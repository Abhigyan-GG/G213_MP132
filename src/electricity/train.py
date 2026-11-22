"""Build the MLflow registry, monitoring report, service model and predictions."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from .data import CATEGORIES, FEATURES, NUMERIC, TARGET, audit, clean_training


def estimator(alpha):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    inputs = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIES),
    ])
    return Pipeline([
        ("inputs", inputs),
        ("interactions", PolynomialFeatures(interaction_only=True, include_bias=False)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def psi(reference, later):
    # Reference deciles. Tied edges collapse; unbounded tails retain shifted values.
    reference = reference.dropna()
    edges = np.unique(reference.quantile(np.linspace(0, 1, 11)))
    bins = np.r_[-np.inf, edges[1:-1], np.inf]
    proportions = []
    for values in (reference, later.dropna()):
        counts = np.histogram(values, bins=bins)[0]
        proportions.append(np.maximum(counts / counts.sum(), 1e-6))
    expected, actual = proportions
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def score(model, rows):
    return float(mean_absolute_error(rows[TARGET], model.predict(rows[FEATURES])))


def register(client, name, version, model, run_id, requirements):
    # Relative sources keep registry aliases usable when the repository is cloned.
    source = Path("models", "registry", f"v{version}")
    mlflow.sklearn.save_model(model, source, serialization_format="skops",
                              skops_trusted_types=["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList"],
                              pip_requirements=requirements)
    result = client.create_model_version(name, source.as_posix(), run_id=run_id)
    if int(result.version) != version:
        raise RuntimeError(f"Expected version {version}, got {result.version}")


def build(config, raw, template, requirements):
    train_file = raw / "train.csv"
    raw_train = pd.read_csv(train_file)
    later = pd.read_csv(raw / "later.csv").sort_values("date", kind="stable")
    holdout = pd.read_csv(raw / "holdout_features.csv")
    issues = audit(raw_train, holdout)
    train = clean_training(raw_train, issues)
    common = {"data_sha256": sha256(train_file.read_bytes()).hexdigest()}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config["experiment"])
    client = MlflowClient()
    name = config["registered_model"]
    client.create_registered_model(name)

    # Keep an entire calendar date on one side of the validation boundary.
    boundary = train.date.iloc[int(len(train) * (1 - config["validation_fraction"]))]
    fit, valid = train[train.date < boundary], train[train.date >= boundary]
    candidates = []
    for alpha in config["ridge_alphas"]:
        with mlflow.start_run(run_name=f"validate-ridge-{alpha}"):
            candidate = estimator(alpha).fit(fit[FEATURES], fit[TARGET])
            mae = score(candidate, valid)
            mlflow.log_params({**common, "alpha": alpha, "validation_start": boundary})
            mlflow.log_metric("validation_mae", mae)
            candidates.append((mae, alpha))
    best_mae, alpha = min(candidates)
    with mlflow.start_run(run_name="v1-full-train") as run:
        v1 = estimator(alpha).fit(train[FEATURES], train[TARGET])
        mlflow.log_params({**common, "alpha": alpha, "fit_rows": len(train)})
        mlflow.log_metric("selection_validation_mae", best_mae)
        v1_run = run.info.run_id
        register(client, name, 1, v1, v1_run, requirements)
    client.set_registered_model_alias(name, "champion", "1")

    # Use the full Jan-Jun medians for missing reference values in drift monitoring.
    drift = {col: psi(train[col].fillna(train[col].median()), later[col]) for col in NUMERIC}
    group_mae = {str(feeder): score(v1, rows) for feeder, rows in later.groupby("feeder")}
    split = int(config["later_fit_fraction"] * len(later))
    recent, check = later.iloc[:split], later.iloc[split:]
    combined = pd.concat([train, recent], ignore_index=True)
    weights = np.r_[np.ones(len(train)), np.full(len(recent), config["recent_weight"])]
    with mlflow.start_run(run_name="v2-with-recent-data") as run:
        v2 = estimator(alpha).fit(combined[FEATURES], combined[TARGET], ridge__sample_weight=weights)
        old, new = score(v1, check), score(v2, check)
        mlflow.log_params({**common, "alpha": alpha, "recent_weight": config["recent_weight"],
                           "fit_rows": len(combined), "check_rows": len(check)})
        mlflow.log_metrics({"v1_check_mae": old, "v2_check_mae": new,
                            "target_psi": psi(train[TARGET], later[TARGET])})
        register(client, name, 2, v2, run.info.run_id, requirements)
    promote = new <= old * (1 - config["promotion_improvement"])
    champion = 2 if promote else 1
    client.set_registered_model_alias(name, "champion", str(champion))
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    joblib.dump(model, "models/model.joblib", compress=3)
    predictions = pd.DataFrame({"record_id": holdout.record_id, "prediction": model.predict(holdout[FEATURES])})
    if not predictions.record_id.is_unique or not np.isfinite(predictions.prediction).all():
        raise ValueError("Invalid holdout identifiers or predictions")
    Path("predictions").mkdir(exist_ok=True)
    predictions.to_csv("predictions/holdout_predictions.csv", index=False)
    template.update(data_issues=issues, tracking={
        "experiment_name": config["experiment"], "best_run_id": v1_run,
        "registered_model_name": name, "champion_version": champion,
    }, monitoring={
        "psi": drift, "most_drifted_feature": max(drift, key=drift.get),
        "mae_by_group": group_mae, "worst_served_group": max(group_mae, key=group_mae.get),
        "mae_v1_on_check": old, "mae_v2_on_check": new,
        "decision": "promote" if promote else "keep",
    })
    Path("reports").mkdir(exist_ok=True)
    Path("reports/findings.yaml").write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    output = parser.parse_args().output_dir.resolve()
    config = yaml.safe_load(Path("params.yaml").read_text())
    raw = Path(config["raw_dir"]).resolve()
    template = yaml.safe_load(Path("findings.yaml").read_text())
    requirements = Path("requirements.txt").read_text().splitlines()
    output.mkdir(exist_ok=True, parents=True)
    if (output / "mlflow.db").exists() or (output / "models/registry").exists():
        parser.error("A registry already exists. Use --output-dir with a fresh directory to reproduce it.")
    os.chdir(output)
    build(config, raw, template, requirements)


if __name__ == "__main__":
    main()
