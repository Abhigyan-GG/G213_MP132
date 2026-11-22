"""Audit raw records and apply the confirmed historical corrections."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NUMERIC = ["rooms", "members", "ac_hours_per_day", "avg_temp_c", "house_area_sqft", "solar_kw"]
CATEGORIES = ["connection", "feeder"]
FEATURES = NUMERIC + CATEGORIES
TARGET = "units_kwh"


def connection_levels(values):
    return (values.str.strip().str.lower().str.replace(" ", "_", regex=False)
            .replace({"dom.": "domestic", "sho.": "shop_cum_home"}))


def flat_period(data):
    daily = data.groupby("date")[NUMERIC].agg(["min", "max"])
    for column in NUMERIC:
        readings = daily[column]
        values = readings["min"].where(readings["min"] == readings["max"])
        dates = pd.Series(pd.to_datetime(values.index), index=values.index)
        blocks = (values.ne(values.shift()) | dates.diff().dt.days.ne(1)).cumsum()
        for _, period in values.dropna().groupby(blocks):
            if len(period) >= 7:
                return column, str(period.index[0]), str(period.index[-1])
    return None, None, None


def audit(train, holdout):
    extra = train.columns.difference([*holdout.columns, TARGET]).tolist()
    maxima = train.groupby("feeder").ac_hours_per_day.max()
    affected = maxima[maxima > 24]
    factor = None
    feeder = None
    if len(affected):
        feeder = affected.idxmax()
        ratio = affected[feeder] / maxima.drop(feeder).median()
        if not np.isclose(ratio, 60, rtol=0.1):
            raise ValueError("Inspect unexpected AC unit ratio")
        factor = 60
    sentinel = int(train.members.eq(-1).sum())
    duplicates = int(train.duplicated().sum())
    sensor, start, end = flat_period(train)
    # Extra-zero errors would form an isolated upper cluster in sorted targets.
    levels = np.sort(train[TARGET].unique())
    gaps = np.flatnonzero(levels[1:] > 2 * levels[:-1])
    typos = int((train[TARGET] >= levels[gaps[-1] + 1]).sum()) if gaps.size else 0
    canonical = connection_levels(train.connection)
    variants = bool(canonical.ne(train.connection).any())
    return {
        "leaking_column": {"present": bool(extra), "column": next(iter(extra), None)},
        "unit_mixup": {"present": bool(factor), "column": "ac_hours_per_day" if factor else None,
                       "group_value": feeder, "factor": factor},
        "sentinel_value": {"present": bool(sentinel), "column": "members" if sentinel else None,
                           "value": -1 if sentinel else None, "count": sentinel or None},
        "duplicate_rows": {"present": bool(duplicates), "count": duplicates or None},
        "stuck_sensor": {"present": sensor is not None, "column": sensor,
                         "start_date": start, "end_date": end},
        "target_typos": {"present": bool(typos), "count": typos or None},
        "spelling_variants": {"present": variants, "column": "connection" if variants else None,
                              "true_levels": int(canonical.nunique()) if variants else None},
    }


def clean_training(train, issues):
    data = train.drop_duplicates().copy()
    units = issues["unit_mixup"]
    if units["present"]:
        affected = data.feeder.eq(units["group_value"])
        data.loc[affected, units["column"]] /= units["factor"]
    data["members"] = data.members.mask(data.members.eq(-1))
    sensor = issues["stuck_sensor"]
    if sensor["present"]:
        blocked = data.date.between(sensor["start_date"], sensor["end_date"])
        data.loc[blocked, sensor["column"]] = np.nan
    data["connection"] = connection_levels(data.connection)
    return data.sort_values("date", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    raw = Path(yaml.safe_load(Path("params.yaml").read_text())["raw_dir"])
    print(json.dumps(audit(pd.read_csv(raw / "train.csv"),
                           pd.read_csv(raw / "holdout_features.csv", nrows=0)), indent=2))
