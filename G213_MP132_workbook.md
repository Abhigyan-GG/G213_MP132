# G213 · MP132 submission workbook

**SCSE3040 MLOps mini project · Household Electricity Use (Patna) · Batch 55**  
**Faculty** Dr. Saurabh Shrivastava

This workbook is for your group's working sessions. The grader receives your shared GitHub repository link. Section 3 of the student handbook allows AI tutoring and debugging. Each member writes and commits their own work and should be able to explain the pipeline and change their code in the viva.

| Enrolment | Member |
| --- | --- |
| S24CSEU0622 | ABHIGYANJHAMANIT |
| S24CSEU2512 | AnshRajput |
| S24CSEU2533 | RohitGangwar |

## 1. Prepare the repository and preserve the data

Use the group's existing repository if one has been created. Make one public repository for G213, invite all three members, and agree who owns the data audit, the training and monitoring, and the API and deployment. Split the remaining work in your group. Put **G213**, **MP132**, and all three enrolment numbers in `README.md`.

I read the attached CSVs without editing them. Here are the file checks you can compare against your copies after copying them to `data/raw/`.

| CSV | Data rows | Period | SHA-256 of the attached bytes |
| --- | ---: | --- | --- |
| `train.csv` | 1,042 | Jan–Jun 2026 | `973781e4ba19cc5ca856b8c90d10845286fc1bea6f91981aa23609835aee72ff` |
| `later.csv` | 300 | Jul–Aug 2026 | `511eaed477aaabcafd23549afaf6929a748ee953a95b39408d3a651088ab0415` |
| `holdout_features.csv` | 250 | Sep 2026 | `09de892bfc53da4c26811cec48f15cd301a515e3f367ff9f7d2d277462c18129` |

The September file has 250 distinct `record_id` values. Use a byte-for-byte copy from the ZIP, and write cleaned data to a separate path or keep it in memory. Compute your own hashes in PowerShell from the repository root.

```powershell
Get-FileHash -Algorithm SHA256 -Path data/raw/train.csv, data/raw/later.csv, data/raw/holdout_features.csv
```

In the Git repository, with DVC installed, run the handbook's setup. Skip `dvc init` if the repo already contains `.dvc/config`.

```text
dvc init
dvc add data/raw/train.csv data/raw/later.csv
git status --short
```

Commit the DVC pointers, configuration, and ignore rules. The default MD5 values for the attached training and later files are `f662e088d2f62acedfb43c1ccc68a3bc` and `257413c4feb1dac91c1de1a67fba3d14`. Compare these with the generated pointers. Your Python program should compute the **SHA-256** of raw training bytes for MLflow's `data_sha256` parameter. The handbook lists a raw holdout path and also says CI has no CSVs. Keep your raw files local for development and check with faculty before committing raw holdout features to the public repo.

Copy the provided `findings.yaml` template to `reports/findings.yaml` and retain its keys. Create a package under `src/` with `__init__.py` and at least two modules. Pin package versions in `requirements.txt`. Put paths and configurable model settings in `params.yaml`, then actually load that file with `yaml.safe_load`.

## 2. Write the data audit before choosing cleaning rules

Have your data lead write and run the audit in `src/`. Use the observed results to decide whether each issue is present. Retain the audit code and keep a record of which raw or deduplicated rows you counted.

| Key in `reports/findings.yaml` | Exercise for your own audit code |
| --- | --- |
| `leaking_column` | Compare the headers of train and holdout; account for removal of the known target, `units_kwh`. For any remaining candidate, check association with the target and whether the column would exist at mid-month. |
| `unit_mixup` | For each numeric input, compare min, mean, and max across `feeder` groups. Check any large ratio against the problem's physical units. Report the exact factor where wrong value = right value × factor. Keep your correction scoped to the affected group. |
| `sentinel_value` | Use value counts on numeric inputs. Examine frequent out-of-domain values and count their occurrences. Consider whether frequent zeros are legitimate, especially for `solar_kw`. |
| `duplicate_rows` | Count extra whole-row copies using `DataFrame.duplicated().sum()`. Investigate repeated IDs separately so you can distinguish identical rows from records that disagree. |
| `stuck_sensor` | Sort records by date and examine consecutive days whose readings are all the same. Record the first and last dates if you confirm a stuck period. Inspect columns that naturally spend long stretches at zero before treating a flat period as an error. |
| `target_typos` | Sort `units_kwh`, inspect large gaps above the normal range, and check whether dividing suspected values by ten returns them to a plausible range. Confirm against the rest of the row before settling on a rule. |
| `spelling_variants` | Compare value counts for the raw category columns with the canonical levels in the problem and `later.csv`. Use an explicit mapping for variants you can justify and count the real levels. |

Set each `present` to a YAML boolean. If it's `false`, its detail fields stay `null`. If it's `true`, use the exact column or feeder spelling and fill all details. Use quoted dates in `YYYY-MM-DD` form if a stuck run is present. For every correction, check how many rows the code changes and keep the model's prediction-time inputs limited to the eight API fields.

## 3. Train and register v1

1. Use only cleaned `train.csv` for v1 model selection and fitting. Try at least three settings on the same validation split. A chronological split within Jan–Jun is one way to mimic predicting a later month. Use **MAE in kWh** to compare settings.
2. Keep preprocessing and the regressor together in a fitted scikit-learn pipeline so the API and batch prediction use the same transformations. When using a validation split, fit imputers and encoders on the fitting portion. Verify you can make a prediction from one eight-field row.
3. Set tracking to `sqlite:///mlflow.db` from the repository root. In **every** run, log the original train hash with parameter name `data_sha256`, a metric name containing `mae`, and the settings for that run. Track all experiments before registering your chosen v1. If you refit it after choosing settings, log the refit and record the run ID tied to the registered model.
4. Use one registered model name. The first model version should be **1** with alias `champion`. Avoid registering candidates as versions of this same model. Save an exported champion pipeline to `models/model.joblib` for the service. Commit `mlflow.db` and the joblib file. The handbook allows MLflow's artifact folders to be ignored, so verify alias resolution locally while the artifacts still exist.

## 4. Build and test the service

In `app/main.py`, make `GET /health` return HTTP 200 and include `{"status": "ok"}`. Make `POST /predict` accept these eight fields exactly.

```text
rooms, members, ac_hours_per_day, avg_temp_c,
house_area_sqft, solar_kw, connection, feeder
```

Put `le=65` on the Pydantic field for `avg_temp_c` and `ge=0` on the field for `ac_hours_per_day`. Configure the request schema to reject extra fields so it enforces the exact request shape. Load the exported pipeline in the service and emit one JSON log line to stdout for each successful prediction. Avoid logging raw requests or customer details.

Write at least three pytest tests in `tests/test_*.py`; at least one must use FastAPI's `TestClient`. A good set checks health, a valid prediction, HTTP 422 for a temperature above 65, and HTTP 422 for negative AC hours. Add boundary cases at 65 and 0. Use a small fixture fitted from rows created inside the tests, or a substitute predictor where you are testing request handling. The CI job should work in a fresh checkout without any of the raw CSVs.

| File | What to verify |
| --- | --- |
| `Dockerfile` | `python:3.12-slim`, install `requirements.txt` before copying code, include the model used by the app, `EXPOSE 8000`, run Uvicorn on `0.0.0.0:8000`. Build with `docker build -t mp132-api:1.0 .`. |
| `.dockerignore` | Leave raw data, virtual environments, and Git metadata outside the build context. Keep required app and model files in it. |
| `.github/workflows/ci.yml` | Begin with the literal key `on:`, trigger on push, install requirements, and run pytest. Verify a green GitHub run for the final code. |
| `k8s/deployment.yaml` | Deployment `mp132-api`, image `mp132-api:1.0`, **3 replicas**, container port **8000**. Both readiness and liveness use `/health` on port 8000, with a longer wait for liveness. |
| `k8s/service.yaml` | Service `mp132-svc`, port **80**, target port **8000**. Match the Deployment pod labels in the Service selector. |

## 5. Monitor, retrain, and produce the final predictions

1. Compute 10-bin PSI between cleaned training data and `later.csv` using your P13 convention. Use the same bin boundaries for both sets and check how your lab handles repeated edges, out-of-range observations, and empty bins. In the template, report PSI for the six listed numeric inputs and use the feature with the highest PSI for `most_drifted_feature`. If you also measure target drift under the problem's wording about every numeric column, keep that statistic separate from the feature ranking.
2. Predict with **v1** on all 300 later rows. Compute MAE separately for each feeder and report the feeder with the highest MAE.
3. Sort the later rows by date, train v2 on cleaned train plus the first **210** later rows, and score v1 and v2 on the same remaining **90** rows. Both the 210th and 211th rows are dated **2026-08-15**. Use a stable date sort to preserve CSV order within the same day, and document that tie convention. Keep the 90 comparison rows out of fitting and model tuning.
4. Register v2 as version **2**. Use unrounded MAEs to decide. If `MAE_v2 <= 0.85 * MAE_v1`, record `decision: promote`, move `champion` to version 2, and export v2 for the service. Otherwise record `decision: keep` and leave version 1 as champion. Verify `champion_version`, the registry alias, and the service's joblib agree.
5. Predict with the model selected by the `champion` alias. Save `predictions/holdout_predictions.csv` with exactly `record_id,prediction` as the header, **250** finite predictions, and all September IDs exactly once. Preserve the input order and write the CSV with the index disabled.

## 6. Final check and submission

| Grader section | Points | Evidence before anyone fills the form |
| --- | ---: | --- |
| A–C | 22 | README identifiers, pinned requirements, params actually read, src package, 3+ tests including TestClient, matching DVC pointer |
| D–E | 26 | Seven evidence-backed findings, committed MLflow DB, 3+ hashed runs, MAE, v1 run ID, two versions and champion alias |
| F–J | 38 | Exact API contract, container, green push workflow, Kubernetes manifests and JSON stdout logs |
| K–L | 14 | PSI and MAE findings, correct promotion decision and complete champion predictions |

- [ ] Validate `reports/findings.yaml` after you fill it. All required values should be set. Detail fields for absent issues remain `null`.
- [ ] Push the final commit and verify the Actions run is green.
- [ ] Check that the public repo opens in a private browser window and that all three students have commits from their own accounts.
- [ ] Each member submits the **same repository URL** from their own Bennett account. Report the files, hours, and work each person actually contributed.
- [ ] Do a short viva dry run. Each person should be able to show how they identified one real data issue, explain why the same 90 later rows score both models, demonstrate an expected HTTP 422, and point to their own commits.

**Deadline** Sunday 29 November 2026 at 11:59 PM IST. The latest form response counts and determines Timely marks. The grader uses the group's last commit before the last member's form time, subject to the grace deadline. Push and verify the work first.  
[Individual submission form](https://forms.cloud.microsoft/r/9ZPB1ZzcAK)
