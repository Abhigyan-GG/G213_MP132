# G213 · MP132 · Household Electricity Use (Patna)

SCSE3040 MLOps mini project, batch 55. The service estimates monthly household electricity consumption in kWh using information available at mid-month.

| Member | Enrolment |
| --- | --- |
| ABHIGYANJHAMANIT | S24CSEU0622 |
| AnshRajput | S24CSEU2512 |
| RohitGangwar | S24CSEU2533 |

Codex generated the implementation and ran the local experiments. The group roster above identifies the assigned students. Each student should describe their own contributions when submitting.

## Run the service

From the repository root, create a Python 3.12 virtual environment, activate it, and run the following commands. The tests and API use the exported model included in the repository.

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --port 8000 --no-access-log
```

Open `http://localhost:8000/docs` to try `POST /predict`. The response contains `prediction` in kWh. Example request body:

```json
{"rooms":3,"members":4,"ac_hours_per_day":6,"avg_temp_c":32,"house_area_sqft":1100,"solar_kw":1.5,"connection":"domestic","feeder":"Feeder-1"}
```

## Reproduce the experiments

Copy the group's three original CSVs into `data/raw/`. The DVC pointers identify the training and later files. To verify their contents with the same DVC environment used here, run `uvx --python 3.10 --with pathspec==0.12.1 --from dvc==3.63.0 dvc status`. The pipeline takes paths, ridge penalties, and split settings from `params.yaml`.

```sh
python -m src.electricity.data
python -m src.electricity.train --output-dir ../G213-reproduction
```

Use a fresh output directory for each reproduction to preserve the two registered versions and their run IDs. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the directory containing the database and its `models/` folder to inspect the experiments.

The audit counts issues before deduplication. Historical corrections convert Feeder-2 AC minutes to hours, normalize connection spellings, and mask missing member counts and stuck room readings. Each fitted pipeline learns its own imputation medians. The service takes the eight clean input fields in their documented units.

The initial model compares three ridge penalties on the final fifth of the training period, keeping the boundary date together. Numeric and categorical interactions let the effect of a feature vary by feeder. The best penalty is refitted on all cleaned training rows. The drift report uses training deciles, merges tied edges, includes out-of-range values in the end bins, and floors bin proportions at 0.000001. Feature PSI is in `reports/findings.yaml`. Target PSI is logged on the v2 MLflow run.

Retraining adds the earliest 210 later records and weights each recent record 20 times as heavily as a historical record to adapt to the new period. A stable date sort preserves input order for ties. The last 90 later records score both versions and are excluded from fitting and selection of the ridge penalty. A reduction of at least 15% in MAE promotes v2. The service model and all 250 September predictions come from the selected registry alias.

## Deployment

```sh
docker build -t mp132-api:1.0 .
docker run --rm -p 8000:8000 mp132-api:1.0
```

For Kubernetes, make this image available on the cluster nodes, then apply `kubectl apply -f k8s/`. The Service exposes port 80 and forwards traffic to three API replicas on port 8000. Both probes call `GET /health`.
