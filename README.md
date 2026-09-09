# MLOps Sentinel

A production MLOps platform demonstrating the operational layer most ML projects skip: model registry with versioning, real A/B traffic routing between model versions, statistical drift detection, and automated retraining triggers — built around a deliberately simple model, since the point of this project is the infrastructure around it, not the model itself.

## What this is

Most ML portfolio projects stop at "trained a model, here's the accuracy." MLOps Sentinel starts where that stops: it takes a registered model and builds the production machinery around it that real ML systems need — versioning so you always know exactly which model produced which prediction, traffic splitting so a new model can be validated against real load before fully replacing the old one, drift detection so degradation gets caught statistically rather than anecdotally, and a trigger mechanism that flags when retraining is warranted (without blindly automating retraining itself, which is a deliberate safety choice, not an oversight).

The underlying model is a synthetic, deliberately simple fraud-detection classifier — chosen because fraud is the textbook domain where drift isn't hypothetical: fraud patterns actively evolve specifically to evade whatever pattern the last model learned, so "the model degrades over time" has a genuine causal story here, not just a theoretical one.

## How it works

1. **Training** (`training/train.py`) trains a model and registers it in MLflow's model registry, tagging it with an alias — `champion` for the currently-serving model, `challenger` for a candidate under evaluation
2. **Serving** (`app/serving/`) loads models by alias, not hardcoded version number — promoting a challenger to champion is a registry alias reassignment, not a code deploy
3. **A/B routing** (`ab_router.py`) uses consistent hashing on each request's ID to deterministically route a configurable percentage of traffic to the challenger — the same request always routes the same way, which matters for reproducible debugging
4. **Every prediction is logged** to Postgres with its model alias/version, features, and output
5. **Drift detection** (`drift_detector.py`) computes PSI (Population Stability Index) between recent production traffic and the training-time feature baseline, on a per-feature basis, on both a manual endpoint and a scheduled background job
6. **Retraining triggers** (`retrain_trigger.py`) watch drift results and log a `RetrainingEvent` when PSI crosses a configured threshold — this only *flags* the need for retraining; it deliberately does not auto-trigger retraining itself, since that's a consequential action (compute cost, a new model version) that should go through explicit review, not fire silently in the background

## Core components

| Layer | What it does |
|---|---|
| `training/` | Trains and registers models; computes the feature-distribution baseline used for drift comparison |
| `registry/` | Thin wrapper around MLflow's model registry — loads models by alias, caches in-process |
| `serving/` | A/B router (consistent hashing) + predictor (runs inference, logs results) |
| `monitoring/` | PSI drift detector + metrics aggregation service |
| `retraining/` | Threshold-based retraining trigger (flags, does not auto-act) |
| `db/` | SQLAlchemy models + schema for prediction logs, feedback, drift scores, retraining events |
| `api/` | FastAPI routes: `/predict`, `/feedback`, `/monitoring/*`, `/registry/promote-challenger`, `/health` |

## Tech stack

- Python 3.11, FastAPI, Uvicorn
- MLflow (model registry + experiment tracking)
- scikit-learn (LogisticRegression champion, RandomForest challenger — deliberately different model types for a meaningful A/B comparison)
- PostgreSQL + SQLAlchemy (prediction logging, drift scores, retraining events)
- APScheduler (background periodic drift checks)
- pandas, numpy (PSI calculation)
- pytest, pytest-mock
- Docker + docker-compose
- GitHub Actions CI

## Project structure

```text
mlops-sentinel/
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── pytest.ini
├── data/
│ └── prepare_dataset.py
├── training/
│ ├── train.py
│ └── feature_baseline.py
├── app/
│ ├── main.py
│ ├── config.py
│ ├── schemas.py
│ ├── db/
│ ├── registry/
│ ├── serving/
│ ├── monitoring/
│ ├── retraining/
│ └── api/
└── tests/
├── test_registry/
├── test_serving/
├── test_monitoring/
├── test_retraining/
└── test_api/
```


## Running it locally

```bash
# 1. Set up
cd mlops-sentinel
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Bring up Postgres and MLflow first
docker-compose up -d postgres mlflow

# 3. Generate synthetic data, train and register the champion model
export MLFLOW_TRACKING_URI=http://localhost:5001   # host-side port; see note below
python -m data.prepare_dataset
python -m training.train
python -m training.feature_baseline

# 4. Run the unit test suite
python -m pytest -v

# 5. Bring up the full stack
docker-compose up --build

# 6. Apply the DB schema (first time only, on a fresh Postgres volume this
#    happens automatically via docker-entrypoint-initdb.d; run manually if
#    you're working against an existing volume that predates that setup)
docker exec -i mlops-postgres psql -U mlops_user -d mlops_sentinel < app/db/schema.sql

# 7. Smoke test
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_0":0.5,"feature_1":0.5,"feature_2":0.5,"feature_3":0.5,"feature_4":0.5,"feature_5":0.5,"feature_6":0.5,"feature_7":0.5,"feature_8":0.5,"feature_9":0.5}'

# 8. Register a challenger to activate A/B routing
python -m training.train --as-challenger

curl http://localhost:8000/monitoring/summary
curl http://localhost:8000/monitoring/drift
```

**Port note:** MLflow's default port 5000 may already be in use on macOS by AirPlay Receiver. This project's `docker-compose.yml` maps MLflow's host-exposed port to `5001` — the app container still reaches MLflow internally via `http://mlflow:5000` (Docker's internal network is unaffected by host port conflicts), but any script run directly on your machine (like `training/train.py`) needs `MLFLOW_TRACKING_URI=http://localhost:5001`.

## Verified working example

The full loop has been run end-to-end against real MLflow and Postgres instances:

- A LogisticRegression "champion" and a RandomForest "challenger" were both trained and registered with distinct MLflow aliases
- 31 real predictions were served, with ~26% correctly routed to the challenger via consistent-hash A/B routing (target: 20%, variance expected at small sample sizes)
- `/monitoring/summary` correctly aggregated the champion/challenger split
- `/monitoring/drift` correctly computed PSI scores per feature and classified them against standard interpretation bands (STABLE / MODERATE_SHIFT / SIGNIFICANT_SHIFT)

## Design notes worth calling out

- **The model is deliberately simple** (a synthetic dataset, a plain LogisticRegression/RandomForest) — the point of this project is the operational layer, not model sophistication. A 3-day feature-engineering effort here would be solving the wrong problem.
- **Retraining triggers flag, they don't act.** `RetrainTrigger` logs a `PENDING` `RetrainingEvent` when drift crosses threshold, but never itself invokes `training/train.py`. Automatically retraining and redeploying a model in response to a drift signal is a real production decision with real cost and risk — conflating "detected a problem" with "took automated action" is how monitoring becomes an incident, not a safeguard.
- **A/B routing is consistent-hash-based**, not per-request-random — the same request ID always resolves to the same model variant, which matters for reproducible debugging and for trusting that an observed traffic split actually held steady over the analysis window.
- **PSI thresholds are industry-standard bands** (< 0.1 stable, 0.1–0.2 moderate, > 0.2 significant), not arbitrary — this is the metric practitioners actually use for this purpose, not a bespoke invention.

## License

MIT
