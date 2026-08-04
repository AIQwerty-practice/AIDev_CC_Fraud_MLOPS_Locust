# Credit Card Fraud Detection MLOps

An end-to-end fraud detection application using H2O AutoML and explicit H2O
estimators, MLflow experiment tracking and model registry, FastAPI inference,
Streamlit, and Docker Compose. Every experiment is launched through the single
`backend/train.py` entry point with a YAML configuration.

## Suggestion
We suggest to run the app and containers on chrome with dark mode theme in your settings. 
You can do this in the Google Chrome browser by going to Settings > Appearance > Mode and selecting "Dark"
## Architecture

```text
YAML experiment config ----\
                            +--> train.py --> H2O training --> MLflow --> Model Registry --\
Local raw/processed data --/                 |                                         |
                                             +--> shared native-model volume -----------+--> FastAPI --> Streamlit
```

## Dataset

The full [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset is not included because it exceeds GitHub's practical file-size limits.
Download `creditcard.csv` separately and place it at:

```text
backend/data/raw/creditcard.csv
```

Generate the processed data from `backend`:

```powershell
cd backend
python preprocess_creditcard.py
```

The preprocessing pipeline removes `Time`, fits `StandardScaler` using only the
training split's `Amount` values, transforms the train and test splits, and
creates the lightweight sample files used for API and Streamlit demonstrations.

Raw and processed data stay local. Git includes only `backend/data/sample_test.csv`
and `backend/data/sample_test_labeled.csv` for lightweight demonstrations.
Use `sample_test.csv` for prediction-only uploads and
`sample_test_labeled.csv` for predictions with evaluation metrics and a
confusion matrix.

## Experiment workflow

```text
Choose YAML config --> validate and load data --> select evaluation data
  --> apply feature version --> train AutoML candidates or explicit estimator
  --> threshold-aware evaluation --> log MLflow run and artifacts
  --> register champion alias --> atomically publish inference metadata
```

From `backend`, launch any experiment without editing Python:

```powershell
python train.py --config configs/baseline.yaml
python train.py --config configs/gbm_tuned.yaml
python train.py --config configs/random_forest.yaml
python train.py --config configs/feature_engineering.yaml
python train.py --config configs/threshold.yaml
```

For a quick pipeline check using only the versioned sample data:

```powershell
python train.py --config configs/smoke_test.yaml
```

To add an experiment, copy one YAML file, choose a unique `experiment_name`,
change the desired values, and run the same command with the new path. Run names
default to `<experiment>-<algorithm>-<UTC timestamp>` for clear comparison;
set `run_name` explicitly when a semantic label is preferable.

## Configuration reference

| Parameter | Meaning |
|---|---|
| `experiment_name` | MLflow experiment used to group related runs. |
| `run_name` | Optional MLflow display name; generated with a UTC timestamp when omitted. |
| `algorithm` | `AutoML`, `GBM`, `RandomForest`/`DRF`, or `XGBoost`. |
| `target` | Binary target column, normally `Class`. |
| `feature_version` | `baseline` or the inference-safe `engineered_v1`. |
| `threshold` | Positive-class probability cutoff used for metrics and serving. |
| `train_path`, `test_path` | Paths resolved from the current working directory. |
| `use_pre_split_test` | Use `test_path` when true; create a stratified split when false. |
| `test_size` | Fraction held out when `use_pre_split_test` is false. |
| `sample_frac` | Reproducible stratified fraction of training rows to use. |
| `max_runtime_secs` | AutoML/estimator runtime limit; `0` means no explicit limit. |
| `max_models` | Maximum AutoML candidate count; informational for explicit estimators. |
| `seed` | Random seed for splitting, sampling, CV, and model training. |
| `stopping_metric` | H2O early-stopping metric, such as `AUC` or `logloss`. |
| `stopping_rounds` | Consecutive non-improving scoring rounds before early stopping; `0` disables it. |
| `sort_metric` | Metric used to order the AutoML leaderboard. |
| `balance_classes` | Enables or disables H2O class balancing. |
| `nfolds` | Cross-validation folds; `0` disables cross-validation. |
| `include_algos` | AutoML allow-list. Cannot be combined with `exclude_algos`. |
| `exclude_algos` | AutoML deny-list. Cannot be combined with `include_algos`. |
| `parameters` | Algorithm-specific estimator options such as `ntrees`. |

## MLflow workflow

Each run logs the full resolved configuration (including defaults), experiment and
run names, UTC timestamps, tags, leaderboard, accuracy, precision, recall, F1,
ROC-AUC, confusion matrix, ROC curve, Precision-Recall curve, feature importance
when supported, prediction sample, model metadata, exact YAML, and MLflow H2O
model. After successful model registration, the configured registry alias is
moved to the new version and deployment files are atomically updated.

Open http://localhost:5000, select an experiment, select multiple runs, and use
MLflow's comparison view to compare parameters and metrics.

## Docker workflow

From the project root:

```powershell
.\start.ps1 -Rebuild
```

Alternatively:

```powershell
docker compose up --build
```

Compose starts MLflow, runs `configs/baseline.yaml`, starts FastAPI after training
completes, and starts Streamlit after the API health check passes.
The trainer receives `backend/data` as a read-only bind mount; datasets are not
copied into the backend image.

- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- FastAPI docs: http://localhost:8000/docs

To deploy a different experiment through Compose, change only the trainer's
`--config` path in `docker-compose.yml` and rebuild.

## Local execution workflow

Install the backend dependencies and start MLflow:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-backend.txt
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal, set the tracking URI, train, then start FastAPI:

```powershell
cd backend
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python train.py --config configs/baseline.yaml
uvicorn main:app --host 0.0.0.0 --port 8000
```

Start Streamlit in a third terminal:

```powershell
cd frontend
pip install -r requirements-frontend.txt
$env:BACKEND_URL = "http://127.0.0.1:8000/predict"
streamlit run app.py
```

## Load testing with Locust

Load-testing dependencies are kept separate from the production backend and
frontend dependencies:

```text
load_tests/
|-- locustfile.py
`-- requirements-load-test.txt
```

The Locust scenario simulates clients using the fraud FastAPI service. It sends
weighted traffic to `POST /predict`, `GET /health`, and `GET /`. Prediction
requests upload a small batch from `backend/data/sample_test.csv` and verify
that the API returns a prediction for every submitted transaction.

Run load testing only after at least one training experiment has completed,
the champion model has been registered, and the FastAPI predictor is healthy.
The recommended workflow is:

1. Start the stack and let the trainer finish successfully.
2. Verify a normal prediction through Streamlit or FastAPI.
3. Start Locust from a separate PowerShell terminal.

Install the isolated load-testing dependency from the repository root:

```powershell
pip install -r load_tests\requirements-load-test.txt
```

Start Locust against the locally published FastAPI port:

```powershell
locust -f load_tests\locustfile.py --host http://127.0.0.1:8000
```

Open http://127.0.0.1:8089/dataset before starting the test. The custom Dataset
Manager displays the active file and lets you upload a labeled or unlabeled CSV,
choose how many rows each prediction request sends, or restore the default
dataset. Uploaded data is validated for the required `V1` through `V28` and
`Amount` columns; `Class` is optional and must contain only `0` or `1`. Uploads
are held in Locust process memory and are not written to disk.

After activating a dataset, select **Return to Locust**, enter a user count and
spawn rate, and start the test. Begin with a small load because each prediction
request invokes the H2O model. Locust reports request throughput, response
times, percentiles, and failures for each grouped endpoint. For predictable
results, stop an active test before changing the active dataset.

The browser upload page is intended for a local, single-process Locust run. By
default, each simulated prediction uploads up to 25 rows from
`backend/data/sample_test.csv`. Environment variables remain available for
headless, scripted, and automated runs:

```powershell
$env:LOCUST_SAMPLE_ROWS = "50"
$env:LOCUST_SAMPLE_CSV = "C:\path\to\demo_fraud_unlabeled.csv"
locust -f load_tests\locustfile.py --host http://127.0.0.1:8000
```

`LOCUST_SAMPLE_ROWS` must be between `1` and `5000`. A labeled file is accepted
because the prediction service removes the `Class` column before inference.
The upload page has a 10 MB file-size limit. It is not synchronized across
distributed Locust workers; use a shared fixture and environment variables for
distributed tests. Stop Locust with `Ctrl+C`; stopping Locust does not stop the
Docker application stack.

## Repository safety

The `.gitignore` excludes raw/processed data, MLflow stores, generated models,
virtual environments, caches, secrets, keys, IDE state, and temporary files.
Never force-add those files or commit credentials. Source code, configs, notebooks,
Docker definitions, requirements, documentation, and sample CSVs remain versioned.

## AWS EC2 production deployment

Production combines the base Compose file with the production overlay. Nginx is
the only service that publishes public ports (80 and 443). H2O is available only
to containers on `project_network`; it has no public Nginx route, DNS record, or
host port.

### Example addresses

| Purpose | Address |
|---|---|
| Apex landing page | https://yourdomain.com |
| Landing page alias | https://fraud.yourdomain.com |
| Streamlit app | https://app.yourdomain.com |
| FastAPI | https://api.yourdomain.com |
| MLflow | https://mlflow.yourdomain.com |
| Locust | https://locust.yourdomain.com |

The FastAPI documentation is at `https://api.yourdomain.com/docs` and its health
endpoint is at `https://api.yourdomain.com/health`. Locust's dataset manager is
at `https://locust.yourdomain.com/dataset`.

### EC2 and DNS prerequisites

Use a current Ubuntu LTS EC2 instance with enough memory for H2O training (8 GB
RAM or more is recommended), Docker Engine with the Compose plugin, Git, an
Elastic IP, and inbound security-group rules for SSH (22, restricted to trusted
addresses), HTTP (80), and HTTPS (443). Do not open ports 5000, 8000, 8089,
8501, or 54321.

Create these DNS A records, all pointing to the EC2 Elastic IP:

| Type | Name | Target |
|---|---|---|
| A | `@` | EC2 Elastic IP |
| A | `fraud` | EC2 Elastic IP |
| A | `app` | EC2 Elastic IP |
| A | `api` | EC2 Elastic IP |
| A | `mlflow` | EC2 Elastic IP |
| A | `locust` | EC2 Elastic IP |

Do not create an H2O DNS record.

### Clone and configure

Replace `your-account` only with the account that hosts the repository; keep
the repository folder name unchanged:

```bash
git clone https://github.com/your-account/AIDev_CC_Fraud_MLOPS_Locust.git
cd AIDev_CC_Fraud_MLOPS_Locust
cp .env.example .env
```

Edit `.env` on EC2, set a real `CERTBOT_EMAIL`, and replace reusable placeholder
values only for the actual deployment. Never commit `.env`.

Ensure the processed training dataset expected by the trainer exists:

```text
backend/data/processed/train.csv
```

### Certificates

After all six DNS records resolve to the EC2 Elastic IP and port 80 is free:

```bash
chmod +x deploy/aws/*.sh
./deploy/aws/bootstrap-certificates.sh
```

The script requests one certificate named `yourdomain.com` covering the apex,
landing alias, Streamlit, FastAPI, MLflow, and Locust addresses. H2O is
intentionally excluded. To renew manually:

```bash
./deploy/aws/renew-certificates.sh
```

Example cron entry:

```cron
15 3 * * * /absolute/path/AIDev_CC_Fraud_MLOPS_Locust/deploy/aws/renew-certificates.sh >> /var/log/fraud-certbot.log 2>&1
```

### Start and operate production

Use both Compose files for every production command:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml config
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml ps
```

The deployment helper runs the same standard startup command:

```bash
./deploy/aws/deploy.sh
```

Useful operations:

```bash
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml logs -f
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml exec proxy nginx -t
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml restart proxy backend frontend locust
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml down
```

The production Locust service runs one process so uploaded datasets and its
in-memory active-dataset state remain consistent. Open
`https://locust.yourdomain.com/dataset` to upload a labeled or unlabeled CSV,
then return to Locust to set users and spawn rate. Locust sends traffic to the
internal FastAPI address `http://backend:8000`.

### Production checks and security

Verify all six HTTPS addresses, the API health endpoint, the Locust dataset page,
and persistent MLflow/model volumes after deployment. Nginx forwards the original
host, client address, and HTTPS scheme. Add authentication before exposing MLflow
or Locust to untrusted users. Keep H2O private to the Docker network.
