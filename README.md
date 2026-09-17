# Age Predictions Based on fMRI Scans

## Overview

This project investigates how resting-state functional magnetic resonance imaging (fMRI) data can be used to predict chronological age in a pediatric cohort. Using the Healthy Brain Network dataset, we developed a machine learning pipeline that addresses dataset imbalances, high-dimensional connectivity features, and biases toward younger ages.

Our final optimized XGBoost model achieved strong predictive performance across all age groups, providing insights into neurodevelopmental patterns such as functional frontalization and the role of frontal–limbic connectivity.

**🚀 Data Engineering Upgrade:** The project was originally prototyped in Jupyter Notebooks. It has since been completely re-architected into a cloud data pipeline utilizing PySpark, Apache Airflow, Terraform, MLflow, and FastAPI.

## Key Contributions

- Built a data preprocessing pipeline to align functional connectivity (FC) matrices with age metadata.
- Reduced 19,900 FC features into 40 principal components using PCA (capturing 90% variance).
- Applied SMOTE oversampling and random undersampling to balance age representation across 5–21 years.
- Developed and tuned multiple machine learning models:
  - Random Forest (baseline, struggled with imbalance)
  - Ridge Regression (baseline linear model)
  - XGBoost (final, with sample weighting + PCA)
- Performed SHAP feature analysis to interpret predictive regions of interest.

## Results

**Final XGBoost Model Performance (internal test set):**

| Metric | Score |
|--------|-------|
| R²     | 0.72  |
| RMSE   | 2.1 years |
| MAE    | 1.6 years |

**Stratified Performance:**

| Age Range | RMSE | R²   |
|-----------|------|------|
| 5–10      | 2.3  | 0.68 |
| 11–15     | 1.8  | 0.75 |
| 16–21     | 2.4  | 0.70 |

Feature Importance: Frontal and limbic connectivity emerged as key predictors, consistent with developmental milestones.

---

## 🏗️ Architecture

This repository serves as a fully containerized, scalable ETL/MLOps portfolio project deployed exclusively on free-tier and developer-focused cloud platforms.

*   **Data Storage:** Google Cloud Storage (GCS) single-region free tier.
*   **Data Warehouse:** Google BigQuery (Free tier).
*   **Data Processing:** Apache Spark (PySpark) for distributed PCA and data balancing.
*   **Orchestration:** Apache Airflow DAGs (Local/Docker) & GitHub Actions (Live execution).
*   **MLOps & Tracking:** MLflow integrated with DagsHub (Free remote tracking server).
*   **Model Serving:** FastAPI REST API, Dockerized and deployed via Hugging Face Spaces.
*   **Infrastructure as Code (IaC):** Terraform.
*   **CI/CD & Testing:** GitHub Actions enforcing `black`, `ruff`, and comprehensive `pytest` suites.

## Project Structure

```
fMRI-Age-Prediction/
├── api/                                 # FastAPI application for model serving
│   └── app.py
├── dags/                                # Apache Airflow DAGs for orchestration
│   └── fmri_pipeline_dag.py
├── data/                                # Local data directory (ignored by git)
├── infra/                               # Terraform IaC for GCP resources
│   ├── main.tf
│   ├── provider.tf
│   └── variables.tf
├── notebooks/                           # Original Jupyter notebooks (prototypes)
├── src/                                 # Modular Python source code
│   ├── cloud_ingest.py                  # GCP GCS and BigQuery ingestion scripts
│   ├── extract.py                       # Data extraction logic
│   ├── pipeline.py                      # Main CLI entrypoint
│   ├── spark_transform.py               # PySpark ETL logic
│   ├── train.py                         # Model training with MLflow integration
│   └── transform.py                     # Local pandas/sklearn transformations
├── tests/                               # Pytest suite
│   ├── test_api.py
│   ├── test_cloud_ingest.py
│   ├── test_spark_transform.py
│   └── test_transform.py
├── .github/workflows/                   # CI/CD pipelines
│   └── ci.yml
├── Dockerfile                           # Dockerfile for Airflow/Spark ETL
├── Dockerfile.api                       # Dockerfile for FastAPI serving
├── requirements.txt                     # Python dependencies
└── README.md
```

## Usage

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/fmri-age-prediction.git
cd fmri-age-prediction
```

**2. Set up the environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Run tests:**
```bash
pytest tests/
```

**4. Execute the pipeline locally (CLI):**
You can use the new `src/pipeline.py` CLI to run specific stages of the local pipeline:
```bash
python src/pipeline.py --extract
python src/pipeline.py --transform
python src/pipeline.py --train
```

**5. Deploy Infrastructure (Terraform):**
```bash
cd infra
terraform init
terraform apply
```

**6. Serve the Model:**
```bash
docker build -t fmri-api -f Dockerfile.api .
docker run -p 8000:8000 fmri-api
```

## Future Directions

- Incorporate longitudinal fMRI datasets for developmental trajectory modeling.
- Test deep learning models (CNNs, GNNs) with larger datasets.
- Explore multimodal data fusion (EEG + fMRI) for richer predictions.

## Authors

- David Coleman
- Maci Sekander
- Caleb Macias
- Angel Serna
- John Barandica
- Anikait Konatolapalli
