# Age Predictions Based on fMRI Scans

## Overview

This project investigates how resting-state functional magnetic resonance imaging (fMRI) data can be used to predict chronological age in a pediatric cohort. Using the Healthy Brain Network dataset, we developed a machine learning pipeline that addressed dataset imbalances, high-dimensional connectivity features, and biases toward younger ages.

Our final optimized XGBoost model achieved strong predictive performance across all age groups, providing insights into neurodevelopmental patterns such as functional frontalization and the role of frontal–limbic connectivity.

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

## Tech Stack

- **Language:** Python
- **Data Handling:** numpy, pandas
- **Machine Learning:** scikit-learn, xgboost
- **Oversampling:** imblearn (SMOTE)
- **Neuroimaging:** nibabel, nilearn
- **Visualization:** matplotlib, seaborn, shap

## Project Structure

```
fMRI-Age-Prediction/
├── data/
│   ├── metadata/                        # CSV files with participant age metadata
│   │   ├── training_metadata.csv
│   │   ├── test_metadata.csv
│   │   └── code-test.csv
│   ├── train_tsv/                       # Training functional connectivity TSV files
│   │   └── sub-[Patient_ID].tsv
│   └── test_tsv/                        # Test functional connectivity TSV files
│       └── sub-[Patient_ID].tsv
├── docs/
│   ├── capstone_article.pdf             # Written capstone report
│   └── presentation.pdf                 # Project presentation slides
├── notebooks/                           # Jupyter notebooks in development order
│   ├── 01_tsv_generation.ipynb          # Merge raw TSV files; generate synthetic data by age bin
│   ├── 02_initial_pca_model.ipynb       # First model: PCA dimensionality reduction + baseline ML
│   ├── 03_rf_xgboost_ensemble.ipynb     # RF + XGBoost ensemble with Ridge regression and bias correction
│   ├── 04_synthetic_data_model.ipynb    # Model trained with synthetic data augmentation
│   ├── 05_reduced_synthetic_data.ipynb  # Reduced-feature variant of synthetic data approach
│   ├── 06_oversampling_model.ipynb      # XGBoost with SMOTE oversampling for age balance
│   └── 07_age_balanced_xgboost.ipynb   # Final model: age-binned balancing + XGBoost (best results)
├── requirements.txt                     # Python dependencies
└── README.md
```

> **Note:** All notebooks were developed in Google Colab with data sourced from Google Drive. Raw fMRI connectivity data is not included in this repository due to size constraints. Synthetic data generated during experiments is also excluded.

## Usage

**Clone the repository:**
```bash
git clone https://github.com/yourusername/fmri-age-prediction.git
cd fmri-age-prediction
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run notebooks in order:**

Open notebooks in the `notebooks/` directory sequentially (01 → 07). Each notebook corresponds to a stage of the pipeline:

1. `01_tsv_generation.ipynb` — Merge per-subject TSV files and generate synthetic samples for underrepresented age groups.
2. `02_initial_pca_model.ipynb` — Establish a PCA-based baseline model.
3. `03_rf_xgboost_ensemble.ipynb` — Explore Random Forest, XGBoost, and Ridge ensemble with early bias correction.
4. `04_synthetic_data_model.ipynb` — Train with synthetically augmented data.
5. `05_reduced_synthetic_data.ipynb` — Reduced-feature version of the synthetic approach.
6. `06_oversampling_model.ipynb` — Apply SMOTE oversampling to improve minority age-group coverage.
7. `07_age_balanced_xgboost.ipynb` — Final optimized XGBoost model with age-binned balancing (best performance).

> Notebooks require a Google Drive mount with the Healthy Brain Network data at the configured paths. Update the path variables in each notebook to point to your local or Drive data location.

## Future Directions

- Incorporate longitudinal fMRI datasets for developmental trajectory modeling.
- Test deep learning models (CNNs, GNNs) with larger datasets.
- Explore multimodal data fusion (EEG + fMRI) for richer predictions.
- Improve rare age prediction with advanced resampling techniques.

## Authors

- David Coleman
- Maci Sekander
- Caleb Macias
- Angel Serna
- John Barandica
- Anikait Konatolapalli
