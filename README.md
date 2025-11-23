Age Predictions Based on fMRI Scans
Overview

This project investigates how resting-state functional magnetic resonance imaging (fMRI) data can be used to predict chronological age in a pediatric cohort. Using the Healthy Brain Network dataset, we developed a machine learning pipeline that addressed dataset imbalances, high-dimensional connectivity features, and biases toward younger ages.

Our final optimized XGBoost model achieved strong predictive performance across all age groups, providing insights into neurodevelopmental patterns such as functional frontalization and the role of frontal–limbic connectivity.

Key Contributions

Built a data preprocessing pipeline to align functional connectivity (FC) matrices with age metadata.

Reduced 19,900 FC features into 40 principal components using PCA (capturing 90% variance).

Applied SMOTE oversampling and random undersampling to balance age representation across 5–21 years.

Developed and tuned multiple machine learning models:

Random Forest (baseline, struggled with imbalance)

Ridge Regression (baseline linear model)

XGBoost (final, with sample weighting + PCA)

Performed SHAP feature analysis to interpret predictive regions of interest.

Results

Final XGBoost Model Performance (internal test set):

R² = 0.72

RMSE = 2.1 years

MAE = 1.6 years

Stratified Performance:

Ages 5–10 → RMSE: 2.3, R²: 0.68

Ages 11–15 → RMSE: 1.8, R²: 0.75

Ages 16–21 → RMSE: 2.4, R²: 0.70

Feature Importance: Frontal and limbic connectivity emerged as key predictors, consistent with developmental milestones.

Tech Stack

Languages: Python

Core Libraries:

Data Handling: numpy, pandas

Machine Learning: scikit-learn, xgboost

Oversampling: imblearn (SMOTE)

Neuroimaging: nibabel, nilearn

Visualization: matplotlib, seaborn, shap

Project Structure
fmri-age-prediction/
│── data/               # Raw & processed fMRI connectivity data/synthetically generated data(not included)
│── notebooks/          # Jupyter notebooks for experiments & visualization
│── results/            # Output metrics(Notebooks were configured for a google drive centered workflow)
│── requirements.txt    # Dependencies
│── README.md           # Documentation

Usage

Clone the repository:

git clone https://github.com/yourusername/fmri-age-prediction.git
cd fmri-age-prediction


Install dependencies:

pip install -r requirements.txt


Run preprocessing on Healthy Brain Network data:

python src/preprocess.py --input data/raw/ --output data/processed/


Train the XGBoost model:

python src/train_xgboost.py --data data/processed/


Visualize results & SHAP feature importance via provided Jupyter notebooks.

Future Directions

Incorporate longitudinal fMRI datasets for developmental trajectory modeling.

Test deep learning models (CNNs, GNNs) with larger datasets.

Explore multimodal data fusion (EEG + fMRI) for richer predictions.

Improve rare age prediction with advanced resampling techniques.

Authors

David Coleman

Maci Sekander

Caleb Macias

Angel Serna

John Barandica

Anikait Konatolapalli