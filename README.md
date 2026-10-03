# Age Predictions Based on fMRI Scans: Neurodevelopmental Connectomics in Pediatric Cohorts

[![Paper PDF](https://img.shields.io/badge/Research_Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](docs/capstone_article.pdf)
[![Defense Slides](https://img.shields.io/badge/Defense_Slides-PDF-blue?style=for-the-badge&logo=googleslides)](docs/presentation.pdf)
[![Interactive Showcase](https://img.shields.io/badge/Live_Showcase-Static_HTML-teal?style=for-the-badge&logo=html5)](index.html)
[![In-Browser ML](https://img.shields.io/badge/In--Browser_ML-XGBoost_JS-emerald?style=for-the-badge&logo=javascript)](assets/xgb_model.js)
[![Tests Passing](https://img.shields.io/badge/Tests-6%2F6_Passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)

---

## 📄 Research Publications & Artifacts

- 📖 **[Read the Full Research Paper (PDF)](docs/capstone_article.pdf)**  
  *Coleman, D., Sekander, M., Macias, C., Serna, A., Barandica, J., & Konatolapalli, A. (2025). "Age Predictions Based on FMRI Scans: Neurodevelopmental Connectomics in Pediatric Cohorts." Saint Mary's College of California, School of Business (31 pages).*
- 📊 **[View Technical Defense Presentation Slides (PDF)](docs/presentation.pdf)**  
  *Covers the complete trial-and-error engineering journey, synthetic data failure modes, and sample-weighted gradient boosting.*
- 🌐 **[Launch Interactive Web Showcase (`index.html`)](index.html)**  
  *Standalone, zero-overhead static showcase with 100% client-side in-browser XGBoost inference, connectome visualizations, patient archetype simulators, and Chart.js benchmarks.*

---

## Overview

This project investigates how resting-state functional magnetic resonance imaging (rs-fMRI) Blood Oxygenation Level Dependent (BOLD) data can predict chronological age across a pediatric cohort (Ages 5–21). Using the open-resource [Healthy Brain Network (HBN) dataset](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/index.html) (Child Mind Institute) via the WiDS Datathon 2025 ($N = 1,578$), we developed an equitable machine learning pipeline that addresses:
1. **The Curse of Dimensionality:** 19,900 pairwise functional connectivity (FC) features against 1,578 subjects ($p \gg n$).
2. **Severe Demographic Imbalance:** Dense concentration in ages 9–14 with extreme scarcity in young children (<8y) and emerging adults (17–21y, ~15%).
3. **No-Outlier-Removal Constraint:** A strict protocol rule prohibiting subject pruning, requiring algorithmic noise resilience.

Our final optimized **Sample-Weighted XGBoost model with 40-PCA latent representation** achieved an **$R^2$ of 0.72**, an **RMSE of 2.1 years**, and an **MAE of 1.6 years** (>50% error reduction over baseline models). SHAP feature attribution corroborated neurodevelopmental **"Functional Frontalization"** (*Rubia et al., 2000*), highlighting prefrontal cortex (dlPFC, mPFC) and limbic circuits (amygdala, hippocampus) as prime predictors of brain maturation.

---

## Key Contributions

- **Feature Engineering:** Extracted upper-triangle Pearson correlation pairs ($\frac{200 \times 199}{2} = 19,900$) from the Schaefer 200-ROI cortical atlas.
- **Dimensionality Reduction:** Compressed 19,900 features into **40 Principal Components** retaining **90% explained macro-variance**, cutting noise by 99.8%.
- **Trial-and-Error Exploration:**
  - Evaluated Ridge Regression ($R^2 = 0.35$, RMSE = 4.2y) and Random Forest ($R^2 = 0.45$, RMSE = 3.8y), uncovering catastrophic failure at boundary ages (<8y and >17y).
  - Attempted raw connectome synthetic data generation across biometric covariates; discovered that artificial matrix noise collapsed training runs and degraded generalization (abandoned).
  - Tested deep learning (2D CNNs and GNNs), which overfitted due to small sample size and topological variance.
- **The Breakthrough Solution:** Balanced the 40-PCA latent space via SMOTE/random undersampling, paired with **inverse-frequency loss sample weighting** inside XGBoost, penalizing errors on rare age cohorts up to 2.0x higher.
- **Explainability:** SHAP feature analysis mapping latent components back to anatomical hubs, demonstrating biological alignment with executive function maturation.
- **Client-Side Deployment Pivot:** Scrapped high-cost, high-latency cloud backends (GCP Cloud Run, GCS, BigQuery, Hugging Face) in favor of an autonomous, zero-cost **static web showcase** (`index.html`) running native JavaScript XGBoost tree evaluation (<1ms latency).

---

## Results & Benchmark Comparison

### Model Benchmark Matrix (Internal Test Set)

| Model Architecture | Input Features | Resampling Strategy | R² Score | RMSE | MAE | Status / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Ridge Regression** | 40 PCA | None (Raw) | 0.35 | 4.2 yrs | 3.4 yrs | Flatlined at boundaries (<8y, >17y) |
| **Random Forest Regressor** | 40 PCA | None (Raw) | 0.45 | 3.8 yrs | 3.0 yrs | Zero valid predictions for ages 5–10 in several folds |
| **Deep Learning (CNN / GNN)** | 200×200 / Graph | None (Raw) | <0.20 | >4.5 yrs | >3.6 yrs | Severe overfitting from small $N$ and spatial variance |
| **Raw Synthetic XGBoost** | 40 PCA | Noisy Matrix Interp | 0.52 | 3.4 yrs | 2.6 yrs | Unstable training, RAM crashes, noisy artifacts |
| **⭐ Final Weighted XGBoost** | **40 PCA** | **Sample Weights + SMOTE** | **0.72** | **2.1 yrs** | **1.6 yrs** | **Winning pipeline; equitable across all cohorts** |

### Stratified Performance Across Developmental Epochs

| Age Range | Developmental Stage | RMSE | R² Score | Clinical / Biological Significance |
| :---: | :--- | :---: | :---: | :--- |
| **5–10** | Early Childhood | **2.3 yrs** | **0.68** | Restored sensitivity where Random Forest failed completely |
| **11–15** | Puberty & Adolescence | **1.8 yrs** | **0.75** | Highest fidelity; tightest error bounds around growth spurt |
| **16–21** | Late Adolescence & Emerging Adulthood | **2.4 yrs** | **0.70** | Eliminated systematic ceiling underprediction |

---

## 🏗️ Architecture & Deployment Pivot

> [!NOTE]
> **Why Cloud Deployment was Scrapped:**  
> Deployment through Google Cloud Platform (Cloud Run, GCS, BigQuery) and Hugging Face Spaces was intentionally **scrapped** during our architecture audit. Replacing heavy containerized microservices with an autonomous **Static Showcase Architecture** achieves:
> - **$0/month Cost:** Hosted permanently on GitHub Pages with zero cloud bills.
> - **Zero Cold Starts:** In-browser tree evaluation executes in **<1ms** natively in JavaScript, vs. 4–8 second Docker container boot times.
> - **100% Patient Privacy:** Connectome matrices are evaluated entirely inside the user's browser (HIPAA-friendly; zero data leaves the machine).
> - **Zero Maintenance:** No Docker images, no CORS proxies, no expired SSL certificates, and no cloud outages.

### 🌐 [Interactive Research Showcase (`index.html`)](index.html)
- **Live In-Browser XGBoost Predictor:** Evaluates the serialized 100-tree model directly in client JavaScript with zero backend calls.
- **Subject Archetype Selector:** Test simulated patient connectomes for Child (6.5y), Middle Child (8.0y), Preadolescent (10.5y), Adolescent (14.5y), Middle Teen (16.0y), and Emerging Adult (19.5y).
- **Interactive PCA Feature Sliders:** Tweak top principal components in real time to observe live age output, brain age gap ($\Delta$), and frontalization maturity index.
- **Interactive Visualizations:** Interactive Chart.js histograms, benchmark comparisons, and authentic publication figures.

---

## Project Structure

```
fMRI-Age-Prediction/
├── index.html                           # 🌟 Interactive Static Research Showcase & In-Browser Predictor
├── assets/                              # Compiled JS model, archetypes, & paper figures
│   ├── xgb_model.js                     # 100-tree XGBoost booster compiled for pure JS execution (<1ms)
│   ├── archetypes.js                    # Pediatric connectome patient profiles (Ages 6.5–19.5)
│   ├── fig7_predicted_vs_actual_scatter.png # Paper Fig 7: Predicted vs Actual Scatter Plot
│   ├── fig8_bias_correction_histograms.png  # Paper Fig 8: Bias Correction Comparison
│   ├── fig2_age_distribution_raw.png        # Paper Fig 2: Raw Skewed Cohort Histogram
│   ├── fig3_resampled_age_distribution.png  # Paper Fig 3: Resampled / Weighted Distribution
│   ├── fig4_ridge_alpha_tuning.png          # Paper Fig 4: Ridge Alpha Optimization
│   └── synthetic_data_attempt.png           # Slide Fig: Why Synthetic Data Generation Failed
├── docs/                                # Peer-formatted research publications
│   ├── capstone_article.pdf             # 📄 Full 31-page research paper
│   └── presentation.pdf                 # 📊 Technical defense presentation slides
├── data/                                # Local dataset directory (ignored by git)
├── notebooks/                           # Research exploration & trial-and-error notebooks
│   ├── 01_tsv_generation.ipynb
│   ├── 02_initial_pca_model.ipynb
│   ├── 03_rf_xgboost_ensemble.ipynb
│   ├── 04_synthetic_data_model.ipynb   # The synthetic data exploration
│   ├── 05_reduced_synthetic_data.ipynb
│   ├── 06_oversampling_model.ipynb     # SMOTE & quantile stretching
│   └── 07_age_balanced_xgboost.ipynb   # Final weighted XGBoost pipeline
├── src/                                 # Lean Python source code
│   ├── extract.py                       # Upper-triangle FC extraction (19,900 features)
│   ├── transform.py                     # Scikit-Learn PCA (40 components) & SMOTE
│   ├── train.py                         # XGBoost training with sample weighting
│   └── pipeline.py                      # CLI entrypoint
├── tests/                               # Fast unit test suite (6/6 passing)
│   ├── test_transform.py                # PCA & SMOTE transformation unit tests
│   └── test_model.py                    # Serialized model booster & web asset tests
├── xgboost_fmri.json                    # Serialized trained 100-tree model booster
├── requirements.txt                     # Scientific dependencies + pytest
└── README.md                            # Comprehensive project documentation
```

---

## Getting Started & Usage

### 1. Run the Interactive Showcase Locally (Recommended)
You do not need to install Python packages or build Docker images to use the interactive application. Simply open [`index.html`](index.html) in your browser:

```bash
# Option A: Open directly in your browser
xdg-open index.html   # Linux
open index.html       # macOS

# Option B: Run via Python's built-in HTTP server
python3 -m http.server 8000
# Navigate to http://localhost:8000
```

### 2. Deploy to GitHub Pages (Zero-Cost Hosting)
To host the interactive showcase live:
1. Navigate to your repository settings on GitHub: **Settings &rarr; Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Set **Branch** to `main` and **Folder** to `/ (root)`, then click **Save**.

### 3. Local Python Development & CLI Pipeline

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/fmri-age-prediction.git
cd fmri-age-prediction

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the automated test suite
python3 -m unittest discover -s tests -p "test_*.py" -v
# or
pytest tests/ -v

# 4. Execute the pipeline via CLI
python3 src/pipeline.py --stage extract
python3 src/pipeline.py --stage transform
python3 src/pipeline.py --stage all
```

---

## Citation

If you use this work, codebase, or findings in your research, please cite our paper:

```bibtex
@article{coleman2025fmri_age,
  title={Age Predictions Based on fMRI Scans: Neurodevelopmental Connectomics in Pediatric Cohorts},
  author={Coleman, David and Sekander, Maci and Macias, Caleb and Serna, Angel and Barandica, John and Konatolapalli, Anikait},
  journal={Saint Mary's College of California, School of Business},
  year={2025},
  url={https://github.com/yourusername/fmri-age-prediction}
}
```

**APA Citation:**  
Coleman, D., Sekander, M., Macias, C., Serna, A., Barandica, J., & Konatolapalli, A. (2025). *Age Predictions Based on fMRI Scans: Neurodevelopmental Connectomics in Pediatric Cohorts*. Saint Mary's College of California, School of Business.

---

## Authors

* **David Coleman** — Saint Mary's College of California
* **Maci Sekander** — Saint Mary's College of California
* **Caleb Macias** — Saint Mary's College of California
* **Angel Serna** — Saint Mary's College of California
* **John Barandica** — Saint Mary's College of California
* **Anikait Konatolapalli** — Saint Mary's College of California
