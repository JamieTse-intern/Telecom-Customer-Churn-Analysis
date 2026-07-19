# Telecom Customer Churn Prediction & Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6+-brightgreen.svg)](https://lightgbm.readthedocs.io/)

A comprehensive end-to-end machine learning analysis of customer churn for a telecom company, covering 1 million customers across 32 features. The project spans data validation, exploratory analysis, statistical testing, predictive modelling, and actionable business recommendations.

---

## 📊 Overview

Customer churn is a critical business problem — telecom companies spend millions annually on retention. This project builds a predictive framework to:

1. **Identify** which customers are likely to churn
2. **Understand** the key drivers behind churn behaviour
3. **Develop** targeted retention strategies segmented by risk profile
4. **Quantify** the financial ROI of proactive intervention

**Baseline churn rate:** 9.92% (classic imbalanced classification — 90/10 split)

---

## 📁 Repository Structure

```
customer-churn-prediction/
├── customer_churn_1M.csv                  # Dataset (1,000,000 rows × 32 columns)
├── customer_churn_prediction_1.ipynb      # Full analysis notebook (97 cells)
└── README.md                              # This file
```

---

## 🔬 Analysis Framework

The notebook follows a structured 8-stage pipeline, each building on the last:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Data Validation & Audit                                 │
│     • Load & copy (df → df1 to protect original)            │
│     • Column classification (categorical vs numerical)      │
│     • Missing value analysis (5 cols, 2-5% missingness)     │
│     • Missingness pattern tests (MCAR/MAR, chi2, t-test)    │
│     • Median imputation                                     │
├─────────────────────────────────────────────────────────────┤
│  2. Univariate Analysis                                     │
│     • Histograms + boxplots for 14 numerical features       │
│     • Skewness analysis (highly skewed: monthlycharges 6.58)│
│     • Count plots for 7 categorical features                │
├─────────────────────────────────────────────────────────────┤
│  3. Bivariate Analysis                                      │
│     • Violin plots by churn status                          │
│     • Welch's t-test + Cohen's d (effect-size ranking)      │
│     • Chi-squared + Cramér's V for categorical features     │
│     • Correlation heatmap + multicollinearity detection      │
├─────────────────────────────────────────────────────────────┤
│  4. Multivariate Analysis                                   │
│     • Pivot heatmaps (contract × senior_citizen, etc.)      │
│     • Pairplot of top 6 features coloured by churn          │
├─────────────────────────────────────────────────────────────┤
│  5. Feature Engineering                                     │
│     • Drop: customer_id, signup_date, totalcharges (r=0.91) │
│     • Ordinal encode education (5 levels)                   │
│     • One-hot encode (gender, marital, contract, etc.)      │
│     • Final matrix: 35 numeric features                     │
├─────────────────────────────────────────────────────────────┤
│  6. Model Training                                          │
│     • Logistic Regression (StandardScaler + class_weight)   │
│     • Random Forest (200 trees, max_depth=15)               │
│     • XGBoost (300 estimators, scale_pos_weight=9)          │
│     • LightGBM (300 estimators, class_weight=balanced)      │
├─────────────────────────────────────────────────────────────┤
│  7. Model Evaluation                                        │
│     • ROC-AUC & PR-AUC (imbalance-aware metrics)            │
│     • Precision, Recall, F1-Score                           │
│     • Confusion matrices, ROC + PR curves                   │
│     • Model comparison table                                │
├─────────────────────────────────────────────────────────────┤
│  8. Business Translation                                    │
│     • Risk decile analysis (2.5× lift in top decile)        │
│     • ROI estimation (149% return on retention spend)       │
│     • Executive report with actionable recommendations      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏆 Key Findings

### Top Churn Drivers (ranked by importance)

| Rank | Feature | Signal |
|------|---------|--------|
| 1 | **Contract type** | Month-to-month customers churn at dramatically higher rates |
| 2 | **Num complaints** | More complaints → stronger churn signal |
| 3 | **Num service calls** | Higher call frequency indicates dissatisfaction |
| 4 | **Tenure** | Customers <6 months are at highest risk |
| 5 | **Late payments** | Financial distress is a key churn precursor |
| 6 | **Data usage** | Heavy data users churn more (service quality sensitivity) |

### Interaction Effects

- **Month-to-month + Senior citizen** = highest-risk segment
- **Electronic check + Paperless billing** = elevated risk
- **Two-year contract holders** are consistently sticky across all tenure levels

---

## 📈 Model Performance

*80/20 stratified split (800K train / 200K test). All models evaluated with imbalance-aware metrics.*

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | 0.6849 | 0.2028 | 15.85% | **63.92%** | 0.2540 |
| Random Forest | 0.6683 | 0.1821 | **17.13%** | 48.48% | 0.2532 |
| XGBoost | 0.6788 | 0.1978 | 16.41% | 59.15% | 0.2570 |
| **LightGBM** ⭐ | **0.6851** | **0.2053** | 15.97% | 63.29% | 0.2551 |

> **Key metric: PR-AUC** — measures the model's ability to identify churners across all probability thresholds. A value of 0.2053 means ~2× improvement over the random baseline (0.10).

### Risk Decile Lift

The top 10% of customers by model risk score have a **24.55% churn rate** — a **2.5× lift** over the 9.92% baseline.

---

## 💰 Business Impact

Targeting the **top 20% highest-risk customers** (~40K from a 200K scoring cycle) with a $50 retention offer:

| Metric | Value |
|--------|-------|
| Churn rate in target segment | 20.0% |
| Preventable churns (30% intervention success) | ~2,405 |
| Revenue saved (lifetime) | ~$4.99M |
| Retention offer cost | ~$2.00M |
| **Net savings per cycle** | **~$2.99M** |
| **ROI** | **149%** |
| **Projected annual savings** | **~$36M** |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.9+
pip install pandas numpy matplotlib seaborn scipy missingno
pip install scikit-learn xgboost lightgbm

# macOS only: XGBoost requires OpenMP runtime
brew install libomp
```

### Running the Analysis

1. Clone the repository:
   ```bash
   git clone https://github.com/JamieTse-intern/Telecom-Customer-Churn-Analysis.git
   cd Telecom-Customer-Churn-Analysis
   ```

2. Open the notebook:
   ```bash
   jupyter notebook customer_churn_prediction_1.ipynb
   ```

3. Run all cells (**Kernel → Restart & Run All**)

The notebook is fully self-contained — all code, visualisations, statistical tests, and the final report are included in one file.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Data manipulation** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn, missingno |
| **Statistical testing** | scipy (t-test, chi-squared, Cohen's d, Cramér's V) |
| **Machine learning** | scikit-learn, XGBoost, LightGBM |
| **Model evaluation** | ROC-AUC, PR-AUC, precision/recall/F1, confusion matrices |

---

## 📝 License

This project is created for educational and portfolio purposes.

---

*Analysis completed July 2026. Data: 1,000,000 customers × 32 features.*
