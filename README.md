# ❤️ Heart Disease Risk Predictor

An end-to-end machine learning web application that predicts the risk of heart
disease from 13 clinical features using the Cleveland Heart Disease Dataset.

Built as a portfolio project for **Heart of Gold** — a student-led nonprofit
focused on advancing cardiovascular health research through AI/ML.

---

## Features

| Capability | Detail |
|---|---|
| **Multi-model training** | Logistic Regression, Random Forest, XGBoost, Gradient Boosting |
| **Hyperparameter tuning** | RandomizedSearchCV · 25 iterations · 5-fold stratified CV |
| **Model selection** | Best model auto-selected by ROC-AUC on held-out test set |
| **SHAP Explainability** | Per-patient waterfall chart — shows *why* the model predicts what it does |
| **Interactive web app** | Streamlit · real-time risk gauge · ROC comparison · model metrics |

---

## Quick Start

```bash
# 1. Clone / download the project
cd heart-disease-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all four models (downloads dataset automatically)
python train.py

# 4. Launch the web app
streamlit run app.py
```

The training script downloads the Cleveland dataset automatically from the
UCI ML Repository on first run and saves it to `data/`.

---

## Project Structure

```
heart-disease-predictor/
├── data/
│   └── heart_cleveland.csv        ← auto-downloaded on first run
├── models/
│   ├── best_model.pkl             ← best pipeline (preprocessor + model)
│   ├── shap_background.npy        ← 100-sample background for SHAP
│   ├── metrics.json               ← accuracy / AUC / F1 for all models
│   └── roc_data.json              ← FPR/TPR arrays for ROC curves
├── train.py                       ← full ML training pipeline
├── app.py                         ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## Dataset

**Cleveland Heart Disease Dataset** — UCI Machine Learning Repository (1988)

303 patient records · 13 clinical features · binary target (disease / no disease)

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type |

---

## ML Pipeline Details

```
Raw CSV
  → ColumnTransformer
      ├── Numeric  : SimpleImputer(median)  → StandardScaler
      └── Categorical: SimpleImputer(most_frequent)
  → Model (best of 4, selected by AUC)
      ├── Logistic Regression   (C, solver, penalty)
      ├── Random Forest         (n_estimators, max_depth, features, …)
      ├── XGBoost               (n_estimators, lr, depth, subsample, …)
      └── Gradient Boosting     (n_estimators, lr, depth, subsample)
  → RandomizedSearchCV (n_iter=25, 5-fold StratifiedKFold)
  → Test-set evaluation (Accuracy · ROC-AUC · F1)
  → Best model saved → Streamlit app
```

---

## Tech Stack

- **ML**: scikit-learn · XGBoost · SHAP
- **App**: Streamlit · Plotly
- **Data**: pandas · NumPy

---

## Deploying to Streamlit Cloud (free)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file to `app.py`
4. Add a startup command to run `train.py` before the app (via `packages.txt` / `setup.sh`)

> **Note:** For cloud deployment, consider committing `models/` after a local training
> run so the app can load without re-training.

---

## Disclaimer

> This tool is for **educational and research purposes only**.  
> It is not a medical device and should not be used to make clinical decisions.  
> Always consult a qualified healthcare provider.
