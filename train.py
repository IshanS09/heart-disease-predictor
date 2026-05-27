"""
Heart Disease Risk Predictor — Training Pipeline
Dataset: Cleveland Heart Disease Dataset (UCI ML Repository)
Author: Ishan Singh | Heart of Gold Research Project

Pipeline:
  1. Auto-download Cleveland dataset
  2. Preprocessing (imputation + scaling via ColumnTransformer)
  3. Train 4 models with RandomizedSearchCV (5-fold stratified CV)
  4. Evaluate on held-out test set (Accuracy, ROC-AUC, F1)
  5. Save best model, SHAP background, metrics, and ROC data
"""

import warnings
warnings.filterwarnings("ignore")

import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_URL  = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)
DATA_PATH = DATA_DIR / "heart_cleveland.csv"

NUMERIC_FEATURES     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("📥  Downloading Cleveland Heart Disease Dataset …")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print(f"    Saved → {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"✅  Loaded {len(df)} records · {df.shape[1]} features")
    print(f"    Missing values: {df.isnull().sum().sum()} total")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    df = df.copy()
    # Target is already binary: 0 = no disease, 1 = disease

    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"\n📊  Class distribution:")
    print(f"    No disease (0): {(y == 0).sum()}"
          f"  |  Disease (1): {(y == 1).sum()}")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL CONFIGS & HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════
def get_model_configs() -> dict:
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "model__C":      [0.001, 0.01, 0.1, 1, 10, 100],
                "model__solver": ["liblinear", "lbfgs"],
                "model__penalty": ["l1", "l2"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "model__n_estimators":    [100, 200, 300, 500],
                "model__max_depth":       [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf":  [1, 2, 4],
                "model__max_features":    ["sqrt", "log2"],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                eval_metric="logloss", random_state=42,
                verbosity=0, n_jobs=-1,
            ),
            "params": {
                "model__n_estimators":     [100, 200, 300],
                "model__max_depth":        [3, 5, 7, 9],
                "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "model__subsample":        [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
                "model__gamma":            [0, 0.1, 0.3],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators":  [100, 200, 300],
                "model__max_depth":     [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample":     [0.7, 0.8, 1.0],
            },
        },
    }


def train_models(X_train, y_train, preprocessor) -> dict:
    configs = get_model_configs()
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, cfg in configs.items():
        print(f"\n🔧  Training {name} …")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        cfg["estimator"]),
        ])

        search = RandomizedSearchCV(
            pipeline,
            cfg["params"],
            n_iter=25,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)

        print(f"    CV AUC  : {search.best_score_:.4f}")
        print(f"    Params  : {search.best_params_}")

        results[name] = {
            "pipeline": search.best_estimator_,
            "cv_auc":   search.best_score_,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model_results: dict, X_test, y_test) -> dict:
    print("\n" + "=" * 58)
    print("📈  TEST SET EVALUATION")
    print("=" * 58)

    metrics = {}
    for name, result in model_results.items():
        pipe   = result["pipeline"]
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)

        print(f"\n  {name}:")
        print(f"    Accuracy {acc:.4f}  |  AUC {auc:.4f}  |  F1 {f1:.4f}")
        report = classification_report(y_test, y_pred,
                     target_names=["No Disease", "Disease"])
        # indent lines manually for readability
        indented = "\n".join("    " + line for line in report.splitlines())
        print(indented)

        metrics[name] = {
            "accuracy": acc, "roc_auc": auc, "f1": f1, "y_prob": y_prob,
        }

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# 5. SHAP — save background data for app-side explanation
# ══════════════════════════════════════════════════════════════════════════
def save_shap_background(best_pipeline, X_train):
    print("\n🔍  Preparing SHAP background data …")
    preprocessor   = best_pipeline.named_steps["preprocessor"]
    X_train_t      = preprocessor.transform(X_train)

    # 100-sample background — enough for fast app-side SHAP
    rng = np.random.default_rng(42)
    idx = rng.choice(X_train_t.shape[0],
                     size=min(100, X_train_t.shape[0]),
                     replace=False)
    background = X_train_t[idx]

    np.save(MODEL_DIR / "shap_background.npy", background)
    print(f"    Saved background shape: {background.shape}")


# ══════════════════════════════════════════════════════════════════════════
# 6. PERSIST ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════
def save_artifacts(best_name, best_pipeline, metrics, X_test, y_test):
    print("\n💾  Saving artifacts …")

    joblib.dump(best_pipeline, MODEL_DIR / "best_model.pkl")

    # metrics (strip non-serialisable arrays)
    clean = {
        name: {k: float(v) for k, v in m.items() if k != "y_prob"}
        for name, m in metrics.items()
    }
    clean["best_model"] = best_name
    (MODEL_DIR / "metrics.json").write_text(json.dumps(clean, indent=2))

    # ROC curve data per model
    roc_data = {}
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_data[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(m["roc_auc"]),
        }
    (MODEL_DIR / "roc_data.json").write_text(json.dumps(roc_data, indent=2))

    # feature lists
    feature_info = {
        "numeric":     NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "all":         ALL_FEATURES,
    }
    (MODEL_DIR / "feature_info.json").write_text(
        json.dumps(feature_info, indent=2)
    )

    print("    ✅  best_model.pkl · metrics.json · roc_data.json · feature_info.json")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 58)
    print("❤️   HEART DISEASE RISK PREDICTOR — TRAINING PIPELINE")
    print("=" * 58)

    df          = load_data()
    X, y        = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n🔀  Train: {len(X_train)}  |  Test: {len(X_test)}")

    preprocessor  = build_preprocessor()
    model_results = train_models(X_train, y_train, preprocessor)
    metrics       = evaluate(model_results, X_test, y_test)

    best_name     = max(metrics, key=lambda n: metrics[n]["roc_auc"])
    best_pipeline = model_results[best_name]["pipeline"]
    print(f"\n🏆  Best model → {best_name}  "
          f"(AUC = {metrics[best_name]['roc_auc']:.4f})")

    save_shap_background(best_pipeline, X_train)
    save_artifacts(best_name, best_pipeline, metrics, X_test, y_test)

    print("\n✅  Done! Run →  streamlit run app.py")


if __name__ == "__main__":
    main()
