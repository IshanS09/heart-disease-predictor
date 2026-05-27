"""
Heart Disease Risk Predictor — Streamlit Web Application
Dataset: Cleveland Heart Disease Dataset (UCI ML Repository)
Author: Ishan Singh | Heart of Gold Research Project

Run:  streamlit run app.py
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# Auto-train if model not found (e.g. on Streamlit Cloud)
if not (MODEL_DIR / "best_model.pkl").exists():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from train import main as train_main
    train_main()

# ── CSS overrides ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .risk-high { background:#fadbd8; border-left:4px solid #e74c3c;
               padding:12px 18px; border-radius:6px; font-size:1.1rem; }
  .risk-low  { background:#d5f5e3; border-left:4px solid #27ae60;
               padding:12px 18px; border-radius:6px; font-size:1.1rem; }
  .metric-card { background:#f8f9fa; border-radius:8px; padding:16px;
                 text-align:center; box-shadow:0 1px 4px rgba(0,0,0,.08); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    pipeline   = joblib.load(MODEL_DIR / "best_model.pkl")
    metrics    = json.loads((MODEL_DIR / "metrics.json").read_text())
    roc_data   = json.loads((MODEL_DIR / "roc_data.json").read_text())
    background = np.load(MODEL_DIR / "shap_background.npy")
    return pipeline, metrics, roc_data, background


try:
    pipeline, metrics, roc_data, bg_data = load_artifacts()
    best_model_name = metrics.get("best_model", "Best Model")
    MODEL_READY = True
except FileNotFoundError:
    MODEL_READY = False


# ══════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════
NUMERIC_FEATURES     = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURE_META = {
    "age":      {"label": "Age (years)",                     "type": "slider",
                 "min": 20, "max": 80, "default": 52},
    "sex":      {"label": "Sex",                             "type": "select",
                 "options": {0: "Female", 1: "Male"},        "default": 1},
    "cp":       {"label": "Chest Pain Type",                 "type": "select",
                 "options": {0: "Typical Angina",
                             1: "Atypical Angina",
                             2: "Non-Anginal Pain",
                             3: "Asymptomatic"},             "default": 0},
    "trestbps": {"label": "Resting Blood Pressure (mmHg)",   "type": "slider",
                 "min": 90, "max": 200, "default": 130},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",       "type": "slider",
                 "min": 100, "max": 600, "default": 240},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl", "type": "select",
                 "options": {0: "No", 1: "Yes"},             "default": 0},
    "restecg":  {"label": "Resting ECG Results",             "type": "select",
                 "options": {0: "Normal",
                             1: "ST-T Wave Abnormality",
                             2: "Left Ventricular Hypertrophy"}, "default": 0},
    "thalach":  {"label": "Max Heart Rate Achieved (bpm)",   "type": "slider",
                 "min": 60, "max": 210, "default": 150},
    "exang":    {"label": "Exercise-Induced Angina",         "type": "select",
                 "options": {0: "No", 1: "Yes"},             "default": 0},
    "oldpeak":  {"label": "ST Depression (Oldpeak)",         "type": "slider_float",
                 "min": 0.0, "max": 6.5, "default": 1.0, "step": 0.1},
    "slope":    {"label": "Peak ST Segment Slope",           "type": "select",
                 "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}, "default": 1},
    "ca":       {"label": "Major Vessels Colored (0–3)",     "type": "slider",
                 "min": 0, "max": 3, "default": 0},
    "thal":     {"label": "Thalassemia",                     "type": "select",
                 "options": {0: "Unknown",
                             1: "Normal",
                             2: "Fixed Defect",
                             3: "Reversible Defect"},        "default": 1},
}


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — PATIENT INPUT FORM
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.title("🩺 Patient Parameters")
st.sidebar.caption("Adjust values to generate a real-time risk prediction.")

user_inputs = {}
for feat, meta in FEATURE_META.items():
    t = meta["type"]
    if t == "slider":
        user_inputs[feat] = st.sidebar.slider(
            meta["label"], meta["min"], meta["max"], meta["default"]
        )
    elif t == "slider_float":
        user_inputs[feat] = st.sidebar.slider(
            meta["label"], meta["min"], meta["max"],
            meta["default"], step=meta.get("step", 0.1),
        )
    elif t == "select":
        opts    = meta["options"]
        keys    = list(opts.keys())
        labels  = list(opts.values())
        def_idx = keys.index(meta["default"])
        picked  = st.sidebar.selectbox(meta["label"], labels, index=def_idx)
        user_inputs[feat] = keys[labels.index(picked)]

X_input = pd.DataFrame([user_inputs])

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational purposes only. Not a medical device. "
    "Always consult a qualified physician."
)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("❤️  Heart Disease Risk Predictor")

if not MODEL_READY:
    st.error(
        "**Model not found.** Run `python train.py` first to train and save the model, "
        "then refresh this page."
    )
    st.stop()

col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Model",    best_model_name)
col_h2.metric("Test AUC", f"{metrics[best_model_name]['roc_auc']:.3f}")
col_h3.metric("Accuracy", f"{metrics[best_model_name]['accuracy']:.3f}")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔮  Risk Prediction",
    "📊  Model Performance",
    "🔬  SHAP Explainability",
])


# ── TAB 1: RISK PREDICTION ───────────────────────────────────────────────
with tab1:
    prob = pipeline.predict_proba(X_input)[0][1]
    pred = int(prob >= 0.5)
    pct  = prob * 100

    col_gauge, col_summary = st.columns([1, 1], gap="large")

    with col_gauge:
        st.subheader("Risk Assessment")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": "Heart Disease Probability", "font": {"size": 17}},
            number={"suffix": "%", "font": {"size": 42,
                    "color": "#e74c3c" if pred else "#27ae60"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#e74c3c" if pred else "#27ae60", "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line":      {"color": "#2c3e50", "width": 3},
                    "thickness": 0.75,
                    "value":     50,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=20, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if pred:
            st.markdown(
                f'<div class="risk-high">⚠️ <b>High Risk</b> — '
                f'{pct:.1f}% probability of heart disease.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-low">✅ <b>Low Risk</b> — '
                f'{pct:.1f}% probability of heart disease.</div>',
                unsafe_allow_html=True,
            )

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        conf = max(prob, 1 - prob)
        st.progress(conf, text=f"Model confidence: {conf*100:.1f}%")

    with col_summary:
        st.subheader("Patient Summary")

        rows = []
        for k, v in user_inputs.items():
            meta  = FEATURE_META[k]
            label = meta["label"]
            if "options" in meta:
                display_val = meta["options"].get(v, v)
            else:
                display_val = v
            rows.append({"Feature": label, "Value": display_val})

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, hide_index=True, use_container_width=True, height=430)


# ── TAB 2: MODEL PERFORMANCE ─────────────────────────────────────────────
with tab2:
    model_names = [k for k in metrics if k != "best_model"]

    # Comparison table
    st.subheader("Model Comparison")
    comp_df = pd.DataFrame([
        {
            "Model":    name,
            "Accuracy": round(metrics[name]["accuracy"], 4),
            "ROC-AUC":  round(metrics[name]["roc_auc"],  4),
            "F1-Score": round(metrics[name]["f1"],       4),
            "Best?":    "🏆" if name == best_model_name else "",
        }
        for name in model_names
    ]).sort_values("ROC-AUC", ascending=False)

    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ROC curves
    st.subheader("ROC Curves")
    COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12"]
    fig_roc = go.Figure()
    fig_roc.add_shape(
        type="line", line=dict(dash="dash", color="#aaaaaa", width=1),
        x0=0, x1=1, y0=0, y1=1,
    )
    for (name, roc), color in zip(roc_data.items(), COLORS):
        width = 3 if name == best_model_name else 1.5
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            name=f"{name}  (AUC = {roc['auc']:.3f})",
            line=dict(color=color, width=width),
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.99),
        height=460, margin=dict(t=10),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Bar chart — metric comparison
    st.subheader("Metric Comparison")
    bar_records = []
    for name in model_names:
        for metric_name in ("accuracy", "roc_auc", "f1"):
            bar_records.append({
                "Model":  name,
                "Metric": metric_name.upper().replace("_", "-"),
                "Score":  metrics[name][metric_name],
            })
    bar_df = pd.DataFrame(bar_records)
    fig_bar = px.bar(
        bar_df, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_map={"ACCURACY": "#2980b9",
                            "ROC-AUC": "#e74c3c",
                            "F1":      "#27ae60"},
        range_y=[0.7, 1.0],
        height=380,
    )
    fig_bar.update_layout(margin=dict(t=10))
    st.plotly_chart(fig_bar, use_container_width=True)


# ── TAB 3: SHAP EXPLAINABILITY ────────────────────────────────────────────
with tab3:
    st.subheader("SHAP — Why This Prediction?")
    st.markdown(
        "**SHAP** (SHapley Additive exPlanations) reveals exactly how each clinical "
        "feature pushes the model's prediction toward or away from a heart disease "
        "diagnosis for *this specific patient*."
    )

    with st.spinner("Computing SHAP values …"):
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            model        = pipeline.named_steps["model"]
            X_t          = preprocessor.transform(X_input)   # shape (1, n_features)

            # Choose explainer type based on model family
            if hasattr(model, "feature_importances_"):
                # Tree-based: TreeExplainer — fast, exact
                explainer   = shap.TreeExplainer(model, bg_data)
                sv          = explainer.shap_values(X_t)
                # binary classifiers return list [class0, class1]
                if isinstance(sv, list):
                    sv = sv[1]
                shap_vals   = sv[0]
                base_val    = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            else:
                # Linear: KernelExplainer with background sample
                explainer   = shap.KernelExplainer(
                    model.predict_proba, bg_data, link="identity"
                )
                sv          = explainer.shap_values(X_t, nsamples=200)
                shap_vals   = sv[1][0] if isinstance(sv, list) else sv[0]
                base_val    = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )

            # ── Waterfall / bar chart ─────────────────────────────────────
            feat_labels = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            shap_df = (
                pd.DataFrame({"Feature": feat_labels, "SHAP": shap_vals})
                  .assign(Abs=lambda d: d["SHAP"].abs())
                  .sort_values("Abs", ascending=False)
            )

            col_bar, col_table = st.columns([3, 2], gap="large")

            with col_bar:
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP"],
                    y=shap_df["Feature"],
                    orientation="h",
                    marker_color=[
                        "#e74c3c" if v > 0 else "#27ae60"
                        for v in shap_df["SHAP"]
                    ],
                    text=[f"{v:+.3f}" for v in shap_df["SHAP"]],
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    title=f"Feature Impact (base probability = {base_val:.2f})",
                    xaxis_title="SHAP Value (impact on log-odds)",
                    yaxis={"categoryorder": "total ascending"},
                    height=460,
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption(
                    "🔴 Red bar → pushes prediction toward **disease**   |   "
                    "🟢 Green bar → pushes prediction toward **healthy**   |   "
                    "Bar length → magnitude of influence"
                )

            with col_table:
                st.markdown("**Top contributing features**")
                display_shap = shap_df[["Feature", "SHAP"]].head(10).copy()
                display_shap["Direction"] = display_shap["SHAP"].apply(
                    lambda v: "↑ Risk" if v > 0 else "↓ Risk"
                )
                display_shap["SHAP"] = display_shap["SHAP"].map("{:+.4f}".format)
                st.dataframe(display_shap, hide_index=True, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    f"**Model output probability**: {prob:.4f}\n\n"
                    f"**Base (average) probability**: {base_val:.4f}\n\n"
                    f"**Sum of SHAP values**: "
                    f"{shap_vals.sum():+.4f}"
                )

        except Exception as exc:
            st.error(f"SHAP computation failed: {exc}")
            st.info(
                "This can happen if model pickles across different library versions. "
                "Re-run `python train.py` to regenerate a fresh model."
            )


# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "❤️  Built for the **Heart of Gold** research project · "
    "Cleveland Heart Disease Dataset (UCI ML Repository, 1988) · "
    "For educational purposes only — not a medical device."
)
