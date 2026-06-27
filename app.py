# =============================================================================
# Heart Disease Prediction — Streamlit Web Application
# Author  : ML Engineer Portfolio Project
# Version : 1.0.0
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI · Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "About": "CardioSense AI — Portfolio-grade heart disease prediction app.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- Hero banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.hero-banner h1 { font-size: 2.6rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
.hero-banner p  { font-size: 1.05rem; opacity: 0.85; margin-top: .5rem; }

/* ---- Metric cards ---- */
.metric-card {
    background: #1e2a3a;
    border: 1px solid #2d3f54;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    color: white;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
.metric-card .label { font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }

/* ---- Result boxes ---- */
.result-positive {
    background: linear-gradient(135deg, #ff416c, #c0392b);
    border-radius: 14px; padding: 1.6rem 1.4rem; color: white; text-align: center;
    box-shadow: 0 8px 30px rgba(255,65,108,.35);
}
.result-negative {
    background: linear-gradient(135deg, #11998e, #1a6b5b);
    border-radius: 14px; padding: 1.6rem 1.4rem; color: white; text-align: center;
    box-shadow: 0 8px 30px rgba(17,153,142,.35);
}
.result-positive h2, .result-negative h2 { font-size: 1.6rem; margin: 0 0 .3rem; }
.result-positive p,  .result-negative p  { opacity: .9; margin: 0; font-size: .95rem; }

/* ---- Risk badge ---- */
.badge-high     { background:#ff4b4b; color:white; border-radius:20px; padding:4px 14px; font-weight:600; }
.badge-moderate { background:#ffa500; color:white; border-radius:20px; padding:4px 14px; font-weight:600; }
.badge-low      { background:#21c354; color:white; border-radius:20px; padding:4px 14px; font-weight:600; }

/* ---- Section divider ---- */
.section-title {
    font-size: 1.2rem; font-weight: 600; color: #4fc3f7;
    border-left: 4px solid #4fc3f7; padding-left: .75rem;
    margin: 1.5rem 0 .8rem;
}

/* ---- Sidebar styling ---- */
[data-testid="stSidebar"] { background: #0d1b2a; }
[data-testid="stSidebar"] * { color: #d0dce8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { font-size: .82rem !important; font-weight: 500 !important; }

/* ---- Footer ---- */
.footer {
    margin-top: 3rem; padding-top: 1.2rem;
    border-top: 1px solid #2d3f54;
    text-align: center; font-size: .8rem; color: #6b7f95;
}

/* ---- Tab styling ---- */
button[data-baseweb="tab"] { font-weight: 600 !important; }

/* ---- Expander ---- */
details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="📦 Loading dataset from UCI…")
def load_data() -> pd.DataFrame:
    """Download and clean the Cleveland Heart Disease dataset from UCI."""
    try:
        # Primary: ucimlrepo (most reliable)
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = [
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal","target"
        ]
        df["target"] = (df["target"] > 0).astype(int)
    except Exception:
        # Fallback: public mirror
        url = ("https://raw.githubusercontent.com/sharmaroshan/"
               "Heart-UCI-Dataset/master/heart.csv")
        df = pd.read_csv(url)
        if df["target"].max() > 1:
            df["target"] = (df["target"] > 0).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

@st.cache_resource(show_spinner="🧠 Training models…")
def train_pipeline(df: pd.DataFrame):
    """
    Full ML pipeline:
      - Feature / target split
      - Stratified train/test split
      - StandardScaler
      - Six classifiers evaluated via accuracy, ROC-AUC, 5-fold CV
      - GridSearchCV tuning on the best model
    Returns scaler, tuned_model, results dict, feature names, best model name.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Six candidate models ──────────────────────────────────────────────────
    candidates = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM":                  SVC(probability=True, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(),
    }

    results = {}
    for name, mdl in candidates.items():
        mdl.fit(X_tr, y_train)
        y_pred  = mdl.predict(X_te)
        y_proba = mdl.predict_proba(X_te)[:, 1]
        cv      = cross_val_score(mdl, X_tr, y_train, cv=5, scoring="accuracy")
        results[name] = {
            "model":    mdl,
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc":  roc_auc_score(y_test, y_proba),
            "cv_mean":  cv.mean(),
            "cv_std":   cv.std(),
            "y_pred":   y_pred,
            "y_proba":  y_proba,
            "y_test":   y_test,
        }

    # ── Pick best by ROC-AUC ──────────────────────────────────────────────────
    best_name = max(results, key=lambda m: results[m]["roc_auc"])

    param_grids = {
        "Logistic Regression":  {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        "Decision Tree":        {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
        "Random Forest":        {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "Gradient Boosting":    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "SVM":                  {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        "K-Nearest Neighbors":  {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    }
    base_map = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
        "SVM":                  SVC(probability=True, random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(),
    }
    grid = GridSearchCV(
        base_map[best_name], param_grids[best_name],
        cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid.fit(X_tr, y_train)
    tuned = grid.best_estimator_

    # Re-evaluate tuned model
    y_pred_t  = tuned.predict(X_te)
    y_proba_t = tuned.predict_proba(X_te)[:, 1]
    results["__tuned__"] = {
        "model":    tuned,
        "accuracy": accuracy_score(y_test, y_pred_t),
        "roc_auc":  roc_auc_score(y_test, y_proba_t),
        "y_pred":   y_pred_t,
        "y_proba":  y_proba_t,
        "y_test":   y_test,
        "best_params": grid.best_params_,
    }

    feature_names = list(X.columns)
    return scaler, tuned, results, feature_names, best_name, X_test, y_test


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_level(prob: float) -> tuple[str, str]:
    """Return (label, css_class) based on probability."""
    if prob >= 0.70:
        return "HIGH", "badge-high"
    elif prob >= 0.40:
        return "MODERATE", "badge-moderate"
    return "LOW", "badge-low"


def make_gauge(prob: float) -> plt.Figure:
    """Render a half-donut gauge chart for probability."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Background arc
    theta1, theta2 = 180, 0
    wedge_bg = mpatches.Wedge(
        (0.5, 0.1), 0.4, theta1, theta2,
        width=0.18, facecolor="#2d3f54", edgecolor="none"
    )
    ax.add_patch(wedge_bg)

    # Foreground arc (proportional to probability)
    color = "#ff4b4b" if prob >= 0.70 else "#ffa500" if prob >= 0.40 else "#21c354"
    angle_range = 180 * prob
    wedge_fg = mpatches.Wedge(
        (0.5, 0.1), 0.4, 180 - angle_range, 180,
        width=0.18, facecolor=color, edgecolor="none"
    )
    ax.add_patch(wedge_fg)

    ax.text(0.5, 0.18, f"{prob:.0%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.06, "Probability", ha="center", va="center",
            fontsize=9, color="#90a4ae", transform=ax.transAxes)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.05, 0.55)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def plot_feature_importance(model, feature_names: list) -> plt.Figure:
    """Horizontal bar chart of Random-Forest-style feature importances."""
    importances = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    colors = ["#4fc3f7" if v < importances.max() else "#ff4b4b" for v in importances.values]
    ax.barh(y=importances.index, width=importances.values, height=0.65, color=colors, edgecolor="none")

    ax.axvline(importances.mean(), color="#90a4ae", ls="--", lw=1, alpha=.7,
               label=f"Mean = {importances.mean():.3f}")
    ax.set_title("Feature Importances · Random Forest", color="white", fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score", color="#90a4ae")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(facecolor="#1e2a3a", labelcolor="white", fontsize=8)
    plt.tight_layout()
    return fig


def plot_roc_all(results: dict) -> plt.Figure:
    """Overlay ROC curves for all 6 models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    palette = plt.cm.tab10(np.linspace(0, 1, 6))
    for (name, res), color in zip(
        {k: v for k, v in results.items() if k != "__tuned__"}.items(), palette
    ):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} ({res['roc_auc']:.3f})")

    ax.plot([0,1],[0,1], "w--", lw=1, alpha=.4)
    ax.set_xlabel("False Positive Rate", color="#90a4ae")
    ax.set_ylabel("True Positive Rate", color="#90a4ae")
    ax.set_title("ROC Curves — All Models", color="white", fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", labelcolor="white", fontsize=8, loc="lower right")
    for spine in ax.spines.values():
        spine.set_color("#2d3f54")
    plt.tight_layout()
    return fig


def plot_confusion(results: dict) -> plt.Figure:
    """Confusion matrix of the tuned model."""
    r   = results["__tuned__"]
    cm  = confusion_matrix(r["y_test"], r["y_pred"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Disease", "Heart Disease"],
        yticklabels=["No Disease", "Heart Disease"],
        linewidths=.5, linecolor="#2d3f54",
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title("Confusion Matrix · Tuned Model", color="white", pad=10)
    ax.set_ylabel("Actual", color="#90a4ae")
    ax.set_xlabel("Predicted", color="#90a4ae")
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> plt.Figure:
    """Grouped bar chart: Accuracy vs ROC-AUC for each model."""
    names = [k for k in results if k != "__tuned__"]
    accs  = [results[n]["accuracy"] for n in names]
    rocs  = [results[n]["roc_auc"]  for n in names]

    x     = np.arange(len(names))
    width = 0.38
    short = [n.replace(" ", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#4fc3f7", edgecolor="none")
    b2 = ax.bar(x + width/2, rocs, width, label="ROC-AUC",  color="#ff7043", edgecolor="none")

    for bars in [b1, b2]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color="white", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(short, color="white", fontsize=9)
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("Score", color="#90a4ae")
    ax.set_title("Model Comparison · Accuracy & ROC-AUC", color="white", fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e2a3a", labelcolor="white")
    ax.axhline(.85, color="#90a4ae", ls="--", lw=.8, alpha=.5)
    for spine in ax.spines.values():
        spine.set_color("#2d3f54")
    plt.tight_layout()
    return fig


# =============================================================================
# SIDEBAR — PATIENT INPUT
# =============================================================================

def render_sidebar() -> dict:
    """Render sidebar inputs and return a dict of patient features."""
    with st.sidebar:
        st.markdown("## 🩺 Patient Input Panel")
        st.markdown("Fill in the clinical data below, then click **Predict**.")
        st.markdown("---")

        st.markdown("**👤 Demographics**")
        age = st.slider("Age (years)", 20, 80, 54)
        sex = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        sex_val = 1 if sex.startswith("Male") else 0

        st.markdown("**💓 Cardiac Symptoms**")
        cp_map = {
            "Typical Angina (0)": 0,
            "Atypical Angina (1)": 1,
            "Non-Anginal Pain (2)": 2,
            "Asymptomatic (3)": 3,
        }
        cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        cp_val = cp_map[cp]

        exang_map = {"No (0)": 0, "Yes (1)": 1}
        exang = st.selectbox("Exercise-Induced Angina", list(exang_map.keys()))
        exang_val = exang_map[exang]

        thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)

        slope_map = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
        slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
        slope_val = slope_map[slope]

        st.markdown("**🔬 Lab Values**")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)

        fbs_map  = {"≤ 120 mg/dL (0)": 0, "> 120 mg/dL (1)": 1}
        fbs = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
        fbs_val = fbs_map[fbs]

        st.markdown("**📊 Diagnostic Tests**")
        ecg_map = {
            "Normal (0)": 0,
            "ST-T Abnormality (1)": 1,
            "LV Hypertrophy (2)": 2,
        }
        restecg = st.selectbox("Resting ECG", list(ecg_map.keys()))
        restecg_val = ecg_map[restecg]

        ca = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])

        thal_map = {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}
        thal = st.selectbox("Thalassemia", list(thal_map.keys()))
        thal_val = thal_map[thal]

        st.markdown("---")
        predict_btn = st.button("🔍  Predict Now", use_container_width=True, type="primary")

    return {
        "features": {
            "age": age, "sex": sex_val, "cp": cp_val, "trestbps": trestbps,
            "chol": chol, "fbs": fbs_val, "restecg": restecg_val,
            "thalach": thalach, "exang": exang_val, "oldpeak": oldpeak,
            "slope": slope_val, "ca": ca, "thal": thal_val,
        },
        "predict": predict_btn,
    }


# =============================================================================
# PREDICTION PANEL
# =============================================================================

def render_prediction(features: dict, scaler, model):
    """Run inference and render the result cards."""
    input_df  = pd.DataFrame([features])
    scaled    = scaler.transform(input_df)
    pred      = model.predict(scaled)[0]
    prob      = model.predict_proba(scaled)[0][1]
    risk, css = get_risk_level(prob)

    c1, c2, c3 = st.columns([1.6, 1.2, 1])

    with c1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h2>❤️‍🩹 Heart Disease Detected</h2>
                <p>The model indicates a significant risk of coronary artery disease.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h2>✅ No Heart Disease</h2>
                <p>The model finds no significant indicators of heart disease.</p>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
        fig_gauge = make_gauge(prob)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Probability",  f"{prob:.1%}")
        st.metric("Prediction",   "Disease" if pred == 1 else "Healthy")
        st.markdown(
            f"**Risk Level:** <span class='{css}'>{risk}</span>",
            unsafe_allow_html=True
        )

    # ── Progress bar ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Prediction Confidence</div>", unsafe_allow_html=True)
    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        st.caption("Heart Disease Probability")
        st.progress(float(prob))
    with bar_col2:
        st.caption("No Heart Disease Probability")
        st.progress(float(1 - prob))

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("📋 View Patient Input Summary"):
        labels = {
            "age": "Age", "sex": "Sex (1=M)", "cp": "Chest Pain",
            "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting BS",
            "restecg": "Resting ECG", "thalach": "Max HR", "exang": "Ex. Angina",
            "oldpeak": "ST Depression", "slope": "ST Slope", "ca": "Major Vessels",
            "thal": "Thalassemia",
        }
        summary = pd.DataFrame(
            [(labels[k], v) for k, v in features.items()],
            columns=["Feature", "Value"]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.info(
        "⚠️ **Medical Disclaimer:** This prediction is for educational and research purposes only. "
        "Always consult a qualified cardiologist for medical decisions.",
        icon="ℹ️"
    )


# =============================================================================
# TABS CONTENT
# =============================================================================

def render_tab_analytics(results: dict, feature_names: list, model):
    """Model analytics tab."""
    st.markdown("<div class='section-title'>📊 Model Comparison</div>", unsafe_allow_html=True)
    st.pyplot(plot_model_comparison(results), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>📈 ROC Curves</div>", unsafe_allow_html=True)
        st.pyplot(plot_roc_all(results), use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>🔲 Confusion Matrix</div>", unsafe_allow_html=True)
        st.pyplot(plot_confusion(results), use_container_width=True)

    st.markdown("<div class='section-title'>🌲 Feature Importances</div>", unsafe_allow_html=True)
    # Use Random Forest regardless of tuned model, for interpretability
    rf_model = results.get("Random Forest", {}).get("model")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        st.pyplot(plot_feature_importance(rf_model, feature_names), use_container_width=True)
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        top3 = importances.nlargest(3)
        st.markdown("**Top 3 Predictive Features:**")
        for feat, score in top3.items():
            st.markdown(f"- **{feat}** → importance score `{score:.4f}`")


def render_tab_dataset(df: pd.DataFrame):
    """Dataset overview tab."""
    st.markdown("<div class='section-title'>📋 Raw Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>📐 Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().round(2), use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>🔗 Correlation Heatmap</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(
            df.corr().round(2), mask=mask, annot=True, fmt=".1f",
            cmap="coolwarm", ax=ax, linewidths=.5,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}
        )
        ax.tick_params(colors="white")
        ax.set_title("Feature Correlation", color="white", pad=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div class='section-title'>🎯 Class Distribution</div>", unsafe_allow_html=True)
    counts = df["target"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    fig2.patch.set_facecolor("#0d1b2a")
    ax2.set_facecolor("#0d1b2a")
    ax2.bar(["No Disease", "Heart Disease"], counts.values,
            color=["#21c354", "#ff4b4b"], edgecolor="none", width=0.5)
    for i, v in enumerate(counts.values):
        ax2.text(i, v + 1, str(v), ha="center", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Count", color="#90a4ae")
    ax2.set_title("Target Distribution", color="white")
    for sp in ax2.spines.values():
        sp.set_color("#2d3f54")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


def render_tab_about():
    """Project description tab."""
    st.markdown("""
## 🫀 About CardioSense AI

**CardioSense AI** is a portfolio-grade machine learning application for predicting
the likelihood of heart disease from structured clinical data.

### 📦 Dataset
The model is trained on the **Cleveland Heart Disease dataset** from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease),
one of the most widely studied medical datasets in the ML community.

| Attribute        | Detail |
|------------------|--------|
| Samples          | 303 patients |
| Features         | 13 clinical variables |
| Target           | Binary (0 = no disease, 1 = disease) |
| Source           | Cleveland Clinic Foundation |

### 🤖 Machine Learning Pipeline
| Stage | Detail |
|-------|--------|
| Preprocessing | StandardScaler (zero mean, unit variance) |
| Evaluation | 5-fold stratified cross-validation |
| Models trained | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN |
| Selection | Best ROC-AUC score |
| Tuning | GridSearchCV on the winning model |

### 🔑 Key Features
- **13 clinical inputs** collected non-invasively or from routine lab work
- **Automatic best-model selection** via ROC-AUC
- **Hyperparameter tuning** for peak performance
- **Interpretability** through Random Forest feature importances
- **Risk stratification** into Low / Moderate / High

### ⚙️ Tech Stack
`Python 3.10+` · `Scikit-learn` · `Streamlit` · `Pandas` · `Matplotlib` · `Seaborn`
""")

    with st.expander("🔬 Feature Glossary — What each input means"):
        feature_info = {
            "age":      ("Age", "Patient age in years. Risk increases significantly after 45 for men and 55 for women."),
            "sex":      ("Sex", "1 = Male, 0 = Female. Men have a higher baseline risk."),
            "cp":       ("Chest Pain Type", "0=Typical angina · 1=Atypical angina · 2=Non-anginal pain · 3=Asymptomatic. Paradoxically, asymptomatic often correlates with disease in this dataset."),
            "trestbps": ("Resting Blood Pressure", "Measured in mm Hg on admission. Normal is < 120/80."),
            "chol":     ("Serum Cholesterol", "Total cholesterol in mg/dL. > 200 is borderline high."),
            "fbs":      ("Fasting Blood Sugar", "1 if > 120 mg/dL, else 0. Elevated fasting glucose is a diabetes marker."),
            "restecg":  ("Resting ECG", "0=Normal · 1=ST-T wave abnormality · 2=Left ventricular hypertrophy."),
            "thalach":  ("Max Heart Rate", "Maximum heart rate achieved during stress test. Lower values in older/sicker patients."),
            "exang":    ("Exercise-Induced Angina", "1=Yes, 0=No. Chest pain during exercise strongly suggests ischemia."),
            "oldpeak":  ("ST Depression (Oldpeak)", "ST depression induced by exercise relative to rest. Higher = more ischemia."),
            "slope":    ("ST Slope", "0=Upsloping (good) · 1=Flat · 2=Downsloping (worst prognosis)."),
            "ca":       ("Major Vessels (ca)", "Number of major vessels (0–3) colored by fluoroscopy. More = worse."),
            "thal":     ("Thalassemia", "1=Normal · 2=Fixed defect · 3=Reversible defect (most predictive)."),
        }
        rows = []
        for feat, (name, desc) in feature_info.items():
            rows.append({"Feature": feat, "Name": name, "Clinical Meaning": desc})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # ── Load data & train ─────────────────────────────────────────────────────
    df = load_data()
    scaler, tuned_model, results, feature_names, best_name, X_test, y_test = train_pipeline(df)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>🫀 CardioSense AI</h1>
        <p>Clinical-grade heart disease risk prediction powered by ensemble machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI metrics row ───────────────────────────────────────────────────────
    tuned = results["__tuned__"]
    m1, m2, m3, m4, m5 = st.columns(5)
    kpis = [
        ("🎯 Accuracy",  f"{tuned['accuracy']:.1%}"),
        ("📈 ROC-AUC",   f"{tuned['roc_auc']:.3f}"),
        ("🏆 Best Model", best_name.split()[0]),
        ("🗂️ Samples",   str(len(df))),
        ("🔢 Features",  str(len(feature_names))),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4, m5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar input ─────────────────────────────────────────────────────────
    user_input = render_sidebar()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_pred, tab_analytics, tab_data, tab_about = st.tabs([
        "🔍 Prediction", "📊 Model Analytics", "📋 Dataset", "ℹ️ About"
    ])

    with tab_pred:
        st.markdown("### 🩻 Patient Risk Assessment")
        st.markdown(
            "Configure the patient's clinical values in the **sidebar** and press "
            "**Predict Now** to run the model."
        )
        if user_input["predict"]:
            with st.spinner("Running inference…"):
                render_prediction(user_input["features"], scaler, tuned_model)
        else:
            st.markdown("""
            <div style="background:#1e2a3a;border-radius:12px;padding:2rem;
                        text-align:center;color:#90a4ae;margin-top:1rem;">
                <span style="font-size:3rem">🫀</span><br><br>
                <strong style="color:white;font-size:1.1rem">Ready for Prediction</strong><br>
                Fill in the patient data in the sidebar and click <strong>Predict Now</strong>.
            </div>
            """, unsafe_allow_html=True)

    with tab_analytics:
        render_tab_analytics(results, feature_names, tuned_model)

    with tab_data:
        render_tab_dataset(df)

    with tab_about:
        render_tab_about()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        Built with ❤️ using <strong>Streamlit</strong> · <strong>Scikit-learn</strong> · <strong>Python</strong><br>
        <em>CardioSense AI — Machine Learning Portfolio Project</em><br>
        ⚠️ For educational purposes only. Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
