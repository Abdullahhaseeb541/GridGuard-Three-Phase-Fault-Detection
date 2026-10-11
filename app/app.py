
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"

sys.path.insert(0, str(SRC))

from logistic import add_intercept, make_set_q, predict_probability
from load_data import load_raw
from prepare import prepare_data_raw

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

st.set_page_config(
    page_title="GridGuard | Transmission Fault Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL STYLING
# =========================================================
st.markdown("""
<style>
    :root {
        --navy: #10243A;
        --blue: #1769AA;
        --cyan: #39A0D6;
        --light-blue: #EAF3FB;
        --page: #F4F7FB;
        --border: #DCE5EF;
        --text: #172B42;
        --muted: #5E7187;
        --fault-bg: #FCE7E8;
        --fault-text: #8A1F2D;
        --ok-bg: #EAF8F2;
        --ok-text: #1B5E4A;
        --warn-bg: #FFF4E6;
        --warn-text: #8A5B0A;
    }

    html, body, [class*="css"] {
        font-size: 16px;
    }

    .stApp {
        background: var(--page);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.95);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.8rem;
        padding-bottom: 2.8rem;
    }

    .hero {
        background: linear-gradient(120deg, #10243A 0%, #174A75 60%, #1769AA 100%);
        padding: 30px 34px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 10px 26px rgba(16, 36, 58, 0.12);
    }

    .hero-eyebrow {
        color: #B9DDF8;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 10px 0 8px 0;
        line-height: 1.15;
    }

    .hero p {
        color: #E0ECF7;
        font-size: 1.07rem;
        margin: 0;
        line-height: 1.5;
    }

    .section-heading {
        color: var(--navy);
        font-size: 1.8rem;
        font-weight: 800;
        margin: 10px 0 4px 0;
        line-height: 1.2;
    }

    .section-description {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 18px;
        line-height: 1.5;
    }

    .model-overview {
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .model-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 160px;
        box-shadow: 0 4px 12px rgba(16, 36, 58, 0.04);
    }

    .model-card-title {
        color: var(--navy);
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .model-card p {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }

    .result-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px 16px 20px;
        min-height: 240px;
        box-shadow: 0 4px 12px rgba(16, 36, 58, 0.05);
    }

    .result-card h4 {
        margin: 0 0 8px 0;
        color: var(--navy);
        font-size: 1.15rem;
    }

    .result-badge {
        display: inline-block;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 7px 12px;
        margin-bottom: 18px;
    }

    .badge-fault {
        background: var(--fault-bg);
        color: var(--fault-text);
    }

    .badge-no-fault {
        background: var(--ok-bg);
        color: var(--ok-text);
    }

    .result-label {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1.15;
        color: var(--navy);
        margin: 0 0 10px 0;
    }

    .result-detail {
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.6;
        margin-top: 12px;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px 20px 10px;
        box-shadow: 0 4px 12px rgba(16, 36, 58, 0.03);
    }

    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stTextArea label {
        color: var(--navy) !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stNumberInputContainer"] {
        padding: 4px 0;
    }

    .stNumberInput input {
        font-size: 1.05rem;
        min-height: 48px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(16, 36, 58, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #52677E;
        font-weight: 700;
        font-size: 0.95rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 800;
        font-size: 1.4rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 800;
        min-height: 48px;
        font-size: 1.04rem;
        background: linear-gradient(180deg, #1A6AA4 0%, #154C7C 100%);
        color: white;
        border: none;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.03);
    }

    div[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div {
        font-size: 1rem;
    }

    .status-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
    }

    .status-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .status-value {
        color: var(--navy);
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 9px;
    }

    .status-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 5px;
    }

    .status-box {
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid transparent;
        margin-top: 10px;
        font-size: 1rem;
        line-height: 1.55;
    }

    .status-fault {
        background: var(--fault-bg);
        border-color: #F1B3B8;
        color: var(--fault-text);
    }

    .status-ok {
        background: var(--ok-bg);
        border-color: #B8E3D0;
        color: var(--ok-text);
    }

    .status-warn {
        background: var(--warn-bg);
        border-color: #F0D1A3;
        color: var(--warn-text);
    }

    .range-warning {
        background: #FFF1C2;
        border: 2px solid #9A5B00;
        border-left: 8px solid #9A5B00;
        border-radius: 12px;
        color: #4A2B00;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.55;
        margin: 12px 0;
        padding: 16px 20px;
    }

    .summary-table {
        font-size: 0.98rem;
    }

    .footer {
        color: #728197;
        font-size: 0.86rem;
        text-align: center;
        padding: 12px 0 0 0;
    }

    hr {
        border-color: var(--border);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS AND TRAINING DATA RANGES
# =========================================================
@st.cache_resource
def load_saved_model(filename):
    path = RESULTS / filename

    if not path.exists():
        return None

    return joblib.load(path)


@st.cache_data
def load_feature_ranges():
    df = load_raw()

    if "Output (S)" in df.columns:
        df = df.rename(columns={"Output (S)": "fault"})

    df = df.dropna(axis=1, how="all")
    df = df.loc[
        :, ~df.columns.astype(str).str.startswith("Unnamed")
    ]

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df[FEATURES].dropna()

    return (
        df.min().to_numpy(dtype=float),
        df.max().to_numpy(dtype=float),
    )


@st.cache_data
def load_training_measurements():
    X_train_raw, _, _, y_train, _, _ = prepare_data_raw()
    return X_train_raw, y_train


logistic_model = load_saved_model("logistic_Q.pkl")
rbf_saved = load_saved_model("svm_rbf.pkl")

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Electrical Analytics</div>
    <h1>GridGuard</h1>
    <p>Three-Phase Transmission Line Fault Detection</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚡ GridGuard")
    st.caption("TRANSMISSION LINE ANALYTICS")
    st.divider()

    st.markdown("**MODEL CONFIGURATION**")
    st.write("Logistic Regression · Set Q")
    st.write("Support Vector Machine · RBF Kernel")

    st.divider()

    st.markdown("**INPUT CHANNELS**")
    st.write("Ia · Ib · Ic — Phase currents")
    st.write("Va · Vb · Vc — Phase voltages")

    st.divider()

    st.markdown("**CLASS DEFINITIONS**")
    st.write("0 — No fault")
    st.write("1 — Fault")

    st.divider()
    st.caption(
        "AI4003 · Fall 2026\n\n"
        "Educational prototype. Not certified for operational "
        "electrical protection."
    )

# =========================================================
# MODEL AVAILABILITY CHECK
# =========================================================
if logistic_model is None or rbf_saved is None:
    st.error(
        "Required model files were not found. Run "
        "`python src/train_logistic.py` and "
        "`python src/train_svm.py` from the project root, "
        "then refresh this page."
    )
    st.stop()

try:
    data_min, data_max = load_feature_ranges()
except Exception as exc:
    st.error(
        "The dataset could not be loaded for input validation. "
        "Verify the dataset path and load_data.py configuration."
    )
    st.exception(exc)
    st.stop()

# =========================================================
# MODEL OVERVIEW
# =========================================================
st.markdown(
    '<div class="section-heading">01 &nbsp; Model Overview</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-description">'
    'Two trained classifiers are available for comparison on the same six '
    'measurement inputs. Each model uses the saved artifact and the '
    'corresponding inference preprocessing defined during training.'
    '</div>',
    unsafe_allow_html=True,
)

model_cols = st.columns(2)
with model_cols[0]:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-card-title">Logistic Regression · Set Q</div>
            <p>Uses six raw measurements and their six squared features created from the original signals before standardization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with model_cols[1]:
    st.markdown(
        """
        <div class="model-card">
            <div class="model-card-title">Support Vector Machine · RBF Kernel</div>
            <p>Uses the six standardized phase-current and phase-voltage measurements with the trained RBF SVM classifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown(
    '<div class="section-heading">02 &nbsp; Measurement Input</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-description">'
    'Enter phase currents in the dataset\u2019s raw units and phase voltages '
    'in per-unit (p.u.). No unit conversion is applied before inference.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("measurement_form"):
    st.markdown("**Phase Currents**")

    c1, c2, c3 = st.columns(3)

    with c1:
        ia = st.number_input(
            "Phase A Current (Ia, dataset units)", value=0.0, format="%.6f"
        )
    with c2:
        ib = st.number_input(
            "Phase B Current (Ib, dataset units)", value=0.0, format="%.6f"
        )
    with c3:
        ic = st.number_input(
            "Phase C Current (Ic, dataset units)", value=0.0, format="%.6f"
        )

    st.markdown("**Phase Voltages**")

    c4, c5, c6 = st.columns(3)

    with c4:
        va = st.number_input(
            "Phase A Voltage (Va, p.u.)", value=0.0, format="%.6f"
        )
    with c5:
        vb = st.number_input(
            "Phase B Voltage (Vb, p.u.)", value=0.0, format="%.6f"
        )
    with c6:
        vc = st.number_input(
            "Phase C Voltage (Vc, p.u.)", value=0.0, format="%.6f"
        )

    submitted = st.form_submit_button(
        "Analyze Measurements",
        type="primary",
        use_container_width=True,
    )

# =========================================================
# PREDICTION ENGINE
# =========================================================
if submitted:
    raw = np.array(
        [[ia, ib, ic, va, vb, vc]],
        dtype=float,
    )

    if not np.all(np.isfinite(raw)):
        st.error("All input measurements must be finite numeric values.")
        st.stop()
    st.session_state["analyzed_measurements"] = raw[0].tolist()

if "analyzed_measurements" in st.session_state:
    raw = np.asarray(
        [st.session_state["analyzed_measurements"]],
        dtype=float,
    )
    # Flag inputs outside the observed dataset range.
    outside = (raw[0] < data_min) | (raw[0] > data_max)

    if np.any(outside):
        out_features = [
            FEATURES[i] for i, is_outside in enumerate(outside)
            if is_outside
        ]
        st.markdown(
            '<div class="range-warning" role="alert">'
            '<strong>Outside the observed data range:</strong> '
            + ", ".join(out_features)
            + ". Predictions may be less reliable for these values."
            + "</div>",
            unsafe_allow_html=True,
        )

    logistic_mean = np.asarray(logistic_model["mean"], dtype=float)
    logistic_std = np.asarray(logistic_model["std"], dtype=float)
    logistic_std = np.where(logistic_std == 0, 1.0, logistic_std)

    X_q_raw = make_set_q(raw)
    X_q_standardized = (X_q_raw - logistic_mean) / logistic_std

    probability = float(
        predict_probability(
            add_intercept(X_q_standardized),
            logistic_model["theta"],
        )[0]
    )
    log_pred = int(probability >= 0.5)

    svm_mean = np.asarray(rbf_saved["mean"], dtype=float)
    svm_std = np.asarray(rbf_saved["std"], dtype=float)
    svm_std = np.where(svm_std == 0, 1.0, svm_std)
    svm_input = (raw - svm_mean) / svm_std
    svm_pred = int(rbf_saved["model"].predict(svm_input)[0])

    standardized = (raw - svm_mean) / svm_std

    # =====================================================
    # RESULTS DASHBOARD
    # =====================================================
    st.divider()
    st.markdown(
        '<div class="section-heading">03 &nbsp; Prediction Results</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">'
        'Both models are evaluated on the current six-measurement input. '
        'The result cards below show each prediction independently before '
        'the agreement check is applied.'
        '</div>',
        unsafe_allow_html=True,
    )

    agreement = log_pred == svm_pred
    logistic_status = "FAULT DETECTED" if log_pred else "NO FAULT DETECTED"
    svm_status = "FAULT DETECTED" if svm_pred else "NO FAULT DETECTED"
    logistic_class = "badge-fault" if log_pred else "badge-no-fault"
    svm_class = "badge-fault" if svm_pred else "badge-no-fault"

    result_cols = st.columns(2)

    with result_cols[0]:
        st.markdown(
            f"""
            <div class="result-card">
                <h4>Logistic Regression · Set Q</h4>
                <div class="result-badge {logistic_class}">{'fault' if log_pred else 'no fault'}</div>
                <div class="result-label">{logistic_status}</div>
                <div class="result-detail">
                    Probability estimate: <strong>{probability * 100:.2f}%</strong><br>
                    Uses the saved Set Q logistic model and the matching training-derived preprocessing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_cols[1]:
        st.markdown(
            f"""
            <div class="result-card">
                <h4>RBF Support Vector Machine</h4>
                <div class="result-badge {svm_class}">{'fault' if svm_pred else 'no fault'}</div>
                <div class="result-label">{svm_status}</div>
                <div class="result-detail">
                    Prediction produced by the saved RBF SVM trained on the six standardized electrical measurements.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading" style="font-size: 1.35rem; margin-top: 1.2rem;">04 &nbsp; Prediction Comparison</div>',
        unsafe_allow_html=True,
    )

    if agreement and log_pred == 1:
        st.markdown(
            """
            <div class="status-box status-fault">
                Both models predict FAULT for the submitted measurements. This is a model alert, not a certified electrical protection diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif agreement and log_pred == 0:
        st.markdown(
            """
            <div class="status-box status-ok">
                Both models predict NO FAULT for the submitted measurements. Agreement improves confidence, but this interface remains an educational prototype.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-box status-warn">
                The models disagree. Treat this case as uncertain and verify the input values before making any engineering interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Estimated fault probability")
    st.metric(
        "Logistic Regression · Set Q",
        f"{probability * 100:.2f}%",
        help=(
            "The saved logistic model's estimated probability that these "
            "measurements belong to the fault class. It is not a measure of "
            "fault severity or the physical amount of damage."
        ),
        border=True,
    )
    st.progress(
        float(np.clip(probability, 0.0, 1.0)),
        text=f"Estimated chance of fault class: {probability * 100:.2f}%",
    )
    st.caption(
        "This percentage comes from Logistic Regression. The RBF SVM provides "
        "a fault/no-fault classification only; it does not produce a probability."
    )

    logistic_reason = (
        f"Fault: estimated fault probability is {probability * 100:.2f}%, at or above the 50% decision threshold."
        if log_pred
        else f"No fault: estimated fault probability is {probability * 100:.2f}%, below the 50% decision threshold."
    )
    svm_reason = (
        "Fault: the standardized measurements fall on the RBF SVM's learned fault-class side of its decision boundary."
        if svm_pred
        else "No fault: the standardized measurements fall on the RBF SVM's learned no-fault side of its decision boundary."
    )
    st.markdown(
        "**Why these predictions?** "
        f"Logistic Regression — {logistic_reason}  \n"
        f"RBF SVM — {svm_reason}"
    )
    st.caption(
        "These are explanations of the models’ classification rules, not proof of a physical fault or its cause."
    )

    with st.expander("Explore prediction graphs and measurement values"):
        explanation_view = st.selectbox(
            "Choose a visualization",
            [
                "Logistic Regression feature contributions",
                "RBF SVM decision boundary",
                "Submitted values within training ranges",
            ],
        )

        if explanation_view == "Logistic Regression feature contributions":
            q_feature_order = logistic_model["feature_order"]
            q_contributions = (
                X_q_standardized[0] * logistic_model["theta"][1:]
            )
            contribution_order = np.argsort(np.abs(q_contributions))
            contribution_data = pd.DataFrame(
                {
                    "Feature": np.asarray(q_feature_order)[contribution_order],
                    "Standardized value": X_q_standardized[0][contribution_order],
                    "Learned weight": logistic_model["theta"][1:][contribution_order],
                    "Log-odds contribution": q_contributions[contribution_order],
                }
            )
            bar_colors = [
                "#B42332" if value > 0 else "#1769AA"
                for value in q_contributions[contribution_order]
            ]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(
                contribution_data["Feature"],
                contribution_data["Log-odds contribution"],
                color=bar_colors,
            )
            ax.axvline(0, color="#10243A", linewidth=1)
            ax.set_xlabel("Contribution to fault log-odds")
            ax.set_title("How Set Q features move the logistic prediction")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            st.pyplot(
                fig,
                clear_figure=True,
                alt="Horizontal contributions of the 12 Set Q features to the logistic model's fault log-odds",
            )
            plt.close(fig)
            st.dataframe(
                contribution_data.sort_values(
                    "Log-odds contribution",
                    key=lambda values: values.abs(),
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"Model intercept: {logistic_model['theta'][0]:.4f}. "
                "Positive bars push the model toward fault; negative bars push it toward no fault. "
                "The intercept plus all bars gives the model's log-odds. These are model contributions, not physical causes."
            )

        elif explanation_view == "RBF SVM decision boundary":
            training_raw, training_labels = load_training_measurements()
            feature_cols = st.columns(2)
            with feature_cols[0]:
                x_feature = st.selectbox(
                    "Horizontal axis",
                    FEATURES,
                    index=0,
                    key="svm_boundary_x",
                )
            with feature_cols[1]:
                y_feature = st.selectbox(
                    "Vertical axis",
                    [feature for feature in FEATURES if feature != x_feature],
                    index=0,
                    key="svm_boundary_y",
                )

            x_index = FEATURES.index(x_feature)
            y_index = FEATURES.index(y_feature)
            x_bounds = np.quantile(training_raw[:, x_index], [0.01, 0.99])
            y_bounds = np.quantile(training_raw[:, y_index], [0.01, 0.99])
            x_low = min(float(x_bounds[0]), float(raw[0, x_index]))
            x_high = max(float(x_bounds[1]), float(raw[0, x_index]))
            y_low = min(float(y_bounds[0]), float(raw[0, y_index]))
            y_high = max(float(y_bounds[1]), float(raw[0, y_index]))
            x_padding = max((x_high - x_low) * 0.06, 1e-9)
            y_padding = max((y_high - y_low) * 0.06, 1e-9)
            x_values = np.linspace(x_low - x_padding, x_high + x_padding, 120)
            y_values = np.linspace(y_low - y_padding, y_high + y_padding, 120)
            grid_x, grid_y = np.meshgrid(x_values, y_values)
            grid_raw = np.repeat(raw, grid_x.size, axis=0)
            grid_raw[:, x_index] = grid_x.ravel()
            grid_raw[:, y_index] = grid_y.ravel()
            grid_scaled = (grid_raw - svm_mean) / svm_std
            grid_predictions = rbf_saved["model"].predict(grid_scaled).reshape(
                grid_x.shape
            )
            grid_decisions = rbf_saved["model"].decision_function(
                grid_scaled
            ).reshape(grid_x.shape)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contourf(
                grid_x,
                grid_y,
                grid_predictions,
                levels=[-0.5, 0.5, 1.5],
                colors=["#DCEBFA", "#F9DCDD"],
                alpha=0.85,
            )
            if grid_decisions.min() <= 0 <= grid_decisions.max():
                ax.contour(
                    grid_x,
                    grid_y,
                    grid_decisions,
                    levels=[0],
                    colors=["#10243A"],
                    linewidths=2,
                )

            rng = np.random.default_rng(42)
            plotted_indices = np.concatenate(
                [
                    rng.choice(class_indices, min(500, len(class_indices)), replace=False)
                    for class_indices in (
                        np.flatnonzero(training_labels == 0),
                        np.flatnonzero(training_labels == 1),
                    )
                    if len(class_indices) > 0
                ]
            )
            for label, color, label_text in (
                (0, "#1769AA", "Training: no fault"),
                (1, "#B42332", "Training: fault"),
            ):
                class_indices = plotted_indices[training_labels[plotted_indices] == label]
                ax.scatter(
                    training_raw[class_indices, x_index],
                    training_raw[class_indices, y_index],
                    s=12,
                    alpha=0.35,
                    color=color,
                    label=label_text,
                    edgecolors="none",
                )
            ax.scatter(
                raw[0, x_index],
                raw[0, y_index],
                marker="*",
                s=260,
                color="#F2A900",
                edgecolor="#10243A",
                linewidth=1.2,
                label=f"Your input: {'fault' if svm_pred else 'no fault'}",
                zorder=5,
            )
            ax.set_xlabel(f"{x_feature} (raw measurement)")
            ax.set_ylabel(f"{y_feature} (raw measurement)")
            ax.set_title("RBF SVM classification slice at your other four values")
            ax.legend(loc="best")
            ax.grid(alpha=0.15)
            fig.tight_layout()
            st.pyplot(
                fig,
                clear_figure=True,
                alt=f"RBF SVM fault and no-fault regions across {x_feature} and {y_feature}, with the submitted measurement marked",
            )
            plt.close(fig)
            st.caption(
                "Shaded regions show the SVM's predicted class. The dark curve is its decision boundary when visible. "
                "The other four inputs are held at your submitted values; training points are shown as context. "
                "This is a two-feature slice of a six-feature model, not a complete physical fault explanation."
            )

        else:
            training_raw, _ = load_training_measurements()
            fig, axes = plt.subplots(
                nrows=len(FEATURES),
                ncols=1,
                figsize=(10, 7),
            )
            for index, (feature, axis) in enumerate(zip(FEATURES, axes)):
                observed_min = float(training_raw[:, index].min())
                observed_max = float(training_raw[:, index].max())
                current_value = float(raw[0, index])
                span = observed_max - observed_min
                padding = max(span * 0.08, abs(observed_min) * 0.02, 1e-9)
                axis.set_xlim(
                    min(observed_min, current_value) - padding,
                    max(observed_max, current_value) + padding,
                )
                axis.hlines(
                    0,
                    observed_min,
                    observed_max,
                    color="#91A4B8",
                    linewidth=8,
                    label="Training range",
                )
                axis.scatter(
                    current_value,
                    0,
                    s=90,
                    color="#B42332" if outside[index] else "#1769AA",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=3,
                    label="Submitted value",
                )
                axis.set_yticks([])
                axis.set_title(
                    f"{feature}: submitted {current_value:.6g} | "
                    f"training range {observed_min:.6g} to {observed_max:.6g}",
                    loc="left",
                    fontsize=9,
                )
                axis.grid(axis="x", alpha=0.2)
            axes[0].legend(loc="upper right", ncol=2, frameon=False)
            fig.suptitle("Submitted measurement locations vs. training ranges")
            fig.tight_layout()
            st.pyplot(
                fig,
                clear_figure=True,
                alt="Submitted six electrical measurements compared with their training data ranges",
            )
            plt.close(fig)
            st.caption(
                "Blue markers are inside the training range; red markers are outside it. "
                "Each row has its own raw-value scale so current and voltage values remain readable."
            )

    # =====================================================
    # INPUT REVIEW
    # =====================================================
    with st.expander("Review Input Measurements and Standardization"):
        input_table = pd.DataFrame({
            "Feature": FEATURES,
            "Raw Measurement": raw[0],
            "Standardized Value": standardized[0],
            "Within Observed Dataset Range": [
                "Yes" if not value else "No"
                for value in outside
            ],
        })

        st.dataframe(
            input_table,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Model Interpretation and Limitations"):
        st.markdown("""
        - **Fault (1):** The model predicts the fault class.
        - **No Fault (0):** The model predicts the no-fault class.
        - **Logistic probability:** An estimate produced by the trained
          Logistic Regression model; calibration has not been established
          by this interface.
        - **Model agreement:** Agreement does not prove the prediction is
          correct. Both models may make the same mistake.
        - **Input range:** Values outside the observed dataset range are
          flagged because the models may be extrapolating.
        - **Operational safety:** This interface is a student project and
          must not replace certified relays, protection systems, or
          qualified engineering assessment.
        """)

    st.markdown(
        '<div class="section-heading" style="font-size: 1.35rem; margin-top: 1.25rem;">05 &nbsp; Measurement Summary</div>',
        unsafe_allow_html=True,
    )
    summary_df = pd.DataFrame(
        {
            "Measurement": FEATURES,
            "Submitted Value": [float(value) for value in raw[0]],
        }
    )
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Measurement": st.column_config.TextColumn("Measurement"),
            "Submitted Value": st.column_config.NumberColumn(
                "Submitted Value",
                format="%.6f",
            ),
        },
    )

else:
    st.info(
        "Enter all six measurements and select "
        "**Analyze Measurements** to view the model predictions."
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown("""
<div class="footer">
    GRIDGUARD &nbsp;|&nbsp; AI4003 MACHINE LEARNING PROJECT
    <br>
    Educational use only · Not a certified electrical protection system
</div>
""", unsafe_allow_html=True)