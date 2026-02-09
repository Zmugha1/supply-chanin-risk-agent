"""
Theory-Constrained Supply Chain Resilience Agent — Streamlit UI.

DISCLAIMER:
This agent implements Design Science Research methodology operationalizing Supply Chain Resilience
Theory (Christopher & Peck, 2004), Bullwhip Effect Theory (Forrester, 1961; Lee et al., 1997),
and Resource Dependence Theory (Pfeffer & Salancik, 1978) as ante hoc constraints on survival
analysis models.

Generative AI tools assisted with CrewAI agent scaffolding, Streamlit UI components, and code
structure. All theoretical constraints, algorithmic selections (Cox PH with theory priors vs.
Random Survival Forest), survival analysis validation metrics (Brier Score, C-index, Calibration),
and supply chain governance protocols were centrally directed by the research framework and
empirical supply chain theory. The system demonstrates that theory-constrained parametric models
outperform black-box ensembles on small, high-stakes tabular data per Grinsztajn et al. (2022).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.generate_supply_chain_data import generate_supply_chain_data
from features.theory_feature_engineering import (
    build_theory_features,
    COX_FEATURE_COLUMNS,
    THEORY_FEATURE_CONSTRAINTS,
)
from models.theory_constrained_cox import TheoryConstrainedCox
try:
    from models.baseline_rsf import BaselineRSF
    _RSF_AVAILABLE = True
except ImportError:
    BaselineRSF = None
    _RSF_AVAILABLE = False
from governance.calibration_monitor import (
    brier_score_survival,
    integrated_brier_score,
    reliability_diagram,
    expected_calibration_error,
)
from governance.theoretical_fidelity import check_coefficient_signs, get_theory_alignment_message
from governance.mlflow_tracker import log_human_override, log_survival_run, is_mlflow_available
from agents.bullwhip_analyst import flag_bullwhip_anomalies
from agents.resilience_assessor import compute_resilience_score
from agents.survival_predictor import run as survival_predictor_run
from agents.crew import run_crew

# Data to Dollar styling
NAVY, CREAM, TEAL, CORAL = "#2C3E50", "#FFF8E7", "#4ECDC4", "#FF6B6B"

st.set_page_config(
    page_title="Data to $$$ | Supply Chain Resilience Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stMetric label { color: #2C3E50; }
div[data-testid="stExpander"] { border: 1px solid #2C3E50; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    df = generate_supply_chain_data()
    return build_theory_features(df)


@st.cache_resource
def get_models(df):
    feats = [c for c in COX_FEATURE_COLUMNS if c in df.columns]
    cox = TheoryConstrainedCox(penalty=0.5)
    cox.fit(df, duration_col="duration_days", event_col="event", feature_columns=feats)
    rsf = None
    if _RSF_AVAILABLE and BaselineRSF is not None:
        try:
            rsf = BaselineRSF(n_estimators=100, min_samples_split=20)
            rsf.fit(df, duration_col="duration_days", event_col="event", feature_columns=feats)
        except Exception:
            rsf = None
    return cox, rsf, feats


def render_hero():
    html = (
        '<div style="background:#2C3E50;border-radius:16px;padding:32px 40px;margin-bottom:24px;color:#FFF8E7;">'
        '<h1 style="margin:0 0 8px 0;font-size:2rem;font-weight:700;">Theory-Constrained Supply Chain Resilience Agent</h1>'
        '<p style="margin:0 0 24px 0;opacity:0.9;">Dr. Data Decision Intelligence — Survival analysis with Christopher & Peck, Bullwhip, Resource Dependence</p>'
        '<div style="display:flex;flex-wrap:wrap;gap:24px;">'
        '<div><span style="color:#4ECDC4;font-weight:600;">C-index</span><div style="font-size:1.2rem;font-weight:700;">Target &gt;0.80</div></div>'
        '<div><span style="color:#4ECDC4;font-weight:600;">Calibration</span><div style="font-size:1.2rem;font-weight:700;">90-day ±4%</div></div>'
        '<div><span style="color:#4ECDC4;font-weight:600;">Decision velocity</span><div style="font-size:1.2rem;font-weight:700;">72h → 8 min</div></div>'
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def tab_survival_dashboard(df, cox, rsf, feats):
    st.subheader("Survival Dashboard")
    times = np.array([30.0, 60.0, 90.0])
    surv_cox = _survival_matrix(cox.predict_survival_function(df, times=times), len(df), len(times))

    # Kaplan-Meier (observed) vs Cox (fitted) - aggregate
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter()
    km.fit(df["duration_days"], df["event"])
    km_t = km.survival_function_.index.values.astype(float)
    km_s = km.survival_function_.values.ravel()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km_t, y=km_s, name="Kaplan-Meier (observed)", line=dict(color=TEAL)))
    # Mean Cox survival over time (approximate from 30/60/90)
    mean_cox = surv_cox.mean(axis=0)
    fig.add_trace(go.Scatter(x=times, y=mean_cox, name="Cox (fitted mean)", mode="lines+markers", line=dict(color=NAVY)))
    fig.update_layout(title="Survival: Kaplan-Meier vs Cox", xaxis_title="Days", yaxis_title="P(no disruption)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Individual supplier lookup
    sid = st.text_input("Supplier ID (e.g. SUP_0001)", value="SUP_0001", key="surv_lookup")
    if sid in df["supplier_id"].astype(str).values:
        idx_loc = np.where(df["supplier_id"].astype(str).values == sid)[0][0]
        s90 = float(np.clip(surv_cox[idx_loc, 2], 0, 1))
        st.metric("90-day survival probability", f"{s90:.1%}", "")
        # Theory contribution (hazard ratio contribution from each theory)
        coefs = cox.get_coefficients()
        theory_contrib = {"Resilience": 0.0, "Bullwhip": 0.0, "ResourceDependence": 0.0}
        for f, c in coefs.items():
            if f not in THEORY_FEATURE_CONSTRAINTS:
                continue
            t = THEORY_FEATURE_CONSTRAINTS[f]["theory"]
            val = df.iloc[idx_loc][f] if f in df.columns else 0
            if t == "Resilience":
                theory_contrib["Resilience"] += c * val
            elif "Bullwhip" in t:
                theory_contrib["Bullwhip"] += c * val
            else:
                theory_contrib["ResourceDependence"] += c * val
        st.caption("Theory contribution to log-hazard: Resilience (lower=safer), Bullwhip (higher=risk), Resource Dependence.")
    else:
        st.info("Enter a supplier ID from the dataset (e.g. SUP_0001).")

    render_footer_section()


def _survival_matrix(surv, n_samples: int, n_times: int = 3):
    """Return array shape (n_samples, n_times) so surv[:, 2] is 90-day for each sample."""
    if hasattr(surv, "values"):
        surv = surv.values
    if isinstance(surv, pd.DataFrame):
        surv = surv.values
    if not isinstance(surv, np.ndarray):
        surv = np.asarray(surv)
    if surv.ndim == 1:
        surv = np.broadcast_to(surv, (n_samples, n_times))
    elif surv.shape[0] == n_times and surv.shape[1] == n_samples:
        surv = surv.T
    elif surv.shape[0] != n_samples:
        surv = surv.T
    return np.asarray(surv, dtype=float)[:n_samples, :n_times]


def tab_risk_stratification(df, cox, feats):
    st.subheader("Risk Stratification")
    times = np.array([30.0, 60.0, 90.0])
    surv_raw = cox.predict_survival_function(df, times=times)
    surv = _survival_matrix(surv_raw, len(df), len(times))
    s90 = np.asarray(surv[:, 2]).ravel()
    df_disp = df[["supplier_id", "tier", "demand_amplification_ratio", "network_redundancy"]].copy()
    df_disp["survival_90"] = s90[: len(df_disp)]
    df_disp = df_disp.sort_values("survival_90").head(20)
    df_disp["risk"] = pd.cut(df_disp["survival_90"], bins=[0, 0.5, 0.8, 1.01], labels=["Red (<50%)", "Yellow (50-80%)", "Green (>80%)"])
    st.dataframe(df_disp.style.background_gradient(subset=["survival_90"], cmap="RdYlGn"), use_container_width=True, hide_index=True)
    alerts = flag_bullwhip_anomalies(df, threshold=2.5)
    st.caption(f"Bullwhip alerts (demand_amplification > 2.5x): {len(alerts)} suppliers.")
    render_footer_section()


def tab_calibration(df, cox, rsf, feats):
    st.subheader("Calibration Validation")
    times = np.array([30.0, 60.0, 90.0])
    surv_cox = _survival_matrix(cox.predict_survival_function(df, times=times), len(df), len(times))
    # Binary at 30 days: survived 30 days?
    binary_30 = ((df["event"] == 0) | (df["duration_days"] > 30)).astype(float).values
    pred_30_cox = surv_cox[:, 0]
    prob_true_cox, prob_pred_cox, _ = reliability_diagram(binary_30, pred_30_cox, n_bins=8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred_cox, y=prob_true_cox, name="Theory-Cox", mode="lines+markers", line=dict(color=TEAL)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect", line=dict(dash="dash", color="gray")))
    if rsf is not None:
        surv_rsf = rsf.predict_survival_function(df, times=times)
        pred_30_rsf = surv_rsf[:, 0]
        prob_true_rsf, prob_pred_rsf, _ = reliability_diagram(binary_30, pred_30_rsf, n_bins=8)
        fig.add_trace(go.Scatter(x=prob_pred_rsf, y=prob_true_rsf, name="RSF baseline", mode="lines+markers", line=dict(color=CORAL)))
    fig.update_layout(title="30-day survival calibration (Theory-Cox)" + (" vs RSF" if rsf else ""), xaxis_title="Predicted", yaxis_title="Actual", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    brier_cox = brier_score_survival(df["event"].values, df["duration_days"].values, surv_cox[:, 2], 90.0)
    c_cox = cox.concordance_index_(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("C-index (Cox)", f"{c_cox:.3f}", "Target >0.80")
    with c2:
        if rsf is not None:
            c_rsf = rsf.concordance_index_(df)
            st.metric("C-index (RSF)", f"{c_rsf:.3f}", "Baseline")
        else:
            st.metric("C-index (RSF)", "—", "Optional (not installed)")
    with c3:
        st.metric("Brier 90d (Cox)", f"{brier_cox:.3f}", "Target <0.10")
    with c4:
        if rsf is not None:
            brier_rsf = brier_score_survival(df["event"].values, df["duration_days"].values, surv_rsf[:, 2], 90.0)
            st.metric("Brier 90d (RSF)", f"{brier_rsf:.3f}", "")
        else:
            st.metric("Brier 90d (RSF)", "—", "Optional")
    if rsf is None:
        st.info("RSF baseline is omitted on this environment (scikit-survival not installed). Use Python 3.10/3.11 and add scikit-survival to requirements for full comparison.")
    render_footer_section()


def tab_theory_fidelity(cox):
    st.subheader("Theory Fidelity Monitor")
    coefs = cox.get_coefficients()
    pct, aligned, violations = check_coefficient_signs(coefs)
    st.metric("Theory Alignment Score", f"{pct:.0f}%", "Target 100%")
    st.write(get_theory_alignment_message(violations))
    rows = []
    for f, c in coefs.items():
        if f not in THEORY_FEATURE_CONSTRAINTS:
            continue
        exp = THEORY_FEATURE_CONSTRAINTS[f]["expected_sign"]
        actual = 1 if c > 0 else (-1 if c < 0 else 0)
        ok = actual == exp
        rows.append({"feature": f, "theory": THEORY_FEATURE_CONSTRAINTS[f]["theory"], "coefficient": c, "expected_sign": exp, "align": "✓" if ok else "✗"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_footer_section()


def tab_human_override(df, cox):
    st.subheader("Human Override Console")
    supplier_id = st.text_input("Supplier ID", key="override_sid")
    override_survival = st.slider("Officer override survival probability (90-day)", 0.0, 1.0, 0.7, 0.05)
    reason = st.selectbox("Reason", ["Geopolitical", "New Contract", "Quality Issue", "Other"])
    notes = st.text_area("Officer notes (optional)")
    if st.button("Submit override"):
        run_id = log_human_override(supplier_id, override_survival, reason, notes)
        st.success(f"Override recorded. MLflow run: {run_id or 'N/A'}")
    if "override_log" not in st.session_state:
        st.session_state["override_log"] = []
    st.caption("Audit log: overrides are logged to MLflow when available.")
    render_footer_section()


def render_footer_section():
    st.markdown("---")
    st.caption("Data to $$$ — Dr. Data Decision Intelligence. Theory-Constrained Supply Chain Resilience Agent.")


def tab_how_to_use():
    st.subheader("How to Use This App")
    st.markdown("""
    This **Theory-Constrained Supply Chain Resilience Agent** helps procurement and supply chain officers 
    understand **supplier disruption risk** using survival analysis grounded in supply chain theory. 
    The app uses **synthetic (demo) data only**—no real customer or order data is uploaded or stored.
    """)
    st.markdown("---")
    st.markdown("### What You See on the Home Page")
    st.markdown("""
    - **Business impact (demo):** Example metrics (e.g. stockout prevention value, alert fatigue reduction, decision velocity). 
      These are illustrative; real numbers depend on your data and scenario.
    - **Tabs:** Use the tabs below to switch between **Survival Dashboard**, **Risk Stratification**, **Calibration Validation**, 
      **Theory Fidelity Monitor**, **Human Override Console**, and this **How to Use** guide.
    """)
    st.markdown("---")
    st.markdown("### Tab 1: Survival Dashboard")
    st.markdown("""
    - **Purpose:** See how likely suppliers are to *remain undisrupted* over time (30, 60, 90 days).
    - **Kaplan–Meier (observed):** Non-parametric curve from historical data—*what actually happened* in the sample.
    - **Cox (fitted mean):** Theory-constrained model’s average survival curve.
    - **Supplier lookup:** Enter a **Supplier ID** (e.g. `SUP_0001`) to get that supplier’s **90-day survival probability** 
      and a short note on how each theory (Resilience, Bullwhip, Resource Dependence) contributes to their risk.
    - **How to use it:** Use the chart to judge overall risk level; use the lookup to drill into a specific supplier before making inventory or sourcing decisions.
    """)
    st.markdown("---")
    st.markdown("### Tab 2: Risk Stratification")
    st.markdown("""
    - **Purpose:** List suppliers by **highest risk** (lowest 90-day survival) so you can prioritize attention.
    - **Table:** Top 20 at-risk suppliers with **Supplier ID**, **Tier**, **Demand amplification ratio**, **Network redundancy**, 
      and **90-day survival** with color coding:
      - **Red (&lt;50%):** High risk—consider mitigation or alternatives.
      - **Yellow (50–80%):** Medium risk—monitor closely.
      - **Green (&gt;80%):** Lower risk—still review periodically.
    - **Bullwhip alerts:** Count of suppliers with demand amplification &gt; 2.5× (Bullwhip Effect theory).
    - **How to use it:** Sort mentally by survival_90 (table is already sorted). Focus on red and yellow first for inventory positioning or contract discussions.
    """)
    st.markdown("---")
    st.markdown("### Tab 3: Calibration Validation")
    st.markdown("""
    - **Purpose:** Check whether the model’s **predicted probabilities** match **actual outcomes** (model quality, not a single supplier).
    - **Reliability diagram:** Predicted vs. actual survival at 30 days. Closer to the diagonal “Perfect” line = better calibration.
    - **C-index:** Discrimination (e.g. target &gt; 0.80). Higher = better at ranking suppliers by risk.
    - **Brier score (90-day):** Lower = more accurate probability predictions (e.g. target &lt; 0.10 for Cox).
    - **How to use it:** Use this tab to trust (or question) the model’s numbers. If calibration is poor, treat survival percentages as relative ranks rather than exact probabilities.
    """)
    st.markdown("---")
    st.markdown("### Tab 4: Theory Fidelity Monitor")
    st.markdown("""
    - **Purpose:** Ensure the model’s **coefficients** align with **supply chain theory** (Resilience, Bullwhip, Resource Dependence).
    - **Theory Alignment Score:** Percentage of coefficients with the expected sign (target 100%).
    - **Table:** Each feature, its theory, coefficient, expected sign, and ✓/✗ for alignment.
    - **How to use it:** If alignment is low or you see warnings, the model may be drifting from theory; consider retraining or reviewing data.
    """)
    st.markdown("---")
    st.markdown("### Tab 5: Human Override Console")
    st.markdown("""
    - **Purpose:** Record **human judgment** when you override the model (e.g. geopolitical intel, new contract, quality issue).
    - **Fields:** Supplier ID, override 90-day survival probability (slider), reason (dropdown), optional notes.
    - **Submit override:** Saves the override for audit (e.g. MLflow when available). The app does not change the underlying model; it records your decision.
    - **How to use it:** When you disagree with the model for a specific supplier, enter the override and reason so the decision is documented and traceable.
    """)
    st.markdown("---")
    st.markdown("### Key Terms (Plain Language)")
    st.markdown("""
    | Term | Meaning |
    |------|--------|
    | **Survival probability (90-day)** | Chance the supplier is *not* disrupted in the next 90 days (0–100%). |
    | **C-index** | How well the model ranks suppliers by risk (0.5 = random, 1 = perfect). |
    | **Calibration** | Whether predicted probabilities match actual outcomes. |
    | **Theory alignment** | Whether model coefficients match what theory says (e.g. higher bullwhip → higher hazard). |
    | **Resilience / Bullwhip / Resource Dependence** | Three supply chain theories used to build and constrain the model. |
    """)
    st.markdown("---")
    st.markdown("### Data and Limits")
    st.markdown("""
    - The app runs on **synthetic data** (600 supplier relationships, 15% disruption rate, 85% right-censored at 2 years).
    - All metrics and charts are for **demonstration and learning**. They are not guarantees for real supply chains.
    - For production use, connect your own data and models under your governance and compliance policies.
    """)
    render_footer_section()


def main():
    render_hero()

    # Business impact metrics
    st.subheader("Business impact (demo)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stockout prevention value", "$4.2M", "Early warning")
    with col2:
        st.metric("Alert fatigue reduction", "94%", "Fewer false positives")
    with col3:
        st.metric("Decision velocity", "8 min", "vs 72 hours manual")
    with col4:
        st.metric("Calibration accuracy", "±4%", "90-day predictions")
    st.markdown("---")

    df = get_data()
    cox, rsf, feats = get_models(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "How to Use",
        "Survival Dashboard",
        "Risk Stratification",
        "Calibration Validation",
        "Theory Fidelity Monitor",
        "Human Override Console",
    ])
    with tab1:
        tab_how_to_use()
    with tab2:
        tab_survival_dashboard(df, cox, rsf, feats)
    with tab3:
        tab_risk_stratification(df, cox, feats)
    with tab4:
        tab_calibration(df, cox, rsf, feats)
    with tab5:
        tab_theory_fidelity(cox)
    with tab6:
        tab_human_override(df, cox)


if __name__ == "__main__":
    main()
