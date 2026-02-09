"""
SurvivalPredictor: Run theory-constrained Cox PH model.

Output: Survival curves (P(no disruption) vs. time), hazard ratios with confidence intervals.
Uses models.theory_constrained_cox and features.theory_feature_engineering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.theory_feature_engineering import COX_FEATURE_COLUMNS, build_theory_features
from models.theory_constrained_cox import TheoryConstrainedCox

DURATION_COL = "duration_days"
EVENT_COL = "event"
SURVIVAL_TIMES = np.array([30.0, 60.0, 90.0])


def run(context: dict) -> dict:
    """
    context['df']: DataFrame with duration_days, event, and theory features.
    context.get('model'): optional pre-fitted TheoryConstrainedCox; else fit one.
    Returns survival predictions, c-index, hazard ratio summary.
    """
    df = context.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        return {"survival_at_90": {}, "c_index": 0.0, "summary": "No data.", "hazard_ratios": {}}
    df = build_theory_features(df)
    feats = [c for c in COX_FEATURE_COLUMNS if c in df.columns]
    if not feats:
        return {"survival_at_90": {}, "c_index": 0.0, "summary": "Missing feature columns.", "hazard_ratios": {}}
    model = context.get("model")
    if model is None:
        model = TheoryConstrainedCox(penalty=0.5)
        model.fit(df, duration_col=DURATION_COL, event_col=EVENT_COL, feature_columns=feats)
    survival = model.predict_survival_function(df, times=SURVIVAL_TIMES)
    # survival shape: (n_times,) or (n_samples, n_times) depending on lifelines version
    if hasattr(survival, "values"):
        survival = survival.values
    if isinstance(survival, np.ndarray):
        if survival.ndim == 1:
            survival = np.broadcast_to(survival, (len(df), len(SURVIVAL_TIMES)))
    else:
        survival = np.array(survival)
    if survival.shape[0] != len(df):
        survival = survival.T
    c_index = model.concordance_index_(df, DURATION_COL, EVENT_COL)
    summary_coef = model.get_coefficients()
    hazard_ratios = {k: np.exp(v) for k, v in summary_coef.items()}
    survival_at_90 = {}
    if survival.shape[1] >= 3:  # 30, 60, 90
        for i, sid in enumerate(df["supplier_id"].astype(str)):
            survival_at_90[sid] = float(np.clip(survival[i, 2], 0, 1))
    return {
        "survival_at_90": survival_at_90,
        "survival_matrix": survival,
        "times": SURVIVAL_TIMES.tolist(),
        "c_index": float(c_index),
        "summary": f"SurvivalPredictor: C-index {c_index:.3f}; survival curves at 30/60/90 days.",
        "hazard_ratios": hazard_ratios,
        "coefficients": summary_coef,
    }
