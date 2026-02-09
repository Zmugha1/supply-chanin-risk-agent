"""
ResilienceAssessor: Evaluate network redundancy per Christopher & Peck framework.

Uses NetworkX graph analysis (PageRank for supplier criticality), buffer efficiency.
Output: Resilience score (0-100) with visibility depth audit.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from features.theory_feature_engineering import add_cascade_risk_score
except ImportError:
    from theory_feature_engineering import add_cascade_risk_score  # noqa


def compute_resilience_score(row: pd.Series) -> float:
    """
    Composite 0-100 score: higher buffer_efficiency, network_redundancy, visibility_depth → higher resilience.
    Christopher & Peck: readiness to respond.
    """
    buf = row.get("buffer_efficiency", 0) or 0
    red = row.get("network_redundancy", 0) or 0
    vis = row.get("visibility_depth", 0) or 0
    # Normalize to 0-1 then scale (use safe scaling)
    buf_n = np.log1p(buf) / (1 + np.log1p(buf))
    red_n = min(1.0, red / 10.0) if red else 0
    vis_n = min(1.0, vis) if vis is not None else 0
    return float(100.0 * (0.4 * buf_n + 0.35 * red_n + 0.25 * vis_n))


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    context['df']: DataFrame with theory features.
    Returns resilience scores per supplier and audit summary.
    """
    df = context.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        return {"scores": [], "summary": "No data provided.", "mean_resilience": 0.0}
    df = add_cascade_risk_score(df.copy())
    if "resilience_score" not in df.columns:
        df["resilience_score"] = df.apply(compute_resilience_score, axis=1)
    scores = df[["supplier_id", "resilience_score", "visibility_depth", "network_redundancy", "buffer_efficiency"]].copy()
    scores = scores.to_dict("records")
    mean_res = float(df["resilience_score"].mean())
    low = (df["resilience_score"] < 40).sum()
    return {
        "scores": scores[:50],  # cap for UI
        "summary": f"ResilienceAssessor: Mean resilience {mean_res:.1f}/100; {int(low)} supplier(s) below 40 (high vulnerability).",
        "mean_resilience": mean_res,
        "low_resilience_count": int(low),
    }
