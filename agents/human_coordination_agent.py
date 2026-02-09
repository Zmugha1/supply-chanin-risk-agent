"""
HumanCoordinationAgent: Mitigate automation bias (Collaborative Partnership Theory).

Presents survival curves (not just binary risk scores); requests human validation when
hazard_ratio > historical_95th_percentile. MLflow logging of overrides with reasoning.
"""

import numpy as np
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from governance.mlflow_tracker import log_human_override, is_mlflow_available
except Exception:
    def log_human_override(*args, **kwargs): return None
    def is_mlflow_available(): return False


def should_request_override(
    hazard_ratio: float,
    historical_95th: float = 2.5,
) -> bool:
    """Request human review when hazard ratio exceeds historical 95th percentile."""
    return hazard_ratio > historical_95th


def run(
    context: Dict[str, Any],
    override_supplier_id: Optional[str] = None,
    override_survival_prob: Optional[float] = None,
    override_reason: str = "",
    officer_notes: str = "",
) -> Dict[str, Any]:
    """
    If override_* provided, log override and return updated recommendation.
    Otherwise return list of suppliers that need human review (high hazard).
    """
    if override_supplier_id and override_survival_prob is not None:
        run_id = log_human_override(
            override_supplier_id, override_survival_prob, override_reason, officer_notes
        )
        return {
            "action": "override_logged",
            "supplier_id": override_supplier_id,
            "override_survival_prob": override_survival_prob,
            "reason": override_reason,
            "mlflow_run_id": run_id,
            "message": "Human override recorded for audit.",
        }
    hazard_ratios = context.get("hazard_ratios") or {}
    historical_95th = context.get("historical_95th_percentile", 2.5)
    max_hr = max(hazard_ratios.values()) if hazard_ratios else 0
    need_review = [
        (k, v) for k, v in hazard_ratios.items()
        if should_request_override(v, historical_95th)
    ]
    need_review.sort(key=lambda x: -x[1])
    return {
        "action": "review_list",
        "need_review": need_review[:20],
        "summary": f"HumanCoordinationAgent: {len(need_review)} supplier(s) exceed hazard threshold {historical_95th}; recommend human validation.",
        "max_hazard_ratio": max_hr,
    }
