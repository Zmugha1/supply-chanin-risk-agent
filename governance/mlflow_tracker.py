"""
MLflow tracking: experiment logging, theoretical fidelity, survival model registry, human override audit.

Supply chain lineage: SKU → Supplier → Theory features → Survival probability → Inventory decision.
"""

import os
from typing import Any, Dict, Optional
import json

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

EXPERIMENT_NAME = "supply_chain_resilience"
RUN_PREFIX = "theory_cox"


def _ensure_experiment():
    if not _MLFLOW_AVAILABLE:
        return
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_survival_run(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    coefficients: Optional[Dict[str, float]] = None,
    theory_alignment_pct: Optional[float] = None,
    run_name: Optional[str] = None,
) -> Optional[str]:
    """Log a training/eval run; return run_id if MLflow available."""
    if not _MLFLOW_AVAILABLE:
        return None
    _ensure_experiment()
    with mlflow.start_run(run_name=run_name or f"{RUN_PREFIX}_{os.getpid()}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if coefficients:
            mlflow.log_dict(coefficients, "coefficients.json")
        if theory_alignment_pct is not None:
            mlflow.log_metric("theory_alignment_pct", theory_alignment_pct)
    return mlflow.active_run().info.run_id if mlflow.active_run() else None


def log_human_override(
    supplier_id: str,
    override_survival_prob: float,
    reason: str,
    officer_notes: str = "",
) -> Optional[str]:
    """Log human override for audit (DoD AI accountability). Returns run_id."""
    if not _MLFLOW_AVAILABLE:
        return None
    _ensure_experiment()
    with mlflow.start_run(run_name=f"override_{supplier_id}"):
        mlflow.log_params({
            "supplier_id": supplier_id,
            "override_survival_prob": override_survival_prob,
            "reason": reason,
            "officer_notes": officer_notes[:500],
        })
        mlflow.set_tag("type", "human_override")
    return mlflow.active_run().info.run_id if mlflow.active_run() else None


def is_mlflow_available() -> bool:
    return _MLFLOW_AVAILABLE
