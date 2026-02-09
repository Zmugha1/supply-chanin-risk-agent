"""
BullwhipAnalyst: Detect demand signal distortion violating Forrester/Lee theory.

Only reports anomalies consistent with Bullwhip causes (forecast updating, order batching,
price fluctuations, shortage gaming). Flags orders with demand_amplification_ratio > 2.0.
"""

import pandas as pd
from typing import Any, Dict, List

BULLWHIP_ALERT_THRESHOLD = 2.0  # Lee et al.: amplification > 2x is material


def calculate_demand_amplification_ratio(df: pd.DataFrame) -> pd.Series:
    """Variance(orders) / Variance(demand) per supplier (already in df as demand_amplification_ratio)."""
    if "demand_amplification_ratio" in df.columns:
        return df["demand_amplification_ratio"]
    return pd.Series(dtype=float)


def flag_bullwhip_anomalies(
    df: pd.DataFrame,
    threshold: float = BULLWHIP_ALERT_THRESHOLD,
    supplier_id_col: str = "supplier_id",
) -> List[Dict[str, Any]]:
    """
    Return list of suppliers with demand_amplification > threshold.
    Constraint: Only Bullwhip-consistent anomalies (amplification, batching, gaming).
    """
    if "demand_amplification_ratio" not in df.columns:
        return []
    flagged = df[df["demand_amplification_ratio"] > threshold]
    return [
        {
            "supplier_id": row[supplier_id_col],
            "demand_amplification_ratio": float(row["demand_amplification_ratio"]),
            "message": f"Demand amplification {row['demand_amplification_ratio']:.2f}x exceeds threshold {threshold} (Lee et al. Bullwhip).",
        }
        for _, row in flagged.iterrows()
    ]


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent entry: context must have 'df' (DataFrame with theory features).
    Returns alerts and summary for HumanCoordinationAgent / UI.
    """
    df = context.get("df")
    if df is None or not isinstance(df, pd.DataFrame):
        return {"alerts": [], "summary": "No data provided.", "count": 0}
    alerts = flag_bullwhip_anomalies(df)
    return {
        "alerts": alerts,
        "summary": f"BullwhipAnalyst: {len(alerts)} supplier(s) with demand amplification > {BULLWHIP_ALERT_THRESHOLD}x.",
        "count": len(alerts),
    }
