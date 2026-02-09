"""
Theory operationalization (ante hoc feature engineering).

Supply Chain Resilience (Christopher & Peck 2004): buffer_efficiency, network_redundancy, visibility_depth.
Bullwhip Effect (Forrester 1961; Lee et al. 1997): demand_amplification_ratio, order_batching_irregularity, forecast_gaming_index.
Resource Dependence (Pfeffer & Salancik 1978): asymmetric_dependence, resource_substitutability, contractual_safeguards.

Constraint: Coefficients must align with theory (buffer_efficiency → negative hazard; demand_amplification → positive).
"""
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple

# Theory-to-feature mapping and expected coefficient sign (for governance)
THEORY_FEATURE_CONSTRAINTS = {
    # Resilience: higher buffer/redundancy/visibility → lower hazard
    "buffer_efficiency": {"theory": "Resilience", "expected_sign": -1},
    "network_redundancy": {"theory": "Resilience", "expected_sign": -1},
    "visibility_depth": {"theory": "Resilience", "expected_sign": -1},
    # Bullwhip: higher amplification/batching/gaming → higher hazard
    "demand_amplification_ratio": {"theory": "Bullwhip", "expected_sign": 1},
    "order_batching_irregularity": {"theory": "Bullwhip", "expected_sign": 1},
    "forecast_gaming_index": {"theory": "Bullwhip", "expected_sign": 1},
    "bullwhip_amplitude": {"theory": "Bullwhip", "expected_sign": 1},
    # Resource Dependence: higher asymmetry → higher hazard; substitutability/safeguards → lower
    "asymmetric_dependence": {"theory": "ResourceDependence", "expected_sign": 1},
    "resource_substitutability": {"theory": "ResourceDependence", "expected_sign": -1},
    "contractual_safeguards": {"theory": "ResourceDependence", "expected_sign": -1},
    # Cascade risk (graph): higher → higher hazard
    "cascade_risk_score": {"theory": "ResourceDependence", "expected_sign": 1},
    # Temporal convergence (Forrester): higher sync → lower hazard
    "temporal_convergence_index": {"theory": "Bullwhip", "expected_sign": -1},
}

COX_FEATURE_COLUMNS = [
    "buffer_efficiency",
    "network_redundancy",
    "visibility_depth",
    "demand_amplification_ratio",
    "order_batching_irregularity",
    "forecast_gaming_index",
    "bullwhip_amplitude",
    "asymmetric_dependence",
    "resource_substitutability",
    "contractual_safeguards",
    "cascade_risk_score",
    "temporal_convergence_index",
    "lead_time_days",
    "lead_time_variance",
    "geographic_risk",
]


def add_cascade_risk_score(
    df: pd.DataFrame,
    tier_col: str = "tier",
    asymmetric_col: str = "asymmetric_dependence",
    supplier_id_col: str = "supplier_id",
) -> pd.DataFrame:
    """
    Graph-based cascade risk using PageRank on supplier dependency graph (Tier 1-3).
    Resource Dependence: weight asymmetric power nodes (high dependence on us = critical).
    """
    df = df.copy()
    n = len(df)
    G = nx.DiGraph()
    G.add_nodes_from(df[supplier_id_col].tolist())
    # Edges: Tier 2 → Tier 1, Tier 3 → Tier 2 (dependencies flow up)
    ids = df[supplier_id_col].values
    tiers = df[tier_col].values
    asym = df[asymmetric_col].values
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if tiers[j] == tiers[i] - 1:  # j is one tier below i (supplies i)
                # Weight by asymmetric dependence: critical suppliers have high our_concentration
                w = asym[j] + 0.1
                G.add_edge(ids[j], ids[i], weight=w)
    if G.number_of_edges() == 0:
        # Fallback: use degree-like centrality from asymmetric_dependence
        df["cascade_risk_score"] = (df[asymmetric_col] / (df[asymmetric_col].max() + 1e-9)).values
        return df
    pr = nx.pagerank(G, alpha=0.85)
    df["cascade_risk_score"] = df[supplier_id_col].map(pr).fillna(0).values
    # Normalize 0-1
    mx = df["cascade_risk_score"].max()
    if mx > 0:
        df["cascade_risk_score"] = df["cascade_risk_score"] / mx
    return df


def add_temporal_convergence_index(
    df: pd.DataFrame,
    lead_time_col: str = "lead_time_days",
    order_col: str = "order_quantity_mean",
) -> pd.DataFrame:
    """
    Synchronization between supplier production cycles and demand seasonality (Forrester system dynamics).
    Higher = more aligned → lower hazard. Derived from lead time and order pattern consistency.
    """
    df = df.copy()
    # Placeholder: inverse of lead_time variance effect (longer, more variable = less convergence)
    lt = df[lead_time_col]
    ord_q = df[order_col]
    # Simple proxy: lower lead time variance and moderate order size = better convergence
    df["temporal_convergence_index"] = 1.0 / (1.0 + np.log1p(lt) * 0.1 + ord_q / (ord_q.max() + 1) * 0.2)
    df["temporal_convergence_index"] = (
        df["temporal_convergence_index"] / (df["temporal_convergence_index"].max() + 1e-9)
    ).clip(0, 1)
    return df


def build_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cascade_risk_score and temporal_convergence_index; ensure all COX_FEATURE_COLUMNS exist."""
    df = add_cascade_risk_score(df)
    df = add_temporal_convergence_index(df)
    for c in COX_FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def get_theory_constraint_signs() -> List[Tuple[str, int]]:
    """Return (feature_name, expected_sign) for Cox coefficient validation."""
    return [
        (k, v["expected_sign"])
        for k, v in THEORY_FEATURE_CONSTRAINTS.items()
        if k in COX_FEATURE_COLUMNS
    ]
