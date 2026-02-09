"""
Theoretical Fidelity Monitoring: Cox coefficients must align with theory signs.

Alert if model contradicts theory (e.g. buffer_efficiency showing positive hazard).
Theory Alignment Score = % of coefficients matching theoretical predictions.
"""

import pandas as pd
from typing import Dict, List, Tuple
from features.theory_feature_engineering import THEORY_FEATURE_CONSTRAINTS, COX_FEATURE_COLUMNS


def check_coefficient_signs(
    coefficients: Dict[str, float],
    constraints: Dict[str, dict] = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Returns: (alignment_pct, list_aligned, list_violations).
    alignment_pct in [0, 100]. Violations are features where sign contradicts theory.
    """
    constraints = constraints or THEORY_FEATURE_CONSTRAINTS
    aligned = []
    violations = []
    checked = 0
    for feat, coef in coefficients.items():
        if feat not in constraints or feat not in COX_FEATURE_COLUMNS:
            continue
        expected = constraints[feat]["expected_sign"]
        actual = 1 if coef > 0 else (-1 if coef < 0 else 0)
        if actual == 0:
            continue
        checked += 1
        if actual == expected:
            aligned.append(feat)
        else:
            violations.append(
                f"{feat}: expected sign {expected}, got {actual} (theory: {constraints[feat]['theory']})"
            )
    pct = (100.0 * len(aligned) / checked) if checked else 100.0
    return pct, aligned, violations


def get_theory_alignment_message(violations: List[str]) -> str:
    """Human-readable alert for UI."""
    if not violations:
        return "All coefficients align with theoretical predictions."
    return "WARNING: " + "; ".join(violations[:5]) + (" ..." if len(violations) > 5 else "")
