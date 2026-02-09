"""
Calibration governance: Brier score, reliability diagrams, Integrated Brier Score.

Recalibration trigger: If Integrated Brier Score > 0.15, retrain with updated constraints.
Proper scoring rule for survival: Brier score on survival probabilities at 30/60/90 days.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.calibration import calibration_curve

try:
    from sksurv.metrics import cumulative_dynamic_auc, brier_score
except Exception:
    brier_score = None
    cumulative_dynamic_auc = None


def brier_score_survival(
    y_true_event: np.ndarray,
    y_true_time: np.ndarray,
    y_pred_survival: np.ndarray,
    time_point: float,
) -> float:
    """
    Brier score at a single time point for survival predictions.
    y_pred_survival: predicted P(T > time_point) for each sample.
    """
    # Binary at time_point: did they survive past time_point?
    binary = (y_true_event == 0) | (y_true_time > time_point)
    binary = binary.astype(float)
    pred = np.clip(y_pred_survival, 1e-6, 1 - 1e-6)
    return np.mean((binary - pred) ** 2)


def integrated_brier_score(
    y_true_event: np.ndarray,
    y_true_time: np.ndarray,
    y_pred_survival_at_times: np.ndarray,
    times: np.ndarray,
) -> float:
    """Integrated Brier Score over time points (mean of Brier at each time)."""
    scores = []
    for j, t in enumerate(times):
        if y_pred_survival_at_times.shape[1] > j:
            scores.append(
                brier_score_survival(
                    y_true_event, y_true_time, y_pred_survival_at_times[:, j], t
                )
            )
    return float(np.mean(scores)) if scores else 0.0


def reliability_diagram(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prob_true, prob_pred, bin edges for calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=n_bins)
    return prob_true, prob_pred, np.linspace(0, 1, n_bins + 1)[: len(prob_true)]


def expected_calibration_error(
    y_true_binary: np.ndarray,
    y_pred_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted mean of |prob_true - prob_pred| per bin."""
    prob_true, prob_pred = calibration_curve(y_true_binary, y_pred_prob, n_bins=n_bins)
    return float(np.abs(prob_true - prob_pred).mean())
