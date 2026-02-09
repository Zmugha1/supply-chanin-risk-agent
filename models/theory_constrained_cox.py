"""
Theory-Constrained Cox Proportional Hazards Model.

Rationale (documented in code):
1. Small data (n=600, p=45): Grinsztajn et al. (2022) — complex models overfit; Cox + L2 generalizes.
2. Interpretability ante hoc (Rudin 2019): Coefficients = hazard ratios for procurement officers.
3. Calibration: Cox aligns with Kaplan-Meier; black-box survival often poor (Guo et al. 2017).
4. Right-censoring: 85% censored; Cox partial likelihood handles this natively.

Uses lifelines.CoxPHFitter with L2 penalty. Theoretical coefficient signs validated post-fit in governance.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Default feature set; must match features/theory_feature_engineering.COX_FEATURE_COLUMNS
DURATION_COL = "duration_days"
EVENT_COL = "event"


class TheoryConstrainedCox:
    """
    Wrapper around lifelines.CoxPHFitter with L2 regularization.
    Theory sign constraints are enforced via governance checks (theoretical_fidelity), not hard constraints in lifelines.
    """

    def __init__(
        self,
        penalty: float = 0.5,
        l1_ratio: float = 0.0,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        penalty: L2 strength (higher = more regularization for small n).
        l1_ratio: 0 = pure L2 (default for stability on small data).
        """
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.feature_columns = feature_columns or []
        self._model: Optional[CoxPHFitter] = None
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = DURATION_COL,
        event_col: str = EVENT_COL,
        feature_columns: Optional[List[str]] = None,
    ) -> "TheoryConstrainedCox":
        cols = feature_columns or self.feature_columns
        if not cols:
            raise ValueError("feature_columns required for fit")
        use = [duration_col, event_col] + cols
        missing = [c for c in use if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        data = df[use].copy().dropna()
        self.feature_columns = cols
        self._model = CoxPHFitter(penalizer=self.penalty, l1_ratio=self.l1_ratio)
        self._model.fit(data, duration_col=duration_col, event_col=event_col)
        self._fitted = True
        return self

    @property
    def model(self) -> CoxPHFitter:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model

    def predict_survival_function(self, df: pd.DataFrame, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Survival probability at given times (default: 30, 60, 90 days)."""
        if times is None:
            times = np.array([30.0, 60.0, 90.0])
        return self.model.predict_survival_function(df[self.feature_columns], times=times)

    def predict_partial_hazard(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_partial_hazard(df[self.feature_columns])

    def concordance_index_(self, df: pd.DataFrame, duration_col: str = DURATION_COL, event_col: str = EVENT_COL) -> float:
        """C-index on provided dataframe."""
        pred = -self.predict_partial_hazard(df)  # higher hazard = shorter survival
        return concordance_index(df[duration_col], pred, df[event_col])

    def summary(self) -> pd.DataFrame:
        """Coefficients and hazard ratios."""
        return self.model.summary

    def get_coefficients(self) -> Dict[str, float]:
        """Feature name -> coefficient (log hazard ratio)."""
        s = self.model.summary
        return s["coef"].to_dict() if "coef" in s.columns else {}
