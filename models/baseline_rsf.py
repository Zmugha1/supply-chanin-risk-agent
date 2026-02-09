"""
Baseline: Random Survival Forest (scikit-survival).

Purpose: Demonstrate that even regularized ensembles underperform theory-constrained Cox on small supply chain data.
Parameters: n_estimators=100, min_samples_split=20 (aggressive regularization to combat overfitting on n=600).
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

DURATION_COL = "duration_days"
EVENT_COL = "event"


class BaselineRSF:
    """Random Survival Forest baseline for comparison."""

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 20,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._model: Optional[RandomSurvivalForest] = None
        self.feature_columns: List[str] = []
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = DURATION_COL,
        event_col: str = EVENT_COL,
        feature_columns: Optional[List[str]] = None,
    ) -> "BaselineRSF":
        if feature_columns is None:
            raise ValueError("feature_columns required")
        self.feature_columns = feature_columns
        X = df[self.feature_columns].fillna(0).values
        y = Surv.from_arrays(df[event_col].astype(bool), df[duration_col].values)
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    @property
    def model(self) -> RandomSurvivalForest:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model

    def predict_survival_function(self, df: pd.DataFrame, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Survival probability at times (default 30, 60, 90 days)."""
        if times is None:
            times = np.array([30.0, 60.0, 90.0])
        X = df[self.feature_columns].fillna(0).values
        surv_fn = self.model.predict_survival_function(X)
        out = np.zeros((len(df), len(times)))
        for i, sf in enumerate(surv_fn):
            for j, t in enumerate(times):
                out[i, j] = np.clip(sf(t), 0, 1)
        return out

    def concordance_index_(self, df: pd.DataFrame, duration_col: str = DURATION_COL, event_col: str = EVENT_COL) -> float:
        return self.model.score(df[self.feature_columns].fillna(0), Surv.from_arrays(df[event_col].astype(bool), df[duration_col].values))
