from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, FEATURE_NAMES, extract_features
from .window import Reading

if TYPE_CHECKING:
    from .baseline import BaselineStats

_MODEL_PATH = Path(__file__).parent.parent / "models" / "stress_model.joblib"
MODEL_VERSION = "3"  # bump when feature vector or architecture changes


class StressModel:
    def __init__(self) -> None:
        self._clf = None
        self._scaler: Optional[StandardScaler] = None
        self.cv_accuracy: Optional[float] = None
        self.active_model: Optional[str] = None
        self.all_scores: dict = {}
        self.feature_importances: list[float] = []

    @property
    def is_ready(self) -> bool:
        return self._clf is not None

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42,
        )

        rf_scores = cross_val_score(rf, Xs, y, cv=5, scoring="accuracy")
        gbm_scores = cross_val_score(gbm, Xs, y, cv=5, scoring="accuracy")

        self.all_scores = {
            "random_forest":       {"cv_mean": round(float(rf_scores.mean()), 4),
                                    "cv_std":  round(float(rf_scores.std()), 4)},
            "gradient_boosting":   {"cv_mean": round(float(gbm_scores.mean()), 4),
                                    "cv_std":  round(float(gbm_scores.std()), 4)},
        }

        # Always fit RF to get stable feature importances
        rf.fit(Xs, y)
        self.feature_importances = rf.feature_importances_.tolist()

        # Use the winner for inference
        if rf_scores.mean() >= gbm_scores.mean():
            self._clf = rf
            self.active_model = "random_forest"
            self.cv_accuracy = float(rf_scores.mean())
        else:
            gbm.fit(Xs, y)
            self._clf = gbm
            self.active_model = "gradient_boosting"
            self.cv_accuracy = float(gbm_scores.mean())

        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "clf": self._clf,
            "scaler": self._scaler,
            "cv_acc": self.cv_accuracy,
            "active_model": self.active_model,
            "all_scores": self.all_scores,
            "feature_importances": self.feature_importances,
            "version": MODEL_VERSION,
        }, _MODEL_PATH)

        return self.cv_accuracy

    def load(self) -> bool:
        if not _MODEL_PATH.exists():
            return False
        data = joblib.load(_MODEL_PATH)
        if data.get("version") != MODEL_VERSION:
            print(f"[model] Version mismatch ({data.get('version')} != {MODEL_VERSION}), retraining")
            return False
        self._clf = data["clf"]
        self._scaler = data["scaler"]
        self.cv_accuracy = data.get("cv_acc")
        self.active_model = data.get("active_model")
        self.all_scores = data.get("all_scores", {})
        self.feature_importances = data.get("feature_importances", [])
        return True

    def predict(
        self,
        readings: list[Reading],
        baseline: Optional[BaselineStats] = None,
    ) -> tuple[str, float, float]:
        """Returns (label, stress_score 0–1, confidence 0–1)."""
        feats = extract_features(readings, baseline=baseline).reshape(1, -1)
        Xs = self._scaler.transform(feats)
        label: str = self._clf.predict(Xs)[0]
        proba: np.ndarray = self._clf.predict_proba(Xs)[0]
        confidence = float(proba.max())
        stressed_idx = list(self._clf.classes_).index("stressed")
        stress_score = float(proba[stressed_idx])
        return label, stress_score, confidence

    def explain(
        self,
        readings: list[Reading],
        baseline: Optional[BaselineStats] = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Top-N features by importance with their current values."""
        if not self.is_ready or not self.feature_importances:
            return []
        feats = extract_features(readings, baseline=baseline)
        items = [
            {
                "name": name,
                "label": FEATURE_LABELS.get(name, name),
                "value": round(float(val), 3),
                "importance": round(imp, 4),
            }
            for name, val, imp in zip(FEATURE_NAMES, feats, self.feature_importances)
        ]
        return sorted(items, key=lambda x: x["importance"], reverse=True)[:top_n]
