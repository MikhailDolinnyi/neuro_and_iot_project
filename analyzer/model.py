from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, FEATURE_NAMES, extract_features
from .window import Reading

if TYPE_CHECKING:
    from .baseline import BaselineStats

_MODEL_PATH = Path(__file__).parent.parent / "models" / "stress_model.joblib"
MODEL_VERSION = "4"  # bump when feature vector or architecture changes

# Якоря Valence-Arousal по модели Рассела для каждого класса:
# calm — позитивная валентность, низкое возбуждение
# cognitive_load — нейтральная валентность, высокое возбуждение
# stressed — негативная валентность, очень высокое возбуждение
_VA_ANCHORS: dict[str, tuple[float, float]] = {
    "calm":           ( 0.70, 0.20),
    "cognitive_load": (-0.10, 0.65),
    "stressed":       (-0.70, 0.88),
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _compute_va(classes: list[str], proba: np.ndarray) -> tuple[float, float]:
    valence = sum(proba[i] * _VA_ANCHORS.get(c, (0.0, 0.5))[0] for i, c in enumerate(classes))
    arousal = sum(proba[i] * _VA_ANCHORS.get(c, (0.0, 0.5))[1] for i, c in enumerate(classes))
    return float(np.clip(valence, -1.0, 1.0)), float(np.clip(arousal, 0.0, 1.0))


def _readings_to_array(readings: list[Reading]) -> np.ndarray:
    """Преобразовать окно измерений в сырой массив (n_time, 3) для ROCKET."""
    bpm = np.array([r.bpm for r in readings], dtype=float)
    temp = np.array([r.temp_c for r in readings], dtype=float)
    fsr = np.array([r.fsr_raw for r in readings], dtype=float)
    return np.column_stack([bpm, temp, fsr])


class _RocketTransformer:
    """Упрощённый ROCKET: случайные дилатированные свёртки → PPV-признаки.

    Работает на многомерных временных рядах (n_time, n_channels).
    Не использует метки при fit() — поэтому при кросс-валидации нет утечки данных.
    """

    def __init__(self, n_kernels: int = 200, random_state: int = 42) -> None:
        self.n_kernels = n_kernels
        self.random_state = random_state
        self._kernels: list[tuple[np.ndarray, int]] = []

    def fit(self, X: np.ndarray) -> "_RocketTransformer":
        # X: (n_samples, n_time, n_channels)
        rng = np.random.default_rng(self.random_state)
        n_time = X.shape[1]
        self._kernels = []
        for _ in range(self.n_kernels):
            k_len = int(rng.choice([7, 9, 11]))
            w = rng.standard_normal((k_len, X.shape[2]))
            w -= w.mean(axis=0, keepdims=True)   # zero-mean per channel
            max_dil = max(1, (n_time - 1) // max(k_len - 1, 1))
            dilation = int(2 ** rng.uniform(0.0, np.log2(max_dil + 1)))
            self._kernels.append((w, dilation))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, n_time, n_channels) → (n_samples, n_kernels)
        out = np.empty((len(X), self.n_kernels), dtype=float)
        for k_idx, (w, dil) in enumerate(self._kernels):
            out[:, k_idx] = self._apply_batch(X, w, dil)
        return out

    @staticmethod
    def _apply_batch(X: np.ndarray, w: np.ndarray, dil: int) -> np.ndarray:
        """PPV (proportion of positive values) после дилатированной свёртки."""
        k_len = w.shape[0]
        effective = k_len + (k_len - 1) * (dil - 1)
        out_len = X.shape[1] - effective + 1
        if out_len <= 0:
            return np.zeros(len(X))
        idx = np.arange(0, effective, dil)   # индексы по ядру с дилатацией
        ppv = np.zeros(len(X))
        for i in range(out_len):
            sliced = X[:, i + idx, :]                        # (n, k_len, ch)
            conv = np.einsum("nkf,kf->n", sliced, w)         # (n,)
            ppv += conv > 0
        return ppv / out_len


class StressModel:
    def __init__(self) -> None:
        # Табличные модели (RF / GBM)
        self._clf = None
        self._scaler: Optional[StandardScaler] = None
        # ROCKET
        self._rocket: Optional[_RocketTransformer] = None
        self._rocket_pipeline = None          # Pipeline(StandardScaler, RidgeCV)

        self.cv_accuracy: Optional[float] = None
        self.active_model: Optional[str] = None
        self.all_scores: dict = {}
        self.feature_importances: list[float] = []

    @property
    def is_ready(self) -> bool:
        return self._clf is not None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_win: Optional[np.ndarray] = None,
    ) -> float:
        """Обучить RF, GBM и (если X_win передан) ROCKET; выбрать лучший."""
        # --- Табличные модели ---
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )

        rf_scores  = cross_val_score(rf,  Xs, y, cv=5, scoring="accuracy")
        gbm_scores = cross_val_score(gbm, Xs, y, cv=5, scoring="accuracy")

        # RF всегда обучаем для feature_importances (нужны в explain())
        rf.fit(Xs, y)
        self.feature_importances = rf.feature_importances_.tolist()

        self.all_scores = {
            "random_forest":     {"cv_mean": round(float(rf_scores.mean()),  4),
                                  "cv_std":  round(float(rf_scores.std()),   4)},
            "gradient_boosting": {"cv_mean": round(float(gbm_scores.mean()), 4),
                                  "cv_std":  round(float(gbm_scores.std()),  4)},
        }

        # Выбираем лучшую табличную модель
        if rf_scores.mean() >= gbm_scores.mean():
            self._clf = rf
            self.active_model = "random_forest"
            self.cv_accuracy = float(rf_scores.mean())
        else:
            gbm.fit(Xs, y)
            self._clf = gbm
            self.active_model = "gradient_boosting"
            self.cv_accuracy = float(gbm_scores.mean())

        # --- ROCKET (если переданы сырые окна) ---
        if X_win is not None and len(X_win) >= 10:
            self._rocket = _RocketTransformer(n_kernels=200, random_state=42)
            X_rocket = self._rocket.fit(X_win).transform(X_win)

            rocket_pipeline = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10),
                    class_weight="balanced",
                ),
            )
            rocket_scores = cross_val_score(
                rocket_pipeline, X_rocket, y, cv=5, scoring="accuracy"
            )
            rocket_pipeline.fit(X_rocket, y)
            self._rocket_pipeline = rocket_pipeline

            self.all_scores["rocket"] = {
                "cv_mean": round(float(rocket_scores.mean()), 4),
                "cv_std":  round(float(rocket_scores.std()),  4),
            }

            if rocket_scores.mean() > self.cv_accuracy:
                self.active_model = "rocket"
                self.cv_accuracy = float(rocket_scores.mean())

        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "clf":                self._clf,
            "scaler":             self._scaler,
            "rocket":             self._rocket,
            "rocket_pipeline":    self._rocket_pipeline,
            "cv_acc":             self.cv_accuracy,
            "active_model":       self.active_model,
            "all_scores":         self.all_scores,
            "feature_importances": self.feature_importances,
            "version":            MODEL_VERSION,
        }, _MODEL_PATH)

        return self.cv_accuracy

    def load(self) -> bool:
        if not _MODEL_PATH.exists():
            return False
        data = joblib.load(_MODEL_PATH)
        if data.get("version") != MODEL_VERSION:
            print(f"[model] Version mismatch ({data.get('version')} != {MODEL_VERSION}), retraining")
            return False
        self._clf             = data["clf"]
        self._scaler          = data["scaler"]
        self._rocket          = data.get("rocket")
        self._rocket_pipeline = data.get("rocket_pipeline")
        self.cv_accuracy      = data.get("cv_acc")
        self.active_model     = data.get("active_model")
        self.all_scores       = data.get("all_scores", {})
        self.feature_importances = data.get("feature_importances", [])
        return True

    def predict(
        self,
        readings: list[Reading],
        baseline: Optional[BaselineStats] = None,
    ) -> tuple[str, float, float, float, float]:
        """Вернуть (label, stress_score, confidence, valence, arousal)."""
        if self.active_model == "rocket" and self._rocket is not None:
            X_win = _readings_to_array(readings)[np.newaxis]   # (1, n_time, 3)
            X_rocket = self._rocket.transform(X_win)
            label: str = self._rocket_pipeline.predict(X_rocket)[0]
            decision = self._rocket_pipeline.decision_function(X_rocket)[0]
            proba = _softmax(np.atleast_1d(decision))
            classes: list[str] = list(self._rocket_pipeline.classes_)
        else:
            feats = extract_features(readings, baseline=baseline).reshape(1, -1)
            Xs = self._scaler.transform(feats)
            label = self._clf.predict(Xs)[0]
            proba = self._clf.predict_proba(Xs)[0]
            classes = list(self._clf.classes_)

        confidence = float(proba.max())
        stressed_idx = classes.index("stressed") if "stressed" in classes else 0
        stress_score = float(proba[stressed_idx])
        valence, arousal = _compute_va(classes, proba)
        return label, stress_score, confidence, valence, arousal

    def explain(
        self,
        readings: list[Reading],
        baseline: Optional[BaselineStats] = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Top-N признаков по важности с текущими значениями.

        Всегда использует важности RF, даже если активна модель ROCKET —
        это даёт физиологически интерпретируемое объяснение.
        """
        if not self.is_ready or not self.feature_importances:
            return []
        feats = extract_features(readings, baseline=baseline)
        items = [
            {
                "name":       name,
                "label":      FEATURE_LABELS.get(name, name),
                "value":      round(float(val), 3),
                "importance": round(imp, 4),
            }
            for name, val, imp in zip(FEATURE_NAMES, feats, self.feature_importances)
        ]
        return sorted(items, key=lambda x: x["importance"], reverse=True)[:top_n]
