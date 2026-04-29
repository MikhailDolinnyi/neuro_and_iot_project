from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, FEATURE_NAMES, extract_features
from .window import Reading

if TYPE_CHECKING:
    from .baseline import BaselineStats

_MODEL_PATH = Path(__file__).parent.parent / "models" / "stress_model.joblib"

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
    fsr = np.log1p(np.minimum(np.array([r.fsr_raw for r in readings], dtype=float), 200.0))
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
        self._ch_mean: Optional[np.ndarray] = None
        self._ch_std:  Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "_RocketTransformer":
        # X: (n_samples, n_time, n_channels)
        # Per-channel z-score so BPM (40-180) doesn't dominate FSR log1p (0-7)
        self._ch_mean = x.mean(axis=(0, 1))
        self._ch_std  = np.where(x.std(axis=(0, 1)) < 1e-8, 1.0, x.std(axis=(0, 1)))

        rng = np.random.default_rng(self.random_state)
        n_time = x.shape[1]
        self._kernels = []
        k_pool = [k for k in (3, 5, 7, 9, 11) if k < n_time]
        if not k_pool:
            k_pool = [3]
        for _ in range(self.n_kernels):
            k_len = int(rng.choice(k_pool))
            w = rng.standard_normal((k_len, x.shape[2]))
            w -= w.mean(axis=0, keepdims=True)
            max_dil = max(1, (n_time - 1) // max(k_len - 1, 1))
            dilation = int(2 ** rng.uniform(0.0, np.log2(max_dil + 1)))
            self._kernels.append((w, dilation))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        # X: (n_samples, n_time, n_channels) → (n_samples, n_kernels)
        if self._ch_mean is not None:
            x = (x - self._ch_mean[np.newaxis, np.newaxis, :]) / self._ch_std[np.newaxis, np.newaxis, :]
        out = np.empty((len(x), self.n_kernels), dtype=float)
        for k_idx, (w, dil) in enumerate(self._kernels):
            out[:, k_idx] = self._apply_batch(x, w, dil)
        return out

    @staticmethod
    def _apply_batch(x: np.ndarray, w: np.ndarray, dil: int) -> np.ndarray:
        """PPV (proportion of positive values) после дилатированной свёртки."""
        k_len = w.shape[0]
        effective = k_len + (k_len - 1) * (dil - 1)
        out_len = x.shape[1] - effective + 1
        if out_len <= 0:
            return np.zeros(len(x))
        idx = np.arange(0, effective, dil)   # индексы по ядру с дилатацией
        ppv = np.zeros(len(x))
        for i in range(out_len):
            sliced = x[:, i + idx, :]                        # (n, k_len, ch)
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
        x: np.ndarray,
        y: np.ndarray,
        x_win: Optional[np.ndarray] = None,
    ) -> float:
        """Обучить RF, GBM и (если X_win передан) ROCKET; выбрать лучший."""
        # --- Табличные модели ---
        self._scaler = StandardScaler()
        xs = self._scaler.fit_transform(x)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        gbm = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_scores  = cross_val_score(rf,  xs, y, cv=cv, scoring="accuracy")
        gbm_scores = cross_val_score(gbm, xs, y, cv=cv, scoring="accuracy")

        # RF всегда обучаем для feature_importances (нужны в explain())
        rf.fit(xs, y)
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
            gbm.fit(xs, y)
            self._clf = gbm
            self.active_model = "gradient_boosting"
            self.cv_accuracy = float(gbm_scores.mean())

        # --- ROCKET (если переданы сырые окна) ---
        if x_win is not None and len(x_win) >= 10:
            self._rocket = _RocketTransformer(n_kernels=200, random_state=42)
            x_rocket = self._rocket.fit(x_win).transform(x_win)

            rocket_pipeline = make_pipeline(
                StandardScaler(),
                RidgeClassifierCV(
                    alphas=np.logspace(-3, 3, 10),
                    class_weight="balanced",
                ),
            )
            rocket_scores = cross_val_score(
                rocket_pipeline, x_rocket, y, cv=cv, scoring="accuracy"
            )
            rocket_pipeline.fit(x_rocket, y)
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
        }, _MODEL_PATH)

        return self.cv_accuracy

    def load(self) -> bool:
        if not _MODEL_PATH.exists():
            return False
        data = joblib.load(_MODEL_PATH)
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
            x_win = _readings_to_array(readings)[np.newaxis]   # (1, n_time, 3)
            x_rocket = self._rocket.transform(x_win)
            label: str = self._rocket_pipeline.predict(x_rocket)[0]
            decision = self._rocket_pipeline.decision_function(x_rocket)[0]
            proba = _softmax(np.atleast_1d(decision))
            classes: list[str] = list(self._rocket_pipeline.classes_)
        else:
            feats = extract_features(readings, baseline=baseline).reshape(1, -1)
            xs = self._scaler.transform(feats)
            label = self._clf.predict(xs)[0]
            proba = self._clf.predict_proba(xs)[0]
            classes = list(self._clf.classes_)

        confidence = float(proba.max())
        _STRESS_WEIGHTS = {"calm": 0.0, "cognitive_load": 0.5, "stressed": 1.0}
        stress_score = float(sum(
            proba[i] * _STRESS_WEIGHTS.get(c, 0.5) for i, c in enumerate(classes)
        ))
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

    @property
    def rocket(self):
        return self._rocket
