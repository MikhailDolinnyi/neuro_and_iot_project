from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .window import Reading

if TYPE_CHECKING:
    from .baseline import BaselineStats

# 9 абсолютных + 3 относительных (к базлайну) = 12 признаков
FEATURE_NAMES = [
    "bpm_mean", "bpm_std", "bpm_range", "bpm_trend",
    "temp_mean", "temp_std", "temp_trend",
    "fsr_mean", "fsr_std",
    "bpm_z", "temp_delta", "fsr_delta",
]

FEATURE_LABELS = {
    "bpm_mean":   "Средний пульс",
    "bpm_std":    "Вариация пульса",
    "bpm_range":  "Размах пульса",
    "bpm_trend":  "Тренд пульса",
    "temp_mean":  "Средняя температура",
    "temp_std":   "Вариация температуры",
    "temp_trend": "Тренд температуры",
    "fsr_mean":   "Среднее давление",
    "fsr_std":    "Вариация давления",
    "bpm_z":      "Пульс (откл. базлайн)",
    "temp_delta": "Температура (откл. базлайн)",
    "fsr_delta":  "Давление (откл. базлайн)",
}


def extract_features(
    readings: list[Reading],
    baseline: Optional[BaselineStats] = None,
) -> np.ndarray:
    bpm = np.array([r.bpm for r in readings if r.bpm_valid and r.bpm > 0], dtype=float)
    temp = np.array([r.temp_c for r in readings if r.temp_valid and r.temp_c > 0], dtype=float)
    fsr = np.array([r.fsr_raw for r in readings], dtype=float)

    def _trend(arr: np.ndarray) -> float:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) >= 2 else 0.0

    bpm_feats = (
        [bpm.mean(), bpm.std(), float(bpm.max() - bpm.min()), _trend(bpm)]
        if len(bpm) >= 2 else [0.0, 0.0, 0.0, 0.0]
    )
    temp_feats = (
        [temp.mean(), temp.std(), _trend(temp)]
        if len(temp) >= 2 else [0.0, 0.0, 0.0]
    )
    fsr_feats = [float(fsr.mean()), float(fsr.std()) if len(fsr) >= 2 else 0.0]

    if baseline is not None and len(bpm) >= 2:
        bpm_z = (float(bpm.mean()) - baseline.bpm_mean) / baseline.bpm_std
        temp_delta = (float(temp.mean()) - baseline.temp_mean) if len(temp) else 0.0
        fsr_delta = float(fsr.mean()) - baseline.fsr_mean
    else:
        bpm_z = temp_delta = fsr_delta = 0.0

    return np.array(bpm_feats + temp_feats + fsr_feats + [bpm_z, temp_delta, fsr_delta], dtype=float)
