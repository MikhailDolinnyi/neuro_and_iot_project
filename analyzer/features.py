from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .window import Reading

if TYPE_CHECKING:
    from .baseline import BaselineStats

# 9 абсолютных + 3 относительных (к базлайну) + 4 HRV = 16 признаков
FEATURE_NAMES = [
    "bpm_mean", "bpm_std", "bpm_range", "bpm_trend",
    "temp_mean", "temp_std", "temp_trend",
    "fsr_mean", "fsr_std",
    "bpm_z", "temp_delta", "fsr_delta",
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_mean_rr",
]

FEATURE_LABELS = {
    "bpm_mean":    "Средний пульс",
    "bpm_std":     "Вариация пульса",
    "bpm_range":   "Размах пульса",
    "bpm_trend":   "Тренд пульса",
    "temp_mean":   "Средняя температура",
    "temp_std":    "Вариация температуры",
    "temp_trend":  "Тренд температуры",
    "fsr_mean":    "Среднее давление",
    "fsr_std":     "Вариация давления",
    "bpm_z":       "Пульс (откл. базлайн)",
    "temp_delta":  "Температура (откл. базлайн)",
    "fsr_delta":   "Давление (откл. базлайн)",
    "hrv_rmssd":   "HRV RMSSD (мс)",
    "hrv_sdnn":    "HRV SDNN (мс)",
    "hrv_pnn50":   "HRV pNN50 (%)",
    "hrv_mean_rr": "Средний RR-интервал (мс)",
}


def _hrv_features(readings: list[Reading]) -> tuple[float, float, float, float]:
    """RMSSD, SDNN, pNN50, mean_rr — из RR-интервалов всего окна.

    RMSSD ↓ при стрессе (симпатика подавляет вариабельность).
    SDNN — общая мера вариабельности.
    pNN50 — доля пар, разница которых > 50 мс.
    mean_rr — обратно пропорционален пульсу.
    """
    per_reading = [np.array(r.rr_intervals) for r in readings if r.rr_intervals]
    if not per_reading:
        return 0.0, 0.0, 0.0, 0.0

    # 300–1500ms = 40–200 BPM; выбрасываем артефакты зажима при BPM<40
    per_reading = [rr[(rr >= 300) & (rr <= 1500)] for rr in per_reading]
    per_reading = [rr for rr in per_reading if len(rr) >= 2]
    if not per_reading:
        return 0.0, 0.0, 0.0, 0.0

    # RMSSD считаем внутри каждого измерения, потом усредняем —
    # иначе переходы между измерениями при дрейфе BPM искажают результат
    rmssd_vals = []
    for rr in per_reading:
        d = np.abs(np.diff(rr))
        rmssd_vals.append(float(np.sqrt(np.mean(d ** 2))))
    rmssd = float(np.mean(rmssd_vals))

    rr = np.concatenate(per_reading)
    mean_rr = float(rr.mean())
    sdnn = float(rr.std())
    diffs = np.abs(np.diff(rr))
    pnn50 = float(np.mean(diffs > 50.0)) if len(diffs) else 0.0
    return rmssd, sdnn, pnn50, mean_rr


def extract_features(
    readings: list[Reading],
    baseline: Optional[BaselineStats] = None,
) -> np.ndarray:
    bpm = np.array([r.bpm for r in readings if r.bpm_valid and r.bpm > 0], dtype=float)
    temp = np.array([r.temp_c for r in readings if r.temp_valid and r.temp_c > 0], dtype=float)
    fsr = np.log1p(np.minimum(np.array([r.fsr_raw for r in readings], dtype=float), 200.0))

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
        fsr_delta = float(fsr.mean()) - float(np.log1p(baseline.fsr_mean))
    else:
        bpm_z = temp_delta = fsr_delta = 0.0

    rmssd, sdnn, pnn50, mean_rr = _hrv_features(readings)

    return np.array(
        bpm_feats + temp_feats + fsr_feats + [bpm_z, temp_delta, fsr_delta]
        + [rmssd, sdnn, pnn50, mean_rr],
        dtype=float,
    )
