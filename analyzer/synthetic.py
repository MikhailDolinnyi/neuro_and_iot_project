from __future__ import annotations

import numpy as np

from .baseline import BaselineStats
from .features import extract_features
from .window import Reading

_WINDOW_SIZE = 20
_N_PER_CLASS = 600

# Спокойный базлайн — точка отсчёта для всех относительных признаков
_CALM_BASELINE = BaselineStats(
    bpm_mean=68.0, bpm_std=4.0, temp_mean=36.75, fsr_mean=85.0, n_readings=20
)

_CLASS_PARAMS: dict[str, dict] = {
    "calm": {
        "bpm_mu": 68.0,  "bpm_sigma": 4.0,  "bpm_noise": 1.5,
        "temp_mu": 36.75, "temp_sigma": 0.12, "temp_noise": 0.015,
        "fsr_mu": 85.0,   "fsr_sigma": 35.0,
        # HRV: высокая вариабельность = парасимпатика доминирует
        "rmssd": 42.0,
    },
    "cognitive_load": {
        "bpm_mu": 83.0,  "bpm_sigma": 5.5,  "bpm_noise": 2.5,
        "temp_mu": 36.45, "temp_sigma": 0.16, "temp_noise": 0.025,
        "fsr_mu": 270.0,  "fsr_sigma": 85.0,
        # HRV: умеренное снижение
        "rmssd": 26.0,
    },
    "stressed": {
        "bpm_mu": 98.0,  "bpm_sigma": 8.0,  "bpm_noise": 4.0,
        "temp_mu": 36.10, "temp_sigma": 0.20, "temp_noise": 0.035,
        "fsr_mu": 560.0,  "fsr_sigma": 140.0,
        # HRV: низкая — симпатика подавляет вариабельность
        "rmssd": 12.0,
    },
}

# Каналы для ROCKET: BPM=0, temp=1, FSR=2
_N_CHANNELS = 3


def _make_rr_intervals(bpm: float, rmssd: float, n: int, rng: np.random.Generator) -> list[float]:
    """RR-интервалы (мс) с заданным RMSSD для одного временного шага."""
    mean_rr = 60000.0 / max(bpm, 30.0)
    # Для независимых N(mu, sigma): RMSSD ≈ sqrt(2)*sigma → sigma = rmssd/sqrt(2)
    sigma = rmssd / np.sqrt(2.0)
    rr = rng.normal(mean_rr, sigma, n)
    return np.clip(rr, 300.0, 2000.0).tolist()


def _make_window(p: dict, rng: np.random.Generator) -> list[Reading]:
    bpm_s = np.clip(
        rng.normal(p["bpm_mu"], p["bpm_sigma"])
        + np.cumsum(rng.normal(0, p["bpm_noise"], _WINDOW_SIZE)),
        40, 180,
    )
    temp_s = np.clip(
        rng.normal(p["temp_mu"], p["temp_sigma"])
        + np.cumsum(rng.normal(0, p["temp_noise"], _WINDOW_SIZE)) * 0.1,
        34.0, 40.0,
    )
    fsr_s = np.clip(
        rng.normal(p["fsr_mu"], p["fsr_sigma"], _WINDOW_SIZE), 0, 1023
    ).astype(int)

    return [
        Reading(
            bpm=float(bpm_s[i]),
            temp_c=float(temp_s[i]),
            fsr_raw=int(fsr_s[i]),
            bpm_valid=True,
            temp_valid=True,
            rr_intervals=_make_rr_intervals(float(bpm_s[i]), p["rmssd"], 4, rng),
        )
        for i in range(_WINDOW_SIZE)
    ]


def _window_to_array(readings: list[Reading]) -> np.ndarray:
    """Преобразовать окно в сырой массив (n_time, 3) для ROCKET."""
    bpm = np.array([r.bpm for r in readings], dtype=float)
    temp = np.array([r.temp_c for r in readings], dtype=float)
    fsr = np.array([r.fsr_raw for r in readings], dtype=float)
    return np.column_stack([bpm, temp, fsr])


def generate_dataset(
    n_per_class: int = _N_PER_CLASS, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вернуть (X_tab, X_win, y).

    X_tab — табличные признаки (n_samples, 16) для RF/GBM.
    X_win — сырые временные ряды (n_samples, window_size, 3) для ROCKET.
    y — метки классов.
    """
    rng = np.random.default_rng(seed)
    X_tab, X_win, y = [], [], []
    for label, params in _CLASS_PARAMS.items():
        for _ in range(n_per_class):
            window = _make_window(params, rng)
            X_tab.append(extract_features(window, baseline=_CALM_BASELINE))
            X_win.append(_window_to_array(window))
            y.append(label)
    return np.array(X_tab, dtype=float), np.array(X_win, dtype=float), np.array(y)
