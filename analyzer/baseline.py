from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .window import Reading

BASELINE_MIN_READINGS = 20  # ~60 с при интервале 3 с


@dataclass
class BaselineStats:
    bpm_mean: float
    bpm_std: float
    temp_mean: float
    fsr_mean: float
    n_readings: int


class BaselineCalibrator:
    def __init__(self) -> None:
        self._buffer: list[Reading] = []
        self._recording: bool = False
        self.stats: Optional[BaselineStats] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_calibrated(self) -> bool:
        return self.stats is not None

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def start(self) -> None:
        self._buffer = []
        self._recording = True

    def add(self, reading: Reading) -> None:
        if self._recording and reading.bpm_valid:
            self._buffer.append(reading)

    def stop(self) -> bool:
        self._recording = False
        if len(self._buffer) < BASELINE_MIN_READINGS:
            return False

        bpms = np.array([r.bpm for r in self._buffer])
        temps = np.array([r.temp_c for r in self._buffer if r.temp_valid])
        fsrs = np.array([r.fsr_raw for r in self._buffer])

        self.stats = BaselineStats(
            bpm_mean=float(bpms.mean()),
            bpm_std=float(max(bpms.std(), 1.0)),
            temp_mean=float(temps.mean()) if len(temps) else 36.6,
            fsr_mean=float(fsrs.mean()),
            n_readings=len(self._buffer),
        )
        return True
