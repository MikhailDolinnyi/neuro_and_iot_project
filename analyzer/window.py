from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

WINDOW_MAXLEN = 25   # ~75 с при интервале 3 с
MIN_WINDOW_SIZE = 8  # минимум для первого предсказания


@dataclass
class Reading:
    bpm: float
    temp_c: float
    fsr_raw: int
    bpm_valid: bool
    temp_valid: bool
    rr_intervals: list[float] = field(default_factory=list)  # RR-интервалы в мс между ударами


class SlidingWindow:
    def __init__(self, maxlen: int = WINDOW_MAXLEN) -> None:
        self._data: deque[Reading] = deque(maxlen=maxlen)

    def push(self, reading: Reading) -> None:
        self._data.append(reading)

    def is_ready(self) -> bool:
        return len(self._data) >= MIN_WINDOW_SIZE

    def as_list(self) -> list[Reading]:
        return list(self._data)

    def size(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
