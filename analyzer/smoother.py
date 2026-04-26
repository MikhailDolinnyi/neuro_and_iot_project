from __future__ import annotations

from collections import Counter, deque

import numpy as np

SMOOTH_SIZE = 5


class PredictionSmoother:
    """Majority-vote smoother over the last N predictions."""

    def __init__(self, size: int = SMOOTH_SIZE) -> None:
        self._labels:   deque[str]   = deque(maxlen=size)
        self._scores:   deque[float] = deque(maxlen=size)
        self._confs:    deque[float] = deque(maxlen=size)
        self._valences: deque[float] = deque(maxlen=size)
        self._arousals: deque[float] = deque(maxlen=size)

    def update(
        self,
        label: str,
        score: float,
        conf: float,
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> tuple[str, float, float, float, float]:
        self._labels.append(label)
        self._scores.append(score)
        self._confs.append(conf)
        self._valences.append(valence)
        self._arousals.append(arousal)

        smoothed = Counter(self._labels).most_common(1)[0][0]
        return (
            smoothed,
            round(float(np.mean(self._scores)),   3),
            round(float(np.mean(self._confs)),     3),
            round(float(np.mean(self._valences)),  3),
            round(float(np.mean(self._arousals)),  3),
        )

    def reset(self) -> None:
        self._labels.clear()
        self._scores.clear()
        self._confs.clear()
        self._valences.clear()
        self._arousals.clear()
