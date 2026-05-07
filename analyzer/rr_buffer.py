from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

# Стандарт ESC/AHA для краткосрочного HRV — 5 минут.
# 300 RR при пульсе ~60 покрывают именно этот интервал.
RR_BUFFER_MAXLEN = 300
RR_BUFFER_MAX_AGE_SEC = 300.0
RR_VALID_MIN_MS = 300.0
RR_VALID_MAX_MS = 2000.0


@dataclass
class _Item:
    rr_ms: float
    ts: float


class RRBuffer:
    """Rolling-буфер RR-интервалов для индекса Баевского.

    Накапливает уникальные RR с привязкой ко времени. Старые значения
    автоматически вытесняются — либо по объёму (maxlen), либо по возрасту
    (старше max_age_sec). Это позволяет считать SI на 5-минутном окне
    независимо от частоты прихода MQTT-сообщений.
    """

    def __init__(
        self,
        maxlen: int = RR_BUFFER_MAXLEN,
        max_age_sec: float = RR_BUFFER_MAX_AGE_SEC,
    ) -> None:
        self._items: deque[_Item] = deque(maxlen=maxlen)
        self._max_age_sec = max_age_sec

    def extend(self, rr_intervals: list[float] | list[int]) -> None:
        now = time.monotonic()
        for v in rr_intervals:
            rr = float(v)
            if RR_VALID_MIN_MS <= rr <= RR_VALID_MAX_MS:
                self._items.append(_Item(rr_ms=rr, ts=now))
        self._evict_old(now)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._max_age_sec
        while self._items and self._items[0].ts < cutoff:
            self._items.popleft()

    def values(self) -> list[float]:
        self._evict_old(time.monotonic())
        return [it.rr_ms for it in self._items]

    @property
    def size(self) -> int:
        self._evict_old(time.monotonic())
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
