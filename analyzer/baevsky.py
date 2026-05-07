from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Минимум RR для устойчивой оценки моды/AMo. По методике Баевского — 100+,
# у нас >=50 даёт ориентировочную оценку, ниже считаем индекс ненадёжным.
MIN_RR_FOR_SI = 50

# Ширина бина гистограммы для AMo. Классический Баевский — 50 мс.
HIST_BIN_MS = 50.0


@dataclass
class BaevskyResult:
    si: Optional[float]            # Индекс напряжения (stress index)
    mode_ms: Optional[float]       # Mo — мода RR (мс)
    amo_pct: Optional[float]       # AMo — % RR в окрестности моды
    mxdmn_ms: Optional[float]      # Размах RR (мс)
    n: int                         # Число RR в расчёте
    zone: str                      # Категория: 'parasympathetic', 'normal', 'tension', 'high_stress', 'overload', 'insufficient'

    def to_dict(self) -> dict:
        return {
            "si":       round(self.si, 1)        if self.si       is not None else None,
            "mode_ms":  round(self.mode_ms, 1)   if self.mode_ms  is not None else None,
            "amo_pct":  round(self.amo_pct, 1)   if self.amo_pct  is not None else None,
            "mxdmn_ms": round(self.mxdmn_ms, 1)  if self.mxdmn_ms is not None else None,
            "n":        self.n,
            "zone":     self.zone,
        }


def _zone_from_si(si: float) -> str:
    if si < 50:    return "parasympathetic"   # < 50: ваготония / расслабление
    if si < 150:   return "normal"            # 50–150: норма
    if si < 500:   return "tension"           # 150–500: умеренное напряжение
    if si < 900:   return "high_stress"       # 500–900: выраженный стресс
    return "overload"                          # > 900: срыв адаптации


def compute_si(rr_intervals: list[float] | np.ndarray) -> BaevskyResult:
    """Индекс напряжения регуляторных систем (Баевский, 1979).

    SI = AMo / (2 · Mo · MxDMn),  где
      Mo     — мода RR (с)
      AMo    — доля RR в бине моды (десятичная)
      MxDMn  — размах RR (с)

    Чем выше SI — тем сильнее доминирует симпатика.
    """
    rr = np.asarray([v for v in rr_intervals if 300.0 <= v <= 2000.0], dtype=float)
    n  = len(rr)
    if n < MIN_RR_FOR_SI:
        return BaevskyResult(
            si=None, mode_ms=None, amo_pct=None, mxdmn_ms=None,
            n=n, zone="insufficient",
        )

    # Гистограмма с фиксированным шагом 50 мс — мода = центр самого популярного бина.
    rr_min = np.floor(rr.min() / HIST_BIN_MS) * HIST_BIN_MS
    rr_max = np.ceil(rr.max() / HIST_BIN_MS) * HIST_BIN_MS
    bins = np.arange(rr_min, rr_max + HIST_BIN_MS, HIST_BIN_MS)
    if len(bins) < 2:
        bins = np.array([rr_min, rr_min + HIST_BIN_MS])

    counts, edges = np.histogram(rr, bins=bins)
    mode_idx  = int(np.argmax(counts))
    mode_ms   = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)
    amo_count = int(counts[mode_idx])
    amo_pct   = 100.0 * amo_count / n
    mxdmn_ms  = float(rr.max() - rr.min())
    if mxdmn_ms < 1e-6:
        # Все RR в одной точке — индекс не определён
        return BaevskyResult(
            si=None, mode_ms=mode_ms, amo_pct=amo_pct, mxdmn_ms=mxdmn_ms,
            n=n, zone="insufficient",
        )

    # Классическая формула: AMo (%) / (2 · Mo (с) · MxDMn (с))
    # Норма ~50–150 получается именно при подстановке AMo в процентах
    # и Mo, MxDMn в секундах.
    mode_s  = mode_ms  / 1000.0
    mxdmn_s = mxdmn_ms / 1000.0
    si = amo_pct / (2.0 * mode_s * mxdmn_s)

    return BaevskyResult(
        si=si, mode_ms=mode_ms, amo_pct=amo_pct, mxdmn_ms=mxdmn_ms,
        n=n, zone=_zone_from_si(si),
    )
