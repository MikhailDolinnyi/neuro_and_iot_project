from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
WINDOW_SIZE = 20


class SessionManager:
    def __init__(self) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        self._current: Optional[dict] = None

    @property
    def is_recording(self) -> bool:
        return self._current is not None

    @property
    def current_label(self) -> Optional[str]:
        return self._current["label"] if self._current else None

    @property
    def current_count(self) -> int:
        return len(self._current["readings"]) if self._current else 0

    def start(self, label: str) -> str:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current = {
            "id": session_id,
            "label": label,
            "started_at": datetime.now().isoformat(),
            "readings": [],
        }
        return session_id

    def add_reading(
        self, bpm: float, temp_c: float, fsr_raw: int, stress_score: Optional[float]
    ) -> None:
        if self._current:
            self._current["readings"].append({
                "ts": datetime.now().isoformat(),
                "bpm": bpm,
                "temp_c": temp_c,
                "fsr_raw": fsr_raw,
                "stress_score": stress_score,
            })

    def stop(self) -> Optional[str]:
        if not self._current:
            return None
        self._current["stopped_at"] = datetime.now().isoformat()
        session_id = self._current["id"]
        label = self._current["label"]
        path = SESSIONS_DIR / f"{session_id}_{label}.json"
        path.write_text(json.dumps(self._current, indent=2, ensure_ascii=False))
        self._current = None
        return session_id

    def list_all(self) -> list[dict]:
        sessions = []
        for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "id": data["id"],
                    "label": data["label"],
                    "started_at": data["started_at"],
                    "readings_count": len(data.get("readings", [])),
                })
            except Exception:
                continue
        return sessions

    def build_training_data(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Извлечь оконные признаки из всех сохранённых сессий.

        Возвращает (X_tab, X_win, y):
          X_tab — табличные признаки (n, 16) для RF/GBM.
          X_win — сырые временные ряды  (n, window_size, 3) для ROCKET.
          y     — метки классов.
        """
        from .features import extract_features
        from .window import Reading

        X_tab, X_win, y = [], [], []
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                raw = data.get("readings", [])
                label = data["label"]
                if len(raw) < WINDOW_SIZE:
                    continue
                step = max(1, WINDOW_SIZE // 2)
                for i in range(0, len(raw) - WINDOW_SIZE + 1, step):
                    chunk = raw[i: i + WINDOW_SIZE]
                    readings = [
                        Reading(
                            bpm=r["bpm"], temp_c=r["temp_c"], fsr_raw=r["fsr_raw"],
                            bpm_valid=True, temp_valid=True,
                        )
                        for r in chunk
                    ]
                    X_tab.append(extract_features(readings))

                    bpm_arr  = np.array([r["bpm"]     for r in chunk], dtype=float)
                    temp_arr = np.array([r["temp_c"]  for r in chunk], dtype=float)
                    fsr_arr  = np.array([r["fsr_raw"] for r in chunk], dtype=float)
                    X_win.append(np.column_stack([bpm_arr, temp_arr, fsr_arr]))

                    y.append(label)
            except Exception:
                continue

        if not X_tab:
            return None, None, None
        return (
            np.array(X_tab, dtype=float),
            np.array(X_win, dtype=float),
            np.array(y),
        )
