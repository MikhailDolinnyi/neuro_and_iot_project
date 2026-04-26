from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class HealthSnapshot(BaseModel):
    # Accept both snake_case (from direct POST) and camelCase (from Kotlin/Jackson)
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    device: str
    state: str
    max30102_ready: bool = False
    finger: bool = False
    warmup: bool = False
    ir: int = 0
    bpm_valid: bool = False
    bpm_current: float = 0.0
    bpm_avg: int = 0
    bpm_samples: int = 0
    temp_valid: bool = False
    temp_c: float = 0.0
    fsr_raw: int = 0
    fsr_pressed: bool = False
    rr_intervals: list[int] = []  # RR-интервалы в мс, присланные Arduino за последние 3 с


class FeatureContrib(BaseModel):
    name: str
    label: str
    value: float
    importance: float


class Assessment(BaseModel):
    stress_level: str
    stress_score: Optional[float] = None
    confidence: Optional[float] = None
    valence: Optional[float] = None   # −1 (негатив) … +1 (позитив)
    arousal: Optional[float] = None   # 0 (покой) … 1 (высокое возбуждение)
    engine: str
    window_size: int
    explanation: list[FeatureContrib] = []


class ReadingResponse(BaseModel):
    status: str
    state: str
    assessment: Optional[Assessment] = None
