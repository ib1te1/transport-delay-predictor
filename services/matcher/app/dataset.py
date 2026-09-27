"""The small, internal types used by the stop detector."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Visit:
    vehicle_id: str
    visit_id: str
    planned_at: datetime
    lon: float | None
    lat: float | None


@dataclass(frozen=True)
class Tick:
    vehicle_id: str
    at: datetime
    lon: float | None
    lat: float | None
    speed: float | None

    @property
    def valid(self) -> bool:
        return self.lon is not None and self.lat is not None
