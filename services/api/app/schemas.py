"""Response and WebSocket message models of the dashboard API.

The frontend generates its TypeScript types from ``openapi.json``. The
WebSocket messages get there too: ``app.main`` adds ``WsMessage`` and the
models it refers to to ``components.schemas``, although no endpoint
returns them.
"""

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, Field, TypeAdapter

from app.models import AlertRow, RiskLevel
from contracts import ReasonCode

Freshness = Literal["active", "stale", "offline"]
DegradedReason = Literal["predictor_unavailable", "stale_telemetry", "warming_up"]


class PredictionView(BaseModel):
    """A prediction as the dashboard shows it; the address comes from ``stops_plan``."""

    sample_id: str
    t: AwareDatetime
    target_stop_id: int
    target_address: str | None
    target_time_begin: AwareDatetime
    prediction_s: float
    p_late: float | None
    reasons: list[ReasonCode]
    risk_level: RiskLevel
    degraded: bool
    degraded_reason: DegradedReason | None
    model_version: str


class VehicleView(BaseModel):
    """A vehicle on the map: last position, freshness and current prediction.

    Position fields come from the last point with a valid position and are
    all ``None`` if the window has none; ``last_seen`` is the last point,
    valid or not.
    """

    tr_id: int
    unit_id: int
    lat: float | None
    lon: float | None
    heading_deg: float | None
    speed_kmh: float | None
    last_seen: AwareDatetime
    freshness: Freshness
    prediction: PredictionView | None


class RiskCounts(BaseModel):
    """Vehicles per risk level; ``none`` counts those without a current prediction."""

    green: int
    yellow: int
    red: int
    none: int


class FreshnessCounts(BaseModel):
    """Vehicles per freshness."""

    active: int
    stale: int
    offline: int


class Summary(BaseModel):
    """Counters over the vehicles of a snapshot."""

    risk: RiskCounts
    freshness: FreshnessCounts


class AlertView(AlertRow):
    """An alert as the dashboard shows it: the fields of an ``alerts`` row."""


class StateSnapshot(BaseModel):
    """The whole dashboard state as of message ``seq``.

    ``alerts`` stays empty and ``live_mae_s`` ``None`` until alerts and
    fact checking exist; the shape does not change when they do.
    """

    seq: int
    clock: AwareDatetime | None
    vehicles: list[VehicleView]
    summary: Summary
    alerts: list[AlertView]
    live_mae_s: float | None
    checked_predictions: int


class TrackPoint(BaseModel):
    """A point with a valid position on the vehicle's recent track."""

    event_time: AwareDatetime
    lat: float
    lon: float
    speed_kmh: float | None


class CardStop(BaseModel):
    """A planned stop near the clock; the fact fields are set once a StopEvent came."""

    stop_id: int
    address: str | None
    lat: float
    lon: float
    time_plan: AwareDatetime
    time_fact: AwareDatetime | None
    delay_s: float | None


class NetworkStop(BaseModel):
    """A stop in one planned vehicle run, ordered by planned arrival time."""

    route_id: int
    stop_order: int
    stop_id: int
    address: str | None
    lat: float
    lon: float
    time_plan: AwareDatetime


class VehicleCard(BaseModel):
    """One vehicle in detail; ``seq`` and ``clock`` as in the snapshot."""

    seq: int
    clock: AwareDatetime | None
    vehicle: VehicleView | None
    predictions: list[PredictionView]
    track: list[TrackPoint]
    stops: list[CardStop]


class ClockData(BaseModel):
    """Sequence number and current clock time."""

    seq: int
    clock: AwareDatetime


class VehiclesData(BaseModel):
    """Vehicles whose view changed since the last ``vehicles`` message, and those gone."""

    seq: int
    clock: AwareDatetime
    vehicles: list[VehicleView]
    removed: list[int]


class PredictionData(BaseModel):
    """Vehicle prediction update with sequence number."""

    seq: int
    tr_id: int
    prediction: PredictionView


class AlertData(BaseModel):
    """Alert with sequence number."""

    seq: int
    alert: AlertView


class ClockMessage(BaseModel):
    """WebSocket message: current clock (type: clock)."""

    type: Literal["clock"]
    data: ClockData


class VehiclesMessage(BaseModel):
    """WebSocket message: vehicle updates (type: vehicles)."""

    type: Literal["vehicles"]
    data: VehiclesData


class PredictionMessage(BaseModel):
    """WebSocket message: prediction update (type: prediction)."""

    type: Literal["prediction"]
    data: PredictionData


class AlertMessage(BaseModel):
    """WebSocket message: alert (type: alert)."""

    type: Literal["alert"]
    data: AlertData


WsMessage = Annotated[
    ClockMessage | VehiclesMessage | PredictionMessage | AlertMessage,
    Field(discriminator="type"),
]
"""Every message the dashboard WebSocket carries, told apart by ``type``."""

WS_MESSAGE = TypeAdapter(WsMessage)


def ws_message_schemas(ref_template: str) -> dict[str, dict[str, Any]]:
    """JSON schemas of ``WsMessage`` and every model it refers to, by name.

    ``ref_template`` is where the schemas will live, e.g.
    ``"#/components/schemas/{model}"`` for OpenAPI.
    """
    schema = WS_MESSAGE.json_schema(mode="serialization", ref_template=ref_template)
    definitions = schema.pop("$defs")
    return definitions | {"WsMessage": schema}
