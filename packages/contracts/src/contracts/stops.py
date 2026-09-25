from contracts._base import Contract, UtcDatetime


class StopEvent(Contract):
    """A detected arrival at a planned stop; ``delay_s`` feeds ``cur_dev_s``."""

    tr_id: int
    stop_id: int
    time_plan: UtcDatetime
    time_fact: UtcDatetime
    delay_s: float
