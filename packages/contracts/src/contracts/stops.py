from pydantic import AwareDatetime

from contracts._base import Contract


class StopEvent(Contract):
    """A detected arrival at a planned stop; ``delay_s`` feeds ``cur_dev_s``."""

    tr_id: int
    stop_id: int
    time_plan: AwareDatetime
    time_fact: AwareDatetime
    delay_s: float
