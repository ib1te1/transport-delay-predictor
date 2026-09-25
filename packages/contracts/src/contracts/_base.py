from datetime import UTC
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict


class Contract(BaseModel):
    """Base for every contract: immutable, and an unknown field is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda v: v.astimezone(UTC))]
"""An aware datetime normalized to UTC, so e.g. ``+03:00`` input round-trips as UTC."""
