from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Base for every contract: immutable, and an unknown field is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)
