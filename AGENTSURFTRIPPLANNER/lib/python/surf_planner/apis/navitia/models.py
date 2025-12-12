from typing import Any

from pydantic import BaseModel, Field


class NavitiaSection(BaseModel):
    from_data: dict[str, Any] | None = Field(default=None, alias="from")
    to_data: dict[str, Any] | None = Field(default=None, alias="to")

class NavitiaJourney(BaseModel):
    departure_date_time: str
    arrival_date_time: str
    duration: int
    sections: list[NavitiaSection] = []
