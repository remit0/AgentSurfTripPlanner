import datetime as dt

from pydantic import BaseModel, Field


class EventTime(BaseModel):
    datetime: dt.datetime | None = Field(default=None, alias="dateTime")
    date: dt.date | None = None


class GoogleCalendarEvent(BaseModel):
    summary: str = Field(default="Busy")
    start: EventTime
    end: EventTime
    status: str | None = "confirmed"

    @property
    def start_string(self) -> str:
        """Helper to get the actual time string regardless of event type."""
        val = self.start.datetime or self.start.date
        if val:
            return val.isoformat()
        return "Unknown Time"
