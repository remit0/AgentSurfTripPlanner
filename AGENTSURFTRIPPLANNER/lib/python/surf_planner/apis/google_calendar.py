import datetime
import logging

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

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


class GoogleCalendarAPIClient:
    """
    A dedicated wrapper for the Google Calendar API.
    Handles authentication and raw API calls.
    """
    def __init__(self, service_account_info: dict):
        self._service_account_info = service_account_info
        self._service = self._authenticate()

    def _authenticate(self):
        credentials = service_account.Credentials.from_service_account_info(self._service_account_info, scopes=SCOPES)
        service = build("calendar", "v3", credentials=credentials)
        return service

    def list_events(self, start_date: datetime.datetime, end_date: datetime.datetime, calendar_id: str, max_results=10):
        """
        Fetches events within a specific time range.
        Input: Python datetime objects.
        Output: List of event dictionaries (raw API response).
        """
        try:
            # Google requires RFC3339 strings with 'Z' for UTC
            time_min = start_date.isoformat() + "Z"
            time_max = end_date.isoformat() + "Z"

            logging.debug(f"Fetching calendar events from {time_min} to {time_max}")
            events_result = (
                self._service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            raw_items = events_result.get("items", [])
            return [GoogleCalendarEvent(**item) for item in raw_items]

        except HttpError as error:
            logging.error(f"An error occurred in Google Calendar API: {error}")
            return []
