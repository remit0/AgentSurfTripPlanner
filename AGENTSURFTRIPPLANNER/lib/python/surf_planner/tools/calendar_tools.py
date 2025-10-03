import datetime

from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from .data_models import DayAvailability


class CheckCalendarArgs(BaseModel):
    from_date: str = Field(description="The start date as a string with YYYY-MM-DD format.")
    to_date: str = Field(description="The end date as a string with YYYY-MM-DD format.")


class GetCalendarEventsTool(BaseTool):
    """A tool to check the personal calendar and find the end of the workday."""

    # --- Tool metadata ---
    name: str = "check_calendar"
    description: str = (
        "Use this tool to check the personal calendar between two dates. "
        "For every day between the two dates, it returns the datetime at which the user ends their last meeting."
    )
    args_schema: type[BaseModel] = CheckCalendarArgs

    # --- Class-specific attributes (passed during initialization) ---
    service: Resource  # The Google Calendar API service object
    calendar_id: str

    def _run(self, from_date: str, to_date: str) -> list[DayAvailability] | str:
        """Use the tool with error handling and correct availability logic."""
        try:
            start_day = datetime.date.fromisoformat(from_date)
            end_day = datetime.date.fromisoformat(to_date)

            # Datetime versions for the API call
            from_dt = datetime.datetime.combine(start_day, datetime.time.min)
            to_dt = datetime.datetime.combine(end_day, datetime.time.max)

            raw_events = self._list_events(from_dt, to_dt)
            last_events_per_day = self._get_last_events_per_day(raw_events)

            availabilities = []

            current_day = start_day
            while current_day <= end_day:
                # Check if we found a last event for the current day
                if current_day in last_events_per_day:
                    # If yes, the user is free after their last meeting
                    end_time = last_events_per_day[current_day]
                else:
                    # If no, the user is free all day. We represent this with the
                    # beginning of the day (midnight).
                    end_time = datetime.datetime.combine(current_day, datetime.time.min)

                availabilities.append(
                    DayAvailability(date=current_day, meetings_end_at=end_time)
                )
                current_day += datetime.timedelta(days=1)

            return availabilities

        except ValueError:
            return "Error: The date format provided is incorrect. Please use 'YYYY-MM-DD'."
        except HttpError as e:
            return f"Error: An error occurred with the Google Calendar API: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def _list_events(self, from_datetime: datetime.datetime, to_datetime: datetime.datetime) -> list[dict]:
        """Private method to fetch raw events from the Google Calendar API."""
        time_min = from_datetime.isoformat() + 'Z'
        time_max = to_datetime.isoformat() + 'Z'

        events_result = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        return events_result.get('items', [])

    def _get_last_events_per_day(self, raw_events: list[dict]) -> dict:
        """Processes a list of events to find the end time of the last event for each day."""
        last_events = {}
        for event in raw_events:
            end_time_str = event['end'].get('dateTime', event['end'].get('date'))
            end_time = datetime.datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            day = end_time.date()

            if day not in last_events or end_time > last_events[day]:
                last_events[day] = end_time
        return last_events
