import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from surf_planner.apis.google_calendar import GoogleCalendarAPIClient, GoogleCalendarEvent


class CheckCalendarArgs(BaseModel):
    from_date: datetime.date = Field(description="The start date (YYYY-MM-DD).")
    to_date: datetime.date = Field(description="The end date (YYYY-MM-DD).")


class DayAvailability(BaseModel):
    date: datetime.date
    meetings_end_at: datetime.datetime

    def to_readable_string(self) -> str:
        """Custom formatting logic for the LLM."""
        if self.meetings_end_at.time() == datetime.time.min:
            return f"{self.date.isoformat()}: Free all day"
        else:
            formatted_time = self.meetings_end_at.strftime('%H:%M')
            return f"{self.date.isoformat()}: Available after {formatted_time}"

    def __str__(self) -> str:
        return self.to_readable_string()

    def __repr__(self) -> str:
        return self.to_readable_string()


def _get_last_events_per_day(events: list[GoogleCalendarEvent]) -> dict[datetime.date, datetime.datetime]:
    """
    Processes a list of Pydantic event models to find the end time of the last event for each day.
    Uses native datetime attributes directly.
    """
    last_events = {}

    for event in events:
        final_end_dt = None

        # Case A: Timed event (e.g., 14:00 - 15:30)
        if event.end.datetime:
            final_end_dt = event.end.datetime

        # Case B: All-day event (e.g., 2023-01-01)
        elif event.end.date:
            # Treat end of day as 00:00 UTC of the requested date (standard for all-day logic)
            final_end_dt = datetime.datetime.combine(event.end.date, datetime.time.min).replace(
                tzinfo=datetime.timezone.utc
            )

        if not final_end_dt:
            continue

        # Normalization: Ensure timezone awareness
        if final_end_dt.tzinfo is None:
            final_end_dt = final_end_dt.replace(tzinfo=datetime.timezone.utc)

        day = final_end_dt.date()

        # Update if this event ends later than what we have recorded for this day
        if day not in last_events or final_end_dt > last_events[day]:
            last_events[day] = final_end_dt

    return last_events


def create_calendar_tool(calendar_client: GoogleCalendarAPIClient, calendar_id: str):
    """
    Factory that builds the 'check_calendar' tool with the API client injected via closure.

    This function implements the Factory Pattern to achieve clean Dependency Injection.
    By defining the `@tool` function inside this factory, the inner `check_calendar` function
    captures the `calendar_client` instance from the outer scope (closure).

    This allows the tool logic to make API calls using the provided client, while keeping
    the client invisible to the LLM (which only sees the public `from_date` and `to_date` arguments).

    Args:
        calendar_client (GoogleCalendarAPIClient): An authenticated client instance for fetching events.
        :param calendar_client:
        :param calendar_id: string

    Returns:
        StructuredTool: A configured LangChain tool ready to be bound to an agent.
    """

    @tool(args_schema=CheckCalendarArgs)
    def check_calendar(from_date: datetime.date, to_date: datetime.date) -> list[DayAvailability] | str:
        """
        Use this tool to check the personal calendar between two dates.
        """
        try:
            # 1. Convert to datetime for the API call (Start of day / End of day)
            from_dt = datetime.datetime.combine(from_date, datetime.time.min)
            to_dt = datetime.datetime.combine(to_date, datetime.time.max)

            # 2. Use the injected client
            events = calendar_client.list_events(from_dt, to_dt, calendar_id)

            # 3. Process events
            last_events_per_day = _get_last_events_per_day(events)

            availabilities = []
            current_day = from_date

            while current_day <= to_date:
                if current_day in last_events_per_day:
                    end_time = last_events_per_day[current_day]
                else:
                    end_time = datetime.datetime.combine(
                        current_day,
                        datetime.time.min
                    ).replace(tzinfo=datetime.timezone.utc)

                availabilities.append(
                    DayAvailability(date=current_day, meetings_end_at=end_time)
                )
                current_day += datetime.timedelta(days=1)

            return availabilities

        except Exception as e:
            return f"An unexpected error occurred: {e}"

    return check_calendar


def create_dummy_calendar_tool():
    """
    Creates a 'check_calendar' tool that always returns 'Free all day'.
    Used when GCP credentials are not provided.
    """

    @tool(args_schema=CheckCalendarArgs)
    def check_calendar(from_date: datetime.date, to_date: datetime.date) -> list[DayAvailability]:
        """
        Use this tool to check the personal calendar between two dates.
        (SIMULATION MODE: Returns free availability for all requested dates)
        """
        availabilities = []
        current_day = from_date

        while current_day <= to_date:
            # Create a "Free all day" entry (end time is 00:00)
            free_time = datetime.datetime.combine(
                current_day,
                datetime.time.min
            ).replace(tzinfo=datetime.timezone.utc)

            availabilities.append(
                DayAvailability(date=current_day, meetings_end_at=free_time)
            )
            current_day += datetime.timedelta(days=1)
        return availabilities

    return check_calendar
