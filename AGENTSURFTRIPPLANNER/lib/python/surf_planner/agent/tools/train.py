import datetime as dt

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from surf_planner.apis.navitia.client import NavitiaAPIClient
from surf_planner.apis.navitia.models import NavitiaJourney


class TrainTicket(BaseModel):
    """
    Represents a single train journey option returned to the agent.
    """
    date: dt.date
    origin: str
    destination: str
    departure_time: dt.datetime
    arrival_time: dt.datetime
    duration: str

    def to_readable_string(self) -> str:
        """Formats the ticket into a natural language string for the LLM."""
        dep_time = self.departure_time.strftime("%H:%M")
        arr_time = self.arrival_time.strftime("%H:%M")

        return (
            f"Train from {self.origin} to {self.destination} on {self.date.isoformat()}: "
            f"Departs {dep_time}, Arrives {arr_time}, "
            f"Duration: {self.duration}"
        )

    def __str__(self) -> str:
        return self.to_readable_string()

    def __repr__(self) -> str:
        return self.to_readable_string()


class FindTrainTicketsArgs(BaseModel):
    origin: str = Field(description="The city of departure (e.g., 'Paris').")
    destination: str = Field(description="The destination city (e.g., 'Bordeaux').")
    from_datetime: dt.datetime = Field(
        description="The start datetime to search from, in ISO format (e.g., '2023-10-01T08:00:00')."
    )


def _process_journeys(journeys: list[NavitiaJourney], target_date: dt.date) -> list[TrainTicket]:
    """
    Filters raw API journeys to strictly match the requested date and formats them.
    """
    tickets = []

    for journey in journeys:
        try:
            # Navitia returns "YYYYMMDDTHHMMSS" format
            dep_dt = dt.datetime.strptime(journey.departure_date_time, "%Y%m%dT%H%M%S")
            arr_dt = dt.datetime.strptime(journey.arrival_date_time, "%Y%m%dT%H%M%S")
        except ValueError:
            continue

        # Filter: Ensure the journey actually starts on the requested day
        if dep_dt.date() != target_date:
            continue

        origin_name = "Unknown Origin"
        dest_name = "Unknown Destination"

        # Navitia journeys consist of 'sections'.
        # Usually, Section 0 is the start and Section -1 is the end.
        if journey.sections:
            if journey.sections[0].from_data:
                origin_name = journey.sections[0].from_data.get("name", origin_name)

            if journey.sections[-1].to_data:
                dest_name = journey.sections[-1].to_data.get("name", dest_name)

        # Duration formatting (seconds -> H h M m)
        hours = journey.duration // 3600
        minutes = (journey.duration % 3600) // 60
        duration_str = f"{hours}h {minutes}m"

        tickets.append(
            TrainTicket(
                origin=origin_name,
                destination=dest_name,
                date=dep_dt.date(),
                departure_time=dep_dt,
                arrival_time=arr_dt,
                duration=duration_str
            )
        )
    return tickets


def create_train_ticket_tool(navitia_client: NavitiaAPIClient):
    """
    Factory that returns a 'find_train_tickets' tool with the Navitia client injected.
    """

    @tool(args_schema=FindTrainTicketsArgs)
    def find_train_tickets(origin: str, destination: str, from_datetime: dt.datetime) -> list[TrainTicket] | str:
        """
        Find available train tickets between two cities for a specific date and time.

        Returns a list of ticket options including departure/arrival times and duration.
        Use this tool to plan travel logistics or check transport availability.
        """
        try:
            # 1. Fetch raw data from the injected client
            raw_journeys = navitia_client.get_journeys(
                origin=origin,
                destination=destination,
                from_datetime=from_datetime
            )

            if not raw_journeys:
                return f"No train journeys were found from {origin} to {destination}."

            # 2. Process and filter results
            tickets = _process_journeys(raw_journeys, from_datetime.date())

            if not tickets:
                return (
                    f"Journeys found, but none match the exact requested date of {from_datetime.date()}. "
                    "They might be on the following day."
                )

            return tickets

        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred while searching for trains: {e}"

    return find_train_tickets
