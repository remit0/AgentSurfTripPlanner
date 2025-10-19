from datetime import datetime

import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from .data_models import TrainTicket


class TrainTicketsArgs(BaseModel):
    origin: str = Field(description="The city of departure.")
    destination: str = Field(description="The destination city.")
    from_datetime: str = Field(description="The datetime from which to check, in ISO format.")


class FindTrainTicketsTool(BaseTool):
    """A tool to find train tickets for a specific day from a given time."""

    # --- Tool metadata ---
    name: str = "find_train_tickets"
    description: str = "Use this tool to find train tickets for a specific day from a given time."
    args_schema: type[BaseModel] = TrainTicketsArgs

    # --- Class-specific attributes (dependencies passed in) ---
    session: requests.Session

    def _run(self, origin: str, destination: str, from_datetime: str) -> list[TrainTicket] | str:
        """Use the tool with error handling."""
        try:
            start_datetime = datetime.fromisoformat(from_datetime)
            origin_id = self._find_station_id(origin)
            destination_id = self._find_station_id(destination)

            params = {
                "from": origin_id,
                "to": destination_id,
                "datetime": start_datetime.strftime("%Y%m%dT%H%M%S"),
                "count": 20,
                "commercial_mode_id[]": "commercial_mode:Train",
            }
            response = self.session.get("https://api.navitia.io/v1/coverage/sncf/journeys", params=params)
            response.raise_for_status()

            journeys = response.json().get("journeys", [])

            if not journeys:
                return "No train journeys were found for the specified route and time."

            return self._parse_journeys(journeys, start_datetime)

        except ValueError as e:
            # Catches errors from fromisoformat() or if a station is not found
            return f"Error: Invalid input provided. Details: {e}"
        except requests.exceptions.RequestException as e:
            # Catches network or HTTP errors from the API call
            return f"Error: The train information service is currently unavailable. Please try again later. Details: {e}"
        except Exception as e:
            # A general catch-all for any other unexpected errors
            return f"An unexpected error occurred: {e}"

    def _find_station_id(self, city_name: str) -> str:
        """Private helper to find the Navitia ID for a station."""
        params = {"q": city_name}
        response = self.session.get("https://api.navitia.io/v1/coverage/sncf/places", params=params)
        response.raise_for_status()
        places = response.json().get("places", [])
        if not places:
            raise ValueError(f"Station '{city_name}' not found.")
        return places[0]["id"]

    def _parse_journeys(self, journeys: list, from_datetime: datetime) -> list[TrainTicket]:
        """Private helper to parse the raw journey JSON into TrainTicket objects."""
        journeys_today = [
            j for j in journeys
            if datetime.strptime(j['departure_date_time'], "%Y%m%dT%H%M%S").date() == from_datetime.date()
        ]

        train_tickets = []
        for journey in journeys_today:
            sections = journey.get('sections', [])
            if not sections: continue

            origin = sections[0].get('from', {}).get('name', 'N/A')
            destination = sections[-1].get('to', {}).get('name', 'N/A')
            duration_s = journey['duration']

            train_tickets.append(
                TrainTicket(
                    origin=origin,
                    destination=destination,
                    date=from_datetime.date(),
                    departure_time=datetime.strptime(journey['departure_date_time'], "%Y%m%dT%H%M%S"),
                    arrival_time=datetime.strptime(journey['arrival_date_time'], "%Y%m%dT%H%M%S"),
                    duration=f"{duration_s // 3600}h {(duration_s % 3600) // 60}m"
                )
            )
        return train_tickets
