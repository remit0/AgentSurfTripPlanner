from datetime import datetime

import requests

from .models import NavitiaJourney


class NavitiaAPIClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.auth = (api_key, "")
        self.base_url = "https://api.navitia.io/v1/coverage/sncf"

    def get_journeys(self, origin: str, destination: str, from_datetime: datetime) -> list[NavitiaJourney]:
        origin_id = self._find_station_id(origin)
        destination_id = self._find_station_id(destination)

        params = {
            "from": origin_id,
            "to": destination_id,
            "datetime": from_datetime.strftime("%Y%m%dT%H%M%S"),
            "count": 20,
            "commercial_mode_id[]": "commercial_mode:Train",
        }

        response = self.session.get(f"{self.base_url}/journeys", params=params)
        response.raise_for_status()

        items = response.json().get("journeys", [])
        return [NavitiaJourney(**item) for item in items]

    def _find_station_id(self, city_name: str) -> str:
        params = {"q": city_name}
        response = self.session.get(f"{self.base_url}/places", params=params)
        response.raise_for_status()

        places = response.json().get("places", [])
        if not places:
            raise ValueError(f"Station '{city_name}' not found.")
        return places[0]["id"]
