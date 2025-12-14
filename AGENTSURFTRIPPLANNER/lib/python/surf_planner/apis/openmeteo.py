import datetime
from datetime import date

import requests
from pydantic import BaseModel, Field


class OpenMeteoForecast(BaseModel):
    date: datetime.date = Field(alias="day")
    wave_height_m: float | None = None
    wave_period_s: float | None = None
    wind_speed_kmh: float | None = None


class OpenMeteoAPIClient:

    def __init__(self):
        self.session = requests.Session()
        self.marine_url = "https://marine-api.open-meteo.com/v1/marine"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"

    def get_forecasts(self, latitude: float, longitude: float, from_date: date, to_date: date) -> list[OpenMeteoForecast]:
        """
        Returns a list of validated Pydantic objects.
        Input dates are standard Python date objects (no string formatting required by caller).
        """
        wind_data = self._get_wind_forecast(from_date, to_date, latitude, longitude)
        wave_data = self._get_wave_forecast(from_date, to_date, latitude, longitude)
        merged_dicts = self._merge_forecast_data(wind_data, wave_data)
        return [OpenMeteoForecast(**item) for item in merged_dicts]

    def _get_wave_forecast(self, from_date: date, to_date: date, latitude: float, longitude: float) -> list[dict]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": from_date.isoformat(),
            "end_date": to_date.isoformat(),
            "daily": ["wave_height_max", "wave_period_max"],
            "timezone": "Europe/Paris"
        }
        response = self.session.get(self.marine_url, params=params)
        response.raise_for_status()
        daily = response.json().get("daily", {})

        return [
            {"day": d, "wave_height_m": wh, "wave_period_s": wp}
            for d, wh, wp in zip(
                daily.get("time", []),
                daily.get("wave_height_max", []),
                daily.get("wave_period_max", [])
            )
        ]

    def _get_wind_forecast(self, from_date: date, to_date: date, latitude: float, longitude: float) -> list[dict]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": from_date.isoformat(),
            "end_date": to_date.isoformat(),
            "daily": ["wind_speed_10m_max"],
            "timezone": "Europe/Paris"
        }
        response = self.session.get(self.forecast_url, params=params)
        response.raise_for_status()
        daily = response.json().get("daily", {})

        return [
            {"day": d, "wind_speed_kmh": ws}
            for d, ws in zip(daily.get("time", []), daily.get("wind_speed_10m_max", []))
        ]

    def _merge_forecast_data(self, wind_data: list, wave_data: list) -> list[dict]:
        """Merges two lists of dicts based on the 'day' key."""
        merged = {item['day']: item for item in wind_data}
        for item in wave_data:
            if item['day'] in merged:
                merged[item['day']].update(item)
        return list(merged.values())
