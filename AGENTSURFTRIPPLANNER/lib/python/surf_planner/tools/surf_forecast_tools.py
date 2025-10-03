import datetime

import requests
from geopy.geocoders import Nominatim
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

from .data_models import DailySurfForecast


class SurfForecastArgs(BaseModel):
    spot: str = Field(description="The name of the surf spot or town to check the forecast for.")
    from_date: str = Field(description="The start date as a string with YYYY-MM-DD format.")
    to_date: str = Field(description="The end date as a string with YYYY-MM-DD format.")


class GetSurfForecastTool(BaseTool):
    """A tool to retrieve the surf forecast between two dates for a specific spot."""

    # --- Tool metadata ---
    name: str = "get_surf_forecast"
    description: str = (
        "Use this tool to retrieve the surf forecast between two dates for a specific spot."
    )
    args_schema: type[BaseModel] = SurfForecastArgs

    # --- Class-specific attributes (dependencies passed in) ---
    session: requests.Session
    geolocator: Nominatim

    def _run(self, spot: str, from_date: str, to_date: str) -> list[DailySurfForecast] | str:
        """Use the tool with error handling."""

        try:

            coordinates = self._get_coordinates(spot)
            latitude, longitude = coordinates

            wind_forecast = self._get_wind_forecast(from_date, to_date, latitude, longitude)
            wave_forecast = self._get_wave_forecast(from_date, to_date, latitude, longitude)

            forecasts = self._merge_forecast_data(wind_forecast, wave_forecast)

            if not forecasts:
                return "No forecast data could be retrieved for the specified dates."

            return [
                DailySurfForecast(
                    spot=spot,
                    date=datetime.date.fromisoformat(f["day"]),
                    wave_height_m=f.get("wave_height_m"),
                    wave_period_s=f.get("wave_period_s"),
                    wind_speed_kmh=f.get("wind_speed_kmh")
                )
                for f in forecasts
            ]
        except ValueError as e:
            # Catches errors from fromisoformat() or if a spot is not found
            return f"Error: Invalid input provided. Details: {e}"
        except requests.exceptions.RequestException:
            # Catches network or HTTP errors from the forecast API call
            return "Error: The weather forecast service is currently unavailable. Please try again later."
        except GeocoderServiceError:
            # Catches errors from the geocoding service
            return "Error: The location service is currently unavailable. Please try again later."
        except Exception as e:
            # A general catch-all for any other unexpected errors
            return f"An unexpected error occurred: {e}"

    def _get_coordinates(self, location_name: str) -> tuple[float, float] | None:
        """Gets the latitude and longitude for a given location name."""
        try:
            location = self.geolocator.geocode(location_name)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Error in geocoding: {e}")
        return None

    def _get_wave_forecast(self, from_date, to_date, latitude, longitude) -> list:
        """Fetches wave forecast data."""
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": from_date, "end_date": to_date,
            "daily": ["wave_height_max", "wave_period_max"],
            "timezone": "Europe/Paris"
        }
        response = self.session.get("https://marine-api.open-meteo.com/v1/marine", params=params)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        return [{"day": d, "wave_height_m": wh, "wave_period_s": wp}
                for d, wh, wp in
                zip(daily.get("time", []), daily.get("wave_height_max", []), daily.get("wave_period_max", []))]

    def _get_wind_forecast(self, from_date, to_date, latitude, longitude) -> list:
        """Fetches wind forecast data."""
        params = {
            "latitude": latitude, "longitude": longitude,
            "start_date": from_date, "end_date": to_date,
            "daily": ["wind_speed_10m_max"],
            "timezone": "Europe/Paris"
        }
        response = self.session.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        daily = response.json().get("daily", {})
        return [{"day": d, "wind_speed_kmh": ws}
                for d, ws in zip(daily.get("time", []), daily.get("wind_speed_10m_max", []))]

    @staticmethod
    def _merge_forecast_data(wind_data: list, wave_data: list) -> list:
        """Merges wind and wave data lists."""
        merged_data = {item['day']: item for item in wind_data}
        for item in wave_data:
            if item['day'] in merged_data:
                merged_data[item['day']].update(item)
        return list(merged_data.values())