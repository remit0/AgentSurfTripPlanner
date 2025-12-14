import datetime
from datetime import date

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from surf_planner.apis.geolocator import GeolocatorAPIClient
from surf_planner.apis.openmeteo import OpenMeteoAPIClient


class SurfForecastArgs(BaseModel):
    spot: str = Field(description="The name of the surf spot or town.")
    from_date: datetime.date = Field(description="The start date (YYYY-MM-DD).")
    to_date: datetime.date = Field(description="The end date (YYYY-MM-DD).")



class DailySurfForecast(BaseModel):
    date: date
    spot: str
    wave_height_m: float
    wave_period_s: float
    wind_speed_kmh: float

    def to_readable_string(self) -> str:
        """Formats the forecast into a concise string for the LLM."""
        return (f"Date: {self.date.isoformat()}, Spot: {self.spot}, "
                f"Waves: {self.wave_height_m}m, Period: {self.wave_period_s}s, "
                f"Wind: {self.wind_speed_kmh}km/h")

    def __str__(self) -> str:
        return self.to_readable_string()

    def __repr__(self) -> str:
        return self.to_readable_string()


def create_surf_forecast_tool(geolocator: GeolocatorAPIClient, openmeteo: OpenMeteoAPIClient):

    @tool(args_schema=SurfForecastArgs)
    def get_surf_forecast(spot: str, from_date: datetime.date, to_date: datetime.date) -> list[DailySurfForecast] | str:
        """
        Retrieve the surf forecast (wave height, period, and wind) for a specific spot between two dates.

        Use this tool when the user asks about surf conditions, waves, or weather for a trip.
        Returns a list of daily forecasts including wave height in meters, period in seconds, and wind speed.
        """
        try:
            location = geolocator.get_coordinates(spot)
            forecasts_data = openmeteo.get_forecasts(
                latitude=location.latitude,
                longitude=location.longitude,
                from_date=from_date,
                to_date=to_date
            )

            if not forecasts_data:
                return "No forecast data could be retrieved for the specified dates."

            return [
                DailySurfForecast(
                    spot=spot,
                    date=f.date,
                    wave_height_m=f.wave_height_m,
                    wave_period_s=f.wave_period_s,
                    wind_speed_kmh=f.wind_speed_kmh
                )
                for f in forecasts_data
            ]

        # Clean error handling based on the specific clients
        except ValueError as e:
            return f"Error: {e}" # "Location not found"
        except RuntimeError as e:
            return f"Service Error: {e}" # "Geocoding service unavailable"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    return get_surf_forecast
