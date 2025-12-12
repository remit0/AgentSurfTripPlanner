import datetime

from pydantic import BaseModel, Field


class OpenMeteoForecast(BaseModel):
    date: datetime.date = Field(alias="day")
    wave_height_m: float | None = None
    wave_period_s: float | None = None
    wind_speed_kmh: float | None = None
