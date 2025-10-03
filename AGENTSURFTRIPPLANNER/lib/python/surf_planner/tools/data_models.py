import datetime
from dataclasses import dataclass


@dataclass
class DayAvailability:
    date: datetime.date
    meetings_end_at: datetime.datetime

    def __str__(self) -> str:
        # Check if the time is midnight, which signifies a full free day.
        if self.meetings_end_at.time() == datetime.time.min:
            return f"{self.date.isoformat()}: Free all day"
        else:
            # Provide the time in a simple HH:MM format.
            formatted_time = self.meetings_end_at.strftime('%H:%M')
            return f"{self.date.isoformat()}: Available after {formatted_time}"


@dataclass
class DailySurfForecast:
    date: datetime.date
    spot: str
    wave_height_m: float
    wave_period_s: float
    wind_speed_kmh: float

    def __str__(self) -> str:
        return (f"Date: {self.date.isoformat()}, Spot: {self.spot}, "
                f"Waves: {self.wave_height_m}m, Period: {self.wave_period_s}s, "
                f"Wind: {self.wind_speed_kmh}km/h")


@dataclass
class TrainTicket:
    date: datetime.date
    origin: str
    destination: str
    departure_time: datetime.datetime
    arrival_time: datetime.datetime
    duration: str
    price_eur: float

    def __str__(self) -> str:
        dep_time = self.departure_time.strftime('%H:%M')
        arr_time = self.arrival_time.strftime('%H:%M')
        return (f"Train from {self.origin} to {self.destination} on {self.date.isoformat()}: "
                f"Departs {dep_time}, Arrives {arr_time}, "
                f"Duration: {self.duration}, Price: {self.price_eur:.2f} EUR")
