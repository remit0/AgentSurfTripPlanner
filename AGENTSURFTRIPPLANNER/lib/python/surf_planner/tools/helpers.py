import json
from datetime import date, timedelta


def get_weekend_dates(departure_date: date) -> tuple[date, date]:
    """
    Calculates the Saturday and Sunday for the weekend following or
    including the given departure date.

    Returns:
        A tuple containing the Saturday and Sunday as date objects.
    """
    day_of_week = departure_date.weekday() # Monday is 0, Sunday is 6

    if day_of_week < 5: # If departure is Monday-Friday
        saturday = departure_date + timedelta(days=(5 - day_of_week))
    else: # If departure is Saturday or Sunday
        saturday = departure_date if day_of_week == 5 else departure_date - timedelta(days=1)

    sunday = saturday + timedelta(days=1)
    return saturday, sunday


def parse_llm_json_response(raw_string: str) -> dict | None:
    """
    Safely extracts a JSON object from a string that might be wrapped in Markdown.
    """
    try:
        # Find the first '{' and the last '}'
        start_index = raw_string.find('{')
        end_index = raw_string.rfind('}') + 1

        if start_index == -1 or end_index == 0:
            return None

        json_string = raw_string[start_index:end_index]
        return json.loads(json_string)

    except (json.JSONDecodeError, IndexError):
        return None
