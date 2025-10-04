import operator
from typing import TypedDict, Annotated, Optional
from datetime import date


class TripDetails(TypedDict, total=False):
    """Holds the specifications for the surf trip plan."""
    departure_city: str
    destination_city: str
    departure_date: date
    return_date: date
    desired_surf_conditions: str

class ToolData(TypedDict):
    """Holds the accumulated raw data from tool calls."""
    calendar_availabilities: Annotated[list, operator.add] # Renamed for clarity
    surf_forecasts: Annotated[list, operator.add]
    train_options: Annotated[list, operator.add]

class AgentState(TypedDict):
    """The overall state of the agent."""
    messages: Annotated[list, operator.add]
    trip_details: TripDetails
    tool_data: ToolData
    current_intent: Optional[str]
    error: Optional[str]
