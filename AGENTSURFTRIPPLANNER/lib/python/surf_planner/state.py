import operator
from datetime import date
from typing import Annotated, TypedDict, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class TripDetails(TypedDict, total=False):
    """Holds the specifications for the surf trip plan."""
    departure_city: str
    destination_city: str
    departure_date: date
    return_date: date
    desired_surf_conditions: str

class AgentState(TypedDict):
    """The overall state of the agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    trip_details: TripDetails
    current_intent: Optional[str]
    error: Optional[str]

    # --- Flattened Tool Data Fields ---
    calendar_availabilities: Annotated[list, operator.add]
    surf_forecasts: Annotated[list, operator.add]
    train_options: Annotated[list, operator.add]


def get_tool_data_for_prompt(state: dict) -> dict:
    """
    Gathers all tool result lists from the flat state and assembles them
    into a single dictionary for a prompt.
    """
    return {
        "surf_forecasts": state.get("surf_forecasts", []),
        "train_options": state.get("train_options", []),
        "calendar_availabilities": state.get("calendar_availabilities", [])
    }
