import datetime
import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    # Fields for accumulating tool results
    messages: Annotated[list, operator.add]

    availabilities: Annotated[list, operator.add]
    surf_forecasts: Annotated[list, operator.add]
    train_options: Annotated[list, operator.add]

    # These will be populated by the gather_info_node
    departure_city: str | None
    destination_city: str | None
    departure_date: datetime.date | None
    return_date: datetime.date | None
    spot: str | None
    surf_conditions: str | None

    # Field to control the graph's flow
    next_step: str | None

    # Error handling
    error: str | None
