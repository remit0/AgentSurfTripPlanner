import logging

from .prompts import validate_surf_forecast_prompt_template
from .state import AgentState


def edge_from_intent(state: AgentState) -> str:
    """
    Reads the 'current_intent' from the state for the main router.
    """
    intent = state.get("current_intent")
    logging.debug("ROUTING from 'route_intent' with intent: %s", intent)

    if not intent:
        logging.error("No intent found in state, routing to 'error'.")
        return "error"

    return intent


def edge_after_update(state: AgentState) -> str:
    """
    Checks if all mandatory details are present after an update.
    """
    logging.debug("ROUTING from 'update_trip_details' to check for completion.")
    details = state.get("trip_details", {})
    mandatory_keys = ["departure_city", "destination_city", "departure_date"]

    if all(details.get(key) for key in mandatory_keys):
        logging.debug("Validation complete: All mandatory details present. Routing to 'details_complete'.")
        return "details_complete"
    else:
        logging.debug("Validation incomplete: Details missing. Routing to 'details_incomplete'.")
        return "details_incomplete"


def edge_after_forecast(state: AgentState, model) -> str:
    """
    Analyzes the forecast.
    NOTE: You must inject 'model' using partial() when building the graph.
    """
    logging.debug("ROUTING from 'check_surf_forecast'")
    forecast_data = state.get("surf_forecasts", [])
    desired_conditions = state.get("trip_details", {}).get("desired_surf_conditions", "any")

    if not forecast_data or state.get("error"):
        logging.warning("No forecast data or error present, routing to 'forecast_is_bad'.")
        return "forecast_is_bad"

    # Use the injected 'model' instead of self.plain_model
    chain = validate_surf_forecast_prompt_template | model
    response = chain.invoke({"desired_conditions": desired_conditions, "forecast_data": str(forecast_data)})

    if "yes" in response.content.lower():
        logging.debug("Validation result: Forecast is good.")
        return "forecast_is_good"
    else:
        logging.debug("Validation result: Forecast is bad.")
        return "forecast_is_bad"


def edge_from_plan(state: AgentState) -> str:
    logging.debug("ROUTING from 'plan_travel_logistics'")
    last_message = state["messages"][-1]

    # Check if the LLM decided to call a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logging.debug("Tool call detected, routing to 'continue_planning'.")
        return "continue_planning"
    else:
        logging.debug("No tool call detected, routing to 'plan_complete'.")
        return "plan_complete"


def edge_after_tools(state: AgentState) -> str:
    logging.debug("ROUTING from 'execute_tools'")
    if state.get("error"):
        logging.debug("Error detected in state, routing to 'error'.")
        return "error"
    else:
        logging.debug("No error detected, routing to 'continue'.")
        return "continue"
