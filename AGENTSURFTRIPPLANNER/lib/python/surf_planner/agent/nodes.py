import json
import logging
from datetime import date

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .helpers import get_weekend_dates, parse_llm_json_response, split_chat_and_scratchpad
from .prompts import (
    inform_user_prompt_template,
    plan_travel_logistics_prompt_template,
    request_missing_details_prompt_template,
    route_intent_prompt_template,
    summarize_plan_prompt_template,
    update_details_prompt_template,
)
from .state import AgentState, get_tool_data_for_prompt


TOOL_TO_STATE_MAP = {
    "check_calendar": "calendar_availabilities",
    "find_train_tickets": "train_options",
    "get_surf_forecast": "surf_forecasts",
}


def node_route_intent(state: AgentState, model):
    logging.debug("ENTERING NODE: route_intent")

    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    trip_details_str = json.dumps(state.get("trip_details", {}), default=str)

    classifier_chain = route_intent_prompt_template | model
    response = classifier_chain.invoke(
        {"trip_details": trip_details_str, "conversation_history": conversation_history}
    )

    try:
        intent_json = parse_llm_json_response(response.content)
        intent = intent_json["intent"]
        logging.debug("LLM classified intent as: %s", intent)
    except Exception as e:
        logging.error("Error parsing intent, defaulting to 'error'. Details: %s", e)
        intent = "error"

    return {"current_intent": intent}


def node_update_trip_details(state: AgentState, model):
    logging.debug("ENTERING NODE: update_trip_details")

    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    current_details_str = json.dumps(state.get("trip_details", {}), default=str)
    current_date_str = date.today().isoformat()

    chain = update_details_prompt_template | model
    response = chain.invoke(
        {
            "current_date": current_date_str,
            "trip_details": current_details_str,
            "conversation_history": conversation_history,
        }
    )

    try:
        parsed_info = parse_llm_json_response(response.content)
        logging.debug("Parsed trip details update: %s", parsed_info)
        new_trip_details = state.get("trip_details", {}).copy()
        new_trip_details.update(parsed_info)

        # Date parsing logic
        if "departure_date" in new_trip_details and isinstance(new_trip_details["departure_date"], str):
            new_trip_details["departure_date"] = date.fromisoformat(new_trip_details["departure_date"])
        if "return_date" in new_trip_details and isinstance(new_trip_details["return_date"], str):
            new_trip_details["return_date"] = date.fromisoformat(new_trip_details["return_date"])

        return {"trip_details": new_trip_details}

    except (json.JSONDecodeError, TypeError) as e:
        logging.error("Error parsing details, state remains unchanged. Details: %s", e)

        return {}


def node_chat_with_user(state: AgentState, model):
    logging.debug("ENTERING NODE: chat_with_user")

    messages = state["messages"]
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful and enthusiastic surf trip assistant. "
                "Answer the user's questions or engage in conversation politely. "
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = chat_prompt_template | model
    response = chain.invoke({"input": messages[-1].content, "chat_history": messages[:-1]})
    llm_response = AIMessage(content=response.content)

    return {"messages": [llm_response]}


def node_request_missing_details(state: AgentState, model):
    logging.debug("ENTERING NODE: request_missing_details")

    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    trip_details_str = json.dumps(state.get("trip_details", {}), default=str)

    chain = request_missing_details_prompt_template | model
    response = chain.invoke({"trip_details": trip_details_str, "conversation_history": conversation_history})
    question_message = AIMessage(content=response.content)

    return {"messages": [question_message]}


def node_check_surf_forecast(state: AgentState, tool_map: dict):
    logging.debug("ENTERING NODE: check_surf_forecast")

    trip_details = state.get("trip_details", {})
    departure_date = trip_details.get("departure_date")
    destination = trip_details.get("destination_city")

    if not departure_date or not destination:
        error_msg = "Missing departure date or destination for forecast."
        logging.warning(error_msg)
        return {"error": error_msg}

    saturday, sunday = get_weekend_dates(departure_date)
    logging.debug("Checking forecast from %s to %s in %s", saturday, sunday, destination)

    # Use injected 'tool_map' instead of 'self.tool_map'
    surf_tool = tool_map.get("get_surf_forecast")
    if not surf_tool:
        error_msg = "Tool 'get_surf_forecast' not found."
        logging.error(error_msg)
        return {"error": error_msg}

    forecast_result = surf_tool.invoke(
        {"spot": destination, "from_date": saturday.isoformat(), "to_date": sunday.isoformat()}
    )

    if isinstance(forecast_result, str) and "error" in forecast_result.lower():
        logging.error("Tool 'get_surf_forecast' returned an error: %s", forecast_result)
        return {"error": forecast_result}

    logging.debug(f"Successfully retrieved surf forecast. {forecast_result}")
    return {"surf_forecasts": forecast_result}


def node_inform_user_of_bad_surf(state: AgentState, model):
    logging.debug("ENTERING NODE: inform_user_of_bad_surf")

    forecast_data = state.get("surf_forecasts", [])
    desired_conditions = state.get("trip_details", {}).get("desired_surf_conditions", "any")

    chain = inform_user_prompt_template | model
    response = chain.invoke({"desired_surf_conditions": desired_conditions, "forecast_data": str(forecast_data)})
    message = AIMessage(content=response.content)

    return {"messages": [message]}


def node_plan_travel_logistics(state: AgentState, model):
    logging.debug("ENTERING NODE: plan_travel_logistics (The Brain)")

    chat_history, scratchpad_messages = split_chat_and_scratchpad(state["messages"])
    chain = plan_travel_logistics_prompt_template | model
    inputs = {
        "input": chat_history[-1].content,
        "chat_history": chat_history[:-1],
        "trip_details": state.get("trip_details", {}),
        "tool_data": get_tool_data_for_prompt(state),
        "agent_scratchpad": scratchpad_messages,
    }
    response_message = chain.invoke(inputs)

    return {"messages": [response_message]}


def node_execute_tools(state: AgentState, tool_map: dict):
    logging.debug("ENTERING NODE: execute_tools (The Hands)")
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    state_updates = {}
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_to_call = tool_map[tool_name]

        # --- 1. Execution & Strict Error Handling ---
        try:
            output = tool_to_call.invoke(tool_call["args"])
        except Exception as e:
            # Critical Error: Stop immediately and route to handle_error
            return {"error": f"Critical failure in tool '{tool_name}': {e}"}

        if isinstance(output, str) and "error" in output.lower():
            # Critical Error: Stop immediately
            return {"error": output}

        # --- 2. Update Chat History (Scratchpad) ---
        tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
        target_state_key = TOOL_TO_STATE_MAP.get(tool_name)

        if target_state_key:
            state_updates[target_state_key] = output
            logging.debug(f"Updated state key '{target_state_key}' with output from {tool_name}")

    # Add the tool messages to the update
    state_updates["messages"] = tool_messages

    return state_updates


def node_handle_error(state: AgentState):
    logging.debug("ENTERING NODE: handle_error")

    error = state.get("error", "An unknown error occurred.")
    error_message = f"I'm sorry, I encountered an error and cannot continue. Details: {error}"

    return {"messages": [AIMessage(content=error_message)]}


def node_summarize_plan(state: AgentState, model):
    logging.debug("ENTERING NODE: summarize_plan")

    trip_details_str = json.dumps(state.get("trip_details", {}), default=str)
    tool_data_str = json.dumps(get_tool_data_for_prompt(state), default=str)

    chain = summarize_plan_prompt_template | model
    response = chain.invoke({"trip_details": trip_details_str, "tool_data": tool_data_str})
    summary_message = AIMessage(content=response.content)

    return {"messages": [summary_message]}
