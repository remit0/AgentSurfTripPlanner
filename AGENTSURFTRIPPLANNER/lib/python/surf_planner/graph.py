import json
import logging
from datetime import date

# Add this to the top of your main script to see the debug messages
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from .prompts import (
    chat_prompt_template,
    clarify_query_prompt_template,
    inform_user_prompt_template,
    plan_travel_logistics_prompt_template,
    request_missing_details_prompt_template,
    route_intent_prompt_template,
    summarize_plan_prompt_template,
    update_details_prompt_template,
    validate_surf_forecast_prompt_template,
)
from .state import AgentState, get_tool_data_for_prompt
from .tools.helpers import get_weekend_dates, parse_llm_json_response




class AgentGraph:
    """Encapsulates the logic for the surf trip agent's graph."""

    def __init__(self, model, tools: list):
        self.plain_model = model
        self.model_with_tools = model.bind_tools(tools)
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        logging.debug("AgentGraph initialized with %d tools.", len(tools))

    # --- NODE IMPLEMENTATIONS ---

    def node_route_intent(self, state: AgentState):
        logging.debug("ENTERING NODE: route_intent")
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        trip_details_str = json.dumps(state.get("trip_details", {}), default=str)

        classifier_chain = route_intent_prompt_template | self.plain_model
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

    def node_update_trip_details(self, state: AgentState):
        logging.debug("ENTERING NODE: update_trip_details")
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        current_details_str = json.dumps(state.get("trip_details", {}), default=str)
        current_date_str = date.today().isoformat()

        chain = update_details_prompt_template | self.plain_model
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

            if "departure_date" in new_trip_details and isinstance(new_trip_details["departure_date"], str):
                new_trip_details["departure_date"] = date.fromisoformat(new_trip_details["departure_date"])
            if "return_date" in new_trip_details and isinstance(new_trip_details["return_date"], str):
                new_trip_details["return_date"] = date.fromisoformat(new_trip_details["return_date"])

            return {"trip_details": new_trip_details}
        except (json.JSONDecodeError, TypeError) as e:
            logging.error("Error parsing details, state remains unchanged. Details: %s", e)
            return {}

    def node_chat_with_user(self, state: AgentState):
        logging.debug("ENTERING NODE: chat_with_user")
        agent = create_tool_calling_agent(self.model_with_tools, self.tools, chat_prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        messages = state["messages"]
        trip_details_str = json.dumps(state.get("trip_details", {}), default=str)
        inputs = {"input": messages[-1].content, "chat_history": messages[:-1], "trip_details": trip_details_str}

        response = agent_executor.invoke(inputs)
        final_message = AIMessage(content=response["output"])
        return {"messages": [final_message]}

    def node_clarify_query(self, state: AgentState):
        logging.debug("ENTERING NODE: clarify_query")
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        chain = clarify_query_prompt_template | self.plain_model
        response = chain.invoke({"conversation_history": conversation_history})
        clarifying_question = AIMessage(content=response.content)
        return {"messages": [clarifying_question]}

    def node_request_missing_details(self, state: AgentState):
        logging.debug("ENTERING NODE: request_missing_details")
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        trip_details_str = json.dumps(state.get("trip_details", {}), default=str)
        chain = request_missing_details_prompt_template | self.plain_model
        response = chain.invoke({"trip_details": trip_details_str, "conversation_history": conversation_history})
        question_message = AIMessage(content=response.content)
        return {"messages": [question_message]}

    def node_check_surf_forecast(self, state: AgentState):
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

        surf_tool = self.tool_map.get("get_surf_forecast")
        if not surf_tool:
            error_msg = "Tool 'get_surf_forecast' not found."
            logging.error(error_msg)
            return {"error": error_msg}

        forecast_result = surf_tool.invoke(
            {"spot": destination, "from_date": saturday.isoformat(), "to_date": sunday.isoformat()}
        )

        if isinstance(forecast_result, str):
            logging.error("Tool 'get_surf_forecast' returned an error: %s", forecast_result)
            return {"error": forecast_result}

        logging.debug(f"Successfully retrieved surf forecast. {forecast_result}")
        return {"surf_forecasts": forecast_result}

    def node_inform_user_of_bad_surf(self, state: AgentState):
        logging.debug("ENTERING NODE: inform_user_of_bad_surf")
        forecast_data = state.get("surf_forecasts", [])
        desired_conditions = state.get("trip_details", {}).get("desired_surf_conditions", "any")
        chain = inform_user_prompt_template | self.plain_model
        response = chain.invoke({"desired_surf_conditions": desired_conditions, "forecast_data": str(forecast_data)})
        message = AIMessage(content=response.content)
        return {"messages": [message]}

    def node_plan_travel_logistics(self, state: AgentState):
        logging.debug("ENTERING NODE: plan_travel_logistics (The Brain)")

        messages = state["messages"]

        # The scratchpad contains the agent's internal tool-related messages.
        # Find the index of the last "real" message (Human or simple AI response)
        last_real_message_idx = -1
        for i, msg in enumerate(reversed(messages)):
            if not (isinstance(msg, AIMessage) and msg.tool_calls) and not isinstance(msg, ToolMessage):
                last_real_message_idx = len(messages) - 1 - i
                break
        # All messages after that index are part of the scratchpad
        chat_history = messages[: last_real_message_idx + 1]
        scratchpad_messages = messages[last_real_message_idx + 1 :]

        chain = plan_travel_logistics_prompt_template | self.model_with_tools
        inputs = {
            "input": chat_history[-1].content,
            "chat_history": chat_history[:-1],
            "trip_details": state.get("trip_details", {}),
            "tool_data": get_tool_data_for_prompt(state),
            "agent_scratchpad": scratchpad_messages,
        }

        response_message = chain.invoke(inputs)
        return {"messages": [response_message]}

    def node_execute_tools(self, state: AgentState):
        """
        Executes one or more tool calls and updates the state with all results.
        """
        logging.debug("ENTERING NODE: execute_tools (The Hands)")
        last_message = state["messages"][-1]

        # Check if there are any tool calls to execute
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logging.debug("No tool calls to execute.")
            return {}

        tool_calls = last_message.tool_calls

        # This dictionary will collect all updates from all tool calls
        state_updates = {}
        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_to_call = self.tool_map[tool_name]
            logging.debug("Calling tool '%s' with args: %s", tool_name, tool_call["args"])

            try:
                output = tool_to_call.invoke(tool_call["args"])
            except Exception as e:
                error_message = f"Tool '{tool_name}' failed with an exception: {e}"
                logging.error(error_message)
                return {"error": error_message}  # Stop immediately on exception

            # Check if the tool itself returned a known error string
            if isinstance(output, str) and "error" in output.lower():
                logging.error("Tool '%s' returned a managed error: %s", tool_name, output)
                return {"error": output}  # Stop immediately on managed error

            # Append the ToolMessage for the agent's scratchpad
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

            # CORRECTED: Add the specific tool output to our updates dictionary instead of returning
            if tool_name == "check_calendar":
                state_updates["calendar_availabilities"] = output
            elif tool_name == "find_train_tickets":  # Corrected name based on previous discussions
                state_updates["train_options"] = output
            elif tool_name == "get_surf_forecast":
                state_updates["surf_forecasts"] = output

        state_updates["messages"] = tool_messages
        return state_updates

    def node_handle_error(self, state: AgentState):
        logging.debug("ENTERING NODE: handle_error")
        error = state.get("error", "An unknown error occurred.")
        error_message = f"I'm sorry, I encountered an error and cannot continue. Details: {error}"
        return {"messages": [AIMessage(content=error_message)]}

    def node_summarize_plan(self, state: AgentState):
        logging.debug("ENTERING NODE: summarize_plan")
        trip_details_str = json.dumps(state.get("trip_details", {}), default=str)
        tool_data_str = json.dumps(get_tool_data_for_prompt(state), default=str)
        logging.debug("Assembled tool_data for summary: %s", tool_data_str)
        chain = summarize_plan_prompt_template | self.plain_model
        response = chain.invoke({"trip_details": trip_details_str, "tool_data": tool_data_str})
        summary_message = AIMessage(content=response.content)
        return {"messages": [summary_message]}

    # --- ROUTING FUNCTION IMPLEMENTATIONS ---
    def route_from_intent(self, state: AgentState) -> str:
        """
        Reads the 'current_intent' from the state for the main router.
        """
        intent = state.get("current_intent")
        logging.debug("ROUTING from 'route_intent' with intent: %s", intent)

        if not intent:
            logging.error("No intent found in state, routing to 'error'.")
            return "error"

        return intent

    def route_after_update(self, state: AgentState) -> str:
        """
        Checks if all mandatory details are present after an update.
        """
        logging.debug("ROUTING from 'update_trip_details' to check for completion.")
        details = state.get("trip_details", {})
        mandatory_keys = ["departure_city", "destination_city", "departure_date"]

        if all(details.get(key) for key in mandatory_keys):
            logging.debug("Validation complete: All mandatory details are present. Routing to 'details_complete'.")
            return "details_complete"
        else:
            logging.debug("Validation incomplete: Mandatory details are missing. Routing to 'details_incomplete'.")
            return "details_incomplete"

    def route_after_forecast(self, state: AgentState) -> str:
        logging.debug("ROUTING from 'check_surf_forecast'")
        forecast_data = state.get("surf_forecasts", [])
        desired_conditions = state.get("trip_details", {}).get("desired_surf_conditions", "any")

        if not forecast_data or state.get("error"):
            logging.warning("No forecast data or error present, routing to 'forecast_is_bad'.")
            return "forecast_is_bad"

        chain = validate_surf_forecast_prompt_template | self.plain_model
        response = chain.invoke({"desired_conditions": desired_conditions, "forecast_data": str(forecast_data)})

        if "yes" in response.content.lower():
            logging.debug("Validation result: Forecast is good.")
            return "forecast_is_good"
        else:
            logging.debug("Validation result: Forecast is bad.")
            return "forecast_is_bad"

    def route_from_plan(self, state: AgentState) -> str:
        logging.debug("ROUTING from 'plan_travel_logistics'")
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logging.debug("Tool call detected, routing to 'continue_planning'.")
            return "continue_planning"
        else:
            logging.debug("No tool call detected, routing to 'plan_complete'.")
            return "plan_complete"

    def route_after_tools(self, state: AgentState) -> str:
        logging.debug("ROUTING from 'execute_tools'")
        if state.get("error"):
            logging.debug("Error detected in state, routing to 'error'.")
            return "error"
        else:
            logging.debug("No error detected, routing to 'continue'.")
            return "continue"

    def create_graph(self):
        """Builds and compiles the final state machine graph."""
        workflow = StateGraph(AgentState)

        # 1. Add all nodes required for the design
        workflow.add_node("route_intent", self.node_route_intent)
        workflow.add_node("update_trip_details", self.node_update_trip_details)
        workflow.add_node("chat_with_user", self.node_chat_with_user)
        workflow.add_node("clarify_query", self.node_clarify_query)
        workflow.add_node("request_missing_details", self.node_request_missing_details)
        # --- Nodes for the structured planning flow ---
        workflow.add_node("check_surf_forecast", self.node_check_surf_forecast)
        workflow.add_node("inform_user_of_bad_surf", self.node_inform_user_of_bad_surf)
        workflow.add_node("plan_travel_logistics", self.node_plan_travel_logistics)
        workflow.add_node("execute_tools", self.node_execute_tools)
        workflow.add_node("summarize_plan", self.node_summarize_plan)
        # --- Helper nodes
        workflow.add_node("handle_error", self.node_handle_error)

        # 2. Set the entry point to the main router
        workflow.set_entry_point("route_intent")

        # 3. Define the main router's logic
        workflow.add_conditional_edges(
            "route_intent",
            self.route_from_intent,
            path_map={
                "update_details": "update_trip_details",
                "chat": "chat_with_user",
                # "plan_trip": "update_trip_details",
                "clarify_query": "clarify_query",
                #"request_missing_details": "request_missing_details",
                "error": "handle_error"
            },
        )

        workflow.add_conditional_edges(
            "update_trip_details",
            self.route_after_update,
            {
                "details_complete": "check_surf_forecast",
                "details_incomplete": "request_missing_details"
            },
        )

        # 4. Define the "fail-fast" surf check logic
        workflow.add_conditional_edges(
            "check_surf_forecast",
            self.route_after_forecast,
            path_map={
                "forecast_is_good": "plan_travel_logistics", # If good, proceed to the planning loop
                "forecast_is_bad": "inform_user_of_bad_surf",
            },
        )

        # 5. Define the explicit planning and tool execution sub-loop
        workflow.add_conditional_edges(
            "plan_travel_logistics",
            self.route_from_plan,
            path_map={
                "continue_planning": "execute_tools",
                "plan_complete": "summarize_plan", # If done, go to the final summary
            },
        )
        workflow.add_conditional_edges(
            "execute_tools",
            self.route_after_tools,
            path_map={
                "continue": "plan_travel_logistics",  # If success, loop back to the brain
                "error": "handle_error",  # If failure, go to the error handler
            },
        )

        # 6. Define the loopbacks for simple conversational turns
        workflow.add_edge("chat_with_user", END)
        workflow.add_edge("request_missing_details", END)
        workflow.add_edge("clarify_query", END)
        workflow.add_edge("handle_error", END)

        # 7. Define the final paths to the end of the graph
        workflow.add_edge("summarize_plan", END)
        workflow.add_edge("inform_user_of_bad_surf", END)

        # 8. Compile and return the graph
        return workflow.compile()
