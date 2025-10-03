import datetime
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .prompts import gather_info_prompt, plan_and_execute_prompt, synthesis_prompt
from .state import AgentState


class AgentGraph:
    """Encapsulates the logic for the surf trip agent's graph."""

    def __init__(self, model, tools: list):
        self.plain_model = model
        self.model_with_tools = model.bind_tools(tools)
        self.tool_map = {tool.name: tool for tool in tools}

    def _extract_and_parse_json(self, raw_string: str) -> dict | None:
        """
        Safely extracts a JSON object from a string that might be wrapped in Markdown.

        Args:
            raw_string: The raw text response from the LLM.

        Returns:
            A dictionary if a valid JSON object is found, otherwise None.
        """
        try:
            # Find the first '{' and the last '}'
            start_index = raw_string.find('{')
            end_index = raw_string.rfind('}') + 1

            # Return None if JSON object markers aren't found
            if start_index == -1 or end_index == 0:
                return None

            # Slice the string to get the pure JSON part
            json_string = raw_string[start_index:end_index]

            # Parse the clean string
            return json.loads(json_string)

        except (json.JSONDecodeError, IndexError):
            # Handle cases where the slice is not valid JSON
            return None

    def _format_state_for_prompt(self, state: AgentState) -> str:
        """
        Converts the entire structured state into a clean "fact sheet" for the LLM.
        --- USER REQUEST ---
        Origin: Paris
        Destination: Biarritz
        Departure Date: 2025-09-05
        Return Date: 2025-09-07
        Surf Spot: Biarritz
        Desired Conditions: waves around 1.5 meters

        --- SURF FORECAST DATA ---
        * Date: 2025-09-06, Spot: Biarritz, Waves: 1.6m, Period: 9s, Wind: 12km/h
        * Date: 2025-09-07, Spot: Biarritz, Waves: 1.4m, Period: 9s, Wind: 15km/h

        --- AGENT STATUS ---
        Next Step: Check the calendar and find train tickets.
        """
        parts = []

        # --- Part 1: Summarize the core request parameters ---
        request_summary_parts = []
        if state.get("departure_city"):
            request_summary_parts.append(f"Departure: {state['departure_city']}")
        if state.get("destination_city"):
            request_summary_parts.append(f"Destination: {state['destination_city']}")
        if state.get("departure_date"):
            request_summary_parts.append(f"Departure Date: {state['departure_date'].isoformat()}")
        if state.get("return_date"):
            request_summary_parts.append(f"Return Date: {state['return_date'].isoformat()}")
        if state.get("spot"):
            request_summary_parts.append(f"Surf Spot: {state['spot']}")
        if state.get("surf_conditions"):
            request_summary_parts.append(f"Desired Conditions: {state['surf_conditions']}")

        if request_summary_parts:
            parts.append("--- USER REQUEST ---\n" + "\n".join(request_summary_parts))

        # --- Part 2: Summarize the data gathered by tools ---
        if state.get("availabilities"):
            avail_summary = "\n* ".join([str(a) for a in state["availabilities"]])
            parts.append(f"--- CALENDAR AVAILABILITY ---\n* {avail_summary}")

        if state.get("surf_forecasts"):
            # --- MODIFICATION: Neutrally reporting the forecast data ---
            forecast_summary = "\n* ".join([str(f) for f in state["surf_forecasts"]])
            parts.append(f"--- SURF FORECAST DATA ---\n* {forecast_summary}")

        if state.get("train_options"):
            train_summary = "\n* ".join([str(t) for t in state["train_options"]])
            parts.append(f"--- TRAIN TICKETS ---\n* {train_summary}")

        # --- Part 3: Add a clear status to guide the agent ---
        status_parts = []
        if not state.get("surf_forecasts"):
            status_parts.append("Next Step: Check the surf forecast.")
        elif not state.get("availabilities") or not state.get("train_options"):
            status_parts.append("Next Step: Check the calendar and find train tickets.")
        else:
            status_parts.append("Status: All information has been collected.")

        parts.append("--- AGENT STATUS ---\n" + "\n".join(status_parts))

        return "\n\n".join(parts)

    def gather_info_node(self, state: AgentState):
        """
        Node 1: Gathers the initial key pieces of information from the user query.
        Decides if it needs to ask for clarification or can proceed.
        """
        user_message = state["messages"][0].content
        current_date_str = datetime.date.today().isoformat()
        prompt = gather_info_prompt.format(user_message=user_message, current_date=current_date_str)
        response = self.plain_model.invoke(prompt)
        updates = {}

        try:
            extracted_info = self._extract_and_parse_json(response.content)

            # Update state with extracted info
            updates['departure_city'] = extracted_info.get("departure_city")
            updates['destination_city'] = extracted_info.get("destination_city")
            updates['spot'] = extracted_info.get("surf_spot")
            updates['surf_conditions'] = extracted_info.get("desired_surf_conditions")

            # Defaulting surf spot to destination.
            if not updates.get("spot") and updates.get("destination_city"):
                updates['spot'] = updates['destination_city']

            if extracted_info.get("departure_date"):
                updates['departure_date'] = datetime.date.fromisoformat(extracted_info["departure_date"])

            if extracted_info.get("return_date"):
                updates['return_date'] = datetime.date.fromisoformat(extracted_info["return_date"])

            # Decide the next step
            if updates.get("departure_city") and updates.get("destination_city"):
                updates['next_step'] = "plan_and_execute"
            else:
                updates['next_step'] = "ask_for_clarification"

        except (json.JSONDecodeError, TypeError):
            updates['next_step'] = "ask_for_clarification"

        return updates

    def plan_and_execute_node(self, state: AgentState):
        """
        The main worker node. Looks at the current state and decides which tool to call next.
        """
        structured_state_summary = self._format_state_for_prompt(state)

        # Keep only the Human and AI messages for the conversational history
        filtered_messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                filtered_messages.append(msg)
            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                filtered_messages.append(msg)

        prompt_input = {
            "messages": filtered_messages,
            "structured_state_summary": structured_state_summary
        }

        chain = plan_and_execute_prompt | self.model_with_tools
        response = chain.invoke(prompt_input)

        return {"messages": [response]}

    def tool_node(self, state: AgentState):
        """The custom tool node for executing tools and updating the state."""
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls

        state_updates = {}
        tool_messages = []

        for call in tool_calls:
            if call['name'] in self.tool_map:
                tool_to_call = self.tool_map[call['name']]
                result = tool_to_call.invoke(call['args'])
            else:
                result = f"Error: Tool '{call['name']}' not found."

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call['id']))

            if isinstance(result, str):
                # Populate the 'error' field in the state to trigger the error handler.
                state_updates['error'] = result
            elif isinstance(result, list):
                # Only update the structured state if the result is the correct type (a list).
                if call['name'] == 'check_calendar':
                    state_updates['availabilities'] = result
                elif call['name'] == 'get_surf_forecast':
                    state_updates['surf_forecasts'] = result
                elif call['name'] == 'find_train_tickets':
                    state_updates['train_options'] = result

        state_updates['messages'] = tool_messages
        return state_updates

    def synthesis_node(self, state: AgentState):
        """
        Generates the final user-facing response, either asking for clarification
        or presenting the final trip options.
        """
        structured_state_summary = self._format_state_for_prompt(state)
        prompt = synthesis_prompt.format(structured_state_summary=structured_state_summary)
        # Use a model without tools to ensure it only generates text
        response = self.plain_model.invoke(prompt)
        return {"messages": [response]}

    def error_node(self, state: AgentState):
        """This node is called when a tool fails, to formulate a final error message."""
        error = state.get("error")
        error_message = f"I'm sorry, I encountered an error and cannot continue. Details: {error}"
        return {"messages": [AIMessage(content=error_message)]}

    def decide_after_gathering(self, state: AgentState) -> str:
        """
        Edge 1: Decides where to go after the info gathering step.
        """
        if state.get("departure_city") and state.get("destination_city"):
            return "execute_plan"
        else:
            # The synthesis node will be prompted to ask for clarification
            return "ask_for_clarification"

    def should_continue_planning(self, state: AgentState) -> str:
        """
        Edge 2: Decides if the agent needs to call more tools or can synthesize a final answer.
        """
        if state["messages"][-1].tool_calls:
            return "use_tool"
        else:
            # The planner has finished, proceed to generate the final answer
            return "generate_final_answer"

    def decide_after_tools(self, state: AgentState) -> str:
        """Decides whether to continue planning or handle an error after a tool run."""
        if state.get("error"):
            return "handle_error"
        else:
            return "plan_and_execute"

    def create_graph(self):
        """Builds and compiles the new state machine graph."""
        workflow = StateGraph(AgentState)

        # Add all the nodes
        workflow.add_node("gather_info", self.gather_info_node)
        workflow.add_node("plan_and_execute", self.plan_and_execute_node)
        workflow.add_node("synthesis", self.synthesis_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("error_handler", self.error_node)

        # --- Define the graph's control flow ---

        # 1. Start with the information gathering step
        workflow.set_entry_point("gather_info")

        # 2. Edge from gathering to either planning or asking for clarification
        workflow.add_conditional_edges(
            "gather_info",
            self.decide_after_gathering,
            {
                "execute_plan": "plan_and_execute",
                "ask_for_clarification": "synthesis"  # The synthesis node will generate the question
            }
        )

        # 3. The main tool-use loop
        workflow.add_conditional_edges(
            "plan_and_execute",
            self.should_continue_planning,
            {
                "use_tool": "tools",
                "generate_final_answer": "synthesis"  # If done, generate the final answer
            }
        )
        workflow.add_conditional_edges(
            "tools",
            self.decide_after_tools,
            {
                "plan_and_execute": "plan_and_execute",  # If success, continue the loop
                "handle_error": "error_handler"  # If failure, go to error node
            }
        )

        # 4. The final step
        workflow.add_edge("synthesis", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()