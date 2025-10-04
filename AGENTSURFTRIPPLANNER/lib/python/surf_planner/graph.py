import datetime
import json
from datetime import date
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain.agents import create_tool_calling_agent, AgentExecutor

from .prompts import route_intent_prompt_template, plan_and_execute_prompt, update_details_prompt_template, chat_prompt_template
from .state import AgentState


class AgentGraph:
    """Encapsulates the logic for the surf trip agent's graph."""

    def __init__(self, model, tools: list):
        self.plain_model = model
        self.model_with_tools = model.bind_tools(tools)
        self.tools = tools
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

    def node_route_intent(self, state: AgentState):
        """
        The central router. Classifies the user's intent and updates the state.
        """
        # 1. Prepare the inputs for the prompt
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        trip_details_str = json.dumps(state.get("trip_details", {}), indent=2)

        # 2. Create the chain and invoke the LLM
        classifier_chain = route_intent_prompt_template | self.plain_model
        response = classifier_chain.invoke(
            {"trip_details": trip_details_str, "conversation_history": conversation_history}
        )

        # 3. Safely parse the JSON output and update the state
        try:
            intent_json = json.loads(response.content)
            intent = intent_json["intent"]
            print(f"Intent classified as: {intent}")
        except Exception as e:
            print(f"Error parsing intent, routing to error handler. Error: {e}")
            intent = "error"

        return {"current_intent": intent}

    def node_update_trip_details(self, state: AgentState):
        """
        Calls an LLM to extract and update trip details from the conversation.
        """
        # 1. Prepare the inputs for the prompt
        conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        current_details_str = json.dumps(state.get("trip_details", {}))
        current_date_str = date.today().isoformat()

        # 2. Create the chain and invoke the LLM
        chain = update_details_prompt_template | self.plain_model
        response = chain.invoke(
            {
                "current_date": current_date_str,
                "trip_details": current_details_str,
                "conversation_history": conversation_history,
            }
        )

        # 3. Safely parse the JSON output and update the state
        try:
            parsed_info = json.loads(response.content)
            new_trip_details = state.get("trip_details", {}).copy()
            new_trip_details.update(parsed_info)

            # 4. Convert date strings from LLM to proper datetime.date objects
            if "departure_date" in new_trip_details and isinstance(new_trip_details["departure_date"], str):
                new_trip_details["departure_date"] = date.fromisoformat(new_trip_details["departure_date"])

            if "return_date" in new_trip_details and isinstance(new_trip_details["return_date"], str):
                new_trip_details["return_date"] = date.fromisoformat(new_trip_details["return_date"])

            return {"trip_details": new_trip_details}

        except (json.JSONDecodeError, TypeError) as e:
            # If parsing fails, we don't change the state and can log the error
            print(f"Error parsing details, state remains unchanged. Error: {e}")
            return {}  # Return an empty dict to signify no changes were made

    def node_chat_with_user(self, state: AgentState):
        """
        The general-purpose worker node. It can call tools or answer directly.
        """
        # 1. Create the Agent and AgentExecutor
        # This creates a powerful "sub-agent" that can reason and use tools.
        agent = create_tool_calling_agent(self.model_with_tools, self.tools, chat_prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        # 2. Prepare the inputs for the AgentExecutor
        messages = state["messages"]
        trip_details_str = json.dumps(state.get("trip_details", {}), indent=2)
        inputs = {"input": messages[-1].content, "chat_history": messages[:-1], "trip_details": trip_details_str}

        # 3. Invoke the AgentExecutor
        response = agent_executor.invoke(inputs)

        # 4. Update the state
        final_message = AIMessage(content=response["output"])
        return {"messages": [final_message]}


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
        """Builds and compiles the final state machine graph."""
        workflow = StateGraph(AgentState)

        # 1. Add all nodes with clear, action-oriented names
        workflow.add_node("route_intent", self.node_route_intent)
        workflow.add_node("update_trip_details", self.node_update_trip_details)
        workflow.add_node("chat_with_user", self.node_chat_with_user)

        workflow.add_node("reason_about_plan", self.node_reason_about_plan)
        workflow.add_node("execute_tools", self.node_execute_tools)
        workflow.add_node("summarize_plan", self.node_summarize_plan)
        workflow.add_node("clarify_query", self.node_clarify_query)
        workflow.add_node("handle_error", self.node_handle_error)

        # 2. Set the entry point to the main router
        workflow.set_entry_point("route_intent")

        # 3. Define the main router's conditional logic
        workflow.add_conditional_edges(
            start_node_key="route_intent",
            condition=self.route_from_intent,
            path_map={
                "update_details": "update_trip_details",
                "chat": "chat_with_user",
                "plan_trip": "reason_about_plan",
                "clarify": "clarify_query",
                "error": "handle_error",
                "end_conversation": END,
            },
        )

        # 4. Define the planning and tool execution sub-loop
        workflow.add_conditional_edges(
            start_node_key="reason_about_plan",
            condition=self.route_from_plan,
            path_map={
                "continue_planning": "execute_tools",
                "plan_complete": "summarize_plan",
            },
        )
        workflow.add_edge("execute_tools", "reason_about_plan")

        # 5. Define the loopbacks for simple conversational turns
        workflow.add_edge("update_trip_details", "route_intent")
        workflow.add_edge("chat_with_user", "route_intent")
        workflow.add_edge("clarify_query", "route_intent")
        workflow.add_edge("handle_error", "route_intent")

        # 6. Define the final step of the graph
        workflow.add_edge("summarize_plan", END)

        # 7. Compile and return the graph
        return workflow.compile()
