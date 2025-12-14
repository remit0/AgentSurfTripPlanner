import logging
from functools import partial

from langgraph.graph import END, StateGraph

from .edges import (
    edge_after_forecast,
    edge_after_tools,
    edge_after_update,
    edge_from_intent,
    edge_from_plan,
)
from .nodes import (
    node_chat_with_user,
    node_check_surf_forecast,
    node_execute_tools,
    node_handle_error,
    node_inform_user_of_bad_surf,
    node_plan_travel_logistics,
    node_request_missing_details,
    node_route_intent,
    node_summarize_plan,
    node_update_trip_details,
)
from .state import AgentState


class AgentGraph:
    """Encapsulates the logic for the surf trip agent's graph."""

    def __init__(self, model, tools: list):
        self.plain_model = model
        self.model_with_tools = model.bind_tools(tools)
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        logging.debug("AgentGraph initialized with %d tools.", len(tools))

    def build(self):
        """Builds and compiles the final state machine graph."""
        workflow = StateGraph(AgentState)

        # --- 1. ADD NODES (Injecting Dependencies via partial) ---

        # Nodes using the Plain Model (Reasoning/Chat)
        workflow.add_node("route_intent", partial(node_route_intent, model=self.plain_model))
        workflow.add_node("update_trip_details", partial(node_update_trip_details, model=self.plain_model))
        workflow.add_node("chat_with_user", partial(node_chat_with_user, model=self.plain_model))
        workflow.add_node("request_missing_details", partial(node_request_missing_details, model=self.plain_model))
        workflow.add_node("inform_user_of_bad_surf", partial(node_inform_user_of_bad_surf, model=self.plain_model))
        workflow.add_node("summarize_plan", partial(node_summarize_plan, model=self.plain_model))

        # Nodes using Model WITH Tools (The Planner)
        workflow.add_node("plan_travel_logistics", partial(node_plan_travel_logistics, model=self.model_with_tools))

        # Nodes using Tool Map (Execution)
        workflow.add_node("check_surf_forecast", partial(node_check_surf_forecast, tool_map=self.tool_map))
        workflow.add_node("execute_tools", partial(node_execute_tools, tool_map=self.tool_map))

        # Nodes with No External Dependencies
        workflow.add_node("handle_error", node_handle_error)


        # --- 2. DEFINE EDGES & ROUTING ---

        workflow.set_entry_point("route_intent")

        # Main Intent Router
        workflow.add_conditional_edges(
            "route_intent",
            edge_from_intent,  # Pure state check, no partial needed
            path_map={
                "update_details": "update_trip_details",
                "chat": "chat_with_user",
                "error": "handle_error"
            },
        )

        # After Updating Details
        workflow.add_conditional_edges(
            "update_trip_details",
            edge_after_update,
            {
                "details_complete": "check_surf_forecast",
                "details_incomplete": "request_missing_details"
            },
        )

        # Forecast Check (Needs Model for Validation)
        workflow.add_conditional_edges(
            "check_surf_forecast",
            partial(edge_after_forecast, model=self.plain_model), # Inject model here!
            path_map={
                "forecast_is_good": "plan_travel_logistics",
                "forecast_is_bad": "inform_user_of_bad_surf",
            },
        )

        # Planning Loop Router
        workflow.add_conditional_edges(
            "plan_travel_logistics",
            edge_from_plan,
            path_map={
                "continue_planning": "execute_tools",
                "plan_complete": "summarize_plan",
            },
        )

        # Tool Execution Router
        workflow.add_conditional_edges(
            "execute_tools",
            edge_after_tools,
            path_map={
                "continue": "plan_travel_logistics",
                "error": "handle_error",
            },
        )

        # --- 3. TERMINAL EDGES ---
        workflow.add_edge("chat_with_user", END)
        workflow.add_edge("request_missing_details", END)
        workflow.add_edge("handle_error", END)
        workflow.add_edge("summarize_plan", END)
        workflow.add_edge("inform_user_of_bad_surf", END)

        return workflow.compile()
