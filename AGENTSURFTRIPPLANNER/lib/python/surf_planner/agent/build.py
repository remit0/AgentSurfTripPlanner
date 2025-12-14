
from dataiku.langchain.dku_llm import DKUChatLLM

from surf_planner.agent.graph import AgentGraph
from surf_planner.agent.tools.calendar import create_calendar_tool, create_dummy_calendar_tool
from surf_planner.agent.tools.surf_forecast import create_surf_forecast_tool
from surf_planner.agent.tools.train import create_train_ticket_tool, create_dummy_train_ticket_tool
from surf_planner.apis.geolocator import GeolocatorAPIClient
from surf_planner.apis.google_calendar import GoogleCalendarAPIClient
from surf_planner.apis.navitia import NavitiaAPIClient
from surf_planner.apis.openmeteo import OpenMeteoAPIClient
from surf_planner.config import ProjectSettings


class AgentBuilder:

    def __init__(self, dku_client, settings: ProjectSettings):
        self._dku_client = dku_client
        self._settings = settings

    def _build_model(self):
        model = DKUChatLLM(llm_id=self._settings.llm_id)
        return model

    def _build_calendar_tool(self):
        gcp_creds = self._settings.gcp_service_account
        calendar_id = self._settings.calendar_id
        if gcp_creds and calendar_id:
            gcp_client = GoogleCalendarAPIClient(gcp_creds.model_dump())
            return create_calendar_tool(gcp_client, calendar_id)
        else:
            return create_dummy_calendar_tool()

    def _build_surf_forecast_tool(self):
        geolocator_client = GeolocatorAPIClient()
        openmeteo_client = OpenMeteoAPIClient()
        return create_surf_forecast_tool(geolocator_client, openmeteo_client)

    def _build_train_tickets_tool(self):
        """Builds and returns the FindTrainTicketsTool."""
        navitia_api_key = self._settings.NAVITIA_API_KEY
        if navitia_api_key:
            navitia_client = NavitiaAPIClient(self._settings.NAVITIA_API_KEY)
            return create_train_ticket_tool(navitia_client)
        else:
            return create_dummy_train_ticket_tool()

    def build(self):
        """
        Assembles and returns the compiled LangGraph agent.
        """
        model = self._build_model()
        tools = [
            self._build_calendar_tool(),
            self._build_surf_forecast_tool(),
            self._build_train_tickets_tool(),
        ]
        graph = AgentGraph(model, tools)
        agent = graph.build()
        return agent
