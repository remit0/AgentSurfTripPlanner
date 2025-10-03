import requests
from geopy.geocoders import Nominatim
from googleapiclient.discovery import build

from .auth import AuthManager
from .graph import AgentGraph
from .tools.calendar_tools import GetCalendarEventsTool
from .tools.surf_forecast_tools import GetSurfForecastTool
from .tools.train_tools import FindTrainTicketsTool


class AgentBuilder:
    """
    Builds a compiled LangGraph agent by configuring and assembling
    the necessary tools, models, and authenticators.
    """
    # Constants can be defined at the class level
    GCP_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
    GEOLOCATOR_USER_AGENT = "my-surf-forecast-app-v1"

    def __init__(self, client, model, calendar_id: str):
        """
        Initializes the builder with essential configuration and components.

        Args:
            client: An authenticated Dataiku API client instance.
            model: An instantiated language model object (e.g., DKUChatLLM).
            calendar_id (str): The Google Calendar ID to check.
        """
        if not client or not model:
            raise ValueError("A Dataiku client and a model instance are required.")

        self.client = client
        self.model = model
        self.calendar_id = calendar_id

        self.auth_manager = AuthManager(self.client)

    def _setup_calendar_tool(self):
        """Builds and returns the CheckCalendarTool."""
        gcp_creds = self.auth_manager.get_gcloud_credentials(scopes=self.GCP_SCOPES)
        service = build('calendar', 'v3', credentials=gcp_creds)
        return GetCalendarEventsTool(service=service, calendar_id=self.calendar_id)

    def _setup_surf_forecast_tool(self):
        """Builds and returns the GetSurfForecastTool."""
        session = requests.Session()
        geolocator = Nominatim(user_agent=self.GEOLOCATOR_USER_AGENT)
        return GetSurfForecastTool(session=session, geolocator=geolocator)

    def _setup_train_tickets_tool(self):
        """Builds and returns the FindTrainTicketsTool."""
        api_key = self.auth_manager.get_navitia_api_key()
        session = requests.Session()
        session.headers.update({"Authorization": api_key})
        return FindTrainTicketsTool(session=session)

    def build(self):
        """
        Assembles and returns the compiled LangGraph agent.
        """
        print(f"Building agent with calendar: {self.calendar_id}")

        tools = [
            self._setup_calendar_tool(),
            self._setup_surf_forecast_tool(),
            self._setup_train_tickets_tool(),
        ]

        # The model is already created, just use it
        graph_builder = AgentGraph(self.model, tools)
        agent = graph_builder.create_graph()

        print("Agent built successfully.")
        return agent
