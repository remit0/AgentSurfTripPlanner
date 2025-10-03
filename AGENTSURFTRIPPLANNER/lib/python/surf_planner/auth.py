from typing import Any

from google.oauth2 import service_account

DEFAULT_GCLOUD_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']


class AuthManager:
    """
    Manages fetching and preparing API credentials from Dataiku secrets.
    """

    def __init__(self, client):
        """
        Initializes the manager and fetches secrets once.
        Args:
            client: An authenticated Dataiku API client instance.
        """
        self.secrets: list[dict[str, Any]] = client.get_auth_info(with_secrets=True).get("secrets", [])

    def get_gcloud_credentials(self, scopes: list[str] = None) -> service_account.Credentials:
        """
        Constructs Google Cloud service account credentials from the fetched secrets.

        Args:
            scopes: A list of Google API scopes required.
        """
        if scopes is None:
            scopes = DEFAULT_GCLOUD_SCOPES

        service_account_info = {
            s["key"].replace("gcp_sa_", ""): s["value"]
            for s in self.secrets
            if s.get("key", "").startswith("gcp_sa_")
        }

        if not service_account_info or "private_key" not in service_account_info:
            raise ValueError("Google Cloud service account secrets (gcp_sa_*) not found or incomplete.")

        # Fix decoding of the private key newline characters
        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=scopes
        )
        return credentials

    def get_navitia_api_key(self) -> str:
        """Retrieves the Navitia API key from the fetched secrets."""
        api_key_dict = next((s for s in self.secrets if s.get("key") == "NAVITIA_API_KEY"), None)

        if not api_key_dict:
            raise ValueError("NAVITIA_API_KEY not found in Dataiku secrets.")

        return api_key_dict["value"]
