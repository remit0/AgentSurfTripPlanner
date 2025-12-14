from typing import Any, Type, Tuple, Dict, Optional

import dataiku
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class GcpServiceAccountInfo(BaseModel):
    type: str
    project_id: str
    private_key_id: str
    private_key: str = Field(validation_alias="private_key") # Handle potential alias if needed
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    universe_domain: str

    model_config = ConfigDict(extra='ignore')


class DataikuSettingsSource(PydanticBaseSettingsSource):
    """
    Fetches settings from Dataiku and groups 'gcp_sa_*' keys into a nested dictionary.
    """
    def __init__(self, settings_cls: Type[BaseSettings], client: Any = None):
        super().__init__(settings_cls)
        self.client = client

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        # A. Resolve Client & Project
        client = self.client if self.client else dataiku.api_client()
        project_key = dataiku.default_project_key()
        project = client.get_project(project_key)

        # B. Fetch Variables (Standard + Local)
        dss_vars = project.get_variables()
        flat_settings = {**dss_vars.get("standard", {}), **dss_vars.get("local", {})}

        # C. Fetch User Secrets
        try:
            auth_info = client.get_auth_info(with_secrets=True)
            raw_secrets = auth_info.get("secrets", [])

            for s in raw_secrets:
                flat_settings[s["key"]] = s["value"]
        except Exception:
            pass

        # D. Group 'gcp_sa_' keys into a nested dict
        gcp_nested = {}
        keys_to_remove = []
        for key, value in flat_settings.items():
            # Handle the newline escape sequence for private keys common in env vars/secrets
            if key == "gcp_sa_private_key":
                value = value.replace("\\n", "\n")

            if key.startswith("gcp_sa_"):
                clean_key = key.replace("gcp_sa_", "")
                gcp_nested[clean_key] = value
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del flat_settings[k]

        if gcp_nested:
            flat_settings["gcp_service_account"] = gcp_nested

        return flat_settings


class ProjectSettings(BaseSettings):
    # From project variables
    llm_id: str
    # From secrets
    calendar_id: Optional[str] = Field(default=None)
    NAVITIA_API_KEY: Optional[str] = Field(default=None)
    gcp_service_account: Optional[GcpServiceAccountInfo] = Field(default=None)

    # --- Runtime Injection Field ---
    dss_client: Any | None = Field(default=None, exclude=True)
    model_config = SettingsConfigDict(case_sensitive=False)

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # MAGIC HAPPENS HERE:
        # We peek into the arguments passed to ProjectSettings(...)
        # init_settings.init_kwargs holds the dict of arguments passed to __init__
        injected_client = init_settings.init_kwargs.get("dss_client")
        return (
            init_settings,
            env_settings,
            DataikuSettingsSource(settings_cls, client=injected_client),
            dotenv_settings,
        )
