import ssl

import certifi
import geopy.geocoders
from geopy.exc import GeocoderServiceError
from geopy.geocoders import Nominatim
from geopy.location import Location


GEOLOCATOR_USER_AGENT = "my-surf-forecast-app-v1"


class GeolocatorAPIClient:
    """
    A dedicated wrapper for the Geopy/Nominatim service.
    Includes SSL certificate fix for environments with strict SSL verification.
    """

    def __init__(self, user_agent: str = GEOLOCATOR_USER_AGENT):
        # --- 1. Apply SSL Fix ---
        # This fixes common SSL errors on Mac/Linux by forcing geopy to use 
        # the certificates provided by the 'certifi' package.
        ctx = ssl.create_default_context(cafile=certifi.where())
        geopy.geocoders.options.default_ssl_context = ctx

        # --- 2. Initialize Service ---
        self._geolocator = Nominatim(user_agent=user_agent)

    def get_coordinates(self, location_name: str) -> Location:
        """
        Retrieves the (latitude, longitude) for a given location name.

        Raises:
            ValueError: If the location cannot be found.
            RuntimeError: If the geocoding service is unavailable.
        """
        try:
            # We assume the SSL fix applied in __init__ handles the connection security
            location = self._geolocator.geocode(location_name)

            if location:
                return location

            raise ValueError(f"Location '{location_name}' could not be found.")

        except GeocoderServiceError as e:
            # This catches specific Nominatim service errors (timeouts, quota, etc.)
            raise RuntimeError(f"The geocoding service is currently unavailable: {e}")
        except Exception as e:
            # This catches network errors (SSL, DNS, etc.)
            raise RuntimeError(f"Geocoding failed for '{location_name}': {e}")
