"""Sets up clients for OpenMeteo API requests."""

from pathlib import Path
import time 

import openmeteo_requests
import requests_cache
from retry_requests import retry


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

CACHE_PATH = Path(".cache") / "openmeteo"


def build_historical_client():
    """Create an Open-Meteo client with caching and retries."""
    cache_session = requests_cache.CachedSession(
        CACHE_PATH,
        expire_after=-1
    )

    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.2
    )

    return openmeteo_requests.Client(session=retry_session)

    

historical_client = build_historical_client()

def get_historical_weather(params):
    """Request historical weather data from Open-Meteo."""
    for attempt in range(1, 6):
        try:
            return historical_client.weather_api(
                ARCHIVE_API_URL,
                params=params
            )
        
        except openmeteo_requests.OpenMeteoRequestsError as e:
            if (
                "Minutely API request limit exceeded" in str(e)
                and attempt < 5
            ): 
                print(
                    f"Rate limit reached. Waiting 65 seconds "
                    f"before retry {attempt + 1} / 5."
                )
                time.sleep(65)
            else:
                raise