"""Calls IOWA crash data API."""

import requests
import json


IOWA_CRASH_URL = "https://gis.iowadot.gov/agshost/rest/services/Traffic_Safety/Crash_Data/FeatureServer/0/"
QUERY_URL = f"{IOWA_CRASH_URL}/query"

RAW_DIR = Path("data/raw")

OUTPUT_PATH = RAW_DIR / "iowa_crash_data_raw.json"

PROCESSED_DIR = Path("data/processed")

STATE_WEBSITE = "Iowadot.gov"

PARAMS = {
    "where": "1=1",
    "outFields": "OBJECTID,CRASH_DATE,COUNTY_NAME",
    "returnGeometry": "false",
    "resultRecordCount": 5,
    "f": "json"
}


def request_json(params):
    """Request JSON from the Iowa Dot AGSHOST data."""

    try:
        response = requests.get(
            QUERY_URL, 
            params=params, 
            timeout=60
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"{STATE_WEBSITE} API request failed: {e}") from e

    try: 
        payload = requests.json()
    except requests.JSONDecodeError as e:
        raise RuntimeError(f"{STATE_WEBSITE} API did not return valid JSON.") from e

    return payload


def get_object_ids():
    """Return all object IDs matching filter."""
    payload = request_json(PARAMS)

    object_ids = payload.get("objectIds")

    if object_ids is None:
        raise RuntimeError(
            "The API response did not contain an objectIds field."
        )
    if not object_ids:
        raise RuntimeError(
            "The API return zero matching observations."
        )

    return sorted(int(object_id) for object_id in object_ids)





