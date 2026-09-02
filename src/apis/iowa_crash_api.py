"""Calls IOWA crash data API."""

import requests
import json


IOWA_CRASH_URL = "https://gis.iowadot.gov/agshost/rest/services/Traffic_Safety/Crash_Data/FeatureServer/0"
QUERY_URL = f"{IOWA_CRASH_URL}/query"

STATE_WEBSITE = "Iowadot.gov"

BATCH_SIZE = 500

OUT_FIELDS = [
    "OBJECTID",
    "CRASH_KEY",
    "CRASH_DATE",
    "CRASH_DATETIME",
    "COUNTY_NUMBER",
    "COUNTY_NAME",
    "CSEV",
    "WEATHER",
    "CSRFCND"
]


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
        payload = response.json()
    except requests.JSONDecodeError as e:
        raise RuntimeError(f"{STATE_WEBSITE} API did not return valid JSON.") from e

    if "error" in payload:
        error = payload["error"]
        message = error.get("message", "Unknown ArcGIS error")
        details = error.get("details", [])
        raise RuntimeError(f"{message}: {details}")

    return payload


def get_object_ids():
    """Return all object IDs matching filter."""
    params = {
        "where": (
            "CRASH_DATE >= TIMESTAMP '2015-01-01 00:00:00' "
            "AND CRASH_DATE < TIMESTAMP '2026-01-01 00:00:00'"
        ),
            "returnIdsOnly": "true",
            "f": "json"

    }

    payload = request_json(params)

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


def batched(values, batch_size):
    """Yield consecutive batches from a sequence."""

    if type(batch_size) != int:
        raise ValueError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def download_batch(object_ids):
    """Download one batch of features as JSON."""

    params = {
        "objectIds": ",".join(str(object_id) for object_id in object_ids),
        "outFields": ",".join(OUT_FIELDS),
        "returnGeometry": "false",
        "f": "json"
    }

    payload = request_json(params)

    features = payload.get("features")

    if features is None:
        raise RuntimeError("JSON response did not contain a features field.")

    return payload


def download_all_features(object_ids):
    """Download and combine every matching trail feature."""

    combined_features = []
    batches = list(batched(object_ids, BATCH_SIZE))

    print(f"Matching object IDs: {len(object_ids):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(batches)}")

    for batch_number, object_id_batch in enumerate(batches, start = 1):
        print(
            f"Downloading batch {batch_number}/{len(batches)} "
            f"({len(object_id_batch)} records...)"
        )

        payload = download_batch(object_id_batch)
        features = payload["features"]

        combined_features.extend(features)

    return {
        "type": "FeatureCollection",
        "features": combined_features
    }