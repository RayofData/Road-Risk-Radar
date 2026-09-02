"""Extracts Iowa crash data using API helper."""

import json
import sys
from pathlib import Path

import pandas as pd 

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.apis.iowa_crash_api import (
    get_object_ids,
    download_all_features
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PATH = RAW_DIR / "iowa_crash_data_raw.json"
PROFILE_PATH = REPORT_DIR / "iowa_crash_data_report.json"


def build_profile(crashes, expected_count):
    """Create a validation and data-quality report."""

    missing_values = {}

    for column in crashes.columns:
        missing_values[column] = int(crashes[column].isna().sum())

    duplicate_object_ids = None

    if "OBJECTID" in crashes.columns:
        duplicate_object_ids = int(crashes["OBJECTID"].duplicated().sum())

    return {
        "expected_record_count": expected_count,
        "downloaded_record_count": len(crashes),
        "counts_match" : len(crashes) == expected_count,
        "duplicated_object_id_count": duplicate_object_ids,
        "missing_values": missing_values
    }


def validate_download(crashes, expected_object_ids):
    """Check that the basic crash download looks complete."""

    if crashes.empty:
        raise RuntimeError("Downloaded crash data is empty.")

    if "OBJECTID" not in crashes.columns:
        raise RuntimeError("Download data does not contain OBJECTID.")

    if crashes["OBJECTID"].duplicated().any():
        duplicated_count = crashes["OBJECTID"].duplicated().sum()
        raise RuntimeError(
            f"Downloaded data contains {duplicated_count} duplicate OBJECTIDs."
        )

    downloaded_ids = set(crashes["OBJECTID"].dropna().astype(int))
    expected_ids = set(expected_object_ids)

    missing_ids = expected_ids - downloaded_ids
    unexpected_ids = downloaded_ids - expected_ids

    if missing_ids:
        preview = sorted(missing_ids)[:10]
        raise RuntimeError(
            f"{len(missing_ids)} object IDs were not downloaded. "
            f"First missing IDs: {preview}"
        )

    if unexpected_ids:
        preview = sorted(unexpected_ids)[:10]
        raise RuntimeError(
            f"{len(unexpected_ids)} object IDs were downloaded. "
            f"First unexpect IDs: {preview}"
        )


def main():
    try: 
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Could not create output directories: {e}"
        )

    print("Requesting all matching object IDs...")
    object_ids = get_object_ids()

    feature_collection = download_all_features(object_ids)
    
    print("\nPreview combined JSON with Pandas DataFrame...")
    crashes = pd.DataFrame(
        feature["attributes"]
        for feature in feature_collection["features"]
    )

    validate_download(crashes, object_ids)

    try: 
        OUTPUT_PATH.write_text(
            json.dumps(feature_collection),
            encoding="utf-8"
        )
    except OSError as e:
        raise RuntimeError(
            f"Could not save raw JSON to {OUTPUT_PATH}: {e}"
        )

    print(crashes.head())
    print("DataFrame shape:", crashes.shape)
    print("Columns:", crashes.columns.tolist())
    print("OBJECTID unique:", crashes["OBJECTID"].is_unique)
    print("Duplicated OBJECTIDs:", crashes["OBJECTID"].duplicated().sum())
    print("Missing values:\n")
    print(crashes.isna().sum())


    profile = build_profile(
        crashes=crashes,
        expected_count=len(object_ids)
    )

    try: 
        PROFILE_PATH.write_text(
            json.dumps(profile, indent=2, default=str),
            encoding="utf-8"
        )
    except OSError as e:
        raise RuntimeError(
            f"Could not save profile to {PROFILE_PATH}: {e}"
        ) from e

    print("\nFull download complete.")
    print(f"Downloaded records: {len(crashes):,}")
    print(f"JSON saved to: {OUTPUT_PATH}")
    print(f"Profile saved to: {PROFILE_PATH}")


if __name__ == "__main__":
    main()