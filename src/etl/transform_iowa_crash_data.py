import json
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


RAW_PATH = Path("data/raw/iowa_crash_data_raw.json")

calendar = USFederalHolidayCalendar()


def load_raw_data():
    """Load raw Iowa crash records from the downloaded JSON file."""
    try:
        with RAW_PATH.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except OSError as e:
        raise RuntimeError(
            f"Could not load JSON: {e}"
        )

    crashes = pd.DataFrame(
        feature["attributes"]
        for feature in raw_data["features"]
    )

    return crashes


def handle_missing_values(crashes):
    """Fill missing county names and remove remaining incomplete records."""
    crashes_clean = crashes.copy()

    crashes_clean["COUNTY_NAME"] = (
        crashes_clean["COUNTY_NAME"].fillna("Unknown")
    )
    crashes_clean.dropna(inplace=True)

    return crashes_clean


def feature_engineer_dates(crashes):
    """Convert crash dates and create calendar-based features."""
    crashes_clean = crashes.copy()

    crashes_clean["CRASH_DATE"] = pd.to_datetime(
        crashes_clean["CRASH_DATE"],
        unit="ms",
    )

    crashes_clean["CRASH_DATETIME"] = pd.to_datetime(
        crashes_clean["CRASH_DATETIME"],
        unit="ms",
    )

    crashes_clean["DATE"] = (
        crashes_clean["CRASH_DATETIME"].dt.normalize()
    )

    start_date = crashes_clean["CRASH_DATE"].min().date()
    end_date = crashes_clean["CRASH_DATE"].max().date()

    federal_holidays = calendar.holidays(
        start=start_date,
        end=end_date,
    )

    crashes_clean["YEAR"] = crashes_clean["DATE"].dt.year
    crashes_clean["MONTH"] = crashes_clean["DATE"].dt.month
    crashes_clean["DAY_OF_WEEK"] = (
        crashes_clean["DATE"].dt.day_name()
    )
    crashes_clean["IS_WEEKEND"] = (
        crashes_clean["DATE"].dt.dayofweek >= 5
    )
    crashes_clean["IS_HOLIDAY"] = (
        crashes_clean["DATE"].isin(federal_holidays)
    )

    return crashes_clean


def add_time_block(crashes):
    """Assign each crash to one of eight 3-hour time blocks."""
    crashes_clean = crashes.copy()

    crashes_clean["TIME_BLOCK"] = (
        crashes_clean["CRASH_DATETIME"].dt.hour // 3
    )

    block_labels = {
        0: "00:00–02:59",
        1: "03:00–05:59",
        2: "06:00–08:59",
        3: "09:00–11:59",
        4: "12:00–14:59",
        5: "15:00–17:59",
        6: "18:00–20:59",
        7: "21:00–23:59",
    }

    crashes_clean["TIME_BLOCK_LABEL"] = (
        crashes_clean["TIME_BLOCK"].map(block_labels)
    )

    return crashes_clean


def group_county_time_blocks(crashes):
    """Count crashes for every county, date, and 3-hour time block."""
    counties = (
        crashes[["COUNTY_NUMBER", "COUNTY_NAME"]]
        .drop_duplicates()
        .sort_values("COUNTY_NUMBER")
    )

    start_date = crashes["CRASH_DATE"].min().date()
    end_date = crashes["CRASH_DATE"].max().date()

    dates = pd.date_range(
        start_date,
        end_date,
        freq="D",
    )

    blocks = range(8)

    full_index = pd.MultiIndex.from_product(
        [
            counties["COUNTY_NUMBER"],
            dates,
            blocks,
        ],
        names=["COUNTY_NUMBER", "DATE", "TIME_BLOCK"],
    )

    county_lookup = (
        counties
        .set_index("COUNTY_NUMBER")["COUNTY_NAME"]
    )

    crash_counts = (
        crashes
        .groupby(
            [
                "COUNTY_NUMBER",
                "COUNTY_NAME",
                "DATE",
                "TIME_BLOCK",
            ]
        )
        .size()
        .rename("crash_count")
        .reset_index()
    )

    observed = (
        crash_counts
        .set_index(
            ["COUNTY_NUMBER", "DATE", "TIME_BLOCK"]
        )["crash_count"]
    )

    county_block_data = (
        observed
        .reindex(full_index, fill_value=0)
        .rename("crash_count")
        .reset_index()
    )

    county_block_data["COUNTY_NAME"] = (
        county_block_data["COUNTY_NUMBER"].map(county_lookup)
    )

    return county_block_data


def main():
    """Run the Iowa crash-data transformation pipeline."""
    crashes = load_raw_data()
    crashes_clean = handle_missing_values(crashes)
    crashes_clean = feature_engineer_dates(crashes_clean)
    crashes_clean = add_time_block(crashes_clean)
    crashes_blocks = group_county_time_blocks(crashes_clean)

    print(crashes_blocks.head())
    print(crashes_blocks.shape)


if __name__ == "__main__":
    main()