"""Transform hourly historical weather into yearly 3-hour block datasets."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import calendar


RAW_DIR = Path("data/raw/historical_weather")
INTERIM_DIR = Path("data/interim/historical_weather")

STATE_CODE = "ia"
YEARS = range(2015, 2026)

EXPECTED_COUNTIES = 99
EXPECTED_TIME_BLOCKS = {0, 3, 6, 9, 12, 15, 18, 21}

INTERIM_COLUMNS = [
    "county_fips",
    "county_name",
    "date",
    "time_block",
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
    "weather_code",
    "wind_direction_10m",
]

def enforce_schema(weather):
    """Enforce consistent columns and key dtypes."""
    df = weather.copy()

    df["county_fips"] = df["county_fips"].astype("string")
    df["county_name"] = df["county_name"].astype("string")
    df["date"] = pd.to_datetime(df["date"])
    df["time_block"] = df["time_block"].astype("int8")
    df["weather_code"] = df["weather_code"].astype("Int16")

    return df[INTERIM_COLUMNS]


def get_year_files(year):
    """Return the 12 expected monthly weather files for a year."""
    files = [
        RAW_DIR / f"{STATE_CODE}_hourly_weather_{year}-{month:02d}.parquet"
        for month in range(1, 13)
    ]

    missing_files = [file for file in files if not file.exists()]

    if missing_files:
        missing_months = [
            file.stem.split("-")[-1]
            for file in missing_files
        ]

        raise ValueError(
            f"{year} is incomplete. "
            f"Missing month(s): {', '.join(missing_months)}"
        )

    return files


def load_dataframe(year):
    """Load one year of monthly hourly weather files."""
    files = get_year_files(year)

    weather_df = pd.concat(
        [pd.read_parquet(file) for file in files],
        ignore_index=True
    )

    return weather_df


def create_time_blocks(weather):
    """Create date and 3-hour block identifiers."""
    df = weather.copy()

    df["date"] = df["time"].dt.date
    df["time_block"] = (df["time"].dt.hour // 3) * 3

    return df


def circular_mean_degrees(mean_sin, mean_cos):
    """Convert mean sine and cosine components to wind direction in degrees."""
    direction = np.rad2deg(
        np.arctan2(mean_sin, mean_cos)
    )

    return direction % 360


def get_weather_code_modes(weather, group_columns):
    """
    Find the most common weather code in each 3-hour block.

    If multiple codes occur equally often, use the highest code.
    """
    code_counts = (
        weather
        .dropna(subset=["weather_code"])
        .groupby(
            group_columns + ["weather_code"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "_code_count"})
    )

    code_counts = code_counts.sort_values(
        group_columns + ["_code_count", "weather_code"],
        ascending=[True] * len(group_columns) + [False, False]
    )

    weather_code_modes = (
        code_counts
        .drop_duplicates(subset=group_columns)
        [group_columns + ["weather_code"]]
    )

    return weather_code_modes


def aggregate_weather(weather):
    """Aggregate hourly weather into county-date 3-hour blocks."""
    start = perf_counter()

    df = weather.copy()

    group_columns = [
        "county_fips",
        "county_name",
        "date",
        "time_block",
    ]

    # Convert wind direction to unit-circle components.
    wind_radians = np.deg2rad(df["wind_direction_10m"])

    df["_wind_sin"] = np.sin(wind_radians)
    df["_wind_cos"] = np.cos(wind_radians)

    weather_3hr = (
        df
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            relative_humidity_2m=("relative_humidity_2m", "mean"),
            pressure_msl=("pressure_msl", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            precipitation=("precipitation", "sum"),
            _wind_sin=("_wind_sin", "mean"),
            _wind_cos=("_wind_cos", "mean"),
        )
    )

    weather_3hr["wind_direction_10m"] = circular_mean_degrees(
        weather_3hr["_wind_sin"],
        weather_3hr["_wind_cos"],
    )

    weather_3hr = weather_3hr.drop(
        columns=["_wind_sin", "_wind_cos"]
    )

    weather_code_modes = get_weather_code_modes(
        df,
        group_columns
    )

    weather_3hr = weather_3hr.merge(
        weather_code_modes,
        on=group_columns,
        how="left"
    )

    elapsed = perf_counter() - start

    return weather_3hr, elapsed


def save_year(weather, year):
    """Save one transformed year to the interim data directory."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        INTERIM_DIR
        / f"{STATE_CODE}_weather_3hr_{year}.parquet"
    )

    weather.to_parquet(
        output_path,
        index=False
    )

    return output_path


def process_year(year):
    """Transform and save one year of historical weather."""
    output_path = (
        INTERIM_DIR
        / f"{STATE_CODE}_weather_3hr_{year}.parquet"
    )

    if output_path.exists():
        print(f"{year}: already processed, skipping.")
        return

    print(f"{year}: processing...")

    start = perf_counter()

    weather_df = load_dataframe(year)
    weather_df = create_time_blocks(weather_df)

    weather_3hr, aggregation_time = aggregate_weather(
        weather_df
    )
    weather_3hr = enforce_schema(weather_3hr)

    output_path = save_year(
        weather_3hr,
        year
    )

    total_time = perf_counter() - start

    print(
        f"{year}: {len(weather_df):,} hourly rows -> "
        f"{len(weather_3hr):,} 3-hour rows | "
        f"aggregation {aggregation_time:.1f}s | "
        f"total {total_time:.1f}s"
    )
    print(f"Saved: {output_path}")


def main():
    """Process each complete year of historical weather."""
    for year in YEARS:
        try:
            process_year(year)

        except ValueError as error:
            print(f"{year}: skipped. {error}")


if __name__ == "__main__":
    main()