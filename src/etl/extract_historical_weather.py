"""Download historical hourly Iowa weather data from Open-Meteo."""

from pathlib import Path

import pandas as pd

from src.apis.openmeteo_api import get_historical_weather


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
WEATHER_RAW_DIR = RAW_DIR / "historical_weather"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
CRASH_PATH = INTERIM_DIR / "iowa_crash_county_counts.parquet"

COUNTY_COORDS_PATH = RAW_DIR / "iowa_county_coordinates.csv"

COUNTY_COORDS_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2025_Gazetteer/2025_gaz_counties_19.txt"
)

LOCAL_TIMEZONE = "America/Chicago"

COUNTY_BATCH_SIZE = 25

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
    "weather_code",
    "wind_direction_10m"
]


def load_date_range():
    """Get the weather date range from the interim crash dataset."""
    crashes = pd.read_parquet(CRASH_PATH, columns=["DATE"])

    start_date = crashes["DATE"].min().normalize()
    end_date = crashes["DATE"].max().normalize()

    return start_date, end_date


def load_county_locations():
    """Load representative coordinates for all Iowa counties."""
    if COUNTY_COORDS_PATH.exists():
        return pd.read_csv(COUNTY_COORDS_PATH, dtype={"county_fips": str})

    counties = pd.read_csv(
        COUNTY_COORDS_URL,
        sep="|",
        dtype={"GEOID": str}
    )

    counties = (
        counties[
            [
                "GEOID",
                "NAME",
                "INTPTLAT",
                "INTPTLONG"
            ]
        ]
        .rename(
            columns={
                "GEOID": "county_fips",
                "NAME": "county_name",
                "INTPTLAT": "latitude",
                "INTPTLONG": "longitude"
            }
        )
    )

    counties["county_name"] = counties["county_name"].str.replace(
        r" County$", "", regex=True
    )
    counties["county_fips"] = counties["county_fips"].str.zfill(5)

    if len(counties) != 99:
        raise ValueError(
            f"Expected 99 Iowa counties, found {len(counties)}."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    counties.to_csv(COUNTY_COORDS_PATH, index=False)

    return counties


def month_file_is_valid(path, start_date, end_date, counties):
    """Check that a monthly weather Parquet file appears complete."""
    if not path.exists():
        return False

    try:
        weather = pd.read_parquet(path)
    except (OSError, ValueError):
        return False

    expected_columns = {
        "time",
        "county_fips",
        "county_name",
        "latitude",
        "longitude",
        *HOURLY_VARIABLES
    }

    if not expected_columns.issubset(weather.columns):
        return False

    if weather[HOURLY_VARIABLES].isna().all().any():
        return False

    expected_counties = set(counties["county_fips"])
    actual_counties = set(weather["county_fips"])

    local_start = start_date.tz_localize(LOCAL_TIMEZONE)

    local_end = (
        end_date + pd.Timedelta(days=1)
    ).tz_localize(LOCAL_TIMEZONE)

    expected_hours = len(
        pd.date_range(
            start=local_start,
            end=local_end,
            freq="h",
            inclusive="left"
        )
    )

    expected_rows = expected_hours * len(counties)

    return (
        len(weather) == expected_rows
        and actual_counties == expected_counties
    )


def build_county_weather(location, response, start_date, end_date):
    """Convert one Open-Meteo response into an hourly county dataframe."""
    hourly = response.Hourly()

    weather_columns = {}

    for i, variable in enumerate(HOURLY_VARIABLES):
        weather_columns[variable] = hourly.Variables(i).ValuesAsNumpy()

    timestamps_utc = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    timestamps_local = timestamps_utc.tz_convert(LOCAL_TIMEZONE)

    county_weather = pd.DataFrame(weather_columns)
    county_weather.insert(0, "time", timestamps_local)

    local_dates = (
        county_weather["time"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    mask = (
        (local_dates >= start_date)
        & (local_dates <= end_date)
    )

    county_weather = county_weather.loc[mask].reset_index(drop=True)

    county_weather.insert(1, "county_fips", location["county_fips"])
    county_weather.insert(2, "county_name", location["county_name"])
    county_weather.insert(3, "latitude", location["latitude"])
    county_weather.insert(4, "longitude", location["longitude"])

    return county_weather


def download_month(counties, start_date, end_date):
    """Download one month of hourly weather for all Iowa counties."""
    month_dfs = []

    for start in range(0, len(counties), COUNTY_BATCH_SIZE):
        batch = counties.iloc[start:start + COUNTY_BATCH_SIZE]

        request_start = start_date
        request_end = end_date + pd.Timedelta(days=1)

        params = {
            "latitude": batch["latitude"].tolist(),
            "longitude": batch["longitude"].tolist(),
            "start_date": request_start.strftime("%Y-%m-%d"),
            "end_date": request_end.strftime("%Y-%m-%d"),
            "hourly": HOURLY_VARIABLES,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "GMT",
            "models": "era5"
        }

        print(
            f"Requesting counties "
            f"{start + 1}-"
            f"{min(start + COUNTY_BATCH_SIZE, len(counties))}"
        )

        responses = get_historical_weather(params)

        if len(responses) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} responses, "
                f"received {len(responses)}."
            )

        for (_, location), response in zip(batch.iterrows(), responses):
            county_weather = build_county_weather(
                location,
                response,
                start_date,
                end_date
            )

            month_dfs.append(county_weather)

    return pd.concat(month_dfs, ignore_index=True)


def main():
    """Download monthly historical weather for all Iowa counties."""
    try:
        WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Could not create weather directory {WEATHER_RAW_DIR}: {e}"
        )

    start_date, end_date = load_date_range()
    counties = load_county_locations()

    print(
        f"Weather date range: "
        f"{start_date.date()} to {end_date.date()}"
    )
    print(f"County locations: {len(counties)}")

    current_start = start_date

    while current_start <= end_date:
        current_end = min(
            current_start + pd.offsets.MonthEnd(0),
            end_date
        )

        filename = (
            f"ia_hourly_weather_"
            f"{current_start.strftime('%Y-%m')}.parquet"
        )

        month_path = WEATHER_RAW_DIR / filename

        if month_file_is_valid(
            month_path,
            current_start,
            current_end,
            counties
        ):
            print(f"Skipping {filename}: already downloaded")
            current_start = current_end + pd.Timedelta(days=1)
            continue

        print(
            f"\nDownloading "
            f"{current_start.date()} to {current_end.date()}"
        )

        month_weather = download_month(
            counties,
            current_start,
            current_end
        )

        month_weather.to_parquet(month_path, index=False)

        print(f"Saved {month_path}")
        print(f"Rows: {len(month_weather):,}")

        current_start = current_end + pd.Timedelta(days=1)


if __name__ == "__main__":
    main()