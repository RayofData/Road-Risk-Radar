import pandas as pd

from src.etl.extract_historical_weather import (
    LOCAL_TIMEZONE,
    month_file_is_valid
)

def make_valid_weather_df(start_date, counties):
    """Create a small valid hourly weather dataframe for testing."""
    times = pd.date_range(
        start=start_date.tz_localize(LOCAL_TIMEZONE),
        periods=24,
        freq="h"
    )

    rows = []

    for _, county in counties.iterrows():
        for time in times:
            row = {
                "time": time,
                "county_fips": county["county_fips"],
                "county_name": county["county_name"],
                "latitude": county["latitude"],
                "longitude": county["longitude"],
                "temperature_2m": 32.0,
                "relative_humidity_2m": 80.0,
                "pressure_msl": 1015.0,
                "wind_speed_10m": 10.0,
                "precipitation": 0.0,
                "weather_code": 0,
                "wind_direction_10m": 180.0
            }

            rows.append(row)

    return pd.DataFrame(rows)


def test_month_file_is_valid_rejects_missing_file(tmp_path):
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2015-01-01")

    counties = pd.DataFrame(
        {
            "county_fips": ["19001"],
            "county_name": ["Adair"],
            "latitude": [41.33],
            "longitude": [-94.47]
        }
    )

    path = tmp_path / "missing.parquet"

    assert not month_file_is_valid(
        path,
        start_date,
        end_date,
        counties
    )


def test_month_file_is_valid_rejects_missing_column(tmp_path):
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2015-01-01")

    counties = pd.DataFrame(
        {
            "county_fips": ["19001"],
            "county_name": ["Adair"],
            "latitude": [41.33],
            "longitude": [-94.47]
        }
    )

    weather = make_valid_weather_df(start_date, counties)
    weather = weather.drop(columns=["precipitation"])

    path = tmp_path / "weather.parquet"
    weather.to_parquet(path, index=False)

    assert not month_file_is_valid(
        path,
        start_date,
        end_date,
        counties
    )


def test_month_file_is_valid_rejects_all_nan_weather_column(tmp_path):
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2015-01-01")

    counties = pd.DataFrame(
        {
            "county_fips": ["19001"],
            "county_name": ["Adair"],
            "latitude": [41.33],
            "longitude": [-94.47]
        }
    )

    weather = make_valid_weather_df(start_date, counties)
    weather["wind_speed_10m"] = float("nan")

    path = tmp_path / "weather.parquet"
    weather.to_parquet(path, index=False)

    assert not month_file_is_valid(
        path,
        start_date,
        end_date,
        counties
    )


def test_month_file_is_valid_accepts_complete_file(tmp_path):
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2015-01-01")

    counties = pd.DataFrame(
        {
            "county_fips": ["19001"],
            "county_name": ["Adair"],
            "latitude": [41.33],
            "longitude": [-94.47]
        }
    )

    weather = make_valid_weather_df(start_date, counties)

    path = tmp_path / "weather.parquet"
    weather.to_parquet(path, index=False)

    assert month_file_is_valid(
        path,
        start_date,
        end_date,
        counties,
    )