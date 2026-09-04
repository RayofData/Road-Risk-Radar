import numpy as np 
import pandas as pd
import pytest

from src.etl.transform_historical_weather import (
    circular_mean_degrees,
    create_time_blocks,
    get_weather_code_modes
)

def test_create_time_blocks():
    df = pd.DataFrame({
        "time": pd.to_datetime([
            "2025-01-01 00:00",
            "2025-01-01 02:00",
            "2025-01-01 03:00",
            "2025-01-01 05:00",
            "2025-01-01 23:00",
        ])
    })

    result = create_time_blocks(df)

    assert result["time_block"].tolist() == [0,0,3,3,21]


def test_circular_mean_handels_zero_boundary():
    directions = np.deg2rad([359,1])

    mean_sin = np.sin(directions).mean()
    mean_cos = np.cos(directions).mean()

    result = circular_mean_degrees(mean_sin, mean_cos)

    distance_from_north = min(abs(result), abs(result-360))

    assert distance_from_north < 0.001


def test_weather_code_mode_uses_highest_code_on_tie():
    df = pd.DataFrame({
        "county_fips": ["19001"] * 4,
        "county_name": ["Adair"] * 4,
        "date": [pd.Timestamp("2025-01-01")] * 4,
        "time_block": [0] * 4,
        "weather_code": [3, 3, 61, 61],
    })

    group_columns = [
        "county_fips",
        "county_name",
        "date",
        "time_block"
    ]

    result = get_weather_code_modes(df, group_columns)

    assert result.iloc[0]["weather_code"] == 61