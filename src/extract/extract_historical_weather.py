"""Download historical Michigan weather data from Open-Meteo."""

from pathlib import Path


import pandas as pd 


from src.apis.openmeteo_api import (
    get_historical_weather
)


START_DATE = pd.Timestamp("2015-12-31")
END_DATE = pd.Timestamp("2023-04-01")

WEATHER_RAW_DIR  = Path("data/raw/historical_weather")

LOCATIONS = [
    # Southeast Michigan / high accident representation
    {"city": "Detroit", "latitude": 42.3834, "longitude": -83.1024},
    {"city": "Southfield", "latitude": 42.4765, "longitude": -83.2605},
    {"city": "Dearborn", "latitude": 42.3127, "longitude": -83.2129},
    {"city": "Warren", "latitude": 42.4934, "longitude": -83.0270},
    {"city": "Ann Arbor", "latitude": 42.2761, "longitude": -83.7311},
    {"city": "Jackson", "latitude": 42.2431, "longitude": -84.4040},
    {"city": "Monroe", "latitude": 41.9155, "longitude": -83.3849},
    {"city": "Adrian", "latitude": 41.8993, "longitude": -84.0447},

    # Flint / Mid-Michigan / east-central
    {"city": "Flint", "latitude": 43.0235, "longitude": -83.6922},
    {"city": "Grand Blanc", "latitude": 42.9258, "longitude": -83.6182},
    {"city": "Lansing", "latitude": 42.7141, "longitude": -84.5605},
    {"city": "Mount Pleasant", "latitude": 43.5966, "longitude": -84.7758},
    {"city": "Saginaw", "latitude": 43.4193, "longitude": -83.9503},
    {"city": "Bay City", "latitude": 43.5902, "longitude": -83.8886},
    {"city": "Port Huron", "latitude": 42.9936, "longitude": -82.4339},

    # West / southwest Michigan
    {"city": "Grand Rapids", "latitude": 42.9619, "longitude": -85.6562},
    {"city": "Kalamazoo", "latitude": 42.2749, "longitude": -85.5882},
    {"city": "Battle Creek", "latitude": 42.2985, "longitude": -85.2297},
    {"city": "Benton Harbor", "latitude": 42.1159, "longitude": -86.4487},
    {"city": "Holland", "latitude": 42.7674, "longitude": -86.0986},
    {"city": "Muskegon", "latitude": 43.2281, "longitude": -86.2564},
    {"city": "Grand Haven", "latitude": 43.0553, "longitude": -86.2201},
    {"city": "Big Rapids", "latitude": 43.6992, "longitude": -85.4805},

    # Northern Lower Peninsula
    {"city": "Traverse City", "latitude": 44.7545, "longitude": -85.6037},
    {"city": "Cadillac", "latitude": 44.2484, "longitude": -85.4094},
    {"city": "Manistee", "latitude": 44.2453, "longitude": -86.3262},
    {"city": "Ludington", "latitude": 43.9573, "longitude": -86.4434},
    {"city": "Alpena", "latitude": 45.0740, "longitude": -83.4399},
    {"city": "Gaylord", "latitude": 45.0199, "longitude": -84.6811},
    {"city": "Cheboygan", "latitude": 45.6412, "longitude": -84.4686},
    {"city": "Petoskey", "latitude": 45.3650, "longitude": -84.9887},
    {"city": "Mackinaw City", "latitude": 45.7879, "longitude": -84.7484},

    # Upper Peninsula
    {"city": "Marquette", "latitude": 46.5507, "longitude": -87.3957},
    {"city": "Sault Ste. Marie", "latitude": 46.4816, "longitude": -84.3727},
    {"city": "Escanaba", "latitude": 45.7466, "longitude": -87.0830},
    {"city": "Iron Mountain", "latitude": 45.8275, "longitude": -88.0599},
    {"city": "Ironwood", "latitude": 46.4522, "longitude": -90.1504},
    {"city": "Houghton", "latitude": 47.1117, "longitude": -88.5669},
    {"city": "Menominee", "latitude": 45.1219, "longitude": -87.6232},
    {"city": "Munising", "latitude": 46.4232, "longitude": -86.6400},
]

latitudes = [location["latitude"] for location in LOCATIONS]
longitudes = [location["longitude"] for location in LOCATIONS]

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
    "weather_code",
    "wind_direction_10m",
]

def month_file_is_valid(path, start_date, end_date):
    """Check that a monthly weather file appears complete."""
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False

    expected_columns = {
        "time",
        "city",
        "latitude",
        "longitude",
        *HOURLY_VARIABLES,
    }

    if not expected_columns.issubset(df.columns):
        return False


    expected_rows = (
        (end_date - start_date).days + 1
        ) * 24 * len(LOCATIONS)


    expected_locations = {
        (
            location["city"],
            round(location["latitude"], 4),
            round(location["longitude"], 4)
        )
        for location in LOCATIONS
    }

    try:
        actual_locations = {
            (
                row.city,
                round(row.latitude, 4),
                round(row.longitude, 4)
            )
            for row in df[["city", "latitude", "longitude"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
    except KeyError as e:
        print(f"File exists but doesn't have expected columns: {e}.")
        return False

    return (
        len(df) == expected_rows
        and actual_locations == expected_locations
    )


def main():
    try:
        WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating weather data directory: {e}")
        raise


    current_start = START_DATE


    while current_start <= END_DATE:
        current_end = min(
            current_start + pd.offsets.MonthEnd(0),
            END_DATE
        )

        filename = (
            f"mi_hourly_weather_{current_start.strftime('%Y-%m')}.csv"
        )

        month_path = WEATHER_RAW_DIR / filename

        if month_file_is_valid(month_path, current_start, current_end):
            print(f"Skipping {filename}: already downloaded")
            current_start = current_end + pd.Timedelta(days=1)
            continue

        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "start_date": current_start.strftime("%Y-%m-%d"),
            "end_date": current_end.strftime("%Y-%m-%d"),
            "hourly": HOURLY_VARIABLES,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "GMT",
            "models": "era5",
        }

        print(f"Requesting {current_start.date()} to {current_end.date()}")

        responses = get_historical_weather(params)

        if len(responses) != len(LOCATIONS):
            raise ValueError(
                f"Expected {len(LOCATIONS)} responses, "
                f"but received {len(responses)}."
            )

        month_dfs = []

        for location, response in zip(LOCATIONS, responses):
            hourly = response.Hourly()

            weather_columns = {}

            for i in range(len(HOURLY_VARIABLES)):
                weather_columns[HOURLY_VARIABLES[i]] = hourly.Variables(i).ValuesAsNumpy()

            timestamps = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )

            city_df = pd.DataFrame(weather_columns)

            city_df.insert(0, "longitude", location["longitude"])
            city_df.insert(0, "latitude", location["latitude"])
            city_df.insert(0, "city", location["city"])
            city_df.insert(0, "time", timestamps)

            month_dfs.append(city_df)

        month_df = pd.concat(month_dfs, ignore_index=True)

        month_df.to_csv(month_path, index=False)

        print(f"Saved {month_path}")

        current_start = current_end + pd.Timedelta(days=1)

if __name__ == "__main__":
    main()