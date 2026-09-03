import pandas as pd 
from pandas.tseries.holiday import USFederalHolidayCalendar

RAW_PATH = Path("../data/raw/iowa_crash_data_raw.json")



calendar = USFederalHolidayCalendar()

federal_holidays = calendar.holidays(
    start="2015-01-01",
    end="2025-12-31"
)


def main():

    try:
        with RAW_PATH.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except OSError as e: 
        raise RuntimeError (
            f"Could not load json: {e}"
        )


    crashes = pd.DataFrame(
        feature["attributes"]
        for feature in raw_data["features"]
    )

    crashes["CRASH_DATE"] = (
        pd.to_datetime(crashes["CRASH_DATE"], unit="ms")
    )

    crashes["CRASH_DATETIME"] = (
        pd.to_datetime(crashes["CRASH_DATETIME"], unit="ms")
        .dt.normalize()
    )

    crashes_clean = crashes.copy()
    crashes_clean["COUNTY_NAME"] = crashes_clean["COUNTY_NAME"].fillna("Unknown")
    crashes_clean.dropna(inplace=True)

    crashes_clean["YEAR"] = crashes_clean["CRASH_DATE"].dt.year
    crashes_clean["MONTH"] = crashes_clean["CRASH_DATE"].dt.month 
    crashes_clean["DAY_OF_WEEK"] = crashes_clean["CRASH_DATE"].dt.day_name()
    crashes_clean["IS_WEEKEND"] = crashes_clean["CRASH_DATE"].dt.dayofweek >=5
    crashes_clean["IS_HOLIDAY"] = crashes_clean["CRASH_DATE"].isin(federal_holidays)