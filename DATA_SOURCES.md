# Data Sources & References

## V1 Overview

RoadRisk V1 uses **2021–2025 Iowa data** with a modeling unit of:

```text
county × day
```

The V1 dataset combines:

| Data                          | Purpose                              |
| ----------------------------- | ------------------------------------ |
| Iowa crash records            | Build the elevated-crash target      |
| Iowa AADT / traffic counts    | Represent traffic exposure           |
| Open-Meteo historical weather | Weather features for training        |
| Open-Meteo forecast           | Weather input for future predictions |
| Calendar features             | Weekday, month, weekend, season      |

---

# Iowa Crash Data

**Source:** Iowa Department of Transportation

Iowa DOT provides statewide point-level crash records through an ArcGIS FeatureServer, including crash location, date/time, severity, and other crash attributes.

[Iowa DOT Crash Data Feature Layer](https://gis.iowadot.gov/agshost/rest/services/Traffic_Safety/Crash_Data/FeatureServer/0)

V1 will use crash records from **2021–2025** and aggregate them to:

```text
county × day
```

The ArcGIS service limits the number of records returned per request, so extraction must support pagination.

---

# Iowa Traffic / AADT Data

**Source:** Iowa Department of Transportation Traffic Information

Iowa DOT provides statewide traffic-count data through an ArcGIS Feature Layer, including AADT and additional traffic-volume fields.

[Iowa DOT Traffic Information Layer](https://gis.iowadot.gov/rams/rest/services/lrs/FeatureServer/102)

V1 will use AADT as a county-level traffic-exposure feature.

The exact county aggregation method will be determined during data exploration, but the initial goal is to create a simple and reproducible county-level AADT feature.

---

# Historical Weather

**Source:** Open-Meteo Historical Weather API

Open-Meteo provides the historical weather used for model training.

[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

Because the modeling unit is **county × day**, V1 will use daily weather for a representative location within each Iowa county.

---

# Forecast Weather

**Source:** Open-Meteo Forecast API

The Streamlit app will use Open-Meteo forecast data for future predictions.

[Open-Meteo Forecast API Documentation](https://open-meteo.com/en/docs)

Prediction flow:

```text
future Open-Meteo forecast
        ↓
county weather features
        ↓
Logistic Regression model
        ↓
predicted probability
        ↓
risk label
```

## Weather Limitation

The model is trained using **historically observed/reconstructed weather**, not archived historical forecasts.

Future predictions therefore assume the forecast conditions occur as predicted.

> Based on historical crash patterns under similar observed conditions, this estimate assumes the forecast conditions occur as predicted.

---

# V1 Data Scope

* Iowa only
* 2021–2025
* county/day observations
* Iowa crash data
* Iowa AADT data
* Open-Meteo historical weather
* Open-Meteo forecast weather
* calendar features

Virginia, Colorado, county maps, additional models, SHAP, probability calibration, and geographic validation are outside the V1 data scope and are documented separately in the project roadmap.
