# Road Risk Radar

**Road Risk Radar** is a machine learning project that predicts whether an Iowa county is likely to experience **elevated crash activity on a future day and time period** based on historical crash patterns, traffic exposure, weather, and calendar conditions.

The project is built with **Python, scikit-learn, Streamlit, and Open-Meteo**.

## MVP Scope

The first version focuses only on **Iowa** so the complete data and machine learning pipeline can be built, validated, and understood before expanding to additional states.

Training data covers **2015–2025** and combines:

* Iowa statewide crash records
* Iowa AADT traffic-volume data
* Open-Meteo historical weather
* calendar features such as weekday, month, weekend, and season

The modeling unit is:

```text
county × date × 3-hour block
```

Each row represents one Iowa county for one date and 3-hour block.

## Prediction Target

RoadRisk predicts whether a county will experience **elevated crash activity compared with its historical baseline**.

Example output:

```text
Polk County, Iowa

Predicted probability of elevated crash activity:
18%

Risk:
Moderate
```

## Modeling

V1 intentionally uses a small model comparison:

1. Historical/base-rate baseline
2. Logistic Regression

The data is split chronologically:

```text
Train:      2015–2023
Validate:   2024
Test:       2025
```

This evaluates whether patterns learned from earlier years can predict crash activity in a future year.

## Streamlit App

The Streamlit application will allow the user to:

1. Select an Iowa county.
2. Select a future date available from Open-Meteo.
3. Retrieve the forecast weather for that county.
4. Receive a predicted probability of elevated crash activity.
5. View a simple risk label and the forecast conditions used by the model.

V1 does not include a map.

## Weather Limitation

The model is trained using **historically observed/reconstructed weather**, not archived historical forecasts.

Future predictions therefore assume the Open-Meteo forecast conditions occur as predicted.

> Based on historical crash patterns under similar observed conditions, this estimate assumes the forecast conditions occur as predicted.


## Goal

RoadRisk is designed as an iterative portfolio project. V1 establishes a complete, reproducible end-to-end machine learning application first, while later versions add modeling depth, geographic generalization, and visualization.
