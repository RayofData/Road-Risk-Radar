# Road Risk Radar V1 Specification

## Goal

Build an end-to-end Streamlit ML application that predicts whether an **Iowa county will experience elevated crash activity on a future day and time period**.

## Scope

* Iowa only
* 2015–2025 data
* modeling unit: `county × date × 3-hour block`
* no map
* no boosting
* no SHAP
* no probability calibration

## Data

Use:

* Iowa statewide crash records
* Iowa AADT / traffic-volume data
* Open-Meteo historical weather
* calendar features

Crash records must be aggregated to county/day.

AADT must provide county-level traffic exposure.

Historical weather must represent each county/day.

## Target

Binary classification:

```text
elevated_crash_activity
```

The target represents whether crash activity for a county/day is elevated relative to that county's historical baseline.

The exact threshold or baseline definition must be determined during EDA and documented.

## Models

Compare:

1. Historical/base-rate baseline
2. Logistic Regression

## Validation

Use a temporal split:

```text
Train:          2015–2023
Validation:     2024
Test:           2025
```

Do not use a random split as the primary evaluation.

## Streamlit

User inputs:

* Iowa county
* future date available from Open-Meteo

The app must:

1. retrieve the forecast weather
2. generate the required model features
3. run the trained Logistic Regression model
4. display predicted probability
5. display a simple risk label
6. display the forecast weather used

## Limitation

The model is trained on historically observed/reconstructed weather, not archived historical forecasts.

Future predictions assume the forecast conditions occur as predicted.

## V1 Done Criteria

V1 is complete when:

* crash extraction is reproducible
* AADT is incorporated
* historical weather extraction is reproducible
* the county/day modeling dataset can be rebuilt
* the baseline is evaluated
* Logistic Regression is evaluated
* 2025 remains held out during training
* Streamlit can generate a future prediction
* limitations are documented
