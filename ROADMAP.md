# Road Risk Radar Roadmap

## V1: Iowa MVP

**Target: 3 weeks**

### Week 1

* extract Iowa crashes
* extract Iowa AADT
* extract historical Open-Meteo weather
* aggregate to county/day
* build modeling dataset

### Week 2

* EDA
* define elevated-crash target
* build historical baseline
* train Logistic Regression
* evaluate on 2025 holdout

### Week 3

* build Streamlit prediction flow
* add future Open-Meteo forecast
* add basic tests
* document methodology and limitations
* deploy

---

## V2: Virginia

* add Virginia crash extraction
* add Virginia AADT
* standardize Iowa/Virginia schemas
* retrain on both states
* compare state performance

## V3: Stronger Modeling

* add XGBoost or LightGBM
* compare with Logistic Regression
* add probability calibration
* add SHAP
* improve imbalance evaluation

## V4: Colorado + Map

* add Colorado crash extraction
* add Colorado AADT
* build three-state dataset
* add leave-one-state-out validation
* add county boundaries
* add Streamlit county risk map
