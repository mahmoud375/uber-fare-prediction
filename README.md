# Uber Fare Prediction — End-to-End Pipeline & Live Deployment

A machine learning system for predicting New York City Uber fares, spanning raw data auditing and OSRM physical road-network enrichment, leak-free preprocessing and gradient boosting regression, through to a production Flask web application with an interactive Leaflet map, real-time road snapping, and live weather and traffic API integrations.

---

## Repository Structure

```text
uber-fare-prediction/
├── data/
│   ├── raw/
│   │   └── final_internship_data.csv          # Raw trip records (500k rows)
│   └── processed/
│       ├── uber_cleaned_data.csv              # Audited & cleaned dataset (489,273 rows)
│       ├── train_processed.csv                # Feature-engineered training split
│       ├── test_processed.csv                 # Feature-engineered test split
│       ├── y_train.csv                        # Outlier-capped training targets
│       └── y_test.csv                         # Outlier-capped test targets
├── models/
│   └── final_pipeline.joblib                  # Fitted ColumnTransformer + HistGradientBoosting pipeline
├── notebooks/
│   ├── 1_Data_Cleaning_Exploration.ipynb       # Data quality audit, bounding box filters, coordinate conversion
│   ├── 2_EDA_Questions_and_Insights.ipynb      # Structured EDA answering 8 core domain hypotheses
│   └── 3_Preprocessing_and_Modeling.ipynb      # Leak-free preprocessing, CV, and 3-model benchmark comparison
├── osrm_nyc/                                   # Local NYC OpenStreetMap routing graph backend (gitignored)
├── scripts/
│   ├── calculate_road_distances.py            # High-concurrency async batch OSRM routing script
│   ├── test_live_features.py                  # Integration tests for live Open-Meteo & TomTom APIs
│   └── test_flask_app.py                      # Integration tests for Flask endpoints, error handling & routes
├── src/
│   ├── __init__.py
│   └── features.py                            # Production 22-feature pipeline & external API clients
├── templates/
│   ├── index.html                             # Web UI: interactive Leaflet map, snap-to-road, parameter form
│   └── result.html                            # Prediction result dashboard & trip summary
├── .env.example                               # Environment template for TomTom API key
├── .gitignore                                 # Git exclusion rules for large datasets, cache, and binaries
├── app.py                                     # Core Flask web server (routes: /, /snap, /predict)
├── pyproject.toml                             # Project metadata and uv dependency definitions
├── requirements.txt                           # Pip-compatible dependency manifest
├── task2_presentation.html                    # Interactive slide deck: Task 2 modeling & evaluation
├── task3_presentation.html                    # Interactive slide deck: Task 3 deployment & pipeline architecture
└── uv.lock                                    # Reproducible environment lockfile
```

---

## Three-Phase Technical Summary

### Phase 1: Data Cleaning & Road-Distance Enrichment
* **Async OSRM Ingestion (`scripts/calculate_road_distances.py`):** Replaced straight-line Euclidean math with true street-network metrics (`road_distance_km`, `estimated_duration_min`) queried asynchronously via a local OSRM Docker daemon with semaphore concurrency (`limit=128`) and atomic checkpointing.
* **Snap-Distance Guard (`radiuses=300`):** Enforced a `radiuses=300;300` parameter on all OSRM queries to prevent points located far off the road network (e.g., across the Hudson River or in water) from silently snapping to distant highways, cleanly converting disconnected coordinates into `NaN` rather than corrupting the feature distribution.
* **Coordinate & Unit Harmonization:** Converted coordinates from radians to decimal degrees and filtered outliers to the Greater New York envelope (`[-75.0, -72.0]` Lon, `[40.0, 42.0]` Lat).
* **Domain Sanitization (`notebooks/1_Data_Cleaning_Exploration.ipynb`):** Clipped coordinates to the active NYC TLC service envelope (`[-74.3, -73.6]` Lon, `[40.4, 41.0]` Lat), bounded fares to realistic operational limits ($2.50 base minimum to $300.00 cap), eliminated non-positive passenger counts (retaining 1–6 passengers), and enforced valid directional bearings `[0°, 360°]`.
* **Dataset Health:** Preserved 489,273 high-fidelity records from 500,000 raw trips (97.85% retention rate) with zero duplicates.

### Phase 2: Preprocessing & Modeling
* **Leak-Free Transformation Pipeline (`notebooks/3_Preprocessing_and_Modeling.ipynb`):** Conducted an 80/20 train/test split prior to computing any transformation statistics, ensuring zero leakage across feature engineering, scaling, and imputation.
* **Feature Engineering & Outlier Treatment:** Transformed pickup timestamps into calendar units (`year`, `month`, `day`, `weekday`, `is_weekend`) and 24-hour circular sine/cosine projections; computed forward azimuth bearings (`bearing_sin`, `bearing_cos`); capped road distances at the training 99th percentile (27.86 km); and flagged 158 anomalous road-vs-haversine discrepancies using an indicator column (`road_distance_unreliable`) with median imputation.
* **Encoding & Scaling:** Applied `StandardScaler` to numeric columns, `OrdinalEncoder` to ordered categoricals (`traffic_condition`, `car_condition`), and `OneHotEncoder` to nominal categoricals (`weather`).
* **Final Model Benchmark Comparison (Held-Out Test Set):**

  | Model | MAE | RMSE | R² |
  |:---|:---:|:---:|:---:|
  | Baseline (Ridge Regression with log1p target) | $2.748 | $5.903 | 0.566 |
  | Random Forest Regressor | $1.566 | $3.281 | 0.866 |
  | **Tuned HistGradientBoostingRegressor (Deployed)** | **$1.499** | **$3.217** | **$0.871** |

* **Deployment Artifact:** Serialized the complete, self-contained `ColumnTransformer` + tuned `HistGradientBoostingRegressor` pipeline to `models/final_pipeline.joblib`.

### Phase 3: Model Deployment
* **Live Feature-Sourcing Architecture:** At inference time, `src/features.py` dynamically assembles the exact 22-column feature vector required by the trained model pipeline across six distinct sources:
  1. **Map Pin Coordinates (4 features):** `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude` derived directly from Leaflet map selections.
  2. **Landmark Proximity (3 features):** `jfk_dist`, `lga_dist`, `nyc_dist` calculated via Haversine distance from coordinates to fixed landmark references (JFK Airport `40.6397° N, -73.7789° W`, LaGuardia `40.7772° N, -73.8726° W`, Central NYC `40.7142° N, -74.0064° W`).
  3. **Trip Bearing (2 features):** `bearing_sin`, `bearing_cos` computed from forward azimuth bearing angles between pickup and dropoff points.
  4. **Physical Road Network (2 features):** `road_distance_km` queried live from the local OSRM Docker instance (`/route/v1/driving/...`); trips exceeding training thresholds are capped and flagged via `road_distance_unreliable`.
  5. **Environmental Telemetry (2 features):** `weather` fetched live via Open-Meteo REST API (categorized into `sunny`, `cloudy`, `rainy`, `stormy`, `windy`); `traffic_condition` fetched live via TomTom Flow API (speed ratio categorized into `Flow Traffic`, `Dense Traffic`, `Congested Traffic`).
  6. **User Input & Calendar Decomposition (8 features):** `passenger_count` entered via the web form; `year`, `month`, `day`, `weekday`, `hour_sin`, `hour_cos`, and `is_weekend` decomposed from the user-selected ISO timestamp.
  7. **Operational Baseline (1 feature):** `car_condition` statically assigned to `'Good'`. Because this application serves as a pre-booking fare estimator, no specific driver or vehicle is assigned at quote time, and no public vehicle telematics API exists; `'Good'` establishes a representative operational baseline across trained classes.
* **Intelligent Snap-to-Road (`/snap`):** A rectangular bounding box (`[-74.3, -73.6]` Lon, `[40.4, 41.0]` Lat) is necessary but insufficient to constrain clicks to valid NYC TLC taxi pickups, as it includes water bodies (East River, Hudson River, Upper Bay) and adjacent non-TLC New Jersey areas (Jersey City, Hoboken, Fort Lee). The frontend communicates with the backend `/snap` endpoint on every click and drag, calling OSRM's Nearest service (`/nearest/v1/driving/...`):
  * Clicks within **150 meters** of a drivable street are snapped to the exact road centerline, updating the coordinates and displaying a success toast with the detected street name.
  * Clicks farther than **150 meters** from any road (or in water/unroutable areas) are rejected immediately (HTTP 422), preventing invalid coordinates from reaching the routing pipeline.
* **Fault-Tolerant Exception Handling:** Custom exception classes (`OSRMConnectionError`, `OSRMRouteNotFoundError`, `WeatherAPIError`, `TrafficAPIError`) wrap external integration stages. Rather than crashing or serving generic 500 error pages, failures are trapped and rendered as user-friendly contextual flash alerts on `templates/index.html` while preserving previously entered form inputs. Predictions are strictly clamped to the legal NYC TLC base minimum of **$2.50**.

---

## Prerequisites

* **Docker:** Required to serve the local OSRM routing engine on port `5000`.
* **TomTom API Key:** Required for real-time traffic flow telemetry. Free-tier keys provide 2,500 daily requests: [TomTom Developer Portal](https://developer.tomtom.com/).
* **Python 3.12+:** Managed via [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`.

---

## Setup & Running the App Locally

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/mahmoud375/uber-fare-prediction.git
cd uber-fare-prediction

# Using uv (recommended)
uv sync

# Or using standard pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your TomTom API key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
TOMTOM_API_KEY=your_actual_tomtom_api_key_here
```

### 3. Start the OSRM Routing Daemon

If the `osrm-service` container was already created during Phase 1:
```bash
docker start osrm-service
```

If setting up OSRM from scratch on a new machine:
```bash
# 1. Ensure osrm_nyc directory exists and contains the New York OSM extract
mkdir -p osrm_nyc
wget -nc https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf -P osrm_nyc/

# 2. Extract, partition, and customize the routing graph
docker run -t -v "${PWD}/osrm_nyc:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/new-york-latest.osm.pbf
docker run -t -v "${PWD}/osrm_nyc:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/new-york-latest.osrm
docker run -t -v "${PWD}/osrm_nyc:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/new-york-latest.osrm

# 3. Launch the routing daemon
docker run -d --name osrm-service -p 5000:5000 -v "${PWD}/osrm_nyc:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/new-york-latest.osrm
```

Verify OSRM health:
```bash
curl "http://127.0.0.1:5000/nearest/v1/driving/-73.9855,40.7580?number=1"
```

### 4. Run Test Suites (Optional)

```bash
# Verify live weather and traffic API integrations
uv run scripts/test_live_features.py

# Verify Flask routing, validation, snapping, and error handling
uv run scripts/test_flask_app.py
```

### 5. Launch the Web Application

```bash
uv run app.py
```

Navigate to **`http://127.0.0.1:5001`** in your browser.

---

## Using the Web Application

1. **Set Pickup Point:** Click anywhere on the map within New York City. The app validates road proximity via `/snap` and places a draggable green pickup pin (`P`) snapped to the nearest street.
2. **Set Dropoff Point:** Click a second location to place the red dropoff pin (`D`). A dashed route line will connect the two locations.
3. **Adjust Points (Optional):** Drag pins to fine-tune positions; invalid placements (>150m from road or outside NYC) revert automatically. Use *Reset Points* to clear the map.
4. **Configure Trip Details:** Select pickup date and time (defaults to current time) and specify passenger count (1 to 6).
5. **Estimate Fare:** Click **Calculate Estimated Fare**. The app queries OSRM, Open-Meteo, and TomTom, evaluates the 22-feature pipeline, and displays the result page with predicted fare, driving distance, and live environmental conditions.

---

## Notebooks & Presentations Reference

* [`notebooks/1_Data_Cleaning_Exploration.ipynb`](notebooks/1_Data_Cleaning_Exploration.ipynb): Raw data auditing, NYC TLC bounding box enforcement, physical/economic boundary clipping, and OSRM road distance validation.
* [`notebooks/2_EDA_Questions_and_Insights.ipynb`](notebooks/2_EDA_Questions_and_Insights.ipynb): Structured exploratory data analysis evaluating 8 core domain hypotheses on passenger volume, weather, traffic states, airport surcharges, and spatial patterns.
* [`notebooks/3_Preprocessing_and_Modeling.ipynb`](notebooks/3_Preprocessing_and_Modeling.ipynb): Leak-free data transformations, cyclical trigonometry, outlier capping, cross-validation, and regression model benchmarking (Ridge, Random Forest, HistGradientBoosting).
* [`task2_presentation.html`](task2_presentation.html): Interactive slide presentation detailing Task 2 modeling methodologies, feature engineering strategies, and performance evaluation metrics.
* [`task3_presentation.html`](task3_presentation.html): Interactive slide presentation with a persistent 22-feature sidebar demonstrating the Task 3 deployment architecture, live feature sourcing, and end-to-end inference flow.

---

## Known Limitations

* **Geographic Coverage Restricted to NYC:** The local OSRM routing database and server-side coordinate checks are constrained strictly to the Greater New York / NYC TLC operational zone (`[-74.3, -73.6]` Lon, `[40.4, 41.0]` Lat). Trips starting or ending outside this envelope cannot be routed or predicted.
* **Static Vehicle Condition Assumption:** `car_condition` defaults unconditionally to `'Good'`. Because the application generates pre-trip fare quotes before driver assignment, vehicle telematics cannot be inspected at inference time.
* **Distributional Shift on Modern Dates:** The model was trained on historical New York City trip records spanning 2009–2015. While cyclical hour and weekday features generalize well, submitting modern years (e.g., 2026) forces tree-based estimators to extrapolate on the `year` feature beyond the training envelope, meaning current macro-inflation and modern TLC tariff updates are not reflected in baseline predictions.
* **Server Infrastructure & Docker Overhead:** Real-time road routing requires an active OSRM backend container consuming ~2.3 GB of disk space and ~500 MB+ RAM. Consequently, standard free-tier static hosts or serverless platforms (such as Vercel, Netlify, or low-memory free PaaS instances) cannot host the complete application without an external, independently hosted OSRM routing service.