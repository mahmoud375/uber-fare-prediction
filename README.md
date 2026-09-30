# Uber Fare Prediction — Data Engineering, EDA & Preprocessing/Modeling

> **Quick Review Navigation**
> All primary deliverables, task answers, and analytical visual interpretations are organized in:
> [`notebooks/2_EDA_Questions_and_Insights.ipynb`](notebooks/2_EDA_Questions_and_Insights.ipynb)
> The preprocessing pipeline and baseline/tuned regression models are in:
> [`notebooks/3_Preprocessing_and_Modeling.ipynb`](notebooks/3_Preprocessing_and_Modeling.ipynb)
>
> * **Dataset Health:** 489,273 clean records retained across 23 features (97.85% retention rate).
> * **Core Pipeline:** Raw Ingestion → OSRM Routing Enrichment → Data Quality Cleaning → Structured EDA → Preprocessing → Model Training.
> * **Best Model:** Tuned Gradient Boosting — RMSE 3.22, R² 0.871 on the held-out test set (vs. RMSE 5.90, R² 0.566 for the linear baseline).

---

## Repository Structure

```text
uber-fare-prediction/
├── data/
│   ├── raw/
│   │   └── final_internship_data.csv          # Original dataset (500k rows)
│   └── processed/
│       ├── uber_with_road_distances.csv       # Enriched with OSRM routes
│       ├── uber_cleaned_data.csv              # Final cleaned dataset (489,273 rows)
│       ├── train_processed.csv                # Feature-engineered train split
│       ├── test_processed.csv                 # Feature-engineered test split
│       ├── y_train.csv                        # Train target (outlier-capped)
│       └── y_test.csv                         # Test target (outlier-capped)
├── models/
│   └── final_pipeline.joblib                  # Fitted preprocessing + tuned Gradient Boosting pipeline
├── notebooks/
│   ├── 1_Data_Cleaning_Exploration.ipynb       # Auditing, unit conversion, filter design
│   ├── 2_EDA_Questions_and_Insights.ipynb      # Primary EDA deliverable
│   └── 3_Preprocessing_and_Modeling.ipynb      # Preprocessing pipeline + baseline/tuned models
├── scripts/
│   └── calculate_road_distances.py            # Asynchronous OSRM routing engine
├── osrm_nyc/                                   # Local NYC OpenStreetMap backend (gitignored)
├── .gitignore                                  # Prevents tracking heavy data & binaries
├── pyproject.toml                              # Project configuration & dependencies
└── uv.lock                                     # Exact reproducible lockfile
```

---

## Week 1 Technical Workflow

### 1. High-Performance Feature Enrichment (`scripts/calculate_road_distances.py`)

Rather than relying solely on straight-line Haversine math, actual street-network metrics were extracted using a local OpenStreetMap routing engine:

* **Async Concurrency:** Built with `aiohttp` and `asyncio.Semaphore(128)` to query a local OSRM Docker instance (`http://127.0.0.1:5000`) without network latency bottleneck.
* **Coordinate Conversion:** Raw coordinate features were stored in radians; converted dynamically to degrees.
* **Bounding Box Filtering:** Filtered pickup and dropoff pairs within the Greater New York metropolitan envelope (Longitude: `[-75.0, -72.0]`, Latitude: `[40.0, 42.0]`).
* **Snap-Distance Guard (`radiuses`):** Points farther than 300 m from any road (e.g. across the Hudson, off the coast) previously got silently snapped to a distant road, producing wildly wrong distances. A `radiuses=300;300` constraint on every OSRM request now returns `NoSegment` for those points instead, converting bad values into `NaN` rather than corrupting the column.
* **Fault Tolerance:** Atomic progress checkpoints via `.npz` files with automated retry logic for dropped connections; fully resumable if interrupted.
* **Output:** Ground-truth values for `road_distance_km` and `estimated_duration_min` (486,674 valid routes, 2,897 unreachable, 10,429 rows outside the coverage envelope).

### 2. Data Quality Audit & Sanitization (`notebooks/1_Data_Cleaning_Exploration.ipynb`)

* **Coordinate Scope:** Clipped physical locations strictly to the active New York City TLC operational zone (`[-74.3, -73.6]` Lon, `[40.4, 41.0]` Lat).
* **Economic Constraints:** Eliminated invalid zero and negative pricing; bounded `fare_amount` to realistic ranges (`$2.50` base minimum to `$300.00` maximum cap).
* **Physical Plausibility:** Removed trips with impossible zero distance or zero passengers (`passenger_count` restricted to integers `1` through `6`).
* **Direction Angle:** Enforced directional `bearing` within the valid `[0°, 360°]` domain.
* **Retention:** Dropped only ~2.15% corrupted or impossible rows, preserving 489,273 high-fidelity records for statistical analysis.

### 3. Exploratory Questions & Domain Insights (`notebooks/2_EDA_Questions_and_Insights.ipynb`)

* **Q1 (Passenger Counts vs. Fare):** Evaluated whether passenger capacity scales ticket pricing.
* **Q2 (Vehicle Condition):** Evaluated pricing variance across differing vehicle conditions.
* **Q3 (Temporal Trends):** Identified hourly pricing surges and ride volume peaks.
* **Q4 (Traffic State):** Quantified ride fare sensitivity to congestion severity.
* **Q5 (Weather Impact):** Analyzed pricing behavior across precipitation and adverse weather.
* **Q6 (Road Distance):** Analyzed the correlation and cost rate per road kilometer.
* **Q7 (Geographic Patterns):** Inspected spatial pickup hotspots across borough corridors.
* **Q8 (Airport Proximities):** Quantified flat-rate fare structures and surcharge premiums around JFK, LGA, and EWR.

---

## Week 2 Technical Workflow — Preprocessing & Modeling (`notebooks/3_Preprocessing_and_Modeling.ipynb`)

Turns `uber_cleaned_data.csv` into a model-ready dataset and trains a baseline and a tuned regression model, with every step justified from the EDA findings and no train/test leakage.

* **Missing Values / Duplicates:** Confirmed zero missing values and zero exact duplicates in the cleaned dataset — no imputation needed.
* **Train/Test Split:** 80/20 random split (`random_state=42`) performed *before* fitting any transformer; rows are independent trips with no entity ID or grouping key, so plain `KFold`-style splitting applies.
* **Feature Engineering:** Coordinates converted from radians to degrees; `hour` and `bearing_deg` cyclically encoded (`sin`/`cos`) instead of raw integers or one-hot; `is_weekend` flag added.
* **Outlier Treatment:** `road_distance_km` / `haversine_dist_km` capped at the train set's 99th percentile (real long trips kept, not deleted); `fare_amount` capped at the 99th percentile on the log scale; 158 rows with an implausible straight-line-vs-road-distance ratio (identified during Week 1's road-distance validation) are flagged with an indicator column and median-imputed rather than dropped.
* **Encoding:** `OrdinalEncoder` for the naturally ordered `car_condition` and `traffic_condition`; `OneHotEncoder` for the nominal, low-cardinality `weather`.
* **Scaling:** `StandardScaler` on numeric features, justified by the linear baseline model's sensitivity to feature scale.
* **Feature Selection:** Correlation matrix removed 3 redundant distance features (`haversine_dist_km`, `sol_dist`, `ewr_dist`, each >0.85 correlated with a kept feature); a Lasso probe flagged `car_condition`/`weather`/`traffic_condition` as nearly zero-signal; `SelectFromModel` is embedded inside the baseline pipeline so selection is refit per CV fold (no leakage).
* **Baseline Model:** `Ridge` regression with a `log1p`-transformed target (`TransformedTargetRegressor`), evaluated with 5-fold `KFold` CV.
* **Tuned Model:** `HistGradientBoostingRegressor` tuned via `RandomizedSearchCV` (25 iterations, same 5-fold CV scheme) to capture non-linear interactions the linear model missed.
* **Results (held-out test set, evaluated once):**

  | Model | MAE | RMSE | R² |
  |---|---|---|---|
  | Baseline (Ridge) | 2.748 | 5.903 | 0.566 |
  | **Tuned (Gradient Boosting)** | **1.499** | **3.217** | **0.871** |

* **Artifact:** The fitted preprocessing + tuned-model pipeline is saved to `models/final_pipeline.joblib` via `joblib`, ready to be loaded for inference on raw (unscaled/unencoded) input rows.

> **Inference note:** `road_distance_km` is not a raw input column — it is computed at request time via a local OSRM routing server (see Week 1 workflow). Serving this model in production requires either running OSRM alongside the model or substituting a Haversine-distance approximation and re-evaluating the accuracy trade-off.

---

## Reproducibility

### Local Environment Setup

Install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/mahmoud375/uber-fare-prediction.git
cd uber-fare-prediction

# Sync environment
uv sync
```

### Running the Notebooks

```bash
# EDA (Week 1 deliverable)
jupyter notebook notebooks/2_EDA_Questions_and_Insights.ipynb

# Preprocessing & Modeling (Week 2 deliverable)
jupyter notebook notebooks/3_Preprocessing_and_Modeling.ipynb
```

### Regenerating Road Distances (optional)

Requires a local OSRM Docker container serving the NYC OSM extract on port 5000:

```bash
docker start osrm-service
uv run ./scripts/calculate_road_distances.py
```