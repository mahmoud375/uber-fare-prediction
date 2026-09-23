# Uber Fare Prediction — Week 1: Data Engineering & Exploratory Analysis

> **Quick Review Navigation**
> All primary deliverables, task answers, and analytical visual interpretations are organized in:
> [`notebooks/2_EDA_Questions_and_Insights.ipynb`](https://www.google.com/search?q=notebooks/2_EDA_Questions_and_Insights.ipynb&utm_source=gemini)
> * **Dataset Health:** 489,273 clean records retained across 23 features (97.85% retention rate).
> 
> 
> * **Core Pipeline:** Raw Ingestion $\rightarrow$ OSRM Routing Enrichment $\rightarrow$ Data Quality Cleaning $\rightarrow$ Structured EDA[cite: 1].
> 
> 

---

## Repository Structure

```text
uber-fare-prediction/
├── data/
│   ├── raw/
│   │   └── final_internship_data.csv       # Original dataset (500k rows)[cite: 3]
│   └── processed/
│       ├── uber_with_road_distances.csv    # Enriched with OSRM routes
│       └── uber_cleaned_data.csv           # Final cleaned dataset[cite: 3]
├── notebooks/
│   ├── 1_Data_Cleaning_Exploration.ipynb   # Auditing, unit conversion, filter design
│   └── 2_EDA_Questions_and_Insights.ipynb  # Primary deliverable for evaluation
├── scripts/
│   └── calculate_road_distances.py         # Asynchronous OSRM routing engine
├── osrm_nyc/                               # Local NYC OpenStreetMap backend (gitignored)[cite: 4]
├── .gitignore                              # Prevents tracking heavy data & binaries[cite: 4]
├── pyproject.toml                          # Project configuration & dependencies
└── uv.lock                                 # Exact reproducible lockfile

```

---

## Week 1 Technical Workflow

### 1. High-Performance Feature Enrichment (`scripts/calculate_road_distances.py`)

Rather than relying solely on straight-line Haversine math, actual street-network metrics were extracted using a local OpenStreetMap routing engine:

* **Async Concurrency:** Built with `aiohttp` and `asyncio.Semaphore(128)` to query a local OSRM Docker instance (`[http://127.0.0.1:5000](http://127.0.0.1:5000)`) without network latency bottleneck.
* **Coordinate Conversion:** Raw coordinate features were stored in radians; converted dynamically to degrees:

$$\text{degrees} = \text{radians} \times \left(\frac{180}{\pi}\right)$$


* **Bounding Box Filtering:** Filtered pickup and dropoff pairs within the Greater New York metropolitan envelope (Longitude: `[-75.0, -72.0]`, Latitude: `[40.0, 42.0]`).
* **Fault Tolerance:** Atomic progress checkpoints via `.npz` files with automated retry logic for dropped connections.
* **Output:** Generated accurate ground-truth values for `road_distance_km` and `estimated_duration_min`.

### 2. Data Quality Audit & Sanitization (`notebooks/1_Data_Cleaning_Exploration.ipynb`)

Before analyzing business trends, the dataset underwent a systematic domain-grounded cleansing pass:

* **Coordinate Scope:** Clipped physical locations strictly to the active New York City TLC operational zone (`[-74.3, -73.6]` Lon, `[40.4, 41.0]` Lat).
* **Economic Constraints:** Eliminated invalid zero and negative pricing; bounded `fare_amount` to realistic ranges (`$2.50` base minimum to `$300.00` maximum cap).
* **Physical Plausibility:** Removed trips with impossible zero distance or zero passengers (`passenger_count` restricted to integers `1` through `6`).
* **Direction Angle:** Enforced directional `bearing` within the valid $[0^\circ, 360^\circ]$ domain.
* **Retention:** Dropped only ~2.15% corrupted or impossible rows, preserving 489,273 high-fidelity records for statistical analysis.



### 3. Exploratory Questions & Domain Insights (`notebooks/2_EDA_Questions_and_Insights.ipynb`)

The second notebook systematically addresses the task's analytical questions with visualizations and quantitative breakdowns[cite: 1]:

* **Q1 (Passenger Counts vs. Fare):** Evaluated whether passenger capacity scales ticket pricing[cite: 1].
* **Q2 (Vehicle Condition):** Evaluated pricing variance across differing vehicle conditions[cite: 1].
* **Q3 (Temporal Trends):** Identified hourly pricing surges and ride volume peaks[cite: 1].
* **Q4 (Traffic State):** Quantified ride fare sensitivity to congestion severity[cite: 1].
* **Q5 (Weather Impact):** Analyzed pricing behavior across precipitation and adverse weather[cite: 1].
* **Q6 (Road Distance):** Analyzed the correlation and cost rate per road kilometer[cite: 1].
* **Q7 (Geographic Patterns):** Inspected spatial pickup hotspots across borough corridors.
* **Q8 (Airport Proximities):** Quantified flat-rate fare structures and surcharge premiums around JFK, LGA, and EWR[cite: 6].

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

### Running the Evaluation Notebook

```bash
jupyter notebook notebooks/2_EDA_Questions_and_Insights.ipynb

```
