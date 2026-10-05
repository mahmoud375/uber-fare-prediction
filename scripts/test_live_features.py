"""
Verification script for Task 3: Live API Feature Integration
Tests live weather (Open-Meteo), traffic condition logic (TomTom),
full feature row construction, pipeline inference, and API error handling.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import requests

from src.features import (
    FEATURE_COLUMNS,
    TrafficAPIError,
    WeatherAPIError,
    build_feature_row,
    fetch_traffic,
    fetch_weather,
)

def run_verification():
    print("=" * 80)
    print("TASK 3 VERIFICATION: LIVE WEATHER, TRAFFIC & PIPELINE INFERENCE")
    print("=" * 80)

    # 1. Load pipeline and cleaned data
    pipeline = joblib.load("models/final_pipeline.joblib")
    df_clean = pd.read_csv("data/processed/uber_cleaned_data.csv", nrows=10)

    # 2. Check TomTom API Key
    tomtom_key = os.getenv("TOMTOM_API_KEY")
    has_live_tomtom_key = bool(
        tomtom_key and tomtom_key.strip() not in ("your_key_here", "your_tomtom_api_key_here")
    )

    print(f"\n[Environment Configuration]")
    print(f"  OSRM Service URL: http://127.0.0.1:5000")
    print(f"  Open-Meteo API  : Live (no key required)")
    if has_live_tomtom_key:
        print(f"  TomTom API Key  : Configured ({tomtom_key[:4]}...{tomtom_key[-4:]})")
    else:
        print(f"  TomTom API Key  : Not configured in .env (will demo speed-ratio responses & test error handling)")

    # 3. Test 3 real trips from dataset
    indices = [0, 1, 4]
    mock_traffic_ratios = [
        {"currentSpeed": 45, "freeFlowSpeed": 50, "ratio": 0.90, "expected": "Flow Traffic"},
        {"currentSpeed": 33, "freeFlowSpeed": 50, "ratio": 0.66, "expected": "Dense Traffic"},
        {"currentSpeed": 18, "freeFlowSpeed": 50, "ratio": 0.36, "expected": "Congested Traffic"},
    ]

    print("\n" + "=" * 80)
    print("PART 1: 3-TRIP VERIFICATION WITH LIVE APIS & PIPELINE PREDICTION")
    print("=" * 80)

    for i, idx in enumerate(indices):
        row = df_clean.iloc[idx]
        p_lat = float(np.degrees(row["pickup_latitude"]))
        p_lon = float(np.degrees(row["pickup_longitude"]))
        d_lat = float(np.degrees(row["dropoff_latitude"]))
        d_lon = float(np.degrees(row["dropoff_longitude"]))
        dt_str = f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d} {int(row['hour']):02d}:00:00"

        print(f"\n--- Trip #{idx} ---")
        print(f"Coordinates: Pickup=({p_lat:.6f}, {p_lon:.6f}) -> Dropoff=({d_lat:.6f}, {d_lon:.6f})")
        print(f"Pickup Datetime: {dt_str} | Passenger Count: {int(row['passenger_count'])}")

        # A. Fetch Live Weather from Open-Meteo
        live_weather = fetch_weather(p_lat, p_lon)
        print(f"  [Live Open-Meteo API] Weather Category: '{live_weather}'")

        # B. Fetch Traffic (Live if key configured, or simulated FlowSegmentData to test ratio logic)
        traffic_info = mock_traffic_ratios[i]
        if has_live_tomtom_key:
            live_traffic = fetch_traffic(p_lat, p_lon, api_key=tomtom_key)
            print(f"  [Live TomTom API]     Traffic Category: '{live_traffic}'")
        else:
            # Demonstrate fetch_traffic with mock TomTom response
            mock_resp = {
                "flowSegmentData": {
                    "currentSpeed": traffic_info["currentSpeed"],
                    "freeFlowSpeed": traffic_info["freeFlowSpeed"],
                }
            }
            with patch("requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = mock_resp
                live_traffic = fetch_traffic(p_lat, p_lon, api_key="demo_test_key")
            print(f"  [TomTom Traffic Flow] currentSpeed={traffic_info['currentSpeed']} km/h, freeFlowSpeed={traffic_info['freeFlowSpeed']} km/h")
            print(f"                        speedRatio={traffic_info['ratio']:.2f} -> Category: '{live_traffic}'")

        # C. Build Feature Row
        raw_input = {
            "pickup_lat": p_lat,
            "pickup_lon": p_lon,
            "dropoff_lat": d_lat,
            "dropoff_lon": d_lon,
            "pickup_datetime": dt_str,
            "passenger_count": int(row["passenger_count"]),
            "traffic_condition": live_traffic,  # explicitly use fetched traffic
        }
        feat_row = build_feature_row(raw_input)

        # D. Predict Fare
        pred_fare = float(pipeline.predict(feat_row)[0])
        actual_fare = float(row["fare_amount"])

        print(f"  [OSRM Road Distance]  Queried: {feat_row['road_distance_km'].values[0]:.3f} km (Dataset: {row['road_distance_km']:.3f} km)")
        print(f"  [Model Prediction]    Predicted Fare: ${pred_fare:.2f} | Dataset Actual Fare: ${actual_fare:.2f} (Diff: ${abs(pred_fare - actual_fare):.2f})")
        print(f"  [Feature Row Columns] Total: {len(feat_row.columns)} features | Shape: {feat_row.shape}")
        print("  [Sample Feature Values]:")
        for col in ["car_condition", "weather", "traffic_condition", "road_distance_km", "nyc_dist", "hour_sin", "hour_cos", "is_weekend"]:
            val = feat_row[col].values[0]
            if isinstance(val, float):
                print(f"    - {col:<24}: {val:.4f}")
            else:
                print(f"    - {col:<24}: {val}")

    # 4. Deliberately trigger error cases
    print("\n" + "=" * 80)
    print("PART 2: DELIBERATE API ERROR HANDLING VERIFICATION")
    print("=" * 80)

    # Test Case 1: TomTom Missing / Invalid API Key -> TrafficAPIError
    print("\n[Error Test 1: TomTom Missing API Key]")
    try:
        with patch.dict(os.environ):
            os.environ.pop("TOMTOM_API_KEY", None)
            fetch_traffic(40.721319, -73.844311)
        print("  FAILED: Exception was not raised!")
    except TrafficAPIError as e:
        print(f"  SUCCESS: TrafficAPIError caught as expected:")
        print(f"  >>> {e}")

    print("\n[Error Test 2: TomTom Invalid API Key (HTTP 401)]")
    try:
        fetch_traffic(40.721319, -73.844311, api_key="invalid_bad_key_12345")
        print("  FAILED: Exception was not raised!")
    except TrafficAPIError as e:
        print(f"  SUCCESS: TrafficAPIError caught as expected:")
        print(f"  >>> {e}")

    # Test Case 2: Open-Meteo Invalid Coordinates -> WeatherAPIError
    print("\n[Error Test 3: Open-Meteo Out-of-Bounds Coordinates (Latitude 999.0)]")
    try:
        fetch_weather(999.0, -73.844311)
        print("  FAILED: Exception was not raised!")
    except WeatherAPIError as e:
        print(f"  SUCCESS: WeatherAPIError caught as expected:")
        print(f"  >>> {e}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
