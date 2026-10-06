"""
Flask Web Application for Uber Fare Prediction.
Provides interactive map-based route selection, real-time weather & traffic integration,
and fare prediction via the trained Scikit-learn regression pipeline.
"""

from datetime import datetime
import os
import joblib
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from src.features import (
    OSRMConnectionError,
    OSRMRouteNotFoundError,
    TrafficAPIError,
    WeatherAPIError,
    build_feature_row,
    fetch_traffic,
    fetch_weather,
    snap_to_nearest_road,
)

# 1. Load environment variables once at startup
load_dotenv()
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# 2. Load trained regression model pipeline once at startup
MODEL_PATH = "models/final_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)

# NYC TLC Operational Service Area Bounding Box (Cleaned dataset envelope)
NYC_LAT_MIN, NYC_LAT_MAX = 40.4, 41.0
NYC_LON_MIN, NYC_LON_MAX = -74.3, -73.6

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the main trip selection form with interactive Leaflet map."""
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M")
    return render_template("index.html", default_datetime=now_str)


@app.route("/snap", methods=["GET", "POST"])
def snap():
    """
    Snap a clicked coordinate to the nearest drivable road using OSRM's Nearest service.
    Accepts JSON body, form data, or query params: 'lat' and 'lon'.
    Returns JSON with snapped_lat, snapped_lon, distance (meters), and street_name if within 150m.
    If no road is within 150m (or over water / off-network), returns HTTP 422 JSON error.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
    else:
        data = request.args

    lat_raw = data.get("lat") or data.get("latitude")
    lon_raw = data.get("lon") or data.get("lng") or data.get("longitude")

    if not lat_raw or not lon_raw:
        return jsonify({
            "success": False,
            "error": "Both 'lat' and 'lon' coordinates are required.",
        }), 400

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Invalid latitude or longitude format.",
        }), 400

    # Snapping threshold: 150 meters
    max_dist = 150.0

    try:
        res = snap_to_nearest_road(lat, lon, max_distance=max_dist)
        return jsonify({
            "success": True,
            "snapped_lat": res["snapped_lat"],
            "snapped_lon": res["snapped_lon"],
            "distance": round(res["distance"], 1),
            "street_name": res["street_name"],
        })
    except OSRMRouteNotFoundError as e:
        return jsonify({
            "success": False,
            "error": "No drivable road found near that point — please click closer to a street.",
            "detail": str(e),
        }), 422
    except OSRMConnectionError as e:
        return jsonify({
            "success": False,
            "error": "OSRM routing service unavailable.",
            "detail": str(e),
        }), 503
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Failed to snap to nearest road.",
            "detail": str(e),
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle trip input submission, query live external APIs (weather, traffic, OSRM),
    construct the feature row, and predict the estimated fare.
    """
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M")

    # 1. Read form inputs
    pickup_lat_raw = request.form.get("pickup_lat", "").strip()
    pickup_lon_raw = request.form.get("pickup_lon", "").strip()
    dropoff_lat_raw = request.form.get("dropoff_lat", "").strip()
    dropoff_lon_raw = request.form.get("dropoff_lon", "").strip()
    pickup_datetime = request.form.get("pickup_datetime", "").strip() or now_str
    passenger_count_raw = request.form.get("passenger_count", "1").strip() or "1"

    form_values = {
        "pickup_lat": pickup_lat_raw,
        "pickup_lon": pickup_lon_raw,
        "dropoff_lat": dropoff_lat_raw,
        "dropoff_lon": dropoff_lon_raw,
        "pickup_datetime": pickup_datetime,
        "passenger_count": passenger_count_raw,
    }

    # 2. Basic coordinate presence validation
    if not (pickup_lat_raw and pickup_lon_raw and dropoff_lat_raw and dropoff_lon_raw):
        return render_template(
            "index.html",
            error="Please click on the map to set both a Pickup point and a Dropoff point before submitting.",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    try:
        pickup_lat = float(pickup_lat_raw)
        pickup_lon = float(pickup_lon_raw)
        dropoff_lat = float(dropoff_lat_raw)
        dropoff_lon = float(dropoff_lon_raw)
        passenger_count = int(passenger_count_raw)
    except (ValueError, TypeError) as e:
        return render_template(
            "index.html",
            error=f"Invalid coordinate or passenger count format: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 3. Server-side bounding box validation (NYC TLC operational area)
    def in_nyc_box(lat: float, lon: float) -> bool:
        return (NYC_LAT_MIN <= lat <= NYC_LAT_MAX) and (NYC_LON_MIN <= lon <= NYC_LON_MAX)

    if not in_nyc_box(pickup_lat, pickup_lon) or not in_nyc_box(dropoff_lat, dropoff_lon):
        return render_template(
            "index.html",
            error=(
                "Coordinates outside the NYC service area. "
                f"Both pickup and dropoff must be within latitude [{NYC_LAT_MIN}, {NYC_LAT_MAX}] "
                f"and longitude [{NYC_LON_MIN}, {NYC_LON_MAX}]."
            ),
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 3. Stage 1: Fetch Live Weather (Open-Meteo)
    try:
        weather = fetch_weather(pickup_lat, pickup_lon)
    except WeatherAPIError as e:
        return render_template(
            "index.html",
            error=f"Weather Service Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Unexpected Weather Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 4. Stage 2: Fetch Live Traffic (TomTom)
    try:
        traffic_condition = fetch_traffic(pickup_lat, pickup_lon, api_key=TOMTOM_API_KEY)
    except TrafficAPIError as e:
        return render_template(
            "index.html",
            error=f"Traffic Service Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Unexpected Traffic Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 5. Stage 3: Feature Engineering & OSRM Road Distance Query
    try:
        raw_input = {
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat,
            "dropoff_lon": dropoff_lon,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "weather": weather,
            "traffic_condition": traffic_condition,
        }
        feature_row = build_feature_row(raw_input)
        road_distance_km = float(feature_row["road_distance_km"].values[0])
    except OSRMConnectionError as e:
        return render_template(
            "index.html",
            error=f"Routing Service Connection Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )
    except OSRMRouteNotFoundError as e:
        friendly_msg = (
            "The pickup or dropoff point you selected doesn't appear to be on a drivable street "
            "(it may be over water, a park, or too isolated). "
            "Please zoom in and click a point directly on or near a road."
        )
        return render_template(
            "index.html",
            error=friendly_msg,
            error_detail=f"Technical detail: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Feature Engineering Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 6. Stage 4: Model Prediction
    try:
        raw_pred = float(pipeline.predict(feature_row)[0])
        # NYC taxi base minimum fare is $2.50
        predicted_fare = max(2.50, round(raw_pred, 2))
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Model Prediction Error: {e}",
            form_values=form_values,
            default_datetime=pickup_datetime,
        )

    # 7. Render Result Page
    return render_template(
        "result.html",
        predicted_fare=f"{predicted_fare:.2f}",
        road_distance_km=f"{road_distance_km:.2f}",
        road_distance_miles=f"{(road_distance_km * 0.621371):.2f}",
        weather=weather,
        traffic_condition=traffic_condition,
        pickup_lat=f"{pickup_lat:.5f}",
        pickup_lon=f"{pickup_lon:.5f}",
        dropoff_lat=f"{dropoff_lat:.5f}",
        dropoff_lon=f"{dropoff_lon:.5f}",
        pickup_datetime=pickup_datetime.replace("T", " "),
        passenger_count=passenger_count,
    )


if __name__ == "__main__":
    # For standalone execution
    app.run(host="0.0.0.0", port=5001, debug=True)
