"""
Feature engineering pipeline for Uber Fare Prediction.
Transforms raw ride inputs into the exact 22 feature columns expected by models/final_pipeline.joblib.
"""

import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Exact 22 feature columns in order expected by models/final_pipeline.joblib
FEATURE_COLUMNS = [
    "car_condition",
    "weather",
    "traffic_condition",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "day",
    "month",
    "weekday",
    "year",
    "jfk_dist",
    "lga_dist",
    "nyc_dist",
    "road_distance_km",
    "hour_sin",
    "hour_cos",
    "bearing_sin",
    "bearing_cos",
    "is_weekend",
    "road_distance_unreliable",
]

# Real-world landmark reference coordinates (latitude, longitude) in decimal degrees:
# - JFK Airport (center of JFK airfield/terminals): 40.6397° N, -73.7789° W
# - LaGuardia Airport (LGA terminal center): 40.7772° N, -73.8726° W
# - Central NYC (City Hall / Lower Manhattan center): 40.7142° N, -74.0064° W
JFK_COORDS = (40.6397, -73.7789)
LGA_COORDS = (40.7772, -73.8726)
NYC_COORDS = (40.7142, -74.0064)

# Constants from Notebook 3 preprocessing
ROAD_DISTANCE_CAP = 27.86  # 99th percentile cap on train set
TRAIN_MEDIAN_ROAD_DISTANCE = 2.8407  # Median road distance of reliable trips in train set
OSRM_DEFAULT_URL = "http://127.0.0.1:5000"


class OSRMError(Exception):
    """Base exception for OSRM routing errors."""
    pass


class OSRMConnectionError(OSRMError):
    """Raised when OSRM routing server is unreachable, times out, or fails to respond."""
    pass


class OSRMRouteNotFoundError(OSRMError):
    """Raised when OSRM returns NoSegment or NoRoute (e.g. coordinates > 300m from road network)."""
    pass


class WeatherAPIError(Exception):
    """Raised when the Open-Meteo weather API fails, times out, or returns invalid data."""
    pass


class TrafficAPIError(Exception):
    """Raised when the TomTom traffic API fails, times out, is missing an API key, or returns invalid data."""
    pass


def fetch_weather(lat: float, lon: float) -> str:
    """
    Fetch real-time weather at the given coordinates using Open-Meteo API.
    Maps WMO weather code and wind_speed_10m to one of the model's 5 trained categories:
    'sunny', 'rainy', 'cloudy', 'stormy', 'windy'.

    Priority rules:
    - If wind_speed_10m > 40.0 km/h: 'windy' (regardless of weather code)
    - Else if weather_code == 0: 'sunny'
    - Else if weather_code in (1, 2, 3) or (45 <= weather_code <= 48): 'cloudy'
    - Else if (51 <= weather_code <= 67) or (80 <= weather_code <= 82): 'rainy'
    - Else if (71 <= weather_code <= 77) or (85 <= weather_code <= 86): 'rainy'
    - Else if 95 <= weather_code <= 99: 'stormy'
    - Else: 'cloudy' (safe default)

    Raises WeatherAPIError if the request fails, times out, or returns invalid data.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.6f}&longitude={lon:.6f}&current=weather_code,wind_speed_10m"
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        raise WeatherAPIError(f"Open-Meteo API connection failed: {e}") from e

    if resp.status_code != 200:
        raise WeatherAPIError(
            f"Open-Meteo API returned HTTP status {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
        current = data.get("current", {})
        weather_code = current.get("weather_code", current.get("weathercode"))
        wind_speed = current.get("wind_speed_10m")
    except Exception as e:
        raise WeatherAPIError(f"Failed to parse Open-Meteo response: {e}") from e

    if weather_code is None or wind_speed is None:
        raise WeatherAPIError(
            f"Open-Meteo response missing required fields ('weather_code' or 'wind_speed_10m'): {data}"
        )

    try:
        weather_code = int(weather_code)
        wind_speed = float(wind_speed)
    except (ValueError, TypeError) as e:
        raise WeatherAPIError(
            f"Invalid weather values received (code={weather_code}, wind={wind_speed}): {e}"
        ) from e

    # Priority 1: High wind speed (> 40 km/h)
    if wind_speed > 40.0:
        return "windy"
    # Priority 2: Clear / sunny (WMO 0)
    elif weather_code == 0:
        return "sunny"
    # Priority 3: Cloudy / overcast / fog (WMO 1-3, 45-48)
    elif weather_code in (1, 2, 3) or (45 <= weather_code <= 48):
        return "cloudy"
    # Priority 4: Drizzle / Rain / Showers (WMO 51-67, 80-82)
    elif (51 <= weather_code <= 67) or (80 <= weather_code <= 82):
        return "rainy"
    # Priority 5: Snow fall / snow showers (WMO 71-77, 85-86) -> mapped to rainy
    elif (71 <= weather_code <= 77) or (85 <= weather_code <= 86):
        return "rainy"
    # Priority 6: Thunderstorm (WMO 95-99)
    elif 95 <= weather_code <= 99:
        return "stormy"
    # Priority 7: Safe default
    else:
        return "cloudy"


def fetch_traffic(lat: float, lon: float, api_key: Optional[str] = None) -> str:
    """
    Fetch real-time traffic condition at the given coordinates using TomTom Flow Segment Data API.
    Calculates currentSpeed / freeFlowSpeed ratio and maps to:
    - > 0.8: 'Flow Traffic'
    - 0.5 to 0.8: 'Dense Traffic'
    - < 0.5: 'Congested Traffic'

    Raises TrafficAPIError if the API key is missing, the request fails/times out, or response is invalid.
    """
    key = api_key or os.getenv("TOMTOM_API_KEY")
    if not key or not key.strip() or key.strip() in ("your_key_here", "your_tomtom_api_key_here"):
        raise TrafficAPIError(
            "TOMTOM_API_KEY is missing or unconfigured. "
            "Please set TOMTOM_API_KEY in your environment or .env file, or pass api_key."
        )

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat:.6f},{lon:.6f}&key={key}"
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        raise TrafficAPIError(f"TomTom Traffic API connection failed: {e}") from e

    if resp.status_code != 200:
        raise TrafficAPIError(
            f"TomTom Traffic API returned HTTP status {resp.status_code}: {resp.text}"
        )

    try:
        data = resp.json()
        flow = data.get("flowSegmentData", {})
        current_speed = flow.get("currentSpeed")
        free_flow_speed = flow.get("freeFlowSpeed")
    except Exception as e:
        raise TrafficAPIError(f"Failed to parse TomTom response: {e}") from e

    if current_speed is None or free_flow_speed is None:
        raise TrafficAPIError(
            f"TomTom response missing speed fields ('currentSpeed' or 'freeFlowSpeed'): {data}"
        )

    try:
        current_speed = float(current_speed)
        free_flow_speed = float(free_flow_speed)
    except (ValueError, TypeError) as e:
        raise TrafficAPIError(
            f"Invalid speed values from TomTom (current={current_speed}, free_flow={free_flow_speed}): {e}"
        ) from e

    if free_flow_speed <= 0:
        return "Congested Traffic"

    ratio = current_speed / free_flow_speed

    if ratio > 0.8:
        return "Flow Traffic"
    elif ratio >= 0.5:
        return "Dense Traffic"
    else:
        return "Congested Traffic"


# Alias for convenience
fetch_traffic_condition = fetch_traffic


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the Earth (specified in decimal degrees).
    Uses the exact Earth radius R = 6371.0088 km from the project's data cleaning notebook.
    """
    r = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return float(r * c)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial forward azimuth (compass bearing) from point 1 to point 2 in degrees [0, 360).
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return float(bearing_deg)


def query_osrm_road_distance(
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
    osrm_url: str = OSRM_DEFAULT_URL,
) -> float:
    """
    Query the local OSRM routing server for driving distance in km with radiuses=300;300.
    Raises OSRMConnectionError if the server cannot be reached or times out.
    Raises OSRMRouteNotFoundError if OSRM returns code != 'Ok' (e.g. NoSegment, NoRoute).
    """
    url = (
        f"{osrm_url}/route/v1/driving/"
        f"{pickup_lon:.6f},{pickup_lat:.6f};{dropoff_lon:.6f},{dropoff_lat:.6f}"
        "?overview=false&steps=false&alternatives=false&skip_waypoints=true"
        "&radiuses=300;300"
    )

    try:
        resp = requests.get(url, proxies={"http": None, "https": None}, timeout=10)
    except requests.exceptions.RequestException as e:
        raise OSRMConnectionError(
            f"Failed to connect to local OSRM routing server at {osrm_url}. "
            "Ensure the Docker container 'osrm-service' is running (docker start osrm-service)."
        ) from e

    # Parse response JSON if possible
    try:
        data = resp.json()
    except Exception:
        data = None

    if resp.status_code == 200 and data and data.get("code") == "Ok":
        routes = data.get("routes", [])
        if not routes or "distance" not in routes[0]:
            raise OSRMRouteNotFoundError("OSRM returned an empty routes list.")
        return float(routes[0]["distance"] / 1000.0)

    # Handle client errors (e.g. 400 NoSegment, NoRoute)
    if data and data.get("code") in ("NoSegment", "NoRoute", "InvalidQuery", "InvalidValue"):
        code = data.get("code")
        msg = data.get("message", "Route not found")
        raise OSRMRouteNotFoundError(
            f"OSRM could not calculate a driving route ({code}: {msg}). "
            "Coordinates may be outside Greater NYC coverage or farther than 300m from any road."
        )

    if resp.status_code != 200:
        raise OSRMConnectionError(
            f"OSRM returned unexpected HTTP status {resp.status_code}: {resp.text}"
        )

    code = data.get("code") if data else "Unknown"
    raise OSRMRouteNotFoundError(f"OSRM routing failed with code: '{code}'.")


def snap_to_nearest_road(
    lat: float,
    lon: float,
    osrm_url: str = OSRM_DEFAULT_URL,
    max_distance: Optional[float] = 150.0,
) -> Dict[str, Any]:
    """
    Query the local OSRM Nearest service to snap a coordinate to the closest drivable road.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    osrm_url : str, optional
        Base URL for the OSRM routing service (default http://127.0.0.1:5000).
    max_distance : float, optional
        Maximum allowed snapping distance in meters (default 150.0).
        If the closest road is farther than this threshold, raises OSRMRouteNotFoundError.

    Returns
    -------
    dict
        Dictionary containing:
          - 'snapped_lat': float
          - 'snapped_lon': float
          - 'distance': float (in meters)
          - 'street_name': str

    Raises
    ------
    OSRMConnectionError
        If local OSRM routing server cannot be reached or times out.
    OSRMRouteNotFoundError
        If no road is found or the nearest road exceeds max_distance.
    """
    url = f"{osrm_url}/nearest/v1/driving/{lon:.6f},{lat:.6f}?number=1"
    try:
        resp = requests.get(url, proxies={"http": None, "https": None}, timeout=5)
    except requests.exceptions.RequestException as e:
        raise OSRMConnectionError(
            f"Failed to connect to local OSRM routing server at {osrm_url}. "
            "Ensure the Docker container 'osrm-service' is running."
        ) from e

    try:
        data = resp.json()
    except Exception:
        data = None

    if resp.status_code == 200 and data and data.get("code") == "Ok":
        waypoints = data.get("waypoints", [])
        if not waypoints:
            raise OSRMRouteNotFoundError("OSRM nearest service returned no waypoints.")

        wp = waypoints[0]
        snapped_lon, snapped_lat = wp["location"]
        distance = float(wp.get("distance", 0.0))
        street_name = wp.get("name", "")

        if max_distance is not None and distance > max_distance:
            raise OSRMRouteNotFoundError(
                f"Nearest road '{street_name}' is {distance:.1f}m away, which exceeds the {max_distance:.0f}m threshold."
            )

        return {
            "snapped_lat": float(snapped_lat),
            "snapped_lon": float(snapped_lon),
            "distance": float(distance),
            "street_name": str(street_name),
        }

    if data and data.get("code") in ("NoSegment", "NoRoute", "InvalidQuery", "InvalidValue"):
        code = data.get("code")
        msg = data.get("message", "No road segment found")
        raise OSRMRouteNotFoundError(f"OSRM nearest service failed ({code}: {msg}).")

    if resp.status_code != 200:
        raise OSRMConnectionError(
            f"OSRM returned unexpected HTTP status {resp.status_code}: {resp.text}"
        )

    code = data.get("code") if data else "Unknown"
    raise OSRMRouteNotFoundError(f"OSRM nearest service failed with code: '{code}'.")


def build_feature_row(
    raw_input: Dict[str, Any],
    osrm_url: str = OSRM_DEFAULT_URL,
    tomtom_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a single-row pandas DataFrame containing the 22 features required by the trained model pipeline.

    Parameters
    ----------
    raw_input : dict
        Dictionary containing raw user inputs:
          - 'pickup_lat': float (decimal degrees)
          - 'pickup_lon': float (decimal degrees)
          - 'dropoff_lat': float (decimal degrees)
          - 'dropoff_lon': float (decimal degrees)
          - 'pickup_datetime': str or datetime-like (ISO format)
          - 'passenger_count': int
          - (optional overrides: 'weather', 'traffic_condition', 'car_condition')
    osrm_url : str, optional
        Base URL for the OSRM routing service (default http://127.0.0.1:5000).
    tomtom_api_key : str, optional
        TomTom API key. If not provided, reads from TOMTOM_API_KEY environment variable.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with exactly the 22 columns in the order expected by models/final_pipeline.joblib.
    """
    pickup_lat = float(raw_input["pickup_lat"])
    pickup_lon = float(raw_input["pickup_lon"])
    dropoff_lat = float(raw_input["dropoff_lat"])
    dropoff_lon = float(raw_input["dropoff_lon"])
    passenger_count = int(raw_input["passenger_count"])

    # 1. Parse timestamp features
    dt = pd.to_datetime(raw_input["pickup_datetime"])
    year = int(dt.year)
    month = int(dt.month)
    day = int(dt.day)
    weekday = int(dt.weekday())  # Monday = 0, Sunday = 6
    hour = int(dt.hour)

    hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
    is_weekend = int(1 if weekday in (5, 6) else 0)

    # 2. Categorical features
    car_condition = raw_input.get("car_condition", "Good")
    weather = raw_input.get("weather") or fetch_weather(pickup_lat, pickup_lon)
    traffic_condition = raw_input.get("traffic_condition") or fetch_traffic(
        pickup_lat, pickup_lon, api_key=tomtom_api_key
    )

    # 3. Landmark distance metrics (JFK, LGA, NYC)
    # Computed as combined trip proximity (pickup to landmark + dropoff to landmark)
    # matching the exact training distribution
    jfk_dist = haversine_distance(pickup_lat, pickup_lon, JFK_COORDS[0], JFK_COORDS[1]) + \
               haversine_distance(dropoff_lat, dropoff_lon, JFK_COORDS[0], JFK_COORDS[1])
    lga_dist = haversine_distance(pickup_lat, pickup_lon, LGA_COORDS[0], LGA_COORDS[1]) + \
               haversine_distance(dropoff_lat, dropoff_lon, LGA_COORDS[0], LGA_COORDS[1])
    nyc_dist = haversine_distance(pickup_lat, pickup_lon, NYC_COORDS[0], NYC_COORDS[1]) + \
               haversine_distance(dropoff_lat, dropoff_lon, NYC_COORDS[0], NYC_COORDS[1])

    # 4. Directional bearing
    bearing_deg = calculate_bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    bearing_sin = float(np.sin(np.radians(bearing_deg)))
    bearing_cos = float(np.cos(np.radians(bearing_deg)))

    # 5. Road distance query via local OSRM
    road_distance_km = query_osrm_road_distance(
        pickup_lat=pickup_lat,
        pickup_lon=pickup_lon,
        dropoff_lat=dropoff_lat,
        dropoff_lon=dropoff_lon,
        osrm_url=osrm_url,
    )

    # 6. Straight-line Haversine distance & unreliable road distance detection
    haversine_dist_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    if haversine_dist_km < 0.5 and road_distance_km > 3.0:
        road_distance_unreliable = 1
        road_distance_km = TRAIN_MEDIAN_ROAD_DISTANCE
    else:
        road_distance_unreliable = 0

    # 7. Outlier cap at 27.86 km (99th percentile cap from Notebook 3)
    road_distance_km = float(min(road_distance_km, ROAD_DISTANCE_CAP))

    # 8. Assemble single-row DataFrame in the exact expected column order
    row_data = {
        "car_condition": car_condition,
        "weather": weather,
        "traffic_condition": traffic_condition,
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passenger_count,
        "day": day,
        "month": month,
        "weekday": weekday,
        "year": year,
        "jfk_dist": float(jfk_dist),
        "lga_dist": float(lga_dist),
        "nyc_dist": float(nyc_dist),
        "road_distance_km": float(road_distance_km),
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "bearing_sin": float(bearing_sin),
        "bearing_cos": float(bearing_cos),
        "is_weekend": int(is_weekend),
        "road_distance_unreliable": int(road_distance_unreliable),
    }

    df_row = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
    return df_row
