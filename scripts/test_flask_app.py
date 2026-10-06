"""
Integration test for Flask app routes and error handling.
"""

import re
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

client = app.test_client()

print("=== 1. Testing GET / ===")
res_get = client.get("/")
assert res_get.status_code == 200, f"GET / failed with {res_get.status_code}"
assert b"NYC Uber Fare Estimator" in res_get.data
assert b"leaflet.js" in res_get.data.lower()
print("GET / SUCCESS! Status:", res_get.status_code)

print("\n=== 2. Testing POST /predict (Missing points validation) ===")
res_missing = client.post("/predict", data={})
assert res_missing.status_code == 200
assert b"Please click on the map to set both a Pickup point and a Dropoff point" in res_missing.data
print("POST /predict (missing points) SUCCESS! Error rendered on page.")

print("\n=== 3. Testing POST /predict (Valid NYC Trip: Times Square -> City Hall) ===")
payload = {
    "pickup_lat": "40.758896",
    "pickup_lon": "-73.985130",
    "dropoff_lat": "40.712776",
    "dropoff_lon": "-74.005974",
    "pickup_datetime": "2026-10-05T15:30",
    "passenger_count": "1",
}
res_pred = client.post("/predict", data=payload)
assert res_pred.status_code == 200, f"POST /predict failed with {res_pred.status_code}"
assert b"Estimated Trip Fare" in res_pred.data, "Estimated Trip Fare not found in response"
assert b"Driving Distance (OSRM)" in res_pred.data, "Road distance not found in response"
assert b"Live Weather (Open-Meteo)" in res_pred.data, "Weather not found in response"
assert b"Live Traffic (TomTom Flow)" in res_pred.data, "Traffic not found in response"
print("POST /predict SUCCESS! Status:", res_pred.status_code)

fare_match = re.search(r'class="fare-amount">\$([0-9\.]+)<', res_pred.data.decode("utf-8"))
if fare_match:
    print(f"Predicted Fare Displayed on Result Page: ${fare_match.group(1)}")

print("\n=== 4. Testing POST /predict (Server-side check: Coordinates outside NYC service box) ===")
# Test 4a: Northern Westchester (lat 41.25 > 41.0)
err_payload_north = {
    "pickup_lat": "41.250000",
    "pickup_lon": "-73.850000",
    "dropoff_lat": "40.712776",
    "dropoff_lon": "-74.005974",
    "pickup_datetime": "2026-10-05T15:30",
    "passenger_count": "1",
}
res_err_north = client.post("/predict", data=err_payload_north)
assert res_err_north.status_code == 200
assert b"Coordinates outside the NYC service area" in res_err_north.data
print("  - North of box (lat 41.25 > 41.0) successfully caught: 'Coordinates outside the NYC service area'")

# Test 4b: West of box in New Jersey (lon -74.50 < -74.30)
err_payload_west = {
    "pickup_lat": "40.758896",
    "pickup_lon": "-74.500000",
    "dropoff_lat": "40.712776",
    "dropoff_lon": "-74.005974",
    "pickup_datetime": "2026-10-05T15:30",
    "passenger_count": "1",
}
res_err_west = client.post("/predict", data=err_payload_west)
assert res_err_west.status_code == 200
assert b"Coordinates outside the NYC service area" in res_err_west.data
print("  - West of box (lon -74.50 < -74.30) successfully caught: 'Coordinates outside the NYC service area'")

print("\n=== 5. Testing POST /predict (OSRMRouteNotFoundError: Water / Non-drivable point) ===")
# Coordinates in Upper New York Bay (inside [40.4, 41.0] and [-74.3, -73.6], but over water >300m from roads)
water_payload = {
    "pickup_lat": "40.675000",
    "pickup_lon": "-74.030000",
    "dropoff_lat": "40.712776",
    "dropoff_lon": "-74.005974",
    "pickup_datetime": "2026-10-05T15:30",
    "passenger_count": "1",
}
res_water = client.post("/predict", data=water_payload)
assert res_water.status_code == 200
assert b"The pickup or dropoff point you selected" in res_water.data
assert b"drivable street" in res_water.data
assert b"Technical detail: OSRM could not calculate a driving route" in res_water.data
print("  - Water / Non-drivable point successfully rendered friendly message and technical detail!")

print("\n=== 6. Testing GET / POST /snap (Real-Time Snap to Road) ===")
# 6a. Manhattan street (Times Square)
snap_res_a = client.post("/snap", json={"lat": 40.7580, "lon": -73.9855})
assert snap_res_a.status_code == 200, f"Point a failed with status {snap_res_a.status_code}"
snap_data_a = snap_res_a.get_json()
assert snap_data_a["success"] is True
assert snap_data_a["distance"] < 10.0, f"Point a snapped distance too large: {snap_data_a['distance']}m"
print(f"  [Point a - Manhattan Street]: status={snap_res_a.status_code}, distance={snap_data_a['distance']}m (<10m), street='{snap_data_a['street_name']}' -> SUCCESS (Snapped)")

# 6b. East River (Water point)
snap_res_b = client.post("/snap", json={"lat": 40.6750, "lon": -74.0300})
assert snap_res_b.status_code == 422, f"Point b should be 422, got {snap_res_b.status_code}"
snap_data_b = snap_res_b.get_json()
assert snap_data_b["success"] is False
assert "No drivable road found" in snap_data_b["error"]
print(f"  [Point b - East River (Water)]: status={snap_res_b.status_code}, error='{snap_data_b['error']}', detail='{snap_data_b['detail']}' -> SUCCESS (Rejected >150m)")

# 6c. Jersey City (Off-network / outside NYC OSRM graph)
snap_res_c = client.post("/snap", json={"lat": 40.7178, "lon": -74.0431})
assert snap_res_c.status_code == 422, f"Point c should be 422, got {snap_res_c.status_code}"
snap_data_c = snap_res_c.get_json()
assert snap_data_c["success"] is False
assert "No drivable road found" in snap_data_c["error"]
print(f"  [Point c - Jersey City]: status={snap_res_c.status_code}, error='{snap_data_c['error']}', detail='{snap_data_c['detail']}' -> SUCCESS (Rejected >150m)")

print("\n==========================================")
print("ALL FLASK INTEGRATION TESTS PASSED!")
print("==========================================")

