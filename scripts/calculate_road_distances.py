"""
Enrich NYC trip data with road distance / duration from a local OSRM instance.
Resumable: if it stops, just run it again and it continues where it left off.
"""
import asyncio
import os
import sys

import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
INPUT_PATH = "data/raw/final_internship_data.csv"
OUTPUT_PATH = "data/processed/uber_with_road_distances.csv"
CHECKPOINT_PATH = "data/processed/temp_progress.npz"

OSRM_URL = "http://127.0.0.1:5000"  # 127.0.0.1 avoids slow IPv6 "localhost" lookups
CONCURRENCY = 128                   # tune: roughly 2-4x the CPU cores given to OSRM
CHUNK_SIZE = 25_000                 # checkpoint after every chunk
MAX_RETRIES = 3
REQUEST_TIMEOUT = 10                # seconds, per request

# Per-row status codes
PENDING, OK, NO_ROUTE, FAILED = 0, 1, 2, 3


# ==========================================
# 2. Data ingestion
# ==========================================
def load_data():
    print("Loading raw dataset...")
    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Radians -> degrees, as a single (n, 4) float array: p_lon, p_lat, d_lon, d_lat
    cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    coords = np.degrees(df[cols].to_numpy(dtype="float64"))
    p_lon, p_lat, d_lon, d_lat = coords.T

    def within(a, lo, hi):  # NaN -> False
        return (a >= lo) & (a <= hi)

    valid = (
        within(p_lon, -75.0, -72.0) & within(p_lat, 40.0, 42.0)
        & within(d_lon, -75.0, -72.0) & within(d_lat, 40.0, 42.0)
    )
    print(f"Total rows: {len(df):,} | Valid NYC coordinates for OSRM: {valid.sum():,}")
    return df, coords, valid


# ==========================================
# 3. Async OSRM client
# ==========================================
async def fetch_route(session, sem, c):
    """Return (distance_km, duration_min, status) for one pickup->dropoff pair."""
    p_lon, p_lat, d_lon, d_lat = c
    url = (
        f"{OSRM_URL}/route/v1/driving/"
        f"{p_lon:.6f},{p_lat:.6f};{d_lon:.6f},{d_lat:.6f}"
        "?overview=false&steps=false&alternatives=false&skip_waypoints=true"
    )
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status == 200 and data.get("code") == "Ok":
                        route = data["routes"][0]
                        return route["distance"] / 1000.0, route["duration"] / 60.0, OK
                    if resp.status < 500:
                        # NoRoute / NoSegment / bad input: retrying will not help
                        return np.nan, np.nan, NO_ROUTE
                    # 5xx -> fall through and retry
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                pass  # network hiccup / timeout / bad JSON -> retry
            await asyncio.sleep(0.2 * 2 ** attempt)  # back off (also throttles a struggling server)
    return np.nan, np.nan, FAILED


async def check_server(session):
    """Fail fast if OSRM is down, instead of silently producing an all-NaN column."""
    url = f"{OSRM_URL}/route/v1/driving/-73.9857,40.7484;-73.9772,40.7527?overview=false"
    try:
        async with session.get(url) as resp:
            data = await resp.json(content_type=None)
    except Exception as e:
        sys.exit(f"Cannot reach OSRM at {OSRM_URL}: {e!r}")
    if data.get("code") != "Ok":
        sys.exit(f"OSRM is up but the test query failed: {data}")


def save_checkpoint(dist, dur, status):
    tmp = CHECKPOINT_PATH + ".tmp.npz"
    np.savez(tmp, dist=dist, dur=dur, status=status)
    os.replace(tmp, CHECKPOINT_PATH)  # atomic: a Ctrl+C mid-write can't corrupt it


async def run(coords, todo, dist, dur, status):
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await check_server(session)
        with tqdm(total=len(todo), desc="OSRM Extraction") as pbar:
            # Chunked: only CHUNK_SIZE coroutines exist at a time (constant memory)
            for start in range(0, len(todo), CHUNK_SIZE):
                chunk = todo[start:start + CHUNK_SIZE]
                results = await asyncio.gather(
                    *(fetch_route(session, sem, coords[i]) for i in chunk)
                )
                d, t, s = zip(*results)
                dist[chunk], dur[chunk], status[chunk] = d, t, s
                save_checkpoint(dist, dur, status)
                pbar.update(len(chunk))


# ==========================================
# 4. Main
# ==========================================
def main():
    df, coords, valid = load_data()
    n = len(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dist = np.full(n, np.nan)
    dur = np.full(n, np.nan)
    status = np.full(n, PENDING, dtype=np.int8)

    if os.path.exists(CHECKPOINT_PATH):
        ck = np.load(CHECKPOINT_PATH)
        if len(ck["status"]) == n:
            print("Found existing checkpoint. Resuming...")
            dist, dur, status = ck["dist"], ck["dur"], ck["status"]
        else:
            print("Checkpoint size does not match the input data -> ignoring it.")

    # Retry PENDING and FAILED; NO_ROUTE / OK are final
    todo = np.flatnonzero(valid & ((status == PENDING) | (status == FAILED)))
    print(f"Rows remaining to process: {len(todo):,}")

    if len(todo):
        asyncio.run(run(coords, todo, dist, dur, status))

    df["road_distance_km"] = dist
    df["estimated_duration_min"] = dur
    df.to_csv(OUTPUT_PATH, index=False)

    n_ok = int((status == OK).sum())
    n_nr = int((status == NO_ROUTE).sum())
    n_fail = int((status == FAILED).sum())
    print(f"OK: {n_ok:,} | no route: {n_nr:,} | failed: {n_fail:,} | invalid coords: {int((~valid).sum()):,}")

    if n_fail:
        print("Some requests still failed - checkpoint kept. Re-run to retry only those.")
    elif os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()