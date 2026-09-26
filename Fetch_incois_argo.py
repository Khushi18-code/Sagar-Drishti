"""Fetch real INCOIS Argo float observations for a region/time window and
build the compact JSON explorer.html expects (argo_data.json).

This is the same pipeline as the original Fetch_incois_argo.py, refactored
so a backend (server.py) can call it directly with the box/date range the
user picked in the frontend, instead of the old fixed constants at the top
of the file. Running it from the command line still works exactly as
before.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import urllib3

BASE_URL = "https://erddap.incois.gov.in/erddap/tabledap/Indian_ARGO_Floats.csv"

VARIABLES = [
    "PLATFORM_NUMBER", "CYCLE_NUMBER", "latitude", "longitude", "time",
    "PRES", "TEMP", "PSAL", "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED",
    "PRES_QC", "TEMP_QC", "PSAL_QC",
]

MAX_LEVELS_PER_CYCLE = 60

# The local .nc snapshot below is write-only — nothing in this project reads
# it back (the frontend only ever fetches the JSON this function returns/
# writes). On a memory-constrained host (e.g. Render's free/Starter 512MB
# tier) building a whole second xarray Dataset copy of the same rows, plus
# loading the netCDF4/HDF5 C extension to write it, is pure overhead that
# can be the difference between fitting in RAM and an OOM kill. Off by
# default; set WRITE_LOCAL_ARGO_NETCDF=true if you actually use that file
# for offline analysis locally.
WRITE_LOCAL_NETCDF = os.environ.get("WRITE_LOCAL_ARGO_NETCDF", "false").lower() == "true"
MAX_CYCLES_PER_FLOAT = 20


def _pick_value(row, adjusted_col, raw_col):
    value = row.get(adjusted_col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        value = row.get(raw_col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), 3)


def _pick_qc(row, qc_col):
    value = row.get(qc_col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ArgoFetchError(RuntimeError):
    """Raised when INCOIS ERDDAP returns nothing usable for the request."""


def fetch_argo(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start: str,
    end: str,
    output_dir: Path = Path("data/incois_argo"),
    frontend_json_path: Optional[Path] = None,
) -> dict:
    """Download INCOIS Argo observations for the given box/time window,
    save the raw CSV + NetCDF under output_dir, and return (and optionally
    write) the compact JSON payload explorer.html renders.

    Raises ArgoFetchError if the request succeeds but no valid float
    observations fall inside the box/time window.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "argo_bay_of_bengal.csv"
    output_netcdf = output_dir / "argo_bay_of_bengal.nc"

    select_part = ",".join(VARIABLES)
    constraints = (
        f"latitude>={min_lat}&latitude<={max_lat}"
        f"&longitude>={min_lon}&longitude<={max_lon}"
        f"&time>={start}&time<={end}"
    )
    url = f"{BASE_URL}?{select_part}&{constraints}"

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        response = requests.get(url, timeout=180, verify=False)
    except requests.exceptions.RequestException as exc:
        raise ArgoFetchError(f"INCOIS ERDDAP request failed: {exc}") from exc

    response.raise_for_status()
    if not response.content:
        raise ArgoFetchError("Empty response from INCOIS ERDDAP.")

    output_csv.write_bytes(response.content)

    df = pd.read_csv(output_csv)
    if df.empty:
        raise ArgoFetchError("No observations returned for this region/time window.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    numeric_columns = [
        "CYCLE_NUMBER", "latitude", "longitude", "PRES", "TEMP", "PSAL",
        "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED",
        "PRES_QC", "TEMP_QC", "PSAL_QC",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["PLATFORM_NUMBER"] = pd.to_numeric(df["PLATFORM_NUMBER"], errors="coerce")

    df = df.dropna(subset=["PLATFORM_NUMBER", "CYCLE_NUMBER", "latitude", "longitude", "time"]).copy()
    if df.empty:
        raise ArgoFetchError("No valid float observations after cleaning.")

    df = df.sort_values(["PLATFORM_NUMBER", "CYCLE_NUMBER", "time"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    # ---- NetCDF (write-only, off by default — see WRITE_LOCAL_NETCDF above) ----
    if WRITE_LOCAL_NETCDF:
        import xarray as xr  # imported lazily: only pulls in netCDF4/HDF5 when actually used

        nc_df = df.copy()
        nc_df["time"] = pd.to_datetime(nc_df["time"], utc=True, errors="coerce")
        nc_df = nc_df.dropna(subset=["time"]).copy()
        nc_df["time"] = nc_df["time"].dt.tz_localize(None)

        try:
            ds = xr.Dataset(
                data_vars={
                    "TEMP": ("observation", nc_df["TEMP"].to_numpy(dtype="float32")),
                    "PSAL": ("observation", nc_df["PSAL"].to_numpy(dtype="float32")),
                    "PRES_ADJUSTED": ("observation", nc_df["PRES_ADJUSTED"].to_numpy(dtype="float32")),
                    "TEMP_ADJUSTED": ("observation", nc_df["TEMP_ADJUSTED"].to_numpy(dtype="float32")),
                    "PSAL_ADJUSTED": ("observation", nc_df["PSAL_ADJUSTED"].to_numpy(dtype="float32")),
                    "PRES_QC": ("observation", nc_df["PRES_QC"].to_numpy(dtype="float32")),
                    "TEMP_QC": ("observation", nc_df["TEMP_QC"].to_numpy(dtype="float32")),
                    "PSAL_QC": ("observation", nc_df["PSAL_QC"].to_numpy(dtype="float32")),
                    "PRES": ("observation", nc_df["PRES"].to_numpy(dtype="float32")),
                    "latitude": ("observation", nc_df["latitude"].to_numpy(dtype="float32")),
                    "longitude": ("observation", nc_df["longitude"].to_numpy(dtype="float32")),
                },
                coords={
                    "time": ("observation", nc_df["time"].to_numpy(dtype="datetime64[ns]")),
                    "PLATFORM_NUMBER": ("observation", nc_df["PLATFORM_NUMBER"].to_numpy(dtype=str)),
                    "CYCLE_NUMBER": ("observation", nc_df["CYCLE_NUMBER"].to_numpy(dtype="float32")),
                },
            )
            ds.attrs.update({
                "title": "INCOIS Indian Argo observations",
                "source": "INCOIS ERDDAP - Indian_ARGO_Floats",
                "region": f"Latitude {min_lat} to {max_lat}, Longitude {min_lon} to {max_lon}",
                "time_range": f"{start} to {end}",
            })
            ds.to_netcdf(output_netcdf, engine="netcdf4")
            del ds
        except Exception as exc:  # noqa: BLE001 - NetCDF is best-effort; JSON is what the frontend needs
            print(f"[fetch_incois_argo] NetCDF write skipped: {exc}")

    # ---- Build the compact JSON the frontend fetches ----
    floats_out = []
    for platform_id, platform_df in df.groupby("PLATFORM_NUMBER"):
        platform_df = platform_df.sort_values(["CYCLE_NUMBER", "time"])
        cycles_out = []
        for cycle_num, cycle_df in platform_df.groupby("CYCLE_NUMBER"):
            cycle_df = cycle_df.dropna(subset=["PRES"]).sort_values("PRES")
            if cycle_df.empty:
                continue
            if len(cycle_df) > MAX_LEVELS_PER_CYCLE:
                step = max(1, len(cycle_df) // MAX_LEVELS_PER_CYCLE)
                cycle_df = cycle_df.iloc[::step]

            levels = []
            for _, row in cycle_df.iterrows():
                depth = _pick_value(row, "PRES_ADJUSTED", "PRES")
                if depth is None:
                    continue
                levels.append({
                    "depth": depth,
                    "temp": _pick_value(row, "TEMP_ADJUSTED", "TEMP"),
                    "psal": _pick_value(row, "PSAL_ADJUSTED", "PSAL"),
                    "temp_qc": _pick_qc(row, "TEMP_QC"),
                    "psal_qc": _pick_qc(row, "PSAL_QC"),
                })
            if not levels:
                continue

            first_row = cycle_df.iloc[0]
            cycles_out.append({
                "cycle": int(cycle_num) if pd.notna(cycle_num) else None,
                "time": first_row["time"].isoformat() if pd.notna(first_row["time"]) else None,
                "lat": round(float(first_row["latitude"]), 4),
                "lon": round(float(first_row["longitude"]), 4),
                "levels": levels,
            })

        if not cycles_out:
            continue

        cycles_out = sorted(cycles_out, key=lambda c: c["time"] or "")[-MAX_CYCLES_PER_FLOAT:]
        last_cycle = cycles_out[-1]
        floats_out.append({
            "id": str(int(platform_id)),
            "wmo": str(int(platform_id)),
            "first_time": cycles_out[0]["time"],
            "last_time": last_cycle["time"],
            "last_lat": last_cycle["lat"],
            "last_lon": last_cycle["lon"],
            "n_cycles": len(cycles_out),
            "cycles": cycles_out,
        })

    if not floats_out:
        raise ArgoFetchError(
            "INCOIS returned rows but none survived cleaning into a usable float profile."
        )

    payload = {
        "meta": {
            "source": "INCOIS ERDDAP - Indian_ARGO_Floats",
            "region": {"minLat": min_lat, "maxLat": max_lat, "minLon": min_lon, "maxLon": max_lon},
            "time_range": {"start": start, "end": end},
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "row_count": int(len(df)),
            "float_count": len(floats_out),
        },
        "floats": floats_out,
    }

    if frontend_json_path is not None:
        frontend_json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    return payload


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-lat", type=float, default=5.0)
    parser.add_argument("--max-lat", type=float, default=15.0)
    parser.add_argument("--min-lon", type=float, default=80.0)
    parser.add_argument("--max-lon", type=float, default=90.0)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end", default="2025-01-21T06:00:00Z")
    parser.add_argument("--output-dir", type=Path, default=Path("data/incois_argo"))
    parser.add_argument("--output", type=Path, default=Path("argo_data.json"),
                         help="Frontend JSON path — keep next to index.html/explorer.html.")
    args = parser.parse_args()

    print("Fetching INCOIS Argo data...")
    try:
        payload = fetch_argo(
            args.min_lat, args.max_lat, args.min_lon, args.max_lon,
            args.start, args.end, args.output_dir, args.output,
        )
    except ArgoFetchError as exc:
        raise SystemExit(f"FAILED: {exc}")

    print(f"SUCCESS: {payload['meta']['float_count']} float(s), "
          f"{payload['meta']['row_count']} row(s) -> {args.output.resolve()}")


if __name__ == "__main__":
    _cli()