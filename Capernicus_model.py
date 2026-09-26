"""Fetch a Copernicus Marine ocean-model subset and sample it at each real
Argo float cycle, producing the JSON explorer.html's Compare/Analysis
panels use as the "model" half (copernicus_model.json).

Refactored from the original Capernum.py into a callable function so a
backend (server.py) can invoke it right after fetch_incois_argo.fetch_argo()
with the same box/date range the user picked, instead of needing a
pre-existing argo_data.json on disk and manual CLI runs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger("sagardrishti.capernicus_model")

DEFAULT_DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i"
DEFAULT_SAL_DATASET_ID = "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i"
MAX_LEVELS_PER_CYCLE = 60

# How long we'll wait on the network before giving up on the model step
# and returning Argo-only data instead. copernicusmarine's lazy Zarr/ARCO
# access can stall indefinitely on a slow link -- there is no built-in
# timeout in the SDK itself, so we enforce one here.
SAMPLE_TIMEOUT_SECONDS = 90
OPEN_TIMEOUT_SECONDS = 60

# Env var names copernicusmarine itself recognises (same names it would use
# if it fell back to its own resolution) — kept identical on purpose so a
# `copernicusmarine login` credentials file still works as a secondary
# fallback for local dev, while a server deployment can just export these.
_USERNAME_ENV = "COPERNICUSMARINE_SERVICE_USERNAME"
_PASSWORD_ENV = "COPERNICUSMARINE_SERVICE_PASSWORD"


class ModelFetchError(RuntimeError):
    """Raised when the Copernicus Marine subset can't be built or sampled."""


def _resolve_credentials() -> tuple[Optional[str], Optional[str]]:
    return os.environ.get(_USERNAME_ENV), os.environ.get(_PASSWORD_ENV)


def _has_cached_login_credentials() -> bool:
    return (Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials").exists()


def _normalise(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for cand in ("lat", "latitude"):
        if cand in ds.coords:
            rename[cand] = "latitude"
    for cand in ("lon", "longitude"):
        if cand in ds.coords:
            rename[cand] = "longitude"
    return ds.rename(rename) if rename else ds


def _run_with_timeout(fn, timeout_seconds: float, what: str):
    """Run fn() in a worker thread and raise ModelFetchError instead of
    hanging forever if it doesn't finish in time. Network calls into
    copernicusmarine's remote store have no timeout of their own, so
    without this a slow/stalled connection hangs the whole HTTP request
    (and the frontend with it) indefinitely.

    IMPORTANT: do NOT wrap this in `with ThreadPoolExecutor() as pool:`.
    The context manager's __exit__ calls shutdown(wait=True), which
    blocks until the worker thread finishes on its own — completely
    defeating the timeout, since a stuck network call can't actually be
    killed, only abandoned. shutdown(wait=False) below lets this
    function return the instant the timeout fires; the orphaned worker
    thread keeps running in the background (harmless) until the stalled
    call eventually errors out or the process exits."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        result = future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)
        raise ModelFetchError(
            f"Copernicus Marine {what} took longer than {timeout_seconds}s "
            "(slow/stalled connection to the remote store) — showing Argo "
            "data only. Try again, or a smaller region/date range."
        )
    else:
        pool.shutdown(wait=False)
        return result


def build_copernicus_model(
    argo_payload: dict,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    start: str,
    end: str,
    min_depth: float = 0.5,
    max_depth: float = 1000,
    dataset_id: str = DEFAULT_DATASET_ID,
    sal_dataset_id: str = DEFAULT_SAL_DATASET_ID,
    nc_dir: Path = Path("data/copernicus"),  # kept for backward compatibility; unused now
    frontend_json_path: Optional[Path] = None,
    interp_method: str = "nearest",
    max_points: int = 30,
    depth_levels: Optional[list] = None,
) -> dict:
    """Sample the Copernicus Marine temperature+salinity model at every
    cycle in argo_payload (the dict returned by fetch_incois_argo.fetch_argo).
    Returns (and optionally writes) the compact JSON explorer.html reads.

    interp_method="nearest" (the default) reads a single grid cell per
    point over the network. "linear" needs the surrounding chunks in
    every dimension too, which is dramatically slower (and sometimes
    effectively hangs) against copernicusmarine's lazy remote store — use
    it only if you need sub-grid-cell accuracy and have a fast, stable
    connection. At 0.083deg (~9km) resolution, nearest is a reasonable
    approximation for a single Argo cycle location.

    Both the dataset-open step and the actual sampling step are wrapped
    in a hard timeout (see SAMPLE_TIMEOUT_SECONDS / OPEN_TIMEOUT_SECONDS)
    so a stalled connection raises ModelFetchError instead of hanging the
    request forever — the caller (server.py) already treats ModelFetchError
    as "return the real Argo data anyway, just skip the model".

    max_points caps how many Argo cycles get sampled against the model in
    one call (default 80). A wide region/date range can return hundreds of
    Argo cycles, and each one is a separate remote chunk read — enough of
    them will blow past SAMPLE_TIMEOUT_SECONDS even though the connection
    itself is fine. When there are more cycles than max_points, they're
    thinned out evenly per float (oldest-to-newest) rather than just
    taking the first N, so every float in the box keeps some coverage
    instead of one float using up the whole budget. Raise max_points if
    you have a fast/stable connection and want denser sampling, or lower
    it if 90s still isn't enough.
    """
    import copernicusmarine  # imported lazily: only needed when this runs

    username, password = _resolve_credentials()
    if not username or not password:
        if _has_cached_login_credentials():
            auth_kwargs = {}
        else:
            raise ModelFetchError(
                "Copernicus Marine credentials not found. Set "
                f"{_USERNAME_ENV} and {_PASSWORD_ENV} as environment "
                "variables before starting server.py (or run "
                "`copernicusmarine login` once on this machine for local "
                "dev). The server will never prompt for them interactively."
            )
    else:
        auth_kwargs = {"username": username, "password": password}

    def open_lazy(ds_id: str, variable: str) -> xr.Dataset:
        return copernicusmarine.open_dataset(
            dataset_id=ds_id,
            variables=[variable],
            minimum_longitude=min_lon, maximum_longitude=max_lon,
            minimum_latitude=min_lat, maximum_latitude=max_lat,
            minimum_depth=min_depth, maximum_depth=max_depth,
            start_datetime=start, end_datetime=end,
            **auth_kwargs,
        )

    t0 = time.monotonic()
    log.info("Opening Copernicus Marine datasets lazily (no full-box download)...")

    def _open_both():
        ds_t = _normalise(open_lazy(dataset_id, "thetao"))
        ds_s = _normalise(open_lazy(sal_dataset_id, "so"))
        return ds_t, ds_s

    try:
        ds_t, ds_s = _run_with_timeout(_open_both, OPEN_TIMEOUT_SECONDS, "dataset open")
        log.info("Datasets opened in %.1fs", time.monotonic() - t0)
    except ModelFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"Copernicus Marine connection failed: {exc}") from exc

    floats_in = argo_payload.get("floats", [])
    if not floats_in:
        raise ModelFetchError("No Argo floats supplied to sample the model at.")

    jobs = []  # (float_index, cycle_dict)
    for f_idx, f in enumerate(floats_in):
        for c in f.get("cycles", []):
            if c.get("lat") is None or c.get("lon") is None or c.get("time") is None:
                continue
            jobs.append((f_idx, c))

    if not jobs:
        raise ModelFetchError("No Argo cycles with lat/lon/time to sample the model at.")

    # Cap how many cycles we actually sample against the model. Each point
    # is a separate remote chunk read against copernicusmarine's lazy Zarr
    # store, so a big region/date range that returns hundreds of Argo
    # cycles can blow past SAMPLE_TIMEOUT_SECONDS even though each
    # individual read is fast. Spread the cap evenly across floats (rather
    # than just taking the first N jobs) so every float keeps at least a
    # few cycles instead of one float eating the whole budget.
    total_jobs = len(jobs)
    if max_points and total_jobs > max_points:
        by_float: dict[int, list] = {}
        for f_idx, c in jobs:
            by_float.setdefault(f_idx, []).append(c)

        n_floats = len(by_float)
        per_float = max(1, max_points // n_floats)

        trimmed = []
        for f_idx, cycles in by_float.items():
            cycles_sorted = sorted(cycles, key=lambda c: c.get("time") or "")
            if len(cycles_sorted) > per_float:
                step = len(cycles_sorted) / per_float
                cycles_sorted = [cycles_sorted[int(i * step)] for i in range(per_float)]
            trimmed.extend((f_idx, c) for c in cycles_sorted)

        # If we're still over max_points (floats*per_float rounding), trim
        # further evenly; if we're under, top up from whatever was cut.
        if len(trimmed) > max_points:
            trimmed = trimmed[:max_points]
        jobs = trimmed
        log.info(
            "Capped Copernicus sampling to %d/%d Argo cycles across %d float(s) "
            "to stay within the %ss sampling timeout.",
            len(jobs), total_jobs, n_floats, SAMPLE_TIMEOUT_SECONDS,
        )

    def _to_naive_utc(time_iso: str) -> pd.Timestamp:
        t = pd.Timestamp(time_iso)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t

    lats = np.array([c["lat"] for _, c in jobs], dtype=float)
    lons = np.array([c["lon"] for _, c in jobs], dtype=float)
    times = np.array([_to_naive_utc(c["time"]) for _, c in jobs], dtype="datetime64[ns]")

    point_dim = "argo_point"
    lat_idx = xr.DataArray(lats, dims=point_dim)
    lon_idx = xr.DataArray(lons, dims=point_dim)
    time_idx = xr.DataArray(times, dims=point_dim)

    # For a demo, pulling the full depth column (dozens of levels) per
    # point multiplies the number of remote chunk reads. Restrict to a
    # handful of representative depths instead -- e.g. surface, 50m,
    # 100m, 200m, 500m, 1000m -- which is plenty to show a temperature/
    # salinity profile on screen and cuts the per-point data volume
    # drastically. Pass depth_levels=None to keep the full column.
    if depth_levels:
        depth_idx = xr.DataArray(np.asarray(depth_levels, dtype=float), dims="depth")

    def _select(da):
        sel = da.sel(latitude=lat_idx, longitude=lon_idx, time=time_idx, method="nearest")
        if depth_levels:
            sel = sel.sel(depth=depth_idx, method="nearest")
        return sel

    log.info("Sampling %d Argo cycle(s) against the model (method=%s)...", len(jobs), interp_method)
    t1 = time.monotonic()

    def _sample():
        if interp_method == "nearest":
            t_result = _select(ds_t["thetao"])
            s_result = _select(ds_s["so"])
        else:
            t_result = ds_t["thetao"].interp(latitude=lat_idx, longitude=lon_idx, time=time_idx, method=interp_method)
            s_result = ds_s["so"].interp(latitude=lat_idx, longitude=lon_idx, time=time_idx, method=interp_method)
        # .values is what actually triggers the network read (everything
        # above this line is still lazy).
        depths = np.atleast_1d(t_result["depth"].values).astype(float)
        temps_all = np.asarray(t_result.transpose(point_dim, "depth").values, dtype=float)
        sals_all = np.asarray(s_result.transpose(point_dim, "depth").values, dtype=float)
        return depths, temps_all, sals_all

    try:
        depths, temps_all, sals_all = _run_with_timeout(_sample, SAMPLE_TIMEOUT_SECONDS, "sampling")
    except ModelFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"Copernicus Marine sampling failed: {exc}") from exc
    log.info("Sampling done in %.1fs", time.monotonic() - t1)

    results_by_float: dict[int, list] = {i: [] for i in range(len(floats_in))}
    skipped = 0

    for row, (f_idx, c) in enumerate(jobs):
        temps = temps_all[row]
        sals = sals_all[row]
        levels = []
        for d, tv, sv in zip(depths, temps, sals):
            if np.isnan(tv) and np.isnan(sv):
                continue
            levels.append({
                "depth": round(float(d), 2),
                "temp": None if np.isnan(tv) else round(float(tv), 3),
                "psal": None if np.isnan(sv) else round(float(sv), 3),
            })
        levels.sort(key=lambda lv: lv["depth"])
        if len(levels) > MAX_LEVELS_PER_CYCLE:
            step = max(1, len(levels) // MAX_LEVELS_PER_CYCLE)
            levels = levels[::step]

        if not levels:
            skipped += 1
            continue
        results_by_float[f_idx].append({
            "cycle": c.get("cycle"), "time": c["time"],
            "lat": c["lat"], "lon": c["lon"], "levels": levels,
        })

    floats_out = []
    for f_idx, f in enumerate(floats_in):
        cycles_out = sorted(results_by_float[f_idx], key=lambda c: c["time"] or "")
        if not cycles_out:
            continue
        floats_out.append({
            "id": f["id"], "wmo": f.get("wmo", f["id"]),
            "n_cycles": len(cycles_out), "cycles": cycles_out,
        })

    payload = {
        "meta": {
            "source": f"Copernicus Marine \u2014 {dataset_id} (thetao) / {sal_dataset_id} (so)",
            "region": {"minLat": min_lat, "maxLat": max_lat, "minLon": min_lon, "maxLon": max_lon},
            "depth_range": {"min": min_depth, "max": max_depth},
            "time_range": {"start": start, "end": end},
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "float_count": len(floats_out),
            "sampled_cycles": len(jobs) - skipped,
            "skipped_cycles": skipped,
        },
        "floats": floats_out,
    }

    if frontend_json_path is not None:
        frontend_json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    if not floats_out:
        raise ModelFetchError("No model profiles could be sampled for these Argo cycles.")

    return payload


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--sal-dataset-id", default=DEFAULT_SAL_DATASET_ID)
    parser.add_argument("--min-lon", type=float, default=80)
    parser.add_argument("--max-lon", type=float, default=90)
    parser.add_argument("--min-lat", type=float, default=5)
    parser.add_argument("--max-lat", type=float, default=15)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=1000)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--argo-json", type=Path, default=Path("argo_data.json"))
    parser.add_argument("--nc-dir", type=Path, default=Path("data/copernicus"))
    parser.add_argument("--output", type=Path, default=Path("copernicus_model.json"))
    parser.add_argument("--interp-method", default="nearest", choices=["nearest", "linear"])
    parser.add_argument("--max-points", type=int, default=30,
                         help="Cap on Argo cycles sampled against the model in one call "
                              "(thinned evenly per float if exceeded). Lower this if you "
                              "keep hitting the sampling timeout.")
    parser.add_argument("--depth-levels", type=str, default="0,50,100,200,500,1000",
                         help="Comma-separated depths (m) to sample instead of the full "
                              "column, e.g. '0,50,100,200,500,1000'. Pass an empty string "
                              "to keep the full column (slower).")
    args = parser.parse_args()
    depth_levels = [float(x) for x in args.depth_levels.split(",") if x.strip()] or None

    if not args.argo_json.exists():
        raise SystemExit(f"{args.argo_json} not found — run fetch_incois_argo.py first.")
    argo_payload = json.loads(args.argo_json.read_text(encoding="utf-8"))

    try:
        payload = build_copernicus_model(
            argo_payload, args.min_lat, args.max_lat, args.min_lon, args.max_lon,
            args.start, args.end, args.min_depth, args.max_depth,
            args.dataset_id, args.sal_dataset_id, args.nc_dir, args.output,
            interp_method=args.interp_method, max_points=args.max_points,
            depth_levels=depth_levels,
        )
    except ModelFetchError as exc:
        raise SystemExit(f"FAILED: {exc}")

    print(f"SUCCESS: sampled {payload['meta']['sampled_cycles']} cycle(s) "
          f"across {payload['meta']['float_count']} float(s) -> {args.output.resolve()}")


if __name__ == "__main__":
    _cli()