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
import gc
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

# Env var names copernicusmarine itself recognises (same names it would use
# if it fell back to its own resolution) — kept identical on purpose so a
# `copernicusmarine login` credentials file still works as a secondary
# fallback for local dev, while a server deployment can just export these.
_USERNAME_ENV = "COPERNICUSMARINE_SERVICE_USERNAME"
_PASSWORD_ENV = "COPERNICUSMARINE_SERVICE_PASSWORD"


class ModelFetchError(RuntimeError):
    """Raised when the Copernicus Marine subset can't be built or sampled."""


def _resolve_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read Copernicus Marine credentials from the environment.

    IMPORTANT: this exists so a running server never falls through to
    copernicusmarine's own interactive `input()` prompt for a username/
    password. A web request has no terminal attached — an interactive
    prompt there just hangs the request instead of failing cleanly.

    Returns (None, None) if the env vars aren't set. Callers should treat
    that as "can't authenticate" and raise ModelFetchError rather than
    calling copernicusmarine.subset() without credentials, UNLESS a
    `copernicusmarine login` credentials file is already present on disk
    (local dev convenience) — see _has_cached_login_credentials().
    """
    return os.environ.get(_USERNAME_ENV), os.environ.get(_PASSWORD_ENV)


def _has_cached_login_credentials() -> bool:
    """True if `copernicusmarine login` was already run once on this
    machine, i.e. a credentials file exists at the SDK's default location.
    Only used as a local-dev fallback — don't rely on this in production."""
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
    nc_dir: Path = Path("data/copernicus"),  # kept for backward compatibility; unused now (see note below)
    frontend_json_path: Optional[Path] = None,
) -> dict:
    """Sample the Copernicus Marine temperature+salinity model at every
    cycle in argo_payload (the dict returned by fetch_incois_argo.fetch_argo).
    Returns (and optionally writes) the compact JSON explorer.html reads.

    SPEED NOTE — why this doesn't download a .nc file to disk anymore:
    the previous version called copernicusmarine.subset(), which downloads
    the ENTIRE lat/lon/depth/time box to a local NetCDF file before any
    interpolation happens — even though only a handful of exact Argo
    cycle points are actually needed. That full-box download is what was
    slow, not a missing "chunking" setting.

    This version uses copernicusmarine.open_dataset() instead, which opens
    the dataset lazily straight from Copernicus's cloud-native ARCO/Zarr
    store — nothing is transferred until you actually read a value. All
    Argo cycle points are then interpolated in ONE vectorized .interp()
    call (xarray's pointwise/advanced indexing: pass same-length arrays
    sharing one dimension name and it interpolates all of them in a
    single request) rather than looping per point. A single call also
    avoids running concurrent reads against the same lazy remote dataset
    from multiple threads, which isn't guaranteed safe with the async
    HTTP access some Zarr/fsspec backends use — that's what produced the
    hang you saw with the previous (threaded) version.

    `nc_dir` is accepted but no longer used (nothing is written to disk
    for the model fetch) — kept only so existing callers (server.py, the
    CLI below) don't need to change their call signature.

    Requires `copernicusmarine login` to have been run once on this
    machine (or the equivalent environment variables set) so the
    Copernicus Marine CLI/SDK can authenticate.
    """
    import copernicusmarine  # imported lazily: only needed when this runs

    # ---- Credentials: resolved BEFORE touching the network, never left to
    # copernicusmarine's own interactive prompt (see _resolve_credentials). ----
    username, password = _resolve_credentials()
    if not username or not password:
        if _has_cached_login_credentials():
            # Local dev convenience: `copernicusmarine login` was run once,
            # so let the SDK read its own credentials file. Don't pass
            # username/password at all in this case (passing None explicitly
            # can short-circuit that fallback in some SDK versions).
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
    try:
        log.info("Opening Copernicus Marine datasets lazily (no full-box download)...")
        ds_t = _normalise(open_lazy(dataset_id, "thetao"))
        ds_s = _normalise(open_lazy(sal_dataset_id, "so"))
        log.info("Datasets opened in %.1fs", time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"Copernicus Marine connection failed: {exc}") from exc

    floats_in = argo_payload.get("floats", [])
    if not floats_in:
        raise ModelFetchError("No Argo floats supplied to sample the model at.")

    # ---- Build the flat list of (float_index, cycle) points to sample -----
    jobs = []  # (float_index, cycle_dict)
    for f_idx, f in enumerate(floats_in):
        for c in f.get("cycles", []):
            if c.get("lat") is None or c.get("lon") is None or c.get("time") is None:
                continue
            jobs.append((f_idx, c))

    if not jobs:
        raise ModelFetchError("No Argo cycles with lat/lon/time to sample the model at.")

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

    # ---- ONE vectorized interpolation call for every point at once --------
    # xarray treats indexers that share a dimension name as pointwise
    # ("advanced indexing"): this returns exactly len(jobs) profiles, not
    # the full outer-product grid of all lats x lons x times.
    log.info("Sampling %d Argo cycle(s) against the model in a single request...", len(jobs))
    t1 = time.monotonic()
    try:
        t_result = ds_t["thetao"].interp(latitude=lat_idx, longitude=lon_idx, time=time_idx, method="linear")
        s_result = ds_s["so"].interp(latitude=lat_idx, longitude=lon_idx, time=time_idx, method="linear")
        # .values is what actually triggers the network read (everything
        # above this line is still lazy).
        depths = np.atleast_1d(t_result["depth"].values).astype(float)
        temps_all = np.asarray(t_result.transpose(point_dim, "depth").values, dtype=float)
        sals_all = np.asarray(s_result.transpose(point_dim, "depth").values, dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise ModelFetchError(f"Copernicus Marine sampling failed: {exc}") from exc
    log.info("Sampling done in %.1fs", time.monotonic() - t1)

    # ---- Free the big lazy dataset handles + raw result arrays as soon as
    # we've pulled what we need out of them (a level dict per point below),
    # rather than waiting for Python's GC to get to it whenever it feels
    # like it. On a memory-constrained host (e.g. Render's free/Starter
    # 512MB tier) this is the difference between fitting in RAM and an OOM
    # kill, since t_result/s_result/temps_all/sals_all can each be sizeable
    # once the underlying remote chunks have actually been pulled in.
    del t_result, s_result, ds_t, ds_s
    gc.collect()

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

    del temps_all, sals_all
    gc.collect()

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
    args = parser.parse_args()

    if not args.argo_json.exists():
        raise SystemExit(f"{args.argo_json} not found — run fetch_incois_argo.py first.")
    argo_payload = json.loads(args.argo_json.read_text(encoding="utf-8"))

    try:
        payload = build_copernicus_model(
            argo_payload, args.min_lat, args.max_lat, args.min_lon, args.max_lon,
            args.start, args.end, args.min_depth, args.max_depth,
            args.dataset_id, args.sal_dataset_id, args.nc_dir, args.output,
        )
    except ModelFetchError as exc:
        raise SystemExit(f"FAILED: {exc}")

    print(f"SUCCESS: sampled {payload['meta']['sampled_cycles']} cycle(s) "
          f"across {payload['meta']['float_count']} float(s) -> {args.output.resolve()}")


if __name__ == "__main__":
    _cli()