"""SagarDrishti backend.

Serves index.html / explorer.html and exposes one endpoint the frontend
calls whenever the user picks a location (region box or map-click point)
and date range and clicks "Fetch live data":

    POST /api/fetch-live-data
    {
      "minLat": 5, "maxLat": 15, "minLon": 80, "maxLon": 90,
      "start": "2025-01-01", "end": "2025-01-07",
      "minDepth": 0.5, "maxDepth": 1000
    }

AUTOMATIC TWO-STEP FETCH
------------------------
As soon as a location + date range comes in, this endpoint ALWAYS runs
both data sources for that exact location, back to back, in one request
— the caller does not need to ask for the model separately:

  1. Fetch_incois_argo.fetch_argo()
         -> real Argo float observations for the box/time window
         -> written to <frontend>/argo_data.json

  2. Capernicus_model.build_copernicus_model()
         -> runs automatically right after step 1, using the SAME
            box/date window and the float cycles step 1 just returned
         -> samples the Copernicus Marine model at every one of those
            cycles' exact lat/lon/time
         -> written to <frontend>/copernicus_model.json

Both payloads are returned in the same response so the frontend can
render Argo + model data together without a second round trip.

Step 2 depends on step 1's output (it samples the model at each real
Argo cycle position), so the two calls are sequential, not parallel —
but from the caller's point of view this is one action: submit a
location, get both datasets back.

If the Copernicus fetch fails (no `copernicusmarine login`, network
issue, no model coverage for that box/date, etc.) the endpoint does
NOT fail the whole request — it still returns the real Argo data and
reports the model failure in `modelWarning`, so the frontend can fall
back to its synthetic offset instead of showing nothing.

An optional `"includeModel": false` in the request body can still be
sent to skip step 2 on purpose (e.g. for a fast Argo-only preview) —
but the default, and the normal path, is that both run automatically.

Run:
    pip install flask flask-cors pandas requests xarray netCDF4 numpy copernicusmarine
    python server.py
    open http://localhost:8000/index.html
"""
from __future__ import annotations

# Load a local .env file (if present) into environment variables BEFORE
# anything else runs, so COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD are
# already set by the time Capernicus_model needs them — no login prompt,
# no re-entering credentials on every restart. Safe to keep even in
# production: if there's no .env file (e.g. credentials come from the
# hosting platform's own environment/secrets instead), this just no-ops.
from dotenv import load_dotenv
load_dotenv()

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from Capernicus_model import ModelFetchError, build_copernicus_model
from Fetch_incois_argo import ArgoFetchError, fetch_argo

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("sagardrishti")

# ---------------------------------------------------------------------------
# Frontend directory resolution.
#
# Works with either layout:
#   backend/server.py  +  frontend/index.html   (sibling folders)
#   server.py + index.html in the same flat folder
# so this doesn't silently 404 / write JSON to the wrong place just
# because the project wasn't split into backend/frontend subfolders.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SIBLING_FRONTEND = _HERE.parent / "frontend"
FRONTEND_DIR = _SIBLING_FRONTEND if (_SIBLING_FRONTEND / "index.html").exists() else _HERE
DATA_DIR = _HERE / "data"

log.info("Serving frontend from: %s", FRONTEND_DIR)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


def _iso(date_str: str, end_of_day: bool = False) -> str:
    """Turn a <input type=date> value ('2025-01-07') into the
    'YYYY-MM-DDTHH:MM:SSZ' shape the ERDDAP/Copernicus APIs expect."""
    suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
    return f"{date_str}{suffix}"


@app.errorhandler(Exception)
def handle_any_uncaught_error(exc):
    """Safety net: if ANYTHING raises inside a route that isn't already
    caught below, this guarantees the browser still gets back valid JSON
    with a real error message instead of a dead/empty connection — which
    is what causes fetch()'s res.json() to fail with
    'Unexpected end of JSON input' on the frontend.

    IMPORTANT: normal HTTP errors (404 for a missing static file like
    copernicus_model.json before the first fetch, or favicon.ico, 405 for
    a wrong method, etc.) are werkzeug HTTPExceptions — they are NOT bugs,
    and must be left to render as their real status code. Only genuinely
    unexpected exceptions get converted to a JSON 500 here.
    Full traceback still goes to the terminal via log.exception."""
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    log.exception("Unhandled error while serving %s", request.path)
    return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/api/fetch-live-data", methods=["POST"])
def fetch_live_data():
    body = request.get_json(force=True, silent=True) or {}

    try:
        min_lat = float(body["minLat"])
        max_lat = float(body["maxLat"])
        min_lon = float(body["minLon"])
        max_lon = float(body["maxLon"])
        start_date = str(body["start"])
        end_date = str(body["end"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "minLat, maxLat, minLon, maxLon, start and end are required."}), 400

    min_depth = float(body.get("minDepth", 0.5))
    max_depth = float(body.get("maxDepth", 1000))
    # Both sources run automatically by default. This flag only exists as
    # an escape hatch to skip the (slower) model fetch on purpose.
    include_model = bool(body.get("includeModel", True))

    start_iso = _iso(start_date, end_of_day=False)
    end_iso = _iso(end_date, end_of_day=True)

    location_label = f"[{min_lat},{min_lon}] to [{max_lat},{max_lon}] over {start_date}..{end_date}"
    log.info("Location received: %s — starting automatic Argo + model fetch", location_label)

    # ---- Step 1: INCOIS Argo (always runs) ----------------------------
    try:
        log.info("Step 1/2: fetching INCOIS Argo floats for %s", location_label)
        argo_payload = fetch_argo(
            min_lat, max_lat, min_lon, max_lon, start_iso, end_iso,
            output_dir=DATA_DIR / "incois_argo",
            frontend_json_path=FRONTEND_DIR / "argo_data.json",
        )
        log.info(
            "Step 1/2 done: %d float(s), %d row(s)",
            argo_payload["meta"]["float_count"], argo_payload["meta"]["row_count"],
        )
    except ArgoFetchError as exc:
        log.warning("Step 1/2 failed: %s", exc)
        return jsonify({"error": f"Argo fetch failed: {exc}"}), 502

    # ---- Step 2: Copernicus Marine model, sampled at the Argo cycles --
    # Runs automatically right after step 1, for the same location, with
    # no extra action needed from the caller. A failure here does not
    # take down the response — Argo data still goes back to the frontend.
    model_payload = None
    model_warning = None
    if include_model:
        try:
            log.info("Step 2/2: sampling Copernicus Marine model at the same location")
            model_payload = build_copernicus_model(
                argo_payload, min_lat, max_lat, min_lon, max_lon,
                start_iso, end_iso, min_depth, max_depth,
                nc_dir=DATA_DIR / "copernicus",
                frontend_json_path=FRONTEND_DIR / "copernicus_model.json",
            )
            log.info(
                "Step 2/2 done: sampled %d cycle(s) across %d float(s)",
                model_payload["meta"]["sampled_cycles"], model_payload["meta"]["float_count"],
            )
        except ModelFetchError as exc:
            # Argo data is still good — let the frontend fall back to the
            # synthetic model offset and say why.
            model_warning = str(exc)
            log.warning("Step 2/2 failed (Argo data still returned): %s", model_warning)
    else:
        log.info("Step 2/2 skipped: includeModel=false was explicitly set in the request")

    return jsonify({
        "argo": argo_payload,
        "model": model_payload,
        "modelWarning": model_warning,
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Hosting platforms (Render, Railway, Heroku, etc.) assign a port at
    # runtime via the PORT env var and expect the app to bind to it —
    # they don't let you hardcode 8000. Falls back to 8000 for local dev.
    port = int(os.environ.get("PORT", 8000))
    # Never run with debug=True in production: it exposes a Python
    # debugger/console to anyone who can trigger an error on your public
    # URL. Set FLASK_DEBUG=true locally if you want it; it defaults to
    # off everywhere else.
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    # threaded=True: a live-data request can legitimately take a while
    # (Copernicus Marine subsets aren't instant) — without this, the dev
    # server can only handle one request at a time.
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)