"""
Parse Copernicus / HYCOM NetCDF data using xarray.

Input:
    data/copernicus/temperature.nc

Output:
    data/copernicus/parsed_temperature.csv
    data/copernicus/parsed_temperature.json

The script:
1. Opens the NetCDF file with xarray
2. Detects latitude/longitude/depth/time coordinates
3. Detects the ocean variable (thetao by default)
4. Prints dataset information
5. Selects a geographic/depth/time subset
6. Converts the selected xarray data to normal Python records
7. Saves CSV and JSON for further processing/frontend use
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/copernicus/temperature_subset.nc.2icw2ds0")

CSV_OUTPUT = Path("data/copernicus/parsed_temperature.csv")
JSON_OUTPUT = Path("data/copernicus/parsed_temperature.json")

# Geographic region
MIN_LON = 45
MAX_LON = 105

MIN_LAT = -5
MAX_LAT = 22

# Depth range in meters
MIN_DEPTH = 0
MAX_DEPTH = 1000

# Maximum number of points written to JSON/CSV.
# This prevents accidentally creating a huge file.
MAX_POINTS = 50000

# Variable to extract
VARIABLE = "thetao"


# ============================================================
# OPEN NETCDF FILE
# ============================================================

def open_netcdf(file_path: Path) -> xr.Dataset:

    if not file_path.exists():
        raise FileNotFoundError(
            f"NetCDF file not found:\n{file_path.resolve()}"
        )

    print("\nOpening NetCDF file...")
    print(f"File: {file_path.resolve()}")

    try:
        # Normal method
        ds = xr.open_dataset(
            file_path,
            decode_times=True
        )

    except Exception as error:
        print("\nNormal time decoding failed.")
        print(f"Reason: {error}")
        print("\nTrying again with decode_times=False...")

        # Fallback for unusual time units such as
        # "hours since analysis"
        ds = xr.open_dataset(
            file_path,
            decode_times=False
        )

    return ds


# ============================================================
# FIND COORDINATE NAMES
# ============================================================

def find_coordinate(ds: xr.Dataset, possible_names: list[str]) -> str | None:

    all_names = list(ds.coords) + list(ds.dims)

    # Exact match first
    for name in possible_names:
        if name in all_names:
            return name

    # Case-insensitive match
    lower_names = {
        name.lower(): name
        for name in all_names
    }

    for name in possible_names:
        if name.lower() in lower_names:
            return lower_names[name.lower()]

    return None


# ============================================================
# FIND OCEAN VARIABLE
# ============================================================

def find_variable(ds: xr.Dataset, requested: str) -> str:

    if requested in ds.data_vars:
        return requested

    # Common ocean variables
    candidates = [
        "thetao",
        "temperature",
        "temp",
        "so",
        "salinity",
        "uo",
        "vo"
    ]

    for variable in candidates:
        if variable in ds.data_vars:
            print(
                f"\nRequested variable '{requested}' was not found."
            )
            print(
                f"Using available variable: '{variable}'"
            )
            return variable

    if len(ds.data_vars) == 0:
        raise ValueError("No data variables were found in the NetCDF file.")

    # Last resort
    first_variable = list(ds.data_vars)[0]

    print(
        f"\nUsing first available data variable: "
        f"'{first_variable}'"
    )

    return first_variable


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

def print_dataset_info(ds: xr.Dataset):

    print("\n" + "=" * 70)
    print("NETCDF DATASET INFORMATION")
    print("=" * 70)

    print("\nDimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print("\nCoordinates:")
    for name in ds.coords:
        try:
            values = ds[name].values

            if values.size > 0:
                print(
                    f"  {name}: "
                    f"shape={values.shape}, "
                    f"first={values.flat[0]}, "
                    f"last={values.flat[-1]}"
                )
            else:
                print(f"  {name}: empty")

        except Exception:
            print(f"  {name}")

    print("\nData variables:")

    for name, variable in ds.data_vars.items():

        print(
            f"  {name}: "
            f"dims={variable.dims}, "
            f"shape={variable.shape}"
        )

        if "units" in variable.attrs:
            print(
                f"      units={variable.attrs['units']}"
            )

        if "long_name" in variable.attrs:
            print(
                f"      description={variable.attrs['long_name']}"
            )

    print("\n" + "=" * 70)


# ============================================================
# SELECT SUBSET
# ============================================================

def select_subset(
    ds: xr.Dataset,
    variable_name: str,
    lat_name: str | None,
    lon_name: str | None,
    depth_name: str | None,
    time_name: str | None,
) -> xr.DataArray:

    data = ds[variable_name]

    print("\nSelecting data...")

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    if lon_name:

        lon_values = ds[lon_name].values

        min_available_lon = float(np.nanmin(lon_values))
        max_available_lon = float(np.nanmax(lon_values))

        min_lon = max(MIN_LON, min_available_lon)
        max_lon = min(MAX_LON, max_available_lon)

        print(
            f"Longitude: {min_lon} -> {max_lon}"
        )

        # Handle ascending/descending coordinates
        if lon_values[0] < lon_values[-1]:
            data = data.sel(
                {lon_name: slice(min_lon, max_lon)}
            )
        else:
            data = data.sel(
                {lon_name: slice(max_lon, min_lon)}
            )

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    if lat_name:

        lat_values = ds[lat_name].values

        min_available_lat = float(np.nanmin(lat_values))
        max_available_lat = float(np.nanmax(lat_values))

        min_lat = max(MIN_LAT, min_available_lat)
        max_lat = min(MAX_LAT, max_available_lat)

        print(
            f"Latitude: {min_lat} -> {max_lat}"
        )

        if lat_values[0] < lat_values[-1]:
            data = data.sel(
                {lat_name: slice(min_lat, max_lat)}
            )
        else:
            data = data.sel(
                {lat_name: slice(max_lat, min_lat)}
            )

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    if depth_name:

        depth_values = ds[depth_name].values

        min_available_depth = float(
            np.nanmin(depth_values)
        )

        max_available_depth = float(
            np.nanmax(depth_values)
        )

        min_depth = max(
            MIN_DEPTH,
            min_available_depth
        )

        max_depth = min(
            MAX_DEPTH,
            max_available_depth
        )

        print(
            f"Depth: {min_depth} -> {max_depth} meters"
        )

        if depth_values[0] < depth_values[-1]:
            data = data.sel(
                {
                    depth_name:
                    slice(min_depth, max_depth)
                }
            )
        else:
            data = data.sel(
                {
                    depth_name:
                    slice(max_depth, min_depth)
                }
            )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if time_name:

        time_values = ds[time_name].values

        if len(time_values) > 0:

            print(
                f"Time points available: "
                f"{len(time_values)}"
            )

            # For this parser we take the first time step.
            # This keeps the exported file manageable.
            data = data.isel(
                {time_name: 0}
            )

            print(
                f"Selected first time step: "
                f"{time_values[0]}"
            )

    return data


# ============================================================
# REDUCE DATA SIZE
# ============================================================

def limit_points(data: xr.DataArray) -> xr.DataArray:

    total_points = data.size

    print(
        f"\nSelected data points: {total_points:,}"
    )

    if total_points <= MAX_POINTS:
        return data

    print(
        f"More than {MAX_POINTS:,} points found."
    )

    print("Downsampling data for export...")

    factor = int(
        np.ceil(
            total_points / MAX_POINTS
        )
    )

    print(
        f"Downsampling factor: {factor}"
    )

    # Reduce each dimension using isel
    indexers = {}

    for dimension in data.dims:

        size = data.sizes[dimension]

        indexers[dimension] = slice(
            0,
            size,
            factor
        )

    data = data.isel(indexers)

    print(
        f"Points after downsampling: "
        f"{data.size:,}"
    )

    return data


# ============================================================
# CONVERT XARRAY DATA TO RECORDS
# ============================================================

def convert_to_records(
    data: xr.DataArray,
    lat_name: str | None,
    lon_name: str | None,
    depth_name: str | None,
    time_name: str | None,
) -> list[dict]:

    print("\nConverting xarray data to records...")

    # Convert to DataFrame.
    # xarray handles the coordinate mapping for us.
    df = data.to_dataframe(
        name="value"
    ).reset_index()

    # Remove missing values
    df = df.dropna(
        subset=["value"]
    )

    records = []

    for _, row in df.iterrows():

        record = {}

        # Latitude
        if lat_name and lat_name in row:
            record["latitude"] = float(
                row[lat_name]
            )

        # Longitude
        if lon_name and lon_name in row:
            record["longitude"] = float(
                row[lon_name]
            )

        # Depth
        if depth_name and depth_name in row:
            record["depth"] = float(
                row[depth_name]
            )

        # Time
        if time_name and time_name in row:

            time_value = row[time_name]

            record["time"] = str(
                time_value
            )

        # Ocean value
        record["value"] = float(
            row["value"]
        )

        records.append(record)

    return records


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records: list[dict], output: Path):

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(records)

    df.to_csv(
        output,
        index=False
    )

    print(
        f"\nCSV saved to:\n{output.resolve()}"
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    records: list[dict],
    output: Path,
    variable_name: str,
    ds: xr.Dataset,
):

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "source_file": str(INPUT_FILE),
        "variable": variable_name,
        "units": ds[variable_name].attrs.get(
            "units",
            ""
        ),
        "long_name": ds[variable_name].attrs.get(
            "long_name",
            ""
        ),
        "number_of_points": len(records),
        "records": records,
    }

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            default=str
        )

    print(
        f"JSON saved to:\n{output.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("COPERNICUS / HYCOM NETCDF PARSER")
    print("=" * 70)

    # --------------------------------------------------------
    # Open dataset
    # --------------------------------------------------------

    ds = open_netcdf(
        INPUT_FILE
    )

    try:

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print_dataset_info(
            ds
        )

        # ----------------------------------------------------
        # Detect coordinates
        # ----------------------------------------------------

        lat_name = find_coordinate(
            ds,
            [
                "latitude",
                "lat",
                "nav_lat",
            ]
        )

        lon_name = find_coordinate(
            ds,
            [
                "longitude",
                "lon",
                "nav_lon",
            ]
        )

        depth_name = find_coordinate(
            ds,
            [
                "depth",
                "deptht",
                "lev",
                "level",
            ]
        )

        time_name = find_coordinate(
            ds,
            [
                "time",
                "time_counter",
                "datetime",
            ]
        )

        print("\nDetected coordinates:")

        print(
            f"  Latitude : {lat_name}"
        )

        print(
            f"  Longitude: {lon_name}"
        )

        print(
            f"  Depth    : {depth_name}"
        )

        print(
            f"  Time     : {time_name}"
        )

        # ----------------------------------------------------
        # Find variable
        # ----------------------------------------------------

        variable_name = find_variable(
            ds,
            VARIABLE
        )

        print(
            f"\nVariable selected: "
            f"{variable_name}"
        )

        # ----------------------------------------------------
        # Select subset
        # ----------------------------------------------------

        data = select_subset(
            ds,
            variable_name,
            lat_name,
            lon_name,
            depth_name,
            time_name,
        )

        # ----------------------------------------------------
        # Limit data
        # ----------------------------------------------------

        data = limit_points(
            data
        )

        # ----------------------------------------------------
        # Convert to records
        # ----------------------------------------------------

        records = convert_to_records(
            data,
            lat_name,
            lon_name,
            depth_name,
            time_name,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_csv(
            records,
            CSV_OUTPUT
        )

        save_json(
            records,
            JSON_OUTPUT,
            variable_name,
            ds,
        )

        # ----------------------------------------------------
        # Display sample
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("SAMPLE EXTRACTED DATA")
        print("=" * 70)

        for record in records[:10]:
            print(record)

        print("\n")
        print("=" * 70)
        print("PARSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            f"\nTotal records exported: "
            f"{len(records):,}"
        )

    finally:

        # Always close the NetCDF file
        ds.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()