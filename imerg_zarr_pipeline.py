"""Convert downloaded IMERG HDF5 granules into cropped monthly Zarr stores."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import logging
import re

import h5py
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# IMERG granule names commonly contain e.g. 20250107-S040000-E042959.
_FILENAME_TIME = re.compile(r"\.(\d{8})-S(\d{6})-E(\d{6})\.")


def _time_from_filename(path: Path):
    match = _FILENAME_TIME.search(path.name)
    if not match:
        return None
    day, start_clock, _ = match.groups()
    return pd.Timestamp(datetime.strptime(day + start_clock, "%Y%m%d%H%M%S"))


def _find_dataset(h5, candidate_paths):
    for path in candidate_paths:
        if path in h5:
            return h5[path]
    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.rsplit("/", 1)[-1].lower() in {p.rsplit("/", 1)[-1].lower() for p in candidate_paths}:
            found.append(obj)
    h5.visititems(visitor)
    return found[0] if found else None


def _read_granule(path: Path, bbox):
    """Read one granule, crop it to bbox, and return (timestamp, lat, lon, rainfall)."""
    with h5py.File(path, "r") as h5:
        lat_ds = _find_dataset(h5, ["Grid/lat", "Grid/latitude", "lat", "latitude"])
        lon_ds = _find_dataset(h5, ["Grid/lon", "Grid/longitude", "lon", "longitude"])
        rain_ds = _find_dataset(h5, ["Grid/precipitationCal", "precipitationCal"])
        time_ds = _find_dataset(h5, ["Grid/time", "time"])
        if lat_ds is None or lon_ds is None or rain_ds is None:
            raise ValueError(f"Could not find latitude, longitude, and precipitationCal in {path.name}")

        lat = np.asarray(lat_ds[...]).squeeze()
        lon = np.asarray(lon_ds[...]).squeeze()
        rain = np.asarray(rain_ds[...]).squeeze()
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError(f"Expected 1-D latitude/longitude arrays in {path.name}")
        if rain.ndim != 2:
            raise ValueError(f"Expected a single 2-D precipitation field in {path.name}; got shape {rain.shape}")

        # Identify the rainfall array's axes by matching them to coordinate lengths.
        if rain.shape == (len(lat), len(lon)):
            pass
        elif rain.shape == (len(lon), len(lat)):
            rain = rain.T
        else:
            raise ValueError(f"Rainfall shape {rain.shape} does not match lat/lon lengths in {path.name}")

        west, south, east, north = bbox
        lat_idx = np.flatnonzero((lat >= south) & (lat <= north))
        lon_idx = np.flatnonzero((lon >= west) & (lon <= east))
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            return None
        rain = rain[np.ix_(lat_idx, lon_idx)].astype(np.float32, copy=False)
        lat = lat[lat_idx].astype(np.float32, copy=False)
        lon = lon[lon_idx].astype(np.float32, copy=False)

        timestamp = _time_from_filename(path)
        if timestamp is None and time_ds is not None:
            raw_time = np.asarray(time_ds[...]).squeeze()
            units = time_ds.attrs.get("units", "seconds since 1980-01-06 00:00:00")
            if isinstance(units, bytes):
                units = units.decode("utf-8", errors="ignore")
            try:
                unit, _, origin = units.partition(" since ")
                origin = pd.Timestamp(origin.strip())
                factor = {"second": "s", "seconds": "s", "minute": "m", "minutes": "m", "hour": "h", "hours": "h"}.get(unit.lower())
                if factor is not None:
                    timestamp = origin + pd.to_timedelta(float(raw_time), unit=factor)
            except Exception:
                timestamp = None
        if timestamp is None:
            raise ValueError(f"Could not determine observation time for {path.name}")
        return pd.Timestamp(timestamp), lat, lon, rain


def convert_imerg_to_monthly_zarr(input_dir, output_dir, bbox=None,
                                  time_chunk=96, lat_chunk=200, lon_chunk=200,
                                  overwrite=False):
    """Create one Zarr store per calendar month.

    Args:
        input_dir: Directory containing downloaded HDF5 granules (searched recursively).
        output_dir: Destination directory for monthly .zarr stores.
        bbox: Optional (west, south, east, north). If None, use the full grid.
        time_chunk: Number of half-hourly steps per Zarr time chunk (96 = 2 days).
        lat_chunk/lon_chunk: Spatial chunk sizes.
        overwrite: Replace existing monthly Zarr stores if True.

    Returns a list of written Zarr store paths.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    granules = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"})
    if not granules:
        raise FileNotFoundError(f"No .H5/.HDF5 files found under {input_dir}")

    monthly = defaultdict(list)
    reference_lat = reference_lon = None
    failures = []
    for i, path in enumerate(granules, 1):
        try:
            item = _read_granule(path, bbox)
            if item is None:
                logger.warning("Skipping %s: bbox does not overlap its coordinate grid", path.name)
                continue
            timestamp, lat, lon, rain = item
            if reference_lat is None:
                reference_lat, reference_lon = lat, lon
            elif not (np.array_equal(reference_lat, lat) and np.array_equal(reference_lon, lon)):
                raise ValueError("Grid coordinates differ between granules; cannot combine into one monthly store")
            monthly[timestamp.strftime("%Y-%m")].append((timestamp, rain, path.name))
            if i % 100 == 0:
                logger.info("Read %s/%s granules", i, len(granules))
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
            logger.exception("Failed to read %s", path)

    written = []
    for month in sorted(monthly):
        rows = sorted(monthly[month], key=lambda row: row[0])
        times = pd.DatetimeIndex([row[0] for row in rows])
        data = np.stack([row[1] for row in rows], axis=0)
        ds = xr.Dataset(
            data_vars={"precipitationCal": (("time", "lat", "lon"), data,
                         {"long_name": "IMERG calibrated precipitation estimate", "units": "mm/hr"})},
            coords={"time": times, "lat": reference_lat, "lon": reference_lon},
            attrs={"source": "NASA GPM IMERG", "product": "GPM_3IMERGHH Version 07",
                   "bbox": "full grid" if bbox is None else str(tuple(bbox)),
                   "time_step": "half-hourly", "input_directory": str(input_dir)},
        )
        store = output_dir / f"imerg_{month}.zarr"
        if store.exists():
            if not overwrite:
                logger.warning("Skipping existing store %s (set overwrite=True to replace)", store)
                written.append(str(store))
                continue
            import shutil
            shutil.rmtree(store)
        chunks = {"time": max(1, int(time_chunk)), "lat": max(1, int(lat_chunk)), "lon": max(1, int(lon_chunk))}
        ds = ds.chunk(chunks)
        try:
            ds.to_zarr(store, mode="w", consolidated=True)
        except Exception:
            # Compatibility fallback for older Zarr/xarray combinations.
            ds.to_zarr(store, mode="w", consolidated=False)
        logger.info("Wrote %s: shape=%s", store, data.shape)
        written.append(str(store))

    if failures:
        failure_csv = output_dir / "conversion_failures.csv"
        pd.DataFrame(failures).to_csv(failure_csv, index=False)
        logger.warning("%s granules failed; details saved to %s", len(failures), failure_csv)
    return written


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--time-chunk", type=int, default=96)
    parser.add_argument("--lat-chunk", type=int, default=200)
    parser.add_argument("--lon-chunk", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    bbox = tuple(args.bbox) if args.bbox else None
    print(convert_imerg_to_monthly_zarr(args.input_dir, args.output_dir, bbox,
          args.time_chunk, args.lat_chunk, args.lon_chunk, args.overwrite))
