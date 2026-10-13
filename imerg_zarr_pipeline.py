"""Convert NASA IMERG HDF5 granules into monthly, validated Zarr v2 stores."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json, logging, re, shutil
import h5py
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)
_TIME_RE = re.compile(r"\.(\d{8})-S(\d{6})-E(\d{6})\.")
_EXCLUDED = {"lat", "lon", "time", "nv", "lonv", "latv", "time_bnds", "lon_bnds", "lat_bnds"}


def _timestamp(path, grid):
    match = _TIME_RE.search(path.name)
    if match:
        day, clock, _ = match.groups()
        return pd.Timestamp(datetime.strptime(day + clock, "%Y%m%d%H%M%S"))
    if "time" in grid:
        ds = grid["time"]
        value = float(np.asarray(ds[...]).reshape(-1)[0])
        units = ds.attrs.get("units", "seconds since 1980-01-06 00:00:00")
        if isinstance(units, bytes): units = units.decode(errors="ignore")
        unit, _, origin = str(units).partition(" since ")
        factor = {"second":"s", "seconds":"s", "minute":"m", "minutes":"m", "hour":"h", "hours":"h"}.get(unit.lower())
        if not factor: raise ValueError(f"Unsupported time units {units!r} in {path.name}")
        return pd.Timestamp(origin.strip()) + pd.to_timedelta(value, unit=factor)
    raise ValueError(f"Could not determine timestamp for {path.name}")


def _decode_array(dataset):
    arr = np.asarray(dataset[...])
    # Decode common HDF5 missing-value conventions before conversion.
    for key in ("_FillValue", "missing_value", "CodeMissingValue"):
        value = dataset.attrs.get(key)
        if value is not None:
            try: arr = np.where(arr == value, np.nan, arr)
            except (TypeError, ValueError): pass
    scale = dataset.attrs.get("scale_factor")
    offset = dataset.attrs.get("add_offset", 0)
    if scale is not None: arr = arr.astype(np.float32) * float(scale) + float(offset)
    return arr


def _read_granule(path, bbox=None, variables=None, attrs_path=None):
    with h5py.File(path, "r") as h5:
        grid = h5["Grid"] if "Grid" in h5 else h5
        if not all(k in grid for k in ("lat", "lon")):
            raise ValueError("Could not find Grid/lat and Grid/lon")
        lat = np.asarray(grid["lat"][...]).squeeze(); lon = np.asarray(grid["lon"][...]).squeeze()
        if lat.ndim != 1 or lon.ndim != 1: raise ValueError("Latitude/longitude must be 1-D")
        ts = _timestamp(path, grid)
        requested = set(variables) if variables else None
        candidates = [k for k in grid.keys() if k not in _EXCLUDED and isinstance(grid[k], h5py.Dataset)]
        if requested is not None:
            missing = requested.difference(candidates)
            if missing: raise ValueError(f"Requested variable(s) absent from {path.name}: {sorted(missing)}")
            candidates = [k for k in candidates if k in requested]
        fields = {}
        for name in candidates:
            raw = _decode_array(grid[name]).squeeze()
            if raw.ndim != 2: continue
            if raw.shape == (len(lat), len(lon)): pass
            elif raw.shape == (len(lon), len(lat)): raw = raw.T
            else: continue
            fields[name] = raw
        if not fields:
            raise ValueError("No 2-D gridded variables could be read")
        if "precipitationCal" not in fields and variables is None:
            raise ValueError("Grid/precipitationCal was not found; check this granule's structure")
        if bbox is not None:
            west, south, east, north = map(float, bbox)
            yi = np.flatnonzero((lat >= south) & (lat <= north)); xi = np.flatnonzero((lon >= west) & (lon <= east))
            if not len(yi) or not len(xi): return None
            lat, lon = lat[yi], lon[xi]
            fields = {k: v[np.ix_(yi, xi)] for k, v in fields.items()}
        var_attrs = {}
        for name in fields:
            var_attrs[name] = {"long_name": name}
            if name == "precipitationCal": var_attrs[name]["units"] = "mm/hr"
        if attrs_path and Path(attrs_path).exists():
            with open(attrs_path, encoding="utf-8") as f: external = json.load(f)
            for name in fields:
                if name in external: var_attrs[name].update(external[name])
        return ts, lat, lon, fields, var_attrs


def convert_imerg_to_monthly_zarr(input_dir, output_dir, years=None, bbox=None,
        variables=None, attrs_path=None, time_chunk=96, lat_chunk=200, lon_chunk=200,
        compressor=None, batch_size=48, max_workers=None, memory_limit_gb=None,
        overwrite=False, start_date=None, end_date=None):
    """Create monthly Zarr stores; optional start/end dates are inclusive calendar days."""
    src, dst = Path(input_dir), Path(output_dir)
    if not src.exists(): raise FileNotFoundError(f"Input directory not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).normalize() if start_date else None
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if end_date else None
    granules = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"})
    if years is not None: granules = [p for p in granules if any(part in set(map(str, years)) for part in p.parts)]
    if not granules: raise FileNotFoundError(f"No HDF5 granules found under {src}")
    monthly = defaultdict(list); failures = []; ref_lat = ref_lon = None; var_attrs = {}
    for i, path in enumerate(granules, 1):
        try:
            # Fast filename filtering prevents parsing granules outside the requested dates.
            match = _TIME_RE.search(path.name)
            if match:
                day, clock, _ = match.groups(); ts_hint = pd.Timestamp(datetime.strptime(day + clock, "%Y%m%d%H%M%S"))
                if start is not None and ts_hint < start: continue
                if end is not None and ts_hint > end: continue
            item = _read_granule(path, bbox, variables, attrs_path)
            if item is None: continue
            ts, lat, lon, fields, attrs = item
            if start is not None and ts < start: continue
            if end is not None and ts > end: continue
            if ref_lat is None: ref_lat, ref_lon = lat, lon
            elif not (np.array_equal(ref_lat, lat) and np.array_equal(ref_lon, lon)):
                raise ValueError("Grid coordinates differ between granules")
            monthly[ts.strftime("%Y-%m")].append((ts, fields, path.name)); var_attrs.update(attrs)
        except Exception as exc:
            failures.append({"file": str(path), "error": str(exc)})
            logger.exception("Could not process %s", path.name)
    if failures:
        pd.DataFrame(failures).to_csv(dst / "conversion_failures.csv", index=False)
    if not monthly:
        raise RuntimeError(f"No granules were successfully converted. See {dst / 'conversion_failures.csv'} if present")
    stores = []
    for month, rows in sorted(monthly.items()):
        rows.sort(key=lambda r: r[0])
        # Deduplicate timestamps deterministically: keep the first granule in sorted order.
        unique = {}
        for row in rows: unique.setdefault(row[0], row)
        rows = [unique[t] for t in sorted(unique)]
        names = sorted(set.intersection(*(set(row[1]) for row in rows)))
        data_vars = {}
        for name in names:
            data = np.stack([row[1][name] for row in rows], axis=0)
            data_vars[name] = (("time", "lat", "lon"), data, var_attrs.get(name, {}))
        ds = xr.Dataset(data_vars=data_vars, coords={"time": pd.DatetimeIndex([r[0] for r in rows]), "lat": ref_lat, "lon": ref_lon},
            attrs={"source":"NASA GPM IMERG", "product":"GPM_3IMERGHH Version 07", "bbox":str(bbox), "time_step":"half-hourly"})
        store = dst / f"imerg_{month}.zarr"
        if store.exists() and not overwrite:
            try:
                with xr.open_zarr(store) as old:
                    if "time" in old.coords and len(old.time) > 0:
                        logger.info("Keeping existing readable store %s", store); stores.append(str(store)); continue
            except Exception: logger.warning("Existing store is invalid; replacing %s", store)
        if store.exists(): shutil.rmtree(store)
        chunks = {"time": min(max(1, int(time_chunk)), len(ds.time)), "lat": min(max(1, int(lat_chunk)), len(ds.lat)), "lon": min(max(1, int(lon_chunk)), len(ds.lon))}
        ds = ds.chunk(chunks)
        # requirements pin Zarr v2 for stable numcodecs/xarray compatibility.
        ds.to_zarr(store, mode="w", consolidated=True, zarr_format=2)
        with xr.open_zarr(store) as check:
            if not check.data_vars or not check.sizes.get("time", 0): raise RuntimeError(f"Validation failed: empty Zarr store {store}")
            if not np.all(np.diff(check.time.values) > np.timedelta64(0, "s")) and len(check.time) > 1:
                raise RuntimeError(f"Validation failed: timestamps not strictly increasing in {store}")
            if bbox is not None:
                west, south, east, north = bbox
                if float(check.lon.min()) < west or float(check.lon.max()) > east or float(check.lat.min()) < south or float(check.lat.max()) > north:
                    raise RuntimeError(f"Validation failed: coordinates outside bbox in {store}")
        stores.append(str(store)); logger.info("Validated %s with %d timestamps", store, len(rows))
    if failures and not stores: raise RuntimeError("All granule conversions failed")
    return stores


def read_imerg_zarr(output_dir, start_date, end_date, variables=None, compute=True):
    root = Path(output_dir)
    stores = sorted(root.rglob("*.zarr")) if root.exists() else []
    if not stores: raise FileNotFoundError(f"No Zarr stores found under {root}")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    datasets = []
    for store in stores:
        ds = xr.open_zarr(store)
        if "time" not in ds.coords or ds.time.size == 0:
            ds.close(); continue
        if ds.time.max().values < np.datetime64(start) or ds.time.min().values > np.datetime64(end):
            ds.close(); continue
        if variables:
            missing = set(variables) - set(ds.data_vars)
            if missing: ds.close(); raise KeyError(f"Variables not found in {store}: {sorted(missing)}")
            ds = ds[variables]
        datasets.append(ds.sel(time=slice(start, end)))
    if not datasets: raise ValueError(f"No Zarr data found for {start_date} to {end_date}")
    combined = xr.concat(datasets, dim="time").sortby("time")
    _, idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=np.sort(idx))
    return combined.load() if compute else combined
