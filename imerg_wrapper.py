"""One-call interface for downloading IMERG and converting it to monthly Zarr stores.

Chains imerg_downloader.download_all() → imerg_zarr_pipeline.convert_imerg_to_monthly_zarr()
and exposes a single get_imerg() function plus a convenience read_imerg() reader.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from imerg_downloader import download_all
from imerg_zarr_pipeline import (
    IMERGProcessor,
    convert_imerg_to_monthly_zarr,
    read_imerg_zarr,
    read_imerg_zarr_all,
)
import xarray as xr

logger = logging.getLogger(__name__)


def get_imerg(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, ...],
    output_dir: str,
    dataset_name: str = "GPM_3IMERGHH",
    # Download options
    max_workers: int = 4,
    # Zarr conversion options
    convert_to_zarr: bool = True,
    variables: Optional[List[str]] = None,
    attrs_path: Optional[str] = None,
    time_chunk: int = 96,
    lat_chunk: int = 200,
    lon_chunk: int = 200,
    compressor=None,
    batch_size: int = 48,
    zarr_workers: Optional[int] = None,
    memory_limit_gb: Optional[float] = None,
    overwrite_zarr: bool = False,
) -> dict:
    """Download IMERG granules and optionally convert them to monthly Zarr stores.

    Parameters
    ----------
    start_date       : "YYYY-MM-DD" start of the requested period.
    end_date         : "YYYY-MM-DD" end of the requested period (inclusive).
    bbox             : (west, south, east, north) bounding box in decimal degrees.
    output_dir       : Root directory for all outputs.
    dataset_name     : NASA CMR short name (default "GPM_3IMERGHH").
    max_workers      : Parallel download threads (one per month).
    convert_to_zarr  : Also run the HDF5 → Zarr pipeline after downloading.
    variables        : HDF5 variable names to include (None = all variables).
    attrs_path       : Path to a JSON file with variable metadata attributes.
    time_chunk       : Zarr time chunk (default 96 = 2 days).
    lat_chunk        : Zarr latitude chunk size.
    lon_chunk        : Zarr longitude chunk size.
    compressor       : numcodecs compressor (default: Blosc lz4 if available).
    batch_size       : HDF5 files per processing batch.
    zarr_workers     : Parallel workers for HDF5 reading (default: cpu_count - 1).
    memory_limit_gb  : Soft RSS memory cap in GB; skips months if exceeded.
    overwrite_zarr   : Re-process months even if their Zarr store already exists.

    Returns
    -------
    dict with keys:
        dataset_name, start_date, end_date, bbox,
        hdf5_directory, download_summary,
        zarr_directory (or None), zarr_stores (list of paths).
    """
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ───────────────────────────────────────────────────── #
    logger.info("=== Step 1: Downloading IMERG granules ===")
    download_summary = download_all(
        dataset_name=dataset_name,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        base_dir=str(root),
        max_workers=max_workers,
    )

    hdf5_dir = root / dataset_name
    result = {
        "dataset_name":    dataset_name,
        "start_date":      str(start_date),
        "end_date":        str(end_date),
        "bbox":            tuple(bbox),
        "hdf5_directory":  str(hdf5_dir),
        "download_summary": download_summary,
        "zarr_directory":  None,
        "zarr_stores":     [],
    }

    # ── 2. Convert to Zarr ────────────────────────────────────────────── #
    if convert_to_zarr:
        logger.info("=== Step 2: Converting HDF5 → monthly Zarr ===")
        zarr_dir = root / "zarr"
        stores = convert_imerg_to_monthly_zarr(
            input_dir=str(hdf5_dir),
            output_dir=str(zarr_dir),
            bbox=tuple(map(float, bbox)),
            variables=variables,
            attrs_path=attrs_path,
            time_chunk=time_chunk,
            lat_chunk=lat_chunk,
            lon_chunk=lon_chunk,
            compressor=compressor,
            batch_size=batch_size,
            max_workers=zarr_workers,
            memory_limit_gb=memory_limit_gb,
            overwrite=overwrite_zarr,
        )
        result["zarr_directory"] = str(zarr_dir)
        result["zarr_stores"]    = stores

    return result


def read_imerg(
    output_dir: str,
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
    compute: bool = True,
) -> xr.Dataset:
    """Convenience wrapper: read processed IMERG Zarr stores for a date range.

    Parameters
    ----------
    output_dir  : Directory that contains the zarr/ sub-folder written by get_imerg().
    start_date  : "YYYY-MM-DD".
    end_date    : "YYYY-MM-DD".
    variables   : Optional variable whitelist.
    compute     : Load arrays into memory if True (default).

    Returns
    -------
    xr.Dataset with dimensions (time, lat, lon).
    """
    zarr_dir = Path(output_dir) / "zarr"
    if not zarr_dir.exists():
        # Fall back: treat output_dir itself as the Zarr root.
        zarr_dir = Path(output_dir)
    return read_imerg_zarr(
        str(zarr_dir),
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        compute=compute,
    )
