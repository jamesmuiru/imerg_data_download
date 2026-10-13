"""One-call API for NASA IMERG download, monthly Zarr conversion, and reading."""
from pathlib import Path
import logging
import xarray as xr
from imerg_downloader import download_all
from imerg_zarr_pipeline import convert_imerg_to_monthly_zarr, read_imerg_zarr

logger = logging.getLogger(__name__)


def get_imerg(start_date, end_date, bbox, output_dir, dataset_name="GPM_3IMERGHH",
              max_workers=4, convert_to_zarr=True, variables=None, attrs_path=None,
              time_chunk=96, lat_chunk=200, lon_chunk=200, compressor=None,
              batch_size=48, zarr_workers=None, memory_limit_gb=None,
              overwrite_zarr=False):
    root = Path(output_dir).expanduser(); root.mkdir(parents=True, exist_ok=True)
    summary = download_all(dataset_name, start_date, end_date, bbox, str(root), max_workers)
    hdf5_dir = root / dataset_name
    result = {"status":"failed", "dataset_name":dataset_name, "start_date":str(start_date),
        "end_date":str(end_date), "bbox":tuple(map(float,bbox)), "hdf5_directory":str(hdf5_dir),
        "download_summary":summary, "zarr_directory":None, "zarr_stores":[],
        "validation_passed":False, "errors":[]}
    if summary.get("failed_months"):
        result["errors"].append(f"Download failed for month(s): {summary['failed_months']}")
        raise RuntimeError(f"IMERG download failed for month(s): {summary['failed_months']}")
    if summary.get("total_local_files", 0) == 0:
        raise RuntimeError("No local IMERG HDF5 granules are available; cannot create final Zarr outputs")
    if not convert_to_zarr:
        result["status"] = "downloaded_only"
        return result
    zarr_dir = root / "zarr"
    stores = convert_imerg_to_monthly_zarr(input_dir=str(hdf5_dir), output_dir=str(zarr_dir),
        bbox=tuple(map(float,bbox)), variables=variables, attrs_path=attrs_path,
        time_chunk=time_chunk, lat_chunk=lat_chunk, lon_chunk=lon_chunk,
        compressor=compressor, batch_size=batch_size, max_workers=zarr_workers,
        memory_limit_gb=memory_limit_gb, overwrite=overwrite_zarr,
        start_date=start_date, end_date=end_date)
    if not stores: raise RuntimeError("Conversion returned no Zarr stores")
    # Validate the requested range can actually be read from the final stores.
    sample = read_imerg_zarr(str(zarr_dir), start_date, end_date,
                             variables=variables, compute=False)
    try:
        if sample.sizes.get("time", 0) == 0: raise RuntimeError("Zarr validation found zero timestamps")
        if not sample.data_vars: raise RuntimeError("Zarr validation found no data variables")
    finally: sample.close()
    result.update({"status":"success", "zarr_directory":str(zarr_dir),
                   "zarr_stores":stores, "validation_passed":True})
    return result


def read_imerg(output_dir, start_date, end_date, variables=None, compute=True):
    root = Path(output_dir) / "zarr"
    if not root.exists(): root = Path(output_dir)
    return read_imerg_zarr(str(root), start_date, end_date, variables=variables, compute=compute)
