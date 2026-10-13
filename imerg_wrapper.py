"""One-call interface for downloading IMERG and converting it to monthly Zarr."""
from pathlib import Path
import logging

from imerg_downloader import download_all
from imerg_zarr_pipeline import convert_imerg_to_monthly_zarr

logger = logging.getLogger(__name__)


def get_imerg(start_date, end_date, bbox, output_dir,
              dataset_name="GPM_3IMERGHH", convert_to_zarr=True,
              max_workers=4, time_chunk=96, lat_chunk=200, lon_chunk=200,
              overwrite_zarr=False):
    """Download IMERG granules and optionally create monthly Zarr stores.

    bbox must be (west, south, east, north). Dates are YYYY-MM-DD.
    The Zarr converter crops the global IMERG grid to bbox.
    """
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

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
        "dataset_name": dataset_name,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "bbox": tuple(bbox),
        "hdf5_directory": str(hdf5_dir),
        "download_summary": download_summary,
        "zarr_directory": None,
        "zarr_stores": [],
    }
    if convert_to_zarr:
        zarr_dir = root / "zarr"
        stores = convert_imerg_to_monthly_zarr(
            input_dir=str(hdf5_dir),
            output_dir=str(zarr_dir),
            bbox=tuple(map(float, bbox)),
            time_chunk=time_chunk,
            lat_chunk=lat_chunk,
            lon_chunk=lon_chunk,
            overwrite=overwrite_zarr,
        )
        result["zarr_directory"] = str(zarr_dir)
        result["zarr_stores"] = stores
    return result
