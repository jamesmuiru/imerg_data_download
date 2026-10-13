"""Example usage in Google Colab or a Python environment."""
from imerg_wrapper import get_imerg, read_imerg

result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-03",
    bbox=(36.0, -1.5, 38.0, 0.5),  # west, south, east, north
    output_dir="/content/IMERG",
    dataset_name="GPM_3IMERGHH",
    max_workers=2,
    convert_to_zarr=True,
    variables=["precipitationCal"],  # use None to read supported 2-D Grid variables
    attrs_path=None,
    time_chunk=96,
    lat_chunk=200,
    lon_chunk=200,
    zarr_workers=None,
    memory_limit_gb=None,
    overwrite_zarr=False,
)
print("Status:", result["status"])
print("Download summary:", result["download_summary"])
print("Zarr stores:", result["zarr_stores"])
print("Validation passed:", result["validation_passed"])

ds = read_imerg("/content/IMERG", "2025-01-01", "2025-01-03",
                variables=["precipitationCal"], compute=False)
print(ds)
print("Time steps:", ds.sizes.get("time", 0))
ds.close()
