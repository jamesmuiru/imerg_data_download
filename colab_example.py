"""Example usage – works in Google Colab or any Python environment.

Install dependencies first:
    pip install -r requirements.txt
"""
from imerg_wrapper import get_imerg, read_imerg

# ── Download + convert to Zarr ──────────────────────────────────────────── #
result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-03",
    bbox=(36.0, -1.5, 38.0, 0.5),      # west, south, east, north (near Nairobi)
    output_dir="/content/IMERG",
    dataset_name="GPM_3IMERGHH",

    # Download options
    max_workers=4,

    # Zarr conversion options
    convert_to_zarr=True,
    variables=None,                     # None = all HDF5 variables
    attrs_path=None,                    # optional path to variable-attrs JSON
    time_chunk=96,                      # 96 half-hourly steps = 2 days
    lat_chunk=200,
    lon_chunk=200,
    zarr_workers=None,                  # default: cpu_count - 1
    memory_limit_gb=None,               # e.g. 8.0 to cap RAM usage
    overwrite_zarr=False,               # skip already-converted months
)

print("Download summary:", result["download_summary"])
print("Zarr stores:", result["zarr_stores"])

# ── Read data back ──────────────────────────────────────────────────────── #
ds = read_imerg(
    output_dir="/content/IMERG",
    start_date="2025-01-01",
    end_date="2025-01-03",
    variables=["precipitationCal"],     # None = all variables
    compute=True,
)

print(ds)
print("Time steps:", len(ds.time))
