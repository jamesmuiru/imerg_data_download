# IMERG Data Download Toolkit

A Python toolkit for downloading **NASA GPM IMERG** satellite rainfall data and converting it into analysis-ready, compressed **Zarr** stores organised by calendar month.

> **Product:** GPM_3IMERGHH — Version 07 (Final, half-hourly, global)
> **Resolution:** 0.1° × 0.1° spatial · 30-minute temporal

---

## Features

| Category | Capability |
|---|---|
| 🔐 **Auth** | NASA Earthdata login via `earthaccess` (interactive, credentials cached to `~/.netrc`) |
| 📅 **Date ranges** | Pandas-vectorised month splitting — handles any multi-year range |
| ⚡ **Parallel downloads** | `ThreadPoolExecutor` — one thread per month, configurable workers |
| 🔍 **Complete search** | `count=-1` ensures every available granule is found |
| ♻️ **Resume-safe downloads** | Skips already-downloaded HDF5 files by filename |
| 📊 **tqdm progress** | Visual progress bars in terminal and Jupyter/Colab |
| 🗂️ **All HDF5 variables** | Reads every variable from the `Grid/` group (not just `precipitationCal`) |
| 🎯 **Variable whitelist** | Optional `variables` list to read only specific fields |
| 📝 **External attributes** | Attach variable metadata from a JSON file (`attrs_path`) |
| 🔀 **Parallel HDF5 reading** | `ProcessPoolExecutor` for CPU-parallel granule parsing |
| 🗜️ **Blosc compression** | lz4-compressed Zarr stores — smaller footprint, faster I/O |
| 🛡️ **Memory monitoring** | `psutil` RSS guard — skips months when RAM limit is approached |
| ⏱️ **Time sort + dedup** | Guaranteed strictly-monotonic time axis before writing |
| ✅ **Zarr verification** | Re-opens each store after writing and checks time monotonicity |
| 🔄 **Resume conversion** | `processing_state.json` tracks done months across sessions |
| 📖 **Built-in reader** | `read_imerg_zarr()` / `read_imerg_zarr_all()` — no manual Zarr handling needed |
| 🗺️ **Bounding box crop** | Spatial subset applied to both HDF5 reading and Zarr output |
| 🛠️ **Failure log** | Bad granules logged to `conversion_failures.csv` instead of crashing |

---

## Project Structure

```
imerg_data_download/
├── imerg_downloader.py      # NASA Earthdata parallel download engine
├── imerg_zarr_pipeline.py   # IMERGProcessor: HDF5 → monthly Zarr + reader
├── imerg_wrapper.py         # One-call facade: get_imerg() + read_imerg()
├── colab_example.py         # Minimal usage demo
└── requirements.txt         # Python dependencies
```

### Module Architecture

```
colab_example.py
        │
        ▼
imerg_wrapper.get_imerg()          ← single user-facing function
        │
        ├──► imerg_downloader.download_all()
        │         ├── validate_inputs()
        │         ├── earthaccess.login()
        │         ├── generate_month_ranges()    pandas vectorised
        │         └── ThreadPoolExecutor
        │               └── _download_month()   per-month, resume-safe
        │
        └──► imerg_zarr_pipeline.convert_imerg_to_monthly_zarr()
                  └── IMERGProcessor
                        ├── _read_files()           ProcessPoolExecutor
                        │     └── _read_single_granule()   all variables
                        ├── _sort_and_dedup()       monotonic time
                        ├── _write_zarr()           Blosc compressed
                        ├── _verify_zarr()          post-write check
                        └── _save_processing_state()  resume JSON

imerg_wrapper.read_imerg()
        └──► imerg_zarr_pipeline.read_imerg_zarr()
                  └── IMERGProcessor.read_imerg_zarr()
```

---

## Requirements

- Python 3.8+
- A free [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new)

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `earthaccess` | NASA Earthdata auth + CMR search + download |
| `h5py` | Reading IMERG HDF5/H5 granule files |
| `numpy` | Array operations and bounding-box masking |
| `pandas` | Vectorised date ranges, timestamp parsing |
| `xarray` | Labelled `(time, lat, lon)` datasets |
| `zarr<3` | Zarr v2 store writing (pinned for xarray compatibility) |
| `dask` | Lazy chunked computation behind xarray |
| `tqdm` | Progress bars (terminal + Jupyter/Colab) |
| `psutil` | RSS memory monitoring during conversion |
| `numcodecs` | Blosc/lz4 compression for Zarr stores |

---

## Quick Start

### Full pipeline — download + convert

```python
from imerg_wrapper import get_imerg

result = get_imerg(
    start_date="2025-01-01",
    end_date="2025-01-31",
    bbox=(36.0, -1.5, 38.0, 0.5),   # west, south, east, north
    output_dir="./data/IMERG",

    # Download
    max_workers=4,

    # Zarr conversion
    convert_to_zarr=True,
    variables=None,                  # None → all HDF5 variables
    attrs_path=None,                 # optional JSON with variable metadata
    time_chunk=96,                   # 96 steps = 2 days
    lat_chunk=200,
    lon_chunk=200,
    zarr_workers=None,               # default: cpu_count - 1
    memory_limit_gb=None,            # e.g. 8.0 to cap RAM usage
    overwrite_zarr=False,
)

print(result["zarr_stores"])
```

### Read data back

```python
from imerg_wrapper import read_imerg

ds = read_imerg(
    output_dir="./data/IMERG",
    start_date="2025-01-01",
    end_date="2025-01-31",
    variables=["precipitationCal"],  # None → all variables
    compute=True,
)
print(ds)
```

### Download only (keep raw HDF5)

```python
from imerg_downloader import download_all

summary = download_all(
    dataset_name="GPM_3IMERGHH",
    start_date="2025-01-01",
    end_date="2025-01-31",
    bbox=(36.0, -1.5, 38.0, 0.5),
    base_dir="./data/IMERG",
    max_workers=4,
)
print(summary)
```

### Convert existing HDF5 files to Zarr

```python
from imerg_zarr_pipeline import convert_imerg_to_monthly_zarr

stores = convert_imerg_to_monthly_zarr(
    input_dir="./data/IMERG/GPM_3IMERGHH",
    output_dir="./data/IMERG/zarr",
    years=["2024", "2025"],          # None → auto-discover
    bbox=(36.0, -1.5, 38.0, 0.5),
    variables=None,                  # all variables
    time_chunk=96,
    lat_chunk=200,
    lon_chunk=200,
    memory_limit_gb=8.0,
    overwrite=False,
)
```

### Use `IMERGProcessor` directly (advanced)

```python
from imerg_zarr_pipeline import IMERGProcessor

processor = IMERGProcessor(
    input_dir="./data/IMERG/GPM_3IMERGHH",
    output_dir="./data/IMERG/zarr",
    bbox=(36.0, -1.5, 38.0, 0.5),
    variables=["precipitationCal", "precipitationUncal"],
    attrs_path="./variable_attrs.json",
    zarr_chunks={"time": 96, "lat": 200, "lon": 200},
    max_workers=4,
    memory_limit_gb=8.0,
)

# Convert
processor.process_years(["2024", "2025"])

# Read back
ds = processor.read_imerg_zarr("2024-06-01", "2024-08-31", compute=True)
print(ds)
```

---

## Command-Line Interface

### Download granules

```bash
python imerg_downloader.py \
  --start-date 2025-01-01 \
  --end-date   2025-01-31 \
  --bbox 36.0 -1.5 38.0 0.5 \
  --output-dir ./data/IMERG \
  --dataset-name GPM_3IMERGHH \
  --max-workers 4
```

### Convert HDF5 → Zarr

```bash
python imerg_zarr_pipeline.py \
  --input-dir  ./data/IMERG/GPM_3IMERGHH \
  --output-dir ./data/IMERG/zarr \
  --years 2024 2025 \
  --bbox 36.0 -1.5 38.0 0.5 \
  --variables precipitationCal precipitationUncal \
  --time-chunk 96 \
  --lat-chunk 200 \
  --lon-chunk 200 \
  --max-workers 4 \
  --memory-limit-gb 8.0 \
  --overwrite
```

---

## Output Structure

```
./data/IMERG/
├── GPM_3IMERGHH/                   # Raw HDF5 granules
│   └── 2025/
│       └── 01/
│           ├── 3B-HHR.MS.MRG.3IMERG.20250101-S000000-E002959.*.HDF5
│           └── ...
├── zarr/                            # Converted Zarr stores
│   └── 2025/
│       └── IMERG_2025_01.zarr/
└── zarr/processing_state.json       # Resume tracker
```

### Zarr Dataset Schema

Each monthly `.zarr` store is an `xarray.Dataset` with:

| Dimension | Description |
|---|---|
| `time` | Half-hourly UTC timestamps |
| `lat` | Latitude (cropped to bbox if specified) |
| `lon` | Longitude (cropped to bbox if specified) |

| Variable | Units | Description |
|---|---|---|
| `precipitationCal` | mm/hr | Gauge-calibrated precipitation estimate |
| `precipitationUncal` | mm/hr | Uncalibrated precipitation estimate |
| `HQprecipitation` | mm/hr | High-quality merged microwave precipitation |
| *(all other Grid/* variables)* | — | Included by default unless `variables` is set |

---

## `get_imerg()` Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_date` | `str` | required | `"YYYY-MM-DD"` |
| `end_date` | `str` | required | `"YYYY-MM-DD"` (inclusive) |
| `bbox` | `tuple` | required | `(west, south, east, north)` degrees |
| `output_dir` | `str` | required | Root output directory |
| `dataset_name` | `str` | `"GPM_3IMERGHH"` | NASA CMR short name |
| `max_workers` | `int` | `4` | Parallel download threads |
| `convert_to_zarr` | `bool` | `True` | Run HDF5 → Zarr pipeline |
| `variables` | `list` | `None` | HDF5 variables to include (None = all) |
| `attrs_path` | `str` | `None` | JSON file with variable metadata |
| `time_chunk` | `int` | `96` | Zarr time chunk (96 = 2 days) |
| `lat_chunk` | `int` | `200` | Zarr latitude chunk |
| `lon_chunk` | `int` | `200` | Zarr longitude chunk |
| `zarr_workers` | `int` | `None` | Workers for HDF5 reading (None = cpu_count−1) |
| `memory_limit_gb` | `float` | `None` | RSS memory cap in GB |
| `overwrite_zarr` | `bool` | `False` | Re-process existing Zarr stores |

---

## Authentication

On first run you will be prompted for your NASA Earthdata credentials:

```
Enter your Earthdata Login username: your_username
Enter your Earthdata Login password: ········
```

Credentials are cached to `~/.netrc` and reused automatically on all future runs.

---

## Variable Attributes JSON (Optional)

Pass `attrs_path` pointing to a JSON file to attach CF-convention metadata to variables:

```json
{
  "precipitationCal": {
    "long_name": "IMERG calibrated precipitation",
    "units": "mm/hr",
    "valid_min": 0.0,
    "valid_max": 9999.9
  },
  "precipitationUncal": {
    "long_name": "IMERG uncalibrated precipitation",
    "units": "mm/hr"
  }
}
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `No granules found` | Check bbox and date range overlap with available IMERG data |
| `Could not find precipitationCal` | HDF5 file may be corrupt — check `conversion_failures.csv` |
| `zarr` import errors | Run `pip install "zarr<3"` |
| Auth failure | Re-run `earthaccess.login(strategy="interactive")` to refresh |
| OOM / memory crash | Set `memory_limit_gb=8.0` (or appropriate for your system) |
| Partial run / missing months | Re-run — `processing_state.json` will skip completed months |

---

## License

MIT — see [LICENSE](LICENSE).