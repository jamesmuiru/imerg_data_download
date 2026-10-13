"""Download NASA GPM IMERG granules with earthaccess.

Expected product: GPM_3IMERGHH, Version 07 (Final half-hourly).
Bounding box order: (west, south, east, north), in geographic degrees.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
import logging

import earthaccess

logger = logging.getLogger(__name__)


def validate_inputs(dataset_name, start_date, end_date, bbox, base_dir):
    start = date.fromisoformat(str(start_date)[:10])
    end = date.fromisoformat(str(end_date)[:10])
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north)")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitude bounds must satisfy -180 <= west < east <= 180")
    if not (-90 <= south < north <= 90):
        raise ValueError("bbox latitude bounds must satisfy -90 <= south < north <= 90")
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return start, end, (west, south, east, north)


def generate_month_ranges(start_date, end_date):
    """Yield inclusive (month_start, month_end) date ranges clipped to request."""
    current = date.fromisoformat(str(start_date)[:10]).replace(day=1)
    final = date.fromisoformat(str(end_date)[:10])
    while current <= final:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        month_end = next_month - timedelta(days=1)
        yield max(current, date.fromisoformat(str(start_date)[:10])), min(month_end, final)
        current = next_month


def _download_month(dataset_name, bbox, month_start, month_end, base_dir):
    month_dir = Path(base_dir) / dataset_name / f"{month_start:%Y}" / f"{month_start:%m}"
    month_dir.mkdir(parents=True, exist_ok=True)

    # earthaccess temporal bounds are ISO datetimes; include the full final day.
    temporal = (
        datetime.combine(month_start, time.min).isoformat(),
        datetime.combine(month_end, time.max).isoformat(timespec="seconds"),
    )
    logger.info("Searching %s from %s to %s", dataset_name, temporal[0], temporal[1])
    granules = earthaccess.search_data(
        short_name=dataset_name,
        version="07",
        temporal=temporal,
        bounding_box=bbox,
    )
    if not granules:
        logger.warning("No granules found for %s to %s", month_start, month_end)
        return {"month": f"{month_start:%Y-%m}", "found": 0, "downloaded": 0, "directory": str(month_dir)}

    # Avoid re-downloading files already present in the month directory.
    existing_names = {p.name for p in month_dir.iterdir() if p.is_file()}
    pending = []
    for granule in granules:
        try:
            links = granule.data_links()
            if links:
                name = Path(links[0].split("?")[0]).name
                if name and name not in existing_names:
                    pending.append(granule)
        except Exception:
            pending.append(granule)

    if pending:
        earthaccess.download(pending, local_path=str(month_dir))
    return {
        "month": f"{month_start:%Y-%m}",
        "found": len(granules),
        "downloaded": len(pending),
        "directory": str(month_dir),
    }


def download_all(dataset_name, start_date, end_date, bbox, base_dir, max_workers=4):
    """Authenticate and download the requested granules into year/month folders."""
    start, end, bbox = validate_inputs(dataset_name, start_date, end_date, bbox, base_dir)
    logger.info("Authenticating with NASA Earthdata")
    earthaccess.login(strategy="interactive", persist=True)

    ranges = list(generate_month_ranges(start.isoformat(), end.isoformat()))
    results = []
    # Separate month folders prevent concurrent downloads from colliding.
    workers = max(1, min(int(max_workers), len(ranges) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_month, dataset_name, bbox, month_start, month_end, base_dir): (month_start, month_end)
            for month_start, month_end in ranges
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["month"])
    summary = {
        "dataset_name": dataset_name,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "bbox": bbox,
        "base_dir": str(Path(base_dir)),
        "months": results,
        "total_found": sum(item["found"] for item in results),
        "total_downloaded": sum(item["downloaded"] for item in results),
    }
    logger.info("Download complete: %s granules found; %s newly downloaded", summary["total_found"], summary["total_downloaded"])
    return summary


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--output-dir", default="data/imerg")
    parser.add_argument("--dataset-name", default="GPM_3IMERGHH")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    print(download_all(args.dataset_name, args.start_date, args.end_date, tuple(args.bbox), args.output_dir, args.max_workers))
