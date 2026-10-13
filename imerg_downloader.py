"""Download NASA GPM IMERG granules using earthaccess."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta
from pathlib import Path
import logging
import earthaccess

logger = logging.getLogger(__name__)


def validate_inputs(dataset_name, start_date, end_date, bbox, base_dir):
    start, end = date.fromisoformat(str(start_date)[:10]), date.fromisoformat(str(end_date)[:10])
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north)")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid bbox: require -180 <= west < east <= 180 and -90 <= south < north <= 90")
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return start, end, (west, south, east, north)


def generate_month_ranges(start_date, end_date):
    """Yield calendar-month ranges clipped to the requested inclusive dates."""
    start = date.fromisoformat(str(start_date)[:10]); end = date.fromisoformat(str(end_date)[:10])
    current = start.replace(day=1)
    while current <= end:
        next_month = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
        yield max(current, start), min(next_month - timedelta(days=1), end)
        current = next_month


def _download_month(dataset_name, bbox, first_day, last_day, base_dir):
    month_dir = Path(base_dir) / dataset_name / f"{first_day:%Y}" / f"{first_day:%m}"
    month_dir.mkdir(parents=True, exist_ok=True)
    temporal = (datetime.combine(first_day, time.min).isoformat(),
                datetime.combine(last_day, time.max).isoformat(timespec="seconds"))
    try:
        granules = list(earthaccess.search_data(short_name=dataset_name, version="07",
            temporal=temporal, bounding_box=bbox, count=-1))
        existing = {p.name for p in month_dir.iterdir() if p.is_file()}
        pending = []
        for granule in granules:
            links = granule.data_links()
            name = Path(links[0].split("?")[0]).name if links else ""
            if not name or name not in existing:
                pending.append(granule)
        if pending:
            earthaccess.download(pending, local_path=str(month_dir))
        # Count actual files present, rather than assuming download() succeeded.
        files = list(month_dir.glob("*.HDF5")) + list(month_dir.glob("*.h5")) + list(month_dir.glob("*.hdf5"))
        return {"month": first_day.strftime("%Y-%m"), "found": len(granules),
                "downloaded": len(pending), "skipped": len(granules)-len(pending),
                "local_files": len(files), "failed": False, "error": None,
                "directory": str(month_dir)}
    except Exception as exc:
        logger.exception("Download failed for %s", first_day.strftime("%Y-%m"))
        return {"month": first_day.strftime("%Y-%m"), "found": 0, "downloaded": 0,
                "skipped": 0, "local_files": 0, "failed": True, "error": str(exc),
                "directory": str(month_dir)}


def download_all(dataset_name, start_date, end_date, bbox, base_dir, max_workers=4):
    start, end, bbox = validate_inputs(dataset_name, start_date, end_date, bbox, base_dir)
    logger.info("Authenticating with NASA Earthdata")
    auth = earthaccess.login(strategy="interactive", persist=True)
    if auth is False:
        raise RuntimeError("NASA Earthdata authentication failed")
    ranges = list(generate_month_ranges(start.isoformat(), end.isoformat()))
    results = []
    workers = max(1, min(int(max_workers), len(ranges)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_month, dataset_name, bbox, a, b, base_dir) for a, b in ranges]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["month"])
    summary = {"dataset_name": dataset_name, "start_date": start.isoformat(), "end_date": end.isoformat(),
        "bbox": bbox, "base_dir": str(Path(base_dir)), "months": results,
        "total_found": sum(x["found"] for x in results),
        "total_downloaded": sum(x["downloaded"] for x in results),
        "total_skipped": sum(x["skipped"] for x in results),
        "total_local_files": sum(x["local_files"] for x in results),
        "failed_months": [x["month"] for x in results if x["failed"]]}
    return summary
