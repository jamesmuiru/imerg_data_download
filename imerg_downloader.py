"""Download NASA GPM IMERG granules with earthaccess.

Expected product: GPM_3IMERGHH, Version 07 (Final half-hourly).
Bounding box order: (west, south, east, north), in geographic degrees.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import earthaccess
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────── validation ────────────────────────────────── #

def validate_inputs(dataset_name: str, start_date: str, end_date: str,
                    bbox: List[float], base_dir: str):
    """Validate all inputs; raise descriptive errors on bad values."""
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty.")

    try:
        start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
        end   = datetime.strptime(str(end_date)[:10],   "%Y-%m-%d")
        if start > end:
            raise ValueError("start_date must be on or before end_date.")
    except ValueError as exc:
        raise ValueError(f"Invalid date format – use YYYY-MM-DD. ({exc})") from exc

    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(
            "bbox must be [west, south, east, north] with west<east and south<north."
        )
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180):
        raise ValueError("bbox longitude bounds must satisfy -180 <= west < east <= 180.")
    if not (-90 <= south < north <= 90):
        raise ValueError("bbox latitude bounds must satisfy -90 <= south < north <= 90.")

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return start, end, (west, south, east, north)


# ─────────────────────────── month ranges ──────────────────────────────── #

def generate_month_ranges(start_date: str, end_date: str) -> pd.DataFrame:
    """Return a DataFrame of month-by-month date ranges (vectorised with pandas)."""
    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    df = pd.DataFrame({
        "year":  dates.year,
        "month": dates.month,
        "start": dates,
        "end":   [d + pd.offsets.MonthEnd(0) for d in dates],
    })
    return df


# ─────────────────────────── per-month worker ──────────────────────────── #

def _download_month(dataset_name: str, bbox: Tuple[float, ...],
                    start_dt: datetime, end_dt: datetime,
                    output_dir: str) -> dict:
    """Search and download all granules for one calendar month."""
    month_label = f"{start_dt.year}-{start_dt.month:02d}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    temporal = (start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    logger.info("🔍 Searching %s granules for %s", dataset_name, month_label)

    try:
        granules = list(earthaccess.search_data(
            short_name=dataset_name,
            bounding_box=tuple(bbox),
            temporal=temporal,
            cloud_hosted=False,
            count=-1,           # retrieve every matching granule
        ))
    except Exception as exc:
        logger.error("❌ Search failed for %s: %s", month_label, exc)
        return {"month": month_label, "found": 0, "downloaded": 0,
                "skipped": 0, "failed": True, "directory": output_dir}

    if not granules:
        logger.warning("⚠️  No granules found for %s", month_label)
        return {"month": month_label, "found": 0, "downloaded": 0,
                "skipped": 0, "failed": False, "directory": output_dir}

    logger.info("✅ Found %d granules for %s", len(granules), month_label)

    # Resume-safe: skip granules whose files already exist locally.
    existing_names = {p.name for p in Path(output_dir).iterdir() if p.is_file()}
    pending, skipped = [], 0
    for granule in granules:
        try:
            links = granule.data_links()
            if links:
                name = Path(links[0].split("?")[0]).name
                if name and name in existing_names:
                    skipped += 1
                    continue
        except Exception:
            pass
        pending.append(granule)

    downloaded = 0
    if pending:
        try:
            earthaccess.download(pending, local_path=output_dir)
            downloaded = len(pending)
        except Exception as exc:
            logger.error("❌ Download failed for %s: %s", month_label, exc)
            return {"month": month_label, "found": len(granules),
                    "downloaded": 0, "skipped": skipped,
                    "failed": True, "directory": output_dir}

    return {
        "month":      month_label,
        "found":      len(granules),
        "downloaded": downloaded,
        "skipped":    skipped,
        "failed":     False,
        "directory":  output_dir,
    }


# ─────────────────────────── public API ────────────────────────────────── #

def download_all(dataset_name: str, start_date: str, end_date: str,
                 bbox, base_dir: str, max_workers: int = 4) -> dict:
    """Authenticate with NASA Earthdata and download IMERG granules.

    Parameters
    ----------
    dataset_name : str   NASA CMR short name (e.g. "GPM_3IMERGHH").
    start_date   : str   ISO date string "YYYY-MM-DD".
    end_date     : str   ISO date string "YYYY-MM-DD" (inclusive).
    bbox         : seq   (west, south, east, north) in degrees.
    base_dir     : str   Root output directory.
    max_workers  : int   Parallel download threads (one per month).

    Returns
    -------
    dict with keys: dataset_name, start_date, end_date, bbox, base_dir,
                    months (list of per-month dicts), total_found,
                    total_downloaded, total_skipped, failed_months.
    """
    start, end, bbox = validate_inputs(dataset_name, start_date, end_date,
                                       list(bbox), base_dir)
    logger.info("🔐 Authenticating with NASA Earthdata …")
    earthaccess.login(strategy="interactive", persist=True)

    months_df = generate_month_ranges(start.strftime("%Y-%m-%d"),
                                      end.strftime("%Y-%m-%d"))
    output_root = os.path.join(base_dir, dataset_name)

    # Pre-create all month directories (vectorised).
    months_df["outdir"] = months_df.apply(
        lambda r: os.path.join(output_root, str(r.year), f"{r.month:02d}"),
        axis=1,
    )
    months_df["outdir"].map(lambda p: os.makedirs(p, exist_ok=True))

    results: List[dict] = []
    workers = max(1, min(int(max_workers), len(months_df)))

    pbar = tqdm(total=len(months_df), desc="📥 Downloading months", ncols=100)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _download_month,
                dataset_name, bbox,
                row.start.to_pydatetime(),
                row.end.to_pydatetime(),
                row.outdir,
            ): (row.year, row.month)
            for _, row in months_df.iterrows()
        }
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✅" if not result["failed"] else "❌"
                pbar.set_postfix_str(f"{year}-{month:02d} {status}")
            except Exception as exc:
                logger.error("❌ Unexpected error for %d-%02d: %s", year, month, exc)
                results.append({"month": f"{year}-{month:02d}", "found": 0,
                                 "downloaded": 0, "skipped": 0, "failed": True,
                                 "directory": ""})
            finally:
                pbar.update(1)

    pbar.close()

    results.sort(key=lambda r: r["month"])
    summary = {
        "dataset_name":     dataset_name,
        "start_date":       start.strftime("%Y-%m-%d"),
        "end_date":         end.strftime("%Y-%m-%d"),
        "bbox":             bbox,
        "base_dir":         str(Path(base_dir)),
        "months":           results,
        "total_found":      sum(r["found"]      for r in results),
        "total_downloaded": sum(r["downloaded"] for r in results),
        "total_skipped":    sum(r["skipped"]    for r in results),
        "failed_months":    [r["month"]         for r in results if r["failed"]],
    }
    logger.info(
        "Download complete: %d found | %d downloaded | %d skipped | %d failed",
        summary["total_found"], summary["total_downloaded"],
        summary["total_skipped"], len(summary["failed_months"]),
    )
    return summary


# ─────────────────────────── CLI entry point ───────────────────────────── #

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date",    required=True)
    parser.add_argument("--end-date",      required=True)
    parser.add_argument("--bbox",          nargs=4, type=float, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--output-dir",    default="data/imerg")
    parser.add_argument("--dataset-name",  default="GPM_3IMERGHH")
    parser.add_argument("--max-workers",   type=int, default=4)
    args = parser.parse_args()
    import json
    print(json.dumps(
        download_all(args.dataset_name, args.start_date, args.end_date,
                     tuple(args.bbox), args.output_dir, args.max_workers),
        indent=2,
    ))
