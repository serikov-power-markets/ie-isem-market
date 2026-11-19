"""EirGrid Smart Grid Dashboard data (all-island, 15-minute), cached on disk.

The public dashboard API returns one field per request. Fields used: wind actual and forecast (MW), system demand (MW) and CO2 intensity (g/kWh). The
dashboard's SNSP series is not exposed under a known area name, so SNSP is reconstructed in prep.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.smartgriddashboard.com/api/chart/"
AREAS = {"wind": "windactual", "wind_fc": "windforecast", "demand": "demandactual",
         "co2": "co2intensity"}
HEADERS = {"User-Agent": "Mozilla/5.0 (research; ie-isem-market)"}


def _month(area: str, start: pd.Timestamp) -> pd.DataFrame:
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)
    q = {"region": "ALL", "chartType": "default", "dateRange": "day", "areas": area,
         "dateFrom": start.strftime("%d-%b-%Y"), "dateTo": end.strftime("%d-%b-%Y")}
    for i in range(12):
        try:
            r = requests.get(URL, params=q, headers=HEADERS, timeout=120)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("{"):
            body = r.json()
            rows = body.get("Rows") or []
            df = pd.DataFrame(rows)
            if df.empty:
                return df
            df["ts"] = pd.to_datetime(df["EffectiveTime"], format="%d-%b-%Y %H:%M:%S")
            return df[["ts", "Value"]]
        time.sleep(min(5 * (i + 1), 30))
    raise RuntimeError(f"dashboard request failed for {area} {start:%Y-%m}: HTTP {r.status_code}")


def _cached_month(cache: Path, area: str, start: pd.Timestamp) -> pd.DataFrame:
    path = cache / "eirgrid_chunks" / f"{area}_{start:%Y-%m}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = _month(area, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch(cache: Path, start: str = "2023-01-01", end: str = "2026-10-01") -> pd.DataFrame:
    """15-minute table with the dashboard fields as columns; timestamps are Irish local time."""
    path = cache / f"eirgrid_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    cols = {}
    for name, area in AREAS.items():
        parts = [_cached_month(cache, area, m) for m in pd.date_range(start, end, freq="MS")]
        d = pd.concat([p for p in parts if not p.empty], ignore_index=True).drop_duplicates("ts")
        cols[name] = d.set_index("ts")["Value"]
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "ts_local"
    cache.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(path)
    return df.reset_index()
