"""ENTSO-E Transparency Platform downloads for the Irish bidding zone IE(SEM), cached on disk.

Each document type is fetched in monthly chunks and parsed into a tidy table with a UTC timestamp.
The token is read from the ENTSOE_TOKEN environment variable or from ~/.entsoe_token and is never
written to disk or logged.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

URL = "https://web-api.tp.entsoe.eu/api"
IE = "10Y1001A1001A59C"      # IE(SEM)
GB = "10YGB----------A"
PSR = {"B16": "solar", "B18": "wind_offshore", "B19": "wind_onshore", "B04": "gas", "B05": "coal",
       "B06": "oil", "B10": "hydro_pumped", "B11": "hydro_ror", "B12": "hydro_res", "B15": "other_re",
       "B17": "waste", "B01": "biomass", "B20": "other", "B09": "geothermal", "B14": "nuclear"}


def _token() -> str:
    t = os.environ.get("ENTSOE_TOKEN")
    if not t:
        t = (Path.home() / ".entsoe_token").read_text().strip()
    return t


def _res_minutes(res: str) -> int:
    return {"PT15M": 15, "PT30M": 30, "PT60M": 60}[res]


def parse(xml: bytes, value_tag: str = "price.amount") -> pd.DataFrame:
    """Parse a publication or generation document into rows (ts_utc, value, series key)."""
    root = ET.fromstring(xml)
    ns = {"n": root.tag.split("}")[0].strip("{")}
    rows = []
    for ts in root.findall("n:TimeSeries", ns):
        psr = ts.find(".//n:MktPSRType/n:psrType", ns)
        key = PSR.get(psr.text, psr.text) if psr is not None else ""
        # generation documents mark consumption (pumping) by inBiddingZone; keep both signs apart
        out = ts.find("n:outBiddingZone_Domain.mRID", ns)
        key = key + ("_consumption" if out is not None else "")
        for p in ts.findall("n:Period", ns):
            start = pd.Timestamp(p.find("n:timeInterval/n:start", ns).text)
            step = _res_minutes(p.find("n:resolution", ns).text)
            for pt in p.findall("n:Point", ns):
                pos = int(pt.find("n:position", ns).text)
                v = float(pt.find(f"n:{value_tag}", ns).text) if pt.find(f"n:{value_tag}", ns) is not None \
                    else float(pt.find("n:quantity", ns).text)
                rows.append((start + pd.Timedelta(minutes=step * (pos - 1)), key, v, step))
    return pd.DataFrame(rows, columns=["ts_utc", "series", "value", "step_min"])


def _get(params: dict, retries: int = 4) -> bytes | None:
    for i in range(retries):
        r = requests.get(URL, params={**params, "securityToken": _token()}, timeout=120)
        if r.status_code == 200:
            return r.content
        if r.status_code == 400 and b"No matching data" in r.content:
            return None
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"ENTSO-E request failed: HTTP {r.status_code} for {params.get('documentType')}")


def fetch(name: str, params: dict, start: str, end: str, cache: Path, value_tag: str = "price.amount") -> pd.DataFrame:
    """Monthly chunks from `start` to `end` (month starts), cached as one parquet file per document."""
    path = cache / f"{name}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for m in pd.date_range(start, end, freq="MS"):
        nxt = m + pd.offsets.MonthBegin(1)
        xml = _get({**params, "periodStart": m.strftime("%Y%m%d0000"), "periodEnd": nxt.strftime("%Y%m%d0000")})
        if xml is not None:
            frames.append(parse(xml, value_tag))
    df = pd.concat(frames, ignore_index=True)
    cache.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def download_all(cache: Path, start: str = "2023-01-01", end: str = "2026-10-01") -> dict[str, pd.DataFrame]:
    zone = {"in_Domain": IE, "out_Domain": IE}
    return {
        "price": fetch("price", {"documentType": "A44", **zone}, start, end, cache),
        "load_actual": fetch("load_actual", {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": IE},
                             start, end, cache, "quantity"),
        "load_fc": fetch("load_fc", {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": IE},
                         start, end, cache, "quantity"),
        "res_fc": fetch("res_fc", {"documentType": "A69", "processType": "A01", "in_Domain": IE},
                        start, end, cache, "quantity"),
        "gen": fetch("gen", {"documentType": "A75", "processType": "A16", "in_Domain": IE},
                     start, end, cache, "quantity"),
        "flow_to_gb": fetch("flow_to_gb", {"documentType": "A11", "in_Domain": GB, "out_Domain": IE},
                            start, end, cache, "quantity"),
        "flow_from_gb": fetch("flow_from_gb", {"documentType": "A11", "in_Domain": IE, "out_Domain": GB},
                              start, end, cache, "quantity"),
    }
