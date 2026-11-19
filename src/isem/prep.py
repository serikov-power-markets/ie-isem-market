"""Hourly market table for IE(SEM) from the cached downloads, with data-quality checks.

Day-ahead prices are published per market time unit (MTU). Every series is averaged to local
(Europe/Dublin) hours. Local days have 24 hours; on clock-change days the missing hour is
interpolated and the doubled hour averaged, as in the German study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Dublin"


def _hourly(df: pd.DataFrame, value_name: str, how: str = "mean") -> pd.Series:
    s = df.groupby("ts_utc")["value"].sum() if how == "sum" else df.groupby("ts_utc")["value"].mean()
    s.index = pd.DatetimeIndex(s.index)
    return s.resample("1h").mean().rename(value_name)


def build(raw: dict[str, pd.DataFrame], dash: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC hour with price, load, forecasts, generation by type, flows and dashboard fields."""
    out = pd.concat([
        _hourly(raw["price"], "price"),
        _hourly(raw["load_actual"], "load"),
        _hourly(raw["load_fc"], "load_fc"),
    ], axis=1)
    fc = raw["res_fc"].copy()
    fc["h"] = fc["ts_utc"].dt.floor("1h")
    fcp = fc.groupby(["h", "series"])["value"].mean().unstack("series")
    out["wind_fc"] = fcp.reindex(out.index)[[c for c in fcp.columns if c.startswith("wind")]].sum(axis=1, min_count=1)
    out["solar_fc"] = fcp.reindex(out.index)["solar"] if "solar" in fcp.columns else np.nan
    g = raw["gen"].copy()
    g["h"] = g["ts_utc"].dt.floor("1h")
    gp = g.groupby(["h", "series"])["value"].mean().unstack("series")
    for c in gp.columns:
        out["gen_" + c] = gp[c].reindex(out.index)
    out["flow_to_gb"] = _hourly(raw["flow_to_gb"], "x").reindex(out.index)
    out["flow_from_gb"] = _hourly(raw["flow_from_gb"], "x").reindex(out.index)
    out = out.reset_index().rename(columns={"index": "ts_utc"})
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    # dashboard (local time, 15 minutes) -> UTC hourly
    d = dash.copy()
    d["ts_utc"] = d["ts_local"].dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    d = d.dropna(subset=["ts_utc"])
    d["ts_utc"] = d["ts_utc"].dt.tz_convert("UTC").dt.floor("1h")
    dh = d.drop(columns="ts_local").groupby("ts_utc").mean()
    dh.columns = ["dash_" + c for c in dh.columns]
    out = out.merge(dh.reset_index(), on="ts_utc", how="left")
    # ENTSO-E actual load has gaps (2023 Q2, 2025 Q1, from Oct 2025): fill with the dashboard system demand
    out["load_src"] = np.where(out["load"].isna(), "dashboard", "entsoe")
    out["load"] = out["load"].fillna(out["dash_demand"])
    out["wind_fc"] = out["wind_fc"].fillna(out["dash_wind_fc"])
    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing share, duplicate timestamps, flat runs and out-of-range values per column."""
    rows = []
    idx = df["ts_utc"]
    rows.append(("ts_utc", "duplicates", int(idx.duplicated().sum())))
    full = pd.date_range(idx.min(), idx.max(), freq="1h")
    rows.append(("ts_utc", "missing hours in range", int(len(full) - idx.nunique())))
    for c in df.columns.drop("ts_utc"):
        x = df[c]
        rows.append((c, "missing share %", round(100 * float(x.isna().mean()), 2)))
        run = (x.diff() == 0).astype(int)
        longest = int(run.groupby((run == 0).cumsum()).sum().max()) if len(run) else 0
        rows.append((c, "longest flat run (h)", longest))
    p = df["price"]
    rows.append(("price", "below -100 or above 1000 EUR/MWh", int(((p < -100) | (p > 1000)).sum())))
    return pd.DataFrame(rows, columns=["column", "check", "value"])


def anomalies(df: pd.DataFrame, col: str = "price", window: int = 24 * 14, z: float = 6.0) -> pd.DataFrame:
    """Hours whose rolling robust z-score (median and MAD of the previous window) exceeds z."""
    x = df.set_index("ts_utc")[col]
    med = x.rolling(window, min_periods=window // 2).median().shift(1)
    mad = (x - med).abs().rolling(window, min_periods=window // 2).median().shift(1) * 1.4826
    score = (x - med) / mad.replace(0, np.nan)
    return pd.DataFrame({col: x, "z": score})[score.abs() > z].reset_index()


def local_days(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Local days x 24 local hours of one column."""
    local = df["ts_utc"].dt.tz_convert(TZ)
    g = pd.DataFrame({"day": local.dt.tz_localize(None).dt.normalize(), "hour": local.dt.hour, "v": df[col].to_numpy()})
    m = g.groupby(["day", "hour"])["v"].mean().unstack("hour").reindex(columns=list(range(24)))
    return m.interpolate(axis=1, limit_direction="both")
