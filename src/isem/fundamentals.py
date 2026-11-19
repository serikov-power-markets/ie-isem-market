"""Price formation, wind capture prices and system non-synchronous penetration (SNSP) in IE(SEM).

Definitions
- Capture price of wind in a period = sum(price x wind output) / sum(wind output); capture rate =
  capture price / time-weighted mean price of the same period.
- SNSP is reconstructed as (wind + solar + HVDC imports) / (demand + HVDC exports), all in MW, with
  HVDC flows taken as the net physical flow between IE(SEM) and Great Britain (EWIC, Moyle and from
  2025 Greenlink are all HVDC). EirGrid's own SNSP series is not available through the open API, so
  this is a reconstruction: levels may differ by a few points from the published value. Operational
  limit assumed 75% (raised from 70% in 2021; 80% is the 2025 programme target).
- Cannibalisation: OLS of the hourly wind capture rate (per month) on the monthly wind share of load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

SNSP_LIMIT = 75.0


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    wind = d["gen_wind_onshore"].fillna(0) + d.get("gen_wind_offshore", 0)
    d["wind"] = wind
    d["solar"] = d["gen_solar"] if "gen_solar" in d else np.nan
    d["net_flow_to_gb"] = d["flow_to_gb"].fillna(0) - d["flow_from_gb"].fillna(0)  # MW exported from IE(SEM)
    exp = d["net_flow_to_gb"].clip(lower=0)
    imp = (-d["net_flow_to_gb"]).clip(lower=0)
    d["snsp"] = 100 * (d["wind"] + d["solar"].fillna(0) + imp) / (d["load"] + exp)
    d["wind_share"] = 100 * d["wind"] / d["load"]
    d["year"] = d["ts_utc"].dt.year
    d["month"] = d["ts_utc"].dt.to_period("M")
    return d


def price_summary(d: pd.DataFrame) -> pd.DataFrame:
    g = d.dropna(subset=["price"]).groupby("year")
    return pd.DataFrame({
        "mean_price": g["price"].mean(), "median": g["price"].median(),
        "share_le_0_%": 100 * g["price"].apply(lambda x: (x <= 0).mean()),
        "share_gt_150_%": 100 * g["price"].apply(lambda x: (x > 150).mean()),
        "std": g["price"].std(), "hours": g["price"].count()})


def capture(d: pd.DataFrame, gen_col: str = "wind", by: str = "year") -> pd.DataFrame:
    x = d.dropna(subset=["price", gen_col])
    x = x[x[gen_col] > 0]
    rows = {}
    for k, g in x.groupby(by):
        cap = float((g["price"] * g[gen_col]).sum() / g[gen_col].sum())
        rows[k] = {"mean_price": float(g["price"].mean()), "capture_price": cap,
                   "capture_rate_%": 100 * cap / float(g["price"].mean()), "gwh": float(g[gen_col].sum() / 1000)}
    return pd.DataFrame(rows).T


def cannibalisation(d: pd.DataFrame) -> dict:
    """Monthly wind capture rate against the monthly wind share of load; slope in points per point."""
    x = d.dropna(subset=["price", "wind", "load"])
    m = x.groupby("month").apply(lambda g: pd.Series({
        "share": 100 * g["wind"].sum() / g["load"].sum(),
        "rate": 100 * ((g["price"] * g["wind"]).sum() / g["wind"].sum()) / g["price"].mean()}), include_groups=False)
    res = sm.OLS(m["rate"], sm.add_constant(m["share"])).fit(cov_type="HC1")
    return {"slope": float(res.params["share"]), "t": float(res.tvalues["share"]), "intercept": float(res.params["const"]),
            "r2": float(res.rsquared), "months": int(len(m))}


def by_band(d: pd.DataFrame, col: str, edges: list[float]) -> pd.DataFrame:
    """Mean price, wind output and wind capture by band of a driver (e.g. SNSP or wind share)."""
    x = d.dropna(subset=["price", col]).copy()
    x["band"] = pd.cut(x[col], edges)
    rows = {}
    for k, g in x.groupby("band", observed=True):
        rows[str(k)] = {"hours_%": 100 * len(g) / len(x), "mean_price": float(g["price"].mean()),
                        "mean_wind_mw": float(g["wind"].mean()),
                        "wind_capture": float((g["price"] * g["wind"]).sum() / g["wind"].sum()) if g["wind"].sum() > 0 else np.nan}
    return pd.DataFrame(rows).T


def price_drivers(d: pd.DataFrame, gas: pd.Series, co2: pd.Series) -> dict:
    """OLS of the hourly price on the gas and carbon cost of a CCGT and on residual load (HC1 errors)."""
    x = d.dropna(subset=["price", "load", "wind"]).copy()
    mo = x["month"].astype(str)
    gas_m = gas.copy(); gas_m.index = gas_m.index.astype(str)
    co2_m = co2.copy(); co2_m.index = co2_m.index.astype(str)
    x["fuel"] = (mo.map(gas_m) + 0.202 * mo.map(co2_m)) / 0.55
    x["resid"] = (x["load"] - x["wind"]) / 1000
    x = x.dropna(subset=["fuel"])
    X = sm.add_constant(x[["fuel", "resid"]])
    res = sm.OLS(x["price"], X).fit(cov_type="HC1")
    return {"fuel_coef": float(res.params["fuel"]), "fuel_t": float(res.tvalues["fuel"]),
            "resid_coef_per_gw": float(res.params["resid"]), "resid_t": float(res.tvalues["resid"]),
            "r2": float(res.rsquared), "n": int(len(x))}
