"""Run the whole study: data checks, fundamentals, SNSP, forecasts and the value of forecasts for a CCGT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from isem import eirgrid, entsoe, forecast, fundamentals as fu, prep, thermal
from isem.fuels import monthly_prices

ROOT = Path(__file__).resolve().parents[2]
RAW, RES = ROOT / "data" / "raw", ROOT / "results"
FUEL_START, FUEL_END = "2023-01", "2025-12"


def load() -> pd.DataFrame:
    raw = entsoe.download_all(RAW)
    dash = eirgrid.fetch(RAW)
    return fu.add_derived(prep.build(raw, dash))


def main() -> None:
    RES.mkdir(exist_ok=True)
    d = load()
    out: dict = {}
    prep.quality_report(d[["ts_utc", "price", "load", "wind_fc", "solar_fc", "wind", "net_flow_to_gb",
                           "dash_wind", "dash_demand"]]).to_csv(RES / "quality_report.csv", index=False)
    prep.anomalies(d).to_csv(RES / "price_anomalies.csv", index=False)
    out["anomalous_hours"] = int(len(prep.anomalies(d)))
    # 1. market summary and capture prices
    fu.price_summary(d).round(2).to_csv(RES / "price_summary.csv")
    cap = fu.capture(d, "wind")
    cap.round(2).to_csv(RES / "wind_capture_by_year.csv")
    capm = fu.capture(d, "wind", by="month")
    capm.round(2).to_csv(RES / "wind_capture_by_month.csv")
    out["cannibalisation"] = fu.cannibalisation(d)
    fu.by_band(d, "snsp", [0, 30, 45, 60, 70, 75, 200]).round(2).to_csv(RES / "by_snsp_band.csv")
    fu.by_band(d, "wind_share", [-1, 10, 20, 30, 40, 60, 200]).round(2).to_csv(RES / "by_wind_share_band.csv")
    out["snsp_share_ge_70_%"] = float(100 * (d["snsp"] >= 70).mean())
    out["snsp_share_ge_75_%"] = float(100 * (d["snsp"] >= fu.SNSP_LIMIT).mean())
    # dashboard wind forecast minus actual: availability gap in hours at the SNSP limit (indicator of dispatch-down)
    gap = (d["dash_wind_fc"] - d["dash_wind"])
    out["wind_fc_minus_actual_mw"] = {"snsp_ge_70": float(gap[d["snsp"] >= 70].mean()),
                                      "snsp_lt_70": float(gap[d["snsp"] < 70].mean())}
    # 2. drivers (fuel data end 2025)
    fuels = monthly_prices(RAW / "fuels", FUEL_START, FUEL_END)
    dd = d[d["ts_utc"] < "2026-01-01"]
    out["drivers"] = fu.price_drivers(dd, fuels["gas_eur_mwh"], fuels["co2_eur_t"])
    # 3. forecasts, test year 2025
    eua = pd.read_pickle(RES / "eua.pkl") if (RES / "eua.pkl").exists() else None
    if eua is None:
        from isem.fuels import eua_auctions
        eua = eua_auctions(RAW / "fuels", range(2022, 2026)); eua.to_pickle(RES / "eua.pkl")
    inp = forecast.build_inputs(dd, eua, fuels["gas_eur_mwh"])
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    act = inp["price"].reindex(days)
    nv = forecast.naive(inp["price"], days)
    lr = forecast.lear(inp, days, 728)
    gb = forecast.gbm(inp, days, 728)
    fcs = {"naive": nv, "lear": lr, "gbm": gb, "avg": (lr + gb) / 2}
    mt = {k: forecast.metrics(act, v, nv) for k, v in fcs.items()}
    out["forecast_metrics"] = mt
    out["dm_lear_vs_naive"] = forecast.dm_test(act, nv, lr)
    out["dm_gbm_vs_lear"] = forecast.dm_test(act, lr, gb)
    pd.DataFrame(mt).T.round(3).to_csv(RES / "forecast_metrics.csv")
    for k, v in fcs.items():
        v.to_pickle(RES / f"fc_{k}.pkl")
    # 4. value of forecasts for a CCGT
    plant = thermal.CCGT()
    rows = []
    for name in ["perfect", "naive", "lear", "gbm", "avg"]:
        state = thermal.State()
        profit = 0.0
        starts = 0
        for d0 in days:
            m = d0.to_period("M")
            fuel = thermal.fuel_cost(float(fuels.loc[m, "gas_eur_mwh"]), float(fuels.loc[m, "co2_eur_t"]))
            real = act.loc[d0].to_numpy()
            guess = real if name == "perfect" else fcs[name].loc[d0].to_numpy()
            s = thermal.schedule(guess, plant, fuel, state)
            profit += thermal.settle(real, s["p"], s["u"], s["v"], plant, fuel)
            starts += int(s["v"].sum())
            state = thermal.next_state(s["u"], state)
        rows.append({"forecast": name, "profit_eur": profit, "profit_eur_per_mw_yr": profit / plant.pmax, "starts": starts})
    vt = pd.DataFrame(rows).set_index("forecast")
    best = vt.loc["perfect", "profit_eur"]
    vt["share_of_perfect_%"] = 100 * vt["profit_eur"] / best
    vt.round(1).to_csv(RES / "ccgt_value_of_forecast.csv")
    out["ccgt"] = vt.round(1).to_dict("index")
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
