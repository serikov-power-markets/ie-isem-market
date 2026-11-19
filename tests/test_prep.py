import numpy as np
import pandas as pd

from isem import fundamentals as fu, prep


def _frame(n=24 * 40):
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 3000, n)
    return pd.DataFrame({"ts_utc": ts, "price": 150 - 0.03 * wind + rng.normal(0, 3, n), "load": 4500.0,
                         "gen_wind_onshore": wind, "flow_to_gb": 0.0, "flow_from_gb": 500.0})


def test_snsp_counts_wind_and_hvdc_imports():
    d = fu.add_derived(_frame())
    row = d.iloc[0]
    assert abs(row["snsp"] - 100 * (row["wind"] + 500) / 4500) < 1e-9


def test_capture_rate_below_100_when_price_falls_with_wind():
    c = fu.capture(fu.add_derived(_frame()), "wind")
    assert (c["capture_rate_%"] < 100).all()


def test_anomaly_detector_flags_a_spike():
    d = _frame()
    d.loc[700, "price"] = 900.0
    a = prep.anomalies(d)
    assert (a["ts_utc"] == d.loc[700, "ts_utc"]).any()


def test_quality_report_counts_duplicates():
    d = pd.concat([_frame(100), _frame(100).iloc[:3]])
    q = prep.quality_report(d[["ts_utc", "price"]])
    assert int(q[(q.column == "ts_utc") & (q.check == "duplicates")]["value"].iloc[0]) == 3
