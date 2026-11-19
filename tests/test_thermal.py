import numpy as np
from isem.thermal import CCGT, State, fuel_cost, next_state, schedule, settle

PL = CCGT()
FUEL = fuel_cost(35.0, 70.0)


def test_heat_rate_curve_hits_both_efficiencies():
    assert abs(PL.pmax / (PL.no_load + PL.slope * PL.pmax) - PL.eta_full) < 1e-9
    assert abs(PL.pmin / (PL.no_load + PL.slope * PL.pmin) - PL.eta_min) < 1e-9


def test_stays_off_when_price_below_marginal_cost():
    r = schedule(np.full(24, 60.0), PL, FUEL)
    assert r["u"].sum() == 0 and r["profit"] == 0


def test_runs_through_high_price_block_and_settles_consistently():
    pr = np.r_[np.full(6, 60.0), np.full(12, 110.0), np.full(6, 70.0)]
    r = schedule(pr, PL, FUEL)
    assert r["u"][6:18].all() and r["v"].sum() == 1
    assert abs(settle(pr, r["p"], r["u"], r["v"], PL, FUEL) - r["profit"]) < 1e-3
    assert np.all(r["p"][r["u"] == 1] >= PL.pmin - 1e-6)


def test_short_spike_does_not_pay_for_a_start():
    pr = np.r_[np.full(6, 60.0), 110.0, np.full(17, 60.0)]
    assert schedule(pr, PL, FUEL)["u"].sum() == 0


def test_minimum_down_time_carries_over_from_previous_day():
    pr = np.r_[np.full(3, 200.0), np.full(21, 20.0)]
    r = schedule(pr, PL, FUEL, State(on=0, hours=1))
    assert r["u"][:3].sum() == 0


def test_perfect_foresight_is_an_upper_bound():
    rng = np.random.default_rng(1)
    for _ in range(5):
        pr = 70 + 40 * rng.standard_normal(24)
        guess = 70 + 40 * rng.standard_normal(24)
        best = schedule(pr, PL, FUEL)["profit"]
        g = schedule(guess, PL, FUEL)
        assert settle(pr, g["p"], g["u"], g["v"], PL, FUEL) <= best + 1e-6


def test_next_state_counts_hours():
    u = np.r_[np.zeros(10), np.ones(14)]
    s = next_state(u, State())
    assert s.on == 1 and s.hours == 14
