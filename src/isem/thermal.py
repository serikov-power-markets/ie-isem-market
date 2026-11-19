"""Day-ahead schedule of a stylised gas-fired CCGT on the I-SEM day-ahead price (unit commitment per day).

The unit is a generic 400 MW combined-cycle plant (not any real station). For every local day the
owner commits the unit on the day-ahead prices it expects, and it is paid the day-ahead price that
is actually realised for the scheduled output (self-schedule, price-taker; no balancing market,
intraday, gas-nomination or capacity-market revenue, no ramp limits).

    max  sum_t  price_t * p_t - fuel_cost * H_t - vom * p_t - start_cost * v_t
    s.t. H_t = a * u_t + b * p_t              heat input, MWh thermal (affine heat-rate curve)
         pmin * u_t <= p_t <= pmax * u_t
         v_t >= u_t - u_{t-1}                 start-up indicator
         minimum up and down time, carried over from the previous day

Fuel cost per MWh thermal = gas price + emission factor x carbon price. Value of information: the
profit of a schedule built on forecast prices against the profit of the schedule built on the realised
prices (perfect foresight) and of one built on yesterday's prices (naive), all settled at the realised prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

EF_GAS = 0.202     # tCO2 per MWh thermal, natural gas


@dataclass(frozen=True)
class CCGT:
    pmax: float = 400.0        # MW
    pmin: float = 160.0        # MW, minimum stable generation
    eta_full: float = 0.55     # electrical efficiency at pmax
    eta_min: float = 0.50      # ... and at pmin
    start_cost: float = 12000.0  # EUR per start (assumption)
    vom: float = 2.0           # EUR per MWh electric
    min_up: int = 4            # hours
    min_down: int = 4          # hours

    @property
    def slope(self) -> float:  # MWh thermal per MWh electric above the no-load heat
        return (self.pmax / self.eta_full - self.pmin / self.eta_min) / (self.pmax - self.pmin)

    @property
    def no_load(self) -> float:  # MWh thermal per hour at u = 1, p = 0 (extrapolated)
        return self.pmin / self.eta_min - self.slope * self.pmin


@dataclass(frozen=True)
class State:
    on: int = 0            # 1 if running at the end of the previous day
    hours: int = 24        # hours since the last change of state, capped at 24


def fuel_cost(gas: float, co2: float) -> float:
    """EUR per MWh thermal."""
    return gas + EF_GAS * co2


def schedule(prices: np.ndarray, plant: CCGT, fuel: float, state: State = State()) -> dict:
    """Optimal commitment and output for the given hourly prices; returns u, p, starts and expected profit."""
    pr = np.asarray(prices, float)
    T = len(pr)
    n = 3 * T    # p, u, v
    ip, iu, iv = 0, T, 2 * T
    cost = np.zeros(n)
    cost[ip:ip + T] = -(pr - plant.vom - fuel * plant.slope)
    cost[iu:iu + T] = fuel * plant.no_load
    cost[iv:iv + T] = plant.start_cost
    rows, lo, hi = [], [], []

    def add(coefs: dict, l: float, h: float) -> None:
        r = np.zeros(n)
        for k, v in coefs.items():
            r[k] += v
        rows.append(r); lo.append(l); hi.append(h)

    for t in range(T):
        add({ip + t: 1, iu + t: -plant.pmax}, -np.inf, 0)
        add({ip + t: 1, iu + t: -plant.pmin}, 0, np.inf)
        prev = {iu + t - 1: 1} if t > 0 else {}
        if t > 0:
            add({iv + t: 1, iu + t: -1, **prev}, 0, np.inf)       # v_t >= u_t - u_{t-1}
        else:
            add({iv + t: 1, iu + t: -1}, -state.on, np.inf)
        # minimum up time: running for min_up hours after a start
        for k in range(1, plant.min_up):
            if t + k < T:
                add({iu + t + k: 1, iv + t: -1}, 0, np.inf)
        # minimum down time: off for min_down hours after a stop (stop_t = u_{t-1} - u_t)
        for k in range(1, plant.min_down):
            if t + k < T:
                if t > 0:
                    add({iu + t + k: 1, iu + t - 1: 1, iu + t: -1}, -np.inf, 1)  # stop at t -> off at t+k
                else:
                    pass
    # carry-over of minimum up and down time from the previous day
    must_hold = max(0, (plant.min_up if state.on else plant.min_down) - state.hours)
    lb_u = np.zeros(T)
    ub_u = np.ones(T)
    for t in range(min(must_hold, T)):
        if state.on:
            lb_u[t] = 1.0
        else:
            ub_u[t] = 0.0
    lb = np.r_[np.zeros(T), lb_u, np.zeros(T)]
    ub = np.r_[np.full(T, plant.pmax), ub_u, np.ones(T)]
    integrality = np.r_[np.zeros(T), np.ones(T), np.zeros(T)]
    res = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi), bounds=Bounds(lb, ub),
               integrality=integrality)
    if not res.success:
        raise RuntimeError(f"CCGT schedule infeasible: {res.message}")
    x = res.x
    p, u, v = np.where(x[ip:ip + T] < 1e-6, 0.0, x[ip:ip + T]), np.round(x[iu:iu + T]), np.round(x[iv:iv + T])
    return {"p": p, "u": u, "v": v, "profit": -float(res.fun)}


def settle(prices: np.ndarray, p: np.ndarray, u: np.ndarray, v: np.ndarray, plant: CCGT, fuel: float) -> float:
    """Profit of a given schedule at the realised prices."""
    heat = plant.no_load * u + plant.slope * p
    return float(np.sum(prices * p - plant.vom * p - fuel * heat) - plant.start_cost * v.sum())


def next_state(u: np.ndarray, state: State) -> State:
    """State at the end of a day: running or not, and hours since the last change."""
    last = int(u[-1])
    k = 0
    for x in u[::-1]:
        if int(x) == last:
            k += 1
        else:
            break
    hours = k + (state.hours if k == len(u) and last == state.on else 0)
    return State(on=last, hours=min(hours, 24))
