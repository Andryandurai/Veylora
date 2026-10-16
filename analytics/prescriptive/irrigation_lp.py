"""Weekly irrigation scheduling as a linear program (PuLP + CBC).

Decision variables per week ``w``: irrigation ``I_w`` (mm), weekly evapotranspiration ``E_w`` (ETa), start-of-week
soil water ``S_w`` and drainage ``DP_w``. The model is a linearisation of the daily FAO-56 balance:

    S_{w+1} = S_w + Pe_w + I_w - E_w - DP_w,        0 <= S_{w+1} <= TAW
    E_w <= ETc_w                                     (cannot exceed the crop demand)
    E_w <= (ETc_w / S_crit) (S_w + Pe_w + I_w)       (piecewise-linear Ks: stress once storage < S_crit = (1-p) TAW)
    E_w <= S_w + Pe_w + I_w                          (cannot use water that is not there)
    sum_w I_w <= budget,    0 <= I_w <= weekly_cap

Ponded paddy (D-029): the storage bound becomes ``TAW + pond depth`` and every week loses ``perc_w`` mm to seepage and
percolation (assuming the field is kept flooded). The loss cannot exceed the water that is there: a slack ``U_w`` returns
the part of ``perc_w`` that is not available, and a large penalty keeps ``U_w`` at zero unless the water really is short.

Objective: maximise ``sum_w (Ky_s(w) / ETc_s) E_w``, which is the first-order (additive) form of the FAO-33
relative yield ``1 - sum_s Ky_s (1 - ETa_s / ETc_s)``, minus a tiny penalty on irrigation so that water is only
used when it buys yield. The LP is a *planning* model; every schedule (the LP's and the baselines') is then
scored with the nonlinear daily simulator in :mod:`analytics.prescriptive.water_balance`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pulp

from analytics.prescriptive.params import CropParams
from analytics.prescriptive.water_balance import stage_index

LOG = logging.getLogger(__name__)

WATER_PENALTY = 1e-5  # per mm of irrigation; far below the value of a mm of ETa (~1e-3 .. 1e-2)
UNREALISED_LOSS_PENALTY = (
    1.0  # per mm of percolation that the model "did not lose": only used when water is really short
)
SOLVER_TIME_LIMIT_S = 5


@dataclass
class LPInputs:
    """Weekly data for the planning model (all arrays have one entry per week)."""

    etc_w: np.ndarray
    peff_w: np.ndarray
    stage_w: np.ndarray  # stage (0..3) of each week
    ky: tuple[float, float, float, float]
    taw: float
    p: float
    s0: float  # soil water at the start of week 0
    perc_w: float = 0.0  # weekly seepage and percolation of a ponded field (mm); 0 for upland crops
    cap: float | None = None  # storage upper bound: TAW plus the pond depth (None = TAW)

    @property
    def n_weeks(self) -> int:
        return len(self.etc_w)

    @property
    def storage_cap(self) -> float:
        return self.taw if self.cap is None else self.cap

    @property
    def s_crit(self) -> float:
        return (1.0 - self.p) * self.taw


@dataclass
class LPResult:
    """Solution of the planning LP."""

    schedule_mm: np.ndarray
    eta_w: np.ndarray
    storage: np.ndarray
    objective: (
        float  # sum_w weight_w * E_w from the LP's own variables (yield proxy, higher is better)
    )
    penalized_objective: float  # what the solver maximised: objective - WATER_PENALTY * water used
    status: str
    solve_seconds: float

    @property
    def water_used_mm(self) -> float:
        return float(self.schedule_mm.sum())


def build_inputs(
    crop: CropParams,
    etc_daily: np.ndarray,
    peff_daily: np.ndarray,
    taw: float,
    initial_depletion: float,
) -> LPInputs:
    """Aggregate daily ETc and effective rain (``(T,)``) into the weekly LP inputs."""
    n_weeks = len(etc_daily) // 7
    weekly = lambda a: np.asarray(a, float)[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)  # noqa: E731
    stage_w = stage_index(crop)[3 : n_weeks * 7 : 7]  # stage of each week's middle day
    return LPInputs(
        weekly(etc_daily),
        weekly(peff_daily),
        stage_w,
        crop.ky,
        taw,
        crop.p,
        taw - initial_depletion,
        perc_w=7.0 * crop.percolation_mm_day,
        cap=taw + crop.pond_max_mm if crop.ponded else None,
    )


def stage_weights(inputs: LPInputs) -> np.ndarray:
    """Objective weight of one mm of ETa in each week: ``Ky_s / sum of ETc over the weeks of stage s``."""
    weights = np.zeros(inputs.n_weeks)
    for s in range(4):
        sel = inputs.stage_w == s
        total = inputs.etc_w[sel].sum()
        if total > 0:
            weights[sel] = inputs.ky[s] / total
    return weights


def max_weekly_eta(
    etc: float, available: float, taw: float, s_crit: float, stress_basis: str = "end"
) -> float:
    """Largest weekly ETa the LP constraints allow for given available water (closed form, per stress basis).

    The stress constraint ``E <= (ETc / S_crit) * basis`` uses start-of-week water (``start``), end-of-week water
    ``S_next = min(TAW, available - E)`` (``end``) or their average (``mid``); the other limits are ``E <= ETc``
    and ``E <= available``.
    """
    c = etc / s_crit
    if stress_basis == "start":
        limit = c * available
    elif stress_basis == "end":
        limit = c * taw if available >= taw * (1.0 + c) else c * available / (1.0 + c)
    else:  # mid
        upper = c * (available + taw) / 2.0
        limit = upper if upper <= available - taw else c * available / (1.0 + c / 2.0)
    return float(max(min(etc, available, limit), 0.0))


def penalized_value(inputs: LPInputs, schedule: np.ndarray, stress_basis: str = "end") -> float:
    """Greedy objective of a schedule minus the water penalty: comparable with ``LPResult.penalized_objective``."""
    return schedule_objective(inputs, schedule, stress_basis) - WATER_PENALTY * float(
        np.sum(schedule)
    )


def schedule_objective(inputs: LPInputs, schedule: np.ndarray, stress_basis: str = "end") -> float:
    """Value of the LP objective (without the water penalty) for a given weekly schedule.

    Evapotranspiration is taken greedily (as much as the model allows each week, see :func:`max_weekly_eta`) and
    surplus above field capacity drains. That ``E_w`` is feasible for the LP, so the LP's *own* optimum
    (``LPResult.penalized_objective`` versus :func:`penalized_value`) is at least this value: a baseline schedule
    cannot beat the LP under the same model. (Greedy-evaluating the LP's schedule is not the LP optimum, because the
    LP may deliberately hold water back for a more sensitive stage.)
    """
    weights = stage_weights(inputs)
    storage, value, cap = inputs.s0, 0.0, inputs.storage_cap
    for w in range(inputs.n_weeks):
        available = storage + inputs.peff_w[w] + schedule[w]
        available -= min(
            inputs.perc_w, available
        )  # seepage and percolation come first and cannot exceed the water there
        eta = max_weekly_eta(inputs.etc_w[w], available, cap, inputs.s_crit, stress_basis)
        value += weights[w] * eta
        storage = min(cap, available - eta)
    return float(value)


def solve_irrigation(
    inputs: LPInputs, budget_mm: float, weekly_cap_mm: float, stress_basis: str = "end"
) -> LPResult:
    """Solve the planning LP; asserts that the budget and weekly cap hold in the returned schedule.

    ``stress_basis`` chooses the storage that drives the piecewise-linear stress constraint:
    ``"start"`` = start-of-week storage plus the week's inflows (optimistic: ignores depletion inside the week),
    ``"end"`` = end-of-week storage (conservative: the lowest point of the week when irrigation and rain arrive
    first), ``"mid"`` = the average of the two.
    """
    if stress_basis not in ("start", "mid", "end"):
        raise ValueError("stress_basis must be 'start', 'mid' or 'end'")
    if budget_mm < 0 or weekly_cap_mm < 0:
        raise ValueError("budget and weekly cap must be non-negative")
    n = inputs.n_weeks
    weights = stage_weights(inputs)
    prob = pulp.LpProblem("irrigation", pulp.LpMaximize)
    irr = [pulp.LpVariable(f"I_{w}", 0, weekly_cap_mm) for w in range(n)]
    eta = [pulp.LpVariable(f"E_{w}", 0, float(inputs.etc_w[w])) for w in range(n)]
    storage = [pulp.LpVariable(f"S_{w}", 0, inputs.storage_cap) for w in range(n + 1)]
    drain = [pulp.LpVariable(f"DP_{w}", 0) for w in range(n)]
    slack = (
        [pulp.LpVariable(f"U_{w}", 0, inputs.perc_w) for w in range(n)] if inputs.perc_w > 0 else []
    )
    prob += (
        pulp.lpSum(float(weights[w]) * eta[w] for w in range(n))
        - WATER_PENALTY * pulp.lpSum(irr)
        - UNREALISED_LOSS_PENALTY * pulp.lpSum(slack)
    )
    prob += storage[0] == float(inputs.s0)
    for w in range(n):
        available = storage[w] + float(inputs.peff_w[w]) + irr[w]
        loss = float(inputs.perc_w) - slack[w] if slack else 0.0
        prob += storage[w + 1] == available - loss - eta[w] - drain[w]
        basis = {
            "start": available,
            "end": storage[w + 1],
            "mid": 0.5 * (available + storage[w + 1]),
        }[stress_basis]
        prob += eta[w] <= (float(inputs.etc_w[w]) / inputs.s_crit) * basis
        prob += eta[w] <= available - loss
    prob += pulp.lpSum(irr) <= budget_mm
    started = time.perf_counter()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVER_TIME_LIMIT_S))
    elapsed = time.perf_counter() - started
    name = pulp.LpStatus[status]
    if name != "Optimal":
        raise RuntimeError(f"irrigation LP did not solve to optimality: {name}")
    schedule = np.clip(np.array([v.value() or 0.0 for v in irr]), 0.0, weekly_cap_mm)
    if (
        schedule.sum() > budget_mm
    ):  # CBC feasibility tolerance (~1e-7): renormalise the numerical overshoot
        schedule = schedule * (budget_mm / schedule.sum())
    assert schedule.sum() <= budget_mm + 1e-9, "LP violated the seasonal budget"
    assert (schedule <= weekly_cap_mm + 1e-9).all(), "LP violated the weekly cap"
    eta_values = np.array([v.value() or 0.0 for v in eta])
    objective = float(np.dot(weights, eta_values))
    return LPResult(
        schedule_mm=schedule,
        eta_w=eta_values,
        storage=np.array([v.value() or 0.0 for v in storage]),
        objective=objective,
        penalized_objective=objective - WATER_PENALTY * float(schedule.sum()),
        status=name,
        solve_seconds=elapsed,
    )


def even_split(budget_mm: float, weekly_cap_mm: float, n_weeks: int) -> np.ndarray:
    """Baseline: the budget spread evenly over all weeks (never above the weekly cap)."""
    return np.full(n_weeks, min(budget_mm / n_weeks, weekly_cap_mm))


def fixed_interval_daily(n_days: int, depth_mm: float = 50.0, every_days: int = 10) -> np.ndarray:
    """Farmer-baseline schedule: ``depth_mm`` every ``every_days`` days starting on day 0 (daily array)."""
    daily = np.zeros(n_days)
    daily[::every_days] = depth_mm
    return daily
