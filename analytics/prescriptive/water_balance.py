"""FAO-56 style daily root-zone water balance and FAO-33 yield response.

All functions are pure and vectorised over scenarios: arrays have shape ``(n_scenarios, n_days)`` (1-D inputs
are treated as one scenario). Equations:

* Extra-terrestrial radiation Ra from latitude and day of year (FAO-56 eq. 21-25), in mm/day of evaporation.
* Reference evapotranspiration, Hargreaves-Samani: ``ET0 = 0.0023 (Tmean + 17.8) sqrt(Tmax - Tmin) Ra``.
* Crop evapotranspiration ``ETc = Kc ET0`` with a four-stage Kc curve.
* Root-zone depletion ``Dr``: ``Dr_t = Dr_{t-1} - Pe_t - I_t + ETa_t + DP_t`` where ``ETa = Ks ETc``,
  ``Ks = (TAW - Dr) / ((1 - p) TAW)`` when ``Dr > p TAW`` (else 1), and water above field capacity drains (DP).
* Yield response: ``Ya/Ym = prod_s (1 - Ky_s (1 - sum ETa_s / sum ETc_s))`` over the four stages.

Ponded fields (paddy rice, D-029): depletion may go *negative* down to ``-pond_max`` (standing water behind the bunds);
water above that spills. While water stands, ``min(percolation, standing water)`` is lost to seepage and percolation each
day. Without the pond parameters the model is the plain root-zone balance above, bit for bit.

Mass is conserved exactly: the change in soil water equals effective rain + irrigation - ETa - drainage - percolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analytics.prescriptive.params import CropParams, Globals, load_globals

GSC = 0.0820  # solar constant, MJ m-2 min-1
LAMBDA_INV = 0.408  # MJ m-2 day-1 -> mm day-1 of evaporation


def extraterrestrial_radiation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Ra in mm/day for a latitude (degrees, + north) and day-of-year array (FAO-56 eq. 21, 22, 23, 24, 25)."""
    phi = np.deg2rad(lat_deg)
    doy = np.asarray(doy, float)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra_mj = (
        (24.0 * 60.0 / np.pi)
        * GSC
        * dr
        * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    )
    return ra_mj * LAMBDA_INV


def hargreaves_et0(tmax: np.ndarray, tmin: np.ndarray, ra_mm: np.ndarray) -> np.ndarray:
    """Reference evapotranspiration (mm/day), Hargreaves-Samani; Tmax - Tmin is floored at 0."""
    tmax, tmin = np.asarray(tmax, float), np.asarray(tmin, float)
    tmean = (tmax + tmin) / 2.0
    return np.maximum(0.0023 * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.0)) * ra_mm, 0.0)


def kc_curve(crop: CropParams, n_days: int | None = None) -> np.ndarray:
    """Daily Kc: constant ``kc_ini``, linear rise to ``kc_mid`` over development, constant, linear fall to ``kc_end``."""
    ini, dev, mid, late = crop.stage_days
    kc = np.concatenate(
        [
            np.full(ini, crop.kc_ini),
            np.linspace(crop.kc_ini, crop.kc_mid, dev + 1)[1:],
            np.full(mid, crop.kc_mid),
            np.linspace(crop.kc_mid, crop.kc_end, late + 1)[1:],
        ]
    )
    return kc if n_days is None else kc[:n_days]


def stage_index(crop: CropParams) -> np.ndarray:
    """Stage number (0 initial, 1 development, 2 mid, 3 late) of every season day."""
    return np.repeat(np.arange(4), crop.stage_days)


def effective_rainfall(rain: np.ndarray, efficiency: float) -> np.ndarray:
    """Rain that enters the root zone (the rest is runoff / interception)."""
    return np.asarray(rain, float) * efficiency


@dataclass
class WaterBalance:
    """Daily results, each array (n_scenarios, n_days) except the scalars."""

    eta: np.ndarray
    etc: np.ndarray
    peff: np.ndarray
    irrigation: np.ndarray
    drainage: (
        np.ndarray
    )  # deep drainage above field capacity, or spill over the bunds of a ponded field
    depletion: np.ndarray  # root-zone depletion at the END of each day (negative = standing water)
    ks: np.ndarray
    taw: float
    initial_depletion: float
    percolation: np.ndarray | None = (
        None  # seepage and percolation of a ponded field (zeros elsewhere)
    )
    pond_max: float = 0.0

    def soil_water(self) -> np.ndarray:
        """Stored water (TAW - depletion) at the end of each day; above TAW only while water stands in a ponded field."""
        return self.taw - self.depletion

    def pond(self) -> np.ndarray:
        """Standing water depth (mm) at the end of each day (zero unless the field is ponded and flooded)."""
        return np.maximum(-self.depletion, 0.0)

    def mass_balance_residual(self) -> np.ndarray:
        """Per scenario: change in storage minus (effective rain + irrigation - ETa - drainage - percolation)."""
        storage_change = (self.taw - self.depletion[:, -1]) - (self.taw - self.initial_depletion)
        loss = 0.0 if self.percolation is None else self.percolation
        flows = (self.peff + self.irrigation - self.eta - self.drainage - loss).sum(axis=1)
        return storage_change - flows


def simulate(
    et0: np.ndarray,
    rain: np.ndarray,
    irrigation: np.ndarray,
    kc: np.ndarray,
    taw: float,
    p: float,
    initial_depletion: float | None = None,
    glob: Globals | None = None,
    pond_max: float = 0.0,
    percolation: float = 0.0,
) -> WaterBalance:
    """Run the daily water balance for one or many scenarios.

    Args:
        et0, rain, irrigation: ``(n, T)`` or ``(T,)`` arrays in mm/day (irrigation = net depth applied that day).
        kc: ``(T,)`` crop coefficients.
        taw: total available water of the root zone, mm.
        p: allowable depletion fraction before stress.
        initial_depletion: depletion at the start (mm); default ``initial_depletion_fraction x TAW``.
        glob: shared settings (rain efficiency etc.).
        pond_max: bund height in mm for a ponded field (0 = ordinary root-zone balance).
        percolation: seepage and percolation loss in mm/day while water stands (only with ``pond_max`` > 0).
    """
    glob = glob or load_globals()
    et0 = np.atleast_2d(np.asarray(et0, float))
    rain = np.atleast_2d(np.asarray(rain, float))
    irr = np.atleast_2d(np.asarray(irrigation, float)) * glob.irrigation_efficiency
    n = max(et0.shape[0], rain.shape[0], irr.shape[0])
    t_len = et0.shape[1]
    et0, rain, irr = (np.broadcast_to(a, (n, t_len)) for a in (et0, rain, irr))
    d0 = (
        glob.initial_depletion_fraction * taw
        if initial_depletion is None
        else float(initial_depletion)
    )
    peff = effective_rainfall(rain, glob.rain_efficiency)
    raw = p * taw
    depletion = np.full(n, d0)
    eta, etc = np.zeros((n, t_len)), np.zeros((n, t_len))
    drainage, dr_hist, ks_hist = np.zeros((n, t_len)), np.zeros((n, t_len)), np.zeros((n, t_len))
    perc_hist = np.zeros((n, t_len))
    for t in range(t_len):
        ks = np.where(depletion > raw, np.clip((taw - depletion) / (taw - raw), 0.0, 1.0), 1.0)
        etc_t = kc[t] * et0[:, t]
        eta_t = ks * etc_t
        after_inputs = depletion - peff[:, t] - irr[:, t]
        if pond_max > 0.0:  # seepage only draws on standing water
            perc_hist[:, t] = np.minimum(percolation, np.maximum(-after_inputs, 0.0))
        unclipped = after_inputs + eta_t + perc_hist[:, t]
        drainage[:, t] = np.maximum(-unclipped - pond_max, 0.0)
        depletion = np.clip(unclipped, -pond_max, taw)
        eta[:, t], etc[:, t], dr_hist[:, t], ks_hist[:, t] = eta_t, etc_t, depletion, ks
    return WaterBalance(
        eta,
        etc,
        peff,
        irr,
        drainage,
        dr_hist,
        ks_hist,
        float(taw),
        float(d0),
        perc_hist,
        float(pond_max),
    )


def stage_factors(eta: np.ndarray, etc: np.ndarray, crop: CropParams) -> np.ndarray:
    """FAO-33 factor of each growth stage, shape ``(n_scenarios, 4)``: ``1 - Ky_s (1 - ETa_s / ETc_s)`` clipped to [0, 1].

    1 means the stage lost nothing; the product over the four stages is :func:`relative_yield`.
    """
    stage = stage_index(crop)
    eta, etc = np.atleast_2d(eta), np.atleast_2d(etc)
    out = np.ones((eta.shape[0], 4))
    for s in range(4):
        sel = stage == s
        et_c, et_a = etc[:, sel].sum(axis=1), eta[:, sel].sum(axis=1)
        ratio = np.where(et_c > 0, et_a / np.maximum(et_c, 1e-9), 1.0)
        out[:, s] = np.clip(1.0 - crop.ky[s] * (1.0 - ratio), 0.0, 1.0)
    return out


def relative_yield(eta: np.ndarray, etc: np.ndarray, crop: CropParams) -> np.ndarray:
    """FAO-33 multiplicative stage response ``Ya/Ym`` in [0, 1] per scenario."""
    return np.clip(stage_factors(eta, etc, crop).prod(axis=1), 0.0, 1.0)


def to_weekly(daily: np.ndarray, n_weeks: int | None = None) -> np.ndarray:
    """Sum a daily array ``(..., T)`` into weeks ``(..., n_weeks)`` (T must be a multiple of 7 up to ``n_weeks``)."""
    daily = np.asarray(daily, float)
    weeks = daily.shape[-1] // 7 if n_weeks is None else n_weeks
    return daily[..., : weeks * 7].reshape(*daily.shape[:-1], weeks, 7).sum(axis=-1)


def weekly_to_daily_irrigation(weekly_mm: np.ndarray, n_days: int) -> np.ndarray:
    """Apply each week's irrigation on its first day (a single event), zeros elsewhere."""
    weekly_mm = np.asarray(weekly_mm, float)
    daily = np.zeros(weekly_mm.shape[:-1] + (n_days,))
    for w in range(weekly_mm.shape[-1]):
        if 7 * w < n_days:
            daily[..., 7 * w] = weekly_mm[..., w]
    return daily
