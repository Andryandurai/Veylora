"""Fixed-effects regressions of yield on weather and vegetation, per crop.

Specifications (OLS via statsmodels, standard errors clustered by district):

* **M1 two-way FE, weather**: yield ~ rain + Tmax + district FE + year FE (the plan's model).
* **M2 two-way FE, weather + satellite**: M1 plus peak NDVI and mean soil moisture.
* **M3 district FE + trend, weather**: yield ~ rain + Tmax + district FE + common linear trend.

Why M3: year fixed effects absorb every shock shared by all districts in a year (a region-wide
drought such as 2009), so M1 identifies weather effects only from *differences between districts in
the same year*. M3 lets common year-to-year weather variation identify the effect while still
removing district levels and the technology trend. Compare M1 and M3 before drawing conclusions.

Diagnostics: VIF on the fixed-effects-residualised regressors, partial R-squared of the weather terms,
Breusch-Pagan test, and a residual-versus-fitted figure. Satellite terms are flagged when the satellite
data is synthetic (they are then rainfall transforms, not independent evidence).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

from analytics import common

LOG = logging.getLogger(__name__)

SCALES = {
    "rain_100mm": ("total_rain", 1 / 100.0, "per +100 mm season rainfall"),
    "tmax_c": ("mean_tmax", 1.0, "per +1 C mean Tmax"),
    "ndvi_01": ("ndvi_peak", 10.0, "per +0.1 peak NDVI"),
    "soilm_01": ("soil_moisture_mean", 10.0, "per +0.1 m3/m3 soil moisture"),
}
SPECS = {
    "M1_twoway_FE_weather": (["rain_100mm", "tmax_c"], "C(district_id) + C(year)"),
    "M2_twoway_FE_weather_satellite": (
        ["rain_100mm", "tmax_c", "ndvi_01", "soilm_01"],
        "C(district_id) + C(year)",
    ),
    "M3_district_FE_trend_weather": (["rain_100mm", "tmax_c"], "C(district_id) + year_c"),
}


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Add scaled regressors and a centred year trend."""
    out = data.copy()
    for name, (source, factor, _) in SCALES.items():
        out[name] = out[source] * factor
    out["year_c"] = out["year"] - out["year"].mean()
    return out


def within_vif(frame: pd.DataFrame, regressors: list[str], fixed_effects: str) -> dict[str, float]:
    """VIF of each regressor after partialling out the fixed effects (the collinearity that matters in an FE model)."""
    dummies = patsy_design(frame, fixed_effects)
    resid = {}
    for name in regressors:
        beta, *_ = np.linalg.lstsq(dummies, frame[name].to_numpy(float), rcond=None)
        resid[name] = frame[name].to_numpy(float) - dummies @ beta
    matrix = np.column_stack([resid[n] for n in regressors])
    vifs = {}
    for j, name in enumerate(regressors):
        others = np.delete(matrix, j, axis=1)
        if others.shape[1] == 0:
            vifs[name] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(others, matrix[:, j], rcond=None)
        fitted = others @ beta
        denom = float(((matrix[:, j] - matrix[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float(((matrix[:, j] - fitted) ** 2).sum()) / denom if denom > 0 else 0.0
        vifs[name] = float(1.0 / max(1.0 - r2, 1e-9))
    return vifs


def patsy_design(frame: pd.DataFrame, rhs: str) -> np.ndarray:
    """Design matrix (with intercept) for the fixed-effects part of a formula."""
    import patsy

    return np.asarray(patsy.dmatrix(rhs, frame, return_type="dataframe"))


def fit_model(frame: pd.DataFrame, regressors: list[str], fixed_effects: str):
    """Fit OLS with district-clustered standard errors; returns the statsmodels results."""
    formula = "yield_label ~ " + " + ".join(regressors) + " + " + fixed_effects
    groups = pd.Categorical(frame["district_id"]).codes
    return smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": groups})


def run_specs(
    data: pd.DataFrame, crop: str, satellite_synthetic: bool
) -> tuple[pd.DataFrame, dict, dict]:
    """Fit every spec for one crop; returns (coefficient rows, diagnostics, residual frames)."""
    frame = prepare(data[data["crop"] == crop])
    rows, diagnostics, residuals = [], {}, {}
    for spec, (regressors, fixed_effects) in SPECS.items():
        needed = ["yield_label", "district_id", "year", "year_c", *regressors]
        sample = frame.dropna(subset=needed).reset_index(drop=True)
        if len(sample) < 50 or sample["district_id"].nunique() < 5:
            continue
        result = fit_model(sample, regressors, fixed_effects)
        base = smf.ols("yield_label ~ " + fixed_effects, data=sample).fit()
        partial_r2 = float(result.rsquared - base.rsquared)
        vifs = within_vif(sample, regressors, fixed_effects)
        _, bp_p, _, _ = het_breuschpagan(result.resid, result.model.exog)
        conf = result.conf_int()
        for term in regressors:
            within_sd = float(
                np.std(sample[term] - sample.groupby("district_id")[term].transform("mean"))
            )
            rows.append(
                {
                    "crop": crop,
                    "model": spec,
                    "term": term,
                    "unit": SCALES[term][2],
                    "coef": float(result.params[term]),
                    "se_cluster": float(result.bse[term]),
                    "t": float(result.tvalues[term]),
                    "p": float(result.pvalues[term]),
                    "ci_lo": float(conf.loc[term, 0]),
                    "ci_hi": float(conf.loc[term, 1]),
                    "pct_of_mean_yield_per_1sd_within": 100.0
                    * float(result.params[term])
                    * within_sd
                    / float(sample["yield_label"].mean()),
                    "vif_within": vifs[term],
                    "n": int(len(sample)),
                    "n_districts": int(sample["district_id"].nunique()),
                    "r2": float(result.rsquared),
                    "partial_r2_weather": partial_r2,
                    "synthetic_regressor": bool(
                        satellite_synthetic and term in {"ndvi_01", "soilm_01"}
                    ),
                }
            )
        diagnostics[spec] = {
            "breusch_pagan_p": float(bp_p),
            "r2": float(result.rsquared),
            "partial_r2": partial_r2,
            "n": int(len(sample)),
        }
        residuals[spec] = pd.DataFrame({"fitted": result.fittedvalues, "resid": result.resid})
    return pd.DataFrame(rows), diagnostics, residuals


def draw_residuals(residual_frames: dict[str, pd.DataFrame]) -> None:
    """Residual versus fitted for the two-way FE weather model (M1), one panel per crop."""
    common.apply_mpl_theme()
    crops = list(residual_frames)
    colors = common.crop_colors()
    fig, axes = plt.subplots(1, len(crops), figsize=(3.6 * len(crops), 3.6), squeeze=False)
    for ax, crop in zip(axes[0], crops, strict=True):
        frame = residual_frames[crop]
        ax.scatter(
            frame["fitted"], frame["resid"], s=9, alpha=0.45, color=colors[crop], edgecolors="none"
        )
        ax.axhline(0, color=common.AXIS, linewidth=1)
        ax.set_title(crop.capitalize())
        ax.set_xlabel("fitted yield, t/ha")
    axes[0][0].set_ylabel("residual, t/ha")
    fig.suptitle(
        "Residuals vs fitted, two-way fixed-effects model M1", x=0.01, ha="left", fontweight="bold"
    )
    common.save_mpl(fig, "regression_residuals")


def run() -> dict:
    """Fit all models, save the coefficient table, VIFs and residual plot."""
    train = common.training_table()
    settings = common.load_settings()
    data = train[
        (train["year"] >= int(settings["years"]["start"]))
        & (train["year"] <= int(settings["years"]["end"]))
    ]
    synthetic = common.synthetic_satellite_used(data)
    tables, diagnostics, resid_m1 = [], {}, {}
    for crop in settings["crops"]:
        if not (data["crop"] == crop).any():
            continue
        table, diag, residuals = run_specs(data, crop, synthetic)
        if table.empty:
            continue
        tables.append(table)
        diagnostics[crop] = diag
        if "M1_twoway_FE_weather" in residuals:
            resid_m1[crop] = residuals["M1_twoway_FE_weather"]
    coefficients = pd.concat(tables, ignore_index=True)
    common.save_table(coefficients, "regression_coefficients", floatfmt=4)
    draw_residuals(resid_m1)

    def pick(crop: str, model: str, term: str) -> dict:
        row = coefficients[
            (coefficients["crop"] == crop)
            & (coefficients["model"] == model)
            & (coefficients["term"] == term)
        ]
        return row.iloc[0].to_dict() if len(row) else {}

    results = {
        "period": [int(data["year"].min()), int(data["year"].max())],
        "satellite_features_synthetic": synthetic,
        "standard_errors": "clustered by district",
        "diagnostics": diagnostics,
        "max_vif_within": {
            m: float(coefficients[coefficients["model"] == m]["vif_within"].max()) for m in SPECS
        },
        "rice": {
            m: {t: pick("rice", m, t) for t in ("rain_100mm", "tmax_c", "ndvi_01", "soilm_01")}
            for m in SPECS
        },
        "coefficients": coefficients.to_dict("records"),
    }
    common.save_results("regression", results)
    return results


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
