"""Altair charts used by the dashboard pages (palette and ink tokens shared with the static report figures)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from analytics.predictive.features import HEAT_TMAX_C
from dashboards.components import theme

BLUE, ORANGE = theme.RAIN, theme.HEAT
PADDING = {
    "left": 14,
    "right": 14,
    "top": 14,
    "bottom": 14,
}  # Streamlit's Vega wrapper needs an object
LEVEL_LABEL = {"low": "Low", "moderate": "Moderate", "high": "High", "severe": "Severe"}


def style(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """Apply the shared look: no frame, no axis lines or ticks, one hairline grid, quiet labels."""
    return (
        chart.properties(height=height, padding=PADDING)
        .configure(background=theme.CHART)
        .configure_view(stroke=None)
        .configure_axis(
            domain=False,
            ticks=False,
            gridColor=theme.LINE,
            labelColor=theme.INK_2,
            titleColor=theme.INK_2,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            labelPadding=6,
        )
        .configure_axisX(grid=False)
        .configure_legend(
            labelColor=theme.INK_2,
            titleColor=theme.INK_2,
            orient="top",
            symbolType="stroke",
        )
        .configure_title(color=theme.INK, fontSize=13, anchor="start", fontWeight=600)
    )


def rain_chart(weather: pd.DataFrame) -> alt.Chart:
    """Daily rainfall bars of the recent window."""
    return style(
        alt.Chart(weather)
        .mark_bar(color=BLUE, size=3, cornerRadiusEnd=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("rain_mm:Q", title="Rainfall (mm/day)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("rain_mm:Q", title="Rain (mm)", format=".1f"),
            ],
        ),
        height=170,
    )


def temperature_chart(weather: pd.DataFrame) -> alt.Chart:
    """Daily Tmax and Tmin with the heat threshold used by the heatwave label."""
    long = weather.melt(
        id_vars="date", value_vars=["tmax_c", "tmin_c"], var_name="series", value_name="temp_c"
    )
    long["series"] = long["series"].map({"tmax_c": "Tmax", "tmin_c": "Tmin"})
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("temp_c:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["Tmax", "Tmin"], range=[ORANGE, BLUE]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("temp_c:Q", title="°C", format=".1f"),
            ],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [HEAT_TMAX_C]}))
        .mark_rule(strokeDash=[4, 4], color=theme.MUTED)
        .encode(y="y:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"y": [HEAT_TMAX_C], "t": [f"heat threshold {HEAT_TMAX_C:.0f} °C"]}))
        .mark_text(align="left", dx=4, dy=-6, color=theme.MUTED, fontSize=10)
        .encode(y="y:Q", text="t:N", x=alt.value(2))
    )
    return style(lines + threshold + label, height=190)


def probability_chart(probabilities: dict[str, float]) -> alt.Chart:
    """Horizontal bars of hazard probabilities (0-100 %), direct-labelled."""
    frame = pd.DataFrame({"hazard": list(probabilities), "p": list(probabilities.values())})
    bars = (
        alt.Chart(frame)
        .mark_bar(color=ORANGE, size=16, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("hazard:N", title=None, sort=list(probabilities)),
            x=alt.X(
                "p:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5),
            ),
            tooltip=[
                alt.Tooltip("hazard:N", title="Hazard"),
                alt.Tooltip("p:Q", title="Probability", format=".1%"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=4, color=theme.INK_2, fontSize=11).encode(
        text=alt.Text("p:Q", format=".0%")
    )
    return style(bars + text, height=40 + 34 * len(frame))


def history_chart(history: pd.DataFrame, forecast: dict | None, year: int | None) -> alt.Chart:
    """Observed yield history (line + points) with the forecast and its 80 % interval at the reference year."""
    line = (
        alt.Chart(history)
        .mark_line(
            strokeWidth=2, color=BLUE, point=alt.OverlayMarkDef(size=45, filled=True, color=BLUE)
        )
        .encode(
            x=alt.X("year:O", title="Crop year"),
            y=alt.Y("yield_t_per_ha:Q", title="Yield (t/ha)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("yield_t_per_ha:Q", title="Observed (t/ha)", format=".2f"),
            ],
        )
    )
    layers = [line]
    if forecast is not None and year is not None:
        point = pd.DataFrame(
            {
                "year": [year],
                "yield": [forecast["yield_t_ha"]],
                "lo": [forecast["lo"]],
                "hi": [forecast["hi"]],
            }
        )
        layers.append(
            alt.Chart(point)
            .mark_rule(strokeWidth=3, color=ORANGE)
            .encode(x="year:O", y="lo:Q", y2="hi:Q")
        )
        layers.append(
            alt.Chart(point)
            .mark_point(size=110, filled=True, color=ORANGE, stroke=theme.CHART, strokeWidth=2)
            .encode(
                x="year:O",
                y="yield:Q",
                tooltip=[
                    alt.Tooltip("yield:Q", title="Forecast (t/ha)", format=".2f"),
                    alt.Tooltip("lo:Q", title="80% low", format=".2f"),
                    alt.Tooltip("hi:Q", title="80% high", format=".2f"),
                ],
            )
        )
    return style(alt.layer(*layers), height=280)


def contributions_chart(contributions: list[dict]) -> alt.Chart:
    """SHAP contributions in t/ha: blue raises, orange lowers the forecast relative to the base value."""
    frame = pd.DataFrame(contributions)
    frame["direction"] = frame["contribution_t_ha"].map(lambda v: "raises" if v >= 0 else "lowers")
    order = frame["feature"].tolist()
    return style(
        alt.Chart(frame)
        .mark_bar(size=14, cornerRadiusEnd=3)
        .encode(
            y=alt.Y("feature:N", title=None, sort=order),
            x=alt.X("contribution_t_ha:Q", title="Effect on forecast (t/ha)"),
            color=alt.Color(
                "direction:N",
                title=None,
                scale=alt.Scale(domain=["raises", "lowers"], range=[theme.RAIN, theme.HEAT]),
                legend=alt.Legend(symbolType="square"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
                alt.Tooltip("contribution_t_ha:Q", title="Effect (t/ha)", format="+.3f"),
            ],
        ),
        height=40 + 28 * len(frame),
    )


def crop_ranking_chart(ranking: pd.DataFrame) -> alt.Chart:
    """Expected net return per hectare (interval midpoint) by crop, best first."""
    order = ranking.sort_values("net_mid_inr_ha", ascending=False)["crop"].tolist()
    bars = (
        alt.Chart(ranking)
        .mark_bar(color=theme.PRIMARY, size=22, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("crop:N", title=None, sort=order),
            x=alt.X("net_mid_inr_ha:Q", title="Expected net return (INR/ha, approximate)"),
            tooltip=[
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip("net_mid_inr_ha:Q", title="Net return (INR/ha)", format=",.0f"),
            ],
        )
    )
    return style(bars, height=40 + 40 * len(ranking))
