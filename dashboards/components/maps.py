"""District choropleths drawn with Altair's geoshape mark (no tile server, so they work offline and in tests)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from dashboards.components import data, theme

SEQ = theme.SEQ_RISK
PADDING = {"left": 14, "right": 14, "top": 14, "bottom": 14}


def choropleth(
    values: pd.DataFrame,
    column: str,
    title: str,
    domain: tuple[float, float] | None = None,
    extra: list[str] | None = None,
    height: int = 520,
) -> alt.Chart:
    """Colour districts by ``values[column]`` (rows keyed by ``district_id``); districts without a value stay grey.

    Args:
        values: DataFrame with ``district_id`` and ``column``.
        column: numeric column to colour by.
        title: legend title.
        domain: colour-scale limits (default: min/max of the data).
        extra: additional columns shown in the tooltip.
        height: chart height in pixels.
    """
    extra = extra or []
    lookup = values.set_index("district_id")[[column, *extra]].to_dict("index")
    features = []
    for feature in data.load_geojson()["features"]:
        props = dict(feature["properties"])
        props.update(lookup.get(props["district_id"], {}))
        features.append({**feature, "properties": props})
    lo, hi = domain or (float(values[column].min()), float(values[column].max()))
    tooltip = [
        alt.Tooltip("properties.district:N", title="District"),
        alt.Tooltip("properties.state:N", title="State"),
        alt.Tooltip(f"properties.{column}:Q", title=title, format=".2f"),
    ]
    tooltip += [alt.Tooltip(f"properties.{name}:N", title=name.replace("_", " ")) for name in extra]
    chart = (
        alt.Chart(alt.Data(values=features))
        .mark_geoshape(stroke=theme.CARD, strokeWidth=0.6)
        .encode(
            color=alt.Color(
                f"properties.{column}:Q",
                title=title,
                scale=alt.Scale(domain=[lo, hi], range=SEQ),
                legend=alt.Legend(orient="right", gradientLength=180),
            ),
            tooltip=tooltip,
        )
        .project(type="mercator")
        .properties(height=height, padding=PADDING)
    )
    return (
        chart.configure(background=theme.CARD)
        .configure_view(stroke=None)
        .configure_legend(labelColor=theme.INK_2, titleColor=theme.INK_2)
    )
