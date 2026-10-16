"""Deterministic farmer advisories from model outputs.

``generate(district_id, crop, as_of_date, lang) -> str`` combines

* the Phase 5 early-warning probabilities and level (``risk_forecast``),
* the Phase 6 optimized irrigation plan for the current week of the crop's season (computed under the district's
  *typical-year* weather, so it is a planning value, not a forecast of this season's weather), and
* the crop calendar (sowing window before the season starts),

with fixed templates (``advisory/templates.yaml``). At most two messages are composed and the SMS text is kept
under ``advisory.sms_max_chars`` characters. No language model writes any text here (safety-critical advice).

Translations are drafts that **need native-speaker and agronomist review** before real use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date
from functools import lru_cache
from typing import Any

import pandas as pd
import yaml

from config import ROOT, load_settings

LOG = logging.getLogger(__name__)

TEMPLATES_PATH = ROOT / "advisory" / "templates.yaml"
LEVEL_ORDER = ("low", "moderate", "high", "severe")
GSM7_BASIC = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞ\x1bÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM7_EXTENDED = set("^{}\\[~]|€")


@dataclass
class Advisory:
    """A composed advisory: the final text plus the reasoning that led to it."""

    text: str
    lang: str
    messages: list[str] = field(default_factory=list)  # message types used, in order
    risk: dict[str, Any] = field(default_factory=dict)
    irrigation_mm: float | None = None
    weeks_into_season: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def sms_segments(self) -> int:
        return sms_segments(self.text)


@lru_cache(maxsize=1)
def load_templates() -> dict[str, Any]:
    """Parse templates.yaml once."""
    with TEMPLATES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def settings() -> dict[str, Any]:
    """The ``advisory`` section of settings.yaml."""
    return load_settings()["advisory"]


def is_gsm7(text: str) -> bool:
    """True if every character fits the GSM 03.38 alphabet (basic + extension table)."""
    return all(ch in GSM7_BASIC or ch in GSM7_EXTENDED for ch in text)


def sms_segments(text: str) -> int:
    """Number of SMS parts: GSM-7 uses 160 / 153 characters per part, Unicode (UCS-2) 70 / 67."""
    if not text:
        return 0
    if is_gsm7(text):
        units = sum(2 if ch in GSM7_EXTENDED else 1 for ch in text)
        return 1 if units <= 160 else -(-units // 153)
    return 1 if len(text) <= 70 else -(-len(text) // 67)


def render(
    message: str,
    lang: str,
    district: str,
    crop: str,
    as_of: str,
    risk: str = "low",
    mm: float | None = None,
) -> str:
    """Fill one template with localised crop and risk words."""
    tpl = load_templates()
    if lang not in tpl["languages"]:
        raise ValueError(f"unsupported language {lang!r}; choose from {tpl['languages']}")
    text = tpl["messages"][message][lang]
    return text.format(
        district=district,
        crop=tpl["crops"][crop][lang],
        date=as_of,
        mm="" if mm is None else f"{int(round(mm / 5.0) * 5)}",
        risk=tpl["risk"][lang][risk],
    )


def pretty_district(district_id: str) -> str:
    """'tamil_nadu__thanjavur' -> 'Thanjavur' (display name used in messages)."""
    return district_id.split("__", 1)[-1].replace("_", " ").title()


def season_position(crop: str, as_of: Date) -> tuple[str, int | None, Date]:
    """(season, weeks since the season start or None if outside the 16-week window, next season start)."""
    from analytics.predictive import features as F

    season = load_settings()["crop_season"][crop]
    stamp = pd.Timestamp(as_of)
    weeks = None
    for year in (stamp.year - 1, stamp.year):
        start, _ = F.season_bounds(year, season)
        days = (stamp - start).days
        if 0 <= days < 112:
            weeks = days // 7
    upcoming = min(
        (
            F.season_bounds(year, season)[0]
            for year in (stamp.year, stamp.year + 1)
            if F.season_bounds(year, season)[0] > stamp
        ),
        default=None,
    )
    return season, weeks, (upcoming.date() if upcoming is not None else as_of)


@lru_cache(maxsize=256)
def planned_irrigation(district_id: str, crop: str) -> tuple[float, ...]:
    """Weekly optimized irrigation (mm) for the district's typical year: the Phase 6 LP schedule (planning value)."""
    from analytics.prescriptive import scenario as S
    from analytics.prescriptive.params import load_crop

    season = load_settings()["crop_season"][crop]
    inputs = S.load_season_inputs(district_id, crop, season)
    table, _ = S.evaluate_strategies(inputs, load_crop(crop).default_budget_mm)
    row = table[table["strategy"] == "optimized"].iloc[0]
    return tuple(float(x) for x in row["weekly_irrigation_mm"])


def choose_messages(
    level_by_hazard: dict[str, str],
    week: int | None,
    irrigation_mm: float | None,
    days_to_sowing: int | None,
) -> list[str]:
    """Deterministic rule engine: which message types to send (at most two).

    1. A hazard at or above ``alert_min_level`` produces its alert (heatwave first if both are equal).
    2. Then, if the crop is in its season and the planned irrigation this week is at least ``irrigation_min_mm``,
       the irrigation plan; otherwise, if the season starts within ``sowing_lead_days``, the sowing window.
    3. If nothing applies: ``all_clear``.
    """
    cfg = settings()
    minimum = LEVEL_ORDER.index(cfg["alert_min_level"])
    chosen: list[str] = []
    hazards = sorted(
        level_by_hazard.items(), key=lambda kv: (-LEVEL_ORDER.index(kv[1]), kv[0] != "heatwave")
    )
    for hazard, level in hazards:
        if LEVEL_ORDER.index(level) >= minimum:
            chosen.append("heat_alert" if hazard == "heatwave" else "drought_watch")
            break
    if week is not None and irrigation_mm is not None and irrigation_mm >= cfg["irrigation_min_mm"]:
        chosen.append("irrigation_plan")
    elif days_to_sowing is not None and 0 < days_to_sowing <= cfg["sowing_lead_days"]:
        chosen.append("sowing_window")
    return chosen[:2] or ["all_clear"]


def compose(district_id: str, crop: str, as_of_date: str | Date, lang: str = "en") -> Advisory:
    """Build the advisory with its supporting facts (used by the dashboard, the bot and the tests)."""
    from analytics.predictive.extreme_weather import level_from_probability, risk_forecast

    as_of = pd.Timestamp(as_of_date).date()
    stamp = as_of.isoformat()
    risk = risk_forecast(district_id, stamp)
    thresholds = load_settings()["risk_levels"]
    levels = {
        "heatwave": level_from_probability("heatwave", risk["heatwave_p"], thresholds),
        "drought": level_from_probability("drought", risk["drought_p"], thresholds),
    }
    season, week, upcoming = season_position(crop, as_of)
    irrigation_mm, notes = None, []
    if not risk.get(
        "drought_relevant", True
    ):  # target month is normally almost dry: SPI is not meaningful
        levels["drought"] = "low"
        notes.append("drought watch suppressed: the coming month is normally almost dry")
    if week is not None:
        try:
            irrigation_mm = planned_irrigation(district_id, crop)[min(week, 15)]
        except (ValueError, KeyError, FileNotFoundError) as exc:
            notes.append(f"irrigation plan unavailable: {exc}")
    days_to_sowing = (upcoming - as_of).days if week is None else None
    kinds = choose_messages(levels, week, irrigation_mm, days_to_sowing)
    name = pretty_district(district_id)
    parts = []
    for kind in kinds:
        hazard_level = levels["heatwave"] if kind == "heat_alert" else levels["drought"]
        parts.append(
            render(
                kind,
                lang,
                name,
                crop,
                upcoming.isoformat() if kind == "sowing_window" else stamp,
                hazard_level,
                irrigation_mm,
            )
        )
    limit = int(settings()["sms_max_chars"])
    text = " ".join(parts)
    if len(text) > limit and len(parts) > 1:
        notes.append("second message dropped to stay within the SMS length limit")
        parts, kinds = parts[:1], kinds[:1]
        text = parts[0]
    if len(text) > limit:
        raise ValueError(
            f"advisory of {len(text)} characters exceeds the {limit}-character limit ({kinds}, {lang})"
        )
    return Advisory(
        text=text,
        lang=lang,
        messages=kinds,
        risk={**risk, "levels": levels},
        irrigation_mm=irrigation_mm,
        weeks_into_season=week,
        notes=notes,
    )


def generate(district_id: str, crop: str, as_of_date: str | Date, lang: str = "en") -> str:
    """The advisory text for a district and crop on a date, in ``lang`` (en, hi, ta, te); at most 320 characters."""
    return compose(district_id, crop, as_of_date, lang).text
