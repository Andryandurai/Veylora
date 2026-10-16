"""Understanding short chat messages: district (fuzzy), crop, language, date and simple commands.

Messages look like ``Thanjavur rice``, ``warangal cotton hindi``, ``advisory for east godavari paddy on 2023-05-20`` or a
single word such as ``help`` or ``stop``. District names must be typed in English letters; crop and language words are
also understood in Hindi, Tamil and Telugu script. Misspellings are tolerated by fuzzy matching with rapidfuzz, and an
uncertain match is *never* used silently: the caller is expected to ask the person to confirm (see ``DistrictMatch``).
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd
from rapidfuzz import fuzz, process

from advisory.generate import load_templates
from config import ROOT, get_path
from etl.names import name_key

ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
SPLIT = re.compile(r"[\s,.;:!?()\[\]{}\"'/\\|<>+=*&^%$#@~`]+")
STOPWORDS = {
    "advisory", "advice", "weather", "for", "in", "on", "of", "the", "my", "at", "please", "pls", "plz", "me", "give",
    "send", "tell", "about", "is", "a", "an", "farm", "field", "district", "crop", "today", "now", "alert", "info",
    "lang", "language", "date", "and", "to", "get", "show",
}  # fmt: skip
CROP_ALIASES = {
    "paddy": "rice", "dhan": "rice", "chawal": "rice", "nel": "rice", "vari": "rice",
    "kapas": "cotton", "kapaas": "cotton", "paruthi": "cotton", "patti": "cotton",
    "ganna": "sugarcane", "cane": "sugarcane", "karumbu": "sugarcane", "cheruku": "sugarcane",
    "gehun": "wheat", "gehu": "wheat", "godhumai": "wheat", "godhuma": "wheat",
}  # fmt: skip
LANGUAGE_WORDS = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te",
    "हिंदी": "hi", "हिन्दी": "hi", "தமிழ்": "ta", "తెలుగు": "te", "अंग्रेजी": "en",
}  # fmt: skip
LANGUAGE_CODES = {"en", "hi", "ta", "te"}
GREETINGS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "namaste",
    "namaskar",
    "vanakkam",
    "namaskaram",
    "good",
    "morning",
    "evening",
}
INTENT_WORDS = {
    "help": {"help", "menu", "how", "commands", "?"},
    "stop": {"stop", "unsubscribe", "cancel", "quit", "end", "optout"},
    "start": {"start", "subscribe", "resume", "restart"},
    "yes": {"yes", "y", "yeah", "yep", "ok", "okay", "haan", "han", "ha", "sure", "correct", "right", "ஆம்", "అవును", "हां", "हाँ"},
    "no": {"no", "n", "nope", "nahi", "illai", "kadu", "नहीं", "இல்லை", "కాదు"},
}  # fmt: skip


@dataclass(frozen=True)
class DistrictMatch:
    """Result of matching a typed place name against the district units."""

    district_id: str
    label: str  # "Thanjavur (Tamil Nadu)"
    score: float  # 0-100
    confident: bool  # clearly the best, safe to use without asking
    alternatives: tuple[
        tuple[str, str, float], ...
    ] = ()  # (district_id, label, score) of the next best units


@dataclass(frozen=True)
class ParsedMessage:
    """What a chat message asked for (any part may be missing)."""

    intent: str  # advisory | help | greeting | stop | start | yes | no | unknown
    district_query: str = ""
    crop: str | None = None
    lang: str | None = None
    as_of: date | None = None


def _clean(token: str) -> str:
    return unicodedata.normalize("NFC", token).casefold()


@lru_cache(maxsize=1)
def crop_vocabulary() -> dict[str, str]:
    """Every accepted crop word (English, Hindi, Tamil, Telugu and common aliases) -> crop name."""
    vocab = dict(CROP_ALIASES)
    for crop, words in load_templates()["crops"].items():
        for word in words.values():
            vocab[_clean(word)] = crop
        vocab[crop] = crop
    return vocab


@lru_cache(maxsize=1)
def district_index() -> tuple[dict[str, str], dict[str, str]]:
    """(alias key -> district_id, district_id -> label) built from the processed districts and the alias file."""
    path = get_path("data_processed") / "districts.parquet"
    frame = pd.read_parquet(
        path, columns=["district_id", "district", "state", "member_districts", "icrisat_name"]
    )
    aliases: dict[str, str] = {}
    labels: dict[str, str] = {}
    member_to_unit: dict[tuple[str, str], str] = {}
    for row in frame.itertuples():
        labels[row.district_id] = f"{row.district} ({row.state})"
        names = {row.district, row.icrisat_name, *str(row.member_districts).split("|")}
        for name in names:
            key = name_key(name)
            if key:
                aliases.setdefault(key, row.district_id)
        for member in str(row.member_districts).split("|"):
            member_to_unit[(name_key(row.state), name_key(member))] = row.district_id
    alias_file = ROOT / "config" / "district_name_map.csv"
    with alias_file.open(encoding="utf-8", newline="") as handle:
        lines = (line for line in handle if not line.startswith("#"))
        for entry in csv.DictReader(lines):
            unit = member_to_unit.get((name_key(entry["state"]), name_key(entry["canonical"])))
            if unit:
                aliases.setdefault(name_key(entry["alias"]), unit)
    return aliases, labels


def _score(query: str, key: str, **_: object) -> float:
    """Similarity of a normalised query and alias (0-100); whole-word containment counts as a strong match."""
    base = max(fuzz.ratio(query, key), fuzz.token_sort_ratio(query, key))
    qt, kt = set(query.split()), set(key.split())
    shared = qt & kt
    if shared and (qt <= kt or kt <= qt) and sum(len(t) for t in shared) >= 4:
        return max(base, 95.0)
    return base


def match_district(query: str, accept: float = 92.0, confirm: float = 70.0) -> DistrictMatch | None:
    """Best district unit for a typed name, or None if nothing is plausible (score below ``confirm``).

    ``confident`` is True only when the best score reaches ``accept`` and the runner-up (a different district) is at
    least 4 points behind, so ambiguous names such as "godavari" are never accepted silently.
    """
    key = name_key(query)
    if not key:
        return None
    aliases, labels = district_index()
    best: dict[str, float] = {}
    for alias, value, _index in process.extract(key, list(aliases), scorer=_score, limit=12):
        unit = aliases[alias]
        best[unit] = max(best.get(unit, 0.0), float(value))
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < confirm:
        return None
    top_id, top_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    confident = top_score >= accept and top_score - runner >= 4
    alternatives = tuple(
        (u, labels[u], v) for u, v in ranked[1:3] if v >= confirm and top_score - v <= 15
    )
    return DistrictMatch(top_id, labels[top_id], top_score, confident, alternatives)


def match_in_text(text: str, accept: float = 92.0, confirm: float = 70.0) -> DistrictMatch | None:
    """Find a district anywhere in free text: the best-scoring word span of one to four words."""
    words = name_key(text).split()
    best: DistrictMatch | None = None
    for size in range(min(4, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            candidate = match_district(" ".join(words[start : start + size]), accept, confirm)
            if candidate is not None and (
                best is None
                or (candidate.score, candidate.confident) > (best.score, best.confident)
            ):
                best = candidate
    return best


def _fuzzy_crop(token: str) -> str | None:
    if not token.isascii() or len(token) < 4:
        return None
    vocab = {w: c for w, c in crop_vocabulary().items() if w.isascii() and len(w) >= 4}
    hit = process.extractOne(token, list(vocab), scorer=fuzz.ratio, score_cutoff=82)
    return vocab[hit[0]] if hit else None


def parse_message(text: str) -> ParsedMessage:
    """Split a chat message into intent, district words, crop, language and date."""
    raw = unicodedata.normalize("NFC", text or "").strip()
    if not raw:
        return ParsedMessage("help")
    when = None
    found = ISO_DATE.search(raw)
    if found:
        try:
            when = date(int(found[1]), int(found[2]), int(found[3]))
        except ValueError:
            when = None
        raw = ISO_DATE.sub(" ", raw)
    tokens = [t for t in SPLIT.split(raw) if t]
    cleaned = [_clean(t) for t in tokens]
    lowered = set(cleaned)
    for intent, words in INTENT_WORDS.items():
        if lowered and lowered <= words:
            return ParsedMessage(intent)
    if lowered and lowered <= GREETINGS | {"hi", "there"}:
        return ParsedMessage("greeting")
    lang = None
    crop = None
    rest: list[str] = []
    skip_next_code = False
    for i, token in enumerate(cleaned):
        if skip_next_code:
            skip_next_code = False
            continue
        if (
            token in {"lang", "language"}
            and i + 1 < len(cleaned)
            and cleaned[i + 1] in LANGUAGE_CODES
        ):
            lang = cleaned[i + 1]
            skip_next_code = True
        elif token in LANGUAGE_WORDS:
            lang = LANGUAGE_WORDS[token]
        elif token in crop_vocabulary():
            crop = crop or crop_vocabulary()[token]
        elif token in STOPWORDS:
            continue
        else:
            rest.append(tokens[i])
    query_words = []
    for token in rest:
        guess = _fuzzy_crop(_clean(token)) if crop is None else None
        if guess is not None and not _plausible_district_word(token):
            crop = guess
        else:
            query_words.append(token)
    query = " ".join(query_words)
    if not query and crop is None and lang is None and when is None:
        return ParsedMessage("unknown")
    return ParsedMessage("advisory", district_query=query, crop=crop, lang=lang, as_of=when)


def _plausible_district_word(token: str) -> bool:
    """True if the word is itself a good district-name match (so it is not mistaken for a misspelt crop)."""
    match = match_district(name_key(token), accept=92.0, confirm=85.0)
    return match is not None
