"""The WhatsApp assistant's conversation logic, independent of any web framework.

``Conversation.handle(sender, text)`` returns the reply text. It remembers each sender's last district, crop and
language for ``bot.session_ttl_minutes`` (in memory only, keyed by a hash of the number; nothing is written to disk
except the opt-out list). It never uses an uncertain district match silently: a medium-confidence match produces a
"Did you mean ...?" question and waits for YES. Replies other than the advisory itself are in English.

The advisory is built by ``advisory.generate.compose`` (fixed templates, no language model). There is no live weather
feed, so the date defaults to the last day of the observed record (``ADVISORY_AS_OF`` overrides it).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from advisory import optout, parse
from advisory.generate import Advisory, compose, settings

LOG = logging.getLogger(__name__)

HELP = (
    "ClimateResilience advisory. Send a district and a crop, e.g. 'Thanjavur rice' or 'Warangal cotton hindi'. "
    "Crops: rice, wheat, cotton, sugarcane. Languages: english, hindi, tamil, telugu. "
    "Type the district in English letters. Send STOP to stop messages."
)
CROP_QUESTION = "Which crop? Reply rice, wheat, cotton or sugarcane."
NOT_FOUND = "I could not find that district. Send the district name in English letters, e.g. 'Thanjavur rice'."
UNAVAILABLE = "Sorry, I cannot make an advisory for that right now. Please try again later."
SLOW_DOWN = "You are sending messages very quickly. Please wait a minute and try again."


@dataclass
class Pending:
    """A district the bot is waiting for the person to confirm."""

    district_id: str
    label: str


@dataclass
class Session:
    """What the bot remembers about one sender for a short time."""

    district_id: str | None = None
    crop: str | None = None
    lang: str = "en"
    pending: Pending | None = None
    updated: float = 0.0
    recent: deque[float] = field(default_factory=deque)


@dataclass(frozen=True)
class Reply:
    """What to send back: the text (empty for no reply), a short kind label and the advisory if there was one."""

    text: str
    kind: str
    advisory: Advisory | None = None


def default_as_of() -> date:
    """Date used when the message names none: ``ADVISORY_AS_OF`` or the last day of the observed weather record."""
    override = os.environ.get("ADVISORY_AS_OF", "").strip()
    if override:
        return date.fromisoformat(override)
    from analytics.predictive.extreme_weather import forecast_date_range

    return forecast_date_range()[1].date()


def default_date_range() -> tuple[date, date]:
    """First and last date an advisory can be made for (the observed record the early-warning models cover)."""
    from analytics.predictive.extreme_weather import forecast_date_range

    first, last = forecast_date_range()
    return first.date(), last.date()


class Conversation:
    """Stateful message handler; create one per process.

    Args:
        advisory_fn: ``(district_id, crop, as_of, lang) -> Advisory`` (default: :func:`advisory.generate.compose`).
        clock: seconds clock (replaced in tests).
        as_of_fn: date used when the message has none.
        date_range_fn: first and last date an advisory can be made for.
    """

    def __init__(
        self,
        advisory_fn: Callable[[str, str, date, str], Advisory] = compose,
        clock: Callable[[], float] = time.time,
        as_of_fn: Callable[[], date] = default_as_of,
        date_range_fn: Callable[[], tuple[date, date]] = default_date_range,
    ) -> None:
        cfg = settings()["bot"]
        self.advisory_fn = advisory_fn
        self.clock = clock
        self.as_of_fn = as_of_fn
        self.date_range_fn = date_range_fn
        self.ttl = float(cfg["session_ttl_minutes"]) * 60
        self.accept = float(cfg["accept_score"])
        self.confirm = float(cfg["confirm_score"])
        self.per_minute = int(cfg["messages_per_minute"])
        self.sessions: dict[str, Session] = {}

    def _session(self, sender: str) -> Session:
        key = optout.digest(sender)
        now = self.clock()
        session = self.sessions.get(key)
        if session is None or now - session.updated > self.ttl:
            session = Session()
            self.sessions[key] = session
        session.updated = now
        return session

    def _too_fast(self, session: Session) -> bool:
        now = self.clock()
        while session.recent and now - session.recent[0] >= 60:
            session.recent.popleft()
        if len(session.recent) >= self.per_minute:
            return True
        session.recent.append(now)
        return False

    def handle(self, sender: str, text: str) -> Reply:
        """Reply to one incoming message from ``sender`` (a phone number, optionally ``whatsapp:`` prefixed)."""
        parsed = parse.parse_message(text)
        if optout.is_opted_out(sender):
            if parsed.intent == "start":
                optout.opt_in(sender)
                return Reply("You will receive advisories again. " + HELP, "start")
            return Reply("", "opted_out")
        session = self._session(sender)
        if self._too_fast(session):
            return Reply(SLOW_DOWN, "rate_limited")
        if parsed.intent == "stop":
            optout.opt_out(sender)
            self.sessions.pop(optout.digest(sender), None)
            return Reply(
                "You will not get more messages. Send START to receive them again.", "stop"
            )
        if parsed.intent in {"help", "greeting", "start", "unknown"}:
            return Reply(HELP, "help")
        if parsed.intent == "no":
            session.pending = None
            return Reply("OK. Send the district name again, e.g. 'Thanjavur rice'.", "ask_district")
        if parsed.intent == "yes":
            if session.pending is None:
                return Reply(HELP, "help")
            session.district_id, session.pending = session.pending.district_id, None
            return self._finish(session, None)
        return self._advisory_request(session, parsed)

    def _advisory_request(self, session: Session, parsed: parse.ParsedMessage) -> Reply:
        if parsed.lang:
            session.lang = parsed.lang
        if parsed.crop:
            session.crop = parsed.crop
        if parsed.district_query:
            match = parse.match_in_text(parsed.district_query, self.accept, self.confirm)
            if match is None:
                return Reply(NOT_FOUND, "district_not_found")
            if not match.confident:
                session.pending = Pending(match.district_id, match.label)
                options = [match.label] + [alt[1] for alt in match.alternatives]
                if len(options) == 1:
                    question = (
                        f"Did you mean {match.label}? Reply YES, or send the district name again."
                    )
                else:
                    question = f"Which one: {' or '.join(options)}? Send the full name, or reply YES for {match.label}."
                return Reply(question, "confirm_district")
            session.district_id, session.pending = match.district_id, None
        return self._finish(session, parsed.as_of)

    def _finish(self, session: Session, as_of: date | None) -> Reply:
        if session.district_id is None:
            return Reply(
                "Which district? Send the district name in English letters.", "ask_district"
            )
        if session.crop is None:
            return Reply(CROP_QUESTION, "ask_crop")
        when = as_of or self.as_of_fn()
        first, last = self.date_range_fn()
        if not first <= when <= last:
            return Reply(
                f"I can only advise for dates from {first} to {last}.", "date_out_of_range"
            )
        try:
            advisory = self.advisory_fn(session.district_id, session.crop, when, session.lang)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            LOG.warning(
                "no advisory for %s/%s on %s: %s", session.district_id, session.crop, when, exc
            )
            return Reply(UNAVAILABLE, "unavailable")
        except Exception:
            LOG.exception(
                "advisory failed for %s/%s on %s", session.district_id, session.crop, when
            )
            return Reply(UNAVAILABLE, "error")
        return Reply(advisory.text, "advisory", advisory)
