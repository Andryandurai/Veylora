"""Outbound SMS and WhatsApp delivery through Twilio, safe by default.

* **Dry run is the default.** Nothing is sent unless ``DRY_RUN`` is explicitly set to ``false`` *and* the Twilio
  credentials are present; in every other case the message is only logged (``SendResult.status == "dry_run"``).
* Recipients must be E.164 numbers (``+919876543210``); a malformed number is rejected before anything else happens.
* Two rate limits protect farmers and the account: ``delivery.max_per_minute`` for all recipients together and
  ``delivery.per_recipient_per_day`` for each phone (``settings.yaml``).
* Logs and the audit file ``advisory_audit.jsonl`` (state folder, ``data/interim`` by default) hold masked numbers and lengths, never the message
  body, the auth token or a full phone number.
* Recipients who replied STOP (``advisory/optout.py``) are skipped.

CLI for the Twilio sandbox test: ``python -m advisory.twilio_client --to +91XXXXXXXXXX --body "test" [--channel whatsapp]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from advisory import optout
from config import load_env, load_settings

LOG = logging.getLogger(__name__)

E164 = re.compile(r"^\+[1-9]\d{7,14}$")
TWILIO_MAX_CHARS = 1600
FALSE_WORDS = {"false", "0", "no", "off"}


class AdvisoryError(Exception):
    """Base class of delivery errors."""


class ConfigError(AdvisoryError):
    """Real sending was requested but the Twilio configuration is incomplete."""


class InvalidRecipient(AdvisoryError, ValueError):
    """The destination is not an E.164 phone number."""


class RateLimitExceeded(AdvisoryError):
    """A delivery rate limit was hit; nothing was sent."""


@dataclass(frozen=True)
class TwilioSettings:
    """Delivery configuration read from the environment (``.env``) and ``settings.yaml``."""

    dry_run: bool = True
    account_sid: str = ""
    auth_token: str = field(default="", repr=False)
    sms_from: str = ""
    whatsapp_from: str = ""
    max_per_minute: int = 30
    per_recipient_per_day: int = 3

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TwilioSettings:
        """Read the configuration; only an explicit ``DRY_RUN=false`` (or 0, no, off) leaves dry-run mode."""
        if env is None:
            load_env()
            env = os.environ
        limits = load_settings()["advisory"]["delivery"]
        return cls(
            dry_run=env.get("DRY_RUN", "true").strip().lower() not in FALSE_WORDS,
            account_sid=env.get("TWILIO_ACCOUNT_SID", "").strip(),
            auth_token=env.get("TWILIO_AUTH_TOKEN", "").strip(),
            sms_from=env.get("TWILIO_SMS_FROM", "").strip(),
            whatsapp_from=env.get("TWILIO_WHATSAPP_FROM", "").strip(),
            max_per_minute=int(limits["max_per_minute"]),
            per_recipient_per_day=int(limits["per_recipient_per_day"]),
        )

    def missing_for(self, channel: str) -> list[str]:
        """Names of the settings that real sending on ``channel`` still needs."""
        needed = {"TWILIO_ACCOUNT_SID": self.account_sid, "TWILIO_AUTH_TOKEN": self.auth_token}
        needed["TWILIO_WHATSAPP_FROM" if channel == "whatsapp" else "TWILIO_SMS_FROM"] = (
            self.whatsapp_from if channel == "whatsapp" else self.sms_from
        )
        return [name for name, value in needed.items() if not value]


@dataclass(frozen=True)
class SendResult:
    """Outcome of one send attempt (never contains the body or the full number)."""

    channel: str
    to_masked: str
    status: str  # dry_run | queued | sent | skipped_opt_out (Twilio statuses are passed through)
    chars: int
    dry_run: bool
    sid: str | None = None


def mask_number(number: str) -> str:
    """``+919876543210`` -> ``+91******3210`` (country code and last four digits only)."""
    digits = number.removeprefix("whatsapp:")
    if len(digits) <= 7:
        return "***"
    return digits[:3] + "*" * (len(digits) - 7) + digits[-4:]


def normalise_number(number: str) -> str:
    """Strip spaces, dashes and a ``whatsapp:`` prefix and check E.164; raises :class:`InvalidRecipient`."""
    cleaned = re.sub(r"[\s\-()]", "", number.strip()).removeprefix("whatsapp:")
    if not E164.match(cleaned):
        raise InvalidRecipient("recipient must be an E.164 number such as +919876543210")
    return cleaned


class RateLimiter:
    """Sliding-window limits: ``max_per_minute`` overall and ``per_recipient_per_day`` per phone."""

    def __init__(
        self,
        max_per_minute: int,
        per_recipient_per_day: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_per_minute = max_per_minute
        self.per_recipient_per_day = per_recipient_per_day
        self.clock = clock
        self._all: deque[float] = deque()
        self._by_recipient: dict[str, deque[float]] = defaultdict(deque)

    def check(self, recipient: str) -> None:
        """Record one send for ``recipient`` or raise :class:`RateLimitExceeded` (nothing is recorded then)."""
        now = self.clock()
        while self._all and now - self._all[0] >= 60:
            self._all.popleft()
        own = self._by_recipient[recipient]
        while own and now - own[0] >= 86400:
            own.popleft()
        if len(self._all) >= self.max_per_minute:
            raise RateLimitExceeded(f"more than {self.max_per_minute} messages in a minute")
        if len(own) >= self.per_recipient_per_day:
            raise RateLimitExceeded(
                f"{mask_number(recipient)} already received {self.per_recipient_per_day} messages in 24 hours"
            )
        self._all.append(now)
        own.append(now)


_DEFAULT_LIMITER: RateLimiter | None = None


def default_limiter(settings: TwilioSettings) -> RateLimiter:
    """Process-wide limiter (limits only mean something if they are shared by every call)."""
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        _DEFAULT_LIMITER = RateLimiter(settings.max_per_minute, settings.per_recipient_per_day)
    return _DEFAULT_LIMITER


def reset_default_limiter() -> None:
    """Forget the shared limiter (tests)."""
    global _DEFAULT_LIMITER
    _DEFAULT_LIMITER = None


def _audit(result: SendResult) -> None:
    """Append one JSON line (masked number, length, status) to the audit file."""
    record = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **result.__dict__}
    path = optout.state_dir() / "advisory_audit.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _real_client(settings: TwilioSettings) -> Any:
    """The Twilio REST client (imported lazily so that dry runs never touch the library)."""
    from twilio.rest import Client

    return Client(settings.account_sid, settings.auth_token)


def send(
    channel: str,
    to: str,
    body: str,
    *,
    settings: TwilioSettings | None = None,
    limiter: RateLimiter | None = None,
    client_factory: Callable[[TwilioSettings], Any] = _real_client,
) -> SendResult:
    """Send (or, in dry-run mode, only log) one message.

    Args:
        channel: ``"sms"`` or ``"whatsapp"``.
        to: E.164 phone number; a ``whatsapp:`` prefix is accepted.
        body: message text (at most 1600 characters, Twilio's limit).
        settings: configuration (default: environment).
        limiter: rate limiter (default: the process-wide one).
        client_factory: builds the Twilio client; replaced in tests.

    Raises:
        InvalidRecipient: malformed number.
        ValueError: unknown channel, empty or oversized body.
        ConfigError: real sending requested without complete credentials.
        RateLimitExceeded: a delivery limit was reached.
    """
    if channel not in {"sms", "whatsapp"}:
        raise ValueError("channel must be 'sms' or 'whatsapp'")
    if not body.strip():
        raise ValueError("message body is empty")
    if len(body) > TWILIO_MAX_CHARS:
        raise ValueError(
            f"message of {len(body)} characters exceeds Twilio's {TWILIO_MAX_CHARS}-character limit"
        )
    settings = settings or TwilioSettings.from_env()
    number = normalise_number(to)
    masked = mask_number(number)
    if optout.is_opted_out(number):
        result = SendResult(channel, masked, "skipped_opt_out", len(body), settings.dry_run)
        LOG.info("skipped %s to %s: recipient opted out", channel, masked)
        _audit(result)
        return result
    (limiter or default_limiter(settings)).check(number)
    if settings.dry_run:
        result = SendResult(channel, masked, "dry_run", len(body), True)
        LOG.info("DRY RUN: would send %s to %s (%d characters)", channel, masked, len(body))
        _audit(result)
        return result
    missing = settings.missing_for(channel)
    if missing:
        raise ConfigError(f"DRY_RUN is false but {', '.join(missing)} is not set")
    sender = settings.whatsapp_from if channel == "whatsapp" else settings.sms_from
    destination = f"whatsapp:{number}" if channel == "whatsapp" else number
    if channel == "whatsapp" and not sender.startswith("whatsapp:"):
        sender = f"whatsapp:{sender}"
    message = client_factory(settings).messages.create(to=destination, from_=sender, body=body)
    result = SendResult(channel, masked, str(message.status), len(body), False, str(message.sid))
    LOG.info(
        "sent %s to %s (%d characters), sid %s, status %s",
        channel,
        masked,
        len(body),
        result.sid,
        result.status,
    )
    _audit(result)
    return result


def send_sms(to: str, body: str, **kwargs: Any) -> SendResult:
    """Send an SMS (dry run unless ``DRY_RUN=false``); see :func:`send`."""
    return send("sms", to, body, **kwargs)


def send_whatsapp(to: str, body: str, **kwargs: Any) -> SendResult:
    """Send a WhatsApp message (dry run unless ``DRY_RUN=false``); see :func:`send`."""
    return send("whatsapp", to, body, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Command-line sender for the Twilio sandbox test."""
    parser = argparse.ArgumentParser(
        description="Send one message through Twilio (dry run unless DRY_RUN=false)."
    )
    parser.add_argument("--to", required=True, help="E.164 number, e.g. +919876543210")
    parser.add_argument("--body", required=True)
    parser.add_argument("--channel", choices=["sms", "whatsapp"], default="whatsapp")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = send(args.channel, args.to, args.body)
    except AdvisoryError as exc:
        LOG.error("not sent: %s", exc)
        return 1
    LOG.info("result: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
