"""Phase 9: advisory templates and rules, message parsing, the conversation, the Twilio wrapper, the webhook and the bulk tool.

Nothing here sends a message or needs credentials. Tests that need the trained models or processed data skip themselves.
"""

from __future__ import annotations

import csv
import json
import logging
import string
from datetime import date
from types import SimpleNamespace

import pytest

from advisory import bot, optout, parse, twilio_client
from advisory import conversation as conv
from advisory import generate as G
from advisory.generate import Advisory
from analytics.predictive import extreme_weather as X
from config import get_path, load_settings

LANGS = ("en", "hi", "ta", "te")
LEVELS = ("low", "moderate", "high", "severe")
HEAT_P = {
    "low": 0.01,
    "moderate": 0.10,
    "high": 0.20,
    "severe": 0.50,
}  # settings.yaml risk_levels.heatwave
DROUGHT_P = {
    "low": 0.10,
    "moderate": 0.40,
    "high": 0.60,
    "severe": 0.80,
}  # settings.yaml risk_levels.drought
PHONE = "+919876543210"
LIMIT = int(load_settings()["advisory"]["sms_max_chars"])

processed = get_path("data_processed")
needs_districts = pytest.mark.skipif(
    not (processed / "districts.parquet").exists(), reason="run `make etl` first"
)
needs_models = pytest.mark.skipif(
    not (
        (processed / "weather_daily.parquet").exists()
        and (get_path("models") / "drought_xgb.joblib").exists()
        and (get_path("models") / "yield_xgb.joblib").exists()
    ),
    reason="run `make etl train-yield train-extreme` first",
)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Opt-outs, audit log and limiter never touch the real data folder or the environment's DRY_RUN."""
    monkeypatch.setenv("ADVISORY_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.delenv("ALLOW_UNSIGNED_WEBHOOK", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    twilio_client.reset_default_limiter()
    yield
    twilio_client.reset_default_limiter()


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the two model calls behind ``compose`` so that every risk level can be produced on demand."""
    state = {"heatwave_p": 0.0, "drought_p": 0.0, "drought_relevant": True, "irrigation": 0.0}

    def risk_forecast(district_id, as_of):
        return {
            "heatwave_p": state["heatwave_p"],
            "drought_p": state["drought_p"],
            "drought_relevant": state["drought_relevant"],
            "drought_normal_month_rain_mm": 80.0,
            "level": "low",
            "lead_days": 14,
        }

    monkeypatch.setattr(X, "risk_forecast", risk_forecast)
    monkeypatch.setattr(
        G, "planned_irrigation", lambda district_id, crop: tuple([state["irrigation"]] * 16)
    )
    return state


# ---------------------------------------------------------------------------------------------------------------
# templates and rules
# ---------------------------------------------------------------------------------------------------------------


def test_every_language_has_every_message_crop_and_risk_word() -> None:
    templates = G.load_templates()
    assert tuple(templates["languages"]) == LANGS
    for kind, texts in templates["messages"].items():
        assert set(texts) == set(LANGS), kind
    for crop, words in templates["crops"].items():
        assert set(words) == set(LANGS), crop
    for lang in LANGS:
        assert set(templates["risk"][lang]) == set(LEVELS), lang


def test_templates_use_only_the_documented_placeholders() -> None:
    allowed = {"district", "crop", "date", "mm", "risk"}
    for kind, texts in G.load_templates()["messages"].items():
        for lang, text in texts.items():
            names = {f for _, f, _, _ in string.Formatter().parse(text) if f}
            assert names <= allowed, (kind, lang, names - allowed)


@needs_districts
def test_every_message_fits_the_sms_limit_for_every_district_crop_language_and_level() -> None:
    ids = sorted(parse.district_index()[1])
    assert len(ids) >= 40
    templates = G.load_templates()
    worst = 0
    for district_id in ids:
        name = G.pretty_district(district_id)
        for crop in templates["crops"]:
            for lang in LANGS:
                for kind in templates["messages"]:
                    for level in LEVELS:
                        text = G.render(kind, lang, name, crop, "2024-12-31", level, 100.0)
                        assert "{" not in text and "}" not in text, (kind, lang)
                        worst = max(worst, len(text))
                        assert len(text) <= LIMIT, (district_id, crop, lang, kind, len(text))
    assert worst > 0


def test_render_rejects_unknown_languages_and_rounds_irrigation_to_five_millimetres() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        G.render("all_clear", "fr", "Guntur", "rice", "2024-01-01")
    assert "about 25 mm" in G.render(
        "irrigation_plan", "en", "Guntur", "rice", "2024-01-01", mm=23.4
    )


def test_sms_segments_follow_gsm7_and_unicode_limits() -> None:
    assert G.sms_segments("") == 0
    assert G.sms_segments("a" * 160) == 1 and G.sms_segments("a" * 161) == 2
    assert G.sms_segments("a" * 306) == 2 and G.sms_segments("a" * 307) == 3
    assert (
        G.sms_segments("€" * 80) == 1 and G.sms_segments("€" * 81) == 2
    )  # extension characters take two units
    assert G.sms_segments("क" * 70) == 1 and G.sms_segments("क" * 71) == 2
    assert G.is_gsm7("Heat alert: water early.") and not G.is_gsm7("गर्मी")


def test_choose_messages_rules() -> None:
    quiet = {"heatwave": "low", "drought": "low"}
    assert G.choose_messages(quiet, None, None, None) == ["all_clear"]
    assert G.choose_messages({"heatwave": "moderate", "drought": "low"}, None, None, None) == [
        "heat_alert"
    ]
    assert G.choose_messages({"heatwave": "low", "drought": "high"}, None, None, None) == [
        "drought_watch"
    ]
    # the more severe hazard wins; a tie goes to the heatwave; only one alert is ever sent
    assert G.choose_messages({"heatwave": "moderate", "drought": "severe"}, None, None, None) == [
        "drought_watch"
    ]
    assert G.choose_messages({"heatwave": "high", "drought": "high"}, None, None, None) == [
        "heat_alert"
    ]
    # irrigation follows an alert; it needs the crop to be in season and at least 5 mm planned
    assert G.choose_messages({"heatwave": "high", "drought": "low"}, 4, 30.0, None) == [
        "heat_alert",
        "irrigation_plan",
    ]
    assert G.choose_messages(quiet, 4, 4.9, None) == ["all_clear"]
    assert G.choose_messages(quiet, 4, 5.0, None) == ["irrigation_plan"]
    # sowing window only before the season, and only within 21 days
    assert G.choose_messages(quiet, None, None, 10) == ["sowing_window"]
    assert G.choose_messages(quiet, None, None, 22) == ["all_clear"]
    assert G.choose_messages(quiet, None, None, 0) == ["all_clear"]
    assert len(G.choose_messages({"heatwave": "severe", "drought": "severe"}, 3, 40.0, 5)) <= 2


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("level", LEVELS)
def test_heat_advisory_for_each_risk_level_and_language(fake_models, lang, level) -> None:
    fake_models["heatwave_p"] = HEAT_P[level]
    result = G.compose("tamil_nadu__thanjavur", "rice", "2024-04-20", lang)
    assert result.risk["levels"]["heatwave"] == level
    assert 0 < len(result.text) <= LIMIT
    if level == "low":
        assert "heat_alert" not in result.messages
    else:
        assert result.messages[0] == "heat_alert"
        assert G.load_templates()["risk"][lang][level] in result.text


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("level", LEVELS)
def test_drought_advisory_for_each_risk_level_and_language(fake_models, lang, level) -> None:
    fake_models["drought_p"] = DROUGHT_P[level]
    result = G.compose("telangana__warangal", "cotton", "2024-06-15", lang)
    assert result.risk["levels"]["drought"] == level
    assert 0 < len(result.text) <= LIMIT
    if level == "low":
        assert "drought_watch" not in result.messages
    else:
        assert result.messages[0] == "drought_watch"
        assert G.load_templates()["risk"][lang][level] in result.text


@pytest.mark.parametrize("lang", LANGS)
def test_two_message_advisories_never_exceed_the_limit(fake_models, lang) -> None:
    fake_models.update(heatwave_p=0.5, irrigation=40.0)
    result = G.compose(
        "telangana__warangal", "rice", "2024-07-10", lang
    )  # kharif week 2, severe heat, 40 mm planned
    assert len(result.text) <= LIMIT
    assert result.messages[0] == "heat_alert"
    if "irrigation_plan" not in result.messages:  # long scripts drop the second message and say so
        assert any("second message dropped" in note for note in result.notes)


def test_a_dry_month_drought_signal_is_suppressed_with_a_note(fake_models) -> None:
    fake_models.update(drought_p=0.9, drought_relevant=False)
    result = G.compose("andhra_pradesh__krishna", "rice", "2005-01-10", "en")
    assert result.messages == ["all_clear"]
    assert result.risk["levels"]["drought"] == "low"
    assert any("suppressed" in note for note in result.notes)


def test_irrigation_message_carries_the_planned_amount_in_season(fake_models) -> None:
    fake_models["irrigation"] = 22.0
    result = G.compose("tamil_nadu__thanjavur", "rice", "2024-07-10", "en")
    assert result.messages == ["irrigation_plan"] and "about 20 mm" in result.text
    assert result.weeks_into_season is not None and 0 <= result.weeks_into_season <= 15


def test_sowing_window_appears_shortly_before_the_season(fake_models) -> None:
    start = G.season_position("rice", date(2024, 5, 20))[2]
    assert start.month == 6 and start.year == 2024  # kharif starts in June
    result = G.compose("tamil_nadu__thanjavur", "rice", "2024-05-20", "en")
    assert result.messages == ["sowing_window"] and "2024-06-01" in result.text


@needs_models
def test_real_models_give_valid_advisories_in_all_languages() -> None:
    for lang in LANGS:
        for district, crop, day in (
            ("tamil_nadu__thanjavur", "rice", "2024-06-15"),
            ("telangana__warangal", "cotton", "2003-06-15"),
            ("andhra_pradesh__krishna", "rice", "2005-01-10"),
        ):
            result = G.compose(district, crop, day, lang)
            assert 0 < len(result.text) <= LIMIT, (district, lang)
            assert result.messages
    assert G.is_gsm7(G.generate("tamil_nadu__thanjavur", "rice", "2024-06-15", "en"))
    krishna = G.compose("andhra_pradesh__krishna", "rice", "2005-01-10", "en")
    assert krishna.messages == ["all_clear"] and krishna.notes


# ---------------------------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------------------------


def test_parse_crops_languages_dates_and_commands() -> None:
    p = parse.parse_message("advisory for east godavari paddy on 2023-05-20")
    assert (p.intent, p.crop, p.as_of, p.district_query) == (
        "advisory",
        "rice",
        date(2023, 5, 20),
        "east godavari",
    )
    assert parse.parse_message("Warangal cotton hindi").lang == "hi"
    assert parse.parse_message("cuddapah rice in telugu").lang == "te"
    assert parse.parse_message("dhan karimnagar lang ta") == parse.ParsedMessage(
        "advisory", "karimnagar", "rice", "ta", None
    )
    assert (
        parse.parse_message("कपास वारंगल").crop == "cotton"
    )  # crop words work in Indic scripts too
    assert parse.parse_message("हिंदी").lang == "hi"
    assert parse.parse_message("riice nizamabad").crop == "rice"  # a misspelt crop word
    for text, intent in (
        ("STOP", "stop"),
        ("help", "help"),
        ("Hi", "greeting"),
        ("yes", "yes"),
        ("No", "no"),
        ("start", "start"),
        ("", "help"),
    ):
        assert parse.parse_message(text).intent == intent, text
    assert parse.parse_message("2023-02-30 rice").as_of is None  # an impossible date is ignored


def test_a_district_name_is_not_mistaken_for_a_crop_or_command() -> None:
    for name in ("Guntur", "Medak", "Krishna", "Bidar", "Kolar"):
        p = parse.parse_message(name)
        assert p.intent == "advisory" and p.crop is None and p.district_query == name


@needs_districts
def test_fuzzy_district_matching_accepts_confident_matches_and_asks_about_the_rest() -> None:
    exact = parse.match_in_text("Thanjavur")
    assert (
        exact
        and exact.district_id == "tamil_nadu__thanjavur"
        and exact.confident
        and exact.score == 100
    )
    typo = parse.match_in_text("thanjavr")
    assert typo and typo.district_id == "tamil_nadu__thanjavur" and typo.confident
    for alias, expected in (
        ("Tanjore", "tamil_nadu__thanjavur"),
        ("Cuddapah", "andhra_pradesh__kadapa_ysr"),
        ("mysuru", "karnataka__mysore"),
    ):
        got = parse.match_in_text(alias)
        assert got and got.district_id == expected and got.confident, alias
    ambiguous = parse.match_in_text("godavari")
    assert ambiguous and not ambiguous.confident
    assert {a[0] for a in ambiguous.alternatives} | {ambiguous.district_id} >= {
        "andhra_pradesh__east_godavari",
        "andhra_pradesh__west_godavari",
    }
    assert parse.match_in_text("xyzzy") is None
    assert parse.match_in_text("தஞ்சாவூர்") is None  # district names must be typed in English letters
    inside = parse.match_in_text("my farm near thanjavur")
    assert inside and inside.district_id == "tamil_nadu__thanjavur"


# ---------------------------------------------------------------------------------------------------------------
# conversation
# ---------------------------------------------------------------------------------------------------------------


class Recorder:
    """Stands in for ``compose`` and remembers every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail: Exception | None = None

    def __call__(self, district_id, crop, as_of, lang):
        self.calls.append((district_id, crop, as_of, lang))
        if self.fail is not None:
            raise self.fail
        return Advisory(
            text=f"ADVICE {district_id} {crop} {as_of} {lang}", lang=lang, messages=["all_clear"]
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def talk():
    recorder, clock = Recorder(), Clock()
    conversation = conv.Conversation(
        advisory_fn=recorder,
        clock=clock,
        as_of_fn=lambda: date(2024, 12, 31),
        date_range_fn=lambda: (date(1991, 5, 31), date(2024, 12, 31)),
    )
    return SimpleNamespace(c=conversation, rec=recorder, clock=clock)


@needs_districts
def test_conversation_remembers_district_crop_and_language(talk) -> None:
    first = talk.c.handle(PHONE, "Thanjavur rice")
    assert first.kind == "advisory" and talk.rec.calls[-1] == (
        "tamil_nadu__thanjavur",
        "rice",
        date(2024, 12, 31),
        "en",
    )
    talk.c.handle(PHONE, "wheat")
    assert talk.rec.calls[-1][:2] == ("tamil_nadu__thanjavur", "wheat")
    talk.c.handle(PHONE, "hindi")
    assert talk.rec.calls[-1][3] == "hi"
    talk.c.handle("+911111111111", "bidar cotton")  # another sender has their own memory
    assert (
        talk.rec.calls[-1][:2] == ("karnataka__bidar", "cotton") and talk.rec.calls[-1][3] == "en"
    )


@needs_districts
def test_uncertain_district_is_confirmed_before_use(talk) -> None:
    ask = talk.c.handle(PHONE, "godavari rice")
    assert ask.kind == "confirm_district" and "East Godavari" in ask.text and talk.rec.calls == []
    assert talk.c.handle(PHONE, "yes").kind == "advisory"
    assert talk.rec.calls[-1][:2] == ("andhra_pradesh__east_godavari", "rice")
    talk.c.handle(PHONE, "godavari")
    assert talk.c.handle(PHONE, "no").kind == "ask_district"
    assert talk.c.handle(PHONE, "yes").kind == "help"  # nothing is pending any more


@needs_districts
def test_missing_crop_unknown_district_and_dates_are_handled_politely(talk) -> None:
    assert talk.c.handle(PHONE, "Warangal").kind == "ask_crop"
    assert talk.c.handle(PHONE, "cotton").kind == "advisory"
    assert talk.c.handle(PHONE, "xyzzy rice").kind == "district_not_found"
    out = talk.c.handle(PHONE, "warangal cotton on 2031-01-01")
    assert out.kind == "date_out_of_range" and "1991-05-31" in out.text
    assert talk.c.handle(PHONE, "warangal cotton on 2003-06-15").kind == "advisory"
    assert talk.rec.calls[-1][2] == date(2003, 6, 15)


@needs_districts
def test_failures_become_apologies_not_stack_traces(talk, caplog) -> None:
    talk.rec.fail = ValueError("not enough history")
    assert talk.c.handle(PHONE, "Guntur rice").text == conv.UNAVAILABLE
    talk.rec.fail = RuntimeError("boom")
    with caplog.at_level(logging.ERROR):
        assert talk.c.handle(PHONE, "Guntur rice").text == conv.UNAVAILABLE
    assert any("advisory failed" in r.message for r in caplog.records)


@needs_districts
def test_sessions_expire_and_senders_are_rate_limited(talk) -> None:
    talk.c.handle(PHONE, "Guntur rice")
    talk.clock.now += 31 * 60
    assert talk.c.handle(PHONE, "Medak").kind == "ask_crop"  # the remembered crop is gone
    kinds = [talk.c.handle("+912222222222", "help").kind for _ in range(11)]
    assert kinds[:10] == ["help"] * 10 and kinds[10] == "rate_limited"
    talk.clock.now += 61
    assert talk.c.handle("+912222222222", "help").kind == "help"


@needs_districts
def test_stop_silences_the_sender_until_start(talk) -> None:
    assert talk.c.handle(PHONE, "stop").kind == "stop" and optout.is_opted_out(PHONE)
    assert (
        talk.c.handle(PHONE, "Guntur rice") == conv.Reply("", "opted_out") and talk.rec.calls == []
    )
    assert talk.c.handle("whatsapp:" + PHONE, "START").kind == "start"
    assert not optout.is_opted_out(PHONE)
    assert talk.c.handle(PHONE, "Guntur rice").kind == "advisory"


def test_opt_out_list_stores_only_hashes() -> None:
    optout.opt_out("whatsapp:+919876543210")
    stored = (optout.state_dir() / "advisory_optout.json").read_text(encoding="utf-8")
    assert "9876543210" not in stored and len(json.loads(stored)) == 1
    assert optout.is_opted_out("+919876543210") and not optout.is_opted_out("+919876543211")


# ---------------------------------------------------------------------------------------------------------------
# Twilio wrapper
# ---------------------------------------------------------------------------------------------------------------


class FakeTwilio:
    """Records ``messages.create`` calls instead of contacting Twilio."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(sid="SM123", status="queued")


def live(**overrides) -> twilio_client.TwilioSettings:
    base = {
        "dry_run": False,
        "account_sid": "AC1",
        "auth_token": "secret-token",
        "sms_from": "+15550001111",
        "whatsapp_from": "whatsapp:+14155238886",
    }
    return twilio_client.TwilioSettings(**{**base, **overrides})


def test_dry_run_is_the_default_and_only_an_explicit_false_leaves_it() -> None:
    assert twilio_client.TwilioSettings.from_env({}).dry_run is True
    for value in ("true", "TRUE", "1", "yes", "maybe", "", "dry"):
        assert twilio_client.TwilioSettings.from_env({"DRY_RUN": value}).dry_run is True, value
    for value in ("false", "False", "0", "no", "off"):
        assert twilio_client.TwilioSettings.from_env({"DRY_RUN": value}).dry_run is False, value


def test_dry_run_never_builds_a_twilio_client_and_logs_only_masked_data(caplog) -> None:
    def forbidden(_settings):
        raise AssertionError("a Twilio client must not be created in dry-run mode")

    settings = twilio_client.TwilioSettings(dry_run=True, auth_token="secret-token")
    with caplog.at_level(logging.INFO):
        result = twilio_client.send_sms(
            PHONE, "Heat alert for Guntur", settings=settings, client_factory=forbidden
        )
    assert result.status == "dry_run" and result.dry_run and result.sid is None
    assert result.to_masked == "+91******3210" and result.chars == len("Heat alert for Guntur")
    log = caplog.text + (optout.state_dir() / "advisory_audit.jsonl").read_text(encoding="utf-8")
    assert "9876543210" not in log and "secret-token" not in log and "Heat alert" not in log
    assert "+91******3210" in log and "DRY RUN" in caplog.text


def test_real_sending_uses_the_right_numbers_and_returns_the_sid() -> None:
    fake = FakeTwilio()
    sms = twilio_client.send_sms(PHONE, "hello", settings=live(), client_factory=lambda _s: fake)
    wa = twilio_client.send_whatsapp(
        "whatsapp:+91 98765-43211",
        "hello",
        settings=live(sms_from="", whatsapp_from="+14155238886"),
        client_factory=lambda _s: fake,
    )
    assert (sms.status, sms.sid, sms.dry_run) == ("queued", "SM123", False) and wa.sid == "SM123"
    assert fake.created[0] == {"to": PHONE, "from_": "+15550001111", "body": "hello"}
    assert fake.created[1] == {
        "to": "whatsapp:+919876543211",
        "from_": "whatsapp:+14155238886",
        "body": "hello",
    }


def test_real_sending_without_credentials_fails_before_any_client_is_built() -> None:
    def forbidden(_settings):
        raise AssertionError("no client without credentials")

    with pytest.raises(twilio_client.ConfigError, match="TWILIO_AUTH_TOKEN"):
        twilio_client.send_sms(
            PHONE, "hello", settings=live(auth_token=""), client_factory=forbidden
        )
    with pytest.raises(twilio_client.ConfigError, match="TWILIO_WHATSAPP_FROM"):
        twilio_client.send_whatsapp(
            PHONE, "hello", settings=live(whatsapp_from=""), client_factory=forbidden
        )


def test_recipients_and_bodies_are_validated() -> None:
    settings = twilio_client.TwilioSettings(dry_run=True)
    for bad in ("9876543210", "+0123456789", "+91 abc", "", "+12345"):
        with pytest.raises(twilio_client.InvalidRecipient):
            twilio_client.send_sms(bad, "hello", settings=settings)
    for body in ("", "   ", "x" * 1601):
        with pytest.raises(ValueError):
            twilio_client.send_sms(PHONE, body, settings=settings)
    with pytest.raises(ValueError, match="channel"):
        twilio_client.send("fax", PHONE, "hello", settings=settings)
    assert twilio_client.send_sms(PHONE, "x" * 1600, settings=settings).chars == 1600


def test_rate_limits_apply_per_minute_and_per_recipient_per_day() -> None:
    now = [0.0]
    limiter = twilio_client.RateLimiter(3, 2, clock=lambda: now[0])
    settings = twilio_client.TwilioSettings(dry_run=True)

    def send(number: str) -> None:
        twilio_client.send_sms(number, "hi", settings=settings, limiter=limiter)

    send("+919876543210")
    send("+919876543210")
    with pytest.raises(twilio_client.RateLimitExceeded, match="24 hours"):
        send("+919876543210")  # the same phone a third time in a day
    send("+919876543211")
    with pytest.raises(twilio_client.RateLimitExceeded, match="in a minute"):
        send("+919876543212")  # 4th message within a minute
    now[0] += 61
    send("+919876543212")  # the minute window has moved on
    now[0] += 86400
    send("+919876543210")  # and so has the day window


def test_a_refused_send_is_not_counted_against_the_limit() -> None:
    limiter = twilio_client.RateLimiter(1, 1, clock=lambda: 0.0)
    limiter.check("+919876543210")
    with pytest.raises(twilio_client.RateLimitExceeded):
        limiter.check("+919876543211")
    assert len(limiter._all) == 1  # the refused call left no trace


def test_opted_out_recipients_are_skipped_without_using_the_limiter() -> None:
    optout.opt_out(PHONE)
    fake = FakeTwilio()
    limiter = twilio_client.RateLimiter(1, 1)
    result = twilio_client.send_whatsapp(
        PHONE, "hello", settings=live(), limiter=limiter, client_factory=lambda _s: fake
    )
    assert result.status == "skipped_opt_out" and fake.created == [] and len(limiter._all) == 0


def test_cli_dry_run_exit_codes(capsys) -> None:
    assert twilio_client.main(["--to", PHONE, "--body", "hello", "--channel", "sms"]) == 0
    assert twilio_client.main(["--to", "12345", "--body", "hello"]) == 1


# ---------------------------------------------------------------------------------------------------------------
# webhook
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def client_factory():
    from fastapi.testclient import TestClient
    from twilio.request_validator import RequestValidator

    def make(token: str = "tok-123", **kwargs):
        recorder = Recorder()
        conversation = conv.Conversation(
            advisory_fn=recorder,
            as_of_fn=lambda: date(2024, 12, 31),
            date_range_fn=lambda: (date(1991, 5, 31), date(2024, 12, 31)),
        )
        app = bot.create_app(
            conversation, twilio_client.TwilioSettings(dry_run=True, auth_token=token)
        )
        client = TestClient(app)

        def post(
            form: dict[str, str], url: str = "http://testserver/whatsapp", signed: bool = True
        ):
            headers = (
                {"X-Twilio-Signature": RequestValidator(token or "x").compute_signature(url, form)}
                if signed
                else {}
            )
            return client.post("/whatsapp", data=form, headers=headers)

        return SimpleNamespace(client=client, post=post, recorder=recorder)

    return make


def test_health_reports_status_without_loading_models(client_factory) -> None:
    body = client_factory().client.get("/health").json()
    assert (
        body["status"] == "ok" and body["dry_run"] is True and body["signature_validation"] is True
    )


@needs_districts
def test_webhook_answers_with_twiml_when_the_signature_is_valid(client_factory) -> None:
    app = client_factory()
    response = app.post({"From": "whatsapp:" + PHONE, "Body": "Thanjavur rice"})
    assert response.status_code == 200 and response.headers["content-type"].startswith(
        "application/xml"
    )
    assert "<Message>ADVICE tamil_nadu__thanjavur rice 2024-12-31 en</Message>" in response.text


def test_webhook_rejects_missing_or_wrong_signatures_and_never_runs_the_conversation(
    client_factory,
) -> None:
    app = client_factory()
    assert app.post({"From": PHONE, "Body": "Thanjavur rice"}, signed=False).status_code == 403
    forged = app.post({"From": PHONE, "Body": "Thanjavur rice"}, url="http://evil.example/whatsapp")
    assert forged.status_code == 403
    tampered = app.client.post(
        "/whatsapp", data={"From": PHONE, "Body": "other"}, headers={"X-Twilio-Signature": "AAAA"}
    )
    assert tampered.status_code == 403
    assert app.recorder.calls == []


def test_webhook_without_an_auth_token_is_refused_unless_explicitly_allowed(
    client_factory, monkeypatch
) -> None:
    app = client_factory(token="")
    response = app.client.post("/whatsapp", data={"From": PHONE, "Body": "help"})
    assert response.status_code == 403 and "TWILIO_AUTH_TOKEN" in response.text
    monkeypatch.setenv("ALLOW_UNSIGNED_WEBHOOK", "true")
    assert app.client.post("/whatsapp", data={"From": PHONE, "Body": "help"}).status_code == 200


def test_webhook_validates_the_public_url_behind_a_tunnel(client_factory, monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.ngrok.app/")
    app = client_factory()
    form = {"From": PHONE, "Body": "help"}
    assert app.post(form, url="https://example.ngrok.app/whatsapp").status_code == 200
    assert app.post(form, url="http://testserver/whatsapp").status_code == 403


def test_webhook_edge_cases(client_factory) -> None:
    app = client_factory()
    assert app.post({"Body": "help"}).status_code == 400  # no sender
    optout.opt_out(PHONE)
    silent = app.post({"From": PHONE, "Body": "Thanjavur rice"})
    assert (
        silent.status_code == 200 and "<Message>" not in silent.text
    )  # opted out: an empty response
    assert app.recorder.calls == []


# ---------------------------------------------------------------------------------------------------------------
# bulk composer
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def bulk():
    import importlib.util
    import sys

    from config import ROOT

    spec = importlib.util.spec_from_file_location(
        "bulk_advisory", ROOT / "scripts" / "bulk_advisory.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["bulk_advisory"] = module
    spec.loader.exec_module(module)
    return module


@needs_districts
def test_bulk_composer_reports_every_row_and_never_sends(bulk, monkeypatch, tmp_path) -> None:
    def no_send(*args, **kwargs):
        raise AssertionError("the bulk composer must not send")

    monkeypatch.setattr(twilio_client, "send", no_send)
    monkeypatch.setattr(bulk, "compose", Recorder())
    optout.opt_out("+919876500003")
    source = tmp_path / "recipients.csv"
    rows = [
        ["phone", "district", "crop", "lang"],
        ["+919876500001", "Thanjavur", "rice", "ta"],
        ["+919876500002", "thanjavr", "Rice", ""],
        ["+919876500003", "Guntur", "rice", "en"],
        ["+919876500004", "xyzzy", "rice", "en"],
        ["+919876500005", "godavari", "rice", "en"],
        ["+919876500006", "Guntur", "barley", "en"],
        ["+919876500007", "Guntur", "rice", "fr"],
        ["12345", "Guntur", "rice", "en"],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    out = tmp_path / "out.csv"
    target, counts = bulk.run(source, date(2024, 6, 15), out, include_phone=False)
    result = list(csv.DictReader(target.open(encoding="utf-8-sig")))
    assert [r["status"].split(":")[0] for r in result] == [
        "ok",
        "ok",
        "skipped_opt_out",
        "error",
        "error",
        "error",
        "error",
        "error",
    ]
    assert counts == {"ok": 2, "skipped_opt_out": 1, "error": 5}
    assert (
        result[0]["district_id"] == "tamil_nadu__thanjavur"
        and result[0]["lang"] == "ta"
        and "ADVICE" in result[0]["text"]
    )
    assert result[1]["lang"] == "en" and result[1]["crop"] == "rice"
    assert result[0]["phone"] == "+91******0001" and not any(
        "9876500" in r["phone"] for r in result
    )
    assert (
        "ambiguous" in result[4]["status"]
        and "unknown crop" in result[5]["status"]
        and "unsupported language" in result[6]["status"]
    )
    raw = bulk.run(source, date(2024, 6, 15), out, include_phone=True)[0]
    assert "+919876500001" in raw.read_text(encoding="utf-8-sig")


@needs_districts
def test_bulk_main_exit_code_reflects_errors(bulk, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bulk, "compose", Recorder())
    good = tmp_path / "good.csv"
    good.write_text("phone,district,crop,lang\n+919876500001,Guntur,rice,en\n", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_text("phone,district,crop,lang\n+919876500001,nowhere,rice,en\n", encoding="utf-8")
    assert bulk.main([str(good), "--as-of", "2024-06-15", "--output", str(tmp_path / "a.csv")]) == 0
    assert bulk.main([str(bad), "--as-of", "2024-06-15", "--output", str(tmp_path / "b.csv")]) == 1


@needs_models
def test_bulk_composer_with_the_real_models(bulk, tmp_path) -> None:
    source = tmp_path / "r.csv"
    source.write_text(
        "phone,district,crop,lang\n+919876500001,Thanjavur,rice,en\n+919876500002,Warangal,cotton,hi\n",
        encoding="utf-8",
    )
    target, counts = bulk.run(source, date(2024, 6, 15), tmp_path / "o.csv", include_phone=False)
    assert counts == {"ok": 2}
    rows = list(csv.DictReader(target.open(encoding="utf-8-sig")))
    assert all(0 < int(r["chars"]) <= LIMIT and int(r["sms_parts"]) >= 1 for r in rows)
