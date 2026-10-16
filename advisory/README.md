# Farmer advisory (Phase 9)

Turns model output into short farmer messages and delivers them through SMS or a WhatsApp assistant.

```
risk_forecast (Phase 5)  ─┐
irrigation plan (Phase 6) ├─►  advisory/generate.py  ─►  text (en / hi / ta / te, <= 320 characters)
crop calendar             ─┘     fixed templates, rules                │
                                                                       ├─► dashboards page "Farmer advisory"
                                                                       ├─► advisory/bot.py  (WhatsApp webhook)
                                                                       ├─► scripts/bulk_advisory.py  (CSV, never sends)
                                                                       └─► advisory/twilio_client.py (outbound, dry run by default)
```

**No language model writes any advice.** Messages are chosen by explicit rules (`generate.choose_messages`) and filled
into the templates in `templates.yaml`.

## Read this before sending anything to a farmer

| Item | State | What you must do |
|---|---|---|
| Hindi, Tamil and Telugu texts | **Unreviewed drafts** written by an AI assistant (decision D-026) | Have a native speaker *and* an agronomist review every line of `templates.yaml` |
| Advice content (water early morning, mulch, avoid midday spraying) | Generic good practice, not tuned to a crop stage | Have an agronomist or the local agriculture office approve the wording |
| Forecast skill | Heatwave probability is climatology (the LSTM did not beat it); the drought model has skill on the held-out years; see `reports/extreme_weather_report.md` | Do not describe the messages as precise forecasts |
| Weather data | There is **no live feed**: the bot answers for the last observed day (or `ADVISORY_AS_OF`) | Connect a real forecast source before any real-time use |
| Irrigation amounts | Optimized schedule for a *typical-rainfall year*, not this season's weather | Present as a planning value |
| Delivery | `DRY_RUN=true` by default; nothing leaves the machine | Switch only for a consented, tested audience |
| Regulation | SMS to Indian numbers is regulated (sender and template registration under India's DLT rules; check Twilio's current guidance) | Complete the registration before SMS at scale |

## Compose an advisory

```bash
python -m advisory.generate tamil_nadu__thanjavur rice 2024-06-15 --lang ta --explain
```

In Python: `generate(district_id, crop, as_of_date, lang) -> str`, or `compose(...)` for the text plus the reasoning
(`messages`, `risk`, `irrigation_mm`, `notes`). `risk_forecast` only accepts dates inside the observed weather record
(`extreme_weather.forecast_date_range()`).

Rules, in order: (1) the more severe of heatwave and drought, if at least `advisory.alert_min_level` (moderate), gives
its alert (a drought signal for a normally dry month is suppressed, D-024); (2) if the crop is in its 16-week season and
this week's planned irrigation is at least `irrigation_min_mm`, the irrigation plan; otherwise, within
`sowing_lead_days` of the season start, the sowing window; (3) if nothing applies, "all clear". At most two messages;
the second is dropped if the text would exceed `sms_max_chars` (320). Indic scripts use Unicode SMS (70 characters
per part), so a message can span several parts; the dashboard shows the count.

## WhatsApp assistant (Twilio sandbox)

The assistant understands `Thanjavur rice`, `warangal cotton hindi`, `advisory for east godavari paddy on 2023-05-20`,
and the commands `help`, `yes`/`no`, `stop`/`start`. District names are matched fuzzily (including old and modern
spellings such as Tanjore or Mysuru) and must be typed in English letters. A match that is not clearly the best
(for example "godavari") is **confirmed first** ("Which one: East Godavari or West Godavari?"). The crop and language
are remembered for 30 minutes, in memory only. `STOP` is honoured permanently (hashed number in
`data/interim/advisory_optout.json`).

Setup (manual steps; they need your Twilio account and a phone):

1. Create a Twilio account and open **Messaging > Try it out > Send a WhatsApp message**. From your phone, send the
   `join <your-code>` message shown there to the sandbox number (the sandbox only reaches phones that joined).
2. Copy the **Account SID** and **Auth Token** into `.env` (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`). Keep
   `DRY_RUN=true`. The webhook refuses all requests when no auth token is set.
3. Start the bot: `make bot` (= `uvicorn advisory.bot:app --port 8000`). Check `curl localhost:8000/health`.
4. Expose it: `ngrok http 8000`, then put the address in `.env` as `PUBLIC_BASE_URL=https://<id>.ngrok-free.app` and
   restart the bot (the signature is validated against the URL Twilio called).
5. In the sandbox settings set **When a message comes in** to `https://<id>.ngrok-free.app/whatsapp`, method POST.
6. Send `Thanjavur rice` from your phone. The reply comes back as TwiML, so it does not use `DRY_RUN`.

Local tests without Twilio: `ALLOW_UNSIGNED_WEBHOOK=true` accepts unsigned calls (never on a public server), and
`make test-bot` starts the server and checks signed, forged and unsigned requests (`reports/bot_smoke.json`).

## Outbound messages

```bash
python -m advisory.twilio_client --to +91XXXXXXXXXX --body "test" --channel whatsapp     # dry run: logs only
DRY_RUN=false python -m advisory.twilio_client --to +91XXXXXXXXXX --body "test"            # real send (needs credentials)
```

`send_sms` / `send_whatsapp` validate the number (E.164), skip opted-out numbers, enforce
`advisory.delivery` limits (30 per minute overall, 3 per phone per day), and log only masked numbers and lengths
(`data/interim/advisory_audit.jsonl`). Real sending needs `DRY_RUN=false` **and** `TWILIO_ACCOUNT_SID`,
`TWILIO_AUTH_TOKEN` and the sender number; with any of them missing it raises `ConfigError` instead of sending.
In the WhatsApp sandbox, free-form messages are only delivered inside the 24-hour window after the person last wrote.

## Bulk composition

```bash
python scripts/bulk_advisory.py recipients.csv --as-of 2024-06-15 [--output out.csv] [--include-phone]
```

`recipients.csv` has the columns `phone,district,crop,lang`. The script writes the composed text, length, SMS parts and
a status per row (`ok`, `skipped_opt_out`, or the reason for an error such as an ambiguous district); phone numbers are
masked unless `--include-phone` is given. **It never sends.** Review the output, then deliver through
`advisory.twilio_client`.

## Tests

`tests/test_advisory.py` covers every risk level in every language, the 320-character limit for all districts, crops,
languages and message types, GSM-7 and Unicode segment counts, the rules, parsing and fuzzy matching, the conversation
(confirmation, memory, expiry, rate limit, STOP), the dry-run guard, rate limits, signature validation of the webhook and
the bulk tool. Safeguards were mutation-checked: breaking the dry-run default, the signature check, the ambiguity rule
or the opt-out check makes the suite fail.
