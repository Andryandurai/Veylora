"""WhatsApp (and SMS) webhook for the advisory assistant: ``uvicorn advisory.bot:app --port 8000``.

* ``POST /whatsapp`` is the Twilio webhook. Every request must carry a valid ``X-Twilio-Signature`` (computed with the
  account auth token over the public URL and the form fields). Without a configured auth token requests are
  refused, unless ``ALLOW_UNSIGNED_WEBHOOK=true`` is set for local testing. Behind a tunnel or proxy set
  ``PUBLIC_BASE_URL`` (for example the ngrok address) so that the validated URL is the one Twilio called.
* The reply is TwiML, so it goes back through Twilio's webhook response; no outbound API message is sent and the
  ``DRY_RUN`` switch of :mod:`advisory.twilio_client` does not apply. The assistant only answers people who wrote to it.
* ``GET /health`` reports readiness without touching any model.

Conversation rules (fuzzy district match with confirmation, remembered crop and language, STOP) live in
:mod:`advisory.conversation`.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

from advisory.conversation import Conversation
from advisory.twilio_client import TwilioSettings, mask_number
from config import ROOT

LOG = logging.getLogger("advisory.bot")
TRUE_WORDS = {"1", "true", "yes", "on"}


def _signature_ok(request: Request, form: dict[str, str], auth_token: str) -> bool:
    """Check the Twilio signature against the URL Twilio called (``PUBLIC_BASE_URL`` replaces the local host)."""
    base = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if base:
        url = base + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    else:
        url = str(request.url)
    signature = request.headers.get("X-Twilio-Signature", "")
    return bool(signature) and RequestValidator(auth_token).validate(url, form, signature)


def _twiml(text: str) -> Response:
    response = MessagingResponse()
    if text:
        response.message(text)
    return Response(content=str(response), media_type="application/xml")


def create_app(
    conversation: Conversation | None = None, settings: TwilioSettings | None = None
) -> FastAPI:
    """Build the FastAPI application (a fresh conversation store and settings unless given)."""
    settings = settings or TwilioSettings.from_env()
    talk = conversation or Conversation()
    app = FastAPI(title="ClimateResilience advisory bot", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, object]:
        models = [ROOT / "models" / "heatwave_lstm.pt", ROOT / "models" / "drought_xgb.joblib"]
        return {
            "status": "ok",
            "dry_run": settings.dry_run,
            "signature_validation": bool(settings.auth_token),
            "models_ready": all(path.exists() for path in models),
        }

    @app.post("/whatsapp")
    async def whatsapp(request: Request) -> Response:
        form = {key: str(value) for key, value in (await request.form()).items()}
        if settings.auth_token:
            if not _signature_ok(request, form, settings.auth_token):
                LOG.warning("rejected a webhook call with a missing or invalid signature")
                return JSONResponse({"detail": "invalid signature"}, status_code=403)
        elif os.environ.get("ALLOW_UNSIGNED_WEBHOOK", "").strip().lower() in TRUE_WORDS:
            LOG.warning("ALLOW_UNSIGNED_WEBHOOK is set: the request signature was not checked")
        else:
            return JSONResponse(
                {
                    "detail": "webhook signature validation is not configured (set TWILIO_AUTH_TOKEN)"
                },
                status_code=403,
            )
        sender = form.get("From", "").strip()
        if not sender:
            return JSONResponse({"detail": "missing From"}, status_code=400)
        reply = talk.handle(sender, form.get("Body", ""))
        LOG.info("%s -> %s (%d characters)", mask_number(sender), reply.kind, len(reply.text))
        return _twiml(reply.text)

    return app


app = create_app()
