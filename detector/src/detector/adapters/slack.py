import logging
import os
import time
from functools import cache

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

from ..model import IS_AI_THRESHOLD, inference

router = APIRouter()
logger = logging.getLogger(__name__)

EVENT_RETENTION = 25 * 60 * 60
FLAGGED_MESSAGE_REPLY = (
    "you sent me AI content, i aint reading allat."
)
_received_events: dict[str, float] = {}


@cache
def _slack_session(token: str) -> tuple[WebClient, str]:
    client = WebClient(token=token, timeout=10, retry_handlers=[])
    return client, client.auth_test()["user_id"]


def _handle_dm(event: dict, authorizations: list, token: str, event_id: str) -> None:
    try:
        client, user_id = _slack_session(token)
        if event["user"] == user_id or not any(
            isinstance(auth, dict)
            and auth.get("user_id") == user_id
            and auth.get("is_bot") is False
            for auth in authorizations
        ):
            return

        if inference(event["text"]).ai_score <= IS_AI_THRESHOLD:
            return

        client.chat_postMessage(
            channel=event["channel"],
            text=FLAGGED_MESSAGE_REPLY,
            thread_ts=event.get("thread_ts"),
        )
    except (SlackApiError, OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.error("Slack DM processing failed for %s (%s)", event_id, type(exc).__name__)


@router.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    secret = os.getenv("SLACK_SIGNING_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="Slack signing secret is missing")

    try:
        valid = SignatureVerifier(secret).is_valid_request(
            await request.body(), request.headers
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Slack payload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid Slack payload")

    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge")
        if not isinstance(challenge, str):
            raise HTTPException(status_code=400, detail="Missing Slack challenge")
        return {"challenge": challenge}

    event = payload.get("event")
    if (
        payload.get("type") != "event_callback"
        or not isinstance(event, dict)
        or event.get("type") != "message"
        or event.get("channel_type") != "im"
        or event.get("subtype")
        or event.get("bot_id")
        or not all(
            isinstance(event.get(key), str) and event[key]
            for key in ("channel", "user", "text")
        )
        or not event["text"].strip()
    ):
        return Response(status_code=200)

    event_id = payload.get("event_id")
    authorizations = payload.get("authorizations")
    if not isinstance(event_id, str) or not event_id or not isinstance(authorizations, list):
        raise HTTPException(status_code=400, detail="Invalid Slack event envelope")

    token = os.getenv("USER_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="Slack user token is missing")

    now = time.monotonic()
    while _received_events:
        oldest = next(iter(_received_events))
        if now - _received_events[oldest] < EVENT_RETENTION:
            break
        del _received_events[oldest]
    if event_id not in _received_events:
        _received_events[event_id] = now
        background_tasks.add_task(_handle_dm, event, authorizations, token, event_id)
    return Response(status_code=200)
