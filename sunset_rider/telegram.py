"""Telegram delivery.

Credentials come from the environment only — TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID, stored as GitHub Actions secrets. Nothing here reads or writes a
token to disk, and the token is never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import requests

from .config import Config

log = logging.getLogger(__name__)

TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ENV = "TELEGRAM_CHAT_ID"


class TelegramError(RuntimeError):
    pass


@dataclass
class Credentials:
    token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> "Credentials":
        token = os.environ.get(TOKEN_ENV, "").strip()
        chat_id = os.environ.get(CHAT_ENV, "").strip()
        missing = [name for name, value in ((TOKEN_ENV, token), (CHAT_ENV, chat_id))
                   if not value]
        if missing:
            raise TelegramError(
                f"missing environment variable(s): {', '.join(missing)}. "
                f"Set them as GitHub Actions secrets — see the README."
            )
        return cls(token=token, chat_id=chat_id)


class TelegramClient:
    def __init__(self, config: Config, credentials: Credentials | None = None,
                 session: requests.Session | None = None) -> None:
        self._config = config
        self._credentials = credentials
        self._session = session or requests.Session()

    @property
    def credentials(self) -> Credentials:
        if self._credentials is None:
            self._credentials = Credentials.from_env()
        return self._credentials

    def _url(self, method: str) -> str:
        return f"{self._config.telegram.api_base}/bot{self.credentials.token}/{method}"

    def send_message(self, text: str, *, disable_preview: bool = True) -> dict:
        """Send one message. Raises TelegramError with the API's own reason on failure."""
        response = self._session.post(
            self._url("sendMessage"),
            data={
                "chat_id": self.credentials.chat_id,
                "text": text,
                "parse_mode": self._config.telegram.parse_mode,
                "disable_web_page_preview": "true" if disable_preview else "false",
            },
            timeout=float(self._config.weather.timeout_seconds),
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TelegramError(
                f"Telegram returned non-JSON (HTTP {response.status_code})"
            ) from exc
        if not payload.get("ok"):
            # Never include the token in an error message.
            raise TelegramError(
                f"Telegram API error {payload.get('error_code')}: "
                f"{payload.get('description')}"
            )
        log.info("sent Telegram message (%d chars)", len(text))
        return payload

    def get_updates(self, offset: int | None = None) -> list[dict]:
        """Poll for replies — used by the weekly calibration job to read /rate N."""
        params: dict[str, object] = {"timeout": 0}
        if offset is not None:
            params["offset"] = offset
        response = self._session.get(
            self._url("getUpdates"), params=params,
            timeout=float(self._config.weather.timeout_seconds),
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TelegramError("Telegram returned non-JSON from getUpdates") from exc
        if not payload.get("ok"):
            raise TelegramError(
                f"Telegram API error {payload.get('error_code')}: "
                f"{payload.get('description')}"
            )
        return list(payload.get("result", []))
