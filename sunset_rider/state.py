"""Send-deduplication state.

``state/last_sent.json`` records the last date each mode was sent for, so that a
30-minute-wide send window crossing two hourly cron runs does not produce two
messages.

This file is committed back to the default branch after every send, and that commit
is doing double duty: GitHub disables scheduled workflows after 60 days with no
commits to the default branch, and does so silently. The dedupe commit is also the
keepalive. See the README before "tidying it up".
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from .config import Config, repo_root

log = logging.getLogger(__name__)

MODES = ("plan", "confirm", "go")


class SendState:
    """Tracks which modes have already been sent for which date."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            self._data = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # A corrupt state file must not stop tonight's forecast. Worst case we
            # send a duplicate, which is far better than sending nothing.
            log.warning("could not read %s (%s); treating as empty", self.path, exc)
            self._data = {}
            return
        self._data = {k: str(v) for k, v in raw.items() if isinstance(raw, dict)}

    def already_sent(self, mode: str, target: dt.date) -> bool:
        return self._data.get(mode) == target.isoformat()

    def mark_sent(self, mode: str, target: dt.date) -> None:
        self._data[mode] = target.isoformat()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {mode: self._data.get(mode, "") for mode in MODES}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    def as_dict(self) -> dict[str, str]:
        return dict(self._data)


def load_state(config: Config) -> SendState:
    return SendState(repo_root() / config.state.path)
