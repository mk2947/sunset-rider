"""Calibration loop.

Every `go` message records what it predicted. A weekly job polls Telegram for
``/rate N`` replies and joins them against the most recent prediction, building
``data/calibration.csv``.

There is deliberately NO automatic weight fitting here. Twenty data points cannot
support it, and a model that quietly retunes itself is a model you can no longer
reason about. This produces evidence for a human to act on. See docs/TUNING.md.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config, repo_root
from .models import RunResult
from .telegram import TelegramClient

log = logging.getLogger(__name__)

RATE_RE = re.compile(r"^/rate\s+([1-5])\b", re.IGNORECASE)

PREDICTION_FIELDS = [
    "sent_at", "date", "mode", "spot_id", "spot_name", "predicted_worth_it",
    "predicted_sky", "best_mode", "photo", "video", "ride", "clearing_front",
    "total_cc", "cloud_cover_mid", "cloud_cover_high", "corridor", "slot",
]

CALIBRATION_FIELDS = PREDICTION_FIELDS + ["actual", "rated_at"]


def _predictions_path(config: Config) -> Path:
    return repo_root() / "data" / "predictions.csv"


def _calibration_path(config: Config) -> Path:
    return repo_root() / config.calibration.csv_path


def _offset_path() -> Path:
    return repo_root() / "state" / "telegram_offset.json"


def _append_row(path: Path, fields: Sequence[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def record_prediction(result: RunResult, config: Config) -> None:
    """Log what a `go` message claimed, so a later rating can be scored against it."""
    if result.mode != "go":
        return
    rideable = [s for s in result.spots if not s.blocked]
    if not rideable:
        return
    spot = rideable[0]
    components = spot.sky.components
    _append_row(_predictions_path(config), PREDICTION_FIELDS, {
        "sent_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        "date": result.target_date.isoformat(),
        "mode": result.mode,
        "spot_id": spot.viewpoint.get("id", ""),
        "spot_name": spot.name,
        "predicted_worth_it": round(spot.worth_it, 1),
        "predicted_sky": round(spot.sky.sky, 1),
        "best_mode": spot.sky.best_mode,
        "photo": round(spot.output.photo, 1),
        "video": round(spot.output.video, 1),
        "ride": round(spot.ride.score, 1),
        "clearing_front": int(spot.sky.clearing_front),
        "total_cc": round(components.get("moody_deck", 0.0), 3),
        "cloud_cover_mid": round(components.get("moody_texture", 0.0), 3),
        "cloud_cover_high": round(components.get("vivid_canvas", 0.0), 3),
        "corridor": round(components.get("corridor", 0.0), 3),
        "slot": round(components.get("moody_slot", 0.0), 3),
    })


def _read_predictions(config: Config) -> list[dict]:
    path = _predictions_path(config)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _already_rated(config: Config) -> set[tuple[str, str]]:
    path = _calibration_path(config)
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {(row.get("date", ""), row.get("spot_id", ""))
                for row in csv.DictReader(handle)}


def _load_offset() -> int | None:
    path = _offset_path()
    if not path.is_file():
        return None
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("offset"))
    except (ValueError, TypeError, OSError):
        return None


def _save_offset(offset: int) -> None:
    path = _offset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"offset": offset}) + "\n", encoding="utf-8")


def parse_rating(text: str) -> int | None:
    """Extract N from a ``/rate N`` reply, or None."""
    match = RATE_RE.match(str(text).strip())
    return int(match.group(1)) if match else None


def collect_ratings(config: Config, client: TelegramClient) -> int:
    """Poll Telegram, join ratings to predictions, append to the calibration CSV.

    Returns the number of new rows written.
    """
    offset = _load_offset()
    updates = client.get_updates(offset=offset)
    if not updates:
        log.info("no new Telegram updates")
        return 0

    predictions = _read_predictions(config)
    rated = _already_rated(config)
    written = 0
    highest = offset or 0

    for update in updates:
        highest = max(highest, int(update.get("update_id", 0)))
        message = update.get("message") or update.get("edited_message") or {}
        rating = parse_rating(message.get("text", ""))
        if rating is None:
            continue

        sent_at = message.get("date")
        when = (dt.datetime.fromtimestamp(sent_at, tz=dt.timezone.utc)
                if sent_at else dt.datetime.now(tz=dt.timezone.utc))

        # Join to the most recent prediction at or before the rating.
        candidates = [p for p in predictions
                      if p.get("sent_at", "") <= when.isoformat(timespec="seconds")]
        if not candidates:
            log.warning("rating %d has no matching prediction; skipping", rating)
            continue
        prediction = candidates[-1]
        key = (prediction.get("date", ""), prediction.get("spot_id", ""))
        if key in rated:
            continue

        row = dict(prediction)
        row["actual"] = rating
        row["rated_at"] = when.isoformat(timespec="seconds")
        _append_row(_calibration_path(config), CALIBRATION_FIELDS, row)
        rated.add(key)
        written += 1

    if highest:
        _save_offset(highest + 1)
    log.info("wrote %d calibration rows", written)
    return written


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Collect /rate replies from Telegram.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    collect_ratings(config, TelegramClient(config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
