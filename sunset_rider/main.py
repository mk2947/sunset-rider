"""CLI entry point.

    python -m sunset_rider.main --dry-run --mode plan --date 2025-06-18

Scheduling philosophy: cron decides whether to wake, Python decides whether to act.
GitHub Actions cron is UTC-only and would drift an hour across the GMT/BST boundary,
so the workflow runs hourly and this module self-gates against Europe/London.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from typing import Sequence
from zoneinfo import ZoneInfo

from .config import load_config
from .feedback import record_prediction
from .gating import due_modes, target_date_for
from .message import render, render_failure
from .models import RunResult
from .pipeline import run_deterministic, run_plan
from .solar import SolarCalculator, UTC
from .state import load_state
from .telegram import TelegramClient, TelegramError
from .weather import OpenMeteoClient, WeatherUnavailable

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sunset_rider",
        description="Tiered sunset-riding forecasts for a CBT rider near Reading.",
    )
    parser.add_argument("--mode", choices=["plan", "confirm", "go"],
                        help="force a mode; omit to self-gate on the schedule")
    parser.add_argument("--date", help="target date, YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print to stdout and make no network writes")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--ignore-state", action="store_true",
                        help="bypass send de-duplication")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="DEBUG logging, including full score breakdowns")
    return parser


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD, got {value!r}") from exc


def execute(mode: str, target_date: dt.date, config, client: OpenMeteoClient) -> RunResult:
    if mode == "plan":
        return run_plan(config, target_date, client)
    return run_deterministic(config, target_date, mode, client)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(args.config)
    tz = ZoneInfo(config.home.timezone)
    now_utc = dt.datetime.now(tz=UTC)
    now_local = now_utc.astimezone(tz)
    solar = SolarCalculator(config)

    explicit_date = _parse_date(args.date)

    # Decide which modes are due. An explicit --mode bypasses the schedule gate,
    # which is what makes --dry-run useful for any date.
    if args.mode:
        modes = [args.mode]
    else:
        today = explicit_date or now_local.date()
        modes = due_modes(now_utc, solar.sunset_utc(today), config)
        if not modes:
            log.info("nothing due at %s local; exiting quietly",
                     now_local.strftime("%Y-%m-%d %H:%M %Z"))
            return 0

    state = load_state(config)
    client = OpenMeteoClient(config)
    telegram: TelegramClient | None = None
    exit_code = 0

    for mode in modes:
        target_date = explicit_date or target_date_for(mode, now_local, config)

        if not args.dry_run and not args.ignore_state:
            if state.already_sent(mode, target_date):
                log.info("%s already sent for %s; skipping", mode, target_date)
                continue

        try:
            result = execute(mode, target_date, config, client)
            text = render(result, config)
        except (WeatherUnavailable, RuntimeError) as exc:
            # Silence is the worst failure mode. Tell the rider it broke, and why.
            log.error("%s run failed: %s", mode, exc)
            text = render_failure(mode, str(exc), config)
            exit_code = 1
            if args.dry_run:
                print(text)
                continue
            try:
                telegram = telegram or TelegramClient(config)
                telegram.send_message(text)
            except TelegramError as send_exc:
                log.error("could not report the failure either: %s", send_exc)
            continue

        if args.dry_run:
            print(text)
            continue

        try:
            telegram = telegram or TelegramClient(config)
            telegram.send_message(text)
        except TelegramError as exc:
            log.error("Telegram send failed: %s", exc)
            exit_code = 1
            continue

        state.mark_sent(mode, target_date)
        state.save()
        record_prediction(result, config)
        log.info("sent %s for %s", mode, target_date)

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
