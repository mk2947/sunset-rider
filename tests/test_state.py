"""Send-dedupe state tests.

The file this manages is also the 60-day Actions keepalive, so its failure modes
matter more than its size suggests.
"""

from __future__ import annotations

import datetime as dt
import json

from sunset_rider.feedback import parse_rating
from sunset_rider.state import MODES, SendState


def test_nothing_is_sent_initially(tmp_path):
    state = SendState(tmp_path / "last_sent.json")
    assert state.already_sent("go", dt.date(2025, 6, 18)) is False


def test_marking_and_saving_round_trips(tmp_path):
    path = tmp_path / "last_sent.json"
    state = SendState(path)
    state.mark_sent("go", dt.date(2025, 6, 18))
    state.save()

    reloaded = SendState(path)
    assert reloaded.already_sent("go", dt.date(2025, 6, 18)) is True


def test_dedupe_is_per_mode(tmp_path):
    state = SendState(tmp_path / "s.json")
    state.mark_sent("go", dt.date(2025, 6, 18))
    assert state.already_sent("confirm", dt.date(2025, 6, 18)) is False
    assert state.already_sent("plan", dt.date(2025, 6, 18)) is False


def test_dedupe_is_per_date(tmp_path):
    """A 60-minute window spans two hourly cron runs; the second must not resend."""
    state = SendState(tmp_path / "s.json")
    state.mark_sent("go", dt.date(2025, 6, 18))
    assert state.already_sent("go", dt.date(2025, 6, 18)) is True
    assert state.already_sent("go", dt.date(2025, 6, 19)) is False


def test_saved_file_always_contains_every_mode(tmp_path):
    """The workflow commits this file; a stable shape keeps diffs meaningful."""
    path = tmp_path / "s.json"
    state = SendState(path)
    state.mark_sent("go", dt.date(2025, 6, 18))
    state.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved) == set(MODES)


def test_a_corrupt_state_file_does_not_stop_tonights_forecast(tmp_path):
    """Worst case we send a duplicate, which beats sending nothing."""
    path = tmp_path / "s.json"
    path.write_text("{not json at all", encoding="utf-8")
    state = SendState(path)
    assert state.already_sent("go", dt.date(2025, 6, 18)) is False
    state.mark_sent("go", dt.date(2025, 6, 18))
    state.save()
    assert json.loads(path.read_text(encoding="utf-8"))["go"] == "2025-06-18"


def test_save_creates_missing_directories(tmp_path):
    state = SendState(tmp_path / "nested" / "deeper" / "s.json")
    state.mark_sent("plan", dt.date(2025, 6, 18))
    state.save()
    assert (tmp_path / "nested" / "deeper" / "s.json").is_file()


# ---------------------------------------------------------------------------
# /rate parsing
# ---------------------------------------------------------------------------

def test_rate_command_parsing():
    assert parse_rating("/rate 4") == 4
    assert parse_rating("  /rate 1  ") == 1
    assert parse_rating("/RATE 5") == 5


def test_non_rate_messages_are_ignored():
    for text in ["hello", "/rate", "/rate 6", "/rate 0", "rate 3", "/rated 3", ""]:
        assert parse_rating(text) is None
