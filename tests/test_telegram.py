"""Telegram delivery tests, against a fake transport.

No test here contacts the real API. The delivery path has never been exercised
live — that is what the README's workflow_dispatch step is for.
"""

from __future__ import annotations

import datetime as dt

import pytest

from sunset_rider.feedback import collect_ratings, parse_rating, record_prediction
from sunset_rider.telegram import Credentials, TelegramClient, TelegramError

TOKEN = "123456789:FAKE-TOKEN-DO-NOT-USE"
CHAT = "987654321"


class FakeResponse:
    def __init__(self, payload, status_code=200, bad_json=False):
        self._payload = payload
        self.status_code = status_code
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, data=None, timeout=None):
        self.calls.append({"url": url, "data": data or {}})
        return self._responses.pop(0)

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}})
        return self._responses.pop(0)


@pytest.fixture()
def creds():
    return Credentials(token=TOKEN, chat_id=CHAT)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    creds = Credentials.from_env()
    assert creds.token == TOKEN and creds.chat_id == CHAT


def test_missing_credentials_name_what_is_missing_and_point_at_the_readme(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    with pytest.raises(TelegramError) as exc:
        Credentials.from_env()
    assert "TELEGRAM_BOT_TOKEN" in str(exc.value)
    assert "TELEGRAM_CHAT_ID" not in str(exc.value)
    assert "README" in str(exc.value)


def test_blank_credentials_are_treated_as_missing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    with pytest.raises(TelegramError, match="TELEGRAM_BOT_TOKEN"):
        Credentials.from_env()


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

def test_send_message_posts_the_expected_payload(config, creds):
    session = FakeSession([FakeResponse({"ok": True, "result": {"message_id": 1}})])
    client = TelegramClient(config, credentials=creds, session=session)
    client.send_message("hello")

    call = session.calls[0]
    assert call["url"].endswith("/sendMessage")
    assert call["data"]["chat_id"] == CHAT
    assert call["data"]["text"] == "hello"
    assert call["data"]["parse_mode"] == config.telegram.parse_mode


def test_api_errors_surface_the_reason(config, creds):
    session = FakeSession([FakeResponse(
        {"ok": False, "error_code": 400, "description": "chat not found"})])
    client = TelegramClient(config, credentials=creds, session=session)
    with pytest.raises(TelegramError) as exc:
        client.send_message("hello")
    assert "chat not found" in str(exc.value)
    assert "400" in str(exc.value)


def test_the_token_never_appears_in_an_error_message(config, creds):
    """A leaked token in a public Actions log would be a real incident."""
    session = FakeSession([FakeResponse(
        {"ok": False, "error_code": 401, "description": "Unauthorized"})])
    client = TelegramClient(config, credentials=creds, session=session)
    with pytest.raises(TelegramError) as exc:
        client.send_message("hello")
    assert TOKEN not in str(exc.value)


def test_non_json_response_is_reported(config, creds):
    session = FakeSession([FakeResponse(None, status_code=502, bad_json=True)])
    client = TelegramClient(config, credentials=creds, session=session)
    with pytest.raises(TelegramError, match="non-JSON"):
        client.send_message("hello")


# ---------------------------------------------------------------------------
# getUpdates
# ---------------------------------------------------------------------------

def test_get_updates_returns_the_result_list(config, creds):
    session = FakeSession([FakeResponse({"ok": True, "result": [{"update_id": 5}]})])
    client = TelegramClient(config, credentials=creds, session=session)
    assert client.get_updates() == [{"update_id": 5}]


def test_get_updates_passes_the_offset(config, creds):
    session = FakeSession([FakeResponse({"ok": True, "result": []})])
    client = TelegramClient(config, credentials=creds, session=session)
    client.get_updates(offset=42)
    assert session.calls[0]["params"]["offset"] == 42


def test_get_updates_reports_api_errors(config, creds):
    session = FakeSession([FakeResponse(
        {"ok": False, "error_code": 409, "description": "conflict"})])
    client = TelegramClient(config, credentials=creds, session=session)
    with pytest.raises(TelegramError, match="conflict"):
        client.get_updates()


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

def _prediction_run(config, tmp_path, monkeypatch):
    """Record one `go` prediction into a temp repo root."""
    import datetime as dt

    from sunset_rider.gating import BlockerResult, leave_by
    from sunset_rider.models import RunResult, SpotForecast
    from sunset_rider.scoring import (corridor_clearness, score_output, score_ride,
                                      score_sky, score_worth_it, combine_output)
    from sunset_rider.solar import SolarCalculator
    from tests.test_scoring import moody_day

    monkeypatch.setattr("sunset_rider.feedback.repo_root", lambda: tmp_path)

    events = SolarCalculator(config).events(dt.date(2025, 6, 18))
    inputs = moody_day()
    sky = score_sky(inputs, config)
    corridor = corridor_clearness(inputs.corridor_low, config)
    output = score_output(sky=sky.sky, inputs=inputs, corridor=corridor,
                          foreground_interest=0.7, foreground_motion=0.2,
                          gusts_kmh=20.0, wind_500hpa=40.0,
                          blue_hour_minutes=events.blue_hour_minutes, config=config)
    ride = score_ride(max_gust_kmh=20.0, max_precip_prob=10.0, apparent_temperature=16.0,
                      precip_preceding_3h=0.0, returns_after_dark_minutes=30.0,
                      config=config)
    viewpoint = {"id": "node_1", "name": "Walbury Hill", "spot_score": 82.0,
                 "minutes_one_way": 59.6, "open_arc": [280, 330]}
    worth, parts = score_worth_it(
        sky=sky.sky, output_score=combine_output(output.photo, output.video, config),
        ride=ride.score, spot=82.0, sun_bearing=events.sun_bearing,
        open_arc=[280, 330], minutes_one_way=59.6, blocked=False, config=config)

    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [SpotForecast(
        viewpoint=viewpoint, events=events, sky=sky, output=output, ride=ride,
        blockers=BlockerResult(blocked=False), worth_it=worth, worth_it_parts=parts,
        leave_by=leave_by(events, 59.6, config), minutes_one_way=59.6,
        max_gust_kmh=20.0, wind_500hpa=40.0)]
    record_prediction(result, config)
    return result


def test_predictions_are_recorded_for_go_runs(config, tmp_path, monkeypatch):
    _prediction_run(config, tmp_path, monkeypatch)
    csv_path = tmp_path / "data" / "predictions.csv"
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "Walbury Hill" in text and "moody" in text


def test_plan_runs_are_not_recorded_as_predictions(config, tmp_path, monkeypatch):
    from sunset_rider.models import RunResult

    monkeypatch.setattr("sunset_rider.feedback.repo_root", lambda: tmp_path)
    record_prediction(RunResult(mode="plan", target_date=dt.date(2025, 6, 18)), config)
    assert not (tmp_path / "data" / "predictions.csv").exists()


def test_a_rating_joins_to_the_prediction(config, tmp_path, monkeypatch):
    _prediction_run(config, tmp_path, monkeypatch)
    later = int(dt.datetime.now(tz=dt.timezone.utc).timestamp()) + 3600
    session = FakeSession([FakeResponse({"ok": True, "result": [
        {"update_id": 10, "message": {"date": later, "text": "/rate 4"}}]})])
    client = TelegramClient(config, credentials=Credentials(TOKEN, CHAT), session=session)

    written = collect_ratings(config, client)
    assert written == 1
    text = (tmp_path / config.calibration.csv_path).read_text(encoding="utf-8")
    assert "Walbury Hill" in text
    assert text.strip().splitlines()[-1].split(",")[-2] == "4"


def test_non_rating_messages_are_ignored(config, tmp_path, monkeypatch):
    _prediction_run(config, tmp_path, monkeypatch)
    later = int(dt.datetime.now(tz=dt.timezone.utc).timestamp()) + 3600
    session = FakeSession([FakeResponse({"ok": True, "result": [
        {"update_id": 11, "message": {"date": later, "text": "nice one"}}]})])
    client = TelegramClient(config, credentials=Credentials(TOKEN, CHAT), session=session)
    assert collect_ratings(config, client) == 0


def test_the_same_evening_is_not_rated_twice(config, tmp_path, monkeypatch):
    _prediction_run(config, tmp_path, monkeypatch)
    later = int(dt.datetime.now(tz=dt.timezone.utc).timestamp()) + 3600
    update = {"update_id": 12, "message": {"date": later, "text": "/rate 5"}}

    client = TelegramClient(config, credentials=Credentials(TOKEN, CHAT),
                            session=FakeSession([FakeResponse({"ok": True,
                                                               "result": [update]})]))
    assert collect_ratings(config, client) == 1

    again = TelegramClient(config, credentials=Credentials(TOKEN, CHAT),
                           session=FakeSession([FakeResponse({"ok": True,
                                                              "result": [update]})]))
    assert collect_ratings(config, again) == 0


def test_no_updates_writes_nothing(config, tmp_path, monkeypatch):
    monkeypatch.setattr("sunset_rider.feedback.repo_root", lambda: tmp_path)
    client = TelegramClient(config, credentials=Credentials(TOKEN, CHAT),
                            session=FakeSession([FakeResponse({"ok": True, "result": []})]))
    assert collect_ratings(config, client) == 0
