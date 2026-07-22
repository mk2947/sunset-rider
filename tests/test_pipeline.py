"""End-to-end pipeline tests against recorded fixtures. No network calls.

These cover the wiring that unit tests miss: corridor placement along the real solar
azimuth, batching, radius gating, and the CLI's dry-run contract.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from sunset_rider.geo import haversine_km, initial_bearing
from sunset_rider.main import main as cli_main
from sunset_rider.message import assert_no_bare_score
from sunset_rider.pipeline import (
    corridor_points,
    good_threshold,
    run_deterministic,
    run_plan,
    select_viewpoints,
)
from sunset_rider.solar import SolarCalculator
from sunset_rider.weather import (
    _ensemble_from_payload,
    _point_from_payload,
    load_ensemble_fixture,
    load_point_fixture,
)

FORECAST_VARS = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "relative_humidity_2m", "precipitation_probability", "precipitation",
    "apparent_temperature", "wind_speed_10m", "wind_gusts_10m", "cape",
    "wind_speed_500hPa", "wind_speed_700hPa",
]
ENSEMBLE_VARS = [
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "precipitation", "cape", "wind_gusts_10m",
]

VIEWPOINTS = [
    {"id": "node_1", "name": "Walbury Hill", "kind": "peak",
     "latitude": 51.3525, "longitude": -1.4650, "elevation_m": 297.0,
     "open_arc": [280, 330], "horizon_openness": 0.93, "elevation_prominence": 0.62,
     "foreground_interest": 0.7, "foreground_motion": 0.2,
     "distance_km": 36.7, "road_distance_km": 47.7, "minutes_one_way": 59.6,
     "close_fallback": False, "gate_closes": None, "spot_score": 82.0},
    {"id": "node_2", "name": "Dinton Pastures", "kind": "park",
     "latitude": 51.4550, "longitude": -0.8900, "elevation_m": 40.0,
     "open_arc": [240, 310], "horizon_openness": 0.71, "elevation_prominence": 0.05,
     "foreground_interest": 0.3, "foreground_motion": 1.0,
     "distance_km": 5.4, "road_distance_km": 7.0, "minutes_one_way": 8.8,
     "close_fallback": True, "gate_closes": "unknown", "spot_score": 55.0},
    {"id": "node_3", "name": "Locked Car Park", "kind": "parking",
     "latitude": 51.5100, "longitude": -1.1400, "elevation_m": 180.0,
     "open_arc": [250, 320], "horizon_openness": 0.85, "elevation_prominence": 0.40,
     "foreground_interest": 0.5, "foreground_motion": 0.0,
     "distance_km": 15.0, "road_distance_km": 19.5, "minutes_one_way": 24.4,
     "close_fallback": False, "gate_closes": "sunset", "spot_score": 70.0},
]


class FixtureClient:
    """Returns the same recorded forecast for every coordinate.

    Sufficient for exercising the wiring: what matters here is that the right number
    of points are requested, batched into one call, and threaded to the right places.
    """

    def __init__(self, fixtures_dir):
        self._forecast = json.loads(
            (fixtures_dir / "forecast_multi.json").read_text(encoding="utf-8"))[0]
        self._ensemble = json.loads(
            (fixtures_dir / "ensemble_ecmwf.json").read_text(encoding="utf-8"))
        self.forecast_calls = []
        self.ensemble_calls = []

    def forecast(self, coordinates, *, start_date, end_date, variables=None):
        self.forecast_calls.append(list(coordinates))
        payload = dict(self._forecast)
        return [_point_from_payload(payload, variables or FORECAST_VARS)
                for _ in coordinates]

    def ensemble(self, coordinates, *, start_date, end_date, variables=None, model=None):
        self.ensemble_calls.append(list(coordinates))
        return [_ensemble_from_payload(dict(self._ensemble), variables or ENSEMBLE_VARS)
                for _ in coordinates]


@pytest.fixture()
def client(fixtures_dir):
    return FixtureClient(fixtures_dir)


@pytest.fixture()
def fixture_date(fixtures_dir):
    """The fixtures were recorded for a specific window; use a date inside it."""
    payload = json.loads((fixtures_dir / "forecast_multi.json").read_text(encoding="utf-8"))
    first = payload[0]["hourly"]["time"][0]
    return dt.date.fromisoformat(first[:10])


# ---------------------------------------------------------------------------
# Corridor placement
# ---------------------------------------------------------------------------

def test_corridor_points_lie_along_the_solar_azimuth(config):
    solar = SolarCalculator(config)
    events = solar.events(dt.date(2025, 6, 21))
    points = corridor_points(config.home.latitude, config.home.longitude,
                             events.sun_bearing, config)
    assert len(points) == len(config.corridor.distances_km)
    for (lat, lon), distance in zip(points, config.corridor.distances_km):
        assert haversine_km(config.home.latitude, config.home.longitude,
                            lat, lon) == pytest.approx(float(distance), abs=0.05)
        assert initial_bearing(config.home.latitude, config.home.longitude,
                               lat, lon) == pytest.approx(events.sun_bearing, abs=0.5)


def test_corridor_moves_between_june_and_december(config):
    """The corridor follows the sun, so it points somewhere different in winter."""
    solar = SolarCalculator(config)
    june = corridor_points(config.home.latitude, config.home.longitude,
                           solar.sun_bearing(dt.date(2025, 6, 21)), config)
    december = corridor_points(config.home.latitude, config.home.longitude,
                               solar.sun_bearing(dt.date(2025, 12, 21)), config)
    assert haversine_km(*june[-1], *december[-1]) > 100.0


def test_good_threshold_comes_from_the_band_table(config):
    assert good_threshold(config) == 60.0


# ---------------------------------------------------------------------------
# Deterministic run
# ---------------------------------------------------------------------------

def test_deterministic_run_scores_and_ranks(config, client, fixture_date):
    result = run_deterministic(config, fixture_date, "go", client,
                               viewpoints=VIEWPOINTS)
    assert result.spots
    worths = [s.worth_it for s in result.spots]
    assert worths == sorted(worths, reverse=True), "spots must be ranked"


def test_a_gate_that_closes_at_sunset_excludes_the_spot(config, client, fixture_date):
    result = run_deterministic(config, fixture_date, "go", client,
                               viewpoints=VIEWPOINTS)
    locked = [s for s in result.spots if s.name == "Locked Car Park"]
    if locked:  # only present if inside tonight's radius
        assert locked[0].blocked is True
        assert locked[0].worth_it == 0.0
        assert any("gate" in r.lower() for r in locked[0].blockers.reasons)


def test_every_coordinate_is_fetched_in_one_batched_call(config, client, fixture_date):
    """Never loop one request per point."""
    run_deterministic(config, fixture_date, "go", client, viewpoints=VIEWPOINTS)
    # One regional call, then one call covering all viewpoints and their corridors.
    assert len(client.forecast_calls) == 2
    per_spot = 1 + len(config.corridor.distances_km)
    in_range = len(client.forecast_calls[1]) // per_spot
    assert len(client.forecast_calls[1]) == in_range * per_spot


def test_leave_by_is_present_and_sane_for_every_spot(config, client, fixture_date):
    result = run_deterministic(config, fixture_date, "go", client,
                               viewpoints=VIEWPOINTS)
    for spot in result.spots:
        assert spot.leave_by < spot.events.sunset
        gap = (spot.events.sunset - spot.leave_by).total_seconds() / 60.0
        assert gap == pytest.approx(
            float(config.rider.setup_minutes) + spot.minutes_one_way, abs=0.01)


def test_dynamic_radius_gates_the_candidate_set(config):
    assert len(select_viewpoints(VIEWPOINTS, 10.0)) == 1
    assert len(select_viewpoints(VIEWPOINTS, 70.0)) == 3


def test_out_of_range_spots_are_reported_as_excluded(config, client, fixture_date):
    result = run_deterministic(config, fixture_date, "go", client,
                               viewpoints=VIEWPOINTS)
    total = len(result.spots) + sum(
        1 for _, reason in result.excluded if "radius" in reason)
    assert total == len(VIEWPOINTS)


# ---------------------------------------------------------------------------
# Ensemble run
# ---------------------------------------------------------------------------

def test_plan_run_produces_five_evenings(config, client, fixture_date):
    result = run_plan(config, fixture_date, client)
    assert len(result.outlooks) == int(config.schedule.plan.horizon_days)


def test_plan_run_scores_every_member(config, client, fixture_date):
    result = run_plan(config, fixture_date, client)
    for outlook in result.outlooks:
        assert outlook.member_count > 30, "should be scoring the full ECMWF ensemble"
        assert len(outlook.member_skies) == outlook.member_count
        assert 0.0 <= outlook.probability_above_good <= 1.0
        assert outlook.stats["iqr"] >= 0.0


def test_plan_requests_a_corridor_for_each_evening(config, client, fixture_date):
    run_plan(config, fixture_date, client)
    assert len(client.ensemble_calls) == 1
    horizon = int(config.schedule.plan.horizon_days)
    expected = 1 + horizon * len(config.corridor.distances_km)
    assert len(client.ensemble_calls[0]) == expected


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

def test_dry_run_prints_and_sends_nothing(config, monkeypatch, capsys,
                                          fixtures_dir, fixture_date):
    """--dry-run must make no network writes; a Telegram call here would raise."""
    fixture_client = FixtureClient(fixtures_dir)
    monkeypatch.setattr("sunset_rider.main.OpenMeteoClient",
                        lambda cfg: fixture_client)

    def explode(*args, **kwargs):
        raise AssertionError("dry-run must not construct a Telegram client")

    monkeypatch.setattr("sunset_rider.main.TelegramClient", explode)
    monkeypatch.setattr("sunset_rider.pipeline.load_viewpoints", lambda cfg: VIEWPOINTS)

    exit_code = cli_main(["--dry-run", "--mode", "go",
                          "--date", fixture_date.isoformat()])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.strip(), "dry run printed nothing"
    assert "LEAVE BY" in out or "NO-GO" in out


def test_dry_run_plan_prints_a_ranking_without_point_scores(config, monkeypatch,
                                                            capsys, fixtures_dir,
                                                            fixture_date):
    """Done-condition #2, and the plan-mode honesty rule, in one test."""
    fixture_client = FixtureClient(fixtures_dir)
    monkeypatch.setattr("sunset_rider.main.OpenMeteoClient",
                        lambda cfg: fixture_client)
    monkeypatch.setattr("sunset_rider.main.TelegramClient",
                        lambda *a, **k: pytest.fail("no send in dry run"))

    assert cli_main(["--dry-run", "--mode", "plan",
                     "--date", fixture_date.isoformat()]) == 0
    out = capsys.readouterr().out
    assert "NEXT 5 EVENINGS" in out
    assert "% of members above" in out
    assert_no_bare_score(out)


def test_a_failed_fetch_reports_the_reason_instead_of_going_silent(
        config, monkeypatch, capsys, fixture_date):
    """Silence is the worst failure mode."""
    from sunset_rider.weather import WeatherUnavailable

    class BrokenClient:
        def forecast(self, *a, **k):
            raise WeatherUnavailable("server returned HTTP 500", source="forecast")

        def ensemble(self, *a, **k):
            raise WeatherUnavailable("server returned HTTP 500", source="forecast")

    monkeypatch.setattr("sunset_rider.main.OpenMeteoClient", lambda cfg: BrokenClient())
    monkeypatch.setattr("sunset_rider.pipeline.load_viewpoints", lambda cfg: VIEWPOINTS)

    exit_code = cli_main(["--dry-run", "--mode", "go",
                          "--date", fixture_date.isoformat()])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "unavailable" in out.lower()
    assert "HTTP 500" in out


def test_bad_date_is_rejected_clearly():
    with pytest.raises(SystemExit, match="YYYY-MM-DD"):
        cli_main(["--dry-run", "--mode", "go", "--date", "18-06-2025"])
