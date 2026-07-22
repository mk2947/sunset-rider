"""Discovery I/O tests: harvest, caching, resumable profiling, YAML round-trip.

All network transports are faked. Nothing here touches Overpass or Open-Meteo.
"""

from __future__ import annotations

import json

import pytest
import requests

from sunset_rider.config import Config
from sunset_rider.discovery import (
    Candidate,
    HorizonProfile,
    harvest_osm,
    harvest_water,
    load_viewpoints,
    profile_horizons,
    run,
    write_viewpoints,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"elements": []}

    def json(self):
        if self._payload == "__bad__":
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers or {}})
        if not self._responses:
            raise AssertionError("FakeSession exhausted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeElevationClient:
    """Returns a constant elevation, optionally failing after N calls."""

    def __init__(self, elevation=100.0, fail_after=None, error=None):
        self.elevation_value = elevation
        self.fail_after = fail_after
        self.error = error
        self.calls = 0
        self.points_requested = 0

    def elevation(self, coordinates):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise self.error
        self.points_requested += len(coordinates)
        return [self.elevation_value] * len(coordinates)


@pytest.fixture()
def tmp_config(config, tmp_path, monkeypatch):
    """Config with all cache paths redirected into a temp directory."""
    monkeypatch.setattr("sunset_rider.discovery.repo_root", lambda: tmp_path)
    return config


def _element(osm_id, lat, lon, tags, type_="node"):
    return {"type": type_, "id": osm_id, "lat": lat, "lon": lon, "tags": tags}


# ---------------------------------------------------------------------------
# harvest_osm
# ---------------------------------------------------------------------------

def test_harvest_merges_the_three_subqueries(tmp_config, tmp_path):
    session = FakeSession([
        FakeResponse(200, {"elements": [_element(1, 51.5, -1.2, {"natural": "peak"})]}),
        FakeResponse(200, {"elements": [_element(2, 51.6, -1.3,
                                                 {"natural": "grassland", "name": "A"},
                                                 "way")]}),
        FakeResponse(200, {"elements": [_element(3, 51.7, -1.4,
                                                 {"amenity": "parking"}, "way")]}),
    ])
    payload = harvest_osm(tmp_config, force=True, session=session)
    assert len(payload["elements"]) == 3
    assert len(session.calls) == 3
    cache = tmp_path / tmp_config.discovery.raw_osm_cache
    assert cache.is_file()


def test_harvest_deduplicates_across_subqueries(tmp_config):
    same = _element(1, 51.5, -1.2, {"natural": "peak"})
    session = FakeSession([
        FakeResponse(200, {"elements": [same]}),
        FakeResponse(200, {"elements": [same]}),
        FakeResponse(200, {"elements": [same]}),
    ])
    payload = harvest_osm(tmp_config, force=True, session=session)
    assert len(payload["elements"]) == 1


def test_harvest_uses_the_cache_and_makes_no_calls(tmp_config, tmp_path):
    cache = tmp_path / tmp_config.discovery.raw_osm_cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"elements": [{"type": "node", "id": 9}]}),
                     encoding="utf-8")
    session = FakeSession([])
    payload = harvest_osm(tmp_config, session=session)
    assert payload["elements"][0]["id"] == 9
    assert session.calls == []


def test_harvest_sends_a_descriptive_user_agent(tmp_config):
    """overpass-api.de returns HTTP 406 for the default python-requests UA."""
    session = FakeSession([FakeResponse(200) for _ in range(3)])
    harvest_osm(tmp_config, force=True, session=session)
    for call in session.calls:
        assert "sunset_rider" in call["headers"]["User-Agent"]


def test_harvest_falls_back_to_a_mirror_on_failure(tmp_config):
    """The primary endpoint 429s and 504s under load; mirrors carry the heavy query."""
    session = FakeSession([
        FakeResponse(429),
        FakeResponse(200, {"elements": [_element(1, 51.5, -1.2, {"natural": "peak"})]}),
        FakeResponse(200),
        FakeResponse(200),
    ])
    payload = harvest_osm(tmp_config, force=True, session=session)
    assert len(payload["elements"]) == 1
    assert session.calls[0]["url"] == tmp_config.discovery.overpass_url
    assert session.calls[1]["url"] != tmp_config.discovery.overpass_url


def test_a_timed_out_subquery_falls_through_to_a_mirror(tmp_config):
    """A timeout arrives as HTTP 200 with a remark, which must not be cached."""
    session = FakeSession([
        FakeResponse(200, {"elements": [], "remark": "runtime error: Query timed out"}),
        FakeResponse(200, {"elements": [_element(1, 51.5, -1.2, {"natural": "peak"})]}),
        FakeResponse(200),
        FakeResponse(200),
    ])
    payload = harvest_osm(tmp_config, force=True, session=session)
    assert len(payload["elements"]) == 1


def test_harvest_raises_when_every_endpoint_fails(tmp_config):
    session = FakeSession([FakeResponse(504), FakeResponse(504), FakeResponse(504)])
    with pytest.raises(RuntimeError, match="failed on all endpoints"):
        harvest_osm(tmp_config, force=True, session=session)


def test_harvest_survives_a_network_exception_on_one_endpoint(tmp_config):
    session = FakeSession([
        requests.ConnectionError("dns"),
        FakeResponse(200, {"elements": [_element(1, 51.5, -1.2, {"natural": "peak"})]}),
        FakeResponse(200),
        FakeResponse(200),
    ])
    payload = harvest_osm(tmp_config, force=True, session=session)
    assert len(payload["elements"]) == 1


# ---------------------------------------------------------------------------
# profile_horizons — resumability is the point
# ---------------------------------------------------------------------------

def _candidates(count):
    return [Candidate(f"node/{i}", f"C{i}", "peak", 51.4 + i * 0.01, -1.0)
            for i in range(count)]


def test_profiles_are_computed_and_cached(tmp_config, tmp_path):
    client = FakeElevationClient()
    candidates = _candidates(3)
    profiles = profile_horizons(candidates, tmp_config, client)

    assert len(profiles) == 3
    cache = tmp_path / tmp_config.discovery.raw_horizon_cache
    assert cache.is_file()
    assert len(json.loads(cache.read_text(encoding="utf-8"))) == 3


def test_a_cached_profile_is_never_refetched(tmp_config, tmp_path):
    """Terrain does not change. Re-fetching is explicitly forbidden."""
    client = FakeElevationClient()
    candidates = _candidates(2)
    profile_horizons(candidates, tmp_config, client)
    first_points = client.points_requested

    second = FakeElevationClient()
    profiles = profile_horizons(candidates, tmp_config, second)
    assert second.points_requested == 0, "cached profiles were re-fetched"
    assert len(profiles) == 2
    assert first_points > 0


def test_only_new_candidates_are_fetched_on_a_top_up(tmp_config):
    client = FakeElevationClient()
    profile_horizons(_candidates(2), tmp_config, client)

    top_up = FakeElevationClient()
    profiles = profile_horizons(_candidates(4), tmp_config, top_up)
    assert len(profiles) == 4
    per_candidate = 1 + 36 * len(tmp_config.discovery.sample_distances_km)
    assert top_up.points_requested == 2 * per_candidate


def test_an_interrupted_run_keeps_its_progress(tmp_config, tmp_path):
    """A rate limit mid-run must not discard hours of expensive work."""
    from sunset_rider.weather import WeatherUnavailable

    client = FakeElevationClient(
        fail_after=2, error=WeatherUnavailable("rate limited", source="elevation"))
    profiles = profile_horizons(_candidates(5), tmp_config, client)

    assert len(profiles) == 2, "should have kept the two completed profiles"
    cache = json.loads(
        (tmp_path / tmp_config.discovery.raw_horizon_cache).read_text(encoding="utf-8"))
    assert len(cache) == 2


def test_an_interrupted_run_resumes_where_it_stopped(tmp_config):
    from sunset_rider.weather import WeatherUnavailable

    interrupted = FakeElevationClient(
        fail_after=2, error=WeatherUnavailable("rate limited", source="elevation"))
    profile_horizons(_candidates(5), tmp_config, interrupted)

    resumed = FakeElevationClient()
    profiles = profile_horizons(_candidates(5), tmp_config, resumed)
    assert len(profiles) == 5
    per_candidate = 1 + 36 * len(tmp_config.discovery.sample_distances_km)
    assert resumed.points_requested == 3 * per_candidate


def test_force_reprofiles_everything(tmp_config):
    client = FakeElevationClient()
    profile_horizons(_candidates(2), tmp_config, client)
    forced = FakeElevationClient()
    profile_horizons(_candidates(2), tmp_config, forced, force=True)
    assert forced.points_requested > 0


# ---------------------------------------------------------------------------
# Water harvest
# ---------------------------------------------------------------------------

def test_water_harvest_is_cached(tmp_config, tmp_path):
    session = FakeSession([FakeResponse(200, {"elements": [
        {"center": {"lat": 51.45, "lon": -0.90}}]})])
    candidates = _candidates(2)
    first = harvest_water(candidates, tmp_config, force=True, session=session)
    assert len(first["elements"]) == 1

    second_session = FakeSession([])
    second = harvest_water(candidates, tmp_config, session=second_session)
    assert second["elements"] == first["elements"]
    assert second_session.calls == []


def test_water_harvest_reports_failure(tmp_config):
    from sunset_rider.weather import WeatherUnavailable

    session = FakeSession([FakeResponse(500)])
    with pytest.raises(WeatherUnavailable, match="500"):
        harvest_water(_candidates(1), tmp_config, force=True, session=session)


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def test_viewpoints_round_trip(tmp_config, tmp_path):
    records = [{
        "id": "node_1", "name": "Walbury Hill", "kind": "peak",
        "latitude": 51.3525, "longitude": -1.465,
        "horizon_profile": {b: 0.5 for b in range(0, 360, 10)},
        "open_arc": [280, 330], "horizon_openness": 0.93,
        "elevation_prominence": 0.62, "gate_closes": None,
        "spot_score": 82.0, "distance_km": 36.7, "minutes_one_way": 59.6,
    }]
    path = write_viewpoints(records, tmp_config)
    assert path.is_file()

    loaded = load_viewpoints(tmp_config)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "Walbury Hill"
    assert len(loaded[0]["horizon_profile"]) == 36
    assert loaded[0]["gate_closes"] is None


def test_generated_file_warns_against_hand_editing(tmp_config, tmp_path):
    path = write_viewpoints([], tmp_config)
    text = path.read_text(encoding="utf-8")
    assert "Do not hand-edit" in text
    assert "gate_closes" in text


def test_loading_a_missing_file_explains_how_to_generate_it(tmp_config):
    with pytest.raises(FileNotFoundError, match="discovery"):
        load_viewpoints(tmp_config)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_refuses_a_suspiciously_small_harvest(tmp_config, tmp_path):
    """Section 13: stop and ask if Overpass returns fewer than 40 raw candidates."""
    cache = tmp_path / tmp_config.discovery.raw_osm_cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"elements": [
        _element(i, 51.5, -1.2, {"natural": "peak"}) for i in range(5)]}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="fewer than"):
        run(tmp_config, stage="harvest")


def test_run_harvest_stage_stops_before_profiling(tmp_config, tmp_path):
    cache = tmp_path / tmp_config.discovery.raw_osm_cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"elements": [
        _element(i, 51.4 + i * 0.02, -1.0, {"natural": "peak", "name": f"P{i}"})
        for i in range(50)]}), encoding="utf-8")
    assert run(tmp_config, stage="harvest") == []
    assert not (tmp_path / tmp_config.discovery.output_viewpoints).is_file()
