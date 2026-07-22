"""Weather client tests.

Recorded fixtures only — no test in this file touches the network. The fixtures
in tests/fixtures/ were captured from the live API on 2026-07-21.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import requests

from sunset_rider.weather import (
    EnsembleForecast,
    OpenMeteoClient,
    PointForecast,
    WeatherUnavailable,
    _interpolate,
    load_ensemble_fixture,
    load_point_fixture,
    summarise_spread,
)

UTC = dt.timezone.utc

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


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", raise_json=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload) if payload is not None else text
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not JSON")
        return self._payload


class FakeSession:
    """Returns a scripted sequence of responses and records the calls made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}})
        if not self._responses:
            raise AssertionError("FakeSession ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Backoff is real in production but must not slow the suite."""
    monkeypatch.setattr("sunset_rider.weather.time.sleep", lambda _s: None)


@pytest.fixture()
def forecast_payload(fixtures_dir):
    return json.loads((fixtures_dir / "forecast_multi.json").read_text(encoding="utf-8"))


@pytest.fixture()
def ensemble_payload(fixtures_dir):
    return json.loads((fixtures_dir / "ensemble_ecmwf.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Parsing recorded fixtures
# ---------------------------------------------------------------------------

def test_forecast_fixture_parses_all_requested_variables(fixtures_dir):
    points = load_point_fixture(str(fixtures_dir / "forecast_multi.json"), FORECAST_VARS)
    assert len(points) == 2
    for point in points:
        assert len(point.times) > 24
        for name in FORECAST_VARS:
            assert name in point.variables


def test_forecast_times_are_utc_aware(fixtures_dir):
    point = load_point_fixture(str(fixtures_dir / "forecast_multi.json"), FORECAST_VARS)[0]
    assert all(t.tzinfo is not None for t in point.times)
    assert point.times[0].utcoffset() == dt.timedelta(0)


def test_forecast_hours_are_one_hour_apart(fixtures_dir):
    point = load_point_fixture(str(fixtures_dir / "forecast_multi.json"), FORECAST_VARS)[0]
    gaps = {(b - a).total_seconds() for a, b in zip(point.times, point.times[1:])}
    assert gaps == {3600.0}


def test_ensemble_fixture_has_the_expected_member_count(fixtures_dir):
    """ECMWF IFS: 50 perturbed members plus the control run."""
    ens = load_ensemble_fixture(str(fixtures_dir / "ensemble_ecmwf.json"), ENSEMBLE_VARS)[0]
    assert ens.member_count == 51
    assert len(ens.member_series("cloud_cover_mid")) == 51


def test_ensemble_carries_split_cloud_layers(fixtures_dir):
    """The reason we use ECMWF and not ICON-EU/GFS: those return total cloud only."""
    ens = load_ensemble_fixture(str(fixtures_dir / "ensemble_ecmwf.json"), ENSEMBLE_VARS)[0]
    for layer in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
        series = ens.member_series(layer)
        assert series, f"{layer} missing entirely"
        assert any(v is not None for v in series[0]), f"{layer} is all None"


def test_ensemble_drops_all_none_series():
    """visibility is accepted by the ensemble API but never populated.

    An all-None series must be dropped rather than counted as a member, otherwise
    the member count and therefore the IQR would be silently wrong.
    """
    payload = {
        "latitude": 51.4, "longitude": -0.9,
        "hourly": {
            "time": ["2025-06-18T00:00", "2025-06-18T01:00"],
            "visibility": [None, None],
            "visibility_member01": [None, None],
            "cloud_cover": [10.0, 20.0],
            "cloud_cover_member01": [12.0, 22.0],
        },
    }
    from sunset_rider.weather import _ensemble_from_payload
    ens = _ensemble_from_payload(payload, ["visibility", "cloud_cover"])
    assert ens.member_series("visibility") == []
    assert len(ens.member_series("cloud_cover")) == 2


# ---------------------------------------------------------------------------
# Interpolation — the spec forbids rounding to the nearest hour
# ---------------------------------------------------------------------------

def _series(values, start=dt.datetime(2025, 6, 18, 18, 0, tzinfo=UTC)):
    times = [start + dt.timedelta(hours=i) for i in range(len(values))]
    return times, values


def test_interpolation_returns_the_midpoint_not_the_nearest_hour():
    times, values = _series([0.0, 100.0])
    half_past = times[0] + dt.timedelta(minutes=30)
    assert _interpolate(times, values, half_past) == pytest.approx(50.0)


def test_interpolation_is_linear_at_arbitrary_offsets():
    times, values = _series([20.0, 80.0])
    at_24_min = times[0] + dt.timedelta(minutes=24)
    assert _interpolate(times, values, at_24_min) == pytest.approx(44.0)


def test_interpolation_returns_exact_value_on_the_hour():
    times, values = _series([5.0, 9.0, 13.0])
    assert _interpolate(times, values, times[1]) == pytest.approx(9.0)


def test_interpolation_returns_none_outside_the_series():
    times, values = _series([1.0, 2.0])
    assert _interpolate(times, values, times[0] - dt.timedelta(hours=1)) is None
    assert _interpolate(times, values, times[-1] + dt.timedelta(hours=1)) is None


def test_interpolation_returns_none_when_a_bracket_is_missing():
    """Better an admitted gap than an invented number."""
    times, values = _series([10.0, None, 30.0])
    assert _interpolate(times, values, times[0] + dt.timedelta(minutes=30)) is None


def test_interpolation_of_empty_series_is_none():
    assert _interpolate([], [], dt.datetime(2025, 6, 18, tzinfo=UTC)) is None


def test_point_forecast_value_at_interpolates():
    times, values = _series([0.0, 60.0])
    point = PointForecast(51.0, -1.0, 50.0, times, {"cloud_cover": values})
    at = times[0] + dt.timedelta(minutes=45)
    assert point.value_at("cloud_cover", at) == pytest.approx(45.0)


def test_point_forecast_unknown_variable_is_none():
    times, values = _series([1.0, 2.0])
    point = PointForecast(51.0, -1.0, 50.0, times, {"cloud_cover": values})
    assert point.value_at("nonexistent", times[0]) is None


def test_window_aggregates_ignore_missing_values():
    times, values = _series([10.0, None, 40.0, 20.0])
    point = PointForecast(51.0, -1.0, 50.0, times, {"wind_gusts_10m": values})
    assert point.max_between("wind_gusts_10m", times[0], times[-1]) == 40.0
    assert point.min_between("wind_gusts_10m", times[0], times[-1]) == 10.0
    assert point.sum_between("wind_gusts_10m", times[0], times[-1]) == 70.0


def test_window_aggregates_return_none_when_empty():
    times, values = _series([1.0, 2.0])
    point = PointForecast(51.0, -1.0, 50.0, times, {"cape": values})
    far = times[-1] + dt.timedelta(days=5)
    assert point.max_between("cape", far, far + dt.timedelta(hours=1)) is None


# ---------------------------------------------------------------------------
# Ensemble spread
# ---------------------------------------------------------------------------

def test_summarise_spread_computes_median_and_iqr():
    stats = summarise_spread([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert stats["median"] == 5.0
    assert stats["q1"] == 3.0
    assert stats["q3"] == 7.0
    assert stats["iqr"] == 4.0
    assert stats["n"] == 9.0


def test_summarise_spread_handles_empty_and_single():
    assert summarise_spread([])["n"] == 0
    single = summarise_spread([42.0])
    assert single["median"] == 42.0 and single["iqr"] == 0.0


def test_tight_spread_has_smaller_iqr_than_wide_spread():
    """A tight IQR at 72h is the signal that an evening is worth holding."""
    tight = summarise_spread([50, 51, 52, 53, 54])
    wide = summarise_spread([10, 30, 52, 75, 95])
    assert tight["iqr"] < wide["iqr"]
    assert tight["median"] == wide["median"]


def test_ensemble_values_at_returns_one_value_per_member(fixtures_dir):
    ens = load_ensemble_fixture(str(fixtures_dir / "ensemble_ecmwf.json"), ENSEMBLE_VARS)[0]
    moment = ens.times[10] + dt.timedelta(minutes=20)
    values = ens.values_at("cloud_cover", moment)
    assert len(values) == 51


def test_ensemble_member_count_is_zero_when_empty():
    assert EnsembleForecast(0.0, 0.0, [], {}).member_count == 0


# ---------------------------------------------------------------------------
# Transport: retries, backoff, graceful degradation
# ---------------------------------------------------------------------------

def test_retries_then_succeeds(config, forecast_payload):
    session = FakeSession([
        FakeResponse(500, {"error": True, "reason": "boom"}),
        FakeResponse(200, forecast_payload),
    ])
    client = OpenMeteoClient(config, session=session)
    points = client.forecast([(51.44, -0.96), (51.78, -1.73)],
                             start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 20))
    assert len(points) == 2
    assert len(session.calls) == 2


def test_persistent_500_raises_with_a_stated_reason(config, fixtures_dir):
    """The malformed-500 fixture: three attempts, then a clear message.

    Silence is the worst failure mode, so the reason must be human-readable and
    must survive to the Telegram layer.
    """
    body = json.loads((fixtures_dir / "malformed_500.json").read_text(encoding="utf-8"))
    session = FakeSession([FakeResponse(500, body) for _ in range(3)])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable) as excinfo:
        client.forecast([(51.44, -0.96)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 20))
    assert "500" in str(excinfo.value)
    assert excinfo.value.source == "forecast"
    assert len(session.calls) == int(config.weather.retries)


def test_client_error_is_not_retried(config):
    """A 4xx means our parameters are wrong; retrying just hammers the server."""
    session = FakeSession([FakeResponse(400, {"reason": "cannot initialize Variable"})])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="cannot initialize"):
        client.forecast([(51.44, -0.96)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 20))
    assert len(session.calls) == 1


def test_invalid_json_is_reported_clearly(config):
    session = FakeSession([FakeResponse(200, None, raise_json=True) for _ in range(3)])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="not valid JSON"):
        client.forecast([(51.44, -0.96)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 20))


def test_network_error_is_retried_then_reported(config):
    session = FakeSession([requests.ConnectionError("dns"), requests.ConnectionError("dns"),
                           requests.ConnectionError("dns")])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="network error"):
        client.forecast([(51.44, -0.96)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 20))
    assert len(session.calls) == 3


def test_payload_without_hourly_block_degrades_gracefully(config, fixtures_dir):
    body = json.loads((fixtures_dir / "forecast_missing_hourly.json").read_text(encoding="utf-8"))
    session = FakeSession([FakeResponse(200, body)])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="no hourly block"):
        client.forecast([(51.44, -0.96)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 20))


def test_absent_variable_becomes_all_none_rather_than_crashing(config):
    """One missing variable should degrade one scoring term, not the whole run."""
    payload = {
        "latitude": 51.4, "longitude": -0.9,
        "hourly": {"time": ["2025-06-18T00:00", "2025-06-18T01:00"], "cloud_cover": [10, 20]},
    }
    session = FakeSession([FakeResponse(200, payload)])
    client = OpenMeteoClient(config, session=session)
    point = client.forecast([(51.4, -0.9)], start_date=dt.date(2025, 6, 18),
                            end_date=dt.date(2025, 6, 18),
                            variables=["cloud_cover", "visibility"])[0]
    assert point.variables["visibility"] == [None, None]
    assert point.variables["cloud_cover"] == [10.0, 20.0]


def test_location_count_mismatch_is_detected(config):
    payload = {"latitude": 51.4, "longitude": -0.9,
               "hourly": {"time": ["2025-06-18T00:00"], "cloud_cover": [10]}}
    session = FakeSession([FakeResponse(200, payload)])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="expected 2 locations"):
        client.forecast([(51.4, -0.9), (51.9, -1.9)], start_date=dt.date(2025, 6, 18),
                        end_date=dt.date(2025, 6, 18), variables=["cloud_cover"])


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_elevation_batches_at_the_hundred_coordinate_cap(config):
    """VERIFIED: the endpoint rejects >100 pairs. 250 points must become 3 calls."""
    coords = [(51.0 + i * 0.001, -1.0) for i in range(250)]
    session = FakeSession([
        FakeResponse(200, {"elevation": [100.0] * 100}),
        FakeResponse(200, {"elevation": [100.0] * 100}),
        FakeResponse(200, {"elevation": [100.0] * 50}),
    ])
    client = OpenMeteoClient(config, session=session)
    out = client.elevation(coords)
    assert len(out) == 250
    assert len(session.calls) == 3
    assert len(session.calls[0]["params"]["latitude"].split(",")) == 100
    assert len(session.calls[2]["params"]["latitude"].split(",")) == 50


def test_elevation_count_mismatch_is_detected(config):
    session = FakeSession([FakeResponse(200, {"elevation": [1.0]})])
    client = OpenMeteoClient(config, session=session)
    with pytest.raises(WeatherUnavailable, match="expected 2 elevations"):
        client.elevation([(51.0, -1.0), (51.1, -1.1)])


def test_empty_coordinate_lists_make_no_calls(config):
    session = FakeSession([])
    client = OpenMeteoClient(config, session=session)
    assert client.elevation([]) == []
    assert client.forecast([], start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 18)) == []
    assert client.ensemble([], start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 18)) == []
    assert session.calls == []


def test_forecast_batches_multiple_coordinates_into_one_call(config, forecast_payload):
    """Never loop one request per point."""
    session = FakeSession([FakeResponse(200, forecast_payload)])
    client = OpenMeteoClient(config, session=session)
    client.forecast([(51.44, -0.96), (51.78, -1.73)],
                    start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 20))
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["latitude"] == "51.440000,51.780000"


# ---------------------------------------------------------------------------
# The two API traps we verified
# ---------------------------------------------------------------------------

def test_all_requests_ask_for_utc(config, forecast_payload, ensemble_payload):
    """Open-Meteo reports utc_offset_seconds=3600 for Europe/London in December.

    Requesting UTC and localising in Python is the only safe option; this test
    fails if anyone reintroduces a named timezone.
    """
    session = FakeSession([FakeResponse(200, forecast_payload),
                           FakeResponse(200, ensemble_payload)])
    client = OpenMeteoClient(config, session=session)
    client.forecast([(51.44, -0.96), (51.78, -1.73)],
                    start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 20))
    client.ensemble([(51.44, -0.96)],
                    start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 19))
    for call in session.calls:
        assert call["params"]["timezone"] == "UTC"


def test_ensemble_uses_the_dedicated_host_and_ecmwf_model(config, ensemble_payload):
    """api.open-meteo.com/v1/ensemble is a 404, and only ECMWF has cloud layers."""
    session = FakeSession([FakeResponse(200, ensemble_payload)])
    client = OpenMeteoClient(config, session=session)
    client.ensemble([(51.44, -0.96)],
                    start_date=dt.date(2025, 6, 18), end_date=dt.date(2025, 6, 19))
    call = session.calls[0]
    assert call["url"].startswith("https://ensemble-api.open-meteo.com")
    assert call["params"]["models"] == "ecmwf_ifs025"
