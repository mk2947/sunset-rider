"""Open-Meteo clients: deterministic forecast, ensemble, and elevation.

Everything here was verified against the live API on 2026-07-21. Three findings
are baked into this module and should not be "simplified" away:

1. The ensemble API is on its own host, ``ensemble-api.open-meteo.com``.
   ``api.open-meteo.com/v1/ensemble`` returns 404.
2. We always request UTC. Asking for ``timezone=Europe/London`` makes Open-Meteo
   return ``utc_offset_seconds=3600`` even in December, which would silently shift
   every hourly value an hour away from the sunset it is supposed to bracket.
3. Only the ECMWF ensembles carry ``cloud_cover_low/mid/high``. ICON-EU and GFS
   return total cloud only, and would quietly produce all-None mid/high series.
   ``visibility`` and ``precipitation_probability`` are null on every ensemble
   model, so ensemble PoP is derived from member agreement instead.

Requests are always batched with comma-separated coordinates. Never loop one
request per point.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Sequence

import requests

from .config import Config

log = logging.getLogger(__name__)

UTC = dt.timezone.utc


class WeatherUnavailable(RuntimeError):
    """Raised when forecast data cannot be obtained.

    Carries a human-readable reason because silence is the worst failure mode:
    the rider must never be left guessing whether it is a dull sky or a dead script.
    """

    def __init__(self, reason: str, *, source: str = "", cause: Exception | None = None):
        self.reason = reason
        self.source = source
        self.cause = cause
        super().__init__(f"{source + ': ' if source else ''}{reason}")


def _parse_times(raw: Sequence[str]) -> list[dt.datetime]:
    """Parse Open-Meteo iso8601 timestamps as UTC.

    Open-Meteo omits the offset suffix, and because we always request UTC the
    bare timestamps are UTC by construction.
    """
    out: list[dt.datetime] = []
    for value in raw:
        parsed = dt.datetime.fromisoformat(value)
        out.append(parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC))
    return out


def _interpolate(times: Sequence[dt.datetime], values: Sequence[float | None],
                 moment: dt.datetime) -> float | None:
    """Linear interpolation between the two samples bracketing ``moment``.

    Returns None if the moment is outside the series or either bracketing value
    is missing — a guessed value is worse than an admitted gap.
    """
    if not times:
        return None
    target = moment.astimezone(UTC)
    if target <= times[0]:
        return values[0] if target == times[0] else None
    if target >= times[-1]:
        return values[-1] if target == times[-1] else None

    lo = 0
    hi = len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= target:
            lo = mid
        else:
            hi = mid

    left, right = values[lo], values[hi]
    if left is None or right is None:
        return None
    span = (times[hi] - times[lo]).total_seconds()
    if span <= 0:
        return left
    frac = (target - times[lo]).total_seconds() / span
    return left + (right - left) * frac


@dataclass
class PointForecast:
    """Deterministic hourly forecast for one coordinate."""

    latitude: float
    longitude: float
    elevation: float | None
    times: list[dt.datetime]
    variables: dict[str, list[float | None]]

    def value_at(self, name: str, moment: dt.datetime) -> float | None:
        """Interpolated value at an arbitrary moment.

        The spec is explicit: interpolate between the two hours bracketing sunset,
        do not round to the nearest hour.
        """
        series = self.variables.get(name)
        if series is None:
            return None
        return _interpolate(self.times, series, moment)

    def _window(self, name: str, start: dt.datetime, end: dt.datetime) -> list[float]:
        series = self.variables.get(name)
        if series is None:
            return []
        lo, hi = start.astimezone(UTC), end.astimezone(UTC)
        return [v for t, v in zip(self.times, series) if lo <= t <= hi and v is not None]

    def max_between(self, name: str, start: dt.datetime, end: dt.datetime) -> float | None:
        values = self._window(name, start, end)
        return max(values) if values else None

    def min_between(self, name: str, start: dt.datetime, end: dt.datetime) -> float | None:
        values = self._window(name, start, end)
        return min(values) if values else None

    def sum_between(self, name: str, start: dt.datetime, end: dt.datetime) -> float | None:
        values = self._window(name, start, end)
        return sum(values) if values else None


@dataclass
class EnsembleForecast:
    """Per-member hourly forecast for one coordinate."""

    latitude: float
    longitude: float
    times: list[dt.datetime]
    # variable -> list of member series
    members: dict[str, list[list[float | None]]] = field(default_factory=dict)

    @property
    def member_count(self) -> int:
        for series in self.members.values():
            return len(series)
        return 0

    def values_at(self, name: str, moment: dt.datetime) -> list[float]:
        """Every member's interpolated value at ``moment``, missing members dropped."""
        out: list[float] = []
        for series in self.members.get(name, []):
            value = _interpolate(self.times, series, moment)
            if value is not None:
                out.append(value)
        return out

    def member_series(self, name: str) -> list[list[float | None]]:
        return self.members.get(name, [])


def summarise_spread(values: Sequence[float]) -> dict[str, float]:
    """Median and interquartile range of a set of member values.

    A tight IQR at 72 hours is the real signal that an evening is worth holding,
    so the IQR is reported as a first-class output, never hidden behind a mean.
    """
    if not values:
        return {"median": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0, "n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def _quantile(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    q1, q3 = _quantile(0.25), _quantile(0.75)
    return {
        "median": median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "n": float(n),
    }


class OpenMeteoClient:
    """Batched, retrying client for the three Open-Meteo endpoints we use."""

    def __init__(self, config: Config, session: requests.Session | None = None) -> None:
        self._config = config
        self._w = config.weather
        self._session = session or requests.Session()
        # Sliding window of (timestamp, locations) for self-throttling.
        self._recent: list[tuple[float, int]] = []

    # -- rate limiting ------------------------------------------------------

    def _throttle(self, locations: int) -> None:
        """Stay under the measured 600-locations-per-minute cap.

        MEASURED: Open-Meteo counts locations, not requests. Six 100-coordinate
        requests in a second is fine; the seventh returns 429. Self-throttling is
        cheaper and more polite than absorbing 429s.
        """
        limit = int(self._w.locations_per_minute)
        while True:
            now = time.monotonic()
            self._recent = [(t, n) for t, n in self._recent if now - t < 60.0]
            used = sum(n for _, n in self._recent)
            if used + locations <= limit or not self._recent:
                self._recent.append((now, locations))
                return
            sleep_for = 60.0 - (now - self._recent[0][0]) + 0.5
            log.info("throttling: %d locations used in the last minute, waiting %.1fs",
                     used, sleep_for)
            time.sleep(max(sleep_for, 1.0))

    # -- transport ----------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any], *, source: str,
             locations: int = 1) -> Any:
        """GET with throttling, retries and exponential backoff, then a clear failure."""
        retries = int(self._w.retries)
        delay = float(self._w.backoff_base_seconds)
        last: Exception | None = None
        rate_waits = 0

        attempt = 0
        while attempt < retries:
            attempt += 1
            self._throttle(locations)
            try:
                response = self._session.get(
                    url, params=params, timeout=float(self._w.timeout_seconds)
                )
                if response.status_code == 429:
                    # Not our bug and it will clear on its own, so wait rather than
                    # failing the whole discovery run.
                    rate_waits += 1
                    if rate_waits > int(self._w.rate_limit_max_waits):
                        raise WeatherUnavailable(
                            "rate limited repeatedly; giving up", source=source
                        )
                    wait = float(self._w.rate_limit_wait_seconds)
                    log.warning("%s rate limited (429); waiting %.0fs", source, wait)
                    time.sleep(wait)
                    self._recent.clear()
                    attempt -= 1  # a 429 is not a failed attempt, just a pause
                    continue
                if response.status_code >= 500:
                    raise WeatherUnavailable(
                        f"server returned HTTP {response.status_code}", source=source
                    )
                if response.status_code >= 400:
                    # 4xx is our bug (bad parameter) and will not fix itself on retry.
                    detail = ""
                    try:
                        detail = response.json().get("reason", "")
                    except (ValueError, AttributeError):
                        detail = response.text[:200]
                    raise WeatherUnavailable(
                        f"HTTP {response.status_code} — {detail}", source=source
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    raise WeatherUnavailable(
                        "response was not valid JSON", source=source, cause=exc
                    ) from exc

            except WeatherUnavailable as exc:
                last = exc
                # Do not burn retries on a request that is malformed by construction.
                if "HTTP 4" in str(exc):
                    raise
            except requests.RequestException as exc:
                last = WeatherUnavailable(
                    f"network error: {type(exc).__name__}", source=source, cause=exc
                )

            if attempt < retries:
                log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                            source, attempt, retries, last, delay)
                time.sleep(delay)
                delay *= float(self._w.backoff_factor)
            else:
                break

        raise last if last else WeatherUnavailable("unknown failure", source=source)

    def _endpoint_for(self, end_date: dt.date) -> str:
        """Forecast endpoint for recent dates, archive for older ones.

        The forecast API only reaches about 92 days into the past. Routing older
        requests to the archive lets --date work for any historical evening, which
        is what makes calibration backfill (and testing December in July) possible.
        """
        cutoff = dt.date.today() - dt.timedelta(days=int(self._w.archive_cutoff_days))
        if end_date < cutoff:
            log.info("using the archive endpoint for %s", end_date)
            return self._w.archive_url
        return self._w.forecast_url

    @staticmethod
    def _as_list(payload: Any) -> list[dict]:
        """Open-Meteo returns a bare object for one coordinate, a list for many."""
        if isinstance(payload, list):
            return payload
        return [payload]

    # -- forecast -----------------------------------------------------------

    def forecast(self, coordinates: Sequence[tuple[float, float]], *,
                 start_date: dt.date, end_date: dt.date,
                 variables: Sequence[str] | None = None) -> list[PointForecast]:
        """Hourly deterministic forecast for many coordinates in as few calls as possible."""
        if not coordinates:
            return []
        names = list(variables) if variables else list(self._w.forecast_hourly_variables)
        chunk_size = int(self._w.forecast_max_coords_per_request)
        results: list[PointForecast] = []
        url = self._endpoint_for(end_date)

        for chunk in _chunks(coordinates, chunk_size):
            payload = self._get(
                url,
                {
                    "latitude": ",".join(f"{lat:.6f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.6f}" for _, lon in chunk),
                    "hourly": ",".join(names),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "timezone": self._w.request_timezone,
                },
                source="forecast",
            )
            for entry in self._as_list(payload):
                results.append(_point_from_payload(entry, names))

        if len(results) != len(coordinates):
            raise WeatherUnavailable(
                f"expected {len(coordinates)} locations, got {len(results)}",
                source="forecast",
            )
        return results

    # -- ensemble -----------------------------------------------------------

    def ensemble(self, coordinates: Sequence[tuple[float, float]], *,
                 start_date: dt.date, end_date: dt.date,
                 variables: Sequence[str] | None = None,
                 model: str | None = None) -> list[EnsembleForecast]:
        """Per-member ensemble forecast — the basis of ``plan`` mode's spread."""
        if not coordinates:
            return []
        names = list(variables) if variables else list(self._w.ensemble_hourly_variables)
        chunk_size = int(self._w.forecast_max_coords_per_request)
        results: list[EnsembleForecast] = []

        for chunk in _chunks(coordinates, chunk_size):
            payload = self._get(
                self._w.ensemble_url,
                {
                    "latitude": ",".join(f"{lat:.6f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.6f}" for _, lon in chunk),
                    "hourly": ",".join(names),
                    "models": model or self._w.ensemble_model,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "timezone": self._w.request_timezone,
                },
                source="ensemble",
            )
            for entry in self._as_list(payload):
                results.append(_ensemble_from_payload(entry, names))

        if len(results) != len(coordinates):
            raise WeatherUnavailable(
                f"expected {len(coordinates)} locations, got {len(results)}",
                source="ensemble",
            )
        return results

    # -- elevation ----------------------------------------------------------

    def elevation(self, coordinates: Sequence[tuple[float, float]]) -> list[float]:
        """Terrain elevation for many points.

        VERIFIED: the endpoint rejects more than 100 coordinate pairs per request,
        so batches are capped at the configured limit.
        """
        if not coordinates:
            return []
        cap = int(self._w.elevation_max_coords_per_request)
        out: list[float] = []

        for chunk in _chunks(coordinates, cap):
            payload = self._get(
                self._w.elevation_url,
                {
                    "latitude": ",".join(f"{lat:.6f}" for lat, _ in chunk),
                    "longitude": ",".join(f"{lon:.6f}" for _, lon in chunk),
                },
                source="elevation",
                locations=len(chunk),
            )
            values = payload.get("elevation") if isinstance(payload, dict) else None
            if not isinstance(values, list) or len(values) != len(chunk):
                raise WeatherUnavailable(
                    f"expected {len(chunk)} elevations, got "
                    f"{len(values) if isinstance(values, list) else 'none'}",
                    source="elevation",
                )
            out.extend(float(v) for v in values)
        return out


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _point_from_payload(entry: dict, names: Sequence[str]) -> PointForecast:
    hourly = entry.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherUnavailable("payload had no hourly block", source="forecast")
    times = _parse_times(hourly["time"])
    variables: dict[str, list[float | None]] = {}
    for name in names:
        series = hourly.get(name)
        if not isinstance(series, list):
            # Requested but absent: record an all-None series rather than crashing,
            # so a single missing variable degrades one term instead of the whole run.
            log.debug("variable %s absent from forecast payload", name)
            variables[name] = [None] * len(times)
        else:
            variables[name] = [None if v is None else float(v) for v in series]
    return PointForecast(
        latitude=float(entry.get("latitude", 0.0)),
        longitude=float(entry.get("longitude", 0.0)),
        elevation=entry.get("elevation"),
        times=times,
        variables=variables,
    )


def _ensemble_from_payload(entry: dict, names: Sequence[str]) -> EnsembleForecast:
    hourly = entry.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherUnavailable("payload had no hourly block", source="ensemble")
    times = _parse_times(hourly["time"])
    members: dict[str, list[list[float | None]]] = {}

    for name in names:
        series_list: list[list[float | None]] = []
        # The control run is unsuffixed; perturbed members are <name>_memberNN.
        for key in sorted(hourly):
            if key == name or key.startswith(f"{name}_member"):
                raw = hourly[key]
                if not isinstance(raw, list):
                    continue
                converted = [None if v is None else float(v) for v in raw]
                # Skip all-None series: some variables are accepted but never populated.
                if any(v is not None for v in converted):
                    series_list.append(converted)
        members[name] = series_list

    return EnsembleForecast(
        latitude=float(entry.get("latitude", 0.0)),
        longitude=float(entry.get("longitude", 0.0)),
        times=times,
        members=members,
    )


def load_point_fixture(path: str, names: Sequence[str]) -> list[PointForecast]:
    """Build PointForecasts from a recorded JSON fixture (tests only)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload if isinstance(payload, list) else [payload]
    return [_point_from_payload(e, names) for e in entries]


def load_ensemble_fixture(path: str, names: Sequence[str]) -> list[EnsembleForecast]:
    """Build EnsembleForecasts from a recorded JSON fixture (tests only)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload if isinstance(payload, list) else [payload]
    return [_ensemble_from_payload(e, names) for e in entries]
