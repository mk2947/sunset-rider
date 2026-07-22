"""Orchestration: fetch, score, rank.

Separated from ``main`` so the whole pipeline is testable without a CLI, and from
``message`` so rendering never triggers a fetch.

Request discipline: every Open-Meteo call here is batched across all coordinates.
The corridor needs four extra sample points per viewpoint, so a naive
one-request-per-point implementation would make hundreds of calls per run and be
rate limited within a minute.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Sequence

from .config import Config
from .discovery import load_viewpoints
from .gating import evaluate_blockers, leave_by, returns_after_dark_minutes
from .geo import destination_point
from .models import EveningOutlook, RunResult, SpotForecast
from .scoring import (
    SkyInputs,
    combine_output,
    corridor_clearness,
    max_radius_km,
    score_output,
    score_ride,
    score_sky,
    score_worth_it,
)
from .solar import SolarCalculator, SolarEvents, UTC
from .weather import (
    EnsembleForecast,
    OpenMeteoClient,
    PointForecast,
    WeatherUnavailable,
    summarise_spread,
)

log = logging.getLogger(__name__)


def good_threshold(config: Config) -> float:
    """The score that counts as "good" — read from the band table, not hardcoded."""
    for entry in config.bands:
        if entry.label == "GOOD":
            return float(entry.min)
    raise ValueError("no GOOD band defined in config.bands")


def corridor_points(latitude: float, longitude: float, bearing: float,
                    config: Config) -> list[tuple[float, float]]:
    """The four sun-to-sky sample points for one viewpoint on one evening."""
    radius = float(config.discovery.earth_radius_km)
    return [
        destination_point(latitude, longitude, bearing, float(distance), radius)
        for distance in config.corridor.distances_km
    ]


def _series_at(point: PointForecast, name: str, moment: dt.datetime,
               default: float = 0.0) -> float:
    value = point.value_at(name, moment)
    return default if value is None else value


def _optional_at(point: PointForecast, name: str, moment: dt.datetime) -> float | None:
    return point.value_at(name, moment)


def build_sky_inputs(viewpoint_forecast: PointForecast,
                     corridor_forecasts: Sequence[PointForecast],
                     events: SolarEvents) -> SkyInputs:
    """Assemble scoring inputs, interpolated to the exact moment of sunset."""
    moment = events.sunset
    after = [
        _series_at(viewpoint_forecast, "cloud_cover", moment + dt.timedelta(hours=h))
        for h in (1, 2)
    ]
    return SkyInputs(
        total_cc=_series_at(viewpoint_forecast, "cloud_cover", moment),
        cloud_cover_low=_series_at(viewpoint_forecast, "cloud_cover_low", moment),
        cloud_cover_mid=_series_at(viewpoint_forecast, "cloud_cover_mid", moment),
        cloud_cover_high=_series_at(viewpoint_forecast, "cloud_cover_high", moment),
        corridor_low=[_series_at(c, "cloud_cover_low", moment) for c in corridor_forecasts],
        visibility_m=_optional_at(viewpoint_forecast, "visibility", moment),
        relative_humidity=_series_at(viewpoint_forecast, "relative_humidity_2m", moment, 60.0),
        precipitation_probability=_series_at(
            viewpoint_forecast, "precipitation_probability", moment),
        cape=_series_at(viewpoint_forecast, "cape", moment),
        total_cc_after=after,
    )


# ---------------------------------------------------------------------------
# Deterministic run (confirm / go)
# ---------------------------------------------------------------------------

def select_viewpoints(viewpoints: Sequence[dict], radius_km: float) -> list[dict]:
    """Gate the candidate set by tonight's dynamic radius."""
    return [v for v in viewpoints if float(v.get("distance_km", 1e9)) <= radius_km]


def run_deterministic(config: Config, target_date: dt.date, mode: str,
                      client: OpenMeteoClient,
                      viewpoints: Sequence[dict] | None = None) -> RunResult:
    """Score every in-range viewpoint for one evening."""
    solar = SolarCalculator(config)
    events = solar.events(target_date)
    all_viewpoints = list(viewpoints if viewpoints is not None else load_viewpoints(config))
    if not all_viewpoints:
        raise WeatherUnavailable("no viewpoints available", source="viewpoints")

    result = RunResult(mode=mode, target_date=target_date)

    # Step 1: one cheap regional read at the home centroid to set tonight's radius.
    home = (config.home.latitude, config.home.longitude)
    regional_coords = [home] + corridor_points(*home, events.sun_bearing, config)
    regional = client.forecast(regional_coords, start_date=target_date,
                               end_date=target_date + dt.timedelta(days=1))
    regional_inputs = build_sky_inputs(regional[0], regional[1:], events)
    regional_result = score_sky(regional_inputs, config)
    result.regional_sky = regional_result.sky
    result.max_radius_km = max_radius_km(regional_result.sky, config)
    log.info("regional sky %.1f -> radius %.1f km",
             result.regional_sky, result.max_radius_km)

    in_range = select_viewpoints(all_viewpoints, result.max_radius_km)
    for viewpoint in all_viewpoints:
        if viewpoint not in in_range:
            result.excluded.append((
                str(viewpoint.get("name")),
                f"{float(viewpoint.get('distance_km', 0)):.0f} km — beyond tonight's "
                f"{result.max_radius_km:.0f} km radius",
            ))
    if not in_range:
        return result

    # Step 2: one batched request covering every viewpoint and every corridor point.
    coords: list[tuple[float, float]] = []
    spans: list[tuple[int, int]] = []
    for viewpoint in in_range:
        start = len(coords)
        coords.append((float(viewpoint["latitude"]), float(viewpoint["longitude"])))
        coords.extend(corridor_points(float(viewpoint["latitude"]),
                                      float(viewpoint["longitude"]),
                                      events.sun_bearing, config))
        spans.append((start, len(coords)))

    forecasts = client.forecast(coords, start_date=target_date,
                                end_date=target_date + dt.timedelta(days=1))

    for viewpoint, (start, end) in zip(in_range, spans):
        block = forecasts[start:end]
        spot = _score_spot(viewpoint, block[0], block[1:], events, config)
        if spot.blocked:
            result.excluded.append((spot.name, "; ".join(spot.blockers.reasons)))
        result.spots.append(spot)

    result.spots.sort(key=lambda s: -s.worth_it)
    return result


def _score_spot(viewpoint: dict, viewpoint_forecast: PointForecast,
                corridor_forecasts: Sequence[PointForecast],
                events: SolarEvents, config: Config) -> SpotForecast:
    inputs = build_sky_inputs(viewpoint_forecast, corridor_forecasts, events)
    sky = score_sky(inputs, config)
    corridor = corridor_clearness(inputs.corridor_low, config)

    minutes_one_way = float(viewpoint.get("minutes_one_way", 0.0))
    depart = leave_by(events, minutes_one_way, config)
    home_again = events.civil_dusk + dt.timedelta(minutes=minutes_one_way)

    max_gust = viewpoint_forecast.max_between("wind_gusts_10m", depart, home_again) or 0.0
    max_precip_prob = viewpoint_forecast.max_between(
        "precipitation_probability", depart, home_again) or 0.0
    min_temp = viewpoint_forecast.min_between("apparent_temperature", depart, home_again)
    min_visibility = viewpoint_forecast.min_between("visibility", depart, home_again)
    precip_before = viewpoint_forecast.sum_between(
        "precipitation", depart - dt.timedelta(hours=3), depart) or 0.0
    wind_500 = viewpoint_forecast.value_at("wind_speed_500hPa", events.sunset)

    blockers = evaluate_blockers(
        max_gust_kmh=max_gust,
        min_apparent_temp_c=min_temp,
        max_precip_probability=max_precip_prob,
        min_visibility_m=min_visibility,
        gate_closes=viewpoint.get("gate_closes"),
        events=events,
        config=config,
    )

    output = score_output(
        sky=sky.sky, inputs=inputs, corridor=corridor,
        foreground_interest=float(viewpoint.get("foreground_interest", 0.0)),
        foreground_motion=float(viewpoint.get("foreground_motion", 0.0)),
        gusts_kmh=max_gust, wind_500hpa=wind_500,
        blue_hour_minutes=events.blue_hour_minutes, config=config,
    )

    ride = score_ride(
        max_gust_kmh=max_gust, max_precip_prob=max_precip_prob,
        apparent_temperature=min_temp if min_temp is not None else 15.0,
        precip_preceding_3h=precip_before,
        returns_after_dark_minutes=returns_after_dark_minutes(events, minutes_one_way),
        config=config,
    )

    worth, parts = score_worth_it(
        sky=sky.sky,
        output_score=combine_output(output.photo, output.video, config),
        ride=ride.score,
        spot=float(viewpoint.get("spot_score", 0.0)),
        sun_bearing=events.sun_bearing,
        open_arc=viewpoint.get("open_arc"),
        minutes_one_way=minutes_one_way,
        blocked=blockers.blocked,
        config=config,
    )

    if config.logging.debug_score_breakdown:
        log.debug("%s: worth=%.1f sky=%.1f(%s) photo=%.1f video=%.1f ride=%.1f %s %s",
                  viewpoint.get("name"), worth, sky.sky, sky.best_mode,
                  output.photo, output.video, ride.score, parts, sky.components)

    return SpotForecast(
        viewpoint=viewpoint, events=events, sky=sky, output=output, ride=ride,
        blockers=blockers, worth_it=worth, worth_it_parts=parts,
        leave_by=depart, minutes_one_way=minutes_one_way,
        max_gust_kmh=max_gust, wind_500hpa=wind_500,
    )


# ---------------------------------------------------------------------------
# Ensemble run (plan)
# ---------------------------------------------------------------------------

def run_plan(config: Config, start_date: dt.date, client: OpenMeteoClient) -> RunResult:
    """Rank the next five evenings against each other using ensemble spread.

    Deliberately does NOT produce a per-spot score. At 72 hours the honest question
    is "which evening should I protect", and the answer is a distribution.
    """
    solar = SolarCalculator(config)
    horizon = int(config.schedule.plan.horizon_days)
    dates = [start_date + dt.timedelta(days=offset) for offset in range(1, horizon + 1)]
    result = RunResult(mode="plan", target_date=start_date)

    home = (config.home.latitude, config.home.longitude)
    coords: list[tuple[float, float]] = [home]
    spans: dict[dt.date, tuple[int, int]] = {}
    events_by_date: dict[dt.date, SolarEvents] = {}
    for date in dates:
        events = solar.events(date)
        events_by_date[date] = events
        start = len(coords)
        coords.extend(corridor_points(*home, events.sun_bearing, config))
        spans[date] = (start, len(coords))

    ensembles = client.ensemble(coords, start_date=dates[0],
                                end_date=dates[-1] + dt.timedelta(days=1))
    home_ensemble = ensembles[0]
    threshold = good_threshold(config)
    wet_mm = float(config.weather.ensemble_precip_wet_threshold_mm)

    for date in dates:
        events = events_by_date[date]
        start, end = spans[date]
        corridor_ensembles = ensembles[start:end]
        outlook = _score_evening(home_ensemble, corridor_ensembles, events,
                                 threshold, wet_mm, config)
        result.outlooks.append(outlook)

    return result


def _score_evening(home: EnsembleForecast, corridor: Sequence[EnsembleForecast],
                   events: SolarEvents, threshold: float, wet_mm: float,
                   config: Config) -> EveningOutlook:
    """Score every ensemble member independently, then summarise the spread."""
    moment = events.sunset
    total = home.values_at("cloud_cover", moment)
    low = home.values_at("cloud_cover_low", moment)
    mid = home.values_at("cloud_cover_mid", moment)
    high = home.values_at("cloud_cover_high", moment)
    rh = home.values_at("relative_humidity_2m", moment)
    cape = home.values_at("cape", moment)
    precip = home.values_at("precipitation", moment)
    after_1 = home.values_at("cloud_cover", moment + dt.timedelta(hours=1))
    after_2 = home.values_at("cloud_cover", moment + dt.timedelta(hours=2))
    corridor_low = [c.values_at("cloud_cover_low", moment) for c in corridor]

    member_count = min([len(total), len(low), len(mid), len(high)]
                       + [len(c) for c in corridor_low] or [0])

    skies: list[float] = []
    modes: dict[str, int] = {}
    fronts = 0

    for index in range(member_count):
        # PoP is derived from member agreement because the ensemble API carries no
        # precipitation_probability. This is what an ensemble PoP actually means.
        wet_members = sum(1 for p in precip if p > wet_mm)
        pop = 100.0 * wet_members / len(precip) if precip else 0.0

        inputs = SkyInputs(
            total_cc=total[index],
            cloud_cover_low=low[index],
            cloud_cover_mid=mid[index],
            cloud_cover_high=high[index],
            corridor_low=[c[index] for c in corridor_low],
            visibility_m=None,  # not available on any ensemble model
            relative_humidity=rh[index] if index < len(rh) else 60.0,
            precipitation_probability=pop,
            cape=cape[index] if index < len(cape) else 0.0,
            total_cc_after=[
                after_1[index] if index < len(after_1) else total[index],
                after_2[index] if index < len(after_2) else total[index],
            ],
        )
        scored = score_sky(inputs, config)
        skies.append(scored.sky)
        modes[scored.best_mode] = modes.get(scored.best_mode, 0) + 1
        if scored.clearing_front:
            fronts += 1

    above = sum(1 for s in skies if s >= threshold)
    return EveningOutlook(
        date=events.date,
        events=events,
        member_skies=skies,
        stats=summarise_spread(skies),
        probability_above_good=(above / len(skies)) if skies else 0.0,
        mode_counts=modes,
        clearing_front_members=fronts,
        member_count=len(skies),
    )
