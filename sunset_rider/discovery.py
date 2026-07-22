"""Viewpoint discovery: geometry first, community data second.

The premise: underrated spots are exactly the ones with great geometry and no blog
posts about them. So we harvest candidates from OpenStreetMap by *type*, compute a
real horizon profile for each from terrain data, and keep the ones whose geometry
actually faces the sunset. Nothing here is hand-guessed.

Stages, each independently runnable and each cached:

  harvest  -> one Overpass call, cached to data/raw/osm_candidates.json
  profile  -> batched Open-Meteo elevation, cached to data/raw/horizons.json
  water    -> one Overpass call for nearby water, cached to data/raw/water.json
  build    -> data/viewpoints.yaml

Terrain does not change. A cached horizon profile is NEVER re-fetched.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
import yaml

from .config import Config, load_config, repo_root
from .geo import (
    arc_span,
    bearing_in_arc,
    destination_point,
    elevation_angle,
    haversine_km,
    normalise_bearing,
)
from .weather import OpenMeteoClient, WeatherUnavailable

log = logging.getLogger(__name__)

# OSM tag -> our candidate kind, most specific first.
#
# The tag set was widened after the first harvest missed 7 of 15 seed spots. The
# diagnostic (an Overpass name search) showed why, and every addition below fixes
# a specific, verified miss rather than being speculative:
#
#   way amenity=parking      -> Bury Down, Cowleaze Wood. The original query had
#                               node-only parking, and Ridgeway car parks are ways.
#                               This is the single most important fix: unbarred
#                               Ridgeway car parks are the exact spot type wanted.
#   natural=grassland (named)-> Lardon Chase, Lough Down (open chalk downland)
#   leisure=nature_reserve   -> Watlington Hill, Whiteleaf Hill
#   leisure=park (named)     -> Dinton Pastures Country Park (close fallback)
#   man_made=bridge (named)  -> Sonning Bridge (close fallback, river reflections)
#
# natural=water is deliberately NOT a candidate type: a lake's centroid is open
# water, which is not somewhere you can stand. Water is used only to enrich
# foreground_motion, via harvest_water().
PRIMARY_KINDS = {
    ("tourism", "viewpoint"): "viewpoint",
    ("historic", "hillfort"): "hillfort",
    ("natural", "peak"): "peak",
    ("natural", "grassland"): "downland",
    ("natural", "ridge"): "ridge",
    ("man_made", "windmill"): "windmill",
    ("leisure", "nature_reserve"): "reserve",
    ("man_made", "survey_point"): "survey_point",
    ("man_made", "bridge"): "bridge",
    ("leisure", "park"): "park",
}


@dataclass
class Candidate:
    """One harvested location, before any terrain is known."""

    id: str
    name: str
    kind: str
    latitude: float
    longitude: float
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def prior(self) -> int:
        return 0  # replaced at selection time using config

    def coord(self) -> tuple[float, float]:
        return self.latitude, self.longitude


@dataclass
class HorizonProfile:
    """Computed terrain horizon for one candidate."""

    elevation_m: float
    # bearing (int degrees) -> max terrain elevation angle along that bearing
    angles: dict[int, float]
    mean_elevation_5km: float

    def open_bearings(self, sector_start: float, sector_end: float,
                      max_angle: float) -> list[int]:
        return sorted(
            b for b, angle in self.angles.items()
            if bearing_in_arc(b, sector_start, sector_end) and angle < max_angle
        )


# ---------------------------------------------------------------------------
# Stage 1: Overpass harvest
# ---------------------------------------------------------------------------

def build_overpass_queries(config: Config) -> dict[str, str]:
    """The harvest, split into three sub-queries.

    A single combined query asking for parking ways across a 70 km radius times out
    on the public Overpass servers (verified: 107 s runtime error, then HTTP 504).
    Splitting it keeps each part inside the server's budget. These run sequentially
    with a delay between them and the merged result is cached permanently, so the
    shared free infrastructure is hit once, not once per forecast.
    """
    d = config.discovery
    radius = int(d.search_radius_m)
    around = f"(around:{radius},{config.home.latitude:.6f},{config.home.longitude:.6f})"
    timeout = int(d.overpass_timeout)

    def wrap(body: str) -> str:
        return f"[out:json][timeout:{timeout}];\n(\n{body}\n);\nout center;\n"

    return {
        "primaries": wrap("\n".join([
            f'  node["tourism"="viewpoint"]{around};',
            f'  way["tourism"="viewpoint"]{around};',
            f'  node["natural"="peak"]{around};',
            f'  node["natural"="ridge"]{around};',
            f'  node["man_made"="survey_point"]{around};',
            f'  node["historic"="hillfort"]{around};',
            f'  way["historic"="hillfort"]{around};',
            f'  node["man_made"="windmill"]{around};',
        ])),
        "areas": wrap("\n".join([
            f'  way["natural"="grassland"]["name"]{around};',
            f'  relation["natural"="grassland"]["name"]{around};',
            f'  way["leisure"="nature_reserve"]["name"]{around};',
            f'  relation["leisure"="nature_reserve"]["name"]{around};',
            f'  way["leisure"="park"]["name"]{around};',
        ])),
        # The heavy one. Ridgeway car parks are ways, and they are exactly the spot
        # type wanted, so this cannot simply be dropped.
        "parking": wrap("\n".join([
            f'  node["amenity"="parking"]["access"!="private"]{around};',
            f'  way["amenity"="parking"]["access"!="private"]{around};',
        ])),
    }


def harvest_osm(config: Config, *, force: bool = False,
                session: requests.Session | None = None) -> dict:
    """Run the Overpass query once and cache the raw response.

    One call per run. The public server is a shared free resource; we do not
    hammer it, and we never re-fetch a cached harvest unless explicitly forced.
    """
    cache = repo_root() / config.discovery.raw_osm_cache
    if cache.is_file() and not force:
        log.info("using cached OSM harvest at %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))

    d = config.discovery
    log.info("querying Overpass (%d km radius around %.4f, %.4f)",
             int(d.search_radius_m) // 1000, config.home.latitude, config.home.longitude)
    http = session or requests.Session()

    merged: list[dict] = []
    seen: set[tuple] = set()
    queries = build_overpass_queries(config)
    for index, (name, query) in enumerate(queries.items()):
        payload = _overpass_post(query, config, http, label=name)
        elements = payload.get("elements", [])
        log.info("  %-10s -> %d elements", name, len(elements))
        for element in elements:
            key = (element.get("type"), element.get("id"))
            if key not in seen:
                seen.add(key)
                merged.append(element)
        if index < len(queries) - 1:
            time.sleep(float(d.overpass_request_delay_seconds))

    payload = {"elements": merged}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    log.info("cached %d raw elements to %s", len(merged), cache)
    return payload


def _overpass_post(query: str, config: Config, http: requests.Session, *,
                   label: str) -> dict:
    """POST one Overpass query, falling back to mirrors on timeout."""
    d = config.discovery
    endpoints = [d.overpass_url] + list(d.overpass_fallback_urls)
    last_error = ""
    for endpoint in endpoints:
        try:
            response = http.post(
                endpoint, data={"data": query},
                headers={"User-Agent": d.user_agent},
                timeout=int(d.overpass_timeout) + 30,
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__} from {endpoint}"
            log.warning("  %s: %s", label, last_error)
            continue
        if response.status_code != 200:
            last_error = f"HTTP {response.status_code} from {endpoint}"
            log.warning("  %s: %s", label, last_error)
            continue
        try:
            payload = response.json()
        except ValueError:
            last_error = f"non-JSON response from {endpoint}"
            log.warning("  %s: %s", label, last_error)
            continue
        try:
            _reject_overpass_remark(payload)
        except RuntimeError as exc:
            last_error = str(exc)
            log.warning("  %s: %s", label, last_error)
            continue
        return payload
    raise RuntimeError(f"Overpass sub-query {label!r} failed on all endpoints: {last_error}")


def _reject_overpass_remark(payload: dict) -> None:
    """Overpass signals timeouts and runtime errors in a 200 response.

    A timed-out query returns HTTP 200 with an empty element list and a ``remark``.
    Without this check we would happily cache the empty result and silently build a
    viewpoint list from nothing.
    """
    remark = str(payload.get("remark", "")).strip()
    if remark and ("error" in remark.lower() or "timed out" in remark.lower()):
        raise RuntimeError(f"Overpass query failed: {remark}")


def parse_candidates(payload: dict, config: Config) -> list[Candidate]:
    """Turn raw Overpass elements into typed candidates, de-duplicated by position."""
    d = config.discovery
    out: list[Candidate] = []

    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        kind = None
        for (key, value), name in PRIMARY_KINDS.items():
            if tags.get(key) == value:
                kind = name
                break
        if kind is None and tags.get("amenity") == "parking":
            kind = "parking"
        if kind is None:
            continue

        if element.get("type") == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:
            centre = element.get("center") or {}
            lat, lon = centre.get("lat"), centre.get("lon")
        if lat is None or lon is None:
            continue

        out.append(Candidate(
            id=f"{element.get('type', 'node')}/{element.get('id')}",
            name=(tags.get("name") or "").strip(),
            kind=kind,
            latitude=float(lat),
            longitude=float(lon),
            tags={k: str(v) for k, v in tags.items()},
        ))

    return _dedupe(out, float(d.dedupe_min_separation_km), config)


class _SpatialGrid:
    """Cheap uniform-grid index for proximity queries.

    A 70 km harvest yields tens of thousands of elements, and comparing every
    candidate against every kept one is quadratic and unusably slow. Bucketing by
    an approximately cell_km grid makes both de-duplication and the parking/primary
    proximity test linear in practice.
    """

    _KM_PER_DEGREE_LAT = 111.32

    def __init__(self, cell_km: float, reference_latitude: float) -> None:
        self._cell_km = cell_km
        self._lat_step = cell_km / self._KM_PER_DEGREE_LAT
        km_per_degree_lon = self._KM_PER_DEGREE_LAT * math.cos(math.radians(reference_latitude))
        self._lon_step = cell_km / max(km_per_degree_lon, 1e-6)
        self._cells: dict[tuple[int, int], list[tuple[float, float, Any]]] = {}

    def _key(self, lat: float, lon: float) -> tuple[int, int]:
        return int(math.floor(lat / self._lat_step)), int(math.floor(lon / self._lon_step))

    def add(self, lat: float, lon: float, payload: Any = None) -> None:
        self._cells.setdefault(self._key(lat, lon), []).append((lat, lon, payload))

    def neighbours(self, lat: float, lon: float) -> Iterable[tuple[float, float, Any]]:
        row, col = self._key(lat, lon)
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                yield from self._cells.get((row + drow, col + dcol), ())

    def has_within(self, lat: float, lon: float, distance_km: float) -> bool:
        return any(
            haversine_km(lat, lon, other_lat, other_lon) < distance_km
            for other_lat, other_lon, _ in self.neighbours(lat, lon)
        )


def _elevation_tag(candidate: Candidate) -> float:
    """The OSM ``ele`` tag, if it parses. A free height signal before any DEM lookup."""
    raw = candidate.tags.get("ele", "")
    try:
        return float(str(raw).split()[0])
    except (ValueError, IndexError):
        return 0.0


def local_relief(candidates: Sequence[Candidate], config: Config) -> dict[str, float]:
    """Height above nearby candidates, from OSM ``ele`` tags alone. Costs no API calls.

    Absolute altitude is the wrong prior in this terrain. Wittenham Clumps is a
    famous viewpoint at 120 m because it stands above the flat Thames floodplain,
    while a 240 m Chiltern peak surrounded by other 230 m peaks sees very little.
    Ranking candidates by raw ``ele`` therefore pushed exactly the wrong spots to the
    front of the profiling queue.

    This approximates prominence using tagged neighbours within the configured
    radius, which is free and available before any elevation is fetched. It is only
    a selection prior — the real horizon profile still decides what survives.
    """
    radius = float(config.discovery.prominence_radius_km)
    tagged = [(c, _elevation_tag(c)) for c in candidates]
    tagged = [(c, e) for c, e in tagged if e > 0.0]
    if not tagged:
        return {}

    grid = _SpatialGrid(radius, config.home.latitude)
    for candidate, elevation in tagged:
        grid.add(candidate.latitude, candidate.longitude, elevation)

    relief: dict[str, float] = {}
    for candidate, elevation in tagged:
        neighbours = [
            value for lat, lon, value in grid.neighbours(candidate.latitude,
                                                         candidate.longitude)
            if haversine_km(candidate.latitude, candidate.longitude, lat, lon) <= radius
        ]
        if len(neighbours) <= 1:
            relief[candidate.id] = 0.0
        else:
            relief[candidate.id] = elevation - (sum(neighbours) / len(neighbours))
    return relief


def _dedupe(candidates: Sequence[Candidate], min_km: float,
            config: Config) -> list[Candidate]:
    """Drop candidates that sit on top of a better one.

    Named candidates beat unnamed ones at the same prior. Without that tiebreak the
    unnamed viewpoint node next to "Whiteleaf Hill" wins and the spot loses the name
    the rider would actually recognise in a message.
    """
    priors = config.discovery.candidate_type_prior
    ordered = sorted(
        candidates,
        key=lambda c: (-int(priors.get(c.kind, 0)), not bool(c.name), -_elevation_tag(c)),
    )
    grid = _SpatialGrid(min_km, config.home.latitude)
    kept: list[Candidate] = []
    for cand in ordered:
        if grid.has_within(cand.latitude, cand.longitude, min_km):
            continue
        grid.add(cand.latitude, cand.longitude, cand)
        kept.append(cand)
    return kept


def select_for_profiling(candidates: Sequence[Candidate], config: Config) -> list[Candidate]:
    """Choose which candidates are worth the elevation budget.

    Parking nodes are the noisy majority of a 70 km harvest, so they are only kept
    when named or when they plausibly serve a primary candidate.
    """
    d = config.discovery
    priors = d.candidate_type_prior
    primaries = [c for c in candidates if c.kind != "parking"]
    parking_limit = float(d.parking_max_distance_to_primary_km)

    primary_grid = _SpatialGrid(parking_limit, config.home.latitude)
    for primary in primaries:
        primary_grid.add(primary.latitude, primary.longitude, primary)

    keep: list[Candidate] = list(primaries)
    for cand in candidates:
        if cand.kind != "parking":
            continue
        if cand.name or primary_grid.has_within(cand.latitude, cand.longitude, parking_limit):
            keep.append(cand)

    home = (config.home.latitude, config.home.longitude)
    relief = local_relief(keep, config)

    def rank(c: Candidate) -> tuple:
        # Type prior, then notability (a name means somebody thought it mattered),
        # then local relief, then proximity. Relief rather than raw altitude: what
        # makes a viewpoint is standing above your surroundings, not absolute height.
        return (
            -int(priors.get(c.kind, 0)),
            not bool(c.name),
            -relief.get(c.id, 0.0),
            -_elevation_tag(c),
            haversine_km(home[0], home[1], c.latitude, c.longitude),
        )

    # Stratify by distance band, then round-robin across kinds within each band.
    #
    # Both halves are load-bearing. Ranking purely by proximity spends the whole
    # elevation budget inside 20 km and never reaches the Berkshire Downs. Ranking
    # purely by type prior is worse: tourism=viewpoint (prior 100) crowds out
    # natural=peak (prior 80) completely, and the best downland — Walbury Hill,
    # Watership Down, Coombe Hill, Dragon Hill — is tagged as peaks. A measured run
    # with plain prior ordering selected 135 viewpoints and only 15 peaks, and
    # missed Walbury Hill by 9.5 km.
    # Spend the elevation budget on kinds that actually produce viewpoints. Parks,
    # car parks and reserves are kept only as close waterside fallbacks, and capped.
    high_value = set(d.high_value_kinds)
    fallback_kinds = set(d.fallback_kinds)
    close_km = float(d.close_fallback_max_km)

    primary_pool = [c for c in keep if c.kind in high_value]
    fallback_pool = [
        c for c in keep
        if c.kind in fallback_kinds
        and haversine_km(home[0], home[1], c.latitude, c.longitude) <= close_km
    ]

    bands = [(float(b[0]), float(b[1])) for b in d.distance_bands_km]
    per_band = int(d.max_per_distance_band)
    kinds_by_prior = sorted({c.kind for c in primary_pool},
                            key=lambda k: -int(priors.get(k, 0)))
    selected: list[Candidate] = []

    for low, high in bands:
        in_band: dict[str, list[Candidate]] = {k: [] for k in kinds_by_prior}
        for cand in primary_pool:
            distance = haversine_km(home[0], home[1], cand.latitude, cand.longitude)
            if low <= distance < high:
                in_band[cand.kind].append(cand)
        for bucket in in_band.values():
            bucket.sort(key=rank)

        picked: list[Candidate] = []
        while len(picked) < per_band and any(in_band[k] for k in kinds_by_prior):
            for kind in kinds_by_prior:
                if len(picked) >= per_band:
                    break
                if in_band[kind]:
                    picked.append(in_band[kind].pop(0))
        selected.extend(picked)

    # A handful of close fallbacks: on a mediocre evening the right answer is a
    # 15-minute hop, not staying home. Ranked by proximity rather than type prior,
    # because being close IS the defining property here — ordering these by prior put
    # six nature reserves ahead of Dinton Pastures, which is the canonical example.
    fallback_pool.sort(key=lambda c: haversine_km(home[0], home[1],
                                                  c.latitude, c.longitude))
    selected.extend(fallback_pool[:int(d.max_fallback_candidates)])

    # Safety cap only. It must not be smaller than the band quotas, or it would
    # silently undo the stratification by dropping the lowest-prior kinds.
    return selected[:int(d.max_profiled_candidates)]


# ---------------------------------------------------------------------------
# Stage 2: horizon profiling
# ---------------------------------------------------------------------------

def horizon_sample_points(candidate: Candidate, config: Config
                          ) -> list[tuple[float, float]]:
    """The 36 x 7 terrain sample grid for one candidate, plus the candidate itself."""
    d = config.discovery
    step = int(d.bearing_step_degrees)
    radius = float(d.earth_radius_km)
    points = [(candidate.latitude, candidate.longitude)]
    for bearing in range(0, 360, step):
        for distance in d.sample_distances_km:
            points.append(destination_point(
                candidate.latitude, candidate.longitude,
                float(bearing), float(distance), radius,
            ))
    return points


def profile_from_elevations(elevations: Sequence[float], config: Config) -> HorizonProfile:
    """Fold a flat elevation list back into a horizon profile.

    The horizon angle for a bearing is the maximum elevation angle along it: a
    single close ridge blocks the view regardless of what lies beyond.
    """
    d = config.discovery
    step = int(d.bearing_step_degrees)
    distances = [float(x) for x in d.sample_distances_km]
    prominence_radius = float(d.prominence_radius_km)

    viewpoint_elev = float(elevations[0])
    angles: dict[int, float] = {}
    within_5km: list[float] = []

    index = 1
    for bearing in range(0, 360, step):
        best = -90.0
        for distance in distances:
            sample = float(elevations[index])
            index += 1
            angle = elevation_angle(viewpoint_elev, sample, distance)
            best = max(best, angle)
            if distance <= prominence_radius:
                within_5km.append(sample)
        angles[bearing] = best

    mean_5km = sum(within_5km) / len(within_5km) if within_5km else viewpoint_elev
    return HorizonProfile(elevation_m=viewpoint_elev, angles=angles,
                          mean_elevation_5km=mean_5km)


def load_cached_profiles(config: Config) -> dict[str, HorizonProfile]:
    """Every horizon profile ever computed, regardless of the current selection.

    Selection heuristics get retuned; horizon profiles are expensive and permanent.
    The build reads from here so that changing the ranking never discards terrain
    data already paid for.
    """
    cache_path = repo_root() / config.discovery.raw_horizon_cache
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        key: HorizonProfile(
            elevation_m=float(entry["elevation_m"]),
            angles={int(k): float(v) for k, v in entry["angles"].items()},
            mean_elevation_5km=float(entry["mean_elevation_5km"]),
        )
        for key, entry in raw.items()
    }


def profile_horizons(candidates: Sequence[Candidate], config: Config,
                     client: OpenMeteoClient, *, force: bool = False,
                     cache_only: bool = False) -> dict[str, HorizonProfile]:
    """Compute horizon profiles, reusing the permanent cache wherever possible.

    Terrain does not change. Anything already in data/raw/horizons.json is reused
    verbatim and never re-fetched.
    """
    d = config.discovery
    cache_path = repo_root() / d.raw_horizon_cache
    cached: dict[str, Any] = {}
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    profiles: dict[str, HorizonProfile] = {}
    todo: list[Candidate] = []
    for cand in candidates:
        entry = cached.get(cand.id)
        if entry is not None:
            profiles[cand.id] = HorizonProfile(
                elevation_m=float(entry["elevation_m"]),
                angles={int(k): float(v) for k, v in entry["angles"].items()},
                mean_elevation_5km=float(entry["mean_elevation_5km"]),
            )
        else:
            todo.append(cand)

    if not todo:
        log.info("all %d horizon profiles served from cache", len(profiles))
        return profiles

    if cache_only:
        # Rebuild viewpoints.yaml from what is already computed, touching no network.
        # Useful after changing the geometry filters, and essential once the daily
        # elevation quota is spent: without this the run would sit in rate-limit
        # backoff for 20 minutes before giving up on data it cannot get today.
        log.info("cache-only: building from %d cached profiles, skipping %d unprofiled",
                 len(profiles), len(todo))
        return profiles

    per_candidate = 1 + (360 // int(d.bearing_step_degrees)) * len(d.sample_distances_km)
    log.info("profiling %d new candidates (%d cached), %d locations each = %d total",
             len(todo), len(profiles), per_candidate, len(todo) * per_candidate)

    delay = float(d.elevation_request_delay_seconds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Profiled one candidate at a time and flushed to disk after each. The elevation
    # API is rate limited to 600 locations/minute, so a full run takes a while and
    # WILL sometimes be interrupted. Writing incrementally means an interrupted run
    # resumes instead of starting over, and no cached profile is ever re-fetched.
    for index, cand in enumerate(todo, start=1):
        points = horizon_sample_points(cand, config)
        try:
            elevations = client.elevation(points)
        except WeatherUnavailable as exc:
            log.error("stopping after %d/%d: %s", index - 1, len(todo), exc)
            log.error("progress is saved; re-run the same command to resume")
            break

        profile = profile_from_elevations(elevations, config)
        profiles[cand.id] = profile
        cached[cand.id] = {
            "elevation_m": profile.elevation_m,
            "angles": {str(k): v for k, v in profile.angles.items()},
            "mean_elevation_5km": profile.mean_elevation_5km,
        }
        cache_path.write_text(json.dumps(cached), encoding="utf-8")

        if index % 5 == 0 or index == len(todo):
            log.info("  profiled %d/%d (%s)", index, len(todo),
                     cand.name or cand.id)
        if delay and index < len(todo):
            time.sleep(delay)

    log.info("cached %d horizon profiles to %s", len(cached), cache_path)
    return profiles


# ---------------------------------------------------------------------------
# Stage 3: derived geometry
# ---------------------------------------------------------------------------

def compute_open_arc(profile: HorizonProfile, config: Config
                     ) -> tuple[tuple[int, int] | None, float]:
    """Longest contiguous open arc within the sunset sector, and overall openness.

    Returns ((start_bearing, end_bearing) or None, openness_fraction).

    The sector runs 200-330 deg because that is the full swing of the Reading
    sunset azimuth: ~232 deg at midwinter to ~311 deg at midsummer.
    """
    d = config.discovery
    step = int(d.bearing_step_degrees)
    start = float(d.sunset_sector_start_deg)
    end = float(d.sunset_sector_end_deg)
    max_angle = float(d.open_horizon_max_angle_deg)

    sector = [b for b in sorted(profile.angles) if bearing_in_arc(b, start, end)]
    if not sector:
        return None, 0.0

    open_flags = [profile.angles[b] < max_angle for b in sector]
    openness = sum(open_flags) / len(open_flags)

    best_run: tuple[int, int] | None = None
    best_len = 0
    run_start: int | None = None
    for bearing, is_open in zip(sector, open_flags):
        if is_open and run_start is None:
            run_start = bearing
        if not is_open and run_start is not None:
            run_len = bearing - run_start
            if run_len > best_len:
                best_len, best_run = run_len, (run_start, bearing - step)
            run_start = None
    if run_start is not None:
        run_len = sector[-1] - run_start
        if run_len >= best_len:
            best_len, best_run = run_len, (run_start, sector[-1])

    return best_run, openness


def normalised_prominence(profile: HorizonProfile, config: Config) -> float:
    """Elevation above the local mean, normalised to roughly 0..1 and clamped."""
    raw = profile.elevation_m - profile.mean_elevation_5km
    normaliser = float(config.discovery.prominence_normaliser_m)
    return max(0.0, min(1.0, raw / normaliser))


def determine_gate_closes(candidate: Candidate) -> str | None:
    """Work out whether a gate will lock us in, from OSM tags only.

    Deliberately conservative. We never invent an opening time: anything we cannot
    establish from tags is reported as "unknown" and flagged in the message, because
    a car park that locks at dusk makes a sunset viewpoint worthless.

    Returns None for open-access/roadside, otherwise "sunset", "dusk", "HH:MM" or "unknown".
    """
    tags = candidate.tags
    opening = (tags.get("opening_hours") or "").strip().lower()
    barrier = (tags.get("barrier") or "").strip().lower()
    access = (tags.get("access") or "").strip().lower()

    if opening in {"24/7", "24 hours", "mo-su 00:00-24:00"}:
        return None
    if "sunset" in opening or "dusk" in opening:
        return "sunset"
    if opening:
        # Pull a closing time out of a simple "Mo-Su 08:00-20:00" style value.
        import re
        times = re.findall(r"(\d{1,2}):(\d{2})", opening)
        if times:
            hour, minute = times[-1]
            return f"{int(hour):02d}:{minute}"
        return "unknown"

    # No opening_hours at all. Roadside laybys and unbarred verges are the good case.
    if barrier in {"gate", "lift_gate", "bollard", "barrier_gate"}:
        return "unknown"
    if candidate.kind == "parking":
        return "unknown"
    if access in {"yes", "permissive", "public"}:
        return None
    # Peaks, hillforts and open-access downland reached on foot from a road.
    return None


# ---------------------------------------------------------------------------
# Stage 4: water proximity (foreground motion)
# ---------------------------------------------------------------------------

def harvest_water(candidates: Sequence[Candidate], config: Config, *,
                  force: bool = False, session: requests.Session | None = None) -> dict:
    """One Overpass call for water near the surviving candidates."""
    d = config.discovery
    cache = repo_root() / d.raw_water_cache
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    radius = int(d.water_search_radius_m)
    clauses = []
    for cand in candidates:
        around = f"(around:{radius},{cand.latitude:.6f},{cand.longitude:.6f})"
        clauses.append(f'  way["natural"="water"]{around};')
        clauses.append(f'  way["waterway"="river"]{around};')
    query = ("[out:json][timeout:%d];\n(\n%s\n);\nout center;\n"
             % (int(d.overpass_timeout), "\n".join(clauses)))

    http = session or requests.Session()
    response = http.post(d.overpass_url, data={"data": query},
                         headers={"User-Agent": d.user_agent},
                         timeout=int(d.overpass_timeout) + 30)
    if response.status_code != 200:
        raise WeatherUnavailable(
            f"Overpass (water) returned HTTP {response.status_code}", source="overpass"
        )
    payload = response.json()
    _reject_overpass_remark(payload)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def water_distance_km(candidate: Candidate, water_payload: dict) -> float | None:
    """Distance to the nearest mapped water feature, or None if none nearby."""
    best: float | None = None
    for element in water_payload.get("elements", []):
        centre = element.get("center") or {}
        lat, lon = centre.get("lat"), centre.get("lon")
        if lat is None or lon is None:
            continue
        distance = haversine_km(candidate.latitude, candidate.longitude,
                                float(lat), float(lon))
        if best is None or distance < best:
            best = distance
    return best


def foreground_motion_score(distance_km: float | None, config: Config) -> float:
    """0..1 — moving water close by means life in the frame, which video needs."""
    d = config.discovery
    if distance_km is None:
        return 0.0
    full = float(d.water_full_credit_km)
    zero = float(d.water_zero_credit_km)
    if distance_km <= full:
        return 1.0
    if distance_km >= zero:
        return 0.0
    return (zero - distance_km) / (zero - full)


# ---------------------------------------------------------------------------
# Stage 5: build viewpoints.yaml
# ---------------------------------------------------------------------------

def spot_quality(openness: float, prominence: float, kind: str, config: Config) -> float:
    """The 0-100 "spot" term used by worth_it."""
    s = config.spot
    priors = config.discovery.candidate_type_prior
    max_prior = max(int(v) for v in priors.as_dict().values()) if hasattr(priors, "as_dict") \
        else max(int(priors[k]) for k in priors)
    type_term = int(priors.get(kind, 0)) / max_prior if max_prior else 0.0
    return 100.0 * (
        float(s.weight_openness) * openness
        + float(s.weight_prominence) * prominence
        + float(s.weight_type) * type_term
    )


def build_viewpoints(candidates: Sequence[Candidate],
                     profiles: dict[str, HorizonProfile],
                     water_payload: dict,
                     config: Config) -> tuple[list[dict], list[tuple[str, str]]]:
    """Assemble the final viewpoint records, keeping only spots with real geometry.

    Returns (records, rejections) so the discovery report can explain what was
    dropped and why, rather than silently shrinking the list.
    """
    d = config.discovery
    s = config.spot
    min_openness = float(d.min_horizon_openness)
    min_prominence = float(d.min_elevation_prominence)
    fallback_water_km = float(d.fallback_water_max_km)
    fallback_min_openness = float(d.fallback_min_openness)
    close_km = float(d.close_fallback_max_km)
    min_distance_km = float(d.min_distance_km)
    home = (config.home.latitude, config.home.longitude)
    road_factor = float(config.rider.road_distance_factor)
    speed = float(config.rider.average_speed_kmh)

    records: list[dict] = []
    rejections: list[tuple[str, str]] = []

    for cand in candidates:
        profile = profiles.get(cand.id)
        if profile is None:
            continue
        label = cand.name or f"unnamed {cand.kind}"
        straight_km_pre = haversine_km(home[0], home[1], cand.latitude, cand.longitude)
        if straight_km_pre < min_distance_km:
            rejections.append((label, f"{straight_km_pre:.1f} km — that is a walk, "
                                      f"not a ride"))
            continue

        arc, openness = compute_open_arc(profile, config)
        if openness < min_openness or arc is None:
            rejections.append((label, f"horizon openness {openness:.2f} < {min_openness}"))
            continue

        prominence = normalised_prominence(profile, config)
        straight_km = haversine_km(home[0], home[1], cand.latitude, cand.longitude)
        road_km = straight_km * road_factor
        minutes = road_km / speed * 60.0
        water_km = water_distance_km(cand, water_payload)
        gate = determine_gate_closes(cand)

        # Flat ground has no terrain obstruction and therefore perfect "openness",
        # but no view either. Genuine waterside close fallbacks are exempt: they earn
        # their place through reflections rather than elevation. The exemption is
        # deliberately narrow — right at the water, and with a wide open western
        # horizon — so that riverside town parks do not qualify as viewpoints.
        water_fallback = (
            water_km is not None
            and water_km <= fallback_water_km
            and straight_km <= close_km
            and openness >= fallback_min_openness
        )
        if prominence < min_prominence and not water_fallback:
            rejections.append((
                label,
                f"flat: prominence {prominence:.2f} < {min_prominence} "
                f"and not a waterside fallback",
            ))
            continue

        foreground_interest = (
            float(s.foreground_interest_openness_weight) * openness
            + float(s.foreground_interest_prominence_weight) * prominence
        )

        records.append({
            "id": cand.id.replace("/", "_"),
            "name": cand.name or f"Unnamed {cand.kind} ({cand.id})",
            "kind": cand.kind,
            "latitude": round(cand.latitude, 6),
            "longitude": round(cand.longitude, 6),
            "elevation_m": round(profile.elevation_m, 1),
            "horizon_profile": {int(b): round(a, 2) for b, a in sorted(profile.angles.items())},
            "open_arc": [int(arc[0]), int(arc[1])],
            "open_arc_span_deg": int(arc_span(arc[0], arc[1])),
            "horizon_openness": round(openness, 3),
            "elevation_prominence": round(prominence, 3),
            "foreground_interest": round(foreground_interest, 3),
            "foreground_motion": round(foreground_motion_score(water_km, config), 3),
            "water_distance_km": None if water_km is None else round(water_km, 2),
            "distance_km": round(straight_km, 2),
            "road_distance_km": round(road_km, 2),
            "minutes_one_way": round(minutes, 1),
            "close_fallback": straight_km <= float(d.close_fallback_max_km),
            "gate_closes": gate,
            "spot_score": round(spot_quality(openness, prominence, cand.kind, config), 1),
            "osm_id": cand.id,
            "osm_tags": {k: v for k, v in cand.tags.items()
                         if k in {"access", "fee", "opening_hours", "surface",
                                  "barrier", "direction", "ele", "operator"}},
        })

    records.sort(key=lambda r: -r["spot_score"])
    return records, rejections


def write_viewpoints(records: Sequence[dict], config: Config) -> Path:
    path = repo_root() / config.discovery.output_viewpoints
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by discovery.py. Do not hand-edit the computed fields.\n"
        "#\n"
        "# horizon_profile / open_arc / horizon_openness / elevation_prominence are\n"
        "# computed from the Open-Meteo 90 m DEM, not guessed. Regenerate with:\n"
        "#   python -m sunset_rider.discovery --build\n"
        "#\n"
        "# gate_closes is the field most likely to ruin an evening. null means open\n"
        "# access or roadside; \"unknown\" is flagged with a warning in messages.\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump({"viewpoints": list(records)}, handle, sort_keys=False,
                       allow_unicode=True, default_flow_style=False)
    return path


def load_viewpoints(config: Config) -> list[dict]:
    """Read the generated viewpoint list."""
    path = repo_root() / config.discovery.output_viewpoints
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run python -m sunset_rider.discovery --build"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("viewpoints", []))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(config: Config, *, force_harvest: bool = False, force_profile: bool = False,
        stage: str = "build", cache_only: bool = False) -> list[dict]:
    payload = harvest_osm(config, force=force_harvest)
    elements = payload.get("elements", [])
    log.info("raw Overpass elements: %d", len(elements))
    if len(elements) < int(config.discovery.min_raw_candidates):
        raise RuntimeError(
            f"Overpass returned only {len(elements)} raw candidates, fewer than the "
            f"{int(config.discovery.min_raw_candidates)} the design expects. Stop and "
            f"investigate before trusting this list."
        )

    candidates = parse_candidates(payload, config)
    log.info("typed + de-duplicated candidates: %d", len(candidates))
    if stage == "harvest":
        return []

    selected = select_for_profiling(candidates, config)
    log.info("selected for profiling: %d", len(selected))

    client = OpenMeteoClient(config)
    profile_horizons(selected, config, client, force=force_profile,
                     cache_only=cache_only)
    if stage == "profile":
        return []

    # Read back everything on disk, not just what this run's selection covered.
    profiles = load_cached_profiles(config)

    # Build from EVERY candidate with a cached profile, not just the current
    # selection. Selection heuristics get tuned; horizon profiles are expensive and
    # permanent. Filtering to the current selection would silently discard terrain
    # data already paid for whenever the ranking changes.
    profiled = [c for c in candidates if c.id in profiles]
    pending = sum(1 for c in selected if c.id not in profiles)
    if pending:
        log.warning("%d of %d selected candidates are not yet profiled; re-run to top up",
                    pending, len(selected))
    log.info("building from %d cached profiles", len(profiled))

    survivors = [c for c in profiled
                 if compute_open_arc(profiles[c.id], config)[1]
                 >= float(config.discovery.min_horizon_openness)]
    log.info("survived the geometry filter: %d", len(survivors))

    water = harvest_water(survivors, config)
    records, rejections = build_viewpoints(profiled, profiles, water, config)
    for name, reason in rejections:
        log.info("  rejected %-34s %s", name[:34], reason)
    path = write_viewpoints(records, config)
    log.info("wrote %d viewpoints to %s (%d rejected)",
             len(records), path, len(rejections))
    if len(records) < int(config.discovery.target_viewpoint_count):
        log.warning(
            "only %d viewpoints — target is %d. Re-run to profile more candidates "
            "(the horizon cache is permanent, so nothing is re-fetched).",
            len(records), int(config.discovery.target_viewpoint_count),
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(description="Discover sunset viewpoints near Reading.")
    parser.add_argument("--stage", choices=["harvest", "profile", "build"], default="build")
    parser.add_argument("--force-harvest", action="store_true",
                        help="re-run the Overpass query even if cached")
    parser.add_argument("--force-profile", action="store_true",
                        help="re-fetch horizon profiles (terrain does not change; "
                             "you almost never want this)")
    parser.add_argument("--cache-only", action="store_true",
                        help="rebuild viewpoints.yaml from cached profiles without "
                             "fetching anything new")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    run(config, force_harvest=args.force_harvest,
        force_profile=args.force_profile, stage=args.stage,
        cache_only=args.cache_only)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
