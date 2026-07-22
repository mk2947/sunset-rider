"""Discovery tests: horizon profiling, open arcs, gate parsing, selection.

The synthetic-DEM test is the important one. It proves the whole premise of the
project: a spot with a ridge at 300 degrees is superb in December and useless in
June, and the pipeline knows that from terrain alone.
"""

from __future__ import annotations

import json
import math

import pytest

from sunset_rider.config import Config
from sunset_rider.discovery import (
    Candidate,
    HorizonProfile,
    _SpatialGrid,
    _dedupe,
    _elevation_tag,
    _reject_overpass_remark,
    build_overpass_queries,
    build_viewpoints,
    compute_open_arc,
    determine_gate_closes,
    foreground_motion_score,
    horizon_sample_points,
    normalised_prominence,
    parse_candidates,
    profile_from_elevations,
    select_for_profiling,
    spot_quality,
    water_distance_km,
)
from sunset_rider.geo import haversine_km
from sunset_rider.scoring import azimuth_match

JUNE_AZIMUTH = 310.95
DECEMBER_AZIMUTH = 231.61


# ---------------------------------------------------------------------------
# Synthetic DEM
# ---------------------------------------------------------------------------

def _flat_elevations(config, base=100.0):
    """Elevations for one candidate on perfectly flat ground."""
    per = 1 + (360 // int(config.discovery.bearing_step_degrees)) * len(
        config.discovery.sample_distances_km)
    return [base] * per


def _elevations_with_wall(config, wall_bearing, wall_height=250.0, base=100.0,
                          spread=10):
    """Flat ground everywhere except a tall ridge on one bearing."""
    step = int(config.discovery.bearing_step_degrees)
    distances = list(config.discovery.sample_distances_km)
    values = [base]
    for bearing in range(0, 360, step):
        offset = min(abs(bearing - wall_bearing), 360 - abs(bearing - wall_bearing))
        for _distance in distances:
            values.append(wall_height if offset <= spread else base)
    return values


def test_sample_grid_has_the_expected_shape(config):
    candidate = Candidate("node/1", "Test", "peak", 51.5, -1.2)
    points = horizon_sample_points(candidate, config)
    bearings = 360 // int(config.discovery.bearing_step_degrees)
    expected = 1 + bearings * len(config.discovery.sample_distances_km)
    assert len(points) == expected
    assert points[0] == (51.5, -1.2)


def test_flat_terrain_is_fully_open(config):
    profile = profile_from_elevations(_flat_elevations(config), config)
    arc, openness = compute_open_arc(profile, config)
    assert openness == pytest.approx(1.0)
    assert arc == (int(config.discovery.sunset_sector_start_deg),
                   int(config.discovery.sunset_sector_end_deg))


def test_a_wall_at_300_degrees_is_excluded_from_the_open_arc(config):
    """THE load-bearing test.

    A 150 m wall at 300 degrees blocks the midsummer sunset (311) but not the
    midwinter one (232). The system knows this without anyone having stood there.
    """
    profile = profile_from_elevations(
        _elevations_with_wall(config, wall_bearing=300), config)
    arc, openness = compute_open_arc(profile, config)

    assert openness < 1.0, "the wall should have closed part of the sector"
    blocked = [b for b, angle in profile.angles.items()
               if angle >= float(config.discovery.open_horizon_max_angle_deg)]
    assert 300 in blocked

    # The open arc must not contain the blocked bearing.
    assert arc is not None
    assert not (arc[0] <= 300 <= arc[1]), f"arc {arc} still contains 300"


def test_the_june_azimuth_is_penalised_and_december_is_not(config):
    """Same spot, opposite verdicts six months apart."""
    profile = profile_from_elevations(
        _elevations_with_wall(config, wall_bearing=300), config)
    arc, _ = compute_open_arc(profile, config)

    june = azimuth_match(JUNE_AZIMUTH, arc, config)
    december = azimuth_match(DECEMBER_AZIMUTH, arc, config)

    assert december == float(config.worth_it.azimuth_in_arc), (
        f"December ({DECEMBER_AZIMUTH}) should sit inside arc {arc}"
    )
    assert june < december, f"June ({JUNE_AZIMUTH}) should be penalised, arc {arc}"


def test_a_wall_at_235_degrees_reverses_the_verdict(config):
    """Mirror image: blocking the south-west ruins December, not June."""
    profile = profile_from_elevations(
        _elevations_with_wall(config, wall_bearing=230), config)
    arc, _ = compute_open_arc(profile, config)
    assert azimuth_match(JUNE_AZIMUTH, arc, config) > azimuth_match(
        DECEMBER_AZIMUTH, arc, config)


def test_horizon_angle_takes_the_maximum_along_a_bearing(config):
    """A single close ridge blocks the view regardless of what lies beyond it."""
    step = int(config.discovery.bearing_step_degrees)
    distances = list(config.discovery.sample_distances_km)
    values = [100.0]
    for bearing in range(0, 360, step):
        for index, _distance in enumerate(distances):
            # A bump only at the nearest sample on bearing 0.
            values.append(400.0 if (bearing == 0 and index == 0) else 100.0)
    profile = profile_from_elevations(values, config)
    assert profile.angles[0] > 10.0
    assert profile.angles[step] == pytest.approx(0.0)


def test_profile_records_all_thirty_six_bearings(config):
    profile = profile_from_elevations(_flat_elevations(config), config)
    assert len(profile.angles) == 36
    assert sorted(profile.angles) == list(range(0, 360, 10))


def test_prominence_is_zero_on_flat_ground_and_positive_on_a_hill(config):
    flat = profile_from_elevations(_flat_elevations(config), config)
    assert normalised_prominence(flat, config) == pytest.approx(0.0)

    hill = HorizonProfile(elevation_m=220.0, angles={b: 0.0 for b in range(0, 360, 10)},
                          mean_elevation_5km=120.0)
    assert normalised_prominence(hill, config) > 0.5


def test_prominence_is_clamped_to_one(config):
    huge = HorizonProfile(elevation_m=2000.0, angles={}, mean_elevation_5km=10.0)
    assert normalised_prominence(huge, config) == 1.0


def test_openness_is_a_fraction_of_the_sunset_sector_only(config):
    """Terrain outside 200-330 must not affect openness."""
    north_wall = profile_from_elevations(
        _elevations_with_wall(config, wall_bearing=20), config)
    _, openness = compute_open_arc(north_wall, config)
    assert openness == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate parsing — the field that will actually ruin evenings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tags, expected", [
    ({"opening_hours": "24/7"}, None),
    ({"opening_hours": "sunrise-sunset"}, "sunset"),
    ({"opening_hours": "Mo-Su 08:00-dusk"}, "sunset"),
    ({"opening_hours": "Mo-Su 08:00-20:00"}, "20:00"),
    ({"opening_hours": "Mo-Su 07:00-21:30"}, "21:30"),
])
def test_gate_closes_is_read_from_opening_hours(tags, expected):
    candidate = Candidate("node/1", "Test", "parking", 51.5, -1.2, tags)
    assert determine_gate_closes(candidate) == expected


def test_a_barrier_means_unknown_not_open():
    candidate = Candidate("node/1", "Test", "peak", 51.5, -1.2, {"barrier": "gate"})
    assert determine_gate_closes(candidate) == "unknown"


def test_an_untagged_car_park_is_unknown_rather_than_assumed_open():
    """Conservative on purpose: a car park that locks at dusk is worthless."""
    candidate = Candidate("node/1", "Test", "parking", 51.5, -1.2, {})
    assert determine_gate_closes(candidate) == "unknown"


def test_an_untagged_peak_is_treated_as_open_access():
    candidate = Candidate("node/1", "Test", "peak", 51.5, -1.2, {})
    assert determine_gate_closes(candidate) is None


# ---------------------------------------------------------------------------
# Harvest parsing and selection
# ---------------------------------------------------------------------------

def test_overpass_queries_are_split_and_cover_parking_ways(config):
    """Ridgeway car parks are OSM ways; a node-only query cannot find Bury Down."""
    queries = build_overpass_queries(config)
    assert set(queries) == {"primaries", "areas", "parking"}
    assert 'way["amenity"="parking"]' in queries["parking"]
    assert 'node["amenity"="parking"]' in queries["parking"]
    combined = "".join(queries.values())
    assert f"{config.home.latitude:.6f}" in combined
    assert str(int(config.discovery.search_radius_m)) in combined


def test_a_timed_out_overpass_response_is_rejected():
    """Overpass signals failure in a 200 response; caching that would be silent death."""
    with pytest.raises(RuntimeError, match="timed out"):
        _reject_overpass_remark({"elements": [],
                                 "remark": 'runtime error: Query timed out in "query"'})


def test_a_normal_overpass_response_passes():
    _reject_overpass_remark({"elements": [{"id": 1}]})


def test_parse_candidates_types_and_locates_elements(config):
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 51.5, "lon": -1.2,
         "tags": {"tourism": "viewpoint", "name": "A"}},
        {"type": "way", "id": 2, "center": {"lat": 51.9, "lon": -1.6},
         "tags": {"amenity": "parking", "name": "B"}},
        {"type": "node", "id": 3, "lat": 51.7, "lon": -1.4, "tags": {"natural": "peak"}},
        {"type": "node", "id": 4, "lat": 51.6, "lon": -1.3, "tags": {"shop": "bakery"}},
    ]}
    candidates = parse_candidates(payload, config)
    kinds = {c.kind for c in candidates}
    assert kinds == {"viewpoint", "parking", "peak"}
    assert all(c.latitude and c.longitude for c in candidates)


def test_dedupe_prefers_the_named_candidate(config):
    """Otherwise an unnamed viewpoint node suppresses "Whiteleaf Hill"."""
    named = Candidate("node/1", "Whiteleaf Hill", "viewpoint", 51.7288, -0.8105)
    unnamed = Candidate("node/2", "", "viewpoint", 51.7289, -0.8106)
    kept = _dedupe([unnamed, named], 0.6, config)
    assert len(kept) == 1
    assert kept[0].name == "Whiteleaf Hill"


def test_dedupe_keeps_spots_that_are_far_enough_apart(config):
    a = Candidate("node/1", "A", "peak", 51.50, -1.20)
    b = Candidate("node/2", "B", "peak", 51.60, -1.30)
    assert len(_dedupe([a, b], 0.6, config)) == 2


def test_selection_is_diversified_across_kinds(config):
    """Prior ordering alone selects 135 viewpoints and 15 peaks, missing Walbury Hill."""
    candidates = []
    for index in range(60):
        candidates.append(Candidate(f"node/v{index}", f"V{index}", "viewpoint",
                                    51.5 + index * 0.01, -1.0))
        candidates.append(Candidate(f"node/p{index}", f"P{index}", "peak",
                                    51.5 + index * 0.01, -1.1))
    selected = select_for_profiling(candidates, config)
    kinds = {c.kind for c in selected}
    assert "peak" in kinds, "peaks were crowded out by viewpoints"
    assert "viewpoint" in kinds


def test_selection_respects_the_candidate_cap(config):
    candidates = [Candidate(f"node/{i}", f"N{i}", "viewpoint", 51.5 + i * 0.005, -1.0)
                  for i in range(500)]
    selected = select_for_profiling(candidates, config)
    assert len(selected) <= int(config.discovery.max_profiled_candidates)


def test_elevation_tag_parsing():
    assert _elevation_tag(Candidate("n/1", "", "peak", 0, 0, {"ele": "297"})) == 297.0
    assert _elevation_tag(Candidate("n/1", "", "peak", 0, 0, {"ele": "297 m"})) == 297.0
    assert _elevation_tag(Candidate("n/1", "", "peak", 0, 0, {"ele": "high"})) == 0.0
    assert _elevation_tag(Candidate("n/1", "", "peak", 0, 0, {})) == 0.0


# ---------------------------------------------------------------------------
# Spatial grid
# ---------------------------------------------------------------------------

def test_spatial_grid_finds_close_points_and_ignores_distant_ones():
    grid = _SpatialGrid(1.0, 51.44)
    grid.add(51.4400, -0.9563)
    assert grid.has_within(51.4405, -0.9560, 1.0) is True
    assert grid.has_within(51.9000, -1.9000, 1.0) is False


def test_spatial_grid_agrees_with_brute_force():
    import random
    random.seed(7)
    points = [(51.0 + random.random(), -1.5 + random.random()) for _ in range(400)]
    grid = _SpatialGrid(2.0, 51.44)
    for lat, lon in points:
        grid.add(lat, lon)
    for lat, lon in [(51.3, -1.2), (51.8, -0.9), (51.05, -1.45)]:
        brute = any(haversine_km(lat, lon, a, b) < 2.0 for a, b in points)
        assert grid.has_within(lat, lon, 2.0) == brute


# ---------------------------------------------------------------------------
# Water and spot quality
# ---------------------------------------------------------------------------

def test_water_distance_finds_the_nearest_feature():
    candidate = Candidate("node/1", "Test", "park", 51.44, -0.96)
    payload = {"elements": [
        {"center": {"lat": 51.45, "lon": -0.96}},
        {"center": {"lat": 51.90, "lon": -1.90}},
    ]}
    assert water_distance_km(candidate, payload) == pytest.approx(1.11, abs=0.1)


def test_water_distance_is_none_when_nothing_is_nearby():
    candidate = Candidate("node/1", "Test", "peak", 51.44, -0.96)
    assert water_distance_km(candidate, {"elements": []}) is None


def test_foreground_motion_falls_off_with_distance_from_water(config):
    assert foreground_motion_score(0.1, config) == 1.0
    assert foreground_motion_score(None, config) == 0.0
    assert foreground_motion_score(5.0, config) == 0.0
    mid = foreground_motion_score(0.9, config)
    assert 0.0 < mid < 1.0


def test_spot_quality_rewards_openness_and_prominence(config):
    good = spot_quality(1.0, 0.8, "viewpoint", config)
    poor = spot_quality(0.45, 0.05, "parking", config)
    assert good > poor
    assert 0.0 <= poor <= 100.0 and 0.0 <= good <= 100.0


# ---------------------------------------------------------------------------
# build_viewpoints filtering
# ---------------------------------------------------------------------------

def _profile(openness_full=True, elevation=200.0, mean=100.0, config=None):
    angles = {b: 0.0 for b in range(0, 360, 10)}
    if not openness_full:
        for b in range(200, 340, 10):
            angles[b] = 10.0
    return HorizonProfile(elevation_m=elevation, angles=angles, mean_elevation_5km=mean)


def test_flat_ground_is_rejected_even_though_it_looks_wide_open(config):
    """A municipal park in Reading scored horizon_openness 1.00 with no view at all.

    Openness measures terrain obstruction, and flat ground has none. Prominence is
    what separates "nothing blocks the view" from "there is a view".
    """
    # Placed well outside the minimum ride distance so this isolates flatness.
    flat = Candidate("node/1", "Flat Municipal Park", "park", 51.60, -1.20)
    profiles = {"node/1": _profile(elevation=50.0, mean=50.0)}
    records, rejections = build_viewpoints([flat], profiles, {"elements": []}, config)
    assert records == []
    assert any("flat" in reason for _, reason in rejections)


def test_a_spot_on_the_doorstep_is_not_a_ride(config):
    """This is a motorcycle app: 300 m is a walk.

    "Cintra Park", a municipal park 0.3 km from home with prominence 0.10, ranked
    ABOVE Walbury Hill on a clear evening because its distance discount was 1.0.
    """
    doorstep = Candidate("node/1", "Cintra Park", "park",
                         config.home.latitude + 0.002, config.home.longitude)
    profiles = {"node/1": _profile(elevation=50.0, mean=48.0)}
    water = {"elements": [{"center": {"lat": config.home.latitude + 0.002,
                                      "lon": config.home.longitude + 0.0005}}]}
    records, rejections = build_viewpoints([doorstep], profiles, water, config)
    assert records == []
    assert any("walk" in reason for _, reason in rejections)


def test_a_genuine_spot_just_beyond_the_doorstep_survives(config):
    """The rule must not swallow real close fallbacks."""
    from sunset_rider.geo import destination_point

    lat, lon = destination_point(config.home.latitude, config.home.longitude,
                                 270.0, float(config.discovery.min_distance_km) + 4.0)
    spot = Candidate("node/2", "River Meadow", "park", lat, lon)
    profiles = {"node/2": _profile(elevation=45.0, mean=45.0)}
    water = {"elements": [{"center": {"lat": lat + 0.0008, "lon": lon}}]}
    records, _ = build_viewpoints([spot], profiles, water, config)
    assert len(records) == 1
    assert records[0]["close_fallback"] is True


def test_a_real_hill_survives_the_filter(config):
    hill = Candidate("node/2", "Walbury Hill", "peak", 51.3525, -1.4650)
    profiles = {"node/2": _profile(elevation=297.0, mean=180.0)}
    records, _ = build_viewpoints([hill], profiles, {"elements": []}, config)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "Walbury Hill"
    assert record["horizon_openness"] > 0.4
    assert record["elevation_prominence"] > 0.12
    assert len(record["horizon_profile"]) == 36
    assert record["minutes_one_way"] > 0


def test_a_waterside_close_fallback_is_exempt_from_the_prominence_rule(config):
    """Reflection spots are low-lying by nature and earn their place another way."""
    lakeside = Candidate("node/3", "Dinton Pastures", "park", 51.45, -0.90)
    profiles = {"node/3": _profile(elevation=40.0, mean=40.0)}
    water = {"elements": [{"center": {"lat": 51.4505, "lon": -0.9005}}]}
    records, _ = build_viewpoints([lakeside], profiles, water, config)
    assert len(records) == 1
    assert records[0]["close_fallback"] is True
    assert records[0]["foreground_motion"] > 0.5


def test_an_enclosed_spot_is_rejected_for_openness(config):
    enclosed = Candidate("node/4", "In A Valley", "peak", 51.5, -1.2)
    profiles = {"node/4": _profile(openness_full=False, elevation=200.0, mean=100.0)}
    records, rejections = build_viewpoints([enclosed], profiles, {"elements": []}, config)
    assert records == []
    assert any("openness" in reason for _, reason in rejections)


def test_candidates_without_a_profile_are_skipped(config):
    candidate = Candidate("node/5", "Unprofiled", "peak", 51.5, -1.2)
    records, _ = build_viewpoints([candidate], {}, {"elements": []}, config)
    assert records == []
