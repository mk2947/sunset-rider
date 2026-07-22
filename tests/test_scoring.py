"""Scoring tests.

Every test here is capable of failing: the synthetic days are constructed so that
a broken formula changes the answer, not just the decimals.
"""

from __future__ import annotations

import pytest

from sunset_rider.scoring import (
    SkyInputs,
    azimuth_match,
    band_for,
    clamp,
    combine_output,
    corridor_clearness,
    detect_clearing_front,
    distance_discount,
    dynamic_range_term,
    gaussian,
    max_radius_km,
    score_output,
    score_ride,
    score_sky,
    score_worth_it,
)


# ---------------------------------------------------------------------------
# Synthetic days
# ---------------------------------------------------------------------------

def vivid_day(**overrides) -> SkyInputs:
    """Ideal fire sky: ~45% mid+high, clean air, clear corridor."""
    base = dict(
        total_cc=45.0, cloud_cover_low=0.0, cloud_cover_mid=25.0, cloud_cover_high=20.0,
        corridor_low=[0.0, 0.0, 0.0, 0.0], visibility_m=30000.0,
        relative_humidity=55.0, precipitation_probability=5.0, cape=0.0,
        total_cc_after=[45.0, 45.0],
    )
    base.update(overrides)
    return SkyInputs(**base)


def moody_day(**overrides) -> SkyInputs:
    """Heavy deck overhead, clear slot toward the horizon. The v1 blind spot."""
    base = dict(
        total_cc=90.0, cloud_cover_low=60.0, cloud_cover_mid=55.0, cloud_cover_high=20.0,
        corridor_low=[10.0, 20.0, 5.0, 5.0], visibility_m=15000.0,
        relative_humidity=80.0, precipitation_probability=20.0, cape=300.0,
        total_cc_after=[88.0, 87.0],
    )
    base.update(overrides)
    return SkyInputs(**base)


def minimal_day(**overrides) -> SkyInputs:
    """Near-cloudless clean gradient."""
    base = dict(
        total_cc=5.0, cloud_cover_low=0.0, cloud_cover_mid=2.0, cloud_cover_high=3.0,
        corridor_low=[0.0, 0.0, 0.0, 0.0], visibility_m=40000.0,
        relative_humidity=50.0, precipitation_probability=0.0, cape=0.0,
        total_cc_after=[5.0, 5.0],
    )
    base.update(overrides)
    return SkyInputs(**base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_clamp_and_gaussian():
    assert clamp(-1.0) == 0.0 and clamp(2.0) == 1.0 and clamp(0.5) == 0.5
    assert clamp(5.0, 0.0, 10.0) == 5.0
    assert gaussian(45.0, 45.0, 22.0) == pytest.approx(1.0)
    assert gaussian(45.0 + 22.0, 45.0, 22.0) == pytest.approx(0.6065, abs=1e-3)


def test_mhc_is_capped_at_100():
    inputs = vivid_day(cloud_cover_mid=70.0, cloud_cover_high=60.0)
    assert inputs.mhc == 100.0


# ---------------------------------------------------------------------------
# THE CORRIDOR — the test that proves it is actually wired up
# ---------------------------------------------------------------------------

def test_corridor_weights_are_distance_ordered(config):
    """Nearest obstruction must be weighted heaviest."""
    weights = [float(w) for w in config.corridor.weights]
    assert weights == sorted(weights, reverse=True)
    assert sum(weights) == pytest.approx(1.0)


def test_clear_corridor_scores_one_and_blocked_corridor_scores_zero(config):
    assert corridor_clearness([0, 0, 0, 0], config) == pytest.approx(1.0)
    assert corridor_clearness([100, 100, 100, 100], config) == pytest.approx(0.0)


def test_low_cloud_at_30km_hurts_more_than_at_250km(config):
    """A wall of cloud near the sun matters far more than one on the horizon."""
    near_blocked = corridor_clearness([100, 0, 0, 0], config)
    far_blocked = corridor_clearness([0, 0, 0, 100], config)
    assert near_blocked < far_blocked


def test_identical_overhead_conditions_score_lower_with_a_blocked_corridor(config):
    """THE corridor regression test.

    Same sky directly overhead, but one evening has a wall of low cloud 30 km
    toward the sun. If the corridor were not wired in, these would score the same.
    """
    clear = vivid_day(corridor_low=[0.0, 0.0, 0.0, 0.0])
    blocked = vivid_day(corridor_low=[100.0, 0.0, 0.0, 0.0])

    clear_result = score_sky(clear, config)
    blocked_result = score_sky(blocked, config)

    assert clear_result.vivid > blocked_result.vivid
    assert clear_result.vivid - blocked_result.vivid > 10.0, (
        "corridor is not materially affecting the score"
    )
    # Overhead conditions really were identical.
    assert clear.mhc == blocked.mhc
    assert clear.total_cc == blocked.total_cc


# ---------------------------------------------------------------------------
# MODE SEPARATION
# ---------------------------------------------------------------------------

def test_vivid_day_selects_vivid(config):
    result = score_sky(vivid_day(), config)
    assert result.best_mode == "vivid"
    assert result.vivid > result.moody and result.vivid > result.minimal


def test_moody_day_selects_moody(config):
    result = score_sky(moody_day(), config)
    assert result.best_mode == "moody"


def test_minimal_day_selects_minimal(config):
    result = score_sky(minimal_day(), config)
    assert result.best_mode == "minimal"


def test_heavy_cloud_with_clear_far_corridor_scores_moody_above_60(config):
    """REGRESSION TEST for the v1 bug where only fire skies scored.

    90% cloud with a clear corridor is a godray setup and often the best photograph
    of the month. v1 would have scored this near zero.
    """
    inputs = moody_day(total_cc=90.0, corridor_low=[10.0, 15.0, 5.0, 5.0])
    result = score_sky(inputs, config)
    assert result.moody > 60.0, f"moody scored only {result.moody:.1f}"
    assert result.best_mode == "moody"


def test_overcast_with_a_blocked_corridor_is_correctly_poor(config):
    """The genuinely bad evening: heavy cloud AND no slot. Must not score moody.

    Guards the no-slot cap. The four moody weights alone pay out 65% for any heavy
    sky (deck + texture + drama, the last with a 0.5 floor), so without the cap this
    evening scored 63 = GOOD despite 95% low cloud from 30-250 km toward the sun.
    """
    inputs = moody_day(corridor_low=[95.0, 95.0, 95.0, 95.0], total_cc_after=[90.0, 90.0])
    result = score_sky(inputs, config)
    assert result.moody <= float(config.scoring.moody.cap_no_slot_value)
    assert result.sky < 60.0
    assert band_for(result.sky, config)[1] == "POOR"


def test_the_no_slot_cap_does_not_touch_a_real_godray_evening(config):
    """The cap must only bite when the contrast has actually vanished."""
    real = score_sky(moody_day(), config)
    assert real.components["moody_slot"] > float(config.scoring.moody.cap_no_slot_below)
    assert real.moody > float(config.scoring.moody.cap_no_slot_value)


def test_moody_degrades_smoothly_as_the_slot_closes(config):
    """Widening low cloud near the sun must monotonically reduce the moody score."""
    scores = [
        score_sky(moody_day(corridor_low=[near, 20.0, 5.0, 5.0]), config).moody
        for near in (5.0, 30.0, 60.0, 90.0)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_mode_bias_shifts_the_winner(config):
    """config.mode_bias is the first dial to turn if the wrong aesthetic keeps winning."""
    inputs = vivid_day()
    neutral = score_sky(inputs, config)
    assert neutral.best_mode == "vivid"

    biased = config.as_dict()
    biased["scoring"]["mode_bias"]["minimal"] = 3.0
    from sunset_rider.config import Config
    assert score_sky(inputs, Config(biased)).best_mode == "minimal"


def test_runner_up_reported_only_when_close(config):
    close = score_sky(vivid_day(cloud_cover_mid=30.0, cloud_cover_high=25.0,
                                total_cc=55.0), config)
    if close.runner_up is not None:
        assert abs(close.sky - close.runner_up_score) <= float(
            config.scoring.runner_up_within)

    clear_winner = score_sky(minimal_day(), config)
    if clear_winner.runner_up is not None:
        assert abs(clear_winner.sky - clear_winner.runner_up_score) <= float(
            config.scoring.runner_up_within)


# ---------------------------------------------------------------------------
# VIVID caps
# ---------------------------------------------------------------------------

def test_low_cloud_overhead_caps_vivid(config):
    capped = score_sky(vivid_day(cloud_cover_low=90.0), config)
    assert capped.vivid <= float(config.scoring.vivid.cap_low_overhead_value)


def test_high_precipitation_probability_caps_vivid(config):
    capped = score_sky(vivid_day(precipitation_probability=80.0), config)
    assert capped.vivid <= float(config.scoring.vivid.cap_precip_prob_value)


def test_total_overcast_caps_vivid(config):
    capped = score_sky(vivid_day(total_cc=99.0), config)
    assert capped.vivid <= float(config.scoring.vivid.cap_total_cc_value)


# ---------------------------------------------------------------------------
# CLEARING FRONT
# ---------------------------------------------------------------------------

def test_falling_cloud_series_triggers_the_clearing_front_flag(config):
    inputs = moody_day(total_cc=90.0, total_cc_after=[75.0, 55.0],
                       corridor_low=[5.0, 5.0, 0.0, 0.0])
    assert detect_clearing_front(inputs, corridor_clearness(inputs.corridor_low, config),
                                 config) is True


def test_flat_cloud_series_does_not_trigger_the_flag(config):
    inputs = moody_day(total_cc=90.0, total_cc_after=[90.0, 89.0],
                       corridor_low=[5.0, 5.0, 0.0, 0.0])
    assert detect_clearing_front(inputs, corridor_clearness(inputs.corridor_low, config),
                                 config) is False


def test_clearing_front_needs_a_clear_corridor_too(config):
    """Cloud clearing overhead is worthless if the light cannot get under it."""
    inputs = moody_day(total_cc=90.0, total_cc_after=[70.0, 50.0],
                       corridor_low=[100.0, 100.0, 100.0, 100.0])
    assert detect_clearing_front(inputs, corridor_clearness(inputs.corridor_low, config),
                                 config) is False


def test_clearing_front_adds_the_bonus_and_is_capped(config):
    falling = moody_day(total_cc=90.0, total_cc_after=[70.0, 50.0],
                        corridor_low=[5.0, 5.0, 0.0, 0.0])
    flat = moody_day(total_cc=90.0, total_cc_after=[90.0, 90.0],
                     corridor_low=[5.0, 5.0, 0.0, 0.0])
    with_front = score_sky(falling, config)
    without = score_sky(flat, config)
    assert with_front.clearing_front is True
    assert without.clearing_front is False
    assert with_front.moody > without.moody
    assert with_front.moody <= float(config.scoring.moody.score_cap)


def test_missing_future_cloud_data_cannot_trigger_a_front(config):
    inputs = moody_day(total_cc_after=[])
    assert detect_clearing_front(inputs, 1.0, config) is False


# ---------------------------------------------------------------------------
# PHOTO vs VIDEO
# ---------------------------------------------------------------------------

def test_high_wind_makes_it_a_stills_evening_not_a_video_one(config):
    """50 km/h gusts with fast 500 hPa wind: photo must beat video by >=20."""
    inputs = moody_day()
    corridor = corridor_clearness(inputs.corridor_low, config)
    result = score_output(
        sky=70.0, inputs=inputs, corridor=corridor,
        foreground_interest=0.9, foreground_motion=0.1,
        gusts_kmh=50.0, wind_500hpa=60.0, blue_hour_minutes=20.0, config=config,
    )
    assert result.photo - result.video >= 20.0, (
        f"photo {result.photo:.1f} vs video {result.video:.1f}"
    )
    assert result.timelapse_flag is False, "gusts of 50 km/h must not flag a timelapse"
    assert result.verdict == "📷 stills evening"


def test_timelapse_flag_needs_fast_air_aloft_and_still_air_at_the_tripod(config):
    inputs = moody_day(total_cc=50.0)
    corridor = corridor_clearness(inputs.corridor_low, config)
    good = score_output(sky=70.0, inputs=inputs, corridor=corridor,
                        foreground_interest=0.5, foreground_motion=0.5,
                        gusts_kmh=15.0, wind_500hpa=60.0,
                        blue_hour_minutes=30.0, config=config)
    assert good.timelapse_flag is True

    # Same aloft, but gusty at ground level.
    assert score_output(sky=70.0, inputs=inputs, corridor=corridor,
                        foreground_interest=0.5, foreground_motion=0.5,
                        gusts_kmh=30.0, wind_500hpa=60.0,
                        blue_hour_minutes=30.0, config=config).timelapse_flag is False

    # Still at ground, but nothing moving aloft: no cloud motion to capture.
    assert score_output(sky=70.0, inputs=inputs, corridor=corridor,
                        foreground_interest=0.5, foreground_motion=0.5,
                        gusts_kmh=15.0, wind_500hpa=10.0,
                        blue_hour_minutes=30.0, config=config).timelapse_flag is False


def test_timelapse_needs_some_cloud_but_not_total_overcast(config):
    corridor = 0.9
    for total_cc, expected in [(5.0, False), (50.0, True), (95.0, False)]:
        inputs = moody_day(total_cc=total_cc)
        result = score_output(sky=70.0, inputs=inputs, corridor=corridor,
                              foreground_interest=0.5, foreground_motion=0.5,
                              gusts_kmh=15.0, wind_500hpa=60.0,
                              blue_hour_minutes=30.0, config=config)
        assert result.timelapse_flag is expected, f"total_cc={total_cc}"


def test_missing_upper_wind_cannot_flag_a_timelapse(config):
    inputs = moody_day(total_cc=50.0)
    result = score_output(sky=70.0, inputs=inputs, corridor=0.9,
                          foreground_interest=0.5, foreground_motion=0.5,
                          gusts_kmh=10.0, wind_500hpa=None,
                          blue_hour_minutes=30.0, config=config)
    assert result.timelapse_flag is False


def test_long_blue_hour_and_moving_water_favour_video(config):
    """A calm lakeside evening in June is the video case."""
    inputs = minimal_day()
    corridor = 1.0
    lake_calm = score_output(sky=70.0, inputs=inputs, corridor=corridor,
                             foreground_interest=0.2, foreground_motion=1.0,
                             gusts_kmh=5.0, wind_500hpa=20.0,
                             blue_hour_minutes=40.0, config=config)
    ridge_windy = score_output(sky=70.0, inputs=inputs, corridor=corridor,
                               foreground_interest=0.2, foreground_motion=0.0,
                               gusts_kmh=40.0, wind_500hpa=20.0,
                               blue_hour_minutes=12.0, config=config)
    assert lake_calm.video > ridge_windy.video


def test_verdict_reports_both_when_scores_are_close(config):
    inputs = vivid_day()
    result = score_output(sky=70.0, inputs=inputs, corridor=1.0,
                          foreground_interest=0.50, foreground_motion=0.48,
                          gusts_kmh=12.0, wind_500hpa=20.0,
                          blue_hour_minutes=20.0, config=config)
    assert abs(result.photo - result.video) < float(config.output.verdict_margin)
    assert result.verdict == "📷🎥 both"


def test_dynamic_range_rewards_bright_path_under_dark_cloud():
    """High DR is a photo bonus because stills can bracket; video has no such term."""
    heavy = SkyInputs(total_cc=90.0, cloud_cover_low=50.0, cloud_cover_mid=50.0,
                      cloud_cover_high=20.0, corridor_low=[0, 0, 0, 0],
                      visibility_m=20000.0, relative_humidity=70.0,
                      precipitation_probability=0.0, cape=0.0)
    clear = SkyInputs(total_cc=5.0, cloud_cover_low=0.0, cloud_cover_mid=2.0,
                      cloud_cover_high=3.0, corridor_low=[0, 0, 0, 0],
                      visibility_m=20000.0, relative_humidity=70.0,
                      precipitation_probability=0.0, cape=0.0)
    assert dynamic_range_term(heavy, 1.0) > dynamic_range_term(clear, 1.0)


def test_optimise_either_takes_the_better_of_the_two(config):
    assert combine_output(80.0, 40.0, config) == 80.0


# ---------------------------------------------------------------------------
# RIDE
# ---------------------------------------------------------------------------

def test_calm_dry_mild_evening_rides_well(config):
    result = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0,
                        apparent_temperature=18.0, precip_preceding_3h=0.0,
                        returns_after_dark_minutes=0.0, config=config)
    # The three weights deliberately sum to 0.90, so a perfect ride scores 90.
    assert result.score == pytest.approx(90.0, abs=0.1)
    assert result.night_penalty_applied is False


def test_wind_and_rain_reduce_the_ride_score(config):
    calm = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0, apparent_temperature=18.0,
                      precip_preceding_3h=0.0, returns_after_dark_minutes=0.0,
                      config=config).score
    windy = score_ride(max_gust_kmh=55.0, max_precip_prob=0.0, apparent_temperature=18.0,
                       precip_preceding_3h=0.0, returns_after_dark_minutes=0.0,
                       config=config).score
    wet = score_ride(max_gust_kmh=10.0, max_precip_prob=50.0, apparent_temperature=18.0,
                     precip_preceding_3h=0.0, returns_after_dark_minutes=0.0,
                     config=config).score
    assert windy < calm and wet < calm


def test_wet_road_applies_a_multiplier(config):
    dry = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0, apparent_temperature=18.0,
                     precip_preceding_3h=0.0, returns_after_dark_minutes=0.0,
                     config=config).score
    damp = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0, apparent_temperature=18.0,
                      precip_preceding_3h=1.2, returns_after_dark_minutes=0.0,
                      config=config).score
    assert damp == pytest.approx(dry * float(config.ride.dry_road_wet_multiplier), abs=0.1)


def test_night_riding_is_a_penalty_not_a_blocker(config):
    """The best light is often after the sun is down, so this must not zero the score."""
    day = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0, apparent_temperature=18.0,
                     precip_preceding_3h=0.0, returns_after_dark_minutes=0.0,
                     config=config)
    night = score_ride(max_gust_kmh=10.0, max_precip_prob=0.0, apparent_temperature=18.0,
                       precip_preceding_3h=0.0, returns_after_dark_minutes=120.0,
                       config=config)
    assert night.night_penalty_applied is True
    assert night.score == pytest.approx(day.score * float(config.rider.night_penalty),
                                        abs=0.1)
    assert night.score > 0.0


# ---------------------------------------------------------------------------
# AZIMUTH MATCH — where the horizon profile earns its keep
# ---------------------------------------------------------------------------

def test_sun_inside_the_open_arc_scores_full_marks(config):
    assert azimuth_match(310.0, (280, 330), config) == float(config.worth_it.azimuth_in_arc)


def test_sun_just_outside_the_arc_is_only_lightly_penalised(config):
    assert azimuth_match(340.0, (280, 330), config) == float(config.worth_it.azimuth_near_arc)


def test_sun_well_outside_the_arc_is_heavily_penalised(config):
    assert azimuth_match(200.0, (280, 330), config) == float(config.worth_it.azimuth_outside)


def test_a_spot_with_no_open_arc_is_penalised(config):
    assert azimuth_match(310.0, None, config) == float(config.worth_it.azimuth_outside)


def test_a_ridge_at_300_degrees_is_useless_in_june_and_fine_in_december(config):
    """The load-bearing insight of the whole project, as a test.

    A spot open only to the south-west scores full marks on the December azimuth
    (232 deg) and is penalised on the June one (311 deg).
    """
    south_west_only = (215, 265)
    june_bearing, december_bearing = 310.95, 231.61
    assert azimuth_match(december_bearing, south_west_only, config) == 1.0
    assert azimuth_match(june_bearing, south_west_only, config) < 1.0


# ---------------------------------------------------------------------------
# worth_it and dynamic radius
# ---------------------------------------------------------------------------

def test_distance_is_a_discount_never_a_filter(config):
    near = distance_discount(10.0, config)
    far = distance_discount(90.0, config)
    assert near > far
    assert far >= float(config.worth_it.distance_discount_floor)
    assert near <= float(config.worth_it.distance_discount_ceiling)


def test_distance_discount_never_falls_below_the_floor(config):
    assert distance_discount(1000.0, config) == float(config.worth_it.distance_discount_floor)


def test_a_hard_blocker_zeroes_worth_it(config):
    score, parts = score_worth_it(sky=95.0, output_score=95.0, ride=95.0, spot=95.0,
                                  sun_bearing=310.0, open_arc=(280, 330),
                                  minutes_one_way=20.0, blocked=True, config=config)
    assert score == 0.0
    assert parts["blocked"] is True


def test_worth_it_rewards_a_well_aligned_nearby_spot(config):
    aligned, _ = score_worth_it(sky=80.0, output_score=80.0, ride=80.0, spot=80.0,
                                sun_bearing=310.0, open_arc=(280, 330),
                                minutes_one_way=20.0, blocked=False, config=config)
    misaligned, _ = score_worth_it(sky=80.0, output_score=80.0, ride=80.0, spot=80.0,
                                   sun_bearing=200.0, open_arc=(280, 330),
                                   minutes_one_way=20.0, blocked=False, config=config)
    assert aligned > misaligned


def test_dynamic_radius_excludes_far_spots_on_a_dull_evening(config):
    """Regional sky 30 -> a 35-mile spot is out; sky 95 -> it is in."""
    thirty_five_miles_km = 35 * 1.609344
    assert max_radius_km(30.0, config) < thirty_five_miles_km
    assert max_radius_km(95.0, config) > thirty_five_miles_km


def test_dynamic_radius_spans_the_intended_range(config):
    assert max_radius_km(0.0, config) == pytest.approx(20.0, abs=0.1)
    assert max_radius_km(100.0, config) == pytest.approx(65.0, abs=0.1)
    assert max_radius_km(0.0, config) < max_radius_km(50.0, config) < max_radius_km(100.0, config)


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score, expected",
    [(95.0, "DROP EVERYTHING"), (80.0, "EXCELLENT"), (65.0, "GOOD"),
     (50.0, "DECENT — stay local"), (10.0, "POOR"), (0.0, "POOR")],
)
def test_bands_map_scores_to_labels(score, expected, config):
    _, label = band_for(score, config)
    assert label == expected


def test_band_boundaries_are_inclusive_at_the_minimum(config):
    assert band_for(90.0, config)[1] == "DROP EVERYTHING"
    assert band_for(89.99, config)[1] == "EXCELLENT"


# ---------------------------------------------------------------------------
# plan-mode degradation (no visibility on the ensemble API)
# ---------------------------------------------------------------------------

def test_missing_visibility_falls_back_to_the_neutral_term(config):
    """The ensemble API carries no visibility, so plan mode must still score."""
    inputs = vivid_day(visibility_m=None)
    result = score_sky(inputs, config)
    assert result.vivid > 0.0
    assert result.minimal > 0.0


def test_neutral_visibility_sits_between_the_extremes(config):
    good = score_sky(vivid_day(visibility_m=40000.0), config).vivid
    neutral = score_sky(vivid_day(visibility_m=None), config).vivid
    poor = score_sky(vivid_day(visibility_m=1000.0), config).vivid
    assert poor < neutral < good


def test_corridor_rejects_a_short_sample_list(config):
    with pytest.raises(ValueError, match="corridor needs"):
        corridor_clearness([0.0, 0.0], config)
