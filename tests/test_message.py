"""Message rendering tests.

The headline requirement: a `plan` message must never present a point score. At 72
hours, cloud-layer forecasts are among the least skilful model outputs, so a
confident number would be a lie. This is asserted by regex on the rendered text.
"""

from __future__ import annotations

import datetime as dt
import re

import pytest

from sunset_rider.gating import BlockerResult, leave_by
from sunset_rider.message import (
    BareScoreError,
    assert_no_bare_score,
    maps_link,
    ordinal,
    render,
    render_failure,
    truncate,
)
from sunset_rider.models import EveningOutlook, RunResult, SpotForecast
from sunset_rider.scoring import (
    combine_output,
    corridor_clearness,
    score_output,
    score_ride,
    score_sky,
    score_worth_it,
)
from sunset_rider.solar import SolarCalculator
from sunset_rider.weather import summarise_spread
from tests.test_scoring import minimal_day, moody_day, vivid_day


VIEWPOINT = {
    "id": "node_1", "name": "Walbury Hill", "kind": "peak",
    "latitude": 51.3525, "longitude": -1.4650, "elevation_m": 297.0,
    "open_arc": [280, 330], "horizon_openness": 0.93, "elevation_prominence": 0.62,
    "foreground_interest": 0.7, "foreground_motion": 0.2,
    "distance_km": 36.7, "road_distance_km": 47.7, "minutes_one_way": 59.6,
    "close_fallback": False, "gate_closes": None, "spot_score": 82.0,
}

CLOSE_FALLBACK = {
    **VIEWPOINT,
    "id": "node_2", "name": "Dinton Pastures", "open_arc": [240, 310],
    "distance_km": 5.4, "road_distance_km": 7.0, "minutes_one_way": 8.8,
    "close_fallback": True, "spot_score": 55.0, "gate_closes": "unknown",
    "foreground_motion": 1.0,
}


def _spot(config, viewpoint, inputs, date=dt.date(2025, 6, 18), gusts=25.0):
    solar = SolarCalculator(config)
    events = solar.events(date)
    sky = score_sky(inputs, config)
    corridor = corridor_clearness(inputs.corridor_low, config)
    output = score_output(
        sky=sky.sky, inputs=inputs, corridor=corridor,
        foreground_interest=viewpoint["foreground_interest"],
        foreground_motion=viewpoint["foreground_motion"],
        gusts_kmh=gusts, wind_500hpa=45.0,
        blue_hour_minutes=events.blue_hour_minutes, config=config,
    )
    ride = score_ride(max_gust_kmh=gusts, max_precip_prob=10.0,
                      apparent_temperature=16.0, precip_preceding_3h=0.0,
                      returns_after_dark_minutes=60.0, config=config)
    worth, parts = score_worth_it(
        sky=sky.sky, output_score=combine_output(output.photo, output.video, config),
        ride=ride.score, spot=viewpoint["spot_score"],
        sun_bearing=events.sun_bearing, open_arc=viewpoint["open_arc"],
        minutes_one_way=viewpoint["minutes_one_way"], blocked=False, config=config,
    )
    blockers = BlockerResult(blocked=False)
    if viewpoint.get("gate_closes") == "unknown":
        blockers.add_warning("⚠️ gate closing time unknown — check before committing")
    return SpotForecast(
        viewpoint=viewpoint, events=events, sky=sky, output=output, ride=ride,
        blockers=blockers, worth_it=worth, worth_it_parts=parts,
        leave_by=leave_by(events, viewpoint["minutes_one_way"], config),
        minutes_one_way=viewpoint["minutes_one_way"],
        max_gust_kmh=gusts, wind_500hpa=45.0,
    )


def _outlook(config, date, skies):
    solar = SolarCalculator(config)
    events = solar.events(date)
    threshold = 60.0
    above = sum(1 for s in skies if s >= threshold)
    return EveningOutlook(
        date=date, events=events, member_skies=skies,
        stats=summarise_spread(skies),
        probability_above_good=above / len(skies),
        mode_counts={"moody": len(skies) - 2, "vivid": 2},
        clearing_front_members=22, member_count=len(skies),
    )


# ---------------------------------------------------------------------------
# THE plan-mode rule
# ---------------------------------------------------------------------------

def test_plan_message_contains_no_bare_point_score(config):
    """§12: assert by regex on the rendered message."""
    result = RunResult(mode="plan", target_date=dt.date(2025, 6, 15))
    result.outlooks = [
        _outlook(config, dt.date(2025, 6, 16), [30.0] * 10 + [45.0] * 21),
        _outlook(config, dt.date(2025, 6, 17), [20.0] * 25 + [40.0] * 6),
        _outlook(config, dt.date(2025, 6, 18), [70.0] * 21 + [45.0] * 10),
        _outlook(config, dt.date(2025, 6, 19), [55.0] * 14 + [65.0] * 17),
        _outlook(config, dt.date(2025, 6, 20), [25.0] * 28 + [35.0] * 3),
    ]
    text = render(result, config)

    # No "84/100", no "score: 84", no "Walbury Hill — 82", no "Sky 88".
    assert not re.search(r"\d{1,3}\s*/\s*100", text)
    assert not re.search(r"(?i)\bscore[:\s]+\d", text)
    assert not re.search(r"[—–-]\s*\d{1,3}\s*(?:$|\n)", text)
    assert not re.search(r"(?i)\b(?:sky|photo|video|ride|spot)\s+\d{1,3}\b", text)
    # And the guard agrees.
    assert_no_bare_score(text)


def test_plan_message_reports_probability_and_spread_instead(config):
    result = RunResult(mode="plan", target_date=dt.date(2025, 6, 15))
    result.outlooks = [_outlook(config, dt.date(2025, 6, 18), [70.0] * 21 + [45.0] * 10)]
    text = render(result, config)
    assert "% of members above" in text
    assert "IQR" in text
    assert "least reliable" in text


def test_plan_names_the_dominant_mode_and_clearing_front(config):
    result = RunResult(mode="plan", target_date=dt.date(2025, 6, 15))
    result.outlooks = [_outlook(config, dt.date(2025, 6, 18), [70.0] * 21 + [45.0] * 10)]
    text = render(result, config)
    assert "MOODY" in text
    assert "clearing-front" in text.lower()


def test_the_guard_actually_catches_a_bare_score():
    """A guard that cannot fire is not a guard."""
    for bad in ["Thursday: 84/100", "score: 84", "Walbury Hill — 82\n", "Sky 88"]:
        with pytest.raises(BareScoreError):
            assert_no_bare_score(bad)


def test_the_guard_permits_honest_plan_language():
    ok = ("🥇 THU 24th   ~21:09 sunset\n"
          "   68% of members above \"good\" · IQR 54–81 (wide, it's 3 days out)\n"
          "   Clearing-front signature in 22 of 31 members.\n")
    assert_no_bare_score(ok)


def test_plan_with_no_data_still_renders(config):
    result = RunResult(mode="plan", target_date=dt.date(2025, 6, 15))
    text = render(result, config)
    assert "No ensemble data" in text


# ---------------------------------------------------------------------------
# go
# ---------------------------------------------------------------------------

def test_go_message_always_includes_leave_by(config):
    """LEAVE BY is the single most actionable field and must never be absent."""
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, moody_day())]
    text = render(result, config)
    assert "LEAVE BY" in text
    assert re.search(r"LEAVE BY \d{2}:\d{2}", text)


def test_go_message_reports_mode_photo_and_video(config):
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, moody_day())]
    text = render(result, config)
    assert "MOODY" in text
    assert "Photo" in text and "Video" in text
    assert "Sunset" in text and "golden" in text and "blue hour" in text


def test_go_message_announces_a_clearing_front(config):
    inputs = moody_day(total_cc=90.0, total_cc_after=[70.0, 50.0],
                       corridor_low=[5.0, 5.0, 0.0, 0.0])
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, inputs)]
    text = render(result, config)
    assert "CLEARING FRONT" in text


def test_go_message_includes_a_motorway_avoiding_link(config):
    """A CBT rider must never be routed onto the M4."""
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, moody_day())]
    text = render(result, config)
    assert "dirflg=h" in text
    assert "avoids motorways" in text


def test_go_message_flags_an_unknown_gate(config):
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, CLOSE_FALLBACK, moody_day())]
    text = render(result, config)
    assert "⚠️" in text and "gate" in text.lower()


def test_a_poor_evening_is_stated_plainly_with_a_fallback(config):
    """Do not dress up a bad forecast."""
    dull = vivid_day(total_cc=99.0, cloud_cover_low=95.0,
                     corridor_low=[95.0, 95.0, 95.0, 95.0], visibility_m=4000.0)
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [
        _spot(config, VIEWPOINT, dull),
        _spot(config, CLOSE_FALLBACK, dull),
    ]
    result.spots.sort(key=lambda s: -s.worth_it)
    text = render(result, config)
    assert "Not a great one" in text or "NO-GO" in text


def test_go_with_everything_blocked_says_so(config):
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    spot = _spot(config, VIEWPOINT, moody_day())
    spot.blockers.add_block("gusts 70 km/h > 60")
    result.spots = [spot]
    result.excluded = [("Walbury Hill", "gusts 70 km/h > 60")]
    text = render(result, config)
    assert "NO-GO" in text
    assert "gusts" in text.lower()


def test_go_message_ends_with_the_rating_prompt(config):
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, moody_day())]
    assert render(result, config).rstrip().endswith("/rate 1-5")


def test_go_reports_timelapse_when_flagged(config):
    inputs = moody_day(total_cc=50.0)
    result = RunResult(mode="go", target_date=dt.date(2025, 6, 18))
    result.spots = [_spot(config, VIEWPOINT, inputs, gusts=10.0)]
    text = render(result, config)
    assert result.spots[0].output.timelapse_flag is True
    assert "Timelapse" in text


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------

def test_confirm_message_names_the_top_spots_and_verdict(config):
    result = RunResult(mode="confirm", target_date=dt.date(2025, 6, 19))
    result.spots = [
        _spot(config, VIEWPOINT, moody_day(), date=dt.date(2025, 6, 19)),
        _spot(config, CLOSE_FALLBACK, moody_day(), date=dt.date(2025, 6, 19)),
    ]
    result.spots.sort(key=lambda s: -s.worth_it)
    text = render(result, config)
    assert "TOMORROW" in text
    assert "Walbury Hill" in text
    assert "leave by" in text.lower()
    assert any(v in text for v in ("📷", "🎥"))


def test_confirm_with_nothing_rideable(config):
    result = RunResult(mode="confirm", target_date=dt.date(2025, 6, 19))
    spot = _spot(config, VIEWPOINT, moody_day(), date=dt.date(2025, 6, 19))
    spot.blockers.add_block("fog")
    result.spots = [spot]
    text = render(result, config)
    assert "no rideable spot" in text.lower()


# ---------------------------------------------------------------------------
# Failure and utilities
# ---------------------------------------------------------------------------

def test_failure_message_explains_it_is_not_a_dull_sky(config):
    """Silence is the worst failure mode; so is an ambiguous message."""
    text = render_failure("go", "server returned HTTP 500", config)
    assert "unavailable" in text.lower()
    assert "HTTP 500" in text
    assert "not a dull evening" in text.lower()


def test_truncate_respects_the_telegram_limit(config):
    limit = int(config.telegram.max_message_chars)
    assert len(truncate("x" * (limit * 2), config)) <= limit
    assert truncate("short", config) == "short"


@pytest.mark.parametrize("day, expected", [(1, "1st"), (2, "2nd"), (3, "3rd"),
                                           (4, "4th"), (11, "11th"), (12, "12th"),
                                           (13, "13th"), (21, "21st"), (22, "22nd"),
                                           (23, "23rd"), (24, "24th"), (30, "30th")])
def test_ordinal_suffixes(day, expected):
    assert ordinal(day) == expected


def test_maps_link_points_at_the_spot_from_home(config):
    link = maps_link(51.3525, -1.4650, config)
    assert "51.352500,-1.465000" in link
    assert f"{config.home.latitude:.6f}" in link


def test_unknown_mode_is_rejected(config):
    with pytest.raises(ValueError, match="unknown mode"):
        render(RunResult(mode="nonsense", target_date=dt.date(2025, 6, 18)), config)
