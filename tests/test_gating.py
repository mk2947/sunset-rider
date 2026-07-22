"""Gating tests: hard blockers, gate closures, and DST-proof send windows.

The DST tests are the reason this logic lives in Python rather than in cron.
GitHub Actions cron is UTC-only, so a fixed UTC schedule silently drifts an hour
across the GMT/BST boundary — fatal for a sunset-timed job.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from sunset_rider.gating import (
    check_gate,
    due_modes,
    evaluate_blockers,
    in_confirm_window,
    in_go_window,
    in_plan_window,
    leave_by,
    returns_after_dark_minutes,
    target_date_for,
)
from sunset_rider.solar import SolarCalculator, UTC

LONDON = ZoneInfo("Europe/London")


@pytest.fixture()
def solar(config):
    return SolarCalculator(config)


@pytest.fixture()
def june_events(solar):
    return solar.events(dt.date(2025, 6, 21))


@pytest.fixture()
def december_events(solar):
    return solar.events(dt.date(2025, 12, 21))


# ---------------------------------------------------------------------------
# Hard blockers — each one individually, with a stated reason
# ---------------------------------------------------------------------------

def _clear_kwargs(events, config):
    return dict(max_gust_kmh=15.0, min_apparent_temp_c=16.0,
                max_precip_probability=10.0, min_visibility_m=25000.0,
                gate_closes=None, events=events, config=config)


def test_a_clear_evening_is_not_blocked(june_events, config):
    result = evaluate_blockers(**_clear_kwargs(june_events, config))
    assert result.blocked is False
    assert result.reasons == []


def test_high_gusts_block_with_a_reason(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["max_gust_kmh"] = 65.0
    result = evaluate_blockers(**kwargs)
    assert result.blocked is True
    assert any("gust" in r.lower() for r in result.reasons)


def test_freezing_temperature_blocks_with_a_reason(december_events, config):
    kwargs = _clear_kwargs(december_events, config)
    kwargs["min_apparent_temp_c"] = 1.0
    result = evaluate_blockers(**kwargs)
    assert result.blocked is True
    assert any("ice" in r.lower() or "feels like" in r.lower() for r in result.reasons)


def test_high_rain_probability_blocks_with_a_reason(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["max_precip_probability"] = 85.0
    result = evaluate_blockers(**kwargs)
    assert result.blocked is True
    assert any("rain" in r.lower() for r in result.reasons)


def test_fog_blocks_with_a_reason(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["min_visibility_m"] = 800.0
    result = evaluate_blockers(**kwargs)
    assert result.blocked is True
    assert any("visibility" in r.lower() or "fog" in r.lower() for r in result.reasons)


def test_gate_closing_at_sunset_blocks_with_a_reason(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["gate_closes"] = "sunset"
    result = evaluate_blockers(**kwargs)
    assert result.blocked is True
    assert any("gate" in r.lower() for r in result.reasons)


def test_blockers_are_reported_together_not_just_the_first(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["max_gust_kmh"] = 70.0
    kwargs["max_precip_probability"] = 90.0
    result = evaluate_blockers(**kwargs)
    assert len(result.reasons) == 2


def test_borderline_values_do_not_block(june_events, config):
    """Blockers are strict inequalities; exactly-at-threshold must pass."""
    kwargs = _clear_kwargs(june_events, config)
    kwargs["max_gust_kmh"] = float(config.blockers.max_gust_kmh)
    kwargs["max_precip_probability"] = float(config.blockers.max_precip_probability)
    kwargs["min_visibility_m"] = float(config.blockers.min_visibility_m)
    kwargs["min_apparent_temp_c"] = float(config.blockers.min_apparent_temp_c)
    assert evaluate_blockers(**kwargs).blocked is False


# ---------------------------------------------------------------------------
# Gate closing times
# ---------------------------------------------------------------------------

def test_gate_closing_at_sunset_or_dusk_is_always_blocking(june_events, config):
    for value in ("sunset", "dusk", "SUNSET"):
        blocked, reason, _ = check_gate(value, june_events, config)
        assert blocked is True, value
        assert reason


def test_a_spot_with_gate_closes_sunset_is_excluded_on_every_date(solar, config):
    """Every date of the year, not just the one we happened to test."""
    for month in range(1, 13):
        events = solar.events(dt.date(2025, month, 15))
        blocked, _, _ = check_gate("sunset", events, config)
        assert blocked is True, f"month {month}"


def test_no_gate_means_open_access(june_events, config):
    blocked, reason, warn = check_gate(None, june_events, config)
    assert blocked is False and reason is None and warn is False


def test_unknown_gate_is_allowed_but_flagged(june_events, config):
    blocked, _, warn = check_gate("unknown", june_events, config)
    assert blocked is False
    assert warn is True


def test_unknown_gate_produces_a_warning_on_the_spot(june_events, config):
    kwargs = _clear_kwargs(june_events, config)
    kwargs["gate_closes"] = "unknown"
    result = evaluate_blockers(**kwargs)
    assert result.blocked is False
    assert any("⚠️" in w for w in result.warnings)


def test_an_early_closing_time_blocks_in_june(june_events, config):
    """June sunset is 21:24; a 20:00 gate locks you in."""
    blocked, reason, _ = check_gate("20:00", june_events, config)
    assert blocked is True
    assert "20:00" in reason


def test_the_same_gate_is_fine_in_december(december_events, config):
    """December sunset is 15:57, so a 20:00 gate is no problem at all.

    The same spot can be excluded in June and perfectly usable in December, which is
    why this is evaluated per-date rather than baked into the viewpoint list.
    """
    blocked, _, _ = check_gate("20:00", december_events, config)
    assert blocked is False


def test_a_late_gate_is_fine_even_in_june(june_events, config):
    blocked, _, _ = check_gate("23:30", june_events, config)
    assert blocked is False


def test_an_unparseable_gate_value_is_treated_as_unknown(june_events, config):
    blocked, _, warn = check_gate("by arrangement", june_events, config)
    assert blocked is False
    assert warn is True


# ---------------------------------------------------------------------------
# Send windows
# ---------------------------------------------------------------------------

def _local(year, month, day, hour, minute=0):
    return dt.datetime(year, month, day, hour, minute, tzinfo=LONDON)


def test_plan_fires_on_sunday_and_wednesday_evenings(config):
    assert in_plan_window(_local(2025, 6, 22, 19, 30), config) is True   # Sunday
    assert in_plan_window(_local(2025, 6, 18, 19, 30), config) is True   # Wednesday
    assert in_plan_window(_local(2025, 6, 19, 19, 30), config) is False  # Thursday


def test_plan_window_respects_its_hours(config):
    assert in_plan_window(_local(2025, 6, 22, 18, 59), config) is False
    assert in_plan_window(_local(2025, 6, 22, 19, 0), config) is True
    assert in_plan_window(_local(2025, 6, 22, 19, 59), config) is True
    assert in_plan_window(_local(2025, 6, 22, 20, 0), config) is False


def test_confirm_window_is_daily(config):
    for day in range(16, 23):
        assert in_confirm_window(_local(2025, 6, day, 20, 0), config) is True
    assert in_confirm_window(_local(2025, 6, 18, 19, 0), config) is False
    assert in_confirm_window(_local(2025, 6, 18, 21, 0), config) is False


def test_send_windows_are_at_least_thirty_minutes_wide(config):
    """Scheduled Actions runs are routinely delayed 5-30 minutes under load.

    A narrow window would simply be missed, so this is a structural requirement.
    """
    def width(start, end):
        s = dt.datetime.strptime(start, "%H:%M")
        e = dt.datetime.strptime(end, "%H:%M")
        return (e - s).total_seconds() / 60.0

    assert width(config.schedule.plan.start, config.schedule.plan.end) >= 30
    assert width(config.schedule.confirm.start, config.schedule.confirm.end) >= 30
    go = config.schedule.go
    assert (float(go.max_minutes_before_sunset)
            - float(go.min_minutes_before_sunset)) >= 30


def test_go_window_is_anchored_to_sunset(config, solar):
    sunset = solar.sunset_utc(dt.date(2025, 6, 21))
    assert in_go_window(sunset - dt.timedelta(minutes=180), sunset, config) is True
    assert in_go_window(sunset - dt.timedelta(minutes=240), sunset, config) is False
    assert in_go_window(sunset - dt.timedelta(minutes=60), sunset, config) is False


def test_confirm_targets_tomorrow(config):
    now = _local(2025, 6, 18, 20, 0)
    assert target_date_for("confirm", now, config) == dt.date(2025, 6, 19)
    assert target_date_for("go", now, config) == dt.date(2025, 6, 18)


# ---------------------------------------------------------------------------
# DST — the reason cron alone is not enough
# ---------------------------------------------------------------------------

def _count_go_fires(date: dt.date, config, solar) -> int:
    """Simulate the hourly cron across one local day and count `go` triggers.

    The workflow runs at :17 past each hour, so that is what is simulated.
    """
    fires = 0
    start = dt.datetime.combine(date, dt.time(0, 17), tzinfo=LONDON)
    moment = start.astimezone(UTC)
    end = (dt.datetime.combine(date + dt.timedelta(days=1), dt.time(0, 17),
                               tzinfo=LONDON)).astimezone(UTC)
    sunset = solar.sunset_utc(date)
    while moment < end:
        if in_go_window(moment, sunset, config):
            fires += 1
        moment += dt.timedelta(hours=1)
    return fires


def test_exactly_one_go_per_day_across_the_spring_dst_boundary(config, solar):
    """30 Mar 2025: clocks go forward. The day is 23 hours long."""
    for offset in range(-2, 3):
        date = dt.date(2025, 3, 30) + dt.timedelta(days=offset)
        assert _count_go_fires(date, config, solar) == 1, f"{date} misfired"


def test_exactly_one_go_per_day_across_the_autumn_dst_boundary(config, solar):
    """26 Oct 2025: clocks go back. The day is 25 hours long and 01:00-02:00 repeats."""
    for offset in range(-2, 3):
        date = dt.date(2025, 10, 26) + dt.timedelta(days=offset)
        assert _count_go_fires(date, config, solar) == 1, f"{date} misfired"


def test_no_gaps_or_doubles_across_a_full_year(config, solar):
    """Every day of 2025 must fire the go window exactly once."""
    date = dt.date(2025, 1, 1)
    misfires = []
    while date <= dt.date(2025, 12, 31):
        count = _count_go_fires(date, config, solar)
        if count != 1:
            misfires.append((date.isoformat(), count))
        date += dt.timedelta(days=1)
    assert misfires == [], f"days that did not fire exactly once: {misfires[:10]}"


def test_due_modes_combines_windows(config, solar):
    """A Wednesday at 19:45 local is inside both the plan and confirm windows."""
    now_local = _local(2025, 6, 18, 19, 45)
    now_utc = now_local.astimezone(UTC)
    sunset = solar.sunset_utc(dt.date(2025, 6, 18))
    modes = due_modes(now_utc, sunset, config)
    assert "plan" in modes and "confirm" in modes


def test_due_modes_is_empty_at_a_quiet_hour(config, solar):
    now_utc = _local(2025, 6, 18, 3, 17).astimezone(UTC)
    sunset = solar.sunset_utc(dt.date(2025, 6, 18))
    assert due_modes(now_utc, sunset, config) == []


# ---------------------------------------------------------------------------
# LEAVE BY — the single most actionable field
# ---------------------------------------------------------------------------

def test_leave_by_subtracts_setup_time_and_ride_time(june_events, config):
    depart = leave_by(june_events, 48.0, config)
    expected = june_events.sunset - dt.timedelta(
        minutes=float(config.rider.setup_minutes) + 48.0)
    assert depart == expected


def test_leave_by_is_earlier_for_a_further_spot(june_events, config):
    near = leave_by(june_events, 15.0, config)
    far = leave_by(june_events, 60.0, config)
    assert far < near


def test_leave_by_is_before_sunset(june_events, config):
    assert leave_by(june_events, 30.0, config) < june_events.sunset


def test_return_after_dark_grows_with_distance(june_events):
    near = returns_after_dark_minutes(june_events, 10.0)
    far = returns_after_dark_minutes(june_events, 60.0)
    assert far > near > 0
