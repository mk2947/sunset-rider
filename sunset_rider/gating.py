"""Gating: hard blockers, gate-closing times, and DST-proof send windows.

Two separate jobs share this module because both are "should this happen at all"
decisions:

1. Per-spot blockers — is this spot rideable and reachable tonight at all?
2. Send-window gating — should this scheduled run send anything?

The send-window gate is deliberately implemented in Python rather than in cron.
GitHub Actions cron is UTC-only, so a fixed UTC schedule drifts by an hour across
the GMT/BST boundary, which is fatal for a sunset-timed job. Cron wakes us hourly;
this module decides whether to act.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from .config import Config
from .solar import SolarEvents

log = logging.getLogger(__name__)

UTC = dt.timezone.utc

# "20:00", "20:30" etc.
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


@dataclass
class BlockerResult:
    """Why a spot is or is not rideable tonight."""

    blocked: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_block(self, reason: str) -> None:
        self.blocked = True
        self.reasons.append(reason)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


def check_gate(gate_closes: str | None, events: SolarEvents,
               config: Config) -> tuple[bool, str | None, bool]:
    """Decide whether a car park gate rules this spot out tonight.

    Returns (blocked, reason, needs_warning).

    A large fraction of the best English viewpoints have car parks that lock at or
    before dusk, which makes them worthless for their single purpose. "unknown" is
    allowed through but flagged, because most roadside laybys are genuinely fine and
    excluding every unknown would throw away half the list.
    """
    b = config.blockers
    buffer_minutes = int(b.gate_buffer_minutes)
    latest_safe = events.sunset + dt.timedelta(minutes=buffer_minutes)

    if gate_closes is None:
        return False, None, False

    value = str(gate_closes).strip().lower()

    if value in {str(x).lower() for x in b.gate_closes_always_blocking}:
        return True, f"gate closes at {value} (need sunset+{buffer_minutes}min)", False

    if value == "unknown":
        return False, None, True

    match = _TIME_RE.match(value)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        local_sunset = events.local(events.sunset)
        closing_local = local_sunset.replace(hour=hour % 24, minute=minute,
                                             second=0, microsecond=0)
        if closing_local < latest_safe.astimezone(ZoneInfo(events.timezone)):
            return (True,
                    f"gate closes {value}, before sunset+{buffer_minutes}min "
                    f"({latest_safe.astimezone(ZoneInfo(events.timezone)):%H:%M})",
                    False)
        return False, None, False

    # Unparseable value: treat like unknown rather than silently blocking.
    return False, None, True


def evaluate_blockers(*, max_gust_kmh: float | None, min_apparent_temp_c: float | None,
                      max_precip_probability: float | None,
                      min_visibility_m: float | None,
                      gate_closes: str | None, events: SolarEvents,
                      config: Config) -> BlockerResult:
    """Apply every hard blocker. A blocked spot is dropped with the reason stated."""
    b = config.blockers
    result = BlockerResult(blocked=False)

    if max_gust_kmh is not None and max_gust_kmh > float(b.max_gust_kmh):
        result.add_block(
            f"gusts {max_gust_kmh:.0f} km/h > {float(b.max_gust_kmh):.0f} "
            f"(128 kg bike on exposed downland)"
        )
    if min_apparent_temp_c is not None and min_apparent_temp_c < float(b.min_apparent_temp_c):
        result.add_block(
            f"feels like {min_apparent_temp_c:.0f}°C < {float(b.min_apparent_temp_c):.0f}°C (ice risk)"
        )
    if (max_precip_probability is not None
            and max_precip_probability > float(b.max_precip_probability)):
        result.add_block(
            f"rain probability {max_precip_probability:.0f}% > {float(b.max_precip_probability):.0f}%"
        )
    if min_visibility_m is not None and min_visibility_m < float(b.min_visibility_m):
        result.add_block(f"visibility {min_visibility_m:.0f} m < {float(b.min_visibility_m):.0f} m (fog)")

    gate_blocked, gate_reason, gate_warn = check_gate(gate_closes, events, config)
    if gate_blocked and gate_reason:
        result.add_block(gate_reason)
    if gate_warn:
        result.add_warning("⚠️ gate closing time unknown — check before committing")

    return result


# ---------------------------------------------------------------------------
# Send windows
# ---------------------------------------------------------------------------

def _parse_hhmm(value: str) -> dt.time:
    match = _TIME_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"expected HH:MM, got {value!r}")
    return dt.time(hour=int(match.group(1)), minute=int(match.group(2)))


def in_plan_window(now_local: dt.datetime, config: Config) -> bool:
    """Sunday and Wednesday evenings, 19:00-20:00 local."""
    s = config.schedule.plan
    if now_local.weekday() not in [int(d) for d in s.weekdays]:
        return False
    return _parse_hhmm(s.start) <= now_local.time() < _parse_hhmm(s.end)


def in_confirm_window(now_local: dt.datetime, config: Config) -> bool:
    """Daily, 19:30-20:30 local."""
    s = config.schedule.confirm
    return _parse_hhmm(s.start) <= now_local.time() < _parse_hhmm(s.end)


def in_go_window(now_utc: dt.datetime, sunset_utc: dt.datetime, config: Config) -> bool:
    """Roughly 3 hours before today's sunset (a 60-minute half-open window).

    Anchored to sunset rather than the clock, so it is inherently DST-proof.

    Half-open on purpose. Cron ticks hourly, and exactly one tick falls inside any
    half-open 60-minute interval — no gaps, no doubles. A closed interval could
    catch a tick at each end on the days the arithmetic lines up exactly.
    """
    s = config.schedule.go
    minutes_before = (sunset_utc - now_utc.astimezone(UTC)).total_seconds() / 60.0
    return (float(s.min_minutes_before_sunset)
            <= minutes_before
            < float(s.max_minutes_before_sunset))


def due_modes(now_utc: dt.datetime, sunset_utc: dt.datetime,
              config: Config) -> list[str]:
    """Which modes, if any, this scheduled run should send.

    Cron decides whether to wake; this decides whether to act. Every window is at
    least 30 minutes wide because scheduled Actions runs are routinely delayed
    5-30 minutes under load, and a single-instant window would simply be missed.
    """
    now_local = now_utc.astimezone(ZoneInfo(config.home.timezone))
    modes: list[str] = []
    if in_plan_window(now_local, config):
        modes.append("plan")
    if in_confirm_window(now_local, config):
        modes.append("confirm")
    if in_go_window(now_utc, sunset_utc, config):
        modes.append("go")
    return modes


def target_date_for(mode: str, now_local: dt.datetime, config: Config) -> dt.date:
    """Which evening a given mode is talking about."""
    if mode == "confirm":
        return now_local.date() + dt.timedelta(
            days=int(config.schedule.confirm.target_offset_days))
    return now_local.date()


def leave_by(events: SolarEvents, minutes_one_way: float,
             config: Config) -> dt.datetime:
    """The single most actionable field in the whole system.

    sunset - setup_minutes - ride_time. Must never be absent from a `go` message.
    """
    return events.sunset - dt.timedelta(
        minutes=float(config.rider.setup_minutes) + float(minutes_one_way)
    )


def returns_after_dark_minutes(events: SolarEvents, minutes_one_way: float) -> float:
    """How long after sunset the return leg ends, for the night-riding penalty.

    Assumes the rider stays until the end of civil twilight, which is what the blue
    hour is for, then rides home.
    """
    end_of_shoot = events.civil_dusk
    home_again = end_of_shoot + dt.timedelta(minutes=float(minutes_one_way))
    return (home_again - events.sunset).total_seconds() / 60.0
