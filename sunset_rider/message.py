"""Message rendering: three modes, three shapes.

The rule that matters most here is in ``render_plan``. At 72 hours, cloud-layer
forecasts are among the least skilful model outputs, and they are exactly what this
system scores on. Presenting "Thursday: 84/100" three days out would be a lie. So
``plan`` mode is structurally forbidden from printing a bare point score, and
``assert_no_bare_score`` enforces it rather than trusting the template.

What plan mode may say: relative ranking, the share of ensemble members clearing a
band, and the interquartile range. What it may not say: any single number presented
as the score of an evening.
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import re
from typing import Sequence

from .config import Config
from .models import EveningOutlook, RunResult, SpotForecast
from .scoring import band_for

log = logging.getLogger(__name__)


class BareScoreError(AssertionError):
    """Raised when a plan-mode message contains a point score."""


# Patterns that would constitute presenting a confident point score. These match the
# exact renderings the confirm/go templates use, which is what makes the guard
# meaningful: all message text is generated here, so nothing else can produce them.
_BARE_SCORE_PATTERNS = (
    re.compile(r"\b\d{1,3}\s*/\s*100\b"),                 # "84/100"
    re.compile(r"\bscore[:\s]+\d{1,3}\b", re.IGNORECASE),  # "score: 84"
    re.compile(r"[—–-]\s*\d{1,3}\s*(?:$|\n)"),             # "Walbury Hill — 82"
    re.compile(r"\b(?:sky|photo|video|ride|spot)\s+\d{1,3}\b", re.IGNORECASE),
)


def assert_no_bare_score(text: str) -> None:
    """Guard for plan mode. Raises rather than quietly shipping a false confidence."""
    for pattern in _BARE_SCORE_PATTERNS:
        match = pattern.search(text)
        if match:
            raise BareScoreError(
                f"plan-mode message contains a bare point score: {match.group(0)!r}. "
                f"At 72 hours only rankings, probabilities and spreads are honest."
            )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return html.escape(str(text), quote=False)


def ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def _hhmm(moment: dt.datetime, events) -> str:
    return events.local(moment).strftime("%H:%M")


def mode_label(mode: str, config: Config) -> tuple[str, str]:
    display = config.mode_display[mode]
    return display.emoji, display.description


def maps_link(latitude: float, longitude: float, config: Config) -> str:
    """Directions link that avoids motorways.

    Uses the legacy maps URL form because ``dirflg=h`` (avoid highways) has no
    equivalent in the newer ``?api=1`` format, and a CBT rider must never be routed
    onto the M4. Listed in the README as behaviour I could not fully verify.
    """
    origin = f"{config.home.latitude:.6f},{config.home.longitude:.6f}"
    destination = f"{latitude:.6f},{longitude:.6f}"
    return (f"https://maps.google.com/maps?saddr={origin}&daddr={destination}"
            f"&dirflg=h")


def _gate_warning(spot: SpotForecast) -> str | None:
    for warning in spot.blockers.warnings:
        if "gate" in warning.lower():
            return warning
    return None


# ---------------------------------------------------------------------------
# plan (T-72h → T-48h)
# ---------------------------------------------------------------------------

_MEDAL = ["🥇", "🥈", "🥉"]


def render_plan(result: RunResult, config: Config) -> str:
    """Relative ranking of the next five evenings. No point scores, ever."""
    lines = ["🗓 NEXT 5 EVENINGS — planning view", ""]

    # An evening with no members is missing data, not a bad forecast. Reporting it
    # as "below 30%" would be indistinguishable from a genuinely dull evening.
    covered = [o for o in result.outlooks if o.member_count > 0]
    uncovered = [o for o in result.outlooks if o.member_count == 0]

    ranked = sorted(covered, key=lambda o: -o.probability_above_good)
    if not ranked:
        lines.append("No ensemble data available for the coming week.")
        text = "\n".join(lines)
        assert_no_bare_score(text)
        return text

    m = config.message
    threshold = float(m.plan_strong_threshold)
    strong = [o for o in ranked if o.probability_above_good >= threshold]
    weak = [o for o in ranked if o.probability_above_good < threshold]

    for index, outlook in enumerate(strong[:int(m.plan_max_ranked)]):
        medal = _MEDAL[index] if index < len(_MEDAL) else "•"
        day = outlook.events.local(outlook.events.sunset)
        emoji, _ = mode_label(outlook.dominant_mode, config)
        lines.append(
            f"{medal} {day.strftime('%a').upper()} {ordinal(day.day)}   "
            f"~{day.strftime('%H:%M')} sunset"
        )
        if index == 0:
            lines.append(f"   Strongest evening of the {len(result.outlooks)}. "
                         f"Ensemble leans {outlook.dominant_mode.upper()} {emoji}.")
        else:
            lines.append(f"   Leaning {outlook.dominant_mode.upper()} {emoji}.")

        if outlook.clearing_front_members:
            lines.append(
                f"   ⚡ Clearing-front signature in {outlook.clearing_front_members} "
                f"of {outlook.member_count} members."
            )

        pct = round(outlook.probability_above_good * 100)
        q1 = round(outlook.stats["q1"])
        q3 = round(outlook.stats["q3"])
        spread = "tight" if outlook.stats["iqr"] < float(m.plan_tight_iqr) else "wide"
        days = (outlook.date - result.target_date).days
        lines.append(
            f"   {pct}% of members above \"good\" · "
            f"IQR {q1}–{q3} ({spread}, it's {days} day{'' if days == 1 else 's'} out)"
        )
        if index == 0:
            lines.append("   → Worth holding the evening. I'll confirm the night before.")
        lines.append("")

    if weak:
        names = " / ".join(
            o.events.local(o.events.sunset).strftime("%a").upper() for o in weak
        )
        prefix = "both" if len(weak) == 2 else ("all" if len(weak) > 2 else "")
        lines.append(f"😐 {names} — {prefix + ' ' if prefix else ''}"
                     f"below {threshold * 100:.0f}% above \"good\"")
        lines.append("")

    if uncovered:
        names = " / ".join(
            o.events.local(o.events.sunset).strftime("%a").upper() for o in uncovered
        )
        lines.append(f"❔ {names} — beyond the ensemble's range, no data yet")
        lines.append("")

    lines.append("⚠️ 3-day cloud forecasts are the least reliable part of any model.")
    lines.append("   Treat this as \"which evening to protect\", not \"what will happen\".")
    lines.append(f"   Basis: {ranked[0].member_count} ECMWF ensemble members.")

    text = "\n".join(lines)
    # Structural guarantee, not a hope.
    assert_no_bare_score(text)
    return text


# ---------------------------------------------------------------------------
# confirm (T-24h)
# ---------------------------------------------------------------------------

def render_confirm(result: RunResult, config: Config) -> str:
    rideable = [s for s in result.spots if not s.blocked]
    day = result.target_date
    header_date = day.strftime("%a ") + ordinal(day.day)

    if not rideable:
        lines = [f"🚫 {header_date} — no rideable spot tomorrow.", ""]
        lines.extend(_render_exclusions(result, config))
        return "\n".join(lines)

    best = rideable[0]
    emoji, description = mode_label(best.sky.best_mode, config)
    band_emoji, band_label = band_for(best.worth_it, config)

    lines = [
        f"📋 TOMORROW — {header_date}",
        f"{band_emoji} {band_label} · {best.worth_it:.0f} · "
        f"{best.sky.best_mode.upper()} {emoji} · {best.output.verdict}",
        f"Sunset {_hhmm(best.events.sunset, best.events)} · "
        f"golden {_hhmm(best.events.golden_hour_start, best.events)} · "
        f"blue hour to {_hhmm(best.events.blue_hour_end, best.events)}",
        "",
        description.capitalize() + ".",
        "",
    ]

    if best.sky.runner_up:
        lines.append(
            f"↔️ Could go either way — {best.sky.runner_up.upper()} is within "
            f"{float(config.scoring.runner_up_within):.0f} points."
        )
        lines.append("")

    lines.append("Top spots:")
    for index, spot in enumerate(rideable[:int(config.message.confirm_max_spots)]):
        medal = _MEDAL[index] if index < len(_MEDAL) else "•"
        lines.append(
            f"{medal} {spot.name} — {spot.worth_it:.0f} · "
            f"{spot.minutes_one_way:.0f} min · "
            f"leave by {_hhmm(spot.leave_by, spot.events)}"
        )
        gate = _gate_warning(spot)
        if gate:
            lines.append(f"   {gate}")

    lines.append("")
    lines.append("The plan is on. Final go/no-go about 3 hours before sunset.")
    lines.extend(_render_exclusions(result, config))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# go (T-3h)
# ---------------------------------------------------------------------------

def render_go(result: RunResult, config: Config) -> str:
    rideable = [s for s in result.spots if not s.blocked]

    if not rideable:
        lines = ["🚫 NO-GO tonight.", ""]
        lines.extend(_render_exclusions(result, config))
        lines.append("")
        lines.append("Not dressing this up: nowhere is worth the ride tonight.")
        return "\n".join(lines)

    best = rideable[0]
    events = best.events
    emoji, description = mode_label(best.sky.best_mode, config)
    band_emoji, band_label = band_for(best.worth_it, config)

    lines = [
        f"{band_emoji} {band_label} — {best.worth_it:.0f} · "
        f"{best.sky.best_mode.upper()} {emoji} · {best.output.verdict}",
        f"Sunset {_hhmm(events.sunset, events)} · "
        f"golden {_hhmm(events.golden_hour_start, events)} · "
        f"blue hour to {_hhmm(events.blue_hour_end, events)} "
        f"({events.blue_hour_minutes:.0f} min usable)",
        "",
    ]

    # A poor evening should be said plainly, with the 15-minute fallback named.
    if band_label == "POOR" or band_label.startswith("DECENT"):
        fallback = next((s for s in rideable if s.viewpoint.get("close_fallback")), None)
        lines.append("Not a great one — no point pretending otherwise.")
        if fallback is not None and fallback is not best:
            lines.append(
                f"If you go, make it the short hop: {fallback.name}, "
                f"{fallback.minutes_one_way:.0f} min."
            )
        lines.append("")

    for index, spot in enumerate(rideable[:int(config.message.go_max_spots)]):
        medal = _MEDAL[index] if index < len(_MEDAL) else "•"
        if index == 0:
            lines.extend(_render_primary_spot(medal, spot, config))
        else:
            lines.append(
                f"{medal} {spot.name} — {spot.worth_it:.0f} · "
                f"{spot.minutes_one_way:.0f} min · "
                f"leave by {_hhmm(spot.leave_by, spot.events)}"
            )
    lines.append("")

    if best.sky.runner_up:
        lines.append(
            f"↔️ Sky could go {best.sky.runner_up.upper()} instead — within "
            f"{float(config.scoring.runner_up_within):.0f} points."
        )

    if best.output.timelapse_flag:
        lines.append("⏱ Timelapse conditions: fast cloud aloft, still air at the tripod.")

    lines.extend(_render_exclusions(result, config))
    lines.append("")
    lines.append("Rate it later: /rate 1-5")
    return "\n".join(lines)


def _render_primary_spot(medal: str, spot: SpotForecast, config: Config) -> list[str]:
    events = spot.events
    viewpoint = spot.viewpoint
    lines = [
        f"{medal} {spot.name} — {spot.worth_it:.0f}",
        f"   {spot.minutes_one_way:.0f} min · {viewpoint.get('road_distance_km', 0):.0f} km · "
        f"🏍 LEAVE BY {_hhmm(spot.leave_by, events)}",
        f"   Sky {spot.sky.sky:.0f} ({spot.sky.best_mode}) · "
        f"Photo {spot.output.photo:.0f} · Video {spot.output.video:.0f} · "
        f"Ride {spot.ride.score:.0f} · Spot {viewpoint.get('spot_score', 0):.0f}",
    ]

    if spot.sky.clearing_front:
        lines.append("   ⚡ CLEARING FRONT detected. This is the good one.")

    arc = viewpoint.get("open_arc")
    bearing = events.sun_bearing
    if arc:
        inside = spot.worth_it_parts.get("azimuth_match", 0) >= float(
            config.worth_it.azimuth_in_arc)
        if inside:
            lines.append(
                f"   Sun sets on {bearing:.0f}°, inside the open arc "
                f"{arc[0]}–{arc[1]}°."
            )
        else:
            lines.append(
                f"   Sun sets on {bearing:.0f}°, outside the open arc "
                f"{arc[0]}–{arc[1]}° — partially blocked."
            )

    if spot.max_gust_kmh:
        if spot.output.video < spot.output.photo:
            lines.append(
                f"   Gusts {spot.max_gust_kmh:.0f} km/h — video will suffer, shoot stills."
            )
        else:
            lines.append(f"   Gusts {spot.max_gust_kmh:.0f} km/h.")

    if spot.ride.night_penalty_applied:
        lines.append("   🌙 Late return — ride home will be in the dark.")

    gate = _gate_warning(spot)
    if gate:
        lines.append(f"   {gate}")

    for note in spot.notes:
        lines.append(f"   {note}")

    link = maps_link(float(viewpoint["latitude"]), float(viewpoint["longitude"]), config)
    lines.append(f'   📍 <a href="{_esc(link)}">directions (avoids motorways)</a>')
    return lines


def _render_exclusions(result: RunResult, config: Config) -> list[str]:
    if not result.excluded:
        return []
    limit = int(config.message.max_exclusions_listed)
    lines = ["", "Excluded tonight:"]
    for name, reason in result.excluded[:limit]:
        lines.append(f"  · {name} — {reason}")
    if len(result.excluded) > limit:
        lines.append(f"  · …and {len(result.excluded) - limit} more")
    return lines


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

def render_failure(mode: str, reason: str, config: Config) -> str:
    """Silence is the worst failure mode.

    The rider must never be left guessing whether it was a dull sky or a dead script.
    """
    return (
        f"⚠️ Sunset forecast unavailable ({mode} run)\n"
        f"\n"
        f"Reason: {reason}\n"
        f"\n"
        f"This is a script or API problem, not a dull evening. "
        f"The next scheduled run will try again."
    )


def truncate(text: str, config: Config) -> str:
    """Telegram rejects messages over 4096 characters."""
    limit = int(config.telegram.max_message_chars)
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n… (truncated)"


def render(result: RunResult, config: Config) -> str:
    renderers = {"plan": render_plan, "confirm": render_confirm, "go": render_go}
    if result.mode not in renderers:
        raise ValueError(f"unknown mode {result.mode!r}")
    return truncate(renderers[result.mode](result, config), config)
