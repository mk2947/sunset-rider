"""Scoring: three aesthetic modes, a photo/video split, ride safety and worth_it.

The design change that matters most here is that "cloudy" is not "bad". v1 optimised
for one thing — the classic vivid fire sky — and would have scored a heavy overcast
evening with a clear slot at the horizon near zero. That evening is often the best
photograph of the month. So three independent mode scores are computed and the
winner is reported by name.

Every constant comes from config.yaml. Nothing here is a literal.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence

from .config import Config
from .geo import angular_difference, bearing_in_arc

log = logging.getLogger(__name__)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def gaussian(value: float, mu: float, sigma: float) -> float:
    """Unit-height Gaussian — 1.0 at the peak, falling off either side."""
    return math.exp(-((value - mu) ** 2) / (2.0 * sigma ** 2))


@dataclass
class SkyInputs:
    """Weather sampled at the viewpoint and along the corridor to the setting sun.

    ``corridor_low`` holds low-cloud percentages at the configured corridor
    distances (30/80/150/250 km by default), ordered nearest first.
    """

    total_cc: float
    cloud_cover_low: float
    cloud_cover_mid: float
    cloud_cover_high: float
    corridor_low: Sequence[float]
    visibility_m: float | None
    relative_humidity: float
    precipitation_probability: float
    cape: float
    # Total cloud at the viewpoint at +1h and +2h after sunset, for clearing-front detection.
    total_cc_after: Sequence[float] = field(default_factory=tuple)

    @property
    def mhc(self) -> float:
        """Combined mid + high cloud, capped at 100 — the canvas the light paints on."""
        return min(100.0, self.cloud_cover_mid + self.cloud_cover_high)


@dataclass
class ModeResult:
    """Outcome of the three-mode comparison."""

    vivid: float
    moody: float
    minimal: float
    weighted: dict[str, float]
    best_mode: str
    sky: float
    runner_up: str | None
    runner_up_score: float | None
    clearing_front: bool
    corridor: float
    components: dict[str, float]


# ---------------------------------------------------------------------------
# Corridor
# ---------------------------------------------------------------------------

def corridor_clearness(corridor_low: Sequence[float], config: Config) -> float:
    """Distance-weighted clear fraction along the sun-to-sky corridor.

    The warm light has to travel *under* the cloud deck from the horizon to reach
    the clouds above you. A perfect canvas overhead is worthless if there is a wall
    of low cloud 80 km toward the sun. Nearest obstruction is weighted heaviest.
    """
    weights = [float(w) for w in config.corridor.weights]
    values = list(corridor_low)[:len(weights)]
    if len(values) < len(weights):
        raise ValueError(
            f"corridor needs {len(weights)} samples, got {len(values)}"
        )
    return sum(w * (1.0 - cl / 100.0) for w, cl in zip(weights, values))


# ---------------------------------------------------------------------------
# Mode 1: VIVID
# ---------------------------------------------------------------------------

def score_vivid(inputs: SkyInputs, corridor: float, config: Config,
                *, neutral_visibility: float | None = None) -> tuple[float, dict]:
    """Classic fire sky. Peak quality near 45% combined mid+high cover."""
    v = config.scoring.vivid
    canvas = gaussian(inputs.mhc, float(v.canvas_peak_mhc), float(v.canvas_sigma))

    if inputs.visibility_m is None:
        # plan mode: the ensemble API does not carry visibility at all, so the
        # visibility half of clarity is held neutral rather than invented.
        visibility_term = (neutral_visibility
                           if neutral_visibility is not None
                           else float(config.weather.ensemble_neutral_visibility_term))
    else:
        visibility_term = clamp(inputs.visibility_m / float(v.clarity_visibility_full_m))

    humidity_term = gaussian(inputs.relative_humidity, float(v.clarity_rh_peak),
                             float(v.clarity_rh_sigma))
    clarity = (float(v.clarity_visibility_weight) * visibility_term
               + float(v.clarity_humidity_weight) * humidity_term)

    score = 100.0 * (float(v.weight_canvas) * canvas
                     + float(v.weight_corridor) * corridor
                     + float(v.weight_clarity) * clarity)

    caps: list[str] = []
    if inputs.cloud_cover_low > float(v.cap_low_overhead_above):
        score = min(score, float(v.cap_low_overhead_value))
        caps.append("low cloud overhead")
    if inputs.precipitation_probability > float(v.cap_precip_prob_above):
        score = min(score, float(v.cap_precip_prob_value))
        caps.append("rain likely")
    if inputs.total_cc > float(v.cap_total_cc_above):
        score = min(score, float(v.cap_total_cc_value))
        caps.append("total overcast")

    return score, {"canvas": canvas, "clarity": clarity, "corridor": corridor,
                   "caps": caps}


# ---------------------------------------------------------------------------
# Mode 2: MOODY — the one v1 got wrong
# ---------------------------------------------------------------------------

def detect_clearing_front(inputs: SkyInputs, corridor: float,
                          config: Config) -> bool:
    """Heavy cloud overhead, edge of the deck approaching, sun dropping into the gap.

    The single best moody setup, and the evening worth cancelling plans for.
    Detected as total cloud at the viewpoint falling by more than the configured
    number of percentage points over the two hours after sunset, while the corridor
    is already clear.
    """
    m = config.scoring.moody
    horizon = int(m.clearing_front_hours_after_sunset)
    later = list(inputs.total_cc_after)[:horizon]
    if len(later) < horizon:
        return False
    if corridor < float(m.clearing_front_corridor_clear_min):
        return False
    drop = inputs.total_cc - min(later)
    return drop > float(m.clearing_front_drop_pp)


def score_moody(inputs: SkyInputs, corridor: float, config: Config
                ) -> tuple[float, bool, dict]:
    """Heavy cloud plus a clear slot at the horizon: godrays and spotlit landscape.

    The detector is a *contrast* between overhead and corridor, not an absolute
    cloud figure. This is what makes a 90%-cloud evening scoreable.
    """
    m = config.scoring.moody
    c = config.corridor

    deck = clamp((inputs.total_cc - float(m.deck_floor)) / float(m.deck_span))

    far_indices = [int(i) for i in c.far_indices]
    corridor_far = sum(1.0 - inputs.corridor_low[i] / 100.0 for i in far_indices) / len(far_indices)
    corridor_near_block = inputs.corridor_low[int(c.near_index)] / 100.0
    slot = clamp(corridor_far - corridor_near_block)

    texture = clamp(1.0 - abs(inputs.cloud_cover_mid - float(m.texture_peak_mid))
                    / float(m.texture_span))
    drama = (clamp(inputs.cape / float(m.drama_cape_full)) * float(m.drama_cape_weight)
             + float(m.drama_base))
    rain_ok = (1.0 if inputs.precipitation_probability < float(m.rain_ok_threshold)
               else float(m.rain_ok_penalty))

    score = 100.0 * (float(m.weight_deck) * deck
                     + float(m.weight_slot) * slot
                     + float(m.weight_texture) * texture
                     + float(m.weight_drama) * drama) * rain_ok

    # No slot, no moody. The weighted sum alone pays out 65% for any heavy sky
    # (deck + texture + drama, the last with a 0.5 floor), so a fully socked-in
    # evening scored 63 without this guard. Moody is defined by the CONTRAST
    # between a heavy deck and a clear horizon; remove the contrast and there is
    # nothing to photograph.
    capped_no_slot = slot < float(m.cap_no_slot_below)
    if capped_no_slot:
        score = min(score, float(m.cap_no_slot_value))

    clearing = detect_clearing_front(inputs, corridor, config)
    if clearing:
        score = min(score + float(m.clearing_front_bonus), float(m.score_cap))

    return score, clearing, {"deck": deck, "slot": slot, "texture": texture,
                             "drama": drama, "rain_ok": rain_ok,
                             "corridor_far": corridor_far,
                             "corridor_near_block": corridor_near_block,
                             "capped_no_slot": float(capped_no_slot)}


# ---------------------------------------------------------------------------
# Mode 3: MINIMAL
# ---------------------------------------------------------------------------

def score_minimal(inputs: SkyInputs, corridor: float, config: Config,
                  *, neutral_visibility: float | None = None) -> tuple[float, dict]:
    """Near-cloudless clean gradient. Weak for stills, often excellent for video."""
    mi = config.scoring.minimal
    cleanliness = clamp(1.0 - inputs.total_cc / float(mi.cleanliness_cc_full))
    if inputs.visibility_m is None:
        air = (neutral_visibility if neutral_visibility is not None
               else float(config.weather.ensemble_neutral_visibility_term))
    else:
        air = clamp(inputs.visibility_m / float(mi.air_visibility_full_m))

    score = 100.0 * (float(mi.weight_cleanliness) * cleanliness
                     + float(mi.weight_air) * air
                     + float(mi.weight_corridor) * corridor)
    return score, {"cleanliness": cleanliness, "air": air, "corridor": corridor}


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def score_sky(inputs: SkyInputs, config: Config,
              *, neutral_visibility: float | None = None) -> ModeResult:
    """Compute all three modes and pick the winner by taste-weighted score."""
    corridor = corridor_clearness(inputs.corridor_low, config)

    vivid, vivid_parts = score_vivid(inputs, corridor, config,
                                     neutral_visibility=neutral_visibility)
    moody, clearing, moody_parts = score_moody(inputs, corridor, config)
    minimal, minimal_parts = score_minimal(inputs, corridor, config,
                                           neutral_visibility=neutral_visibility)

    bias = config.scoring.mode_bias
    raw = {"vivid": vivid, "moody": moody, "minimal": minimal}
    weighted = {name: value * float(bias[name]) for name, value in raw.items()}
    best_mode = max(weighted, key=lambda k: weighted[k])
    sky = weighted[best_mode]

    # The sky may go either way; the rider should know when it is close.
    others = sorted(((k, v) for k, v in weighted.items() if k != best_mode),
                    key=lambda kv: -kv[1])
    runner_up, runner_up_score = None, None
    if others and (sky - others[0][1]) <= float(config.scoring.runner_up_within):
        runner_up, runner_up_score = others[0]

    components = {
        **{f"vivid_{k}": v for k, v in vivid_parts.items() if not isinstance(v, list)},
        **{f"moody_{k}": v for k, v in moody_parts.items()},
        **{f"minimal_{k}": v for k, v in minimal_parts.items()},
        "corridor": corridor,
    }
    if config.logging.debug_score_breakdown:
        log.debug("sky vivid=%.1f moody=%.1f minimal=%.1f -> %s (%.1f) %s",
                  vivid, moody, minimal, best_mode, sky, components)

    return ModeResult(
        vivid=vivid, moody=moody, minimal=minimal, weighted=weighted,
        best_mode=best_mode, sky=sky, runner_up=runner_up,
        runner_up_score=runner_up_score, clearing_front=clearing,
        corridor=corridor, components=components,
    )


# ---------------------------------------------------------------------------
# Photo vs video
# ---------------------------------------------------------------------------

@dataclass
class OutputScores:
    photo: float
    video: float
    timelapse_flag: bool
    verdict: str
    components: dict[str, float]


def dynamic_range_term(inputs: SkyInputs, corridor: float) -> float:
    """How much tonal separation the scene offers, 0..1.

    NOTE: the design document lists this as a 0.15-weight photo term but does not
    give a formula, so this definition is inferred and is listed in the README's
    "could not verify" section. It is defined as bright light path multiplied by
    dark cloud overhead — the classic high-dynamic-range sunset. Photo gets a bonus
    because it can bracket; video has no equivalent term because it cannot.
    """
    return clamp(corridor * clamp(inputs.total_cc / 100.0))


def score_output(sky: float, inputs: SkyInputs, corridor: float, *,
                 foreground_interest: float, foreground_motion: float,
                 gusts_kmh: float, wind_500hpa: float | None,
                 blue_hour_minutes: float, config: Config) -> OutputScores:
    """Separate photo and video scores.

    These genuinely diverge — wind ruins video far faster than stills, while fast
    high-altitude wind is a timelapse *bonus* — so they must never be averaged
    into mush.
    """
    o = config.output
    p, v = o.photo, o.video
    sky_norm = clamp(sky / 100.0)

    dr = dynamic_range_term(inputs, corridor)
    photo = 100.0 * (
        float(p.weight_sky) * sky_norm
        + float(p.weight_foreground_interest) * clamp(foreground_interest)
        + float(p.weight_dynamic_range) * dr
        + float(p.weight_wind) * clamp(1.0 - gusts_kmh / float(p.gust_divisor))
    )

    blue_norm = clamp(blue_hour_minutes / float(v.blue_hour_full_minutes))
    video = 100.0 * (
        float(v.weight_sky) * sky_norm
        + float(v.weight_wind) * clamp(1.0 - gusts_kmh / float(v.gust_divisor))
        + float(v.weight_blue_hour) * blue_norm
        + float(v.weight_foreground_motion) * clamp(foreground_motion)
    )

    t = o.timelapse
    timelapse = (
        wind_500hpa is not None
        and wind_500hpa > float(t.min_wind_500hpa)
        and gusts_kmh < float(t.max_gusts_10m)
        and float(t.min_total_cc) < inputs.total_cc < float(t.max_total_cc)
    )

    margin = float(o.verdict_margin)
    if photo - video >= margin:
        verdict = "📷 stills evening"
    elif video - photo >= margin:
        verdict = "🎥 video evening"
    else:
        verdict = "📷🎥 both"

    return OutputScores(
        photo=photo, video=video, timelapse_flag=timelapse, verdict=verdict,
        components={"sky_norm": sky_norm, "dynamic_range": dr,
                    "blue_hour_norm": blue_norm,
                    "foreground_interest": foreground_interest,
                    "foreground_motion": foreground_motion, "gusts": gusts_kmh},
    )


# ---------------------------------------------------------------------------
# Ride safety
# ---------------------------------------------------------------------------

@dataclass
class RideResult:
    score: float
    night_penalty_applied: bool
    components: dict[str, float]


def score_ride(*, max_gust_kmh: float, max_precip_prob: float,
               apparent_temperature: float, precip_preceding_3h: float,
               returns_after_dark_minutes: float, config: Config) -> RideResult:
    """Rideability for a 128 kg 125 on L-plates.

    NOTE: the three weights sum to 0.90, exactly as specified. That is deliberate
    and preserved — it scales the ride term down slightly relative to sky.
    """
    r = config.ride
    gust_term = clamp(1.0 - (max_gust_kmh - float(r.gust_floor_kmh)) / float(r.gust_span_kmh))
    rain_term = clamp(1.0 - max_precip_prob / float(r.rain_span_pct))
    temp_term = gaussian(apparent_temperature, float(r.temp_mu_c), float(r.temp_sigma_c))
    dry_road = 1.0 if precip_preceding_3h == 0 else float(r.dry_road_wet_multiplier)

    score = 100.0 * (float(r.weight_gust) * gust_term
                     + float(r.weight_rain) * rain_term
                     + float(r.weight_temp) * temp_term) * dry_road

    # A penalty, not a blocker: the best light is often after the sun is down.
    night = returns_after_dark_minutes > float(config.rider.night_tolerance_min)
    if night:
        score *= float(config.rider.night_penalty)

    return RideResult(score=score, night_penalty_applied=night,
                      components={"gust_term": gust_term, "rain_term": rain_term,
                                  "temp_term": temp_term, "dry_road": dry_road})


# ---------------------------------------------------------------------------
# worth_it
# ---------------------------------------------------------------------------

def azimuth_match(sun_bearing: float, open_arc: Sequence[float] | None,
                  config: Config) -> float:
    """How well the sunset azimuth lines up with the spot's computed open arc.

    This is where the horizon profile earns its keep: a spot with a ridge at 300 deg
    is superb in December and useless in June, and the system knows that without
    anyone having stood there.
    """
    w = config.worth_it
    if not open_arc:
        return float(w.azimuth_outside)
    start, end = float(open_arc[0]), float(open_arc[1])
    if bearing_in_arc(sun_bearing, start, end):
        return float(w.azimuth_in_arc)
    tolerance = float(w.azimuth_near_tolerance_deg)
    if min(angular_difference(sun_bearing, start),
           angular_difference(sun_bearing, end)) <= tolerance:
        return float(w.azimuth_near_arc)
    return float(w.azimuth_outside)


def distance_discount(minutes_one_way: float, config: Config) -> float:
    """Distance is never a filter, always a discount."""
    w = config.worth_it
    raw = 1.0 - float(w.distance_discount_rate) * (
        minutes_one_way / float(w.distance_discount_reference_minutes)
    )
    return clamp(raw, float(w.distance_discount_floor), float(w.distance_discount_ceiling))


def max_radius_km(regional_sky: float, config: Config) -> float:
    """How far it is worth riding tonight: ~20 km on a dull evening, ~65 km on a great one."""
    w = config.worth_it
    return float(w.radius_base_km) + float(w.radius_span_km) * (
        clamp(regional_sky / 100.0) ** float(w.radius_exponent)
    )


def combine_output(photo: float, video: float, config: Config) -> float:
    if config.output.optimise == "either":
        return max(photo, video)
    return (float(config.output.blend_photo_weight) * photo
            + float(config.output.blend_video_weight) * video)


def score_worth_it(*, sky: float, output_score: float, ride: float, spot: float,
                   sun_bearing: float, open_arc: Sequence[float] | None,
                   minutes_one_way: float, blocked: bool, config: Config
                   ) -> tuple[float, dict]:
    """The single ranking number. Zero if any hard blocker fired."""
    if blocked:
        return 0.0, {"blocked": True}

    w = config.worth_it
    az = azimuth_match(sun_bearing, open_arc, config)
    discount = distance_discount(minutes_one_way, config)
    base = (float(w.weight_sky) * sky
            + float(w.weight_output) * output_score
            + float(w.weight_ride) * ride
            + float(w.weight_spot) * spot)
    score = base * az * discount
    return score, {"base": base, "azimuth_match": az,
                   "distance_discount": discount, "blocked": False}


def band_for(score: float, config: Config) -> tuple[str, str]:
    """(emoji, label) for a worth_it score."""
    for entry in config.bands:
        if score >= float(entry.min):
            return entry.emoji, entry.label
    last = config.bands[-1]
    return last.emoji, last.label
