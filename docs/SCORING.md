# Scoring

Every constant referenced here lives in `config.yaml`. Nothing in this document is
hardcoded in the source.

---

## 1. The corridor — why this beats a cloud-percentage app

Warm light has to travel **under** the cloud deck, from the horizon to the clouds
above you. A perfect canvas overhead is worthless if there is a wall of low cloud
80 km toward the sun. So the system samples low cloud along the sun-to-sky corridor,
weighted with the nearest obstruction heaviest:

```
corridor = Σ wᵢ · (1 − low_cloudᵢ / 100)     weights [0.40, 0.30, 0.20, 0.10]
                                             at      [30,   80,   150,  250] km
```

Sample points are placed along the **actual solar azimuth at sunset**, recomputed
daily. Never "west": at Reading the sunset bearing swings from ~311° at midsummer to
~232° at midwinter.

Weather is **interpolated between the two hours bracketing sunset**, never rounded to
the nearest hour.

`tests/test_scoring.py::test_identical_overhead_conditions_score_lower_with_a_blocked_corridor`
is the proof this is actually wired in: identical sky overhead, one with a clear
corridor and one with a wall at 30 km, must differ by more than 10 points.

---

## 2. Three aesthetic modes

All three are computed every time; the winner is reported by name. The runner-up is
also reported when it is within `scoring.runner_up_within` (10) points, because the
sky may genuinely go either way.

### VIVID 🔥 — classic fire sky

Peak quality near **45% combined mid+high cloud**: enough canvas to catch the light,
enough gaps to let it through.

```
canvas  = exp(−(mhc − 45)² / (2 · 22²))          mhc = mid + high, capped at 100
clarity = 0.6 · clamp(visibility / 25000) + 0.4 · exp(−(rh − 55)² / (2 · 20²))
vivid   = 100 · (0.45·canvas + 0.35·corridor + 0.20·clarity)
```

Capped at 25 if low cloud overhead > 85%, at 30 if rain probability > 60%, at 15 if
total cloud > 95%.

### MOODY 🌧️ — heavy deck with a horizon slot

The setup that produces godrays, spotlit landscape and dramatic broken light. The
detector is a **contrast** between overhead and corridor, not an absolute cloud
figure — which is what lets a 90%-cloud evening score highly.

```
deck    = clamp((total_cc − 55) / 35)
slot    = clamp(corridor_far − corridor_near_block)
          corridor_far        = mean clear fraction at 150 and 250 km
          corridor_near_block = low cloud fraction at 30 km
texture = clamp(1 − |mid − 55| / 55)
drama   = clamp(cape / 400) · 0.5 + 0.5
rain_ok = 1.0 if precip_prob < 45 else 0.4
moody   = 100 · (0.30·deck + 0.35·slot + 0.20·texture + 0.15·drama) · rain_ok
```

**The no-slot cap.** If `slot < 0.15`, moody is capped at 30.

This guard is not in the original design and is worth understanding. The four weights
alone pay out 65% of the score for *any* heavy sky, because `deck`, `texture` and
`drama` do not depend on the slot and `drama` has a hard floor of 0.5. A
comprehensively socked-in evening — 90% cloud overhead **and** 95% low cloud from
30–250 km toward the sun — therefore scored **63 (GOOD)** with no cap. There is no
sunset to photograph on such an evening. Moody is *defined* by the contrast between
a heavy deck and a clear horizon; remove the contrast and there is nothing left.

All four specified weights are unchanged. The cap is a guard in the same idiom vivid
already uses, and it does not touch a real godray evening (slot ≈ 0.85 → ~93).

**Clearing front.** Total cloud falling by more than 25 percentage points over the
two hours after sunset, while the corridor is already clear. Adds +12, capped at 100,
and is called out explicitly in the message. This is the evening worth cancelling
plans for.

### MINIMAL 🌌 — clean gradient

Near-cloudless. Weak for stills, often **excellent for video**: smooth exposure ramp,
no blown highlights, strong silhouettes, long usable window.

```
cleanliness = clamp(1 − total_cc / 20)
air         = clamp(visibility / 30000)
minimal     = 100 · (0.45·cleanliness + 0.35·air + 0.20·corridor)
```

### Selection

```
weighted  = {mode: score · mode_bias[mode]}
best_mode = argmax(weighted)
sky       = weighted[best_mode]
```

---

## 3. Photo vs video

These diverge genuinely and are never averaged.

| Factor | Photo | Video |
|---|---|---|
| Ground wind | mild penalty (tripod shake) | **heavy** (shake + mic noise) |
| High-altitude wind | neutral | **bonus** — drives timelapse motion |
| Blue-hour length | minor | **major** — it is the shot budget |
| Foreground motion | minor | **major** — water and grass = life in frame |
| Dynamic range | high tolerance (bracket/HDR) | low tolerance (no bracketing) |

```
photo = 100 · (0.55·sky + 0.20·foreground_interest + 0.15·dynamic_range
             + 0.10·clamp(1 − gusts/70))
video = 100 · (0.40·sky + 0.25·clamp(1 − gusts/45)
             + 0.20·blue_hour_norm + 0.15·foreground_motion)

timelapse_flag = wind_500hPa > 40  and  gusts_10m < 25  and  20 < total_cc < 80
```

Note the different gust divisors — 70 for photo, 45 for video. At 50 km/h gusts the
photo term is still 0.29 while the video term has collapsed to 0.

**`dynamic_range` is an inferred definition.** The brief weights it at 0.15 but never
gives a formula. It is implemented as `corridor × (total_cc/100)` — a bright light
path under dark cloud. Photo gets the bonus because it can bracket; video has no
equivalent term because it cannot. Flagged in the README as unverified.

---

## 4. Ride safety

```
gust_term = clamp(1 − (max_gust − 25) / 40)
rain_term = clamp(1 − max_precip_prob / 60)
temp_term = exp(−(apparent_temp − 18)² / (2 · 10²))
dry_road  = 1.0 if no precipitation in the preceding 3 h else 0.6
ride      = 100 · (0.35·gust_term + 0.35·rain_term + 0.20·temp_term) · dry_road
```

Those three weights sum to **0.90**, exactly as specified. That is deliberate and
preserved: a perfect ride scores 90, not 100, which scales the ride term down
slightly against sky. It is not a typo and should not be "corrected".

**Night riding is a penalty (×0.85), never a blocker.** The best light is often after
the sun is down.

### Hard blockers — spot dropped entirely, reason always stated

| Blocker | Threshold | Why |
|---|---|---|
| Gusts | > 60 km/h | 128 kg bike, exposed downland, crosswind |
| Apparent temp | < 2 °C | ice |
| Rain probability | > 70% | |
| Visibility | < 2000 m | fog |
| Gate closes | before sunset + 45 min | locked in |

**`gate_closes` is the field most likely to ruin an evening.** A large fraction of the
best English viewpoints have car parks that lock at or before dusk, which makes them
worthless for their single purpose. `sunset`/`dusk` is always blocking. A clock time
is evaluated **per date** — a 20:00 gate blocks a June evening (sunset 21:24) but is
irrelevant in December (sunset 15:57). `unknown` is allowed through but flagged ⚠️.

---

## 5. worth_it

```
distance_discount = clamp(1 − 0.20 · (minutes/45), 0.55, 1.0)
azimuth_match     = 1.0 in the open arc · 0.7 within 15° · 0.35 outside
output_score      = max(photo, video)          when optimise = "either"

worth_it = (0.50·sky + 0.20·output + 0.15·ride + 0.15·spot)
           · azimuth_match · distance_discount

worth_it = 0 if any hard blocker fired
```

**Distance is never a filter, always a discount.** The candidate set is gated
separately by a dynamic radius:

```
max_radius_km = 20 + 45 · (regional_sky / 100)^1.5
```

≈20 km on a dull evening, ≈65 km on an exceptional one. `regional_sky` is computed
once at the Reading centroid before any per-spot work.

| worth_it | Band |
|---|---|
| 90–100 | 🔥 DROP EVERYTHING |
| 75–89 | ⭐ EXCELLENT |
| 60–74 | ✅ GOOD |
| 45–59 | 🙂 DECENT — stay local |
| 0–44 | 😐 POOR |

Every rideable evening is sent regardless of band. The score *is* the information;
nothing is suppressed. A poor evening is stated plainly and the 15-minute fallback is
named.

---

## 6. Horizon profiles

The load-bearing idea. For each candidate, terrain is sampled along **36 bearings**
(every 10°) at **0.25, 0.5, 1, 2, 4 and 8 km**, and the elevation angle to each
sample is computed:

```
angle = atan2(elev_sample − elev_viewpoint, distance)
```

The horizon angle for a bearing is the **maximum** along it — a single close ridge
blocks the view regardless of what lies beyond.

Derived per spot:

- `horizon_profile` — all 36 bearings
- `open_arc` — longest contiguous run in the 200°–330° sector below 1.5°
- `horizon_openness` — fraction of that sector below 1.5°
- `elevation_prominence` — height above the mean within 5 km, normalised by 120 m

A spot with a ridge at 300° is superb in December (sun sets at 232°) and useless in
June (311°). The system knows this for every spot without anyone having stood there.

**Two filters, not one.** `horizon_openness > 0.4` **and**
`elevation_prominence ≥ 0.12`. Openness alone measures *terrain* obstruction, and
flat ground has none — a municipal park in central Reading scored openness 1.00 with
no view whatsoever, and the 90 m DEM cannot see buildings or trees either. Waterside
close-fallbacks are exempt from the prominence rule, since they earn their place
through reflections rather than elevation.

**The 15 km ring was dropped.** Blocking 1.5° at 15 km needs terrain 393 m *above the
viewpoint*; the highest point in South East England is Walbury Hill at 297 m. It is
provably dead computation here. The 8 km ring (209 m) can bind and is kept.

---

## 7. plan mode is a different product

At 72 hours, mid- and high-cloud fields are among the least skilful model outputs —
and they are exactly what this system scores on. A confident number would be a lie.

So `plan` mode reports **relative ranking, the share of members clearing "good", and
the interquartile range**, across 51 ECMWF IFS ensemble members. Each member is
scored independently through the full three-mode pipeline, including its own
corridor, then summarised.

A **tight IQR at 72 hours** is the real signal that an evening is worth holding.

`message.py` **forbids** plan mode from printing a bare point score. This is enforced
by `assert_no_bare_score()`, which raises rather than trusting the template, and is
asserted by regex in `tests/test_message.py`.
