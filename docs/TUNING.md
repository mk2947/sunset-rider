# Tuning

Everything tunable is in `config.yaml`. There are no magic numbers in the code, so
you never need to touch Python to change behaviour.

**There is no automatic weight fitting, deliberately.** Twenty data points cannot
support fitting a dozen weights, and a model that quietly retunes itself is one you
can no longer reason about. This is a human-in-the-loop process.

---

## Reading the calibration data

Rate an evening by replying in Telegram:

```
/rate 4
```

A weekly job joins that rating to what was predicted and appends a row to
`data/calibration.csv`:

| column | meaning |
|---|---|
| `date`, `spot_id`, `spot_name` | which evening and where |
| `predicted_worth_it` | the headline number you were sent |
| `predicted_sky`, `best_mode` | the sky score and which aesthetic won |
| `photo`, `video` | the two output scores |
| `clearing_front` | 1 if a clearing front was called |
| `corridor`, `slot` | the two corridor-derived terms |
| `actual` | your 1–5 |

**Wait until roughly 20 entries before changing anything.** Below that you are
fitting noise. Then:

```python
import pandas as pd
df = pd.read_csv("data/calibration.csv")

# Is the headline number tracking reality at all?
print(df[["predicted_worth_it", "actual"]].corr())

# Which aesthetic is being over- or under-called?
print(df.groupby("best_mode")[["predicted_worth_it", "actual"]].mean())

# Did the clearing-front flag actually earn its +12?
print(df.groupby("clearing_front")["actual"].mean())
```

(`pandas` is not a project dependency — this is throwaway analysis, not something the
service needs.)

Rough reading of `actual`: 5 = worth cancelling plans, 4 = glad I went, 3 = fine,
2 = should have stayed home, 1 = actively bad.

---

## The first dial: `mode_bias`

**If the system keeps picking the wrong aesthetic, turn this first.** It is the only
dial that changes *what kind of evening* gets recommended rather than how confident
the numbers are.

```yaml
scoring:
  mode_bias:
    vivid: 1.0
    moody: 1.0
    minimal: 1.0
```

Each mode score is multiplied by its bias before the winner is chosen, so the biases
are relative — only their ratios matter.

| Symptom in the data | Change |
|---|---|
| `best_mode == "vivid"` rows average low `actual` | drop `vivid` to 0.85 |
| Moody evenings consistently beat their prediction | raise `moody` to 1.15 |
| Clear evenings under-delivering for stills but you shoot a lot of video | raise `minimal` |

Move in steps of 0.1–0.15. Anything larger and one mode simply always wins, which is
just v1's bug in a different costume.

---

## If moody evenings consistently under-deliver

This is the most likely failure mode in a **maritime climate**, and worth
understanding before you reach for `mode_bias`.

The `slot` term rewards a gap between a heavy deck overhead and a clear corridor
toward the sun. In the UK that gap is often *real in the model and gone by the time
you arrive* — our low cloud is mobile, ragged and poorly resolved, and a slot
forecast 12 hours out frequently fills in.

**The `slot` weight is probably too generous for our climate.** It carries 0.35, the
largest share of the moody score. If your calibration shows moody evenings
systematically over-promising:

```yaml
scoring:
  moody:
    weight_slot: 0.30      # from 0.35
    weight_deck: 0.35      # from 0.30 — a heavy deck is at least reliably there
```

Keep the four moody weights summing to 1.0. Prefer this over blunting `mode_bias`,
because it fixes *why* moody is wrong rather than just how often it wins.

The related dial is the no-slot cap:

```yaml
scoring:
  moody:
    cap_no_slot_below: 0.15   # raise to 0.20 to be stricter about needing a real gap
    cap_no_slot_value: 30.0
```

Raising `cap_no_slot_below` demands a more convincing gap before an overcast evening
can score at all. See `docs/SCORING.md` §2 for why this cap exists.

---

## Known issue: the top two bands are nearly unreachable

Worth understanding before you conclude the system is pessimistic.

The four `worth_it` weights sum to 1.0, so the base score is essentially a weighted
average — and it is then **multiplied** by `azimuth_match` and `distance_discount`.
Two things cap it hard:

- `ride` can never exceed **90**, because its own three weights deliberately sum to
  0.90 (see `docs/SCORING.md` §4).
- `distance_discount` is 0.735 for a spot 60 minutes away, and 0.55 at the floor.

Worked example — a *perfect* moody clearing-front evening at Walbury Hill, the best
spot in the region: sky 100, output 86, ride 59, spot 84 gives a base of 88.7, times
0.735 for the ride out, equals **65 — "GOOD"**.

So `🔥 DROP EVERYTHING` (90+) and `⭐ EXCELLENT` (75+) are effectively out of reach
for anything more than about 20 minutes away, and the five-band table compresses into
three. The ranking between spots is still correct; it is the *absolute* labels that
are misleading.

This is the specified formula and it has not been changed. If the bands read too
harshly once you have real ratings, the dial is the discount rate, not the weights:

```yaml
worth_it:
  distance_discount_rate: 0.12   # from 0.20 — 60 min then costs 16%, not 27%
  distance_discount_floor: 0.70  # from 0.55
```

Change one of these, not both, and re-read `predicted_worth_it` against `actual`
before going further.

---

## Other dials, roughly in order of usefulness

### Being sent too far, or not far enough

```yaml
worth_it:
  radius_base_km: 20.0     # floor on a dull evening
  radius_span_km: 45.0     # extra reach on an exceptional one
  distance_discount_rate: 0.20
```

Raise `distance_discount_rate` to bias harder toward local spots. Distance is always
a discount, never a filter — a distant spot can still win if it is clearly better.

### Rides that feel too hairy, or too cautious

```yaml
blockers:
  max_gust_kmh: 60.0       # lower if 60 km/h gusts already feel unpleasant
ride:
  gust_floor_kmh: 25.0     # where the gust penalty starts biting
```

The blocker is about safety on a 128 kg bike; the ride weight is about comfort. They
are separate on purpose.

### Arriving with too little or too much time

```yaml
rider:
  setup_minutes: 45        # parking, walking in, setting up
  average_speed_kmh: 48.0
  road_distance_factor: 1.30
```

`LEAVE BY = sunset − setup_minutes − ride_time`. If you keep arriving rushed, raise
`setup_minutes` before touching the speed. If ride times are consistently optimistic,
raise `road_distance_factor` — it converts straight-line to road distance and is the
crudest approximation in the system (no routing API is used, because they all need
keys).

### Photo/video verdict flipping too readily

```yaml
output:
  verdict_margin: 8.0      # points of separation before it calls one over "both"
  video:
    gust_divisor: 45.0     # lower = harsher wind penalty for video
```

### Wrong aesthetic *band*, not wrong mode

```yaml
scoring:
  vivid:
    canvas_peak_mhc: 45.0  # where the fire-sky sweet spot sits
    canvas_sigma: 22.0     # how forgiving that peak is
```

Widening `canvas_sigma` makes vivid less picky about cloud amount. Only touch this if
the calibration data shows vivid mis-scoring across a range of cloud amounts, rather
than one mode being wrong overall.

---

## Discovery dials

Re-run `python -m sunset_rider.discovery --stage build` after changing these. The
horizon cache is permanent, so already-profiled spots are never re-fetched.

```yaml
discovery:
  min_horizon_openness: 0.4        # geometry filter
  min_elevation_prominence: 0.12   # rejects flat ground that "looks" open
  max_profiled_candidates: 42      # the expensive dial
  max_per_distance_band: 7
  search_radius_m: 70000           # needs --force-harvest to take effect
```

**Cost warning.** Each candidate costs 217 elevation samples, and Open-Meteo's free
tier counts *locations*: 600/min, 5,000/hour, under 10,000/day (all measured). That is
roughly 23 candidates per hour and 45 per day. Raising `max_profiled_candidates`
above ~45 means the run will span multiple days — which is fine, because it is
resumable and incremental.

If `viewpoints.yaml` has fewer entries than you want, raise
`max_per_distance_band` and re-run on a later day; only the new candidates cost
anything.

Lowering `min_elevation_prominence` below about 0.10 starts admitting flat suburban
parks and car parks with no actual view. Lowering `min_horizon_openness` admits spots
where terrain blocks part of the sunset arc — the `azimuth_match` term will already
penalise those on the dates it matters, so this is less dangerous than it sounds.
