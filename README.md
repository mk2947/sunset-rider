# sunset_rider

Tiered sunset-riding forecasts for a CBT rider near Reading, delivered to Telegram
and running free on GitHub Actions.

It answers three different questions at three different lead times, because a
3-day sunset forecast and a 3-hour one are not the same product:

| Lead time | Mode | Question | Confidence basis |
|---|---|---|---|
| T-72h → T-48h | `plan` | Which evening this week should I hold? | Ensemble spread, 51 ECMWF members |
| T-24h | `confirm` | Is the evening I held still the one? | Deterministic forecast |
| T-3h | `go` | Leave when, to where? | Freshest run, with a leave-by time |

It scores three aesthetic modes — **vivid**, **moody**, **minimal** — and reports
the winner by name. "Cloudy" is not "bad": a heavy deck with a clear slot at the
horizon is a godray evening and often the best photograph of the month.

Photo and video are scored **separately**, because they want genuinely different
weather (wind ruins video far faster than stills; fast high-altitude wind is a
timelapse bonus).

---

## Setup

### 1. Create the bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram and send `/newbot`.
2. Follow the prompts. BotFather replies with a **token** like
   `123456789:AAF...`. Keep it secret.
3. **Send your new bot a message** — any message. A bot cannot start a conversation
   with you, so it has no chat to reply to until you do this.

### 2. Find your chat ID

Easiest route: message [@userinfobot](https://t.me/userinfobot). It replies with your
numeric user ID, which **is** your chat ID for a one-to-one chat with your own bot.

Otherwise, open this URL in a browser, substituting your token:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

Look for `"chat":{"id":123456789,...}`. That number is your **chat ID**.

**If the response is empty**, it is one of three things, and they look identical from
the outside:

1. You have not messaged the bot yet (step 1.3) — a bot cannot open a conversation.
2. **A webhook is set on the bot.** Telegram will not deliver updates by both webhook
   and `getUpdates`, so the latter returns empty however many messages you send.
   This is the one people miss.
3. The update expired. Telegram keeps updates for about 24 hours, and any earlier
   `getUpdates` call using an `offset` will already have consumed them.

To find out which:

```bash
python scripts/get_chat_id.py
```

It reads `TELEGRAM_BOT_TOKEN` from the environment, never prints it, checks all three
causes in order and tells you which applies.

### 3. Add the secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**

| Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | the token from BotFather |
| `TELEGRAM_CHAT_ID` | the number from `getUpdates` |

Secrets stay private even on a public repo. Use a **public** repo — Actions minutes
are unlimited on public repositories, which is what makes this free forever.

### 4. Enable Actions and test

1. **Actions** tab → enable workflows if prompted.
2. Select **sunset forecast** → **Run workflow**.
3. Set `dry_run: true` first to see the message in the job log without sending.
4. Then run with `mode: go` and `dry_run: false` to get a real message.

---

## ⚠️ Do not "tidy up" the state commit

`state/last_sent.json` is committed back to the default branch after every run, and
that commit is doing **two** jobs:

1. **Dedupe.** Send windows are ≥60 minutes wide and cron fires hourly, so without
   this the same message could be sent twice.
2. **Keepalive.** *GitHub automatically disables scheduled workflows after 60 days
   with no commits to the default branch, and it does so silently.* The state commit
   resets that timer. On a quiet run with nothing to send, the workflow writes
   `state/heartbeat.txt` instead so the timer still resets.

Delete that commit step and the service stops working roughly two months later,
with no error, no notification, and no obvious cause. This is the single easiest way
to break this repo.

## Why the send windows are wide

Scheduled GitHub Actions runs are routinely delayed **5–30 minutes** under load, so
a narrow window would simply be missed.

There is a second, sharper reason. Cron fires **hourly**, so a window narrower than
60 minutes contains an hourly tick only *some* of the time. The design originally
specified a 30-minute `go` window (2h45m–3h15m before sunset); simulating every day
of 2025 showed that produced **zero** `go` messages on **165 days**. All windows are
now exactly 60 minutes and half-open, which guarantees precisely one tick per day —
no gaps, no doubles. `tests/test_gating.py` asserts this for every day of the year
and across both DST boundaries.

## Why the schedule is UTC cron plus a Python gate

GitHub Actions cron is UTC-only, so any fixed UTC schedule drifts by an hour across
the GMT/BST boundary — fatal for a sunset-timed job. **Cron decides whether to wake;
Python decides whether to act**, using `zoneinfo.ZoneInfo("Europe/London")`. This is
DST-proof by construction. Do not try to replace the Python gate with a cleverer
cron expression.

---

## Usage

```bash
# Print a 5-evening planning view; no network writes, nothing sent
python -m sunset_rider.main --dry-run --mode plan --date 2025-06-18

# Tomorrow's confirmation, and tonight's go/no-go
python -m sunset_rider.main --dry-run --mode confirm --date 2025-06-18
python -m sunset_rider.main --dry-run --mode go --date 2025-06-18

# Full component breakdown for every spot
python -m sunset_rider.main --dry-run --mode go --date 2025-06-18 -v
```

Without `--mode`, the run self-gates on the schedule and exits quietly if nothing
is due — which is what the hourly cron relies on.

## Tests

```bash
pytest
pytest --cov=sunset_rider --cov-report=term-missing
```

---

## Re-running discovery

`data/viewpoints.yaml` is generated, not hand-written. Every horizon profile is
computed from the Open-Meteo 90 m DEM.

```bash
python -m sunset_rider.discovery --stage build
```

The run is **resumable and incremental**. Horizon profiles are cached permanently in
`data/raw/horizons.json` and are never re-fetched — terrain does not change. If the
run stops on a rate limit, just run it again; it picks up where it left off.

**To widen the search**, edit `config.yaml`:

- `discovery.search_radius_m` — the Overpass harvest radius (default 70 km). Changing
  this requires `--force-harvest`.
- `discovery.max_profiled_candidates` and `max_per_distance_band` — how many
  candidates get a horizon profile. This is the expensive dial (see rate limits).
- `discovery.min_horizon_openness` / `min_elevation_prominence` — the geometry filter.

### Rate limits, measured

Open-Meteo's free tier counts **locations**, not requests. A 429 arrives at exactly
the 600th location within a minute; the documented limits are 600/min, 5,000/hour,
and under 10,000/day. Each candidate costs 217 elevation samples (36 bearings × 6
distances + the viewpoint itself), so roughly **23 candidates per hour** and about
**45 per day** is the ceiling. The client self-throttles and waits out 429s.

---

## Tuning

Start with `docs/TUNING.md`. The short version: **`config.mode_bias` is the first
dial to turn** if the system keeps picking the wrong aesthetic.

```yaml
scoring:
  mode_bias:
    vivid: 1.0
    moody: 1.3    # bias toward moody evenings
    minimal: 0.8
```

Every tunable number lives in `config.yaml`. There are no magic numbers in the code.

Rate an evening by replying `/rate 1` … `/rate 5` in Telegram. A weekly job collects
those into `data/calibration.csv`. There is deliberately **no automatic weight
fitting** — twenty data points cannot support it.

---

## Deviations from the original design

Each of these was a measured finding, not a preference.

1. **RG2 7BD is at 51.443326, -0.956334**, not the 51.4180, -0.9600 in the brief —
   about 2.9 km out. Verified against `api.postcodes.io`.

2. **The ensemble API is on `ensemble-api.open-meteo.com`**, not
   `api.open-meteo.com/v1/ensemble`, which returns 404.

3. **`plan` mode uses ECMWF IFS (51 members), not ICON-EU/GFS.** Checked all 10
   ensemble models: only the ECMWF ensembles expose `cloud_cover_low/mid/high`.
   ICON-EU and GFS return total cloud only, which cannot support the corridor or the
   three-mode split at all. *(Confirmed with you before changing.)*

4. **Ensemble PoP is derived from member agreement.** No ensemble model exposes
   `precipitation_probability` or `visibility` — both are accepted by the API and
   silently return all-nulls. PoP is computed as the fraction of members with
   precipitation > 0.1 mm, which is what an ensemble PoP actually means. The
   visibility term is held neutral in `plan` mode only. *(Confirmed with you.)*

5. **All Open-Meteo requests ask for UTC**, and localisation happens in Python.
   Requesting `timezone=Europe/London` makes the API return
   `utc_offset_seconds=3600` **even in December**, which would shift every hourly
   value an hour away from the sunset it is meant to bracket.

6. **The `go` window is 60 minutes, not 30.** See "Why the send windows are wide".

7. **A no-slot cap was added to the moody score.** All four specified moody weights
   are unchanged. Without the cap, a fully overcast evening — 90% cloud overhead
   *and* 95% low cloud from 30–250 km toward the sun — scored 63 (GOOD), because
   `deck + texture + drama` pay out 65% regardless of the slot and `drama` has a hard
   floor of 0.5. There is no sunset to photograph on such an evening. *(Confirmed
   with you before changing.)*

8. **The 15 km horizon sample was dropped** (6 rings, not 7). To subtend the 1.5°
   openness threshold at 15 km, terrain must rise 393 m *above the viewpoint*; the
   highest point in South East England is Walbury Hill at 297 m. The ring is
   provably incapable of affecting any result here, and elevation samples are the
   binding cost. The 8 km ring (209 m) *can* bind and is kept.

9. **The Overpass harvest was widened and split.** The brief's query missed 7 of 15
   seed spots. Most importantly it asked only for `node["amenity"="parking"]`, and
   Ridgeway car parks — Bury Down, Cowleaze Wood, the exact spot type the brief calls
   ideal — are OSM **ways**. The query also needed `natural=grassland` (Lardon Chase,
   Lough Down), `leisure=nature_reserve` (Watlington Hill, Whiteleaf Hill) and
   `leisure=park` (Dinton Pastures). It is now three sequential sub-queries because
   a single combined one times out on the public servers.

10. **A minimum prominence filter was added.** `horizon_openness` measures *terrain*
    obstruction only, so flat ground scores 1.00 — a municipal park in central
    Reading qualified as a sunset viewpoint. Waterside close-fallbacks are exempt,
    since they earn their place through reflections rather than elevation.

11. **A minimum ride distance was added (2 km).** "Cintra Park", a municipal park
    300 m from the house with prominence 0.10, ranked *above* Walbury Hill on a clear
    evening (84 vs 63) purely because its distance discount was 1.0. This is a
    motorcycle app: 300 m is a walk, not a ride, and anything that close is
    invariably parkland rather than a viewpoint.

12. **Candidate selection ranks by local relief, not raw altitude.** Computed free
    from OSM `ele` tags. Absolute height is the wrong prior here — Wittenham Clumps
    is a famous viewpoint at 120 m because it stands above a flat floodplain, while a
    240 m Chiltern peak among other 240 m peaks sees very little.

13. **Historical dates route to the archive endpoint.** The forecast API only reaches
    ~92 days back, so `--date` would fail for any older evening. Older requests now
    use `archive-api.open-meteo.com`, which makes calibration backfill possible. Note
    the archive lacks `visibility`, `precipitation_probability`, `cape` and
    pressure-level wind; those degrade to neutral rather than failing the run.
    `plan` mode cannot run historically at all — the ensemble spans only about
    ±4 months — and it reports that reason rather than going silent.

---

## Things I could not verify

Stated plainly, as requested.

- **`dirflg=h` for motorway avoidance.** The `?api=1` Google Maps URL format has no
  documented avoid-motorways parameter, so the links use the legacy
  `maps.google.com/maps?...&dirflg=h` form. I did not confirm Google still honours
  `dirflg=h`, and I could not test the routing itself. **Check the first route before
  trusting it — you are on L-plates and must not be sent onto the M4.**

- **Ride times are estimates, not routed.** Every routing API needs a key, so
  `minutes_one_way` is straight-line distance × `rider.road_distance_factor` (1.30) ÷
  48 km/h. It is a consistent basis for ranking, but a real route on a 125 will
  differ — treat `LEAVE BY` as approximate until you have ridden a spot once.

- **The `dynamic_range_term` formula is inferred.** The brief lists it as a
  0.15-weight photo term but never defines it. I defined it as
  `corridor × (total_cc/100)` — a bright light path under dark cloud. Photo gets the
  bonus because it can bracket; video has no equivalent term because it cannot.

- **No message has been sent to Telegram.** Sending during development was
  explicitly out of scope, so the delivery path is covered by tests against a fake
  transport but has never talked to the real API. Step 4 of Setup is the real test.

- **Scoring accuracy is unvalidated.** Every formula is implemented and tested to
  spec, but no prediction has been checked against a real evening. That is what
  `/rate` and `data/calibration.csv` are for.

- **`viewpoints.yaml` currently holds 12 spots, not the 25 target.** Open-Meteo's
  daily elevation quota (<10,000 locations) allows ~45 candidates per day, and the
  queue is set to 200 — about 4–5 days. **Run this once a day until it stops warning:**

  ```bash
  python -m sunset_rider.discovery --stage build
  ```

  Nothing already computed is re-fetched, and `viewpoints.yaml` is rebuilt from the
  full cache on every run, so the list is usable throughout and simply gets better.
  Six seed spots (Wittenham Clumps, Bury Down, Watlington Hill, Uffington, Coombe
  Hill, Dinton Pastures) are in the queue but not yet profiled.

- **The top two bands are nearly unreachable.** A perfect evening at the best spot in
  the region scores about 65 ("GOOD"), because `ride` caps at 90 by design and the
  distance discount multiplies the total. Ranking between spots is unaffected; the
  absolute labels compress. Documented with the dial to turn in `docs/TUNING.md`.

- **Community cross-referencing was not done.** Komoot ratings, ShotHotSpot and
  bestbikingroads are manual per-spot research with no free API. `gate_closes` comes
  only from OSM tags, and anything unestablished is `unknown` and flagged ⚠️ rather
  than guessed. Those ⚠️ spots are where your local knowledge is worth most.

---

## Layout

```
sunset_rider/
  config.py      dotted-access config loader; raises on missing keys
  solar.py       sunset time, azimuth, golden/blue hour (astral, UTC internally)
  geo.py         great-circle maths for corridor and horizon sampling
  weather.py     Open-Meteo forecast/ensemble/elevation, batched and throttled
  discovery.py   Overpass harvest -> horizon profiles -> viewpoints.yaml
  scoring.py     three modes, photo/video split, ride, worth_it
  gating.py      hard blockers, gate closures, DST-proof send windows
  pipeline.py    fetch -> score -> rank
  message.py     three message shapes; plan mode cannot print a point score
  telegram.py    delivery (env-only credentials)
  state.py       send dedupe + keepalive
  feedback.py    /rate collection into calibration.csv
  main.py        CLI
docs/
  SCORING.md     what every number means and why
  TUNING.md      how to read the calibration data and what to change
```
