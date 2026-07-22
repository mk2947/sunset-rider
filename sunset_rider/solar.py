"""Solar geometry.

Load-bearing module. The corridor bearing, the azimuth/open-arc match and every
"leave by" time depend on the sun's position being right, so this is computed
locally with astral rather than trusted to a weather API.

Two hard rules, both learned the hard way:

1. Everything internal is timezone-aware UTC. Localisation to Europe/London
   happens only at the display edge.
2. We never ask Open-Meteo for local timestamps. Verified 2026-07-21: the API
   returns ``utc_offset_seconds=3600`` for Europe/London even on December dates,
   i.e. it applies one fixed offset per request and gets GMT wrong by an hour.
   Requesting UTC and localising here sidesteps that entirely.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from astral import Observer
from astral.sun import azimuth, elevation, sunset, time_at_elevation, SunDirection

from .config import Config

UTC = dt.timezone.utc


@dataclass(frozen=True)
class SolarEvents:
    """Sunset-centred solar events for one date at one location.

    All datetimes are timezone-aware UTC.
    """

    date: dt.date
    sunset: dt.datetime
    sun_bearing: float
    golden_hour_start: dt.datetime
    blue_hour_start: dt.datetime
    blue_hour_end: dt.datetime
    civil_dusk: dt.datetime
    timezone: str

    @property
    def blue_hour_minutes(self) -> float:
        """Length of the blue hour proper (sun -4 deg to -6 deg).

        This is the "usable window" — the video shot budget.
        """
        return (self.blue_hour_end - self.blue_hour_start).total_seconds() / 60.0

    @property
    def usable_minutes_after_sunset(self) -> float:
        """Sunset to civil dusk: the whole post-sunset shooting window."""
        return (self.civil_dusk - self.sunset).total_seconds() / 60.0

    def local(self, moment: dt.datetime) -> dt.datetime:
        """Convert one of these UTC moments to the configured local timezone."""
        return moment.astimezone(ZoneInfo(self.timezone))

    @property
    def sunset_local(self) -> dt.datetime:
        return self.local(self.sunset)


class SolarCalculator:
    """Computes sunset geometry for a fixed observer."""

    def __init__(self, config: Config, latitude: float | None = None,
                 longitude: float | None = None) -> None:
        self._config = config
        self._solar = config.solar
        self.timezone = config.home.timezone
        self.latitude = config.home.latitude if latitude is None else latitude
        self.longitude = config.home.longitude if longitude is None else longitude
        self.observer = Observer(latitude=self.latitude, longitude=self.longitude,
                                 elevation=0.0)

    # -- core ---------------------------------------------------------------

    def sunset_utc(self, date: dt.date) -> dt.datetime:
        """Sunset for ``date``, timezone-aware UTC."""
        return sunset(self.observer, date, tzinfo=UTC)

    def azimuth_at(self, moment: dt.datetime) -> float:
        """Solar azimuth in degrees clockwise from true north."""
        return azimuth(self.observer, moment.astimezone(UTC))

    def elevation_at(self, moment: dt.datetime) -> float:
        """Solar elevation in degrees above the horizon."""
        return elevation(self.observer, moment.astimezone(UTC))

    def sun_bearing(self, date: dt.date) -> float:
        """Solar azimuth at the moment of sunset — the corridor bearing.

        Never hardcode "west". At Reading this swings from ~311 deg in midsummer
        to ~232 deg at midwinter, which is the whole reason horizon profiles matter.
        """
        return self.azimuth_at(self.sunset_utc(date))

    def _at_elevation(self, date: dt.date, elevation_deg: float) -> dt.datetime:
        """Time the descending sun passes a given elevation, UTC."""
        return time_at_elevation(
            self.observer,
            elevation_deg,
            date,
            direction=SunDirection.SETTING,
            tzinfo=UTC,
        )

    def events(self, date: dt.date) -> SolarEvents:
        """All sunset-centred events for ``date``."""
        s = self._solar
        set_time = self.sunset_utc(date)
        return SolarEvents(
            date=date,
            sunset=set_time,
            sun_bearing=self.azimuth_at(set_time),
            golden_hour_start=self._at_elevation(date, s.golden_hour_elevation_deg),
            blue_hour_start=self._at_elevation(date, s.blue_hour_start_elevation_deg),
            blue_hour_end=self._at_elevation(date, s.blue_hour_end_elevation_deg),
            civil_dusk=self._at_elevation(date, s.civil_dusk_elevation_deg),
            timezone=self.timezone,
        )

    # -- helpers used by gating / messaging ---------------------------------

    def local_now(self, now: dt.datetime | None = None) -> dt.datetime:
        """Current time in the configured local timezone."""
        moment = dt.datetime.now(tz=UTC) if now is None else now
        if moment.tzinfo is None:
            raise ValueError("naive datetime passed to local_now; always pass tz-aware")
        return moment.astimezone(ZoneInfo(self.timezone))

    def minutes_before_sunset(self, moment: dt.datetime, date: dt.date) -> float:
        """How many minutes ``moment`` falls before sunset on ``date``.

        Positive means before sunset, negative means after.
        """
        return (self.sunset_utc(date) - moment.astimezone(UTC)).total_seconds() / 60.0
