"""Shared result types.

Kept in their own module so ``message`` can render them and ``pipeline`` can build
them without importing each other.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .gating import BlockerResult
from .scoring import ModeResult, OutputScores, RideResult
from .solar import SolarEvents


@dataclass
class SpotForecast:
    """Everything known about one viewpoint on one evening."""

    viewpoint: dict[str, Any]
    events: SolarEvents
    sky: ModeResult
    output: OutputScores
    ride: RideResult
    blockers: BlockerResult
    worth_it: float
    worth_it_parts: dict[str, float]
    leave_by: dt.datetime
    minutes_one_way: float
    max_gust_kmh: float
    wind_500hpa: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.viewpoint.get("name", "unnamed"))

    @property
    def blocked(self) -> bool:
        return self.blockers.blocked


@dataclass
class EveningOutlook:
    """Ensemble view of one evening, for `plan` mode.

    Deliberately holds no single headline score. At 72 hours the honest output is a
    distribution: how many members clear "good", and how tightly they agree.
    """

    date: dt.date
    events: SolarEvents
    member_skies: list[float]
    stats: dict[str, float]
    probability_above_good: float
    mode_counts: dict[str, int]
    clearing_front_members: int
    member_count: int

    @property
    def dominant_mode(self) -> str:
        if not self.mode_counts:
            return "unknown"
        return max(self.mode_counts, key=lambda k: self.mode_counts[k])


@dataclass
class RunResult:
    """What a single pipeline run produced."""

    mode: str
    target_date: dt.date
    spots: list[SpotForecast] = field(default_factory=list)
    outlooks: list[EveningOutlook] = field(default_factory=list)
    regional_sky: float = 0.0
    max_radius_km: float = 0.0
    excluded: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
