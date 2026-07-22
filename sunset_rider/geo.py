"""Spherical geometry helpers.

Used by ``discovery`` to walk out along 36 bearings when profiling a horizon, and
by ``scoring`` to place the sun-to-sky corridor sample points. Bearings are
degrees clockwise from true north throughout, matching astral's azimuth convention.
"""

from __future__ import annotations

import math

# Mean Earth radius. Passed in from config at call sites that have one; this
# default exists so the pure-geometry functions stay independently testable.
DEFAULT_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float,
                 earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM) -> float:
    """Great-circle distance between two points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float,
                      earth_radius_km: float = DEFAULT_EARTH_RADIUS_KM
                      ) -> tuple[float, float]:
    """Point reached by travelling ``distance_km`` from (lat, lon) on ``bearing_deg``.

    Standard great-circle direct solution. Returns (latitude, longitude) degrees,
    longitude normalised to [-180, 180).
    """
    angular = distance_km / earth_radius_km
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    sin_phi2 = (math.sin(phi1) * math.cos(angular)
                + math.cos(phi1) * math.sin(angular) * math.cos(theta))
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    y = math.sin(theta) * math.sin(angular) * math.cos(phi1)
    x = math.cos(angular) - math.sin(phi1) * sin_phi2
    lambda2 = lambda1 + math.atan2(y, x)

    lon_out = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return math.degrees(phi2), lon_out


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def normalise_bearing(bearing: float) -> float:
    """Wrap a bearing into [0, 360)."""
    return bearing % 360.0


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute angle between two bearings, in degrees (0..180)."""
    diff = abs(normalise_bearing(a) - normalise_bearing(b)) % 360.0
    return min(diff, 360.0 - diff)


def bearing_in_arc(bearing: float, arc_start: float, arc_end: float) -> bool:
    """Whether ``bearing`` lies inside the arc running clockwise from start to end.

    Handles arcs that wrap through 0 deg. An arc where start == end is treated as
    a single bearing, not a full circle.
    """
    b = normalise_bearing(bearing)
    start = normalise_bearing(arc_start)
    end = normalise_bearing(arc_end)
    if start <= end:
        return start <= b <= end
    return b >= start or b <= end


def arc_span(arc_start: float, arc_end: float) -> float:
    """Angular width of a clockwise arc, in degrees."""
    return (normalise_bearing(arc_end) - normalise_bearing(arc_start)) % 360.0


def elevation_angle(observer_elev_m: float, sample_elev_m: float,
                    distance_km: float) -> float:
    """Elevation angle from an observer to a terrain sample, in degrees.

    Positive means the sample rises above the observer's eye level and therefore
    obstructs the horizon on that bearing.
    """
    if distance_km <= 0:
        raise ValueError("distance_km must be positive")
    return math.degrees(math.atan2(sample_elev_m - observer_elev_m, distance_km * 1000.0))
