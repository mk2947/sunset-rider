"""Geometry tests.

These underpin both the 36-bearing horizon walk and the corridor sample points,
so an error here would silently corrupt every score.
"""

from __future__ import annotations

import pytest

from sunset_rider.geo import (
    angular_difference,
    arc_span,
    bearing_in_arc,
    destination_point,
    elevation_angle,
    haversine_km,
    initial_bearing,
    normalise_bearing,
)

READING = (51.443326, -0.956334)


def test_haversine_matches_known_distance_reading_to_oxford():
    """Reading to Oxford city centre is about 40 km great-circle."""
    oxford = (51.7520, -1.2577)
    assert haversine_km(*READING, *oxford) == pytest.approx(40.0, abs=2.0)


def test_haversine_is_zero_for_identical_points():
    assert haversine_km(*READING, *READING) == pytest.approx(0.0, abs=1e-9)


def test_destination_then_haversine_round_trips():
    """Walking 15 km on a bearing must land 15 km away."""
    for bearing in (0, 45, 137, 235, 310, 359):
        lat, lon = destination_point(*READING, bearing, 15.0)
        assert haversine_km(*READING, lat, lon) == pytest.approx(15.0, abs=0.01)


def test_destination_bearing_round_trips():
    """The initial bearing back to the destination must equal the bearing walked."""
    for bearing in (10.0, 235.0, 310.0):
        lat, lon = destination_point(*READING, bearing, 30.0)
        assert initial_bearing(*READING, lat, lon) == pytest.approx(bearing, abs=0.1)


def test_destination_due_north_increases_latitude_only():
    lat, lon = destination_point(*READING, 0.0, 10.0)
    assert lat > READING[0]
    assert lon == pytest.approx(READING[1], abs=1e-6)


def test_destination_normalises_longitude_across_the_antimeridian():
    lat, lon = destination_point(0.0, 179.5, 90.0, 200.0)
    assert -180.0 <= lon < 180.0
    assert lon < 0, "should have wrapped to a negative longitude"


# -- bearings ---------------------------------------------------------------

def test_normalise_bearing_wraps():
    assert normalise_bearing(370.0) == pytest.approx(10.0)
    assert normalise_bearing(-10.0) == pytest.approx(350.0)


@pytest.mark.parametrize(
    "a, b, expected",
    [(0.0, 10.0, 10.0), (350.0, 10.0, 20.0), (10.0, 350.0, 20.0),
     (0.0, 180.0, 180.0), (0.0, 190.0, 170.0), (310.0, 310.0, 0.0)],
)
def test_angular_difference_takes_the_short_way_round(a, b, expected):
    assert angular_difference(a, b) == pytest.approx(expected)


def test_bearing_in_arc_simple_range():
    assert bearing_in_arc(250.0, 200.0, 330.0)
    assert not bearing_in_arc(100.0, 200.0, 330.0)


def test_bearing_in_arc_handles_wraparound_through_zero():
    """An arc from 340 through 0 to 20 must contain 350 and 10, but not 180."""
    assert bearing_in_arc(350.0, 340.0, 20.0)
    assert bearing_in_arc(10.0, 340.0, 20.0)
    assert bearing_in_arc(340.0, 340.0, 20.0)
    assert not bearing_in_arc(180.0, 340.0, 20.0)


def test_arc_span_handles_wraparound():
    assert arc_span(200.0, 330.0) == pytest.approx(130.0)
    assert arc_span(340.0, 20.0) == pytest.approx(40.0)


# -- elevation angle --------------------------------------------------------

def test_elevation_angle_is_positive_for_higher_terrain():
    """A 100 m hill 1 km away subtends about 5.7 degrees."""
    assert elevation_angle(50.0, 150.0, 1.0) == pytest.approx(5.71, abs=0.01)


def test_elevation_angle_is_negative_when_terrain_falls_away():
    assert elevation_angle(150.0, 50.0, 1.0) < 0


def test_elevation_angle_is_zero_on_flat_ground():
    assert elevation_angle(80.0, 80.0, 4.0) == pytest.approx(0.0)


def test_elevation_angle_falls_off_with_distance():
    """The same hill matters less further away — why the 15 km sample is the weakest."""
    near = elevation_angle(0.0, 100.0, 1.0)
    far = elevation_angle(0.0, 100.0, 15.0)
    assert near > far > 0


def test_elevation_angle_rejects_zero_distance():
    with pytest.raises(ValueError):
        elevation_angle(0.0, 100.0, 0.0)
