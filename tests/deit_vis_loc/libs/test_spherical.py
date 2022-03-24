#!/usr/bin/env python3

from math import pi
import src.deit_vis_loc.libs.spherical as spherical


def test_distance_m():
    tests = [
        {'sphere-lat':  0, 'diff-lat': 111194, 'diff-lon': 111194},
        {'sphere-lat': 15, 'diff-lat': 111194, 'diff-lon': 107405},
        {'sphere-lat': 30, 'diff-lat': 111194, 'diff-lon': 96297},
        {'sphere-lat': 45, 'diff-lat': 111194, 'diff-lon': 78626},
        {'sphere-lat': 60, 'diff-lat': 111194, 'diff-lon': 55596},
        {'sphere-lat': 75, 'diff-lat': 111194, 'diff-lon': 28779},
        {'sphere-lat': 90, 'diff-lat': 111194, 'diff-lon': 0},
        #^ Effect of change one degree of lat/lon on sphere
    ]

    def assert_distance(coord1, coord2, result):
        assert spherical.distance_m(coord1, coord2) == result
        assert spherical.distance_m(coord2, coord1) == result

    for t in tests:
        coord1 = {'lat': t['sphere-lat'],     'lon': 0}
        coord2 = {'lat': t['sphere-lat'] + 1, 'lon': 0}
        assert_distance(coord1, coord2, t['diff-lat'])

        coord1 = {'lat': t['sphere-lat'], 'lon': 0}
        coord2 = {'lat': t['sphere-lat'], 'lon': 1}
        assert_distance(coord1, coord2, t['diff-lon'])

    paris  = {'lat': 48.8566, 'lon':  2.3522}
    madrid = {'lat': 40.4168, 'lon': -3.7038}
    assert_distance(paris, madrid, 1052892)


def test_azimut_deg():
    coord1 = {'lat': 0, 'lon': 0}
    coord2 = {'lat': 1, 'lon': 0}
    assert round(spherical.azimut_deg(coord1, coord2)) == 0
    assert round(spherical.azimut_deg(coord2, coord1)) == 180

    coord1 = {'lat': 0, 'lon': 0}
    coord2 = {'lat': 0, 'lon': 1}
    assert round(spherical.azimut_deg(coord1, coord2)) == 90
    assert round(spherical.azimut_deg(coord2, coord1)) == 270

    coord1 = {'lat': 0, 'lon': 0}
    coord2 = {'lat': 1, 'lon': 1}
    assert round(spherical.azimut_deg(coord1, coord2)) == 45
    assert round(spherical.azimut_deg(coord2, coord1)) == 225

    paris  = {'lat': 48.8566, 'lon':  2.3522}
    madrid = {'lat': 40.4168, 'lon': -3.7038}
    assert round(spherical.azimut_deg(paris, madrid), 4) == 209.2255
    assert round(spherical.azimut_deg(madrid, paris), 4) == 24.9569


def test_endpoint():
    result = spherical.endpoint({'lat': 0, 'lon': 0}, 0, 111194)
    assert round(result['lat'], 4) == 1
    assert round(result['lon'], 4) == 0.0

    result = spherical.endpoint({'lat': 0, 'lon': 0}, 0, -111194)
    assert round(result['lat'], 4) == -1
    assert round(result['lon'], 4) == 0.0

    result = spherical.endpoint({'lat': 0, 'lon': 0}, 90, 111194)
    assert round(result['lat'], 4) == 0.0
    assert round(result['lon'], 4) == 1

    result = spherical.endpoint({'lat': 0, 'lon': 0}, 90, -111194)
    assert round(result['lat'], 4) == 0.0
    assert round(result['lon'], 4) == -1

    paris  = {'lat': 48.8566, 'lon':  2.3522}
    madrid = {'lat': 40.4168, 'lon': -3.7038}
    result = spherical.endpoint(paris, 209.2255, 1052892)
    assert round(result['lat'], 4) == madrid['lat']
    assert round(result['lon'], 4) == madrid['lon']


def test_circle_difference_rad():
    assert spherical.circle_diff_rad(2*pi, 2*pi) == 0
    assert spherical.circle_diff_rad(0, 2*pi)    == 0

    assert spherical.circle_diff_rad(pi, 2*pi) == pi
    assert spherical.circle_diff_rad(2*pi, pi) == pi

    assert spherical.circle_diff_rad(0, 2*pi - pi/2) == pi/2
    assert spherical.circle_diff_rad(0, pi/2)        == pi/2

