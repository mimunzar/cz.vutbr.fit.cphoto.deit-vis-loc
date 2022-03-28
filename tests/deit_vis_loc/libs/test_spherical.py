#!/usr/bin/env python3

from math import pi
import src.deit_vis_loc.libs.spherical as spherical


def test_dist_m():
    tests = [
        {'sphere-lat':  0, 'diff-lat': 111195, 'diff-lon': 111195},
        {'sphere-lat': 15, 'diff-lat': 111195, 'diff-lon': 107406},
        {'sphere-lat': 30, 'diff-lat': 111195, 'diff-lon': 96297},
        {'sphere-lat': 45, 'diff-lat': 111195, 'diff-lon': 78626},
        {'sphere-lat': 60, 'diff-lat': 111195, 'diff-lon': 55597},
        {'sphere-lat': 75, 'diff-lat': 111195, 'diff-lon': 28779},
        {'sphere-lat': 90, 'diff-lat': 111195, 'diff-lon': 0},
        #^ Effect of change one degree of lat/lon on sphere
    ]

    for t in tests:
        latlon1 = (t['sphere-lat'],     0)
        latlon2 = (t['sphere-lat'] + 1, 0)
        assert spherical.dist_m(latlon1, latlon2) == t['diff-lat']
        assert spherical.dist_m(latlon2, latlon1) == t['diff-lat']

        latlon1 = (t['sphere-lat'], 0)
        latlon2 = (t['sphere-lat'], 1)
        assert spherical.dist_m(latlon1, latlon2) == t['diff-lon']
        assert spherical.dist_m(latlon2, latlon1) == t['diff-lon']

    paris  = (48.8566,  2.3522)
    madrid = (40.4168, -3.7038)
    assert spherical.dist_m(paris, paris)  == 0
    assert spherical.dist_m(paris, madrid) == 1052892
    assert spherical.dist_m(madrid, paris) == 1052892


def test_circle_dist_rad():
    assert spherical.circle_dist_rad(2*pi, 2*pi) == 0
    assert spherical.circle_dist_rad(0, 2*pi)    == 0

    assert spherical.circle_dist_rad(pi, 2*pi) == pi
    assert spherical.circle_dist_rad(2*pi, pi) == pi

    assert spherical.circle_dist_rad(0, 2*pi - pi/2) == pi/2
    assert spherical.circle_dist_rad(0, pi/2)        == pi/2

