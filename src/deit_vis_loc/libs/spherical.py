#!/usr/bin/env python3

import functools as ft
import math      as ma

import src.deit_vis_loc.libs.util as util


pluck_coords = ft.partial(util.pluck, ['lat', 'lon'])
EARTH_RADIUS = 6371e3


def distance_m(ll, ll_other, radius=EARTH_RADIUS):
    la1, lo1 = map(ma.radians, pluck_coords(ll))
    la2, lo2 = map(ma.radians, pluck_coords(ll_other))
    return int(radius*ma.acos(
            ma.sin(la1)*ma.sin(la2) + ma.cos(la1)*ma.cos(la2)*ma.cos(lo2 - lo1)))
    #^ http://www.movable-type.co.uk/scripts/latlong.html#cosine-law


def azimut_deg(ll, ll_other):
    la1, lo1 = map(ma.radians, pluck_coords(ll))
    la2, lo2 = map(ma.radians, pluck_coords(ll_other))
    x = ma.sin(lo2 - lo1)*ma.cos(la2)
    y = ma.cos(la1)*ma.sin(la2) - ma.sin(la1)*ma.cos(la2)*ma.cos(lo2-lo1)
    return (360 + ma.degrees(ma.atan2(x, y)))%360;
    #^ http://www.movable-type.co.uk/scripts/latlong.html#bearing


def endpoint(ll, azimut_deg, distance_m, radius=EARTH_RADIUS):
    la1, lo1 = map(ma.radians, pluck_coords(ll))
    dr  = distance_m/radius
    lat = ma.asin(ma.sin(la1)*ma.cos(dr)
          + ma.cos(la1)*ma.sin(dr)*ma.cos(ma.radians(azimut_deg)))
    lon = lo1 + ma.atan2(ma.sin(ma.radians(azimut_deg))*ma.sin(dr)*ma.cos(la1),
            ma.cos(dr) - ma.sin(la1)*ma.sin(lat))
    return {'lat': ma.degrees(lat), 'lon': ma.degrees(lon)}
    #^ http://www.movable-type.co.uk/scripts/latlong.html#dest-point


def circle_diff_rad(l_rad, r_rad):
    distance = abs(l_rad - r_rad) % (2*ma.pi)
    return 2*ma.pi - distance if distance > ma.pi else distance
    #^ If distance is longer than half circle, there is a shorter way

