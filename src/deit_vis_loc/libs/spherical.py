#!/usr/bin/env python3

import math as ma

import src.deit_vis_loc.libs.util as util


EARTH_RADIUS = 6371e3

def dist_m(latlon, latlon_other, radius=EARTH_RADIUS):
    la1, lo1 = map(ma.radians, latlon)
    la2, lo2 = map(ma.radians, latlon_other)
    x = ma.sin(la1)*ma.sin(la2) + ma.cos(la1)*ma.cos(la2)*ma.cos(lo2 - lo1)
    return round(radius*ma.acos(util.clamp(-1, 1, x)))
    #^ http://www.movable-type.co.uk/scripts/latlong.html#cosine-law


def circle_dist_rad(l_rad, r_rad):
    distance = abs(l_rad - r_rad)%(2*ma.pi)
    return 2*ma.pi - distance if distance > ma.pi else distance
    #^ If distance is longer than half circle, there is a shorter way

