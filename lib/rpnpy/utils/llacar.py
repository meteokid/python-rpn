#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Copyright (C) 2010 Edward d'Auvergne                                        #
#                                                                             #
# This file is part of the program relax (http://www.nmr-relax.com).          #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                             #
###############################################################################

# Module docstring.
"""
Module for transforming between different coordinate systems.
"""

# Python module imports.
import math as _math
_cst_DEG2RAD = _math.pi/180.
_cst_RAD2DEG = 180./_math.pi

def llacar_py(lon, lat):
    """
    Transformation from a set of points in the spherical coordinate
    system to cartesian space

    xyz = llacar_py(lon, lat)

    Args:
        lat, lon: (float, float) [degree]
    Returns:
        (x, y, z) : coordinates in cartesian space (float, float, float)
    Raises:
        TypeError
    """
    rlat = _cst_DEG2RAD*lat
    rlon = _cst_DEG2RAD*lon
    x = _math.cos(rlat) * _math.cos(rlon)
    y = _math.cos(rlat) * _math.sin(rlon)
    z = _math.sin(rlat)
    return (x, y, z)


def cartall_py(xyz):
    """
    Computes the lon, lat positions for a rotated system

    (lon, lat) = cartall_py(xyz)
    Args:
        xyz : rotation matrix (float, float, float)
    Returns:
        (lon, lat), spherical coor of points (float, float) [degree]
    Raises:
        TypeError
   """
    lat = _math.asin(max(-1., min(1., xyz[2]))) * _cst_RAD2DEG
    lon = _math.atan2(xyz[1], xyz[0]) * _cst_RAD2DEG
    lon = lon % 360.
    if lon < 0.:
        lon += 360.
    return (lon, lat)


def cartesian_to_spherical(vector):
    """
    Convert the Cartesian vector [x, y, z]
    to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle,
    and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """
    from numpy import array, float64
    from numpy.linalg import norm

    # The radial distance.
    r = norm(vector)

    # Unit vector.
    unit = vector / r

    # The polar angle.
    theta = _math.acos(unit[2])

    # The azimuth.
    phi = _math.atan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return array([r, theta, phi], float64)


def spherical_to_cartesian(spherical_vect, cart_vect):
    """
    Convert the spherical coordinate vector [r, theta, phi]
    to the Cartesian vector [x, y, z].

    The parameter r is the radial distance,
    theta is the polar angle, and phi is the azimuth.

    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """

    # Trig alias.
    sin_theta = sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * _math.cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * _math.sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * _math.cos(spherical_vect[1])
