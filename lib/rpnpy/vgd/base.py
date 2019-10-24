#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd.base contains python wrapper to main vgrid C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[Vgrid]] package.<br>
    You may want to refer to the [[Vgrid]] documentation for more details.

See Also:
    rpnpy.vgd.proto
    rpnpy.vgd.const
"""
import ctypes as _ct
import numpy  as _np
# import numpy.ctypeslib as _npc
from rpnpy.vgd import proto as _vp
from rpnpy.vgd import const as _vc
from rpnpy.vgd import VGDError
import rpnpy.librmn.all as _rmn

from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR as _C_MKSTR

_MB2PA = 100.

## VGD_TYPE_CODE(kind, version): Compute VGD type code from kind and version.
VGD_TYPE_CODE = lambda k, v: int(k) * 1000 + int(v)


def vgd_new_sigm(hyb, ip1=-1, ip2=-1):
    """
    Build a Sigma (1001) based VGridDescriptor initialized with provided info.

    Args:
        hyb      (list): list of sigma level values
        ip1      (int) : Ip1 of the vgrid record
        ip2      (int) : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls = (0.980000, 0.993000, 1.000000)
    >>> try:
    ...     myvgd = vgd.vgd_new_sigm(lvls)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    vgd_put_opt('ALLOW_SIGMA', _vc.VGD_ALLOW_SIGMA)
    (kind, version) = _vc.VGD_KIND_VER['sigm']
    return vgd_new(kind, version, hyb, ip1=ip1, ip2=ip2)


vgd_new_1001 = vgd_new_sigm


def vgd_new_pres(pres, ip1=-1, ip2=-1):
    """
    Build a Pressure (2001) based VGridDescriptor initialized with provided info.

    Args:
        pres     (list): list of pressure level values [hPa]
        ip1      (int) : Ip1 of the vgrid record
        ip2      (int) : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['pres']
    return vgd_new(kind, version, hyb=pres, ip1=ip1, ip2=ip2)


vgd_new_2001 = vgd_new_pres


def vgd_new_eta(hyb, ptop, ip1=-1, ip2=-1):
    """
    Build an Eta (1002) based VGridDescriptor initialized with provided info.

    Args:
        hyb      (list) : list of Eta level values
        ptop     (float): Top level pressure [Pa]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.000,   0.011,    0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.993,    1.000)
    >>> ptop  = 1000.
    >>> try:
    ...     myvgd = vgd.vgd_new_eta(lvls, ptop)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['eta']
    return vgd_new(kind, version, hyb=hyb, ptop=ptop, ip1=ip1, ip2=ip2)


vgd_new_1002 = vgd_new_eta


## def vgd_new_hybn(hyb, rcoef1, ptop, pref, ip1=-1, ip2=-1):
##     """
##     """
##     (kind, version) = _vc.VGD_KIND_VER['hybn']
##     return vgd_new(kind, version, hyb=hyb,
##                    rcoef1=rcoef1, ptop=ptop, pref=pref, ip1=ip1, ip2=ip2)
## vgd_new_1003 = vgd_new_hybn


def vgd_new_hyb(hyb, rcoef1, ptop, pref, ip1=-1, ip2=-1):
    """
    Build an Hybrid Un-staggered (5001) VGridDescriptor initialized with provided info.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
        pref     (float): Reference level pressure [Pa]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.011,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.993,    1.000)
    >>> rcoef1 = 1.6
    >>> ptop   = 110.
    >>> pref   = 80000.
    >>> try:
    ...     myvgd = vgd.vgd_new_hyb(lvls, rcoef1, ptop, pref)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hyb']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, ptop=ptop, pref=pref, ip1=ip1, ip2=ip2)


vgd_new_5001 = vgd_new_hyb


def vgd_new_hybs(hyb, rcoef1, rcoef2, ptop, pref, ip1=-1, ip2=-1):
    """
    Build an Hybrid Staggered (5002) VGridDescriptor initialized with provided info.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
        pref     (float): Reference level pressure [Pa]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 10.
    >>> ptop   = 805.
    >>> pref   = 100000.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybs(lvls, rcoef1, rcoef2, ptop, pref)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybs']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, ptop=ptop, pref=pref,
                   ip1=ip1, ip2=ip2)


vgd_new_5002 = vgd_new_hybs


def vgd_new_hybt(hyb, rcoef1, rcoef2, ptop, pref, ip1=-1, ip2=-1):
    """
    Build an Hybrid Staggered (5003) VGridDescriptor initialized with provided info.

    First level is a thermo level, unstaggered last Thermo level.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
        pref     (float): Reference level pressure [Pa]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 10.
    >>> ptop   = 1000.
    >>> pref   = 100000.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybt(lvls, rcoef1, rcoef2, ptop, pref)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybt']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, ptop=ptop, pref=pref,
                   ip1=ip1, ip2=ip2)


vgd_new_5003 = vgd_new_hybt


def vgd_new_hybm(hyb, rcoef1, rcoef2, ptop, pref, ip1=-1, ip2=-1):
    """
    Build an Hybrid Staggered (5004) VGridDescriptor initialized with provided info.

    First level is a momentum level, same number of thermo and momentum levels.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
                          Possible values -1, -2 or any pressure value.
                          Recommended value is -2 which gives a flat
                          first (top) momentum level [B(1)=0]
        pref     (float): Reference level pressure [Pa]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> ptop   = -1.
    >>> pref   = 100000.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybm(lvls, rcoef1, rcoef2, ptop, pref)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybm']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, ptop=ptop, pref=pref,
                   ip1=ip1, ip2=ip2)


vgd_new_5004 = vgd_new_hybm


def vgd_new_hybmd(hyb, rcoef1, rcoef2, pref, dhm, dht,
                  ip1=-1, ip2=-1):
    """
    Build an Hybrid Staggered (5005) VGridDescriptor initialized with provided info.

    First level is a momentum level, same number of thermo and momentum levels.
    Diag level heights (m AGL) encoded.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        pref     (float): Reference level pressure [Pa]
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybmd']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, pref=pref,
                   dhm=dhm, dht=dht, ip1=ip1, ip2=ip2)


vgd_new_5005 = vgd_new_hybmd


def vgd_new_hybps(hyb, rcoef1, rcoef2, rcoef3, rcoef4, pref, dhm, dht,
                  ip1=-1, ip2=-1):
    """
    Build an Hybrid Presure Staggered SLEVE (5100) VGridDescriptor initialized
    with provided info.

    Args:
        hyb      (list) : list of hybrid level values
        rcoef1   (float): 1st large scale Coordinate recification R-coefficient
        rcoef2   (float): 2nd large scale Coordinate recification R-coefficient
        rcoef3   (float): 1st small scale Coordinate recification R-coefficient
        rcoef4   (float): 2nd small scale Coordinate recification R-coefficient
        pref     (float): Reference level pressure [Pa]
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 5.
    >>> rcoef3 = 0.
    >>> rcoef4 = 100.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybps(lvls, rcoef1, rcoef2, rcoef2, rcoef4,
    ...                               pref, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybps']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, rcoef3=rcoef3, rcoef4=rcoef4,
                   pref=pref, dhm=dhm, dht=dht, ip1=ip1, ip2=ip2)


vgd_new_5100 = vgd_new_hybps


def vgd_new_hybh(hyb, rcoef1, rcoef2, dhm, dht, ip1=-1, ip2=-1):
    """
    Build an Hybrid CP Staggered height (21001) VGridDescriptor initialized
    with provided info.

    Args:
        hyb      (list) : list of hybrid height level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (30968.,  24944., 20493., 16765., 13525., 10814.,  8026., 5477.,
    ...          3488., 1842., 880., 0.)
    >>> rcoef1 = 0.
    >>> rcoef2 = 5.
    >>> dhm    = 10.
    >>> dht    = 1.5
    >>> try:
    ...     myvgd = vgd.vgd_new_hybh(lvls, rcoef1, rcoef2, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybh']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, dhm=dhm, dht=dht, ip1=ip1,
                   ip2=ip2)


vgd_new_21001 = vgd_new_hybh


def vgd_new_hybhs(hyb, rcoef1, rcoef2, rcoef3, rcoef4, dhm, dht, ip1=-1,
                  ip2=-1):
    """
    Build an Hybrid CP Staggered height SLEVE (21001) VGridDescriptor
    initialized with provided info.

    Args:
        hyb      (list) : list of hybrid height level values
        rcoef1   (float): 1st large scale coordinate recification R-coefficient
        rcoef2   (float): 2nd large scale coordinate recification R-coefficient
        rcoef3   (float): 1st small scale coordinate recification R-coefficient
        rcoef4   (float): 2nd small scale Coordinate recification R-coefficient
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (30968.,  24944., 20493., 16765., 13525., 10814.,  8026., 5477.,
    ...          3488., 1842., 880., 0.)
    >>> rcoef1 = 0.
    >>> rcoef2 = 5.
    >>> rcoef3 = 0.
    >>> rcoef4 = 100.
    >>> dhm    = 10.
    >>> dht    = 1.5
    >>> try:
    ...     myvgd = vgd.vgd_new_hybhs(lvls, rcoef1, rcoef2, rcoef3, rcoef4,
    ...                               dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybhs']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, rcoef3=rcoef3, rcoef4=rcoef4,
                   dhm=dhm, dht=dht, ip1=ip1, ip2=ip2)


vgd_new_21001_SLEVE = vgd_new_hybhs


def vgd_new_hybhl(hyb, rcoef1, rcoef2, dhm, dht, dhw, ip1=-1, ip2=-1):
    """
    Build an Hybrid Lorenz Staggered height (21002) VGridDescriptor
    initialized with provided info.

    Args:
        hyb      (list) : list of hybrid height level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        dhw      (float): Height of the Diagnostic vertical velocyty level
                          [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (30968.,  24944., 20493., 16765., 13525., 10814.,  8026., 5477.,
    ...          3488., 1842., 880., 0.)
    >>> rcoef1 = 0.
    >>> rcoef2 = 5.
    >>> dhm    = 10.
    >>> dht    = 1.5
    >>> dhw    = 10.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybhl(lvls, rcoef1, rcoef2, dhm, dht, dhw)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybhl']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, dhm=dhm, dht=dht, dhw=dhw,
                   ip1=ip1, ip2=ip2)


vgd_new_21002 = vgd_new_hybhl


def vgd_new_hybhls(hyb, rcoef1, rcoef2, rcoef3, rcoef4, dhm, dht, dhw, ip1=-1,
                   ip2=-1):
    """
    Build an Hybrid Lorenz Staggered SLEVE (21001) VGridDescriptor initialized
    with provided info.

    Args:
        hyb      (list) : list of hybrid height level values
        rcoef1   (float): 1st large scale coordinate recification R-coefficient
        rcoef2   (float): 2nd large scale coordinate recification R-coefficient
        rcoef3   (float): 1st small scale coordinate recification R-coefficient
        rcoef4   (float): 2nd small scale Coordinate recification R-coefficient
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        dhw      (float): Height of the Diagnostic vertical velocity level
                          [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (30968.,  24944., 20493., 16765., 13525., 10814.,  8026., 5477.,
    ...          3488., 1842., 880., 0.)
    >>> rcoef1 = 0.
    >>> rcoef2 = 5.
    >>> rcoef3 = 0.
    >>> rcoef4 = 100.
    >>> dhm    = 10.
    >>> dht    = 1.5
    >>> dhw    = 10.
    >>> try:
    ...     myvgd = vgd.vgd_new_hybhls(lvls, rcoef1, rcoef2, rcoef3, rcoef4,
    ...                                dhm, dht, dhw)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    (kind, version) = _vc.VGD_KIND_VER['hybhls']
    return vgd_new(kind, version, hyb=hyb,
                   rcoef1=rcoef1, rcoef2=rcoef2, rcoef3=rcoef3, rcoef4=rcoef4,
                   dhm=dhm, dht=dht, dhw=dhw, ip1=ip1, ip2=ip2)


vgd_new_21002_SLEVE = vgd_new_hybhls


def vgd_new(kind, version, hyb,
            rcoef1=None, rcoef2=None, ptop=None, pref=None, dhm=None, dht=None,
            ip1=-1, ip2=-1, rcoef3=None, rcoef4=None, dhw=None, avg=0):
    """
    General function to Build an VGridDescriptor initialized with provided info.

    Deprecated; see vgd_new2 for arguments description.
    Kept for backward compatibility of arguments order.

    See Also:
        vgd_new2
    """
    return vgd_new2(kind, version, hyb, rcoef1, rcoef2, rcoef3, rcoef4,
                    ptop, pref, dhm, dht, dhw, ip1, ip2, avg)


def vgd_new2(kind, version, hyb,
             rcoef1=None, rcoef2=None, rcoef3=None, rcoef4=None, ptop=None,
             pref=None, dhm=None, dht=None, dhw=None, ip1=-1, ip2=-1, avg=0):
    """
    General function to Build an VGridDescriptor initialized with provided info.

    Args:
        kind     (int)  : Kind of vertical coor
        version  (int)  : Version of vertical coor
        hyb      (list) : list of level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef2   (float): 2nd Coordinate recification R-coefficient
        rcoef3   (float): third Coordinate recification R-coefficient
        rcoef4   (float): fourth Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
        pref     (float): Reference level pressure [Pa]
        dhm      (float): Height of the Diagnostic Momentum level [m AGL]
        dht      (float): Height of the Diagnostic Thermodynamic level [m AGL]
        dhw      (float): Height of the Diagnostic vertical velocity level
                          [m AGL]
        ip1      (int)  : Ip1 of the vgrid record
        ip2      (int)  : Ip2 of the vgrid record
        avg      (int)  : if avg=1 last thermo level is in middle 5100 only
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> (kind, version) = vgd.VGD_KIND_VER['hybmd']
    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    ...     myvgd = vgd.vgd_new(kind, version, lvls, rcoef1, rcoef2, pref=pref,
    ...                         dhm=dhm, dht=dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    See Also:
        rpnpy.vgd.const.VGD_KIND_VER
        vgd_new_sigm
        vgd_new_pres
        vgd_new_eta
        vgd_new_hyb
        vgd_new_hybs
        vgd_new_hybt
        vgd_new_hybm
        vgd_new_hybmd
        vgd_new_hybps
        vgd_new_hybh
        vgd_new_hybhs
        vgd_new_hybhl
        vgd_new_hybhls
        vgd_write
        vgd_levels
        vgd_free
    """
    if isinstance(hyb, (list, tuple)):
        hyb = _np.array(hyb, copy=True, dtype=_np.float32, order='F')
    elif isinstance(hyb, _np.ndarray):
        hyb = _np.array(hyb.flatten(), copy=True, dtype=_np.float32,
                        order='F')
    else:
        raise TypeError('hyb should be list or ndarray: {0}'.
                        format(str(type(hyb))))
    if not rcoef1 is None:
        rcoef1 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef1))
    if not rcoef2 is None:
        rcoef2 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef2))
    if not rcoef3 is None:
        rcoef3 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef3))
    if not rcoef4 is None:
        rcoef4 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef4))
    if not ptop is None:
        ptop = _ct.POINTER(_ct.c_double)(_ct.c_double(ptop))
    if not pref is None:
        pref = _ct.POINTER(_ct.c_double)(_ct.c_double(pref))
    if not dhm is None:
        dhm = _ct.POINTER(_ct.c_float)(_ct.c_float(dhm))
    if not dht is None:
        dht = _ct.POINTER(_ct.c_float)(_ct.c_float(dht))
    if not dhw is None:
        dhw = _ct.POINTER(_ct.c_float)(_ct.c_float(dhw))
    vgd_ptr = _vp.c_vgd_construct()
    p_ptop_out = None
    if kind == 5 and version in (4, 5, 100):
        if not ptop is None:
            p_ptop_out = ptop
        else:
            ptop_out = 100.
            p_ptop_out = _ct.POINTER(_ct.c_double)(_ct.c_double(ptop_out))
    ok = _vp.c_vgd_new_gen2(vgd_ptr, kind, version, hyb, hyb.size, rcoef1,
                            rcoef2, rcoef3, rcoef4, ptop, pref, p_ptop_out,
                            ip1, ip2, dhm, dht, dhw, avg)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem building VGD (kind={0}, version={1}): Error={2})'.
                       format(kind, version, ok))
    return vgd_ptr


## def vgd_new_vert(kind, version, nk,
##                  rcoef1=1, rcoef2=1, ptop=1., pref=1.,
##                  a_m_8=None, b_m_8=None, a_t_8=None, b_t_8=None,
##                  ip1_m=None, ip1_t=None, ip1=-1, ip2=-1):
##     """
##     """
##     raise VGDError('Not Implemented yet')


def vgd_read(fileId, ip1=-1, ip2=-1, kind=-1, version=-1):
    """
    Construct a vgrid descriptor from the vgrid record in a RPN standard file.

    Args:
        fileId   (int)  : Opened RPN Std file unit number
        ip1      (int)  : Ip1 of the vgrid record to find, use -1 for any (I)
        ip2      (int)  : Ip2 of the vgrid record to find, use -1 for any (I)
        kind     (int)  : Kind of vertical coor
        version  (int)  : Version of vertical coor
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import os, os.path, sys
    >>> import rpnpy.vgd.all as vgd
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> fileId = rmn.fstopenall(fileName, rmn.FST_RO)
    >>> try:
    ...     myvgd = vgd.vgd_read(fileId)
    ... except:
    ...     sys.stderr.write("There was a problem reading the VGridDescriptor")
    ... finally:
    ...     rmn.fstcloseall(fileId)

    See Also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.const.FST_RO
        rpnpy.vgd.const.VGD_KIND_VER
        vgd_new
        vgd_write
        vgd_levels
        vgd_free
    """
    vgd_ptr = _vp.c_vgd_construct()
    ok = _vp.c_vgd_new_read(vgd_ptr, fileId, ip1, ip2, kind, version)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting vgd from file ' +
                       '(id={0}, ip1={1}, ip2={2}, kind={3}, version={4})'.
                       format(fileId, ip1, ip2, kind, version))
    return vgd_ptr


def vgd_write(vgd_ptr, fileId):
    """
    Write a vgrid descriptor in a previously opened RPN standard file.

    Args:
        vgd_ptr  (VGridDescriptor ref):
                          Reference/Pointer to the VGridDescriptor to write
        fileId   (int)  : Opened RPN Std file unit number
    Returns:
        None
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> import rpnpy.librmn.all as rmn
    >>> lvls = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> fileName = 'myfstfile.fst'
    >>> fileId   = rmn.fstopenall(fileName, rmn.FST_RW)
    >>> try:
    ...     vgd.vgd_write(myvgd, fileId)
    ... except:
    ...     sys.stderr.write("There was a problem writing the VGridDescriptor")
    ... finally:
    ...     rmn.fstcloseall(fileId)

    See Also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.const.FST_RW
        rpnpy.vgd.const.VGD_KIND_VER
        vgd_new
        vgd_read
        vgd_levels
        vgd_free
    """
    ok = _vp.c_vgd_write_desc(vgd_ptr, fileId)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem writing vgd to file (id={0})'.format(fileId))
    return


def vgd_free(vgd_ptr):
    """
    Free memory from previously created vgrid descriptor.

    Args:
        vgd_ptr (VGridDescriptor ref):
                          Reference/Pointer to the VGridDescriptor to free
    Returns:
        None
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> ## Do some work with myvgd
    >>> vgd.vgd_free(myvgd)

    See Also:
        vgd_new
        vgd_read
    """
    _vp.c_vgd_free(vgd_ptr)
    return


def vgd_tolist(vgd_ptr):
    """
    Get a previously created vgrid descriptor as an encoded list of values for pickling

    Args:
        vgd_ptr (VGridDescriptor ref):
                          Reference/Pointer to the VGridDescriptor
    Returns:
        list : Encoded VGridDescriptor values in a list
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> try:
    ...     vgdtable = vgd.vgd_tolist(myvgd)
    ... except:
    ...     sys.stderr.write("There was a problem encoding the VGridDescriptor in a list")

    See Also:
        vgd_fromlist
        vgd_new
        vgd_read
        vgd_levels
        vgd_free
    """
    return vgd_get(vgd_ptr, 'VTBL')


def vgd_fromlist(vgd_table):
    """
    Build a VGridDescriptor from previously encoded table values.

    Args:
        vgd_table (list): Encoded VGridDescriptor values in a list
                          This is obtained with vgd_tolist
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    ...     myvgd    = vgd.vgd_new_pres(lvls)
    ...     vgdtable = vgd.vgd_tolist(myvgd)
    ... except:
    ...     sys.stderr.write("There was a problem creating/encoding the VGridDescriptor")
    ...     sys.exit(1)
    >>> ## ...
    >>> try:
    ...     myvgd2 = vgd.vgd_fromlist(vgdtable)
    ... except:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor in a list")

    See Also:
        vgd_tolist
        vgd_new
        vgd_read
        vgd_levels
        vgd_free
    """
    vgd_ptr = _vp.c_vgd_construct()
    vtbl = vgd_table.ctypes.data_as(_ct.POINTER(_ct.c_double))
    (ni, nj, nk) = vgd_table.shape
    ok = _vp.c_vgd_new_from_table(vgd_ptr, vtbl, ni, nj, nk)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem Creating a new VGD from provided table)')
    return vgd_ptr


def vgd_copy(vgd_ptr):
    """
    Deep copy of a VGridDescriptor ref object.

    Args:
        vgd_ptr (VGridDescriptor ref):
                      Reference/Pointer to the VGridDescriptor
    Returns:
        VGridDescriptor ref
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except:
    ...     sys.stderr.write("There was a problem creating/encoding the VGridDescriptor")
    ...     sys.exit(1)
    >>> ## ...
    >>> try:
    ...     myvgd2 = vgd.vgd_copy(myvgd)
    ... except:
    ...     sys.stderr.write("There was a problem copying the VGridDescriptor")

    See Also:
        vgd_tolist
        vgd_fromlist
        vgd_cmp
        vgd_levels
        vgd_free
    """
    vgdtable = vgd_tolist(vgd_ptr)
    return vgd_fromlist(vgdtable)


def vgd_get_opt(key, quiet=1):
    """
    Get a vgd package global option value.

    Args:
        key   (int) : Global option name
                      Possible values: 'ALLOW_SIGMA'
        quiet (int) : Quite mode on off
                      1 for quiet; 0 for verbose
    Returns:
        mixed type: option value, type depends on option name
    Raises:
        TypeError
        KeyError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> try:
    ...     allow_signma = vgd.vgd_get_opt('ALLOW_SIGMA')
    ... except:
    ...     sys.stderr.write("There was a problem getting vgd gloabl option")
    >>> print("allow_signma={}".format(allow_signma))
    allow_signma=0

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        rpnpy.vgd.const.VGD_ALLOW_SIGMA
        rpnpy.vgd.const.VGD_DISALLOW_SIGMA
        vgd_put_opt
        vgd_put
        vgd_get
    """
    key2 = key.upper()
    if not key2 in _vc.VGD_OPR_KEYS['getopt_int']:
        raise KeyError('Problem getting opt, invalid key (key={0})'.format(key))
    v1 = _ct.c_int(0)
    ok = _vp.c_vgd_getopt_int(_C_WCHAR2CHAR(key2), _ct.byref(v1), quiet)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting opt (key={0})'.format(key))
    if isinstance(v1.value, bytes):
        return _C_CHAR2WCHAR(v1.value)
    else:
        return v1.value


def vgd_put_opt(key, value):
    """
    Set a vgd package global option value.

    Args:
        key   (int)   : Global option name
                        Possible values: 'ALLOW_SIGMA'
        value (mixed) : Value to be set
                        Type of values depends on option name
    Returns:
        None
    Raises:
        TypeError
        KeyError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> try:
    ...     vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
    ... except:
    ...     sys.stderr.write("There was a problem setting vgd gloabl option")

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        rpnpy.vgd.const.VGD_ALLOW_SIGMA
        rpnpy.vgd.const.VGD_DISALLOW_SIGMA
        vgd_get_opt
        vgd_put
        vgd_get
    """
    key2 = key.upper()
    if not key2 in _vc.VGD_OPR_KEYS['putopt_int']:
        raise VGDError('Problem setting opt, invalid key (key={0})'.format(key))
    ok = _vp.c_vgd_putopt_int(_C_WCHAR2CHAR(key2), value)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem setting opt (key={0}, value={1})'.
                       format(key, repr(value)))
    return


def vgd_get(vgd_ptr, key, quiet=1, defaultOnFail=False, defaultValue=None):
    """
    Get a vgd object parameter value

    Args:
        vgd_ptr (VGridDescriptor ref):
                        Reference/Pointer to the VGridDescriptor
        key   (str)   : Parameter name
                        Possible values: see VGD_KEYS, VGD_OPR_KEYS
        quiet (int)   : Quite mode on off
                        1 for quiet; 0 for verbose
        defaultOnFail : return default instead of raising VGDError
                        Does not prevent KeyError
        defaultValue  : default value to return if defaultOnFail and VGDError
    Returns:
        mixed type: option value, type depends on option name
    Raises:
        TypeError
        KeyError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> try:
    ...     vkind = vgd.vgd_get(myvgd, 'KIND')
    ... except:
    ...     sys.stderr.write("There was a problem getting vgd parameter value")
    >>> try:
    ...     rfld = vgd.vgd_get(myvgd, 'RFLD')
    ...     sys.stderr.write("vgd_get RFLD should have raised an error.")
    ... except VGDError:
    ...     rfld = -1
    ...     print("rfld = {}".format(rfld))
    ... except:
    ...     sys.stderr.write("vgd_get RFLD should have raised VGDError.")
    rfld = -1
    >>> rfld = vgd.vgd_get(myvgd, 'RFLD', defaultOnFail=True, defaultValue=-1)
    >>> print("rfld = {}".format(rfld))
    rfld = -1
    >>> try:
    ...     rfld = vgd.vgd_get(myvgd, 'AAAA')
    ...     sys.stderr.write("vgd_get AAAA should have raised an error.")
    ... except KeyError:
    ...     print("No such VGD key: AAAA")
    ... except:
    ...     sys.stderr.write("vgd_get AAAA should have raised KeyError.")
    No such VGD key: AAAA

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        vgd_put
        vgd_get_opt
        vgd_put_opt
    """
    v1 = None
    key2 = key.upper()[0:4]
    key2b = _C_WCHAR2CHAR(key2)
    if key2 in _vc.VGD_OPR_KEYS['get_char']:
        v1 = _C_MKSTR(' ' * _vc.VGD_MAXSTR_ETIKET)
        ok = _vp.c_vgd_get_char(vgd_ptr, key2b, v1, quiet)
        if ok == _vc.VGD_OK:
            v1 = _C_CHAR2WCHAR(v1.value).strip()
    elif key2 in _vc.VGD_OPR_KEYS['get_int']:
        v1 = _ct.c_int(0)
        ok = _vp.c_vgd_get_int(vgd_ptr, key2b, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_float']:
        v1 = _ct.c_float(0.)
        ok = _vp.c_vgd_get_float(vgd_ptr, key2b, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_int_1d']:
        v1 = _ct.POINTER(_ct.c_int)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_int_1d(vgd_ptr, key2b, _ct.byref(v1),
                                  _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_float_1d']:
        v1 = _ct.POINTER(_ct.c_float)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_float_1d(vgd_ptr, key2b, _ct.byref(v1),
                                    _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_double']:
        v1 = _ct.c_double(0.)
        ok = _vp.c_vgd_get_double(vgd_ptr, key2b, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_double_1d']:
        v1 = _ct.POINTER(_ct.c_double)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_double_1d(vgd_ptr, key2b, _ct.byref(v1),
                                     _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_double_3d']:
        v1 = _ct.POINTER(_ct.c_double)()
        ni = _ct.c_int(0)
        nj = _ct.c_int(0)
        nk = _ct.c_int(0)
        ok = _vp.c_vgd_get_double_3d(vgd_ptr, key2b, _ct.byref(v1),
                                     _ct.byref(ni), _ct.byref(nj),
                                     _ct.byref(nk), quiet)
        if ok == _vc.VGD_OK:
            nv = ni.value * nj.value * nk.value
            # v1 = [v for v in v1[0:nv]]
            v1 = _np.asarray(v1[0:nv], dtype=_np.float64, order='F')
            v1 = _np.reshape(v1, (ni.value, nj.value, nk.value), order='F')

    else:
        raise KeyError('Problem getting val, invalid key (key={0})'.format(key))

    if ok != _vc.VGD_OK:
        if defaultOnFail:
            return defaultValue
        else:
            raise VGDError('Problem getting val (key={0})'.format(key))
    return v1


def vgd_put(vgd_ptr, key, value):
    """
    Set a vgd object parameter value.

    Args:
        vgd_ptr (VGridDescriptor ref):
                        Reference/Pointer to the VGridDescriptor
        key   (int)   : Parameter name
                        Possible values: see VGD_KEYS, VGD_OPR_KEYS
        value (mixed) : Value to be set
                        Type of values depends on option name
    Returns:
        None
    Raises:
        TypeError
        KeyError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    ...     myvgd = vgd.vgd_new_pres(lvls)
    ... except:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> try:
    ...     vgd.vgd_put(myvgd, 'IP_1', 2)
    ... except:
    ...     sys.stderr.write("There was a problem setting vgd parameter value")

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        vgd_get
        vgd_get_opt
        vgd_put_opt
    """
    key2 = key.upper()[0:4]
    key2b = _C_WCHAR2CHAR(key2)
    if key2 in _vc.VGD_OPR_KEYS['put_char']:
        v1 = _C_MKSTR(str(value))
        ok = _vp.c_vgd_put_char(vgd_ptr, key2b, v1)
    elif key2 in _vc.VGD_OPR_KEYS['put_int']:
        ok = _vp.c_vgd_put_int(vgd_ptr, key2b, value)
    ## elif key2 in _vc.VGD_OPR_KEYS['put_double']: #removed from vgd 6.2.1
    ##     ok = _vp.c_vgd_put_double(vgd_ptr, key2b, value)
    else:
        raise KeyError('Problem setting val, invalid key (key={0})'.
                       format(key))

    if ok != _vc.VGD_OK:
        raise VGDError('Problem setting val (key={0}, value={1})'.
                       format(key, repr(value)))
    return


def vgd_cmp(vgd0ptr, vgd1ptr):
    """
    Compate 2 VGridDescriptors, return (vgd0ptr == vgd1ptr)

    Args:
        vgd0ptr (VGridDescriptor ref):
                          Reference/Pointer to a VGridDescriptor
        vgd1ptr (VGridDescriptor ref):
                          Reference/Pointer to another VGridDescriptor
    Returns:
        bool : True if vgd0ptr == vgd1ptr
    Raises:
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> try:
    ...     myvgd0 = vgd.vgd_new_pres((500.,850.,1000.))
    ...     myvgd1 = vgd.vgd_new_pres((550.,900.,1013.))
    ... except:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    ...     sys.exit(1)
    >>> if vgd.vgd_cmp(myvgd0, myvgd1):
    ...     print("# The 2 VGridDescriptors are identical.")
    ... else:
    ...     print("# The 2 VGridDescriptors differ.")
    # The 2 VGridDescriptors differ.

    See Also:
        vgd_new
        vgd_read
        vgd_free
    """
    ok = _vp.c_vgd_vgdcmp(vgd0ptr, vgd1ptr)
    return ok == _vc.VGD_OK


def vgd_levels(vgd_ptr, rfld=None, ip1list='VIPM', in_log=_vc.VGD_DIAG_PRES,
               dpidpis=_vc.VGD_DIAG_DPIS, double_precision=False, rfls=None):
    """
    Compute level positions (pressure or log(p)) for the given ip1 list and surface field

    Deprecated; see vgd_levels2 for arguments description.
    Kept for backward compatibility of arguments order.

    See Also:
        vgd_levels2
    """
    return vgd_levels2(vgd_ptr, rfld, rfls, ip1list,
                       in_log, dpidpis, double_precision)


def vgd_levels2(vgd_ptr, rfld=None, rfls=None, ip1list='VIPM',
                in_log=_vc.VGD_DIAG_PRES, dpidpis=_vc.VGD_DIAG_DPIS,
                double_precision=False):
    """
    Compute level positions (pressure or log(p)) for the given ip1 list and surface field

    Args:
        vgd_ptr (VGridDescriptor ref):
                           Reference/Pointer to the VGridDescriptor
        rfld     (mixed) : Reference surface field
                           Possible values:
                           (int)    : RPNStd unit where to read the RFLD
                           (float)  : RFLD values
                           (list)   : RFLD values
                           (ndarray): RFLD values
        rfls     (mixed) : Reference surface field (large scale)
                           Possible values:
                           (int)    : RPNStd unit where to read the RFLD
                           (float)  : RFLS values
                           (list)   : RFLS values
                           (ndarray): RFLS values
        ip1list (mixed)  : ip1 list of destination levels
                           (str) : get the ip1 list form the vgd object
                                   possible value: 'VIPM' or 'VIPT'
                           (int) or (list): ip1 values
        in_log   (int)   : VGD_DIAG_LOGP or VGD_DIAG_PRES
        dpidpis  (int)   : VGD_DIAG_DPI or VGD_DIAG_DPIS
        double_precision (bool) : True for double precision computations
    Returns:
        ndarray : numpy array with shape of...
                  if type(rfld) == float:   shape = [len(ip1list),]
                  if type(rfld) == list:    shape = [len(list), len(ip1list)]
                  if type(rfld) == ndarray: shape = rfld.shape + [len(ip1list)]
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.000,   0.011,    0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.993,    1.000)
    >>> ptop  = 1000.
    >>> try:
    ...     myvgd = vgd.vgd_new_eta(lvls, ptop)
    ... except vgd.VGDError:
    ...     sys.stderr.write('There was a problem creating the VGridDescriptor')
    ...     sys.exit(1)
    >>> try:
    ...     levels = vgd.vgd_levels(myvgd, rfld=100130.)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem computing VGridDescriptor levels")

    See Also:
        rpnpy.vgd.const.VGD_DIAG_LOGP
        rpnpy.vgd.const.VGD_DIAG_PRES
        rpnpy.vgd.const.VGD_DIAG_DPI
        rpnpy.vgd.const.VGD_DIAG_DPIS
        vgd_new
        vgd_read
        vgd_free
        vgd_get
    """
    if isinstance(ip1list, str):
        ip1list0 = vgd_get(vgd_ptr, ip1list)
        ip1list1 = _np.array(ip1list0, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _integer_types):
        ip1list1 = _np.array([ip1list], dtype=_np.int32, order='F')
    elif isinstance(ip1list, (list, tuple)):
        ip1list1 = _np.array(ip1list, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _np.ndarray):
        ip1list1 = _np.array(ip1list.flatten(), dtype=_np.int32, order='F')
    else:
        raise TypeError('ip1list should be string, list or int: {0}'.
                        format(str(type(ip1list))))
    nip1 = ip1list1.size
    ip1list = ip1list1.ctypes.data_as(_ct.POINTER(_ct.c_int))

    vkind = vgd_get(vgd_ptr, 'KIND')
    vvers = vgd_get(vgd_ptr, 'VERS')
    vcode = VGD_TYPE_CODE(vkind, vvers)
    rfld_nomvar = vgd_get(vgd_ptr, 'RFLD', defaultOnFail=True)
    rfls_nomvar = vgd_get(vgd_ptr, 'RFLS', defaultOnFail=True)

    rank0 = False
    if rfld_nomvar:
        if isinstance(rfld, float):
            rfld = _np.array([rfld], dtype=_np.float32, order='F')
            rank0 = True
        elif isinstance(rfld, (list, tuple)):
            rfld = _np.array(rfld, dtype=_np.float32, order='F')
        elif isinstance(rfld, _integer_types):
            fileId = rfld
            rfld = _rmn.fstlir(fileId, nomvar=rfld_nomvar.strip())['d']
            if rfld_nomvar.upper() in _vc.VGD_RFLD_CONV_KEYS:
                rfld = _vc.VGD_RFLD_CONV[rfld_nomvar.upper()](rfld)
        elif rfld is None:
            if rfld_nomvar is None:
                raise TypeError('RFLD needs to be provided for vcode={0}'.format(vcode))
            else:
                rfld = _np.array([1000.], dtype=_np.float32, order='F')
                rank0 = True
        elif not isinstance(rfld, _np.ndarray):
            raise TypeError('rfld should be ndarray, list or float: {0}'.
                            format(str(type(ip1list))))

    if rfls_nomvar:
        if isinstance(rfls, float):
            rfls = _np.array([rfls], dtype=_np.float32, order='F')
            rank0 = True
        elif isinstance(rfls, (list, tuple)):
            rfls = _np.array(rfls, dtype=_np.float32, order='F')
        elif isinstance(rfls, _integer_types):
            fileId = rfls
            rfls = _rmn.fstlir(fileId, nomvar=rfls_nomvar.strip())['d']
            if rfls_nomvar.upper() in _vc.VGD_RFLD_CONV_KEYS:
                rfls = _vc.VGD_RFLD_CONV[rfls_nomvar.upper()](rfls)
        elif rfls is None:
            if rfls_nomvar:
                raise TypeError('RFLS needs to be provided for vcode={0}'.
                                format(vcode))
            else:
                rfls = _np.array([1000.], dtype=_np.float32, order='F')
                rank0 = True
        elif not isinstance(rfls, _np.ndarray):
            raise TypeError('rfls should be ndarray, list or float: {0}'.
                            format(str(type(ip1list))))
        if rfls.size != rfld.size:
            raise RuntimeError("rfls is not of same size as rfld")

    if double_precision:
        dtype = _np.float64
        rfld8 = _np.array(rfld, copy=True, dtype=dtype, order='F')
        if rfls_nomvar:
            rfls8 = _np.array(rfls, copy=True, dtype=dtype, order='F')
    else:
        dtype = _np.float32
        rfld8 = rfld
        rfls8 = rfls
    if rfld is None:
        shape = [nip1]
        rfld8 =  _np.array([-1.],dtype=_np.float32)
    else:
        shape = list(rfld.shape) + [nip1]

    levels8 = _np.empty(shape, dtype=dtype, order='F')

    ok = _vc.VGD_OK
    if double_precision:
        if rfls_nomvar is None:
            ok = _vp.c_vgd_diag_withref_8(vgd_ptr, rfld8.size, 1, nip1,
                                          ip1list, levels8, rfld8, in_log,
                                          dpidpis)
        else:
            ok = _vp.c_vgd_diag_withref_2ref_8(vgd_ptr, rfld8.size, 1, nip1,
                                               ip1list, levels8, rfld8,
                                               rfls8, in_log, dpidpis)
    else:
        if rfls_nomvar is None:
            ok = _vp.c_vgd_diag_withref(vgd_ptr, rfld8.size, 1, nip1,
                                        ip1list, levels8, rfld8, in_log,
                                        dpidpis)
        else:
            ok = _vp.c_vgd_diag_withref_2ref(vgd_ptr, rfld8.size, 1, nip1,
                                        ip1list, levels8, rfld8, rfls8,
                                        in_log, dpidpis)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem computing levels.')
    if rank0:
        levels8 = levels8.flatten()
    return levels8


def vgd_stda76_temp(vgd_ptr, ip1list='VIPM'):
    """
    Get the standard atmosphere 1976 temperature
    for the given vertical structure.

    Args:
        vgd_ptr (VGridDescriptor ref): Reference/Pointer to the VGridDescriptor
        ip1list : ip1 list of destination levels
                  (str) : get the ip1 list form the vgd object
                          possible value: 'VIPM' or 'VIPT'
                  (int) or (list): ip1 values
    Returns:
        ndarray, temperature profile
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.000,   0.011,    0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.993,    1.000)
    >>> ptop  = 1000.
    >>> try:
    ...     myvgd = vgd.vgd_new_eta(lvls, ptop)
    ... except vgd.VGDError:
    ...     sys.stderr.write('There was a problem creating the VGridDescriptor')
    ...     sys.exit(1)
    >>> try:
    ...     stda76_temp = vgd.vgd_stda76_temp(myvgd)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting the stda76 temperture")
    >>> print('stda76_temp={}'.format(stda76_temp))
    stda76_temp=[ 227.70584106  222.80267334  219.13482666  216.6499939   216.6499939
      216.6499939   216.6499939   216.6499939   216.6499939   217.26150513
      223.86172485  230.43205261  236.91807556  243.28668213  249.11688232
      254.51049805  259.62689209  264.26861572  268.58728027  272.56045532
      276.04125977  278.97198486  281.53799438  283.77682495  285.66186523
      287.05548096  287.7689209   288.1499939 ]
    """
    if isinstance(ip1list, str):
        ip1list0 = vgd_get(vgd_ptr, ip1list)
        ip1list1 = _np.array(ip1list0, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _integer_types):
        ip1list1 = _np.array([ip1list], dtype=_np.int32, order='F')
    elif isinstance(ip1list, (list, tuple)):
        ip1list1 = _np.array(ip1list, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _np.ndarray):
        ip1list1 = _np.array(ip1list.flatten(), dtype=_np.int32, order='F')
    else:
        raise TypeError('ip1list should be string, list or int: {0}'.
                        format(str(type(ip1list))))
    nip1 = ip1list1.size
    ip1list = ip1list1.ctypes.data_as(_ct.POINTER(_ct.c_int))
    temp = _np.empty(nip1, dtype=_np.float32, order='F')
    ok = _vp.c_vgd_stda76_temp(vgd_ptr, ip1list, nip1, temp)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting stda 1976 temperature.')
    return temp


def vgd_stda76_pres(vgd_ptr, ip1list='VIPM', sfc_temp=None, sfc_pres=None):
    """
    Get the standard atmosphere 1976 pressure
    for the given height vertical structure.

    Args:
        vgd_ptr (VGridDescriptor ref): Reference/Pointer to the VGridDescriptor
        ip1list  : ip1 list of destination levels
                   (str) : get the ip1 list form the vgd object
                           possible value: 'VIPM' or 'VIPT'
                   (int) or (list): ip1 values
        sfc_temp : Optional surface temperature
        sfc_pres : Optional surface pressure
    Returns:
        ndarray, pressure profile
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> hyb = (30968.,  16765., 5477., 880., 0.)
    >>> (rcoef1, rcoef2, rcoef3, rcoef4) = (0., 5., 0., 100.)
    >>> (kind, version) = (21, 2)
    >>> (dhm, dht, dhw) = (10., 1.5, 10.)
    >>> my_vgd = vgd.vgd_new_hybhls(hyb, rcoef1, rcoef2, rcoef3, rcoef4,
    ...                             dhm, dht, dhw)
    >>> try:
    ...     stda76_pres = vgd.vgd_stda76_pres(my_vgd)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting the stda76 pressure")
    >>> print('stda76_pres={}'.format(stda76_pres))
    stda76_pres=[   1013.26861572    9119.25683594   50665.609375     91190.859375    101325.
      101325.          101204.9296875 ]
    >>> sfc_temp = 273.
    >>> try:
    ...     stda76_pres = vgd.vgd_stda76_pres(my_vgd, sfc_temp=sfc_temp)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting the stda76 pressure")
    >>> print('stda76_pres={}'.format(stda76_pres))
    stda76_pres=[    730.36712646    7728.3359375    48616.93359375   90653.484375    101325.
      101325.          101198.2734375 ]
    >>> sfc_pres = 100000.
    >>> try:
    ...     stda76_pres = vgd.vgd_stda76_pres(my_vgd, sfc_pres=sfc_pres)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting the stda76 pressure")
    >>> print('stda76_pres={}'.format(stda76_pres))
    stda76_pres=[   1000.01843262    9000.00683594   50003.0703125    89998.375       100000.
      100000.           99881.5       ]
    """
    if isinstance(ip1list, str):
        ip1list0 = vgd_get(vgd_ptr, ip1list)
        ip1list1 = _np.array(ip1list0, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _integer_types):
        ip1list1 = _np.array([ip1list], dtype=_np.int32, order='F')
    elif isinstance(ip1list, (list, tuple)):
        ip1list1 = _np.array(ip1list, dtype=_np.int32, order='F')
    elif isinstance(ip1list, _np.ndarray):
        ip1list1 = _np.array(ip1list.flatten(), dtype=_np.int32, order='F')
    else:
        raise TypeError('ip1list should be string, list or int: {0}'.
                        format(str(type(ip1list))))
    if not sfc_temp is None:
        sfc_temp = _ct.POINTER(_ct.c_float)(_ct.c_float(sfc_temp))
    if not sfc_pres is None:
        sfc_pres = _ct.POINTER(_ct.c_float)(_ct.c_float(sfc_pres))
    nip1 = ip1list1.size
    ip1list = ip1list1.ctypes.data_as(_ct.POINTER(_ct.c_int))
    pres = _np.empty(nip1, dtype=_np.float32, order='F')
    ok = _vp.c_vgd_stda76_pres(vgd_ptr, ip1list, nip1, pres, sfc_temp, sfc_pres)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting stda 1976 pressure.')
    return pres


def vgd_stda76_hgts_from_pres_list(pres=None):
    """
    Compute standard atmosphere 1976 height from a list of pressure values

    Agrs:
        pres (mixed) : Pressure in Pa for which to get stda 1976 heights in m
                       Possible values:
                       (float)  : pressure values
                       (list)   : pressure values
                       (ndarray): pressure values
    Returns:
       ndarray : numpy array with shape of pres
                 TODO float list ndarray???
    Raises:
       TypeError
       VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> pres = (100000., 85000., 50000., 25000., 10000., 1000., 100., 10.)
    >>> try:
    ...     hgts = vgd.vgd_stda76_hgts_from_pres_list(pres=pres)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting stda76 heights from pressure list")
    >>> print('hgts={}'.format(hgts))
    hgts=[   110.8887558    1457.35668945   5574.64160156  10363.29980469
      16180.29980469  31055.84375     47822.2265625   64949.40234375]
    """
    if isinstance(pres, (list, tuple)):
        shape = None
        pres = _np.array(pres, copy=True, dtype=_np.float32)
    elif isinstance(pres, _np.ndarray):
        shape = pres.shape
        pres = _np.array(pres.flatten(), copy=True, dtype=_np.float32)
    else:
        raise TypeError('pres should be list or ndarray: {0}'.
                        format(str(type(pres))))
    hgts = _np.empty(pres.shape, dtype=_np.float32)
    ok = _vp.c_vgd_stda76_hgts_from_pres_list(hgts, pres, pres.size)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem computing stda 1976 heights from pressure list.')
    if shape:
        hgts.shape = shape
    return hgts


def vgd_stda76_pres_from_hgts_list(hgts=None):
    """
    Compute standard atmosphere 1976 pressure from a list of heights values

    Agrs:
        hgts (mixed) : heights in m for which to get stda 1976 pressure in Pa
                       Possible values:
                       (float)  : height values
                       (list)   : height values
                       (ndarray): height values
    Returns:
       ndarray : numpy array with shape of hgts
                 TODO float list ndarray???
    Raises:
       TypeError
       VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> hgts=(110.8887558, 1457.35668945, 5574.64160156, 10363.29980469,
    ...       16180.29980469, 31055.84375, 47822.2265625, 64949.40234375)
    >>> try:
    ...     pres = vgd.vgd_stda76_pres_from_hgts_list(hgts=hgts)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem getting stda76 pressure from height list")
    >>> print('pres={}'.format(pres))
    pres=[  1.00000016e+05   8.50000078e+04   4.99999961e+04   2.50000020e+04
       1.00000010e+04   1.00000079e+03   9.99999924e+01   1.00000019e+01]
    """
    if isinstance(hgts, (list, tuple)):
        shape = None
        hgts = _np.array(hgts, copy=True, dtype=_np.float32)
    elif isinstance(hgts, _np.ndarray):
        shape = hgts.shape
        hgts = _np.array(hgts.flatten(), copy=True, dtype=_np.float32)
    else:
        raise TypeError('hgts should be list or ndarray: {0}'.
                        format(str(type(hgts))))
    pres = _np.empty(hgts.shape, dtype=_np.float32)
    ok = _vp.c_vgd_stda76_pres_from_hgts_list(pres, hgts, hgts.size)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem computing stda 1976 pressure from height list.')
    if shape:
        pres.shape = shape
    return pres


def vgd_print_desc(vgd_ptr, convip=-1):
    """
    Print vgrid descriptor parameters to standard output, a debugging tool.

    Args:
        vgd_ptr (VGridDescriptor ref): Reference/Pointer to the VGridDescriptor
        convip  (int)                : use convip>0 to convert ip1, -1 otherwise
    Returns:
        None
    Raises:
        TypeError
        VGDError
    """
    _vp.c_vgd_print_desc(vgd_ptr, -1, convip)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
