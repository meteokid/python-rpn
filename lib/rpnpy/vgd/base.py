#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd.base contains python wrapper to main vgrid C functions

See Also:
    rpnpy.vgd.proto
    rpnpy.vgd.const

Notes:
    This Module is available from python-rpn version 2.0.b6
"""
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from rpnpy.vgd import proto as _vp
from rpnpy.vgd  import const as _vc
from rpnpy.vgd  import VGDError
import rpnpy.librmn.all as _rmn

_C_MKSTR = _ct.create_string_buffer
_MB2PA   = 100.

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
    >>>     myvgd = vgd.vgd_new_sigm(lvls)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
    >>> lvls  = (0.000,   0.011,    0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.993,    1.000)
    >>> ptop  = 1000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_eta(lvls, ptop)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
    >>> lvls  = (0.011,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.993,    1.000)
    >>> rcoef1 = 1.6
    >>> ptop   = 110.
    >>> pref   = 80000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_hyb(lvls, rcoef1, ptop, pref)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
        rcoef1   (float): 2nd Coordinate recification R-coefficient
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
    >>> lvls  = (0.013,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 10.
    >>> ptop   = 805.
    >>> pref   = 100000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_hybs(lvls, rcoef1, rcoef2, ptop, pref)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
        rcoef1   (float): 2nd Coordinate recification R-coefficient
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
    >>> lvls  = (0.013,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 10.
    >>> ptop   = 1000.
    >>> pref   = 100000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_hybt(lvls, rcoef1, rcoef2, ptop, pref)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
        rcoef1   (float): 2nd Coordinate recification R-coefficient
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
    >>> lvls  = (0.013,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> ptop   = -1.
    >>> pref   = 100000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_hybm(lvls, rcoef1, rcoef2, ptop, pref)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
        rcoef1   (float): 2nd Coordinate recification R-coefficient
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
    >>> lvls  = (0.013,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    >>>     myvgd = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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


def vgd_new(kind, version, hyb,
            rcoef1=None, rcoef2=None, ptop=None, pref=None, dhm=None, dht=None,
            ip1=-1, ip2=-1):
    """
    General function to Build an VGridDescriptor initialized with provided info.

    Args:
        kind     (int)  : Kind of vertical coor
        version  (int)  : Version of vertical coor
        hyb      (list) : list of level values
        rcoef1   (float): 1st Coordinate recification R-coefficient
        rcoef1   (float): 2nd Coordinate recification R-coefficient
        ptop     (float): Top level pressure [Pa]
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
    >>> (kind, version) = vgd.VGD_KIND_VER['hybmd']
    >>> lvls  = (0.013,   0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    >>>     myvgd = vgd.vgd_new(kind, version, lvls, rcoef1, rcoef2, pref=pref, dhm=dhm, dht=dht)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    
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
        vgd_write
        vgd_levels
        vgd_free
    """
    if isinstance(hyb,(list,tuple)):
        hyb = _np.array(hyb, copy=True, dtype=_np.float32, order='FORTRAN')
    elif isinstance(hyb,_np.ndarray):
        hyb = _np.array(hyb.flatten(), copy=True, dtype=_np.float32, order='FORTRAN')
    else:
        raise TypeError('hyb should be list or ndarray: %s' % str(type(ip1list)))
    if not rcoef1 is None:
        rcoef1 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef1))
    if not rcoef2 is None:
        rcoef2 = _ct.POINTER(_ct.c_float)(_ct.c_float(rcoef2))
    if not ptop   is None:
        ptop   = _ct.POINTER(_ct.c_double)(_ct.c_double(ptop))
    if not pref   is None:
        pref   = _ct.POINTER(_ct.c_double)(_ct.c_double(pref))
    if not dhm    is None:
        dhm    = _ct.POINTER(_ct.c_float)(_ct.c_float(dhm))
    if not dht    is None:
        dht    = _ct.POINTER(_ct.c_float)(_ct.c_float(dht))
    vgd_ptr = _vp.c_vgd_construct()
    p_ptop_out = None
    if kind == 5 and version in (4,5):
        if not ptop is None:
            p_ptop_out = ptop
        else:
            ptop_out = 100.
            p_ptop_out = _ct.POINTER(_ct.c_double)(_ct.c_double(ptop_out))
    ok = _vp.c_vgd_new_gen(vgd_ptr, kind, version,
                           hyb, hyb.size, rcoef1, rcoef2, ptop, pref,
                           p_ptop_out, ip1, ip2, dhm, dht)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem building VGD (kind=%d, version=%d): Error=%d)' % (kind, version,ok))
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
    >>>     myvgd = vgd.vgd_read(fileId)
    >>> except:
    >>>     sys.stderr.write("There was a problem reading the VGridDescriptor")
    >>> finally:
    >>>     rmn.fstcloseall(fileId)

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
        raise VGDError('Problem getting vgd from file (id=%d, ip1=%d, ip2=%d, kind=%d, version=%d)' % (fileId, ip1, ip2, kind, version))
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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
    >>> fileName = 'myfstfile.fst'
    >>> fileId   = rmn.fstopenall(fileName, rmn.FST_RW)
    >>> try:
    >>>     vgd.vgd_write(myvgd, fileId)
    >>> except:
    >>>     sys.stderr.write("There was a problem writing the VGridDescriptor")
    >>> finally:
    >>>     rmn.fstcloseall(fileId)

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
        raise VGDError('Problem writing vgd to file (id=%d)' % fileId)
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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
    >>> try:
    >>>     vgdtable = vgd.vgd_tolist(myvgd)
    >>> except:
    >>>     sys.stderr.write("There was a problem encoding the VGridDescriptor in a list")

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
    >>>     myvgd    = vgd.vgd_new_pres(lvls)
    >>>     vgdtable = vgd.vgd_tolist(myvgd)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating/encoding the VGridDescriptor")
    >>>     sys.exit(1)
    >>> ## ...
    >>> try:
    >>>     myvgd2 = vgd.vgd_fromlist(vgdtable)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor in a list")

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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating/encoding the VGridDescriptor")
    >>>     sys.exit(1)
    >>> ## ...
    >>> try:
    >>>     myvgd2 = vgd.vgd_copy(myvgd)
    >>> except:
    >>>     sys.stderr.write("There was a problem copying the VGridDescriptor")

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
    >>>     allow_signma = vgd.vgd_get_opt('ALLOW_SIGMA')
    >>> except:
    >>>     sys.stderr.write("There was a problem getting vgd gloabl option")

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
        raise KeyError('Problem getting opt, invalid key (key=%s)' % key)
    v1 = _ct.c_int(0)
    ok = _vp.c_vgd_getopt_int(key2, _ct.byref(v1), quiet)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting opt (key=%s)' % key)
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
    >>>     vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
    >>> except:
    >>>     sys.stderr.write("There was a problem setting vgd gloabl option")

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
        raise VGDError('Problem setting opt, invalid key (key=%s)' % key)
    ok = _vp.c_vgd_putopt_int(key2, value)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem setting opt (key=%s, value=%s)' % (key, repr(value)))
    return


def vgd_get(vgd_ptr, key, quiet=1):
    """
    Get a vgd object parameter value

    Args:
        vgd_ptr (VGridDescriptor ref):
                      Reference/Pointer to the VGridDescriptor
        key   (int) : Parameter name
                      Possible values: see VGD_KEYS, VGD_OPR_KEYS
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
    >>> lvls  = (500.,850.,1000.)
    >>> try:
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
    >>> try:
    >>>     vkind = vgd.vgd_get(myvgd, 'KIND')
    >>> except:
    >>>     sys.stderr.write("There was a problem getting vgd parameter value")

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        vgd_put
        vgd_get_opt
        vgd_put_opt
    """
    v1 = None
    key2 = key.upper()[0:4]
    if   key2 in _vc.VGD_OPR_KEYS['get_char']:
        v1 = _C_MKSTR(' '*_vc.VGD_MAXSTR_ETIKET)
        ok = _vp.c_vgd_get_char(vgd_ptr, key2, v1, quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value.strip()
    elif key2 in _vc.VGD_OPR_KEYS['get_int']:
        v1 = _ct.c_int(0)
        ok = _vp.c_vgd_get_int(vgd_ptr, key2, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_float']:
        v1 = _ct.c_float(0.)
        ok = _vp.c_vgd_get_float(vgd_ptr, key2, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_int_1d']:
        v1 = _ct.POINTER(_ct.c_int)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_int_1d(vgd_ptr, key2, _ct.byref(v1),
                                  _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_float_1d']:
        v1 = _ct.POINTER(_ct.c_float)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_float_1d(vgd_ptr, key2, _ct.byref(v1),
                                    _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_double']:
        v1 = _ct.c_double(0.)
        ok = _vp.c_vgd_get_double(vgd_ptr, key2, _ct.byref(v1), quiet)
        if ok == _vc.VGD_OK:
            v1 = v1.value
    elif key2 in _vc.VGD_OPR_KEYS['get_double_1d']:
        v1 = _ct.POINTER(_ct.c_double)()
        nv = _ct.c_int(0)
        ok = _vp.c_vgd_get_double_1d(vgd_ptr, key2, _ct.byref(v1),
                                     _ct.byref(nv), quiet)
        if ok == _vc.VGD_OK:
            v1 = [v for v in v1[0:nv.value]]
    elif key2 in _vc.VGD_OPR_KEYS['get_double_3d']:
        v1 = _ct.POINTER(_ct.c_double)()
        ni = _ct.c_int(0)
        nj = _ct.c_int(0)
        nk = _ct.c_int(0)
        ok = _vp.c_vgd_get_double_3d(vgd_ptr, key2, _ct.byref(v1),
                                     _ct.byref(ni), _ct.byref(nj),
                                     _ct.byref(nk), quiet)
        if ok == _vc.VGD_OK:
            nv = ni.value * nj.value * nk.value
            #v1 = [v for v in v1[0:nv]]
            v1 = _np.asarray(v1[0:nv], dtype=_np.float64, order='F')
            v1 = _np.reshape(v1, (ni.value, nj.value, nk.value), order='F')

    else:
        raise KeyError('Problem getting val, invalid key (key=%s)' % key)

    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting val (key=%s)' % key)
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
    >>>     myvgd = vgd.vgd_new_pres(lvls)
    >>> except:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
    >>> try:
    >>>     vgd.vgd_put(myvgd, 'IP_1', 2)
    >>> except:
    >>>     sys.stderr.write("There was a problem setting vgd parameter value")

    See Also:
        rpnpy.vgd.const.VGD_KEYS
        rpnpy.vgd.const.VGD_OPR_KEYS
        vgd_get
        vgd_get_opt
        vgd_put_opt
    """
    key2 = key.upper()[0:4]
    if   key2 in _vc.VGD_OPR_KEYS['put_char']:
        v1 = _C_MKSTR(str(value))
        ok = _vp.c_vgd_put_char(vgd_ptr, key2, v1)
    elif key2 in _vc.VGD_OPR_KEYS['put_int']:
        ok = _vp.c_vgd_put_int(vgd_ptr, key2, value)
    elif key2 in _vc.VGD_OPR_KEYS['put_double']:
        ok = _vp.c_vgd_put_double(vgd_ptr, key2, value)
    else:
        raise KeyError('Problem setting val, invalid key (key=%s)' % key)
    
    if ok != _vc.VGD_OK:
        raise VGDError('Problem setting val (key=%s, value=%s)' % (key, repr(value)))
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
    >>>     myvgd0 = vgd.vgd_new_pres((500.,850.,1000.))
    >>>     myvgd1 = vgd.vgd_new_pres((550.,900.,1013.))
    >>> except:
    >>>     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>     sys.exit(1)
    >>> if vgd.vgd_cmp(myvgd0, myvgd1):
    >>>     print("The 2 VGridDescriptors are identical.")
    >>> else:
    >>>     print("The 2 VGridDescriptors differ.")

    See Also:
        vgd_new
        vgd_read
        vgd_free
    """
    ok = _vp.c_vgd_vgdcmp(vgd0ptr, vgd1ptr)
    return ok == _vc.VGD_OK


def vgd_levels(vgd_ptr, rfld=None, ip1list='VIPM',  in_log=_vc.VGD_DIAG_PRES, dpidpis=_vc.VGD_DIAG_DPIS, double_precision=False):
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
        ip11list (mixed) : ip1 list of destination levels
                           (str) : get the ip1 list form the vgd object
                                   possible value: 'VIPM' or 'VIPT'
                           (int) or (list): ip1 values
        in_log   (int)   : VGD_DIAG_LOGP or VGD_DIAG_PRES
        dpidpis  (int)   : VGD_DIAG_DPI or VGD_DIAG_DPIS
        double_precision (bool) : True for double precision computations
    Returns:
        ndarray : numpy array with shape of...
                  if type(rfld) == float:   shape = [len(ip11list),]
                  if type(rfld) == list:    shape = [len(list), len(ip11list)]
                  if type(rfld) == ndarray: shape = rfld.shape + [len(ip11list)]
    Raises:
        TypeError
        VGDError

    Examples:
    >>> import sys
    >>> import rpnpy.vgd.all as vgd
    >>> lvls  = (0.000,   0.011,    0.027,    0.051,    0.075, \
                 0.101,   0.127,    0.155,    0.185,    0.219, \
                 0.258,   0.302,    0.351,    0.405,    0.460, \
                 0.516,   0.574,    0.631,    0.688,    0.744, \
                 0.796,   0.842,    0.884,    0.922,    0.955, \
                 0.980,   0.993,    1.000)
    >>> ptop  = 1000.
    >>> try:
    >>>     myvgd = vgd.vgd_new_eta(lvls, ptop)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write('There was a problem creating the VGridDescriptor')
    >>>     sys.exit(1)
    >>> try:
    >>>     levels = vgd.vgd_levels(myvgd, rfld=100130.)
    >>> except vgd.VGDError:
    >>>     sys.stderr.write("There was a problem computing VGridDescriptor levels")

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
    if isinstance(ip1list,str):
        ip1list0 = vgd_get(vgd_ptr, ip1list)
        ip1list1 = _np.array(ip1list0, dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,(int,long)):
        ip1list1 = _np.array([ip1list], dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,(list,tuple)):
        ip1list1 = _np.array(ip1list, dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,_np.ndarray):
        ip1list1 = _np.array(ip1list.flatten(), dtype=_np.int32, order='FORTRAN')
    else:
        raise TypeError('ip1list should be string, list or int: %s' % str(type(ip1list)))
    nip1    = ip1list1.size
    ip1list = ip1list1.ctypes.data_as(_ct.POINTER(_ct.c_int))

    vkind = vgd_get(vgd_ptr, 'KIND')
    vvers = vgd_get(vgd_ptr, 'VERS')
    vcode = int(vkind)*1000+int(vvers)
    
    rank0 = False
    if isinstance(rfld,float):
        rfld = _np.array([rfld], dtype=_np.float32, order='FORTRAN')
        rank0 = True
    elif isinstance(rfld,(list,tuple)):
        rfld = _np.array(rfld, dtype=_np.float32, order='FORTRAN')
    elif isinstance(rfld,int):
        if _vc.VGD_VCODE_NEED_RFLD[vcode]:
            fileId = rfld
            rfld_name = vgd_get(vgd_ptr, 'RFLD')
            if not rfld_name:
                raise VGDError('Problem getting RFLD to compute levels')
            rfld = _rmn.fstlir(fileId, nomvar=rfld_name.strip())['d']
            MB2PA = 100.
            rfld = rfld * MB2PA
        else:
            rfld = _np.array([float(fileId)], dtype=_np.float32, order='FORTRAN')
            rank0 = True            
    elif rfld is None:
        if _vc.VGD_VCODE_NEED_RFLD[vcode]:
            raise TypeError('RFLD needs to be provided for vcode=%d' % vcode)
        else:
            rfld = _np.array([1000.], dtype=_np.float32, order='FORTRAN')
            rank0 = True
    elif not isinstance(rfld,_np.ndarray):
        raise TypeError('rfld should be ndarray, list or float: %s' % str(type(ip1list)))

    if double_precision:
        dtype = _np.float64
        rfld8 = _np.array(rfld, copy=True, dtype=dtype, order='FORTRAN')
    else:
        dtype = _np.float32
        rfld8 = rfld
    shape = list(rfld.shape) + [nip1]
    levels8 = _np.empty(shape, dtype=dtype, order='FORTRAN')

    ok = _vc.VGD_OK
    if vkind == 2:  # Workaround for pressure levels
        for k in xrange(nip1):
            (value, kind) = _rmn.convertIp(_rmn.CONVIP_DECODE, int(ip1list1[k]))
            levels8[:,:,k] = value * _MB2PA
    else:
        if double_precision:
            ok = _vp.c_vgd_diag_withref_8(vgd_ptr, rfld8.size, 1, nip1, ip1list, levels8, rfld8, in_log, dpidpis)
        else:
            ok = _vp.c_vgd_diag_withref(vgd_ptr, rfld8.size, 1, nip1, ip1list, levels8, rfld8, in_log, dpidpis)
        
    if ok != _vc.VGD_OK:
        raise VGDError('Problem computing levels.')
    if rank0:
        levels8 = levels8.flatten()
    return levels8


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
