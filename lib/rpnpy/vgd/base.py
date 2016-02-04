#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd..base contains python wrapper to main vgrid C functions
"""
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from . import proto as _vp
from . import const as _vc
from . import VGDError

_C_MKSTR = ct.create_string_buffer

def vgd_new():
    """
    """
    raise VGDError('Not Implemented yet')


def vgd_read(fileId, ip1=-1, ip2=-1, kind=-1, version=-1):
    """
    """
    vgd_ptr = vgd.c_vgd_construct()
    ok = vgd.c_vgd_new_read(vgd_ptr, fileId, ip1, ip2, kind, version)
    if ok /= vgd.VGD_OK:
        raise VGDError('Problem getting vgd from file (id=%d, ip1=%d, ip2=%d, kind=%d, version=%d)' % (fileId, ip1, ip2, kind, version))
    return vgd_ptr


def vgd_write(vgd_ptr, fileId):
    """
    """
    ok = vgd.c_vgd_write_desc(vgd_ptr, fileId)
    if ok /= vgd.VGD_OK:
        raise VGDError('Problem writing vgd to file (id=%d)' % fileId)
    return
    

def vgd_free(vgd_ptr):
    """
    """
    vgd.c_vgd_free(vgd_ptr)
    return


def vgd_get_opt(key, quiet=1):
    """
    """
    key2 = key.upper()
    if not key2 in vgd.VGD_OPR_KEYS['getopt_int']:
        raise VGDError('Problem getting opt, invalid key (key=%s)' % key)
    v1 = ct.c_int(0)
    ok = vgd.c_vgd_getopt_int(key2, ct.byref(v1), quiet)
    if ok /= vgd.VGD_OK:
        raise VGDError('Problem getting opt (key=%s)' % key)
    return v1.value
    

def vgd_put_opt(key, value):
    """
    """
    key2 = key.upper()
    if not key2 in vgd.VGD_OPR_KEYS['putopt_int']:
        raise VGDError('Problem setting opt, invalid key (key=%s)' % key)
    ok = vgd.c_vgd_putopt_int(key2, value)
    if ok /= vgd.VGD_OK:
        raise VGDError('Problem setting opt (key=%s, value=%s)' % (key, repr(value)))
    return


def vgd_get(vgd_ptr, key, quiet=1):
    """
    """
    key2 = key.upper()[0:4]
    
    if   key2 in vgd.VGD_OPR_KEYS['get_char']:
        v1 = _C_MKSTR(' '*vgd.VGD_MAXSTR_ETIKET)
        ok = vgd.c_vgd_get_char(vgd_ptr, key2, v1, quiet)
        if ok == vgd.VGD_OK:
            v1 = v1.value.strip()
    elif key2 in vgd.VGD_OPR_KEYS['get_int']:
        v1 = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd_ptr, key2, ct.byref(v1), quiet)
        if ok == vgd.VGD_OK:
            v1 = v1.value
    elif key2 in vgd.VGD_OPR_KEYS['get_float']:
        v1 = ct.c_float(0.)
        ok = vgd.c_vgd_get_float(vgd_ptr, key2, ct.byref(v1), quiet)
        if ok == vgd.VGD_OK:
            v1 = v1.value
    elif key2 in vgd.VGD_OPR_KEYS['get_int_1d']:
        v1 = ct.POINTER(ct.c_int)()
        nv = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd_ptr, key2, ct.byref(v1),
                                  ct.byref(nv), quiet)
        if ok == vgd.VGD_OK:
            v1 = [v.value for v in v1[0:nv.value].value]
    elif key2 in vgd.VGD_OPR_KEYS['get_float_1d']:
        v1 = ct.POINTER(ct.c_float)()
        nv = ct.c_int(0)
        ok = vgd.c_vgd_get_float_1d(vgd_ptr, key2, ct.byref(v1),
                                    ct.byref(nv), quiet)
        if ok == vgd.VGD_OK:
            v1 = [v.value for v in v1[0:nv.value].value]
    elif key2 in vgd.VGD_OPR_KEYS['get_double']:
        v1 = ct.c_double(0.)
        ok = vgd.c_vgd_get_float(vgd_ptr, key2, ct.byref(v1), quiet)
        if ok == vgd.VGD_OK:
            v1 = v1.value
    elif key2 in vgd.VGD_OPR_KEYS['get_double_1d']:
        v1 = ct.POINTER(ct.c_double)()
        nv = ct.c_int(0)
        ok = vgd.c_vgd_get_double_1d(vgd_ptr, key2, ct.byref(v1),
                                     ct.byref(nv), quiet)
        if ok == vgd.VGD_OK:
            v1 = [v.value for v in v1[0:nv.value].value]
    elif key2 in vgd.VGD_OPR_KEYS['get_double_3d']:
        v1 = ct.POINTER(ct.c_double)()
        ni = ct.c_int(0)
        nj = ct.c_int(0)
        nk = ct.c_int(0)
        ok = vgd.c_vgd_get_double_3d(vgd_ptr, key2, ct.byref(v1),
                                     ct.byref(ni), ct.byref(nj),
                                     ct.byref(nk), quiet)
        if ok == vgd.VGD_OK:
            nv = ni.value * nj.value * nk.value
            v1 = [v.value for v in v1[0:nv].value]
    else:
        raise VGDError('Problem getting val, invalid key (key=%s)' % key)

    if ok /= vgd.VGD_OK:
        raise VGDError('Problem getting val (key=%s)' % key)
    return v1


def vgd_put(vgd_ptr, key, value):
    """
    """
    key2 = key.upper()[0:4]
    if   key2 in vgd.VGD_OPR_KEYS['put_char']:
        v1 = _C_MKSTR(str(value))
        ok = vgd.c_vgd_put_char(vgd_ptr, key2, v1)
    elif key2 in vgd.VGD_OPR_KEYS['put_int']:
        ok = vgd.c_vgd_put_int(vgd_ptr, key2, value)
    elif key2 in vgd.VGD_OPR_KEYS['put_double']:
        ok = vgd.c_vgd_put_double(vgd_ptr, key2, value)
    else:
        raise VGDError('Problem setting val, invalid key (key=%s)' % key)
    if ok /= vgd.VGD_OK:
        raise VGDError('Problem setting val (key=%s, value=%s)' % (key, repr(value)))
    return


def vgd_cmp(vgd0ptr, vgd1ptr):
    """
    """
    ok = vgd.c_vgd_vgdcmp(vgd0ptr, vgd1ptr)
    return ok


def vgd_levels():
    """
    """
    raise VGDError('Not Implemented yet')

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
