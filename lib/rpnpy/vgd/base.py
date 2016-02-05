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
import rpnpy.librmn.all as _rmn

_C_MKSTR = _ct.create_string_buffer

def vgd_new():
    """
    """
    raise VGDError('Not Implemented yet')


def vgd_read(fileId, ip1=-1, ip2=-1, kind=-1, version=-1):
    """
    """
    vgd_ptr = _vp.c_vgd_construct()
    ok = _vp.c_vgd_new_read(vgd_ptr, fileId, ip1, ip2, kind, version)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting vgd from file (id=%d, ip1=%d, ip2=%d, kind=%d, version=%d)' % (fileId, ip1, ip2, kind, version))
    return vgd_ptr


def vgd_write(vgd_ptr, fileId):
    """
    """
    ok = _vp.c_vgd_write_desc(vgd_ptr, fileId)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem writing vgd to file (id=%d)' % fileId)
    return
    

def vgd_free(vgd_ptr):
    """
    """
    _vp.c_vgd_free(vgd_ptr)
    return


def vgd_tolist(vgd_ptr):
    """
    """
    return vgd_get(vgd_ptr, 'VTBL')


def vgd_fromlist(vgd_table):
    """
    """
    vgd_ptr = _vp.c_vgd_construct()
    vtbl = vgd_table.ctypes.data_as(_ct.POINTER(_ct.c_double))
    (ni, nj, nk) = vgd_table.shape
    ok = _vp.c_vgd_new_from_table(vgd_ptr, vtbl, ni, nj, nk)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem Creating a new VGD from provided table)')
    return vgd_ptr


def vgd_get_opt(key, quiet=1):
    """
    """
    key2 = key.upper()
    if not key2 in _vc.VGD_OPR_KEYS['getopt_int']:
        raise VGDError('Problem getting opt, invalid key (key=%s)' % key)
    v1 = _ct.c_int(0)
    ok = _vp.c_vgd_getopt_int(key2, _ct.byref(v1), quiet)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting opt (key=%s)' % key)
    return v1.value
    

def vgd_put_opt(key, value):
    """
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
        raise VGDError('Problem getting val, invalid key (key=%s)' % key)

    if ok != _vc.VGD_OK:
        raise VGDError('Problem getting val (key=%s)' % key)
    return v1


def vgd_put(vgd_ptr, key, value):
    """
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
        raise VGDError('Problem setting val, invalid key (key=%s)' % key)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem setting val (key=%s, value=%s)' % (key, repr(value)))
    return


def vgd_cmp(vgd0ptr, vgd1ptr):
    """
    """
    ok = _vp.c_vgd_vgdcmp(vgd0ptr, vgd1ptr)
    return ok


def vgd_levels(vgd_ptr, rfld=None, ip1list='VIPM',  in_log=_vc.VGD_DIAG_PRES, dpidpis=_vc.VGD_DIAG_DPIS, double_precision=False):
    """
    """
    if isinstance(ip1list,str):
        ip1list0 = vgd_get(vgd_ptr, ip1list)
        ip1list1 = _np.array(ip1list0, dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,(int,long)):
        ip1list1 = _np.array([ip1list], dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,(list,tuple)):
        ip1list1 = _np.array(ip1list, dtype=_np.int32, order='FORTRAN')
    elif isinstance(ip1list,_np.ndarray):
        ip1list1 = ip1list.flatten()
    else:
        raise TypeError('ip1list should be string, list or int: %s' % str(type(ip1list)))
    nip1    = ip1list1.size
    ip1list = ip1list1.ctypes.data_as(_ct.POINTER(_ct.c_int))

    rfld_rank = 0
    if isinstance(rfld,float):
        rfld = _np.array([rfld], dtype=_np.float32, order='FORTRAN')
    elif isinstance(rfld,(list,tuple)):
        rfld = _np.array(rfld, dtype=_np.float32, order='FORTRAN')
        rfld_rank = 1
    elif isinstance(rfld,int):
        rfld_rank = 2
        fileId = rfld
        #TODO: check if rfld is needed for vgd_ptr kind,version
        rfld_name = vgd_get(vgd_ptr, 'RFLD')
        if not rfld_name:
            raise VGDError('Problem getting RFLD to compute levels')
        rfld = _rmn.fstlir(fileId, nomvar=rfld_name.strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA
    elif rfld is None:
        raise TypeError('Need to check if RFLD is needed')
    elif not isinstance(rfld,_np.ndarray):
        raise TypeError('rfld should be ndarray, list or float: %s' % str(type(ip1list)))

    if len(rfld.shape) == 1:
        (ni,nj) = (rfld.shape[0],1)
    else:
        (ni,nj) = rfld.shape[0:2]
    if double_precision:
        dtype = _np.float64
        rfld8 = _np.array(rfld, copy=True, dtype=dtype, order='FORTRAN')
    else:
        dtype = _np.float32
        rfld8 = rfld
    levels8 = _np.empty((ni, nj, nip1), dtype=dtype, order='FORTRAN')
        
    if double_precision:
        ok = _vp.c_vgd_diag_withref_8(vgd_ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log, dpidpis)
    else:
        ok = _vp.c_vgd_diag_withref(vgd_ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log, dpidpis)
    if ok != _vc.VGD_OK:
        raise VGDError('Problem computing levels.')
    if (rfld_rank,nj) == (1,1):
        levels8 = levels8.reshape(levels8, (nj, nip1), order='F')
    elif (rfld_rank,ni,nj) == (0,1,1):
        levels8 = levels8.flatten()
    return levels8

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
