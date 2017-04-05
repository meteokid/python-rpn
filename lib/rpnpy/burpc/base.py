#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc.base contains python wrapper to main burp_c C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] documentation for more details.

See Also:
    rpnpy.burpc.brpobj
    rpnpy.burpc.proto
    rpnpy.burpc.const
"""
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from rpnpy.burpc import proto as _bp
from rpnpy.burpc import const as _bc
from rpnpy.burpc import brpobj as _bo
from rpnpy.burpc import BurpcError
import rpnpy.librmn.all as _rmn

from rpnpy import integer_types as _integer_types

## _C_MKSTR = _ct.create_string_buffer

def brp_opt(optName, optValue=None):
    """
    Set/Get BURP file options

    brp_opt(optName, optValue)

    Args:
        optName  : name of option to be set or printed
                   or one of these constants:
                   BURPOP_MISSING, BURPOP_MSGLVL
        optValue : value to be set (float or string) (optional)
                   If not set or is None mrfopt will get the value
                   otherwise mrfopt will set to the provided value
                   for optName=BURPOP_MISSING:
                      a real value for missing data
                   for optName=BURPOP_MSGLVL, one of these constants:
                      BURPOP_MSG_TRIVIAL,   BURPOP_MSG_INFO,  BURPOP_MSG_WARNING,
                      BURPOP_MSG_ERROR,     BURPOP_MSG_FATAL, BURPOP_MSG_SYSTEM
    Returns:
        str or float, optValue
    Raises:
        KeyError   on unknown optName
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> # Restrict to the minimum the number of messages printed by librmn
    >>> brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    'SYSTEM   '

    See Also:
        rpnpy.librmn.burp.mrfopt
        rpnpy.librmn.burp_const
    """
    if not optName in (_rmn.BURPOP_MISSING, _rmn.BURPOP_MSGLVL):
        raise KeyError("Uknown optName: {}".format(optName))

    if optValue is None:
        if optName == _rmn.BURPOP_MISSING:
            return _bp.c_brp_msngval() #TODO: should it be option.value?
        else:
            raise KeyError("Cannot get value for optName: {}".format(optName))

    if isinstance(optValue, str):
        istat = _bp.c_brp_SetOptChar(optName, optValue)
        if istat != 0:
            raise BurpcError('c_brp_SetOptChar: {}={}'.format(optName,optValue))
    elif isinstance(optValue, float):
        istat = _bp.c_brp_SetOptFloat(optName, optValue)
        if istat != 0:
            raise BurpcError('c_mrfopr:{}={}'.format(optName,optValue), istat)
    else:
        raise TypeError("Cannot set optValue of type: {0} {1}"\
                        .format(type(optValue), repr(optValue)))
    return optValue


def brp_open(filename, filemode='r', funit=0, getnbr=False):
    """
    #TODO: desc
    """
    fstmode, brpmode, brpcmode = _bp.brp_filemode(filemode)
    #TODO: Check format/existence of file depending on mode as in burp_open
    if not funit:
        try:
            funit = _rmn.get_funit(filename, fstmode)
        except _rmn.RMNBaseError:
            funit = 0
    if not funit:
        raise BurpcError('Problem associating a unit with file: {} (mode={})'
                        .format(filename, filemode))
    nrep = _bp.c_brp_open(funit, filename, brpcmode)
    if getnbr:
        return (funit, nrep)
    return funit


def brp_close(funit):
    """
    #TODO: desc
    """
    if isinstance(funit, _bo.BurpcFile):
        funit.close()
    elif isinstance(funit, _integer_types):
        istat = _bp.c_brp_close(funit)
        if istat < 0:
            raise BurpcError('Problem closing burp file unit: "{}"'
                             .format(funit))
    else:
        raise TypeError('funit is type="{}"'.format(str(type(funit))) +
                        ', should be an "int" or a "BurpcFile"')


def brp_free(*args):
    """
    Free pointer intances to BURP_RPT and BURP_BLK

    brpc_free(myBURP_RPTptr)
    brpc_free(myBURP_BLKptr, myBURP_RPTptr)

    Args:

    Return:
        None
    Raises:
        TypeError on not supported types or args
    """
    for x in args:
        if isinstance(x, _ct.POINTER(_bp.BURP_RPT)):
            _bp.c_brp_freerpt(x)
        elif isinstance(x, _ct.POINTER(_bp.BURP_BLK)):
            _bp.c_brp_freeblk(x)
        ## elif isinstance(x, _bo.BurpcRpt, _bo.BurpcBlk):
        ##     x.__del__()
        else:
            raise TypeError("Not Supported Type: "+str(type(x)))


def brp_findrpt(funit, rpt=None): #TODO: rpt are search keys, change name
    """
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if not rpt:
        rpt = _bo.BurpcRpt()
        rpt.handle = 0
    elif isinstance(rpt, _integer_types):
        handle = rpt
        rpt = _bo.BurpcRpt()
        rpt.handle = handle
    elif not isinstance(rpt, _bo.BurpcRpt):
        rpt = _bo.BurpcRpt(rpt)
    if _bp.c_brp_findrpt(funit, rpt.getptr()) >= 0:
        return rpt
    return None


def brp_getrpt(funit, handle=0, rpt=None):
    """
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if isinstance(handle, _bo.BurpcRpt):
        if not rpt:
            rpt = handle
        handle = handle.handle
    if not isinstance(rpt, _bo.BurpcRpt):
        rpt = _bo.BurpcRpt(rpt)
    if _bp.c_brp_getrpt(funit, handle, rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_getrpt')
    return rpt


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = _bo.BurpcRpt(rpt)
    if not isinstance(rpt, _bo.BurpcRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not blk:
        blk = _bo.BurpcBlk()
        blk.bkno = 0
    elif isinstance(blk, _integer_types):
        bkno = blk
        blk = _bo.BurpcBlk()
        blk.bkno = bkno
    elif not isinstance(blk, _bo.BurpcBlk):
        blk = _bo.BurpcBlk(blk)
    if _bp.c_brp_findblk(blk.getptr(), rpt.getptr()) >= 0:
        return blk
    return None


def brp_getblk(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = _bo.BurpcRpt(rpt)
    if not isinstance(rpt, _bo.BurpcRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not isinstance(blk, _bo.BurpcBlk):
        blk = _bo.BurpcBlk(blk)
    else:
        blk.reset_arrays()
    if _bp.c_brp_getblk(bkno, blk.getptr(), rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_getblk')
    return blk


## def brp_allocrpt(rpt, size):
##     """
##     """
##     if isinstance(rpt, _bo.BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_allocrpt(rpt, size) < 0:
##         raise BurpcError('Problem in brp_allocrpt')

## def brp_resizerpt(rpt, size):
##     """
##     """
##     if isinstance(rpt, _bo.BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_resizerpt(rpt, size) < 0:
##         raise BurpcError('Problem in brp_resizerpt')

## def brp_clrrpt(rpt):
##     """
##     """
##     if isinstance(rpt, _bo.BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_clrrpt(rpt) < 0:
##         raise BurpcError('Problem in c_brp_clrrpt')

## def brp_putrpthdr(funit, rpt):
##     """
##     """
##     if isinstance(funit, _bo.BurpcFile):
##         funit = funit.funit
##     if isinstance(rpt, _bo.BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_putrpthdr(funit, rpt) < 0:
##         raise BurpcError('Problem in c_brp_putrpthdr')

def brp_updrpthdr(funit, rpt):
    """
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_updrpthdr(funit, rpt) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')

def brp_writerpt(funit, rpt, where=_bc.BRP_END_BURP_FILE):
    """
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_writerpt(funit, rpt, where) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')

def brp_allocblk(blk, nele=None, nval=None, nt=None):
    """
    """
    brp_resizeblk(blk, nele, nval, nt)

def brp_resizeblk(blk, nele=None, nval=None, nt=None):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if nele is None: nele = blk[0].max_nele
    if nval is None: nval = blk[0].max_nval
    if nt   is None: nt   = blk[0].max_nt
    nele, nval, nt = int(max(1, nele)), int(max(1, nval)), int(max(1, nt))
    if blk[0].max_nele == 0:
        _bp.c_brp_allocblk(blk, nele, nval, nt)
    else:
        _bp.c_brp_resizeblk(blk, nele, nval, nt)

## def brp_encodeblk(blk):
##     """
##     """
##     if isinstance(blk, _bo.BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_encodeblk(blk) < 0:
##         raise BurpcError('Problem in c_brp_encodeblk')

## def brp_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
##     """
##     """
##     if isinstance(blk, _bo.BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_convertblk(blk, mode) < 0:
##         raise BurpcError('Problem in c_brp_convertblk')

## def brp_putblk(rpt, blk):
##     """
##     """
##     if isinstance(rpt, _bo.BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if isinstance(blk, _bo.BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_putblk(rpt, blk) < 0:
##         raise BurpcError('Problem in c_brp_putblk')

## def c_brp_copyblk(dst_blk, src_blk):
##     """
##     """
##     if isinstance(dst_blk, _bo.BurpcBlk):
##         dst_blk = dst_blk.getptr()
##     if not isinstance(dst_blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use dst_blk or type={}'+str(type(dst_blk)))
##     if isinstance(src_blk, _bo.BurpcBlk):
##         src_blk = src_blk.getptr()
##     if not isinstance(src_blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use src_blk or type={}'+str(type(src_blk)))
##     if _bp.c_brp_copyblk(dst_blk, src_blk) < 0:
##         raise BurpcError('Problem in c_brp_copyblk')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
