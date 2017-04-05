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

def brp_SetOptFloat(optName, optValue):
    """
    Set BURP file float option (alias to brp_opt)

    brp_SetOptFloat(optName, optValue)

    Args:
        optName  : name of option to be set
        optValue : value to be set (float)
    Returns:
        None
    Raises:
        KeyError   on unknown optName
        TypeError  on wrong input arg types
        BurpError  on any other error

    See Also:
        brp_opt
        rpnpy.librmn.burp.mrfopt
        rpnpy.librmn.burp_const
        rpnpy.burpc.const
    """
    brp_opt(optName, optValue)


def brp_SetOptChar(optName, optValue):
    """
    Set BURP file char option (alias to brp_opt)

    brp_SetOptChar(optName, optValue)

    Args:
        optName  : name of option to be set
        optValue : value to be set (char)
    Returns:
        None
    Raises:
        KeyError   on unknown optName
        TypeError  on wrong input arg types
        BurpError  on any other error

    See Also:
        brp_opt
        rpnpy.librmn.burp.mrfopt
        rpnpy.librmn.burp_const
        rpnpy.burpc.const
    """
    brp_opt(optName, optValue)


def brp_msngval():
    """
    return the floating point constant used for missing values

    missingVal = brp_msngval()

    Args:
        None
    Returns:
        float, constant used for missing values

    See Also:
        brp_opt
    """
    return _bp.c_brp_msngval()


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
        rpnpy.burpc.const
    """
    if not optName in (_rmn.BURPOP_MISSING, _rmn.BURPOP_MSGLVL):
        raise KeyError("Uknown optName: {}".format(optName))

    if optValue is None:
        if optName == _rmn.BURPOP_MISSING:
            return _bp.c_brp_msngval()
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

    See Also:
        brp_close
        rpnpy.burpc.brpobj.BurpcFile
        rpnpy.burpc.const
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

    See Also:
        brp_open
        rpnpy.burpc.brpobj.BurpcFile
        rpnpy.burpc.const
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


##---- allocators and constructors --------------------------------------

def brp_newrpt():
    """
    
    See Also:
        brp_free
        rpnpy.burpc.brpobj.BurpcRpt
    """
    return _bp.c_brp_newrpt() #TODO: should it return BurpcRpt?

def brp_newblk():
    """
    
    See Also:
        brp_free
        rpnpy.burpc.brpobj.BurpcBlk
    """
    return _bp.c_brp_newblk() #TODO: should it return BurpcBlk?


def brp_allocrpt(rpt, size):
    """
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_allocrpt(rpt, size)
    if _bp.RPT_NSIZE(rpt) != size:
        raise BurpcError('Problem allocating report')

def brp_resizerpt(rpt, size):
    """
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_resizerpt(rpt, size)
    if _bp.RPT_NSIZE(rpt) != size:
        raise BurpcError('Problem resizing report')

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
    if ((_bp.BLK_NELE(blk), _bp.BLK_NVAL(blk), _bp.BLK_NT(blk))
        != (nele, nval, nt)):
        raise BurpcError('Problem allocating/resizing block')

##---- destructors and deallocators -------------------------------------

def brp_freerpt(rpt):
    """
    Free pointer intances to BURP_RPT

    brp_freerpt(myBURP_RPTptr)

    Args:
        rpt : pointer to BURP_RPT structure
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_free
        brp_newrpt
        rpnpy.burpc.brpobj.BurpcRpt
    """
    brp_free(rpt)
    
def brp_freeblk(blk):
    """
    Free pointer intances to BURP_BLK

    brp_freeblk(myBURP_BLKptr)

    Args:
        blk : pointer to BURP_BLK structure
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_free
        brp_newblk
        rpnpy.burpc.brpobj.BurpcBlk
    """
    brp_free(blk)
    
def brp_free(*args):
    """
    Free pointer intances to BURP_RPT and BURP_BLK

    brp_free(myBURP_RPTptr)
    brp_free(myBURP_BLKptr, myBURP_RPTptr)

    Args:
        args: list of pointers to BURP_RPT or BURP_BLK structure
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_free
        brp_newrpt
        brp_newblk
        rpnpy.burpc.brpobj.BurpcRpt
        rpnpy.burpc.brpobj.BurpcBlk
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


##---- reinitializer ----------------------------------------------------

def brp_clrrpt(rpt):
    """
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_clrrpt(rpt)

def brp_clrblkv(blk, val=None):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if val is None:
        _bp.c_brp_clrblk(blk)
    else:
        _bp.c_brp_clrblkv(blk, val)

def brp_clrblk(blk):
    """
    """
    brp_clrblkv(blk)

def brp_resetrpthdr(rpt):
    """
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_resetrpthdr(rpt)

def brp_resetblkhdr(blk):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    _bp.c_brp_resetblkhdr(blk)

##---- converters -------------------------------------------------------

def brp_encodeblk(blk):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_encodeblk(blk) < 0:
        raise BurpcError('Problem in c_brp_encodeblk')

def brp_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_convertblk(blk, mode) < 0:
        raise BurpcError('Problem in c_brp_convertblk')

def brp_safe_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
    """
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_safe_convertblk(blk, mode) < 0:
        raise BurpcError('Problem in c_brp_convertblk')

##---- find report and block before reading -----------------------------

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

def brp_searchdlste(code, blk):
    """
    find elements matching code in block
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _bo.BurpcBlk):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    return c_brp_searchdlste(code, blk)

##---- read in data -----------------------------------------------------

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

def brp_safe_getblk(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
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
    if _bp.c_brp_safe_getblk(bkno, blk.getptr(), rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_safe_getblk')
    return blk

def brp_readblk(bkno, blk=None, rpt=None, cvt=0): #TODO: how can we get a block in an empty report?
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
    if _bp.c_brp_readblk(bkno, blk.getptr(), rpt.getptr(), cvt) < 0:
        raise BurpcError('Problem in c_brp_readblk')
    return blk

##---- read in header ---------------------------------------------------

def brp_rdrpthdr(handle=0, rpt=None):
    """
    """
    if isinstance(handle, _bo.BurpcRpt):
        if not rpt:
            rpt = handle
        handle = handle.handle
    if not isinstance(rpt, _bo.BurpcRpt):
        rpt = _bo.BurpcRpt(rpt)
    if _bp.c_brp_rdrpthdr(handle, rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_rdrpthdr')
    return rpt

def brp_rdblkhdr(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
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
    if _bp.c_brp_rdblkhdr(bkno, blk.getptr(), rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_rdblkhdr')
    return blk

##---- writing ----------------------------------------------------------

def brp_initrpthdr(funit, rpt):
    """
    prepare a report for writing
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if not isinstance(rpt, _bo.BurpcRpt):
        rpt = _bo.BurpcRpt(rpt)
    if _bp.c_brp_initrpthdr(funit, rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_initrpthdr')
    return rpt

def brp_putrpthdr(funit, rpt):
    """
    prepare a report for writing (alias of brp_initrpthdr)
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_putrpthdr(funit, rpt) < 0:
        raise BurpcError('Problem in c_brp_putrpthdr')

def brp_updrpthdr(funit, rpt):
    """
    modify only the header of a report
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
    write out report to a file
    """
    if isinstance(funit, _bo.BurpcFile):
        funit = funit.funit
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_writerpt(funit, rpt, where) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')

def brp_putblk(rpt, blk):
    """
    add new blocks into a report
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_putblk(rpt, blk) < 0:
        raise BurpcError('Problem in c_brp_putblk')

##---- utilisites -------------------------------------------------------

def brp_copyrpthdr(dst_rpt, src_rpt):
    """
    copy rpt header
    """
    if isinstance(src_rpt, _bo.BurpcRpt):
        src_rpt = rpt.getptr()
    if not isinstance(src_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not isinstance(dst_rpt, _bo.BurpcRpt):
        dst_rpt = _bo.BurpcRpt(dst_rpt)
    _bp.c_brp_copyrpthdr(dst_rpt.getptr(), src_rpt)
    return dst_rpt

def brp_copyrpt(dst_rpt, src_rpt):
    """
    copy the whole rpt
    """
    if isinstance(src_rpt, _bo.BurpcRpt):
        src_rpt = rpt.getptr()
    if not isinstance(src_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not isinstance(dst_rpt, _bo.BurpcRpt):
        dst_rpt = _bo.BurpcRpt(dst_rpt)
    _bp.c_brp_copyrpt(dst_rpt.getptr(), src_rpt)
    return dst_rpt

def brp_copyblk(dst_blk, src_blk):
    """
    duplicate block
    """
    if isinstance(src_blk, _bo.BurpcBlk):
        src_blk = src_blk.getptr()
    if not isinstance(src_blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use src_blk or type={}'+str(type(src_blk)))
    if not isinstance(dst_blk, _bo.BurpcBlk):
        dst_blk = _bo.BurpcBlk(dst_blk)
    _bp.c_brp_copyblk(dst_blk.getptr(), src_blk)
    return dst_blk

##---- deleting ---------------------------------------------------------

def brp_delblk(rpt, blk):
    """
    delete block from report
    """
    if not isinstance(rpt, _bo.BurpcRpt):
        rpt = _bo.BurpcRpt(rpt)    
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_delblk(rpt, blk) <0:
        raise BurpcError('Problem in c_brp_delblk')
    return rpt

def brp_delblk(rpt):
    """
    delete report
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_delrpt(rpt) <0:
        raise BurpcError('Problem in c_brp_delrpt')

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
