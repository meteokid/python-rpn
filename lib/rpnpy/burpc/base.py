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

    For an OO API to burp files, see: rpnpy.burpc.brpobj

    This module is new in version 2.1.b2

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
    >>> import rpnpy.librmn.all as rmn
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


def brp_open(filename, filemode=_bc.BRP_FILE_READ, funit=0, getnbr=False):
    """
    Open a BURP file

    funit = brp_open(filename)
    funit = brp_open(filename, filemode)
    funit, nrpt = brp_open(filename, getnbr=True)

    Args:
        filename : Name of the file to open
        filemode : (opitonal) open mode, one of:
                   BRP_FILE_READ, BRP_FILE_WRITE, BRP_FILE_APPEND
                   (default: BRP_FILE_READ
        funit    : (opitonal) file unit number
                   (default=0, auto-select an avail. unit)
        getnbr   : (opitonal) if True, will return number of report
                   in the file as well as the file unit number
    Returns:
        funit         : (if getnbr==False) file unit number [int]
        (funit, nrpt) : (if getnbr==True) where
                        funit: file unit number [int]
                        nrpt : number of report in the file [int]
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> filename = 'tmpburpfile.brp'
    >>> funit = brp.brp_open(filename, brp.BRP_FILE_WRITE)
    >>> brp.brp_close(funit)

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
    Close a previously Opened BURP file

    brp_close(funit)

    Args:
        funit
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> filename = 'tmpburpfile.brp'
    >>> funit = brp.brp_open(filename, brp.BRP_FILE_WRITE)
    >>> brp.brp_close(funit)

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
    Create a new BURP_RPT structure

    rpt = brp_newrpt()

    Args:
        None
    Returns:
        ctypes.POINTER(BURP_RPT)

    See Also:
        brp_free
        rpnpy.burpc.brpobj.BurpcRpt
    """
    return _bp.c_brp_newrpt() #TODO: should it return BurpcRpt?

def brp_newblk():
    """
    Create a new BURP_BLK structure

    blk = brp_newblk()

    Args:
        None
    Returns:
        ctypes.POINTER(BURP_BLK)

    See Also:
        brp_free
        rpnpy.burpc.brpobj.BurpcBlk
    """
    return _bp.c_brp_newblk() #TODO: should it return BurpcBlk?


def brp_allocrpt(rpt, size):
    """
    Allocate report Memory, in place, to add blocks in the report

    brp_allocrpt(rpt, size)

    Args:
        rpt  : report pointer to allocate memory into [ctypes.POINTER(BURP_BLK)]
               as obtained from brp_newrpt()
        size : size to be allocated (TODO: units?) [int]
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> rpt = brp.brp_newrpt()
    >>> brp.brp_allocrpt(rpt, 999)

    See Also:
        brp_newrpt
        brp_resizerpt
        brp_freerpt
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
    Rezise report Memory allocation, in place

    brp_allocrpt(rpt, size)

    Args:
        rpt  : report pointer to resize memory into [ctypes.POINTER(BURP_BLK)]
               as obtained from brp_newrpt()
        size : size to be allocated (TODO: units?) [int]
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> rpt = brp.brp_newrpt()
    >>> brp.brp_allocrpt(rpt, 999)
    >>> brp.brp_resizerpt(rpt, 2222)

    See Also:
        brp_newrpt
        brp_allocrpt
        brp_freerpt
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
    Allocate block Memory, in place, to add elements to the block

    brp_allocblk(blk, nele, nval, nt)

    Args:
        blk  : report pointer to allocate memory into [ctypes.POINTER(BURP_BLK)]
               as obtained from brp_newblk()
        nele : number of elements [int]
        nval : number of values per element[int]
        nt   : Number of groups of NELE by NVAL values [int]
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> brp.brp_allocblk(blk, 5, 1, 1)

    See Also:
        brp_newblk
        brp_resizeblk
        brp_freeblk
     """
    brp_resizeblk(blk, nele, nval, nt)

def brp_resizeblk(blk, nele=None, nval=None, nt=None):
    """
    Resize block Memory, in place

    brp_resizeblk(blk, nele, nval, nt)

    Args:
        blk  : report pointer to allocate memory into [ctypes.POINTER(BURP_BLK)]
               as obtained from brp_newblk()
        nele : number of elements [int]
        nval : number of values per element[int]
        nt   : Number of groups of NELE by NVAL values [int]
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> brp.brp_allocblk(blk, 5, 1, 1)
    >>> brp.brp_resizeblk(blk, 9, 1, 1)

    See Also:
        brp_newblk
        brp_allocblk
        brp_freeblk
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

    brp_freerpt(rpt)

    Args:
        rpt : pointer to BURP_RPT structure [ctypes.POINTER(BURP_RPT)]
              as obtained from brp_newrpt()
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

    brp_freeblk(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
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

    brp_free(rpt)
    brp_free(blk, rpt)

    Args:
        args: list of pointers to BURP_RPT or BURP_BLK structure
              as obtained from brp_newrpt() or brp_newblk()
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
    Init report data buffer before adding blocks.
    Needs to be followd by a call to brp_putrpthdr.

    brp_clrrpt(rpt)

    Args:
        rpt : pointer to BURP_RPT structure [ctypes.POINTER(BURP_RPT)]
              as obtained from brp_newrpt()
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_newrpt
        brp_resetrpthdr
        brp_putrpthdr
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_clrrpt(rpt)

def brp_clrblkv(blk, val=None):
    """
    Init block data with provided value

    brp_clrblkv(blk, val)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        val : value used to init the block data (float)
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblk
        brp_resetblkhdr
        brp_newblk
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
    Init block data with missing value

    brp_clrblkv(blk, val)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        val : value used to init the block data (float)
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblkv
        brp_resetblkhdr
        brp_newblk
        brp_msngval
    """
    brp_clrblkv(blk)

def brp_resetrpthdr(rpt):
    """
    Reset report header info

    brp_resetrpthdr(rpt)

    Args:
        rpt : pointer to BURP_RPT structure [ctypes.POINTER(BURP_RPT)]
              as obtained from brp_newrpt()
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrrpt
        brp_newrpt
    """
    if isinstance(rpt, _bo.BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_resetrpthdr(rpt)

def brp_resetblkhdr(blk):
    """
    Reset report header info

    brp_resetblkhdr(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
    Return:
        None
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblk
        brp_newblk
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    _bp.c_brp_resetblkhdr(blk)

##---- converters -------------------------------------------------------

def brp_encodeblk(blk):
    """
    Encode block elements (dlstele) into lstele
    To be used after fill block elements code with BLK_SetDLSTELE

    brp_encodeblk(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
    Return:
        None
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> brp.brp_allocblk(blk, 1,1,1)
    >>> brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> brp.brp_encodeblk(blk)
    >>> print('# '+repr(brp.BLK_LSTELE(blk, 0)))
    # 2564

    See Also:
        brp_newblk
        brp_allocblk
        brp_clrblk
        brp_convertblk
        rpnpy.burpc.proto
     """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_encodeblk(blk) < 0:
        raise BurpcError('Problem in c_brp_encodeblk')

def brp_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
    """
    Convert block's element encoded values there real counterpart
    or the reverse operation depending on mode

    brp_encodeblk(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        mode: one of MKSA_to_BUFR, BUFR_to_MKSA
              if BLK_STORE_TYPE == STORE_FLOAT then
              MKSA_to_BUFR would convert from rval to tblval 
              BUFR_to_MKSA would convert from tblval to rval 
    Return:
        None
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> brp.brp_allocblk(blk, 1,1,1)
    >>> brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> brp.brp_encodeblk(blk)
    >>> brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
    >>> brp.brp_convertblk(blk, brp.BRP_MKSA_to_BUFR)
    >>> print('# '+repr(brp.BLK_TBLVAL(blk, 0, 0, 0)))
    # 10

    See Also:
        brp_safe_convertblk
        brp_newblk
        brp_allocblk
        brp_clrblk
        brp_encodeblk
        rpnpy.burpc.proto
        rpnpy.burpc.const
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_convertblk(blk, mode) < 0:
        raise BurpcError('Problem in c_brp_convertblk')

def brp_safe_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
    """
    Convert block's element encoded values there real counterpart
    or the reverse operation depending on mode

    brp_safe_encodeblk(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        mode: one of MKSA_to_BUFR, BUFR_to_MKSA
              if BLK_STORE_TYPE == STORE_FLOAT then
              MKSA_to_BUFR would convert from rval to tblval 
              BUFR_to_MKSA would convert from tblval to rval 
    Return:
        None
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> brp.brp_allocblk(blk, 1,1,1)
    >>> brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> brp.brp_encodeblk(blk)
    >>> brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
    >>> brp.brp_safe_convertblk(blk, brp.BRP_MKSA_to_BUFR)
    >>> print('# '+repr(brp.BLK_TBLVAL(blk, 0, 0, 0)))
    # 10

    See Also:
        brp_convertblk
        brp_newblk
        brp_allocblk
        brp_clrblk
        brp_encodeblk
        rpnpy.burpc.proto
        rpnpy.burpc.const
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
    """
    if isinstance(blk, _bo.BurpcBlk):
        blk = blk.getptr()
    if not isinstance(blk, _bo.BurpcBlk):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    return c_brp_searchdlste(code, blk)

##---- read in data -----------------------------------------------------

def brp_getrpt(funit, handle=0, rpt=None):
    """

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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

    Args:
    Return:
    Raises:
    See Also:
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
