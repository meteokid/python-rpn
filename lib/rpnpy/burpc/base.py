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


_RPTPTR = lambda rpt: rpt.getptr() if (isinstance(rpt, _bo.BurpcRpt)) else rpt
_BLKPTR = lambda blk: blk.getptr() if (isinstance(blk, _bo.BurpcBlk)) else blk

## /*
##  * Macros for getting values of a Report
##  * acces to field in the BURP_RPT structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  * current definition do not cover all available field
##  * since they are not used by anyone yet.
##  */

#TODO: for optimization purpose... avoid using _RPTPTR, _BLKPTR, especially in table data

# Get burp report handle
RPT_HANDLE = lambda rpt: _RPTPTR(rpt)[0].handle
# Get burp report nsize
RPT_NSIZE  = lambda rpt: _RPTPTR(rpt)[0].nsize
# Get burp report temps
RPT_TEMPS  = lambda rpt: _RPTPTR(rpt)[0].temps
# Get burp report flgs
RPT_FLGS   = lambda rpt: _RPTPTR(rpt)[0].flgs
# Get burp report stnid
RPT_STNID  = lambda rpt: _RPTPTR(rpt)[0].stnid
# Get burp report idtyp
RPT_IDTYP  = lambda rpt: _RPTPTR(rpt)[0].idtype
# Get burp report lati
RPT_LATI   = lambda rpt: _RPTPTR(rpt)[0].lati
# Get burp report long
RPT_LONG   = lambda rpt: _RPTPTR(rpt)[0].longi
# Get burp report dx
RPT_DX     = lambda rpt: _RPTPTR(rpt)[0].dx
# Get burp report dy
RPT_DY     = lambda rpt: _RPTPTR(rpt)[0].dy
# Get burp report elev
RPT_ELEV   = lambda rpt: _RPTPTR(rpt)[0].elev
# Get burp report drnd
RPT_DRND   = lambda rpt: _RPTPTR(rpt)[0].drnd
# Get burp report date
RPT_DATE   = lambda rpt: _RPTPTR(rpt)[0].date
# Get burp report oars
RPT_OARS   = lambda rpt: _RPTPTR(rpt)[0].oars
# Get burp report runn
RPT_RUNN   = lambda rpt: _RPTPTR(rpt)[0].runn
# Get burp report nblk
RPT_NBLK   = lambda rpt: _RPTPTR(rpt)[0].nblk
# Get burp report lngr
RPT_LNGR   = lambda rpt: _RPTPTR(rpt)[0].lngr


## /*
##  * Macros for setting values of a Report
##  * acces to field in the BURP_RPT structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  * current definition do not cover all available field
##  * since they are not used by anyone yet.
##  */

# Set burp report handle
def RPT_SetHANDLE(rpt, val): _RPTPTR(rpt)[0].handle = val
# Set burp report temps
def RPT_SetTEMPS (rpt, val): _RPTPTR(rpt)[0].temps = val
# Set burp report flgs
def RPT_SetFLGS  (rpt, val): _RPTPTR(rpt)[0].flgs
# Set burp report stnid
def RPT_SetSTNID (rpt, val): _bp.c_brp_setstnid(_RPTPTR(rpt), val)
# Set burp report idtyp
def RPT_SetIDTYP (rpt, val): _RPTPTR(rpt)[0].idtype = val
# Set burp report lati
def RPT_SetLATI  (rpt, val): _RPTPTR(rpt)[0].lati = val
# Set burp report long
def RPT_SetLONG  (rpt, val): _RPTPTR(rpt)[0].longi = val
# Set burp report dx
def RPT_SetDX    (rpt, val): _RPTPTR(rpt)[0].dx = val
# Set burp report dy
def RPT_SetDY    (rpt, val): _RPTPTR(rpt)[0].dy = val
# Set burp report evel
def RPT_SetELEV  (rpt, val): _RPTPTR(rpt)[0].elev = val
# Set burp report drnd
def RPT_SetDRND  (rpt, val): _RPTPTR(rpt)[0].drnd = val
# Set burp report date
def RPT_SetDATE  (rpt, val): _RPTPTR(rpt)[0].date = val
# Set burp report oars
def RPT_SetOARS  (rpt, val): _RPTPTR(rpt)[0].oars = val
# Set burp report runn
def RPT_SetRUNN  (rpt, val): _RPTPTR(rpt)[0].runn = val


## /*
##  * Macros for getting values of a Block
##  * acces to field in the BURP_BLK structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  */

# Get burp block number BKNO
BLK_BKNO  = lambda blk: _BLKPTR(blk)[0].bkno
# Get burp block NELE
BLK_NELE  = lambda blk: _BLKPTR(blk)[0].nele
# Get burp block NVAL
BLK_NVAL  = lambda blk: _BLKPTR(blk)[0].nval
# Get burp block NT
BLK_NT    = lambda blk: _BLKPTR(blk)[0].nt
# Get burp block BFAM
BLK_BFAM  = lambda blk: _BLKPTR(blk)[0].bfam
# Get burp block BDESC
BLK_BDESC = lambda blk: _BLKPTR(blk)[0].bdesc
# Get burp block BTYP
BLK_BTYP  = lambda blk: _BLKPTR(blk)[0].btyp
# Get burp block BKNAT
BLK_BKNAT = lambda blk: _BLKPTR(blk)[0].bknat
# Get burp block BKTYP
BLK_BKTYP = lambda blk: _BLKPTR(blk)[0].bktyp
# Get burp block BKSTP
BLK_BKSTP = lambda blk: _BLKPTR(blk)[0].bkstp
# Get burp block NBIT
BLK_NBIT  = lambda blk: _BLKPTR(blk)[0].nbit
# Get burp block BIT0
BLK_BIT0  = lambda blk: _BLKPTR(blk)[0].bit0
# Get burp block DATYP
BLK_DATYP = lambda blk: _BLKPTR(blk)[0].datyp
# Get burp block Data
BLK_Data  = lambda blk: _BLKPTR(blk)[0].data
# Get burp block DLSTELE
BLK_DLSTELE= lambda blk,e : _BLKPTR(blk)[0].dlstele[e]
# Get burp block LSTELE
BLK_LSTELE = lambda blk,e : _BLKPTR(blk)[0].lstele[e]
# Get burp block TBLVAL
BLK_TBLVAL = lambda blk,e,v,t: _BLKPTR(blk)[0].tblval[e + _BLKPTR(blk)[0].nele*(v + (_BLKPTR(blk)[0].nval)*t)]
# Get burp block RVAL
BLK_RVAL   = lambda blk,e,v,t: _BLKPTR(blk)[0].rval[e + _BLKPTR(blk)[0].nele*(v + (_BLKPTR(blk)[0].nval)*t)]
# Get burp block DVAL
BLK_DVAL   = lambda blk,e,v,t: _BLKPTR(blk)[0].drval[e + _BLKPTR(blk)[0].nele*(v + (_BLKPTR(blk)[0].nval)*t)]
# Get burp block CHARVAL
BLK_CHARVAL= lambda blk,l,c: _BLKPTR(blk)[0].charval[l * _BLKPTR(blk)[0].nt + c]
# Get burp block STORE_TYPE
BLK_STORE_TYPE = lambda blk: _BLKPTR(blk)[0].store_type

## /*
##  * Macros for setting values of a Block
##  * acces to field in the BURP_BLK structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  */

# Set burp block NELE
def BLK_SetNELE (blk, val): _BLKPTR(blk)[0].nele = val
# Set burp block NVAL
def BLK_SetNVAL (blk, val): _BLKPTR(blk)[0].nval = val
# Set burp block NT
def BLK_SetNT   (blk, val): _BLKPTR(blk)[0].nt = val
# Set burp block BKNO
def BLK_SetBKNO (blk, val): _BLKPTR(blk)[0].bkno = val
# Set burp block BFAM
def BLK_SetBFAM (blk, val): _BLKPTR(blk)[0].bfam = val
# Set burp block BDESC
def BLK_SetBDESC(blk, val): _BLKPTR(blk)[0].bdesc = val
# Set burp block BTYP
def BLK_SetBTYP (blk, val): _BLKPTR(blk)[0].btyp = val
# Set burp block BKNAT
def BLK_SetBKNAT(blk, val): _BLKPTR(blk)[0].bknat = val
# Set burp block BKTYP
def BLK_SetBKTYP(blk, val): _BLKPTR(blk)[0].bktyp = val
# Set burp block BKSTP
def BLK_SetBKSTP(blk, val): _BLKPTR(blk)[0].bkstp = val
# Set burp block NBIT
def BLK_SetNBIT (blk, val): _BLKPTR(blk)[0].nbit = val
# Set burp block DATYP
def BLK_SetDATYP(blk, val): _BLKPTR(blk)[0].datyp = val
# Set burp block DVAL
def BLK_SetDVAL(blk,e,v,t,val):
    _BLKPTR(blk)[0].drval [e + _BLKPTR(blk)[0].nele * (v + _BLKPTR(blk)[0].nval * t)] = val
# Set burp block TBLVAL
def BLK_SetTBLVAL(blk,e,v,t,val):
    _BLKPTR(blk)[0].tblval[e + _BLKPTR(blk)[0].nele * (v + _BLKPTR(blk)[0].nval * t)] = val
# Set burp block RVAL
def BLK_SetRVAL(blk,e,v,t,val):
    _BLKPTR(blk)[0].rval[e + _BLKPTR(blk)[0].nele * (v + _BLKPTR(blk)[0].nval * t)] = val
# Set burp block CVAL
def BLK_SetCVAL(blk,l,c,val):
    _BLKPTR(blk)[0].charval[l * _BLKPTR(blk)[0].nt + c] = val;
# Set burp block LSTELE
def BLK_SetLSTELE(blk,i,val):
    _BLKPTR(blk)[0].lstele[i] = val
# Set burp block DLSTELE
def BLK_SetDLSTELE(blk,i,val):
    _BLKPTR(blk)[0].dlstele[i] = val
# Set burp block STORE_TYPE
def BLK_SetSTORE_TYPE(blk,val):
    _BLKPTR(blk)[0].store_type = val


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
        Allocated rpt provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> rpt = brp.brp_newrpt()
    >>> rpt = brp.brp_allocrpt(rpt, 999)

    See Also:
        brp_newrpt
        brp_resizerpt
        brp_freerpt
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_allocrpt(prpt, size)
    if RPT_NSIZE(prpt) != size:
        raise BurpcError('Problem allocating report')
    return rpt


def brp_resizerpt(rpt, size):
    """
    Rezise report Memory allocation, in place

    brp_allocrpt(rpt, size)

    Args:
        rpt  : report pointer to resize memory into [ctypes.POINTER(BURP_BLK)]
               as obtained from brp_newrpt()
        size : size to be allocated (TODO: units?) [int]
    Returns:
        Resized rpt provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> rpt = brp.brp_newrpt()
    >>> rpt = brp.brp_allocrpt(rpt, 999)
    >>> rpt = brp.brp_resizerpt(rpt, 2222)

    See Also:
        brp_newrpt
        brp_allocrpt
        brp_freerpt
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_resizerpt(prpt, size)
    if RPT_NSIZE(prpt) != size:
        raise BurpcError('Problem resizing report')
    return rpt


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
        Allocated blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> blk = brp.brp_allocblk(blk, 5, 1, 1)

    See Also:
        brp_newblk
        brp_resizeblk
        brp_freeblk
     """
    return brp_resizeblk(blk, nele, nval, nt)


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
        Resized blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> blk = brp.brp_allocblk(blk, 5, 1, 1)
    >>> blk = brp.brp_resizeblk(blk, 9, 1, 1)

    See Also:
        brp_newblk
        brp_allocblk
        brp_freeblk
    """
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if nele is None: nele = pblk[0].max_nele
    if nval is None: nval = pblk[0].max_nval
    if nt   is None: nt   = pblk[0].max_nt
    nele, nval, nt = int(max(1, nele)), int(max(1, nval)), int(max(1, nt))
    if pblk[0].max_nele == 0:
        _bp.c_brp_allocblk(pblk, nele, nval, nt)
    else:
        _bp.c_brp_resizeblk(pblk, nele, nval, nt)
    if ((BLK_NELE(pblk), BLK_NVAL(pblk), BLK_NT(pblk))
        != (nele, nval, nt)):
        raise BurpcError('Problem allocating/resizing block')
    return blk

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
        Cleared rpt provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_newrpt
        brp_resetrpthdr
        brp_putrpthdr
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_clrrpt(prpt)
    return rpt


def brp_clrblkv(blk, val=None):
    """
    Init block data with provided value

    brp_clrblkv(blk, val)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        val : value used to init the block data (float)
    Return:
        Cleared blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblk
        brp_resetblkhdr
        brp_newblk
    """
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if val is None:
        _bp.c_brp_clrblk(pblk)
    else:
        _bp.c_brp_clrblkv(pblk, val)
    return blk


def brp_clrblk(blk):
    """
    Init block data with missing value

    brp_clrblkv(blk, val)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
        val : value used to init the block data (float)
    Return:
        Cleared blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblkv
        brp_resetblkhdr
        brp_newblk
        brp_msngval
    """
    return brp_clrblkv(blk)


def brp_resetrpthdr(rpt):
    """
    Reset report header info

    brp_resetrpthdr(rpt)

    Args:
        rpt : pointer to BURP_RPT structure [ctypes.POINTER(BURP_RPT)]
              as obtained from brp_newrpt()
    Return:
        Reseted rpt provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrrpt
        brp_newrpt
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    _bp.c_brp_resetrpthdr(prpt)
    return rpt


def brp_resetblkhdr(blk):
    """
    Reset report header info

    brp_resetblkhdr(blk)

    Args:
        blk : pointer to BURP_BLK structure [ctypes.POINTER(BURP_BLK)]
              as obtained from brp_newblk()
    Return:
        Reseted blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args

    See Also:
        brp_clrblk
        brp_newblk
    """
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    _bp.c_brp_resetblkhdr(pblk)
    return blk

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
        Encoded blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> blk = brp.brp_allocblk(blk, 1,1,1)
    >>> blk = brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> blk = brp.brp_encodeblk(blk)
    >>> print('# '+repr(brp.BLK_LSTELE(blk, 0)))
    # 2564

    See Also:
        brp_newblk
        brp_allocblk
        brp_clrblk
        brp_convertblk
        rpnpy.burpc.proto
     """
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_encodeblk(pblk) < 0:
        raise BurpcError('Problem in c_brp_encodeblk')
    return blk


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
        Converted blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> blk = brp.brp_allocblk(blk, 1,1,1)
    >>> blk = brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> blk = brp.brp_encodeblk(blk)
    >>> brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
    >>> blk = brp.brp_convertblk(blk, brp.BRP_MKSA_to_BUFR)
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
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_convertblk(pblk, mode) < 0:
        raise BurpcError('Problem in c_brp_convertblk')
    return blk


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
        Converted blk provided as input [ctypes.POINTER(BURP_RPT)]
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> blk = brp.brp_newblk()
    >>> blk = brp.brp_allocblk(blk, 1,1,1)
    >>> blk = brp.brp_clrblk(blk)
    >>> brp.BLK_SetDLSTELE(blk, 0, 10004)
    >>> blk = brp.brp_encodeblk(blk)
    >>> brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
    >>> blk = brp.brp_safe_convertblk(blk, brp.BRP_MKSA_to_BUFR)
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
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_safe_convertblk(pblk, mode) < 0:
        raise BurpcError('Problem in c_brp_convertblk')
    return blk

##---- find report and block before reading -----------------------------

def brp_findrpt(funit, rpt=None): #TODO: rpt are search keys, change name
    """
    Find, in a burp file, a report matching criterions set in rpt

    Args:
        funit : opened burp file unit number of open/obtained with brp_open()
        rpt   : (optional) search criterions
                if None, will match the first report in file
                if int,  will match next report after provided handle
                if BURP_RPT pointer,  will search for matching parameters
                                      brp_newrpt() returns a 'wildcard' rpt
    Return:
        report (w/o data), ctypes.POINTER(BURP_RPT) if a match is found
        None otherwise
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_getrpt
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    if not rpt:
        rpt = brp_newrpt()
        RPT_SetHANDLE(rpt, 0)
    elif isinstance(rpt, _integer_types):
        handle = rpt
        rpt = brp_newrpt()
        RPT_SetHANDLE(rpt, handle)
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_findrpt(funit, prpt) >= 0:
        return rpt
    return None


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    Find, in burp report rpt, a block matching criterions set in blk

    Args:
        blk   : search criterions
                if None, will match the first block in rpt (equivalent to 0)
                if int,  will match next block after provided bkno
                if BURP_BLK pointer,  will search for matching parameters
                                      brp_newblk() returns a 'wildcard' blk
        rpt   : report to look into [ctypes.POINTER(BURP_RPT)]
                rpt must have been previously read with brp_getrpt()
    Return:
        block (w/o data), ctypes.POINTER(BURP_BLK) if a match is found
        None otherwise
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_getblk
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not blk:
        blk = brp_newblk()
        BLK_SetBKNO(blk, 0)
    elif isinstance(blk, _integer_types):
        bkno = blk
        blk = brp_newblk()
        BLK_SetBKNO(blk, bkno)
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_findblk(pblk, prpt) >= 0:
        return blk
    return None


def brp_searchdlste(code, blk):
    """
    Find elements matching code in block

    Args:
        code  : burp code to search for in block
        blk   : block to look into [ctypes.POINTER(BURP_BLK)]
                blk must have been previously read with brp_getblk()
    Return:
        TODO:
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_getblk
    """
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    return c_brp_searchdlste(code, pblk)

##---- read in data -----------------------------------------------------

def brp_getrpt(funit, handle=0, rpt=None):
    """
    Read burp report at provided handle from file

    rpt = brp_getrpt(funit, handle, rpt)

    Args:
        funit  : opened burp file unit number of open/obtained with brp_open()
        handle : handle of the report to read
                 handle is obtained from brp_findrpt()
        rpt    : (optional) report to put data into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
    Return:
        report (with data), ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_findrpt
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    if rpt is None:
        if isinstance(handle, _ct.POINTER(_bp.BURP_RPT)):
            rpt = handle
            handle = RPT_HANDLE(rpt)
        elif isinstance(handle, _bo.BurpcRpt):
            rpt = handle
            handle = rpt.handle
        elif isinstance(handle, _integer_types):
            rpt = brp_newrpt()
        else:
            raise TypeError('Need to provide a valid handle')
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_getrpt(funit, handle, prpt) < 0:
        raise BurpcError('Problem in c_brp_getrpt')
    return rpt


def brp_getblk(bkno, blk=None, rpt=None):
    """
    Excract and decode block from burp report at specified block number

    blk = brp_getblk(bkno, blk, rpt)

    Args:
        bkno  : block number to get [1:nblk] (int)
        blk   : (optional) block to put data into (ctypes.POINTER(BURP_BLK))
                empty blk can be obtained from brp_newblk()
        rpt   : report to look into (ctypes.POINTER(BURP_RPT))
                rpt must have been previously read with brp_getrpt()
    Return:
        block (with data), ctypes.POINTER(BURP_BLK)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_safe_getblk
    """
    if rpt is None:
        if isinstance(blk, _bo.BurpcRpt):
            rpt = blk
            blk = None
        else:
            raise TypeError('Provided rpt must be of a pointer to a BURP_RPT')
    if blk is None:
        blk = brp_newblk()
    elif isinstance(blk, _bo.BurpcBlk):
        blk.reset_arrays()
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_getblk(bkno, pblk, prpt) < 0:
        raise BurpcError('Problem in c_brp_getblk')
    return blk


def brp_safe_getblk(bkno, blk=None, rpt=None):
    """
    Excract and decode block from burp report at specified block number

    blk = brp_safe_getblk(bkno, blk, rpt)

    Args:
        bkno  : block number to get [1:nblk] (int)
        blk   : (optional) block to put data into (ctypes.POINTER(BURP_BLK))
                empty blk can be obtained from brp_newblk()
        rpt   : report to look into (ctypes.POINTER(BURP_RPT))
                rpt must have been previously read with brp_getrpt()
    Return:
        block (with data), ctypes.POINTER(BURP_BLK)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_getblk
    """
    if rpt is None:
        if isinstance(blk, _bo.BurpcRpt):
            rpt = blk
            blk = None
        else:
            raise TypeError('Provided rpt must be of a pointer to a BURP_RPT')
    if blk is None:
        blk = brp_newblk()
    elif isinstance(blk, _bo.BurpcBlk):
        blk.reset_arrays()
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_safe_getblk(bkno, pblk, prpt) < 0:
        raise BurpcError('Problem in c_brp_safe_getblk')
    return blk


def brp_readblk(bkno, blk=None, rpt=None, cvt=False):
    """
    Excract and optionally decode block from burp report at specified block number

    blk = brp_readblk(bkno, blk, rpt, cvt)

    Args:
        bkno  : block number to get [1:nblk] (int)
        blk   : (optional) block to put data into (ctypes.POINTER(BURP_BLK))
                empty blk can be obtained from brp_newblk()
        rpt   : report to look into (ctypes.POINTER(BURP_RPT))
                rpt must have been previously read with brp_getrpt()
        cvt   : (optional) Decode block data if True (default: False)
    Return:
        block (with data), ctypes.POINTER(BURP_BLK)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_getblk
        brp_safe_getblk
    """
    if rpt is None:
        if isinstance(blk, _bo.BurpcRpt):
            rpt = blk
            blk = None
        else:
            raise TypeError('Provided rpt must be of a pointer to a BURP_RPT')
    if blk is None:
        blk = brp_newblk()
    elif isinstance(blk, _bo.BurpcBlk):
        blk.reset_arrays()
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    cvt = 1 if cvt else 0
    if _bp.c_brp_readblk(bkno, pblk, prpt, cvt) < 0:
        raise BurpcError('Problem in c_brp_readblk')
    return blk

##---- read in header ---------------------------------------------------

def brp_rdrpthdr(handle=0, rpt=None):
    """
    Read report header at provided handle

    rpt = brp_rdrpthdr(handle, rpt)

    Args:
        handle : handle of the report to read
                 handle is obtained from brp_findrpt()
        rpt    : (optional) report to put params into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
    Return:
        report (w/o data), ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_findrpt
        brp_getrpt
    """
    if rpt is None:
        if isinstance(handle, _ct.POINTER(_bp.BURP_RPT)):
            rpt = handle
            handle = RPT_HANDLE(rpt)
        elif isinstance(handle, _bo.BurpcRpt):
            rpt = handle
            handle = rpt.handle
        elif isinstance(handle, _integer_types):
            rpt = brp_newrpt()
        else:
            raise TypeError('Need to provide a valid handle')
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_rdrpthdr(handle, prpt) < 0:
        raise BurpcError('Problem in c_brp_rdrpthdr')
    return rpt


def brp_rdblkhdr(bkno, blk=None, rpt=None):
    """
    Extract block header parameters from burp report at specified block number

    blk = brp_rdblkhdr(bkno, blk, rpt)

    Args:
        bkno  : block number to get [1:nblk] (int)
        blk   : (optional) block to put data into (ctypes.POINTER(BURP_BLK))
                empty blk can be obtained from brp_newblk()
        rpt   : report to look into (ctypes.POINTER(BURP_RPT))
                rpt must have been previously read with brp_getrpt()
    Return:
        block (with data), ctypes.POINTER(BURP_BLK)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_getblk
        brp_safe_getblk
        brp_readblk
    """
    if rpt is None:
        if isinstance(blk, _bo.BurpcRpt):
            rpt = blk
            blk = None
        else:
            raise TypeError('Provided rpt must be of a pointer to a BURP_RPT')
    if blk is None:
        blk = brp_newblk()
    elif isinstance(blk, _bo.BurpcBlk):
        blk.reset_arrays()
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_rdblkhdr(bkno, pblk, prpt) < 0:
        raise BurpcError('Problem in c_brp_rdblkhdr')
    return blk

##---- writing ----------------------------------------------------------

def brp_initrpthdr(funit, rpt):
    """
    Prepare a burp report for writing

    rpt = brp_initrpthdr(funit, rpt)

    Args:
        funit  : opened burp file unit number of open/obtained with brp_open()
        rpt    : report to put data into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
    Return:
        burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_putrpthdr
        brp_updrpthdr
        brp_writerpt
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_initrpthdr(funit, prpt) < 0:
        raise BurpcError('Problem in c_brp_initrpthdr')
    return rpt


def brp_putrpthdr(funit, rpt):
    """
    Prepare a burp report for writing (alias of brp_initrpthdr)

    rpt = brp_putrpthdr(funit, rpt)

    Args:
        funit  : opened burp file unit number of open/obtained with brp_open()
        rpt    : report to put data into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
    Return:
        burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_initrpthdr
        brp_updrpthdr
        brp_writerpt
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_putrpthdr(funit, prpt) < 0:
        raise BurpcError('Problem in c_brp_putrpthdr')
    return rpt


def brp_updrpthdr(funit, rpt):
    """
    Modify only the header of a burp report

    rpt = brp_updrpthdr(funit, rpt)

    Args:
        funit  : opened burp file unit number of open/obtained with brp_open()
        rpt    : report to put data into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
    Return:
        burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_initrpthdr
        brp_putrpthdr
        brp_writerpt
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_updrpthdr(funit, prpt) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')
    return rpt


def brp_writerpt(funit, rpt, where=_bc.BRP_END_BURP_FILE):
    """
    Write out report to a file

    rpt = brp_writerpt(funit, rpt, where)

    Args:
        funit  : opened burp file unit number of open/obtained with brp_open()
        rpt    : report to put data into (ctypes.POINTER(BURP_RPT))
                 empty rpt can be obtained from brp_newrpt()
        where  : (optional) handle where to write the burp report
                 (default: BRP_END_BURP_FILE)
    Return:
        burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_initrpthdr
        brp_putrpthdr
        brp_updrpthdr
    """
    funit = funit.funit if isinstance(funit, _bo.BurpcFile) else funit
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_writerpt(funit, prpt, where) < 0:
        raise BurpcError('Problem in brp_writerpt')
    return rpt


def brp_putblk(rpt, blk):
    """
    Add a new block into a burp report

    rpt = brp_putblk(rpt, blk)

    Args:
        rpt : burp report to be updated (ctypes.POINTER(BURP_RPT))
              empty rpt can be obtained from brp_newrpt()
        blk : block data and meta to put in report (ctypes.POINTER(BURP_BLK))
              empty blk can be obtained from brp_newblk()
    Return:
        updated burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newrpt
        brp_newblk
        brp_initrpthdr
        brp_putrpthdr
        brp_updrpthdr
        brp_writerpt
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_putblk(prpt, pblk) < 0:
        raise BurpcError('Problem in c_brp_putblk')
    return rpt

##---- utilisites -------------------------------------------------------

def brp_copyrpthdr(dst_rpt, src_rpt=None):
    """
    Copy burp report header

    dst_rpt = brp_copyrpthdr(src_rpt)
    dst_rpt = brp_copyrpthdr(dst_rpt, src_rpt)

    Args:
        dst_rpt : (optional) report to be updated (ctypes.POINTER(BURP_RPT))
                  empty rpt can be obtained from brp_newrpt()
        src_rpt : report to copy from (ctypes.POINTER(BURP_RPT))
                  rpt must have been previously read with brp_getrpt()
    Return:
        updated burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_copyrpt
    """
    if src_rpt is None:
        src_rpt = dst_rpt
        dst_rpt = brp_newrpt()
    pdst_rpt = _RPTPTR(dst_rpt)
    if not isinstance(pdst_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(dst_rpt)))
    psrc_rpt = _RPTPTR(src_rpt)
    if not isinstance(psrc_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(src_rpt)))
    _bp.c_brp_copyrpthdr(pdst_rpt, psrc_rpt)
    return dst_rpt


def brp_copyrpt(dst_rpt, src_rpt=None):
    """
    Copy burp report header and data

    dst_rpt = brp_copyrpt(src_rpt)
    dst_rpt = brp_copyrpt(dst_rpt, src_rpt)

    Args:
        dst_rpt : (optional) burp to be updated (ctypes.POINTER(BURP_RPT))
                  empty rpt can be obtained from brp_newrpt()
        src_rpt : report to copy from (ctypes.POINTER(BURP_RPT))
                  rpt must have been previously read with brp_getrpt()
    Return:
        updated burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_copyrpthdr
    """
    if src_rpt is None:
        src_rpt = dst_rpt
        dst_rpt = brp_newrpt()
    pdst_rpt = _RPTPTR(dst_rpt)
    if not isinstance(pdst_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(dst_rpt)))
    psrc_rpt = _RPTPTR(src_rpt)
    if not isinstance(psrc_rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(src_rpt)))
    _bp.c_brp_copyrpt(pdst_rpt, psrc_rpt)
    return dst_rpt


def brp_copyblk(dst_blk, src_blk=None):
    """
    Duplicate a burp block

    dst_blk = brp_copyblk(src_blk)
    dst_blk = brp_copyblk(dst_blk, src_blk)

    Args:
        dst_blk : (optional) block to be updated (ctypes.POINTER(BURP_BLK))
                  empty blk can be obtained from brp_newblk()
        src_blk : block to copy from (ctypes.POINTER(BURP_BLK))
                  blk must have been previously read with brp_getblk()
    Return:
        updated burp block, ctypes.POINTER(BURP_BLK)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_copyrpthdr
        brp_findblk
        brp_getblk
    """
    if src_blk is None:
        src_blk = dst_blk
        dst_blk = brp_newblk()
    pdst_blk = _BLKPTR(dst_blk)
    if not isinstance(pdst_blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(dst_blk)))
    psrc_blk = _BLKPTR(src_blk)
    if not isinstance(psrc_blk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(src_blk)))
    _bp.c_brp_copyblk(pdst_blk, psrc_blk)
    return dst_blk

##---- deleting ---------------------------------------------------------

def brp_delrpt(rpt):
    """
    Delete burp report

    brp_delrpt(rpt)

    Args:
        rpt : report to be deleted (ctypes.POINTER(BURP_RPT))
    Return:
        None
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_free
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_delrpt(prpt) <0:
        raise BurpcError('Problem in c_brp_delrpt')


def brp_delblk(rpt, blk):
    """
    Delete block from burp report

    rpt = brp_delblk(rpt, blk)

    Args:
        rpt : burp report to be updated (ctypes.POINTER(BURP_RPT))
        blk : block data and meta to deleted (ctypes.POINTER(BURP_BLK))
    Return:
        updated burp report, ctypes.POINTER(BURP_RPT)
    Raises:
        TypeError on not supported types or args
        BurpcError on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> #TODO

    See Also:
        brp_open
        brp_newblk
        brp_findrpt
        brp_getrpt
        brp_findblk
        brp_getblk
        brp_putblk
        brp_delrpt
    """
    prpt = _RPTPTR(rpt)
    if not isinstance(prpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    pblk = _BLKPTR(blk)
    if not isinstance(pblk, _ct.POINTER(_bp.BURP_BLK)):
        raise TypeError('Cannot use blk or type={}'+str(type(blk)))
    if _bp.c_brp_delblk(prpt, pblk) <0:
        raise BurpcError('Problem in c_brp_delblk')
    return rpt

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
