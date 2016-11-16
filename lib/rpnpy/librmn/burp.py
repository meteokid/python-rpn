#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Michael Sitwell <michael.sitwell@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Python interface for BUPR files. Contains wrappers for ctypes functions in
proto_burp and the BurpError class.
"""

import ctypes as _ct
import numpy as _np
from rpnpy.librmn import proto_burp as _rp
from rpnpy.librmn import const as _rc
from rpnpy.librmn import RMNError

_C_MKSTR = _ct.create_string_buffer
_C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer'

BURP_STNID_STRLEN = 9

class BurpError(RMNError):
    """
    General librmn.burfile module error/exception

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    >>>    #... a burfile operation ...
    >>> except(rmn.BurpError):
    >>>    pass #ignore the error
    >>> #...
    >>> raise rmn.BurpError()

    See Also:
       rpnpy.librmn.RMNError
    """

    error_codes = {
        1:  "Trivial Error",
        2:  "Informative messages for the user",
        3:  "Warning Error",
        4:  "Utmost Important Error",
        5:  "Errors that the user should know",
        6:  "Overflow error",
        7:  "Intolerable error, program crashed",
        16: "Invalid datyp",
        30: "File is not a report file",
        31: "Error in opening. Only READ, CREATE and APPEND are allowed.",
        32: "Too many supplementary keys",
        33: "Block number invalid",
        34: "Option name unknown",
        35: "FATAL error related to TABLEBURP",
        36: "TRIVIAL error related TABLEBURP",
        37: "Invalid element name",
        38: "Invalid BTP (smaller than zero)",
        39: "Incorrect NCELL dimension",
        40: "Incorrect TBLPRM dimension",
        41: "Value too big for 32 bits and DATYP=2",
        42: "File created with non-official TABLEBURP",
        43: "Bad initialization of BDESC",
        44: "Element code invalid for DATYP=(7 to 9)"
        }

    def __init__(self, fnc_name, ier):
        istat = abs(ier)
        self.msg = "Error occured while executing {0}".format(fnc_name)
        if istat in BurpError.error_codes.keys():
            self.msg += " - {0} (ISTAT={1})".format(BurpError.error_codes[istat], istat)

    def __str__(self):
        return repr(self.msg)


def mrfopt(optName, optValue):
    """
    Set BURP file options

    mrfopt(optName, optValue)

    Args:
        optName  : name of option to be set or printed
                   or one of these constants:
                   BURPOP_MISSING, BURPOP_MSGLVL
        optValue : value to be set (float or string)
                   for optName=BURPOP_MISSING:
                      a real value for missing data
                   for optName=BURPOP_MSGLVL, one of these constants:
                      BURPOP_MSG_TRIVIAL,   BURPOP_MSG_INFO,  BURPOP_MSG_WARNING,
                      BURPOP_MSG_ERROR,     BURPOP_MSG_FATAL, BURPOP_MSG_SYSTEM
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> # Restrict to the minimum the number of messages printed by librmn
    >>> rmn.mrfopt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)

    See Also:
        rpnpy.librmn.const
    """
    if isinstance(optValue, str):
        istat = _rp.c_mrfopc(optName, optValue)
        if istat != 0:
            raise BurpError('c_mrfopc', istat)
    elif isinstance(optValue, float):
        istat = _rp.c_mrfopr(optName, optValue)
        if istat != 0:
            raise BurpError('c_mrfopr', istat)
    else:
        raise TypeError("mrfopt: cannot set optValue of type: {0} {1}"\
                        .format(type(optValue), repr(optValue)))
    return


def mrfopn(funit, mode=_rc.FILE_MODE_RW):
    """
    Opens a BURP file.

    nrec = mrfopn(funit, mode)

    Args:
        funit : file unit number (int)
        mode  : file open mode, one of:
                BURP_MODE_READ, BURP_MODE_CREATE, BURP_MODE_APPEND
    Returns:
        int, number of active records in the file
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import os
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)
    >>> os.unlink('myburfile.brp')  # Remove test file

    See Also:
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    nrep = _rp.c_mrfopn(funit, mode)
    if nrep < 0:
        raise BurpError('c_mrfopn', nrep)
    return nrep


def mrfcls(funit):
    """
    Closes a BURP file.

    mrfcls(funit)

    Args:
        funit : file unit number (int)
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import os
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)
    >>> os.unlink('myburfile.brp')  # Remove test file

    See Also:
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    istat = _rp.c_mrfcls(funit)
    if istat != 0:
        raise BurpError('c_mrfcls', istat)
    return


##TODO: burpopen "a la" fstopenall
##TODO: burpclose "a la" fstcloseall


def mrfnbr(funit):
    """
    Returns number of reports in file.

    nrep = mrfnbr(funit)

    Args:
        funit : file unit number (int)
    Returns:
        int, number of active records in the file
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RO)
    >>> nrep  = rmn.mrfnbr(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    nrep = _rp.c_mrfnbr(funit)
    if nrep < 0:
        raise BurpError('c_mrfnbr', nrep)
    return nrep


def mrfmxl(funit):
    """
    Returns lenght of longest report in file.

    maxlen = mrfmxl(funit)

    Args:
        funit : file unit number (int)
    Returns:
        int, lenght of longest report in file
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfmxl(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfbfl
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    nmax = _rp.c_mrfmxl(funit)
    if nmax < 0:
        raise BurpError('c_mrfmxl', nmax)
    return nmax


def mrfbfl(funit):
    """
    Returns needed lenght to store longest report in file.

    maxlen = mrfbfl(funit)

    Args:
        funit : file unit number (int)
    Returns:
        int, needed lenght to store longest report in file
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfbfl(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfmxl
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    nmax = _rp.c_mrfmxl(funit)
    if nmax < 0:
        raise BurpError('c_mrfmxl', nmax)
    return nmax + 10  ## +10 comes from librmn source code


##TODO: mrfrwd
##TODO: mrfapp

def mrfloc(funit, handle=0, stnid='*********', idtyp=-1, lati=-1, long=-1, date=-1, temps=-1, sup=None):
    """
    Locate position of report in file.

    handle = mrfloc(funit, handle, stnid, idtyp, lati, long, date, temps, sup)
    handle = mrfloc(funit, handle, stnid)

    Args:
        funit  : File unit number
        handle : Start position for the search,
                 0 (zero) for beginning of file
        stnid  : Station ID
                 '*' for any name
        idtyp  : Report Type (-1 as wildcard)
        lat    : Station latitude (1/100 of degrees)  (-1 as wildcard)
        lon    : Station longitude (1/100 of degrees) (-1 as wildcard)
        date   : Report valid date (-1 as wildcard)
        temps  : Observation time/hour (-1 as wildcard)
        sup    : Additional search keys (array of int)
    Returns:
        int, report handle
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_READ)
    >>> handle = mrfloc(funit)
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfget
        mrfput
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    #TODO: should accept handle as a dict with all agrs as keys/val
    nsup = 0
    if sup is None:
        sup = _np.empty((1,), dtype=_np.int32)
    elif isinstance(sup, (list,tuple)):
        nsup = len(sup)
        if nsup == 0:
            sup = _np.empty((1,), dtype=_np.int32)
        else:
            sup = _np._np.asfortranarray(sup, dtype=_np.int32)
    #TODO: providing sup as ndarray of size > 0 with value zero cause a seg fault
    ## elif isinstance(sup, _np.ndarray):
    ##     nsup = sup.size
    else:
        raise TypeError('sup should be a None, list, tuple or ndarray')
    if nsup > 0: ##TODO: remove this condition when implemented/fixed
        raise TypeError('sup is not supported in this version of librmn, should prived None or empty list')
    handle = _rp.c_mrfloc(funit, handle, stnid, idtyp, lati, long, date, temps, sup, nsup)
    if handle < 0:
        raise BurpError('c_mrfloc', handle)
    return handle


def mrfget(handle, buf=None, funit=None):
    """
    Read/Put report pointed to by handle into buffer.

    buf = mrfget(handle, funit=funit)

    Args:
        handle : Report handle
        buf    : Report data buffer (ndarray of dtype int32)
                 or max report size in file (int)
        funit  : file unit number where to read report,
                 only needed if buf is None
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> nbrp   = rmn.mrfopn(funit, rmn.BURP_MODE_READ)
    >>> handle = mrfloc(funit)
    >>> buf    = mrfget(handle, funit=funit)
    >>> #TODO: describe what tools can be used to get info from buf
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfput
        mrbhdr
        mrbprm
        mrbxtr
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    if buf is None or isinstance(buf, int):
        nbuf = buf
        if buf is None:
            nbuf = mrfmxl(funit)
            ##TODO: nbuf = max(64, rmn.mrfmxl(funit))+10
        nbuf *= 2 ##TODO: should we remove this?
        buf = _np.empty((nbuf,), dtype=_np.int32)
        buf[0] = nbuf
    elif not isinstance(buf, _np.ndarray):
        raise TypeError('buf should be an ndarray')
    istat = _rp.c_mrfget(handle, buf)
    if istat != 0:
        raise BurpError('c_mrfget', istat)
    return buf


def mrfput(funit, handle, buf):
    """
    Write a report.

    mrfput(funit, handle, buf)

    Args:
        funit  : File unit number
        handle : Report handle
        buf    : Report data
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> nbuf  = 1024 ## Set nbuf to appropriate size
    >>> buf   =_np.empty((nbuf,), dtype=_np.int32)
    >>> ## Fill buf with relevant info
    >>> #TODO: describe what tools can be used to fill buf
    >>> handle = 0
    >>> mrfput(funit, handle, buf)
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfget
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    istat = _rp.c_mrfput(funit, handle, buf)
    if istat != 0:
        raise BurpError('c_mrfput', istat)
    return


##TODO: provide set/get function for flag bits with flag consts
def mrbhdr(buf):
    """
    Returns report header information.

    params = mrbhdr(buf)
    
    Args:
        buf : Report data (array)
    Returns:
        {
            'temps' : (int)   Observation time/hour (HHMM)
            'flgs'  : (int)   Global flags
                              (24 bits, Bit 0 is the right most bit of the word)
                              Bits/flags description:
                              0  assembled stations
                              1  surface wind used
                              2  message unreliable (p/t)
                              3  incorrect coordinates
                              4  message corrected
                              5  message amended
                              6  station rejected by AO
                              7  station on black list
                              8  station to evaluate
                              9  superobservation
                              10 data observed
                              11 data derived
                              12 residues
                              13 verifications
                              14 TEMP, part RADAT
                              15 TEMP, part A
                              16 TEMP, part B
                              17 TEMP, part C
                              18 TEMP, part D
                              19..23 reserved
            'stnid' : (str)   Station ID
                              If it is a surface station, STNID = WMO number.
                              The name is aligned at left and filled with
                              spaces. In the case of regrouped data,
                              STNID contains blanks.
            'idtyp' : (int)   Report Type
            'lati ' : (int)   Station latitude (1/100 of degrees)
                              with respect to the south pole. (0 to 1800)
                              (100+(latitude+90)) of a station or the
                              lower left corner of a box.
            'long'  : (int)   Station longitude (1/100 of degrees)
                              (0 to 36000) of a station or lower left corner
                              of a box.
            'dx'    : (int)   Width of a box for regrouped data
                              (delta lon, 1/10 of degrees)
            'dy'    : (int)   Height of a box for regrouped data
                              (delta lat, 1/10 of degrees)
            'elev'  : (int)   Station altitude (metres + 400.) (0 to 8191)
            'drnd'  : (int)   Reception delay: difference between the
                              reception time at CMC and the time of observation
                              (TEMPS). For the regrouped data, DRND indicates
                              the amount of data. DRND = 0 in other cases.
            'date'  : (int)   Report valid date (YYYYMMDD)
            'oars'  : (int)   Reserved for the Objective Analysis. (0-->65535)
            'runn'  : (int)   Operational pass identification.
            'nblk'  : (int)   number of blocks
            'sup'   : (array) supplementary primary keys array
                              (reserved for future expansion).
            'nsup'  : (int)   number of sup
            'xaux'  : (array) supplementary auxiliary keys array
                              (reserved for future expansion).
            'nxaux' : (int)   number of xaux
        }
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> nbrp   = rmn.mrfopn(funit, rmn.BURP_MODE_READ)
    >>> handle = mrfloc(funit)
    >>> buf    = mrfget(handle, funit=funit)
    >>> params = mrfhdr(buf)
    >>> #TODO: describe what tools can be used to decode info from params
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfput
        mrfget
        mrbprm
        mrbxtr
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    itime  = _ct.c_int()
    iflgs  = _ct.c_int()
    stnids = _C_MKSTR(' '*BURP_STNID_STRLEN)
    idburp = _ct.c_int()
    ilat = _ct.c_int()
    ilon = _ct.c_int()
    idx  = _ct.c_int()
    idy  = _ct.c_int()
    ialt = _ct.c_int()
    idelay = _ct.c_int()
    idate = _ct.c_int()
    irs   = _ct.c_int()
    irunn = _ct.c_int()
    nblk  = _ct.c_int()
    nsup  = _ct.c_int(0)
    nxaux = _ct.c_int(0)
    sup   = _np.empty((1, ), dtype=_np.int32)
    xaux  = _np.empty((1, ), dtype=_np.int32)
    istat = _rp.c_mrbhdr(buf, _ct.byref(itime), _ct.byref(iflgs),
                         stnids, _ct.byref(idburp),
                         _ct.byref(ilat), _ct.byref(ilon),
                         _ct.byref(idx), _ct.byref(idy), _ct.byref(ialt), 
                         _ct.byref(idelay), _ct.byref(idate), _ct.byref(irs),
                         _ct.byref(irunn), _ct.byref(nblk), 
                         sup, nsup, xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbhdr', istat)
    return {
            'temps' : itime.value,
            'flgs'  : iflgs.value,
            'stnid' : stnids.value,
            'idtyp' : idburp.value,
            'lati ' : ilat.value,
            'long'  : ilon.value,
            'dx'    : idx.value,
            'dy'    : idy.value,
            'elev'  : ialt.value,
            'drnd'  : idelay.value,
            'date'  : idate.value,
            'oars'  : irs.value,
            'runn'  : irunn.value,
            'nblk'  : nblk.value,
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
        }


##TODO: provide datyp consts
def mrbprm(buf, bkno):
    """
    Returns block header information.

    params = mrbprm(buf, bkno)

    Args:
        buf  : Report data  (array)
        bkno : block number (int > 0)
    Returns:
        {
            'nele'  : (int) Number of meteorological elements in a block.
                            1st dimension of the array TBLVAL(block). (0-127)
            'nval'  : (int) Number of values per element.
                            2nd dimension of TBLVAL(block). (0-255)
            'nt'    : (int) Number of groups of NELE by NVAL values in a block.
                            3rd dimension of TBLVAL(block).
                            (ie: time-series). (0- 255)
            'bfam'  : (int) Family block descriptor. (0-31)
            'bdesc' : (int) Block descriptor. (0-2047)
            'btyp'  : (int) Block type (0-2047), made from 3 components:
                            BKNAT: kind component of Block type
                            BKTYP: Data-type component of Block type
                            BKSTP: Sub data-type component of Block type
            'nbit'  : (int) Number of bits per value.
                            When we add a block, we should insure that the
                            number of bits specified is large enough to
                            represent the biggest value contained in the array
                            of values in TBLVAL.
                            The maximum number of bits is 32.
            'bit0'  : (int) Number of the first right bit from block,
                            calculated automatically by the software.  
                            (0-->2**26-1) (always a multiple of 64 minus 1)
            'datyp' : (int) Data type (for packing/unpacking).  
                            0 = string of bits (bit string)  
                            2 = unsigned integers  
                            3 = characters (NBIT must be equal to 8)  
                            4 = signed integers  
                            5 = uppercase characters (the lowercase characters
                                will be converted to uppercase during the read.
                                (NBIT must be equal to 8)  
                            6 = real*4 (ie: 32bits)  
                            7 = real*8 (ie: 64bits)  
                            8 = complex*4 (ie: 2 times 32bits)  
                            9 = complex*8 (ie: 2 times 64bits)  
                            Note: Type 3 and 5 are processed like strings of
                                  bits thus, the user should do the data
                                  compression himself.
        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg values
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        mrfmxl
        mrfloc
        mrfput
        mrfget
        mrbhdr
        mrbxtr
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    if bkno <= 0:
        raise ValueError('Provided bkno must be > 0')
    try:
        maxbkno  = mrbhdr(buf)['nblk']  ##TODO should we do this?
    except:
        maxbkno = -1
    if maxbkno > 0 and bkno > maxbkno:
        raise ValueError('Provided bkno must be < nb of block in report')
    nele = _ct.c_int()
    nval = _ct.c_int()
    nt = _ct.c_int()
    bfam = _ct.c_int()
    bdesc = _ct.c_int()
    btyp = _ct.c_int()
    nbit = _ct.c_int()
    bit0 = _ct.c_int()
    datyp = _ct.c_int()
    istat = _rp.c_mrbprm(buf, bkno, _ct.byref(nele), _ct.byref(nval),
                         _ct.byref(nt), _ct.byref(bfam), _ct.byref(bdesc),
                         _ct.byref(btyp), _ct.byref(nbit),
                         _ct.byref(bit0), _ct.byref(datyp))
    if istat != 0:
        raise BurpError('c_mrbprm', istat)
    return {
            'nele'  : nele.value,
            'nval'  : nval.value,
            'nt'    : nt.value,
            'bfam'  : bfam.value,
            'bdesc' : bdesc.value,
            'btyp'  : btyp.value,
            'nbit'  : nbit.value,
            'bit0'  : bit0.value,
            'datyp' : datyp.value
        }


def mrbxtr(buf, bkno, lstele=None, tblval=None):
    """
    Extract block of data from the buffer.
    
    blockdata = mrbxtr(buf, bkno)
    blockdata = mrbxtr(buf, bkno, lstele, tblval)

    Args:
        buf  : Report data  (array)
        bkno : block number (int > 0)
        lstele, tblval: (optional) return data arrays
    Returns
        {
            'lstele' : (array) List of element names in the report in numeric
                               BUFR codes. (Size: NELE; type: int)
                               NELE: Number of meteorological elements in a
                                     block. 1st dimension of the array
                                     TBLVAL(block). (0-127)
                               ##TODO: See the code repr in the FM 94 BUFR man
            'tblval' : (array) Block data
                               (Shape: NELE, NVAL, NT; type: int)
                               NELE: Number of meteorological elements in block
                               NVAL: Number of values per element.
                               NT  : Nb of groups of NELE x NVAL vals in block.
        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg values
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        mrfmxl
        mrfloc
        mrfput
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    dtype = _np.int32  ##TODO: should this take another value?
    if bkno <= 0:
        raise ValueError('Provided bkno must be > 0')
    try:
        maxbkno  = mrbhdr(buf)['nblk']  ##TODO should we do this?
    except:
        maxbkno = -1
    if maxbkno > 0 and bkno > maxbkno:
        raise ValueError('Provided bkno must be < nb of block in report')
    bparams = mrbprm(buf, bkno)
    (nele, nval, nt) = (bparams['nele'], bparams['nval'], bparams['nt'])
    if isinstance(tblval, _np.ndarray):
        if not tblval.flags['F_CONTIGUOUS']:
            raise TypeError('Provided tblval should be F_CONTIGUOUS')
        if dtype != tblval.dtype:
            raise TypeError('Expecting tblval of type {0}, got: {1}'.format(repr(dtype),repr(tblval.dtype)))
        if len(tblval.shape) != 3:
            raise TypeError('tblval should be a ndarray of rank 3')
        if tblval.shape != (nele, nval, nt):
            raise TypeError('tblval has wrong shape')
    elif tblval is None:
        tblval = _np.empty((nele, nval, nt), dtype=dtype, order='FORTRAN')
    else:
        raise TypeError('tblval should be a ndarray of rank 3')
    if isinstance(lstele, _np.ndarray):
        if not lstele.flags['F_CONTIGUOUS']:
            raise TypeError('Provided lstele should be F_CONTIGUOUS')
        if dtype != lstele.dtype:
            raise TypeError('Expecting lstele of type {0}, got: {1}'.format(repr(dtype),repr(lstele.dtype)))
        if len(lstele.shape) != 1 or lstele.size != nele:
            raise TypeError('lstele should be a ndarray of rank 1 (nele)')
    elif lstele is None:
        lstele = _np.empty(nele, dtype=dtype, order='FORTRAN')
    else:
        raise TypeError('lstele should be a ndarray of rank 1 (nele)')
    istat = _rp.c_mrbxtr(buf, bkno, lstele, tblval)
    if istat != 0:
        raise BurpError('c_mrbxtr', istat)
    bparams['lstele'] = lstele
    bparams['tblval'] = tblval
    return bparams     


def mrbdcl(liste, dliste, nele):
    """
    Convert CMC codes to BUFR codes.
    #TODO: mrbdcl(liste)  return dliste
    
    Args:
        #TODO: 
    Returns
        #TODO: 
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        #TODO: 
    """
    istat = _rp.c_mrbdcl(liste, dliste, nele)
    if istat != 0:
        raise BurpError('c_mrbdcl', istat)
    return istat


def mrbcvt(liste, tblval, rval, nele, nval, nt, mode):
    """
    Convert real values to table values.
    #TODO: mrbcvt should have only input args, output args should be returned
    
    Args:
        #TODO: 
    Returns
        #TODO: 
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        #TODO: 
    """
    istat = _rp.c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
    if istat != 0:
        raise BurpError('c_mrbcvt', istat)
    return istat


def mrbini(funit, buf, temps, flgs, stnid, idtp, lati, lon, dx, dy, elev, drnd,
           date, oars, runn, sup, nsup, xaux, nxaux):
    """
    Writes report header.
    #TODO: should accept a dict as input for all args
    
    Args:
        #TODO: 
    Returns
        #TODO: 
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        #TODO: 
    """
    istat = _rp.c_mrbini(funit, buf, temps, flgs, stnid, idtp, lati, lon, dx, dy,
                         elev, drnd, date, oars, runn, sup, nsup, xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbini', istat)
    return istat


def mrbcol(dliste, liste, nele):
    """
    Convert BUFR codes to CMC codes.
    #TODO: mrbcol(dliste) returns liste
    
    Args:
        #TODO: 
    Returns
        #TODO: 
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        #TODO: 
    """
    istat = _rp.c_mrbcol(dliste, liste, nele)
    if istat != 0:
        raise BurpError('c_mrbcol', istat)
    return istat


def mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp,
           lstele, tblval):
    """
    Adds a block to a report.
    #TODO: should accept a dict as input for all args
    
    Args:
        #TODO: 
        buf    (array) : vector to contain the report
        nele   (int)   : number of meteorogical elements in block
        nval   (int)   : number of data per elements
        nt     (int)   : number of group of nelenval values in block
        bfam   (int)   : block family (12 bits, bdesc no more used)
        bdesc  (int)   : kept for backward compatibility
        btyp   (int)   : block type
        nbit   (int)   : number of bit to keep per values
        datyp  (int)   : data type for packing
        lstele (array) : list of nele meteorogical elements
        tblval (array) : array of values to write (nele*nval*nt)
    Returns
        #TODO: 
        buf    (array) : vector to contain the report
        bkno   (int)   : number of blocks in buf
        bit0   (int)   : position of first bit of the report
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        #TODO: 
    """
    istat = _rp.c_mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit,
                         bit0, datyp, lstele, tblval)
    if istat != 0:
        raise BurpError('c_mrbadd', istat)
    return buf


def mrbdel(buf, bkno):
    """
    Delete a particular block of the report.

    mrbdel(buf, bkno)

    Args:
        buf  : report data (array) 
        bkno : number of blocks in buf (int)
    Returns
        array, modified report data
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    #TODO: 

    See Also:
        mrbadd
        #TODO: 
    """
    istat = _rp.c_mrbdel(buf, bkno)
    if istat != 0:
        raise BurpError('c_mrbdel', istat)
    return buf

# =========================================================================

if __name__ == "__main__":
    print("Python interface for BUPR files.")

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
