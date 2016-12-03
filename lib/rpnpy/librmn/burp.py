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
from rpnpy.librmn import burp_const as _rbc
from rpnpy.librmn import base as _rb
from rpnpy.librmn import RMNError

_C_MKSTR = _ct.create_string_buffer
_C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer'

_ERR_INV_DATYP = 16

class BurpError(RMNError):
    """
    General librmn.burp module error/exception

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    >>>    #... a burpfile operation ...
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

    def __init__(self, fnc_name='', ier=0):
        istat = abs(ier)
        self.msg = "Error occured while executing; {0}".format(fnc_name)
        if istat in BurpError.error_codes.keys():
            self.msg += " - {0} (ISTAT={1})" \
                .format(BurpError.error_codes[istat], istat)

    def __str__(self):
        return repr(self.msg)


def isBURP(filename):
    """
    Return True if file is of BURP type

    isburp = isBURP(filename)
    
    Args:
        filename : path/name of the file to examine (str)
    Returns:
        True or False
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> isburp = rmn.isBURP(filename)

    See Also:
       rpnpy.librmn.base.wkoffit
    """
    if not (type(filename) == str):
        raise TypeError("isBURP: Expecting arg of type str, Got {0}"\
                        .format(type(filename)))
    if filename.strip() == '':
        raise ValueError("isBURP: must provide a valid filename")
    return _rb.wkoffit(filename) in \
        (_rc.WKOFFIT_TYPE_LIST['BURP'], )
    #TODO: should we also accept 'BUFR', 'BLOK'... ?


def burp_open(filename, filemode=_rbc.BURP_MODE_READ):
    """
    Open the specified burp file
    Shortcut for fnom+mrfopn

    iunit = burp_open(filename)
    iunit = burp_open(filename, FST_RO)

    Args:
        paths    : path/name of the file to open
                   if paths is a list, open+link all files
                   if path is a dir, open+link all fst files in dir
        filemode : a string with the desired filemode (see librmn doc)
                   or one of these constants:
                   BURP_MODE_READ, BURP_MODE_CREATE, BURP_MODE_APPEND
    Returns:
        int, file unit number associated with provided path
        None in ReadOnly mode if no burp file was found in path
    Raises:
        TypeError  on wrong input arg types    
        ValueError on invalid input arg value
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit1 = rmn.burp_open(filename)
    >>> funit2 = rmn.burp_open('newfile.brp', rmn.BURP_MODE_CREATE)
    >>> #...
    >>> rmn.burp_close(funit1)
    >>> rmn.burp_close(funit2)
    >>> os.unlink('newfile.fst')  # Remove test file
    
    See Also:
       mrfopn
       mrfcls
       burp_close
       rpnpy.librmn.base.fnom
       rpnpy.librmn.burp_const
       BurpError
    """
    if not isinstance(filename, str):
        raise TypeError("burp_open: Expecting arg of type str, Got {0}"\
                        .format(type(filename)))
    if filename.strip() == '':
        raise ValueError("burp_open: must provide a valid filename")
    if filemode == _rbc.BURP_MODE_CREATE:
        fstmode = _rc.FST_RW
    elif filemode == _rbc.BURP_MODE_APPEND:
        fstmode = _rc.FST_RW_OLD
    elif filemode == _rbc.BURP_MODE_READ:
        fstmode = _rc.FST_RO
    else:
        raise ValueError('filemode should be one of BURP_MODE_READ, BURP_MODE_CREATE, BURP_MODE_APPEND')
    if filemode != _rbc.BURP_MODE_CREATE:
        if not isBURP(filename):
            raise BurpError('Not a burp file: {0}'.format(filename))
    funit = _rb.fnom(filename, fstmode)
    if not funit:
        raise BurpError('Problem associating a unit with the file: {0}'
                        .format(filename))
    nrep = mrfopn(funit, filemode)
    return funit


def burp_close(iunit):
    """
    Close the burp file associated with provided file unit number.
    Shortcut for fclos+mrfcls
    
    Args:
        iunit    : unit number associated to the file
                   obtained with fnom or burp_open
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        FSTDError  on any other error
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit1 = rmn.burp_open(filename)
    >>> #...
    >>> rmn.burp_close(funit1)

    See Also:
       mrfopn
       mrfcls
       burp_open
       rpnpy.librmn.base.fnom
       rpnpy.librmn.base.fclos
       rpnpy.librmn.burp_const
       BurpError
    """
    mrfcls(iunit)
    _rb.fclos(iunit)


def mrfopt(optName, optValue=None):
    """
    Set/Get BURP file options

    mrfopt(optName, optValue)

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
    >>> # Restrict to the minimum the number of messages printed by librmn
    >>> rmn.mrfopt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)

    See Also:
        rpnpy.librmn.burp_const
    """
    if not optName in (_rbc.BURPOP_MISSING, _rbc.BURPOP_MSGLVL):
        raise KeyError("mrfopt: uknown optName: {}".format(optName))
    
    if optValue is None:
        if optName == _rbc.BURPOP_MSGLVL:
            optValue = _C_MKSTR(' '*_rbc.BURP_OPTC_STRLEN)
            istat = _rp.c_mrfgoc(optName, optValue)
        else:
            optValue = _ct.c_float(0.)
            istat = _rp.c_mrfgor(optName, _ct.byref(optValue))
        if istat != 0:
            raise BurpError('c_mrfgocr:'+optName, istat)
        return optValue.value
        
    if isinstance(optValue, str):
        istat = _rp.c_mrfopc(optName, optValue)
        if istat != 0:
            raise BurpError('c_mrfopc:{}={}'.format(optName,optValue), istat)
    elif isinstance(optValue, float):
        #TODO: check c_mrfopr, not working, set value to 0. apparently
        istat = _rp.c_mrfopr(optName, optValue)
        if istat != 0:
            raise BurpError('c_mrfopr:{}={}'.format(optName,optValue), istat)
    else:
        raise TypeError("mrfopt: cannot set optValue of type: {0} {1}"\
                        .format(type(optValue), repr(optValue)))
    return optValue


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
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)
    >>> os.unlink('myburpfile.brp')  # Remove test file

    See Also:
        mrfcls
        burp_open
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
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
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)
    >>> os.unlink('myburpfile.brp')  # Remove test file

    See Also:
        mrfopn
        burp_open
        burp_close
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    istat = _rp.c_mrfcls(funit)
    if istat != 0:
        raise BurpError('c_mrfcls', istat)
    return


def mrfvoi(funit):
    """
    Print the content of a previously opened BURP file

    mrfvoi(funit)

    Args:
        funit : file unit number (int)
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RO)
    >>> nrep  = rmn.mrfvoi(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
    """
    istat = _rp.c_mrfvoi(funit)
    if istat < 0:
        raise BurpError('c_mrfvoi', istat)
    return


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
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RO)
    >>> nrep  = rmn.mrfnbr(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
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
    >>> funit  = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfmxl(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfbfl
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
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
    >>> funit  = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfbfl(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfmxl
        mrfopn
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    nmax = _rp.c_mrfmxl(funit)
    if nmax < 0:
        raise BurpError('c_mrfmxl', nmax)
    return nmax + 10  ## +10 comes from librmn source code


##TODO: mrfrwd
##TODO: mrfapp

def mrfloc(funit, handle=0, stnid='*********', idtyp=-1, lat=-1, lon=-1,
           date=-1, time=-1, sup=None):
    """
    Locate position of report in file.

    handle = mrfloc(funit, handle, stnid, idtyp, lat, lon, date, time, sup)
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
        time   : Observation time/hour (-1 as wildcard)
        sup    : Additional search keys (array of int)
    Returns:
        int, report handle
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rmn.burp_close(funit)

    See Also:
        mrfget
        mrfput
        burp_open
        burp_close
        rpnpy.librmn.burp_const
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
        raise TypeError('sup is not supported in this version of librmn, ' +
                        'should prived None or empty list')
    handle = _rp.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, time,
                          sup, nsup)
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
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> #TODO: describe what tools can be used to get info from buf
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfput
        mrbhdr
        mrbprm
        mrbxtr
        burp_open
        burp_close
        rpnpy.librmn.burp_const
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

#TODO: review
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
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> nbuf  = 1024 ## Set nbuf to appropriate size
    >>> buf   =_np.empty((nbuf,), dtype=_np.int32)
    >>> ## Fill buf with relevant info
    >>> #TODO: describe what tools can be used to fill buf
    >>> handle = 0
    >>> rmn.mrfput(funit, handle, buf)
    >>> rmn.mrfcls(funit)
    >>> rmn.fclos(funit)

    See Also:
        mrfget
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    istat = _rp.c_mrfput(funit, handle, buf)
    if istat != 0:
        raise BurpError('c_mrfput', istat)
    return


def mrbhdr(buf):
    """
    Returns report header information.

    params = mrbhdr(buf)
    
    Args:
        buf : Report data (array)
    Returns:
        {
            'time'  : (int)   Observation time/hour (HHMM)
            'timehh': (int)   Observation time hour part (HH)
            'timemm': (int)   Observation time minutes part (MM)
            'flgs'  : (int)   Global flags
                              (24 bits, Bit 0 is the right most bit of the word)
                              See BURP_FLAGS_IDX_NAME for Bits/flags desc.
            'flgsl' : (list)  Global flags as a list of bool
                              See BURP_FLAGS_IDX for Bits/flags desc.
            'flgs_desc' :     List of set falg desc (str)
            'stnid' : (str)   Station ID
                              If it is a surface station, STNID = WMO number.
                              The name is aligned at left and filled with
                              spaces. In the case of regrouped data,
                              STNID contains blanks.
            'idtyp' : (int)   Report Type
            'idtyp_desc' :    Report Type description (str)
            'ilat'  : (int)   Station latitude (1/100 of degrees)
                              with respect to the south pole. (0 to 1800)
                              (100*(latitude+90)) of a station or the
                              lower left corner of a box.
                              #TODO: check ilat desc 
            'lat'   : (float) Station latitude (degrees)
            'ilon'  : (int)   Station longitude (1/100 of degrees)
                              (0 to 36000) of a station or lower left corner
                              of a box.
            'lon'   : (float) Station longitude (degrees)
            'idx'   : (int)   Width of a box for regrouped data
                              (delta lon, 1/10 of degrees)
            'dx'    : (float) Width of a box for regrouped data (degrees)
            'idy'   : (int)   Height of a box for regrouped data
                              (delta lat, 1/10 of degrees)
            'dy'    : (float) Height of a box for regrouped data (degrees)
            'ielev' : (int)   Station altitude (metres + 400.) (0 to 8191)
            'elev'  : (float) Station altitude (metres)
            'drnd'  : (int)   Reception delay: difference between the
                              reception time at CMC and the time of observation
                              (TIME). For the regrouped data, DRND indicates
                              the amount of data. DRND = 0 in other cases.
            'date'  : (int)   Report valid date (YYYYMMDD)
            'dateyy': (int)   Report valid date (YYYY)
            'datemm': (int)   Report valid date (MM)
            'datedd': (int)   Report valid date (DD)
            'oars'  : (int)   Reserved for the Objective Analysis. (0-->65535)
            'runn'  : (int)   Operational pass identification.
                              #TODO: provide decoded runn?
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
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> #TODO: describe what tools can be used to decode info from params
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbprm
        mrbxtr
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    itime  = _ct.c_int()
    iflgs  = _ct.c_int()
    stnids = _C_MKSTR(' '*_rbc.BURP_STNID_STRLEN)
    idtyp  = _ct.c_int()
    ilat   = _ct.c_int()
    ilon   = _ct.c_int()
    idx    = _ct.c_int()
    idy    = _ct.c_int()
    ialt   = _ct.c_int()
    idelay = _ct.c_int()
    idate  = _ct.c_int()
    irs    = _ct.c_int()
    irunn  = _ct.c_int()
    nblk   = _ct.c_int()
    nsup   = _ct.c_int(0)
    nxaux  = _ct.c_int(0)
    sup    = _np.empty((1, ), dtype=_np.int32)
    xaux   = _np.empty((1, ), dtype=_np.int32)
    istat  = _rp.c_mrbhdr(buf, _ct.byref(itime), _ct.byref(iflgs),
                          stnids, _ct.byref(idtyp),
                          _ct.byref(ilat), _ct.byref(ilon),
                          _ct.byref(idx), _ct.byref(idy), _ct.byref(ialt), 
                          _ct.byref(idelay), _ct.byref(idate), _ct.byref(irs),
                          _ct.byref(irunn), _ct.byref(nblk), 
                          sup, nsup, xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbhdr', istat)
    flgsl = [i == '1' for i in list("{0:024b}".format(iflgs.value))]
    flgsl.reverse()
    try:
        idtyp_desc = _rbc.BURP_IDTYP_DESC[str(idtyp.value)]
    except KeyError:
        idtyp_desc = ''
    flgs_desc = "".join([_rbc.BURP_FLAGS_IDX_NAME[i]+", " if flgsl[i] else "" for i in range(len(flgsl))])
    if flgs_desc:
        if flgs_desc[-2:] == ', ':
            flgs_desc = flgs_desc[:-2]
    return {
            'time'  : itime.value,
            'timehh': itime.value // 100,
            'timemm': itime.value % 100,
            'flgs'  : iflgs.value,
            'flgsl' : flgsl,
            'flgs_desc' : flgs_desc,
            'stnid' : stnids.value,
            'idtyp' : idtyp.value,
            'idtyp_desc' : idtyp_desc,
            'ilat'  : ilat.value,
            'lat'   : (float(ilat.value)/100.) - 90.,
            'ilon'  : ilon.value,
            'lon'   : float(ilon.value)/100.,
            'idx'   : idx.value,
            'dx'    : float(idx.value)/10.,
            'idy'   : idy.value,
            'dy'    : float(idy.value)/10.,
            'ielev' : ialt.value,
            'elev'  : float(ialt.value) - 400.,
            'drnd'  : idelay.value,
            'date'  : idate.value,
            'dateyy': idate.value // 10000,
            'datemm': (idate.value % 10000) // 100,
            'datedd': (idate.value % 10000) % 100,
            'oars'  : irs.value,
            'runn'  : irunn.value,
            'nblk'  : nblk.value,
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
        }


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
                            #TODO: provide decoded bfam?
            'bdesc' : (int) Block descriptor. (0-2047)
                            #TODO: provide decoded bdesc?
            'btyp'  : (int) Block type (0-2047), made from 3 components:
                            BKNAT: kind component of Block type
                            BKTYP: Data-type component of Block type
                            BKSTP: Sub data-type component of Block type
                            #TODO: extract BKNAT, BKTYP, BKSTP
                            #TODO: provide decoded BKNAT, BKTYP, BKSTP?
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
                            See rpnpy.librmn.burp_const BURP_DATYP_LIST
                                                    and BURP_DATYP2NUMPY_LIST
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
            'datyp_name' : (str) Data type name
        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg values
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkparams = rmn.mrbprm(buf, iblk+1)
    >>> #TODO: describe what tools can be used to decode info from blkparams
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbxtr
        burp_open
        burp_close
        rpnpy.librmn.burp_const
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
            'datyp' : datyp.value,
            'datyp_name' : _rbc.BURP_DATYP_NAMES[datyp.value]
        }


def mrbxtr(buf, bkno, lstele=None, tblval=None, dtype=_np.int32):
    """
    Extract block of data from the buffer.
    
    blockdata = mrbxtr(buf, bkno)
    blockdata = mrbxtr(buf, bkno, lstele, tblval)

    Args:
        buf  : Report data  (array)
        bkno : block number (int > 0)
        lstele, tblval: (optional) return data arrays
        dtype: (optional) numpy type for tblval creation, if tblval is None
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
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(buf, iblk+1)
    >>> #TODO: describe what tools can be used to decode info from blkdata
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
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
        ## if dtype != tblval.dtype:
        ##     raise TypeError('Expecting tblval of type {0}, got: {1}'
        ##                     .format(repr(dtype),repr(tblval.dtype)))
        if len(tblval.shape) != 3:
            raise TypeError('tblval should be a ndarray of rank 3')
        if tblval.shape != (nele, nval, nt):
            raise TypeError('tblval has wrong shape')
    elif tblval is None:
        tblval = _np.empty((nele, nval, nt), dtype=dtype, order='FORTRAN')
    else:
        raise TypeError('tblval should be a ndarray of rank 3')
    dtype0 = _np.int32
    if isinstance(lstele, _np.ndarray):
        if not lstele.flags['F_CONTIGUOUS']:
            raise TypeError('Provided lstele should be F_CONTIGUOUS')
        if dtype0 != lstele.dtype:
            raise TypeError('Expecting lstele of type {0}, got: {1}'
                            .format(repr(dtype0), repr(lstele.dtype)))
        if len(lstele.shape) != 1 or lstele.size != nele:
            raise TypeError('lstele should be a ndarray of rank 1 (nele)')
    elif lstele is None:
        lstele = _np.empty(nele, dtype=dtype0, order='FORTRAN')
    else:
        raise TypeError('lstele should be a ndarray of rank 1 (nele)')
    istat = _rp.c_mrbxtr(buf, bkno, lstele, tblval)
    if istat != 0:
        raise BurpError('c_mrbxtr', istat)
    bparams['lstele'] = lstele
    bparams['tblval'] = tblval
    return bparams     


def mrbdcl(lstele):
    """
    Convert CMC codes to BUFR codes.

    lstelebufr = mrbdcl(lstele)
    
    Args:
        lstele : List of element names in the report in numeric BUFR codes.
                 ##TODO: See the code repr in the FM 94 BUFR man
    Returns
        array, list of  lstele converted to BUFR codes
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata    = rmn.mrbxtr(buf, iblk+1)
    >>>     lstelebufr = rmn.mrbdcl(blkdata['lstele'])
    >>> #TODO: describe what tools can be used to decode info from blkdata
    >>> rmn.burp_close(funit)

    See Also:
        mrbcol
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbxtr
        mrbcvt
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    dtype = _np.int32  ##TODO: should this take another value?
    if isinstance(lstele, _np.ndarray):
        if not lstele.flags['F_CONTIGUOUS']:
            raise TypeError('Provided lstele should be F_CONTIGUOUS')
        if dtype != lstele.dtype:
            raise TypeError('Expecting lstele of type {0}, got: {1}'
                            .format(repr(dtype),repr(lstele.dtype)))
    elif isinstance(lstele, (tuple, list)):
        lstele = _np.array(lstele, dtype=dtype)
    elif isinstance(lstele, (int, long)):
        lstele = _np.array([lstele], dtype=dtype)
    else:
        raise TypeError('lstele should be a ndarray of rank 1')
    lstelebufr = _np.empty(lstele.size, dtype=dtype, order='FORTRAN')
    istat = _rp.c_mrbdcl(lstele, lstelebufr, lstele.size)
    if istat != 0:
        raise BurpError('c_mrbdcl', istat)
    return lstelebufr


def mrbcvt_decode(lstele, tblval=None, datyp=_rbc.BURP_DATYP_LIST['float']):  ##TODO: rval=None, nbits?
    """
    Convert table/BURF values to 'real' values.

    rval = mrbcvt_decode(lstele, tblval)
    rval = mrbcvt_decode(lstele, tblval, datyp)
    rval = mrbcvt_decode(blockdata)
    rval = mrbcvt_decode(blockdata, datyp)

    Args:
        lstele : List of element names in the report in numeric BUFR codes.
                 ##TODO: See the code repr in the FM 94 BUFR man
        tblval : BURF code values (array of int or float)
        datyp' : (optional) Data type as obtained from mrbprm (int)
                 See rpnpy.librmn.burp_const BURP_DATYP_LIST
                                         and BURP_DATYP2NUMPY_LIST
                 Default: 6 (float)
        blockdata : (dict) Block data as returned by mrbxtr,
                           must contains 2 keys: 'lstele', 'tblval'
    Returns
        array, Real values (array of float)
    Raises:
        KeyError   on missing blockdata keys
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata  = rmn.mrbxtr(buf, iblk+1)
    >>>     rval     = rmn.mrbcvt_decode(blkdata)
    >>> #TODO: describe what tools can be used to decode info from blkdata
    >>> rmn.burp_close(funit)
    #TODO: 

    See Also:
        mrbcvt_encode
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbxtr
        mrbdcl
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    if isinstance(lstele, dict):
        try:
            tblval = lstele['tblval']
            lstele = lstele['lstele']
        except:
            raise KeyError('Provided blockdata should have these 2 keys: lstele, tblval')
    if isinstance(tblval, _np.ndarray):
        if not tblval.flags['F_CONTIGUOUS']:
            raise TypeError('Provided tblval should be F_CONTIGUOUS')
        if len(tblval.shape) != 3:
            raise TypeError('Provided tblval should be en ndarray of rank 3')
    else:
        raise TypeError('Provided tblval should be an ndarray of rank 3')
    (nele, nval, nt) = tblval.shape
    dtype = _np.int32
    if isinstance(lstele, _np.ndarray):
        if not lstele.flags['F_CONTIGUOUS']:
            raise TypeError('Provided lstele should be F_CONTIGUOUS')
        if dtype != lstele.dtype:
            raise TypeError('Expecting lstele of type {0}, got: {1}'
                            .format(repr(dtype),repr(lstele.dtype)))
        if len(lstele.shape) != 1 or lstele.size != nele:
            raise TypeError('lstele should be a ndarray of rank 1 (nele)')
    else:
        raise TypeError('lstele should be a ndarray of rank 1 (nele)')
            
    ## Conversion of values from RVAL (real) to TBLVAL (integer) or the inverse
    ## depending on the MODE.
    ## If MODE=0, convert from TBLVAL to RVAL and
    ## if MODE=1, convert from RVAL to TBLVAL.
    ## When we are dealing with variables described by a code instead of a value
    ## (eg: present time, type of cloud), we don't do any conversion and the
    ## value stays in the original input array.
    ## A missing data is indicated by putting all the bits ON of the
    ## corresponding element in the array TBLVAL while for the RVAL array,
    ## we insert the attributed value to the option 'MISSING'.
    ## Note: Type 3 and 5 are processed like strings of bits thus,
    ##       the user should do the data compression himself.

    if not datyp in _rbc.BURP_DATYP_NAMES.keys():
        raise ValueError('Out of range datyp={0}'.format(datyp))

    #TODO: review c_mrbcvt
    
    ## if _rbc.BURP_DATYP_NAMES[datyp] in ('complex', 'dcomplex'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]),
    ##                     _ERR_INV_DATYP)
    ## elif _rbc.BURP_DATYP_NAMES[datyp] in ('char', 'upchar'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    ## elif _rbc.BURP_DATYP_NAMES[datyp] in ('float','double'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    ## elif _rbc.BURP_DATYP_NAMES[datyp] in ('binary',):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    ## else:  ## ('uint', 'int'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))

    dtype = _rbc.BURP_DATYP2NUMPY_LIST[datyp]
    rval   = tblval.astype(dtype)    
    istat = _rp.c_mrbcvt(lstele, tblval, rval, nele, nval, nt,
                         _rbc.MRBCVT_DECODE)
    if istat != 0:
        raise BurpError('c_mrbcvt', istat)
    
    return rval

#TODO: function for mrbxtr+cvt+dcl (and maybe hrd,prm)
def mrbxtr_dcl_cvt(buf, bkno, lstele=None, tblval=None, dtype=_np.int32):
    """
    Extract block of data from the buffer and decode its values
    
    blockdata = mrbxtr(buf, bkno)
    blockdata = mrbxtr(buf, bkno, lstele, tblval)

    Args:
        buf  : Report data  (array)
        bkno : block number (int > 0)
        lstele, tblval: (optional) return data arrays
        dtype: (optional) numpy type for tblval creation, if tblval is None
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
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> buf    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(buf)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(buf, iblk+1)
    >>> #TODO: describe what tools can be used to decode info from blkdata
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    raise BurpError('mrbxtr_dcl_cvt not yet implemented')

#TODO: mrbcvt_encode
## def mrbcvt_encode(lstele, rval): ##TODO:

#TODO: review
def mrbini(funit, buf, time, flgs, stnid, idtp, lat, lon, dx, dy, elev, drnd,
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
    istat = _rp.c_mrbini(funit, buf, time, flgs, stnid, idtp, lat, lon,  
                         dx, dy, elev, drnd, date, oars, runn, sup, nsup,
                         xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbini', istat)
    return istat


#TODO: review
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


#TODO: review
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


#TODO: review
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
