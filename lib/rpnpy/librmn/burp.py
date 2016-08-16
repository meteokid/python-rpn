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

from rpnpy.librmn import proto_burp as _rp
from rpnpy.librmn import const as _rc
from rpnpy.librmn import RMNError

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
    if type(optValue) == str:
        istat = _rp.c_mrfopc(optName, optValue)
        if istat != 0:
            raise BurpError('c_mrfopc', istat)
    elif type(optValue) == float:
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


def mrfloc(funit, handle=0, stnid='*', idtyp=-1, lati=-1, long=-1, date=-1, temps=-1, sup=None):
    """
    Locate position of report in file.

    handle = mrfloc(funit, handle, stnid, idtyp, lati, long, date, temps, sup)
    handle = mrfloc(funit, handle, stnid)

    Args:
        iun    : File unit number
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
    #TODO: should accept funit as a dict will all agrs as keys/val
    nsup = 0
    if sup is None:
        sup = _np.empty((1,), dtype=_np.int32)
    elif isinstance(sup, (list,tuple)):
        nsup = len(sup)
        if nsup == 0:
            sup = _np.empty((1,), dtype=_np.int32)
        else:
            sup = _np._np.asfortranarray(sup, dtype=_np.int32)
    elif isinstance(sup, _np.ndarray)
        nsup = sup.size
    else:
        raise TypeError('sup should be a None, list, tuple or ndarray')
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
                 only needed if buf is not provided
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_READ)
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
    if buf is None or type(buf) == int:
        nbuf = buf
        if buf is None:
            nbuf = rmn.mrfmxl(funit)
        nbuf *= 2
        buf = _np.empty((nbuf,), dtype=_np.int32)
        buf[0] = nbuf
    elif not isinstance(buf, _np.ndarray)
        raise TypeError('buf should be an ndarray')
    istat = _rp.c_mrfget(handle, buf)
    if istat != 0:
        raise BurpError('c_mrfget', istat)
    return buf


def mrfput(funit, handle, buffer):
    """
    Write a report.

    mrfput(funit, handle, buffer)

    Args:
        iun    : File unit number
        handle : Report handle
        buffer : Report data
    
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> nbuf   = 1024 ## Set nbuf to appropriate size
    >>> buffer =_np.empty((nbuf,), dtype=_np.int32)
    >>> ## Fill buf with relevant info
    >>> #TODO: describe what tools can be used to fill buf
    >>> handle = 0
    >>> mrfput(funit, handle, buffer)
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
    istat = _rp.c_mrfput(funit, handle, buffer)
    if istat != 0:
        raise BurpError('c_mrfput', istat)
    return


def mrbhdr(buf, temps, flgs, stnid, idtyp, lat, lon, dx, dy, elev, drnd, date,
           oars, runn, nblk, sup, nsup, xaux, nxaux):
    """
    Returns report header information.
    #TODO: mrbhdr(buf) returns dict with all agrs as key/val
    """
    istat = _rp.c_mrbhdr(buf, temps, flgs, stnid, idtyp, lat, lon, dx, dy, elev, drnd, date, oars, runn, nblk, sup, nsup, xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbhdr', istat)
    return istat


def mrbprm(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp):
    """
    Returns block header information.
    #TODO: mrbprm(buf, bkno) returns dict with all agrs as key/val
    """
    istat = _rp.c_mrbprm(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit,
                         bit0, datyp)
    if istat != 0:
        raise BurpError('c_mrbprm', istat)
    return istat


def mrbxtr(buf, bkno, lstele, tblval):
    """
    Extract block of data from the buffer.
    #TODO: mrbxtr(buf, bkno) returns dict with all lstele, tblval as key/val
    """
    istat = _rp.c_mrbxtr(buf, bkno, lstele, tblval)
    if istat != 0:
        raise BurpError('c_mrbxtr', istat)
    return istat


def mrbdcl(liste, dliste, nele):
    """
    Convert CMC codes to BUFR codes.
    #TODO: mrbdcl(liste)  return dliste
    """
    istat = _rp.c_mrbdcl(liste, dliste, nele)
    if istat != 0:
        raise BurpError('c_mrbdcl', istat)
    return istat


def mrbcvt(liste, tblval, rval, nele, nval, nt, mode):
    """
    Convert real values to table values.
    #TODO: mrbcvt should have only input args, output args should be returned
    """
    istat = _rp.c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
    if istat != 0:
        raise BurpError('c_mrbcvt', istat)
    return istat


def mrbini(funit, buf, temps, flgs, stnid, idtp, lati, lon, dx, dy, elev, drnd,
           date, oars, runn, sup, nsup, xaux, nxaux):
    """
    Writes report header.
    #TODO: should accept a dict for all args
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
    """
    istat = _rp.c_mrbcol(dliste, liste, nele)
    if istat != 0:
        raise BurpError('c_mrbcol', istat)
    return istat


def mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp,
           lstele, tblval):
    """
    Adds a block to a report.
    #TODO: should accept a dict for all args
    """
    istat = _rp.c_mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit,
                         bit0, datyp, lstele, tblval)
    if istat != 0:
        raise BurpError('c_mrbadd', istat)
    return buf


def mrbdel(buf, bkno):
    """
    Delete a particular block of the report.
    Args:
        buffer (array) : report data
        bkno   (int)   : number of blocks in buf
    Returns
        array, modified report data
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
