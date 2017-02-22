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

BURP: Binary Universal Report Protocol.

BURP est depuis juin 1992 le format des bases de donnees operationnelles du CMC pour tous les fichiers de type 'observations', au sens large du terme.

On peut voir BURP comme un complement du format Standard de RPN.

Un enregistrement de fichier standard RPN contient un ensemble de valeurs
d'une variable donnee, reparties sur une grille geo-referencee dont les
points sont espaces selon des specifications mathematiques.

Par opposition (ou complementarite), BURP est concu pour emmagasiner un
ensemble de variables a une 'station', ce dernier terme representant un
point geographique de latitude et longitude determinees.

Un fichier BURP est constitues de plusieurs "reports".
Chaque "report" est constitues de plusieurs "blocks".
Chaque "block"  est constitues de plusieurs "elements".
"""

import os
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

_mrbcvt_dict_full = {}
#_mrbcvt_dict_full.__doc__ = 'Parsed BUFR table B into a python dict, see mrbcvt_dict function'

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


def mrfloc(funit, handle=0, stnid='*********', idtyp=-1, ilat=-1, ilon=-1,
           date=-1, time=-1, sup=None):
    """
    Locate position of report in file.

    handle = mrfloc(funit, handle, stnid, idtyp, lat, lon, date, time, sup)
    handle = mrfloc(funit, handle, stnid)
    handle = mrfloc(funit, params)

    Args:
        funit  : File unit number
        handle : Start position for the search,
                 0 (zero) for beginning of file
        stnid  : Station ID
                 '*' for any name
        idtyp  : Report Type (-1 as wildcard)
        ilat   : Station latitude (1/100 of degrees)  (-1 as wildcard)
                 with respect to the south pole. (0 to 1800)
                 (100*(latitude+90)) of a station or the
                 lower left corner of a box.
        ilon   : Station longitude (1/100 of degrees) (-1 as wildcard)
                 (0 to 36000) of a station or lower left corner of a box.
        date   : Report valid date (YYYYMMDD) (-1 as wildcard)
        time   : Observation time/hour (HHMM) (-1 as wildcard)
        sup    : Additional search keys (array of int)
        params : dict with the above values as key/vals
                 keys not specified are taken as wildcads
    Returns:
        int, report handle
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> ## See mrfget, mrbhdr, mrbprm, mrbxtr for how to get the meta + data
    >>> rmn.burp_close(funit)

    See Also:
        mrfget
        mrfput
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    params = {
        'handle' : handle,
        'stnid'  : stnid,
        'idtyp'  : idtyp,
        'ilat'   : ilat,
        'ilon'   : ilon,
        'date'   : date,
        'time'   : time,
        'sup'    : sup
        }
    if isinstance(handle, dict):
        params['handle'] = 0
        params.update(handle)
    nsup = 0
    if params['sup'] is None:
        params['sup'] = _np.empty((1,), dtype=_np.int32)
    elif isinstance(params['sup'], (list,tuple)):
        nsup = len(params['sup'])
        if nsup == 0:
            params['sup'] = _np.empty((1,), dtype=_np.int32)
        else:
            params['sup'] = _np._np.asfortranarray(params['sup'], dtype=_np.int32)
    #NOTE: providing sup as ndarray of size > 0 with value zero cause a seg fault, apparently sup is not supported by the librmn api
    ## elif isinstance(sup, _np.ndarray):
    ##     nsup = sup.size
    else:
        raise TypeError('sup should be a None, list, tuple or ndarray')
    if nsup > 0:
        raise TypeError('sup is not supported in this version of librmn, ' +
                        'should prived None or empty list')
    handle = _rp.c_mrfloc(funit, params['handle'], params['stnid'],
                          params['idtyp'], params['ilat'], params['ilon'],
                          params['date'], params['time'],
                          params['sup'], nsup)
    if handle < 0:
        raise BurpError('c_mrfloc', params['handle'])
    return handle


def mrfget(handle, rpt=None, funit=None):
    """
    Read/Put report pointed to by handle into buffer.

    rpt = mrfget(handle, funit=funit)

    Args:
        handle : Report handle
        rpt    : Report data buffer (ndarray of dtype int32)
                 or max report size in file (int)
        funit  : file unit number where to read report,
                 only needed if rpt is None
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> ## See mrbhdr, mrbprm, mrbxtr for how to get the meta + data
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
    if rpt is None or isinstance(rpt, int):
        nrpt = rpt
        if rpt is None:
            nrpt = mrfmxl(funit)
            #TODO: nrpt = max(64, rmn.mrfmxl(funit))+10
        nrpt *= 2  #TODO: should we remove this?
        rpt = _np.empty((nrpt,), dtype=_np.int32)
        rpt[0] = nrpt
    elif not isinstance(rpt, _np.ndarray):
        raise TypeError('rpt should be an ndarray')
    istat = _rp.c_mrfget(handle, rpt)
    if istat != 0:
        raise BurpError('c_mrfget', istat)
    return rpt

#TODO: review
def mrfput(funit, handle, rpt):
    """
    Write a report.

    mrfput(funit, handle, rpt)

    Args:
        funit  : File unit number
        handle : Report handle
        rpt    : Report data
    Returns:
       array, Report data buffer
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RW)
    >>> rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> nrpt  = 1024 ## Set nrpt to appropriate size
    >>> rpt   =_np.empty((nrpt,), dtype=_np.int32)
    >>> ## Fill rpt with relevant info
    >>> #TODO: describe what tools can be used to fill rpt
    >>> handle = 0
    >>> rmn.mrfput(funit, handle, rpt)
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
    istat = _rp.c_mrfput(funit, handle, rpt)
    if istat != 0:
        raise BurpError('c_mrfput', istat)
    return


def mrbhdr(rpt):
    """
    Returns report header information.

    params = mrbhdr(rpt)
    
    Args:
        rpt : Report data (array)
    Returns:
        {
            'time'  : (int)   Observation time/hour (HHMM)
            'timehh': (int)   Observation time hour part (HH)
            'timemm': (int)   Observation time minutes part (MM)
            'flgs'  : (int)   Global flags
                              (24 bits, Bit 0 is the right most bit of the word)
                              See BURP_FLAGS_IDX_NAME for Bits/flags desc.
            'flgsl' : (list)  Global flags as a list of int
                              See BURP_FLAGS_IDX for Bits/flags desc.
            'flgsd' : (str)   Description of set flgs, comma separated
            'stnid' : (str)   Station ID
                              If it is a surface station, STNID = WMO number.
                              The name is aligned at left and filled with
                              spaces. In the case of regrouped data,
                              STNID contains blanks.
            'idtyp' : (int)   Report Type
            'idtypd': (str)   Report Type description 
            'ilat'  : (int)   Station latitude (1/100 of degrees)
                              with respect to the south pole. (0 to 1800)
                              (100*(latitude+90)) of a station or the
                              lower left corner of a box.
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
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> ## See mrbprm, mrbxtr for how to get the meta + data
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
    istat  = _rp.c_mrbhdr(rpt, _ct.byref(itime), _ct.byref(iflgs),
                          stnids, _ct.byref(idtyp),
                          _ct.byref(ilat), _ct.byref(ilon),
                          _ct.byref(idx), _ct.byref(idy), _ct.byref(ialt), 
                          _ct.byref(idelay), _ct.byref(idate), _ct.byref(irs),
                          _ct.byref(irunn), _ct.byref(nblk), 
                          sup, nsup, xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbhdr', istat)
    flgsl = _rbc.BURP2BIN2LIST(iflgs.value,len(_rbc.BURP_FLAGS_IDX_NAME))
    try:
        idtyp_desc = _rbc.BURP_IDTYP_DESC[str(idtyp.value)]
    except KeyError:
        idtyp_desc = ''
    try:
        flgs_desc = "".join([_rbc.BURP_FLAGS_IDX_NAME[i]+", " if flgsl[i] else "" for i in range(len(flgsl))])
    except KeyError:
        flgs_desc = ''
    if flgs_desc:
        if flgs_desc[-2:] == ', ':
            flgs_desc = flgs_desc[:-2]
    return {
            'time'  : itime.value,
            'timehh': itime.value // 100,
            'timemm': itime.value % 100,
            'flgs'  : iflgs.value,
            'flgsl' : flgsl,
            'flgsd' : flgs_desc,
            'stnid' : stnids.value,
            'idtyp' : idtyp.value,
            'idtypd': idtyp_desc,
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
            'runn'  : irunn.value,  #TODO: provide decoded runn?
            'nblk'  : nblk.value,
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
        }


def mrbprm(rpt, blkno):
    """
    Returns block header information.

    params = mrbprm(rpt, blkno)

    Args:
        rpt   : Report data  (array)
        blkno : block number (int > 0)
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
            'bdesc' : (int) Block descriptor. (0-2047) (not used)
            'btyp'  : (int) Block type (0-2047), made from 3 components:
                            BKNAT: kind component of Block type
                            BKTYP: Data-type component of Block type
                            BKSTP: Sub data-type component of Block type
            'bknat'       : (int) block type, kind component
            'bknat_multi' : (int) block type, kind component, uni/multi bit
                                  0=uni, 1=multi
            'bknat_kind'  : (int) block type, kind component, kind value
                                  See BURP_BKNAT_KIND_DESC
            'bknat_kindd' : (str) desc of bknat_kind
            'bktyp'       : (int) block type, Data-type component
            'bktyp_alt'   : (int) block type, Data-type component, surf/alt bit
                                  0=surf, 1=alt
            'bktyp_kind'  : (int) block type, Data-type component, flags
                                  See BURP_BKTYP_KIND_DESC
            'bktyp_kindd' : (str) desc of bktyp_kind
            'bkstp'       : (int) block type, Sub data-type component
            'bkstpd'      : (str)  desc of bktyp_kindd
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
            'datypd': (str) Data type name/desc
        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg values
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkparams = rmn.mrbprm(rpt, iblk+1)
    >>> ## See mrbxtr for how to get the data
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbxtr
        mrbtyp_decode
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    if blkno <= 0:
        raise ValueError('Provided blkno must be > 0')
    try:
        maxblkno  = mrbhdr(rpt)['nblk']  #TODO: should we do this?
    except:
        maxblkno = -1
    if maxblkno > 0 and blkno > maxblkno:
        raise ValueError('Provided blkno must be < nb of block in report')
    nele = _ct.c_int()
    nval = _ct.c_int()
    nt = _ct.c_int()
    bfam = _ct.c_int()
    bdesc = _ct.c_int()
    btyp = _ct.c_int()
    nbit = _ct.c_int()
    bit0 = _ct.c_int()
    datyp = _ct.c_int()
    istat = _rp.c_mrbprm(rpt, blkno, _ct.byref(nele), _ct.byref(nval),
                         _ct.byref(nt), _ct.byref(bfam), _ct.byref(bdesc),
                         _ct.byref(btyp), _ct.byref(nbit),
                         _ct.byref(bit0), _ct.byref(datyp))
    if istat != 0:
        raise BurpError('c_mrbprm', istat)
    try:
        datypd = _rbc.BURP_DATYP_NAMES[datyp.value]
    except:
        datypd = ''
    params =  {
            'nele'  : nele.value,
            'nval'  : nval.value,
            'nt'    : nt.value,
            'bfam'  : bfam.value,  #TODO: provide decoded bfam?
            'bdesc' : bdesc.value,
            'btyp'  : btyp.value,
            'nbit'  : nbit.value,
            'bit0'  : bit0.value,
            'datyp' : datyp.value,
            'datypd': datypd
        }
    params.update(mrbtyp_decode(btyp.value))
    return params


def mrbtyp_decode(btyp):
    """
    Decode btyp into bknat, bktyp, bkstp

    params = mrbtyp_decode(btyp)

    Args:
        btyp  : (int) Block type (0-2047), made from 3 components:
                      BKNAT: kind component of Block type
                      BKTYP: Data-type component of Block type
                      BKSTP: Sub data-type component of Block type
    Returns:
        {
            'bknat'       : (int) block type, kind component
            'bknat_multi' : (int) block type, kind component, uni/multi bit
                                  0=uni, 1=multi
            'bknat_kind'  : (int) block type, kind component, kind value
                                  See BURP_BKNAT_KIND_DESC
            'bknat_kindd' : (str) desc of bknat_kind
            'bktyp'       : (int) block type, Data-type component
            'bktyp_alt'   : (int) block type, Data-type component, surf/alt bit
                                  0=surf, 1=alt
            'bktyp_kind'  : (int) block type, Data-type component, flags
                                  See BURP_BKTYP_KIND_DESC
            'bktyp_kindd' : (str) desc of bktyp_kind
            'bkstp'       : (int) block type, Sub data-type component
            'bkstpd'      : (str)  desc of bktyp_kindd
        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkparams = rmn.mrbprm(rpt, iblk+1)
    >>>     params    = rmn.mrbtyp_decode(blkparams['btyp'])
    >>> rmn.burp_close(funit)

    See Also:
        mrbtyp_encode_bknat
        mrbtyp_encode_bktyp
        mrbtyp_encode
        burp_open
        mrfloc
        mrfget
        mrfhdr
        mrbprm
    """
    if btyp < 0:
        raise ValueError('Provided btyp must be >= 0: {}'.format(btyp))    
    bknat = _ct.c_int()
    bktyp = _ct.c_int()
    bkstp = _ct.c_int()
    istat = _rp.c_mrbtyp(_ct.byref(bknat), _ct.byref(bktyp),
                         _ct.byref(bkstp), btyp)
    if istat != 0:
        raise BurpError('c_mrbtyp', istat)
    bknat_multi = int(_rbc.BURP2BIN(bknat.value,4)[0:2],2)
    bknat_kind  = int(_rbc.BURP2BIN(bknat.value,4)[2:],2)
    bktyp_alt   = int(_rbc.BURP2BIN(bktyp.value,4)[0],2)
    bktyp_kind  = int(_rbc.BURP2BIN(bktyp.value,4)[1:],2)
    try:
        bknat_kindd = _rbc.BURP_BKNAT_KIND_DESC[bknat_kind]
    except:
        bknat_kindd = ''
    try:
        bktyp_kindd = _rbc.BURP_BKTYP_KIND_DESC[bktyp_kind]
    except:
        bktyp_kindd = ''
    try:
        bkstpd = _rbc.BURP_BKSTP_DESC[(bktyp_alt,bktyp_kind,bkstp.value)]
    except:
        bkstpd = ''
    return {
        'bknat'       : bknat.value,
        'bknat_multi' : bknat_multi,
        'bknat_kind'  : bknat_kind,
        'bknat_kindd' : bknat_kindd,
        'bktyp'       : bktyp.value,
        'bktyp_alt'   : bktyp_alt, 
        'bktyp_kind'  : bktyp_kind,
        'bktyp_kindd' : bktyp_kindd,
        'bkstp'       : bkstp.value,
        'bkstpd'      : bkstpd
        }

def mrbtyp_encode_bknat(bknat_multi, bknat_kind):
    """
    Encode bknat_multi, bknat_kind into bknat

    bknat = mrbtyp_encode_bknat(bknat_multi, bknat_kind)

    Args:
        bknat_multi : (int) block type, kind component, uni/multi bit
                            0=uni, 1=multi
        bknat_kind  : (int) block type, kind component, kind value
                            See BURP_BKNAT_KIND_DESC
    Returns:
        int, encoded block type, kind component

    Examples:
    >>> import rpnpy.librmn.all as rmn

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bktyp        
        rpnpy.librmn.burp_const
    """
    return int(_rbc.BURP2BIN(bknat_multi,2)+_rbc.BURP2BIN(bknat_kind,2),2)

    
def mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind):
    """
    Encode bktyp_alt, bktyp_kind into bktyp

    bktyp = mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)

    Args:
        bktyp_alt  : (int)  block type, Data-type component, surf/alt bit
                            0=surf, 1=alt
        bktyp_kind : (int)  block type, Data-type component, flags
                            See BURP_BKTYP_KIND_DESC
    Returns:
        int, block type, Data-type component

    Examples:
    >>> import rpnpy.librmn.all as rmn

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bknat
        rpnpy.librmn.burp_const
    """
    return int(_rbc.BURP2BIN(bktyp_alt,1)+_rbc.BURP2BIN(bktyp_kind,6),2)

    
#TODO: mrbtyp_encode_bkstp() #?

    
def mrbtyp_encode(bknat, bktyp=None, bkstp=None):
    """
    Encode bknat, bktyp, bkstp to btyp

    btyp = mrbtyp_encode(bknat, bktyp, bkstp)
    btyp = mrbtyp_encode(params)

    Args:
        bknat  : (int) block type, kind component
        bktyp  : (int) block type, Data-type component
        bkstp  : (int) block type, Sub data-type component
        params : dict with the above key:values
    Returns:
        int, encode btyp value
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> #TODO

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bknat
        mrbtyp_encode_bktyp
        rpnpy.librmn.burp_const
        #TODO: 
    """
    if isinstance(bknat, dict):
        try:
            bktyp = bknat['bktyp']
            bkstp = bknat['bkstp']
            bknat = bknat['bknat']
        except:
            raise BurpError('mrbtyp_encode: must provied all 3 sub values: bknat, bktyp, bkstp',)
    if bknat < 0 or bktyp < 0 or bkstp < 0:
        raise ValueError('Provided bknat, bktyp, bkstp must all be > 0')
    #TODO: use librmn c funtion (investigate why it does not work as expected)
    ## c_bknat = _ct.c_int(bknat)
    ## c_bktyp = _ct.c_int(bktyp)
    ## c_bkstp = _ct.c_int(bkstp)
    ## btyp = 0
    ## istat = _rp.c_mrbtyp(_ct.byref(c_bknat), _ct.byref(c_bktyp),
    ##                      _ct.byref(c_bkstp), btyp)
    ## if istat <= 0:
    ##     raise BurpError('c_mrbtyp', istat)
    ## return istat
    return int("{0:04b}{1:07b}{2:04b}".format(bknat, bktyp, bkstp), 2)


def mrbxtr(rpt, blkno, cmcids=None, tblval=None, dtype=_np.int32):
    """
    Extract block of data from the buffer.
    
    blkdata = mrbxtr(rpt, blkno)
    blkdata = mrbxtr(rpt, blkno, cmcids, tblval)

    Args:
        rpt    : Report data  (array)
        blkno  : block number (int > 0)
        cmcids : (optional) return data arrays for cmcids
        tblval : (optional) return data arrays for tblval
        dtype  : (optional) numpy type for tblval creation, if tblval is None
    Returns
        {
            'cmcids' : (array) List of element names in the report in numeric
                               CMC codes. (Size: NELE; type: int)
                               NELE: Number of meteorological elements in a
                                     block. 1st dimension of the array
                                     TBLVAL(block). (0-127)
                               #TODO: check if cmcids are CMC or BUFR codes (most probable CMC codes)... make name consistent across all fn
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
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(rpt, iblk+1)
    >>> ## See mrbdcl, mrbcvt_decode, mrb_prm_xtr_dcl_cvt for how to decode the data
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt_dict
        mrbcvt_decode
        mrb_prm_xtr_dcl_cvt
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    if blkno <= 0:
        raise ValueError('Provided blkno must be > 0')
    try:
        maxblkno  = mrbhdr(rpt)['nblk']  ##TODO should we do this?
    except:
        maxblkno = -1
    if maxblkno > 0 and blkno > maxblkno:
        raise ValueError('Provided blkno must be < nb of block in report')
    bparams = mrbprm(rpt, blkno)
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
    if isinstance(cmcids, _np.ndarray):
        if not cmcids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided cmcids should be F_CONTIGUOUS')
        if dtype0 != cmcids.dtype:
            raise TypeError('Expecting cmcids of type {0}, got: {1}'
                            .format(repr(dtype0), repr(cmcids.dtype)))
        if len(cmcids.shape) != 1 or cmcids.size != nele:
            raise TypeError('cmcids should be a ndarray of rank 1 (nele)')
    elif cmcids is None:
        cmcids = _np.empty(nele, dtype=dtype0, order='FORTRAN')
    else:
        raise TypeError('cmcids should be a ndarray of rank 1 (nele)')
    istat = _rp.c_mrbxtr(rpt, blkno, cmcids, tblval)
    if istat != 0:
        raise BurpError('c_mrbxtr', istat)
    bparams['cmcids'] = cmcids
    bparams['tblval'] = tblval
    return bparams     


def mrbdcl(cmcids):
    """
    Convert CMC codes to BUFR codes.

    bufrids = mrbdcl(cmcids)
    
    Args:
        cmcids : List of element names in the report in numeric CMC codes.
                 See the code desc in the FM 94 BUFR man
    Returns
        array, list of  lstele converted to BUFR codes
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(rpt, iblk+1)
    >>>     bufrids = rmn.mrbdcl(blkdata['cmcids'])
    >>> ## See mrbcvt_decode, mrb_prm_xtr_dcl_cvt for how to decode the data
    >>> rmn.burp_close(funit)

    See Also:
        mrbcol
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbxtr
        mrbcvt_dict
        mrbcvt_decode
        mrb_prm_xtr_dcl_cvt
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    if isinstance(cmcids, (int, _np.int32)):
        v = _rp.c_mrbdcv(cmcids)
        return v
    dtype = _np.int32
    if isinstance(cmcids, _np.ndarray):
        if not cmcids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided cmcids should be F_CONTIGUOUS')
        if dtype != cmcids.dtype:
            raise TypeError('Expecting cmcids of type {0}, got: {1}'
                            .format(repr(dtype),repr(cmcids.dtype)))
    elif isinstance(cmcids, (tuple, list)):
        cmcids = _np.array(cmcids, dtype=dtype)
    elif isinstance(cmcids, (int, long)):
        cmcids = _np.array([cmcids], dtype=dtype)
    else:
        raise TypeError('cmcids should be a ndarray of rank 1')
    bufrids = _np.empty(cmcids.size, dtype=dtype, order='FORTRAN')
    istat = _rp.c_mrbdcl(cmcids, bufrids, cmcids.size)
    if istat != 0:
        raise BurpError('c_mrbdcl', istat)
    return bufrids


def mrbcol(bufrids):
    """
    Convert BUFR codes to CMC codes.
    
    cmcids = mrbcol(bufrids)

    Args:
        bufrids : List of element names in the report in numeric BUFR codes.
    Returns
        array, list of bufrids converted to CMC codes
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(rpt, iblk+1)
    >>>     bufrids = rmn.mrbdcl(blkdata['cmcids'])
    >>>     cmcids  = rmn.mrbcol(bufrids)
    >>> rmn.burp_close(funit)

    See Also:
        mrbdcl
        mrbcvt_dict
        mrfhdr
        mrbxtr
    """
    if isinstance(bufrids, (int, _np.int32)):
        v = _rp.c_mrbcov(bufrids)
        return v
    dtype = _np.int32
    if isinstance(bufrids, _np.ndarray):
        if not bufrids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided bufrids should be F_CONTIGUOUS')
        if dtype != bufrids.dtype:
            raise TypeError('Expecting bufrids of type {0}, got: {1}'
                            .format(repr(dtype),repr(bufrids.dtype)))
    elif isinstance(bufrids, (tuple, list)):
        bufrids = _np.array(bufrids, dtype=dtype)
    elif isinstance(bufrids, (int, long)):
        bufrids = _np.array([bufrids], dtype=dtype)
    else:
        raise TypeError('bufrids should be a ndarray of rank 1')
    cmcids = _np.empty(bufrids.size, dtype=dtype, order='FORTRAN')
    istat = _rp.c_mrbcol(bufrids, cmcids, bufrids.size)
    if istat != 0:
        raise BurpError('c_mrbcol', istat)
    return cmcids


def mrbcvt_dict(cmcid, raise_error=True):
    """
    Extract BUFR table B info for cmcid

    cvtdict = mrbcvt_dict(bufrid)
    
    Args:
        cmcid       : Element CMC code name
        raise_error : Specify what to do when bufrid is not found in BURP table B
                      if True,  raise an error 
                      if False, return an empty dict with key error=-1
                      (optional, default=True)
    Returns
        {
        'e_cmcid'   : (int) Element CMC code
        'e_bufrid'  : (int) Element BUFR code
        'e_bufrid_F': (int) Type part of Element code (e.g. F=0 for obs)
        'e_bufrid_X': (int) Class part of Element code 
        'e_bufrid_Y': (int) Class specific Element code part of Element code 
        'e_desc'    : (str) Element description
        'e_cvt'     : (int) Flag for conversion (1=need units conversion)
        'e_units'   : (str) Units desciption
        'e_scale'   : (int) Scaling factor for element value conversion
        'e_bias'    : (int) Bias for element value conversion
        'e_nbits'   : (int) nb of bits for encoding value
        'e_multi'   : (int) 1 means descriptor is of the "multi" or 
                          repeatable type (layer, level, etc.) and
                          it can only appear in a "multi" block of data
        'e_error'   : (int) 0 if bufrid found in BURP table B, -1 otherwise
        }
    Raises:
        KeyError   on key not found in burp table b dict
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrbxtr(rpt, iblk+1)
    >>>     for cmcid in blkdata['cmcids']:
    >>>         cvtdict = rmn.mrbcvt_dict(cmcid)
    >>>         print('{e_bufrid:0>6} {e_desc} [{e_units}]'.format(**cvtdict))
    >>> rmn.burp_close(funit)

    See Also:
        mrfget
        mrbcvt_dict
        mrbdcl
        mrbcol
        mrbxtr
        mrfhdr
        mrfget
        mrfloc
        burp_open
    """
    if not len(_mrbcvt_dict_full.keys()):
        AFSISIO = os.getenv('AFSISIO', '')
        if not AFSISIO:
            AFSISIO = os.getenv('rpnpy', '/')
            mypath = os.path.join(AFSISIO.strip(), 'share',
                                  _rbc.BURP_TABLE_B_FILENAME)
        else:
            mypath = os.path.join(AFSISIO.strip(), 'datafiles/constants',
                                  _rbc.BURP_TABLE_B_FILENAME)
        try:
            fd = open(mypath, "r")
            try:     rawdata = fd.readlines()
            finally: fd.close()
        except IOError:
            raise IOError(" Oops! File does not exist or is not readable: {0}".format(mypath))
        for item in rawdata:
            if item[0] != '*':
                id = int(item[0:6])
                d = {
                    'e_error'   : 0,
                    'e_cmcid'   : mrbcol(id),
                    'e_bufrid'  : id,
                    'e_bufrid_F': int(item[0]),
                    'e_bufrid_X': int(item[1:3]),
                    'e_bufrid_Y': int(item[3:6]),
                    'e_cvt'     : 1,
                    'e_desc'    : item[8:51].strip(),
                    'e_units'   : item[52:63].strip(),
                    'e_scale'   : int(item[63:66]),
                    'e_bias'    : int(item[66:77]),
                    'e_nbits'   : int(item[77:83]),
                    'e_multi'   : 0
                    }
                if item[50] == '*':
                    d['e_cvt']  = 0
                    d['e_desc'] = item[8:50].strip()
                ## elif d['e_units'] in ('CODE TABLE', 'FLAG TABLE', 'NUMERIC'):
                elif d['e_units'] in ('CODE TABLE', 'FLAG TABLE'):  #TODO: check if NUMERIC should be included
                    d['e_cvt']  = 0
                if len(item) > 84 and item[84] == 'M':
                    d['e_multi'] = 1
                _mrbcvt_dict_full[id] = d
    bufrid =  mrbdcl(cmcid)
    try:
        return _mrbcvt_dict_full[bufrid]
    except:
        if raise_error:
            raise
        else:
            id = "{0:0>6}".format(bufrid)
            return {
                'e_error'   : -1, 
                'e_cmcid'   : cmcid,
                'e_bufrid'  : bufrid,
                'e_bufrid_F': int(id[0]),
                'e_bufrid_X': int(id[1:3]),
                'e_bufrid_Y': int(id[3:6]),
                'e_cvt'     : 0,
                'e_desc'      : '',
                'e_units'   : '',
                'e_scale'   : 0,
                'e_bias'    : 0,
                'e_nbits'   : 0,
                'e_multi'   : 0
                }
 
def mrbcvt_decode(cmcids, tblval=None, datyp=_rbc.BURP_DATYP_LIST['float']):
    """
    Convert table/BURF values to real values.

    rval = mrbcvt_decode(cmcids, tblval)
    rval = mrbcvt_decode(cmcids, tblval, datyp)
    rval = mrbcvt_decode(blkdata)
    rval = mrbcvt_decode(blkdata, datyp)

    Args:
        cmcids : List of element names in the report in numeric BUFR codes.
                 See the code desc in the FM 94 BUFR man
        tblval : BURF code values (array of int or float)
        datyp' : (optional) Data type as obtained from mrbprm (int)
                 See rpnpy.librmn.burp_const BURP_DATYP_LIST
                                         and BURP_DATYP2NUMPY_LIST
                 Default: 6 (float)
        blkdata : (dict) Block data as returned by mrbxtr,
                           must contains 2 keys: 'cmcids', 'tblval'
    Returns
        array, dtype depends on datyp
    Raises:
        KeyError   on missing blkdata keys
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata  = rmn.mrbxtr(rpt, iblk+1)
    >>>     rval     = rmn.mrbcvt_decode(blkdata)
    >>> rmn.burp_close(funit)

    See Also:
        mrbcvt_encode
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbxtr
        mrbdcl
        mrbcvt_dict
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    if isinstance(cmcids, dict):
        try:
            tblval = cmcids['tblval']
            cmcids = cmcids['cmcids']
        except:
            raise KeyError('Provided blkdata should have these 2 keys: cmcids, tblval')
    if isinstance(tblval, _np.ndarray):
        if not tblval.flags['F_CONTIGUOUS']:
            raise TypeError('Provided tblval should be F_CONTIGUOUS')
        if len(tblval.shape) != 3:
            raise TypeError('Provided tblval should be an ndarray of rank 3')
    else:
        raise TypeError('Provided tblval should be an ndarray of rank 3')
    (nele, nval, nt) = tblval.shape
    dtype = _np.int32
    if isinstance(cmcids, _np.ndarray):
        if not cmcids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided cmcids should be F_CONTIGUOUS')
        if dtype != cmcids.dtype:
            raise TypeError('Expecting cmcids of type {0}, got: {1}'
                            .format(repr(dtype),repr(cmcids.dtype)))
        if len(cmcids.shape) != 1 or cmcids.size != nele:
            raise TypeError('cmcids should be a ndarray of rank 1 (nele)')
    else:
        raise TypeError('cmcids should be a ndarray of rank 1 (nele)')
            
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
    ## Note: Type 3 (char) and 5 (upchar) are processed like strings of bits
    ##       thus, the user should do the data compression himself.

    if not datyp in _rbc.BURP_DATYP_NAMES.keys():
        raise ValueError('Out of range datyp={0}'.format(datyp))

    #TODO: complete dtype support for mrbcvt_decode
    
    if _rbc.BURP_DATYP_NAMES[datyp] in ('complex', 'dcomplex'):
        raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
                        .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]),
                        _ERR_INV_DATYP)
    elif _rbc.BURP_DATYP_NAMES[datyp] in ('char', 'upchar'):
        raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
                        .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    elif _rbc.BURP_DATYP_NAMES[datyp] in ('double'):
        raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
                        .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    ## elif _rbc.BURP_DATYP_NAMES[datyp] in ('float'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    elif _rbc.BURP_DATYP_NAMES[datyp] in ('binary',):
        raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
                        .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))
    ## else:  ## ('uint', 'int'):
    ##     raise BurpError('Conversion not Yet implemented for datyp={0} ({1})'
    ##                     .format(datyp, _rbc.BURP_DATYP_NAMES[datyp]))

    dtype = _rbc.BURP_DATYP2NUMPY_LIST[datyp]
    rval  = tblval.astype(dtype)    
    istat = _rp.c_mrbcvt(cmcids, tblval, rval, nele, nval, nt,
                         _rbc.MRBCVT_DECODE)
    if istat != 0:
        raise BurpError('c_mrbcvt', istat)
    try:
        rval_missing = _rbc.BURP_RVAL_MISSING[datyp]
    except:
        rval_missing = _rbc.BURP_TBLVAL_MISSING
    rval[tblval == _rbc.BURP_TBLVAL_MISSING] = rval_missing
    #TODO: ie e_cvt == 0: put tblval

    return rval


#TODO: function for mrbprm+xtr+cvt+dcl (and maybe hrd,prm)
## def mrb_hdr_prm_xtr_dcl_cvt(rpt, blkno, cmcids=None, tblval=None, rval=None, dtype=_np.int32):
        ## cmcids, tblval: (optional) return data arrays
        ## dtype : (optional) numpy type for tblval creation, if tblval is None
def mrb_prm_xtr_dcl_cvt(rpt, blkno):
    """
    Extract block of data from the buffer and decode its values
    
    blkdata = mrb_prm_xtr_dcl_cvt(rpt, blkno)

    Args:
        rpt   : Report data  (array)
        blkno : block number (int > 0)
    Returns
        {
            #TODO: full list of returned parameters
            'cmcids' : (array) List of element names in the report in numeric
                               BUFR codes. (Size: NELE; type: int)
                               NELE: Number of meteorological elements in a
                                     block. 1st dimension of the array
                                     TBLVAL(block). (0-127)
                               See the code desc in the FM 94 BUFR man
            'tblval' : (array) Block data
                               (Shape: NELE, NVAL, NT; type: int)
                               NELE: Number of meteorological elements in block
                               NVAL: Number of values per element.
                               NT  : Nb of groups of NELE x NVAL vals in block.
            'rval'   : (array) Decoded Block data

        }        
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg values
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit  = rmn.burp_open('myburpfile.brp')
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrfhdr(rpt)
    >>> for iblk in xrange(params['nblk']):
    >>>     blkdata = rmn.mrb_prm_xtr_dcl_cvt(rpt, iblk+1)
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt_decode
        mrbcvt_dict
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    blkdata = mrbprm(rpt, blkno)
    blkdata.update(mrbxtr(rpt, blkno))
    blkdata['bufrids'] = mrbdcl(blkdata['cmcids'])
    if blkdata['bknat_kindd'] != 'flags':
        blkdata['rval'] = mrbcvt_decode(blkdata['cmcids'], blkdata['tblval'])
    else:
        blkdata['rval'] = blkdata['tblval'][:,:,:]
    return blkdata


#TODO: mrbcvt_encode
## def mrbcvt_encode(cmcids, rval): ##TODO:

#TODO: review
def mrbini(funit, rpt, time, flgs, stnid, idtp, lat, lon, dx, dy, elev, drnd,
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
    istat = _rp.c_mrbini(funit, rpt, time, flgs, stnid, idtp, lat, lon,  
                         dx, dy, elev, drnd, date, oars, runn, sup, nsup,
                         xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbini', istat)
    return istat


#TODO: review
def mrbadd(rpt, blkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp,
           cmcids, tblval): #TODO change cmcids for consistency (more explict name)
    """
    Adds a block to a report.
    #TODO: should accept a dict as input for all args
    
    Args:
        #TODO: 
        rpt    (array) : vector to contain the report
        nele   (int)   : number of meteorogical elements in block
        nval   (int)   : number of data per elements
        nt     (int)   : number of group of nelenval values in block
        bfam   (int)   : block family (12 bits, bdesc no more used)
        bdesc  (int)   : kept for backward compatibility
        btyp   (int)   : block type
        nbit   (int)   : number of bit to keep per values
        datyp  (int)   : data type for packing
        cmcids (array) : list of nele meteorogical elements
        tblval (array) : array of values to write (nele*nval*nt)
    Returns
        #TODO: 
        rpt    (array) : vector to contain the report
        blkno  (int)   : number of blocks in rpt
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
    istat = _rp.c_mrbadd(rpt, blkno, nele, nval, nt, bfam, bdesc, btyp, nbit,
                         bit0, datyp, cmcids, tblval)
    if istat != 0:
        raise BurpError('c_mrbadd', istat)
    return rpt


#TODO: review
def mrbdel(rpt, blkno):
    """
    Delete a particular block of the report.

    mrbdel(rpt, blkno)

    Args:
        rpt   : report data (array) 
        blkno : number of blocks in rpt (int)
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
    istat = _rp.c_mrbdel(rpt, blkno)
    if istat != 0:
        raise BurpError('c_mrbdel', istat)
    return rpt


#TODO: review
def mrfdel(handle):
    """
    Delete a particular report from a burp file.

    mrfdel(handle)

    Args:
        handle : Report handle
    Returns
        None
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
    istat = _rp.c_mrfdel(handle)
    if istat != 0:
        raise BurpError('c_mrfdel', istat)
    return

##TODO: mrfrwd
##TODO: mrfapp

# =========================================================================

if __name__ == "__main__":
    print("Python interface for BUPR files.")

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
