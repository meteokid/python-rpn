#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Michael Sitwell <michael.sitwell@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Python interface for [[BURP]] (Binary Universal Report Protocol) files.
It Contains wrappers for ctypes functions in proto_burp and the BurpError class.

BURP est depuis juin 1992 le format des bases de donnees operationnelles du
CMC pour tous les fichiers de type 'observations', au sens large du terme.

On peut voir BURP comme un complement du format Standard de RPN.

Un enregistrement de fichier standard RPN contient un ensemble de valeurs
d'une variable donnee, reparties sur une grille geo-referencee dont les
points sont espaces selon des specifications mathematiques.

Par opposition (ou complementarite), BURP est concu pour emmagasiner un
ensemble de variables a une 'station', ce dernier terme representant un
point geographique de latitude et longitude determinees.

* Un fichier BURP est constitues de plusieurs "reports".
* Chaque "report" est constitues de plusieurs "blocks".
* Chaque "block"  est constitues de plusieurs "elements".

Notes:
    The functions described below are a very close ''port'' from the
    original [[librmn]]'s [[BURP]] package.
    You may want to refer to the  [[BURP]] documentation for more details.

See Also:
    rpnpy.librmn.burp_const
    rpnpy.utils.burpfile
    rpnpy.burpc.base
    rpnpy.burpc.brpobj
    rpnpy.librmn.proto_burp
"""

import os
import re as _re
import sys
import copy
import ctypes as _ct
import numpy as _np
from rpnpy.librmn import proto_burp as _rp
from rpnpy.librmn import const as _rc
from rpnpy.librmn import burp_const as _rbc
from rpnpy.librmn import base as _rb
from rpnpy.librmn import RMNError
from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR as _C_MKSTR

_ERR_INV_DATYP = 16

_mrbcvt_dict = {
    'path'  : '',
    'raise' : False,
    'init'  : False,
    'dict'  : {}
    }
#_mrbcvt_dict_full.__doc__ = """
#    Parsed BUFR table B into a python dict,
#    see mrbcvt_dict function
#    """

_list2ftnf32 = lambda x: \
    x if isinstance(x, _np.ndarray) \
      else _np.asfortranarray(x, dtype=_np.float32)

def _getCheckArg(okTypes, value, valueDict, key):
    if isinstance(valueDict, dict) and (value is None or value is valueDict):
        if key in valueDict.keys():
            value = valueDict[key]
    if (okTypes is not None) and not isinstance(value, okTypes):
        raise BurpError('For {0} type, Expecting {1}, Got {2}'.
                           format(key, repr(okTypes), type(value)))
    return value


class BurpError(RMNError):
    """
    General librmn.burp module error/exception

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    ...    pass #... a burpfile operation ...
    ... except(rmn.BurpError):
    ...    pass #ignore the error
    >>> #...
    >>> raise rmn.BurpError()
    Traceback (most recent call last):
      File "/usr/lib/python2.7/doctest.py", line 1289, in __run
        compileflags, 1) in test.globs
      File "<doctest __main__.BurpError[4]>", line 1, in <module>
        raise rmn.BurpError()
    BurpError: 'Error occured while executing; '

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
    if not isinstance(filename, str):
        raise TypeError("isBURP: Expecting arg of type str, Got {0}"\
                        .format(type(filename)))
    if filename.strip() == '':
        raise ValueError("isBURP: must provide a valid filename")
    return _rb.wkoffit(filename) in (
        _rc.WKOFFIT_TYPE_LIST['BURP'],
        _rc.WKOFFIT_TYPE_LIST['BUFR'],
        ## _rc.WKOFFIT_TYPE_LIST['BLOK'], #TODO?: accept 'BLOK'... ?
        )


def burp_open(filename, filemode=_rbc.BURP_MODE_READ):
    """
    Open the specified burp file
    Shortcut for fnom+mrfopn

    iunit = burp_open(filename)
    iunit = burp_open(filename, FST_RO)

    Args:
        filename : path/name of the file to open
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
    >>> os.unlink('newfile.brp')  # Remove test file

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
        raise ValueError('filemode should be one of BURP_MODE_READ,' +
                         ' BURP_MODE_CREATE, BURP_MODE_APPEND')
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


def mrfopt(name, value=None):
    """
    Set/Get BURP file options

    mrfopt(name, value)

    Args:
        name  : name of option to be set or printed
                or one of these constants:
                BURPOP_MISSING, BURPOP_MSGLVL
        value : value to be set (float or string) (optional)
                If not set or is None mrfopt will get the value
                otherwise mrfopt will set to the provided value
                for name=BURPOP_MISSING:
                   a real value for missing data
                for name=BURPOP_MSGLVL, one of these constants:
                   BURPOP_MSG_TRIVIAL,   BURPOP_MSG_INFO,  BURPOP_MSG_WARNING,
                   BURPOP_MSG_ERROR,     BURPOP_MSG_FATAL, BURPOP_MSG_SYSTEM
    Returns:
        str or float, value
    Raises:
        KeyError   on unknown name
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> # Restrict to the minimum the number of messages printed by librmn
    >>> rmn.mrfopt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    'SYSTEM   '

    See Also:
        rpnpy.librmn.burp_const
    """
    if not name in (_rbc.BURPOP_MISSING, _rbc.BURPOP_MSGLVL):
        raise KeyError("mrfopt: uknown name: {}".format(name))

    if value is None:
        if name == _rbc.BURPOP_MSGLVL:
            value = _C_MKSTR(' '*_rbc.BURP_OPTC_STRLEN)
            istat = _rp.c_mrfgoc(_C_WCHAR2CHAR(name), value)
            value = _C_CHAR2WCHAR(value.value).strip()
        else:
            value = _ct.c_float(0.)
            istat = _rp.c_mrfgor(_C_WCHAR2CHAR(name), _ct.byref(value))
            value = value.value
        if istat != 0:
            raise BurpError('c_mrfgocr:'+name, istat)
        return value

    if isinstance(value, str):
        istat = _rp.c_mrfopc(_C_WCHAR2CHAR(name), _C_WCHAR2CHAR(value))
        if istat != 0:
            raise BurpError('c_mrfopc:{}={}'.format(name, value), istat)
    elif isinstance(value, float):
        istat = _rp.c_mrfopr(_C_WCHAR2CHAR(name), value)
        if istat != 0:
            raise BurpError('c_mrfopr:{}={}'.format(name, value), istat)
    else:
        raise TypeError("mrfopt: cannot set value of type: {0} {1}"\
                        .format(type(value), repr(value)))
    return value


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
    >>> n = rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> istat = rmn.fclos(funit)
    >>> os.unlink('myburpfile.brp')  # Remove test file

    See Also:
        mrfcls
        burp_open
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    nrep = _rp.c_mrfopn(funit, _C_WCHAR2CHAR(mode))
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
    >>> n = rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> # ...
    >>> rmn.mrfcls(funit)
    >>> istat = rmn.fclos(funit)
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit = rmn.fnom(filename, rmn.FILE_MODE_RO)
    >>> # nrep  = rmn.mrfvoi(funit)
    >>> istat = rmn.fclos(funit)

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit = rmn.fnom(filename, rmn.FILE_MODE_RO)
    >>> nrep  = rmn.mrfnbr(funit)
    >>> istat = rmn.fclos(funit)

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit = rmn.fnom(filename, rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfmxl(funit)
    >>> istat = rmn.fclos(funit)

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit = rmn.fnom(filename, rmn.FILE_MODE_RO)
    >>> maxlen = rmn.mrfbfl(funit)
    >>> istat = rmn.fclos(funit)

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
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
    elif isinstance(params['sup'], (list, tuple)):
        nsup = len(params['sup'])
        if nsup == 0:
            params['sup'] = _np.empty((1,), dtype=_np.int32)
        else:
            params['sup'] = _np.asfortranarray(params['sup'], dtype=_np.int32)
    else:
        raise TypeError('sup should be a None, list, tuple')
    #NOTES: providing sup as ndarray of size > 0 with value zero cause
    #       a seg fault, apparently sup is not supported by the librmn api
    ## elif isinstance(sup, _np.ndarray):
    ##     nsup = sup.size
    ## else:
    ##     raise TypeError('sup should be a None, list, tuple or ndarray')
    if nsup > 0:
        raise TypeError('sup is not supported in this version of librmn, ' +
                        'should prived None or empty list')
    handle = _rp.c_mrfloc(funit, params['handle'],
                          _C_WCHAR2CHAR(params['stnid']),
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
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
    if rpt is None or isinstance(rpt, _integer_types):
        nrpt = rpt
        if rpt is None:
            nrpt = mrfmxl(funit)  #TODO?: nrpt = max(64, rmn.mrfmxl(funit))+10
        nrpt *= 2  #TODO?: should we remove this?
        rpt = _np.empty((nrpt,), dtype=_np.int32)
        rpt[0] = nrpt
    elif not isinstance(rpt, _np.ndarray):
        raise TypeError('rpt should be an ndarray')
    istat = _rp.c_mrfget(handle, rpt)
    if istat != 0:
        raise BurpError('c_mrfget', istat)
    return rpt


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
    >>> import os
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myburpfile.brp', rmn.FILE_MODE_RW)
    >>> n = rmn.mrfopn(funit, rmn.BURP_MODE_CREATE)
    >>> nrpt  = 1024 ## Set nrpt to appropriate size
    >>> rpt   =_np.empty((nrpt,), dtype=_np.int32)
    >>> ## Fill rpt with relevant info; See mrbini, mrbadd
    >>> handle = 0
    >>> rmn.mrfput(funit, handle, rpt)
    >>> rmn.mrfcls(funit)
    >>> istat = rmn.fclos(funit)
    >>> os.unlink('myburpfile.brp')  # Remove test file

    See Also:
        mrbini
        mrbadd
        mrfget
        mrfopn
        mrfcls
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
        rpnpy.librmn.burp_const
    """
    funit = _getCheckArg(int, funit, funit, 'funit')
    handle = _getCheckArg(int, handle, handle, 'handle')
    rpt = _getCheckArg(_np.ndarray, rpt, rpt, 'rpt')

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> ## See mrbprm, mrbxtr for how to get the meta + data
    >>> rmn.burp_close(funit)

    See Also:
        mrfini
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
    flgs_dict = flags_decode(iflgs.value)
    try:
        idtyp_desc = _rbc.BURP_IDTYP_DESC[str(idtyp.value)]
    except KeyError:
        idtyp_desc = ''
    return {
            'time'  : itime.value,
            'timehh': itime.value // 100,
            'timemm': itime.value % 100,
            'flgs'  : flgs_dict['flgs'],
            'flgsl' : flgs_dict['flgsl'],
            'flgsd' : flgs_dict['flgsd'],
            'stnid' : _C_CHAR2WCHAR(stnids.value),
            'idtyp' : idtyp.value,
            'idtypd': idtyp_desc,
            'ilat'  : ilat.value,
            'lat'   : _rbc.BRP_ILAT2RLAT(ilat.value),
            'ilon'  : ilon.value,
            'lon'   : _rbc.BRP_ILON2RLON(ilon.value),
            'idx'   : idx.value,
            'dx'    : _rbc.BRP_IDX2RDX(idx.value),
            'idy'   : idy.value,
            'dy'    : _rbc.BRP_IDY2RDY(idy.value),
            'ielev' : ialt.value,
            'elev'  : _rbc.BRP_IELEV2RELEV(ialt.value),
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


def flags_decode(iflgs, raise_error=True):
    """
    Decode report header flags information.

    flags_dict = flags_decode(flgs)

    Args:
        flgs        : integer value of the flags
        raise_error : (optional) if True rais an error when convertin error occurs
                      otherwise return a dict with defaults (default=False)
    Returns:
        {
            'flgs'  : (int)   Global flags
                              (24 bits, Bit 0 is the right most bit of the word)
                              See BURP_FLAGS_IDX_NAME for Bits/flags desc.
            'flgsl' : (list)  Global flags as a list of int
                              See BURP_FLAGS_IDX for Bits/flags desc.
            'flgsd' : (str)   Description of set flgs, comma separated
        }

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> print('# ' + rmn.flags_decode(1024)['flgsd'])
    # data observed

    Notes:
        This is a new function in version 2.1.b2

    See Also:
        mrfhdr
        rpnpy.librmn.burp_const
    """
    try:
        flgsl = _rbc.BURP2BIN2LIST(iflgs, len(_rbc.BURP_FLAGS_IDX_NAME))
    except:
        if raise_error:
            raise TypeError('Invalid flags')
        else:
            return {
                'flgs'  : iflgs,
                'flgsl' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0],
                'flgsd' : 'Error While decoding flags'
            }
    try:
        flgs_desc = "".join(
        [_rbc.BURP_FLAGS_IDX_NAME[i] + ", " if flgsl[i] else ""
         for i in range(len(flgsl))]
        )
    except KeyError:
        flgs_desc = ''
    if flgs_desc:
        if flgs_desc[-2:] == ', ':
            flgs_desc = flgs_desc[:-2]
    return {
            'flgs'  : iflgs,
            'flgsl' : flgsl,
            'flgsd' : flgs_desc
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
            'bkno'  : (int) block number
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
            'bkstpd'      : (str) desc of bktyp_kindd
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkparams = rmn.mrbprm(rpt, iblk+1)
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
        maxblkno  = mrbhdr(rpt)['nblk']  #TODO?: should we do this?
    except:
        maxblkno = -1
    if 0 < maxblkno < blkno:
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
    except KeyError:
        datypd = ''
    params =  {
            'bkno'  : blkno,
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
            'bkstpd'      : (str) desc of bktyp_kindd
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkparams = rmn.mrbprm(rpt, iblk+1)
    ...     params    = rmn.mrbtyp_decode(blkparams['btyp'])
    >>> rmn.burp_close(funit)

    See Also:
        mrbtyp_encode_bknat
        mrbtyp_encode_bktyp
        mrbtyp_encode
        burp_open
        mrfloc
        mrfget
        mrbhdr
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
    params = mrbtyp_decode_bknat(bknat)
    params.update(mrbtyp_decode_bktyp(bktyp))
    params['bkstp'] = bkstp.value
    try:
        params['bkstpd'] = _rbc.BURP_BKSTP_DESC[(
            params['bktyp_alt'], params['bktyp_kind'], params['bkstp']
            )]
    except KeyError:
        params['bkstpd'] = ''
    return params


def mrbtyp_decode_bknat(bknat):
    """
    Decode bknat intp bknat_multi, bknat_kind

    bknat_dict = mrbtyp_decode_bknat(bknat)

    Args:
        bknat : (int) encoded block type, kind component
    Returns:
        {
            'bknat'       : (int) block type, kind component
            'bknat_multi' : (int) block type, kind component, uni/multi bit
                                  0=uni, 1=multi
            'bknat_kind'  : (int) block type, kind component, kind value
                                  See BURP_BKNAT_KIND_DESC
            'bknat_kindd' : (str) desc of bknat_kind
        }
    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> bknatdict = rmn.mrbtyp_decode_bknat(7)
    >>> print('# {} {}'.format(bknatdict['bknat_multi'], bknatdict['bknat_kindd']))
    # 1 flags

    Notes:
        This is a new function in version 2.1.b2

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bknat
        rpnpy.librmn.burp_const
    """
    if isinstance(bknat, _ct.c_int):
        bknat = bknat.value
    bknat_multi = int(_rbc.BURP2BIN(bknat, 4)[0:2], 2)
    bknat_kind  = int(_rbc.BURP2BIN(bknat, 4)[2:], 2)
    try:
        bknat_kindd = _rbc.BURP_BKNAT_KIND_DESC[bknat_kind]
    except KeyError:
        bknat_kindd = ''
    return {
        'bknat'       : bknat,
        'bknat_multi' : bknat_multi,
        'bknat_kind'  : bknat_kind,
        'bknat_kindd' : bknat_kindd,
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

    See Also:
        mrbtyp_decode
        mrbtyp_decode_bknat
        mrbtyp_encode_bktyp
        rpnpy.librmn.burp_const
    """
    #TODO: check bit order in mrbtyp_encode_bknat
    return int(_rbc.BURP2BIN(bknat_multi, 2)+_rbc.BURP2BIN(bknat_kind, 2), 2)


def mrbtyp_decode_bktyp(bktyp):
    """
    Decode bktyp into bktyp_alt, bktyp_kind

    bktyp_dict = mrbtyp_decode_bktyp(bktyp)

    Args:
        bktyp : (int) block type, Data-type component
    Returns:
        {
            'bktyp'       : (int) block type, Data-type component
            'bktyp_alt'   : (int) block type, Data-type component, surf/alt bit
                                  0=surf, 1=alt
            'bktyp_kind'  : (int) block type, Data-type component, flags
                                  See BURP_BKTYP_KIND_DESC
            'bktyp_kindd' : (str) desc of bktyp_kind
        }

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> bktypdict = rmn.mrbtyp_decode_bktyp(0)
    >>> print('# {} {}'.format(bktypdict['bktyp_alt'], bktypdict['bktyp_kindd']))
    # 0 observations (ADE)

    Notes:
        This is a new function in version 2.1.b2

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bktyp
        mrbtyp_encode_bknat
        rpnpy.librmn.burp_const
    """
    if isinstance(bktyp, _ct.c_int):
        bktyp = bktyp.value
    bktyp_alt   = int(_rbc.BURP2BIN(bktyp, 4)[0], 2)
    bktyp_kind  = int(_rbc.BURP2BIN(bktyp, 4)[1:], 2)
    try:
        bktyp_kindd = _rbc.BURP_BKTYP_KIND_DESC[bktyp_kind]
    except KeyError:
        bktyp_kindd = ''
    return {
        'bktyp'       : bktyp,
        'bktyp_alt'   : bktyp_alt,
        'bktyp_kind'  : bktyp_kind,
        'bktyp_kindd' : bktyp_kindd
        }


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

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bknat
        rpnpy.librmn.burp_const
    """
    #TODO: check bit order in mrbtyp_encode_bktyp
    return int(_rbc.BURP2BIN(bktyp_alt, 1) + _rbc.BURP2BIN(bktyp_kind, 6), 2)


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

    See Also:
        mrbtyp_decode
        mrbtyp_encode_bknat
        mrbtyp_encode_bktyp
        rpnpy.librmn.burp_const
    """
    if isinstance(bknat, dict):
        try:
            bktyp = bknat['bktyp']
            bkstp = bknat['bkstp']
            bknat = bknat['bknat']
        except:
            raise BurpError('mrbtyp_encode: must provied all 3 sub values: bknat, bktyp, bkstp',)
    if (bknat < 0 or bktyp < 0 or bkstp < 0 or
        bknat is None or bktyp is None or bkstp is None):
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
    #TODO: check bit order in mrbtyp_encode
    return int("{0:04b}{1:07b}{2:04b}".format(bknat, bktyp, bkstp), 2)


def mrbxtr(rpt, blkno, cmcids=None, tblval=None, dtype=_np.int32):
    """
    Extract block of data from the buffer.
    Also calls mrbprm for metadata

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
            'tblval' : (array) Block data
                               (Shape: NELE, NVAL, NT; type: int)
                               NELE: Number of meteorological elements in block
                               NVAL: Number of values per element.
                               NT  : Nb of groups of NELE x NVAL vals in block.
            'bkno'  : (int) block number
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
            'bkstpd'      : (str) desc of bktyp_kindd
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(rpt, iblk+1)
    >>> ## See mrbdcl, mrbcvt_decode, mrb_prm_xtr_dcl_cvt for how to decode the data
    >>> rmn.burp_close(funit)

    See Also:
        mrb_prm_xtr_dcl_cvt
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbdcl
        mrbcvt_dict
        mrbcvt_decode
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    if blkno <= 0:
        raise ValueError('Provided blkno must be > 0')
    try:
        maxblkno = mrbhdr(rpt)['nblk']  ##TODO?: should we do this?
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
        tblval = _np.empty((nele, nval, nt), dtype=dtype, order='F')
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
        cmcids = _np.empty(nele, dtype=dtype0, order='F')
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(rpt, iblk+1)
    ...     bufrids = rmn.mrbdcl(blkdata['cmcids'])
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
    if isinstance(cmcids, (_integer_types, _np.int32)):
        v = _rp.c_mrbdcv(cmcids)
        return v
    dtype = _np.int32
    if isinstance(cmcids, _np.ndarray):
        if not cmcids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided cmcids should be F_CONTIGUOUS')
        if dtype != cmcids.dtype:
            raise TypeError('Expecting cmcids of type {0}, got: {1}'
                            .format(repr(dtype), repr(cmcids.dtype)))
    elif isinstance(cmcids, (tuple, list)):
        cmcids = _np.array(cmcids, dtype=dtype)
    elif isinstance(cmcids, _integer_types):
        cmcids = _np.array([cmcids], dtype=dtype)
    else:
        raise TypeError('cmcids should be a ndarray of rank 1')
    bufrids = _np.empty(cmcids.size, dtype=dtype, order='F')
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(rpt, iblk+1)
    ...     bufrids = rmn.mrbdcl(blkdata['cmcids'])
    ...     cmcids  = rmn.mrbcol(bufrids)
    >>> rmn.burp_close(funit)

    See Also:
        mrbdcl
        mrbcvt_dict
        mrbhdr
        mrbxtr
    """
    if isinstance(bufrids, (_integer_types, _np.int32)):
        v = _rp.c_mrbcov(bufrids)
        return v
    dtype = _np.int32
    if isinstance(bufrids, _np.ndarray):
        if not bufrids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided bufrids should be F_CONTIGUOUS')
        if dtype != bufrids.dtype:
            raise TypeError('Expecting bufrids of type {0}, got: {1}'
                            .format(repr(dtype), repr(bufrids.dtype)))
    elif isinstance(bufrids, (tuple, list)):
        bufrids = _np.array(bufrids, dtype=dtype)
    elif isinstance(bufrids, _integer_types):
        bufrids = _np.array([bufrids], dtype=dtype)
    else:
        raise TypeError('bufrids should be a ndarray of rank 1')
    cmcids = _np.empty(bufrids.size, dtype=dtype, order='F')
    istat = _rp.c_mrbcol(bufrids, cmcids, bufrids.size)
    if istat != 0:
        raise BurpError('c_mrbcol', istat)
    return cmcids


def mrbcvt_dict_path_set(filepath='', raiseError=False):
    """
    Override default BURP_TABLE_B path/filename and reset the dict content.

    Args:
        filepath   : BURP_TABLE_B path/filename
        raiseError : raise an exception on table decoding error (default: False)
    Returns
        None
    Raises:
        IOError if file not found

    See Also:
        mrbcvt_dict
        mrbcvt_dict_bufr
    """
    if filepath.strip() != '' and not os.path.isfile(filepath):
        raise IOError(" Oops! File does not exist or is not readable: {0}".format(filepath))
    _mrbcvt_dict.update({
        'path'  : filepath.strip(),
        'raise' : raiseError,
        'init'  : False,
        'dict'  : {}
        })


def mrbcvt_dict_get():
    """
    Return a copy decoded BURP_TABLE_B used by mrbcvt

    Args:
        None
    Returns
        dict, decoded BURP_TABLE_B used by mrbcvt
    Raises:
        IOError if file not found

    See Also:
        mrbcvt_dict
        mrbcvt_dict_bufr
    """
    if not _mrbcvt_dict['init']:
        _mrbcvt_dict_full_init()
    return copy.deepcopy(_mrbcvt_dict['dict'])


def _mrbcvt_dict_full_init():
    """
    Read BUFR table B and parse into a dict
    in preparation for use in other functions
    """
    if _mrbcvt_dict['init']:
        return

    mypath = _mrbcvt_dict['path']
    if not mypath:
        AFSISIO = os.getenv('AFSISIO', '')
        mypath = os.path.join(AFSISIO.strip(), 'datafiles/constants',
                              _rbc.BURP_TABLE_B_FILENAME)
        if not (AFSISIO and os.path.isfile(mypath)):
            AFSISIO2 = os.getenv('rpnpy', '/')
            mypath = os.path.join(AFSISIO2.strip(), 'share',
                                  _rbc.BURP_TABLE_B_FILENAME)

    try:
        ## sys.stderr.write('_mrbcvt_dict_full_init: '+mypath+"\n") #TODO: print this in verbose mode
        fd = open(mypath, "r")
        try: rawdata = fd.readlines()
        finally: fd.close()
    except IOError:
        raise IOError(" Oops! File does not exist or is not readable: {0}".format(mypath))

    hasError = False
    for item in rawdata:
        if item[0] == '*' or len(item.strip()) == 0:
            continue
        try:
            id1 = int(item[0:6])
            d = {
                'e_error'   : 0,
                'e_cmcid'   : mrbcol(id1),
                'e_bufrid'  : id1,
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
                d['e_cvt'] = 0
                d['e_desc'] = item[8:50].strip()
            ## elif d['e_units'] in ('CODE TABLE', 'FLAG TABLE', 'NUMERIC'):
            elif d['e_units'] in ('CODE TABLE', 'FLAG TABLE'):  #TODO: check if NUMERIC should be included
                d['e_cvt'] = 0
            if len(item) > 84 and item[84] == 'M':
                d['e_multi'] = 1
            _mrbcvt_dict['dict'][id1] = d
        except:
            if not hasError:
                hasError = True
                sys.stderr.write("WARNING: mrbcvt_dict_full_init - problem decoding line in file: {}\n".format(mypath))
            sys.stderr.write("WARNING, offending line: {}\n".format(item.strip()))
            if _mrbcvt_dict['raise']:
                raise
    _mrbcvt_dict['init'] = True


def mrbcvt_dict_find_id(desc, nmax=999, flags=_re.IGNORECASE):
    """
    Find bufrid matching description in BUFR table B

    bufrid = mrbcvt_dict_find_id(desc)

    Args:
        desc : (str) Description of the BUFR elem, can use a regexp
        nmax : (int) Max number of matching desc (default=999)
        flags: re.match flags (default=re.IGNORECASE)
    Returns
        e_bufrid  : (list of int) Element BUFR code as found in BUFR table B

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> bufridlist = rmn.mrbcvt_dict_find_id('.*ground\s+temperature.*')

    Notes:
        This is a new function in version 2.1.b2

    See Also:
        mrbcvt_dict
        mrbcvt_dict_bufr
    """
    if not _mrbcvt_dict['init']:
        _mrbcvt_dict_full_init()
    e_bufrid = []
    for k, v in _mrbcvt_dict['dict'].items():
        if _re.match(desc, v['e_desc'], flags):
            e_bufrid.append(v['e_bufrid'])
        if len(e_bufrid) >= nmax:
            break
    return e_bufrid


def mrbcvt_dict_bufr(bufrid, raise_error=True, cmcid=None):
    """
    Extract BUFR table B info for bufrid

    cvtdict = mrbcvt_dict_bufr(bufrid)

    Args:
        bufrid      : Element BUFR code name
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
    >>> bufrid = 10031
    >>> cvtdict = rmn.mrbcvt_dict_bufr(bufrid, raise_error=False)
    >>> # print('{e_bufrid:0>6} {e_desc} [{e_units}]'.format(**cvtdict))

    Notes:
        This is a new function in version 2.1.b2

    See Also:
        mrbcvt_dict
        mrbcvt_dict_find_id
        mrfget
        mrbdcl
        mrbcol
        mrbxtr
        mrbhdr
        mrfget
        mrfloc
        burp_open
    """
    if not _mrbcvt_dict['init']:
        _mrbcvt_dict_full_init()
    if not cmcid:
        cmcid = mrbcol(bufrid)
    try:
        return _mrbcvt_dict['dict'][bufrid]
    except KeyError:
        if raise_error:
            raise
        else:
            id1 = "{0:0>6}".format(bufrid)
            return {
                'e_error'   : -1,
                'e_cmcid'   : cmcid,
                'e_bufrid'  : bufrid,
                'e_bufrid_F': int(id1[0]),
                'e_bufrid_X': int(id1[1:3]),
                'e_bufrid_Y': int(id1[3:6]),
                'e_cvt'     : 0,
                'e_desc'    : '',
                'e_units'   : '',
                'e_scale'   : 0,
                'e_bias'    : 0,
                'e_nbits'   : 0,
                'e_multi'   : 0
                }

def mrbcvt_dict(cmcid, raise_error=True):
    """
    Extract BUFR table B info for cmcid

    cvtdict = mrbcvt_dict(cmcid)

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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(rpt, iblk+1)
    ...     for cmcid in blkdata['cmcids']:
    ...         try:
    ...             cvtdict = rmn.mrbcvt_dict(cmcid)
    ...             # print('{e_bufrid:0>6} {e_desc} [{e_units}]'.format(**cvtdict))
    ...         except:
    ...             pass  # Description not found
    >>> rmn.burp_close(funit)

    See Also:
        mrbcvt_dict_bufr
        mrbcvt_dict_find_id
        mrfget
        mrbdcl
        mrbcol
        mrbxtr
        mrbhdr
        mrfget
        mrfloc
        burp_open
    """
    bufrid = mrbdcl(cmcid)
    return mrbcvt_dict_bufr(bufrid, raise_error, cmcid)


def mrbcvt_decode(cmcids, tblval=None, datyp=_rbc.BURP_DATYP_LIST['float']):
    """
    Convert table/BUFR values to real values.

    rval = mrbcvt_decode(cmcids, tblval)
    rval = mrbcvt_decode(cmcids, tblval, datyp)
    rval = mrbcvt_decode(blkdata)
    rval = mrbcvt_decode(blkdata, datyp)

    Args:
        cmcids : List of element names in the report in numeric BUFR codes.
                 See the code desc in the FM 94 BUFR man
        tblval : BUFR code values (array of int or float)
                 Note: tblval is modified by mrbcvt_decode for negative values
                       where(tblval < 0) tblval += 1
        datyp' : (optional) Data type as obtained from mrbprm (int)
                 See rpnpy.librmn.burp_const BURP_DATYP_LIST
                                         and BURP_DATYP2NUMPY_LIST
                 Default: 6 (float)
        blkdata: (dict) Block data as returned by mrbxtr,
                           must contains 2 keys: 'cmcids', 'tblval'
                 Note: tblval is modified by mrbcvt_decode for negative values
                       where(tblval < 0) tblval += 1
    Returns
        array, dtype depends on datyp, converted tblval to rval
               missing values will be set to mrfopt("MISSING")
               not convertable values will be copied from tblval to rval
    Raises:
        KeyError   on missing blkdata keys
        TypeError  on wrong input arg types
        ValueError on wrong input arg value
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata  = rmn.mrbxtr(rpt, iblk+1)
    ...     rval     = rmn.mrbcvt_decode(blkdata)
    >>> rmn.burp_close(funit)

    See Also:
        mrbcvt_encode
        mrfopt
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
                            .format(repr(dtype), repr(cmcids.dtype)))
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
    ## Notes: Type 3 (char) and 5 (upchar) are processed like strings of bits
    ##        thus, the user should do the data compression himself.

    ## *       LA MATRICE TABLEAU CONTIENT:
    ## *          TABLEAU(1,I) - CODE DE L'ELEMENT I
    ## *          TABLEAU(2,I) - FACTEUR A APPLIQUER A L'ELEMENT I
    ## *          TABLEAU(3,I) - VALEUR DE REFERENCE A AJOUTER A L'ELEMENT I
    ## *       LA VARIABLE NELELU INDIQUE LE NOMBRE D'ELEMENTS PRESENT DANS LE
    ## *       FICHIER BUFR
    ## *
    ## *       POUR CODER LA VALEUR D'UN ELEMENT AVANT UN APPEL A MRFPUT, ON FAIT
    ## *       L'OPERATION SUIVANTE:
    ## *       ELEMENT(I)_CODE = ELEMENT(I) * TABLEAU(2,I) - TABLEAU(3,I)
    ## *
    ## *       ON NE FAIT AUCUNE CONVERSION LORSQUE QU'UN ELEMENT EST DEJA CODE
    ## *       COMME PAR EXEMPLE POUR LES DIFFERENTS MARQUEURS.
    ## *
    ## *       POUR DECODER UN ELEMENT ON FAIT L'OPERATION INVERSE.  DANS LE CAS
    ## *       DES ELEMENTS NE REQUERANT AUCUN DECODAGE (E.G. MARQUEURS), ON INSERE
    ## *       DANS LE TABLEAU RVAL LA VALEUR -1.1E30 CE QUI INDIQUE A L'USAGER
    ## *       QU'IL DOIT CONSULTER LE TABLEAU TBLVAL POUR OBTENIR CET ELEMEMT


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
    elif _rbc.BURP_DATYP_NAMES[datyp] in ('double', ):
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

    dtype = _np.float32 ##_rbc.BURP_DATYP2NUMPY_LIST[datyp]
    rval = tblval.astype(dtype)

    ## try:
    ##     rval_missing = _rbc.BURP_RVAL_MISSING[datyp]
    ## except KeyError:
    ##     rval_missing = _rbc.BURP_RVAL_MISSING0  ##TODO: check: _rbc.BURP_TBLVAL_MISSING
    ## rval[:, :, :] = rval_missing
    istat = _rp.c_mrbcvt(cmcids, tblval, rval, nele, nval, nt,
                         _rbc.MRBCVT_DECODE)
    if istat != 0:
        raise BurpError('c_mrbcvt', istat)
    ## try:
    ##     rval_missing = _rbc.BURP_RVAL_MISSING[datyp]
    ## except:
    ##     rval_missing = _rbc.BURP_RVAL_MISSING0  ##TODO: check: _rbc.BURP_TBLVAL_MISSING
    ## rval[tblval == _rbc.BURP_TBLVAL_MISSING] = rval_missing

    return rval


#TODO: add optional args?
## def mrb_hdr_prm_xtr_dcl_cvt(rpt, blkno, cmcids=None, tblval=None, rval=None, dtype=_np.int32):
        ## cmcids, tblval: (optional) return data arrays
        ## dtype : (optional) numpy type for tblval creation, if tblval is None
def mrb_prm_xtr_dcl_cvt(rpt, blkno):
    """
    Extract block of data from the buffer and decode its values
    Calls mrbprm, mrbxtr, mrbdcl, mrbcvt_decode

    blkdata = mrb_prm_xtr_dcl_cvt(rpt, blkno)

    Args:
        rpt   : Report data  (array)
        blkno : block number (int > 0)
    Returns
        {
            'bkno'  : (int) block number
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
            'bkstpd'      : (str) desc of bktyp_kindd
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
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrb_prm_xtr_dcl_cvt(rpt, iblk+1)
    >>> rmn.burp_close(funit)

    See Also:
        mrfmxl
        mrfloc
        mrfget
        mrbhdr
        mrbprm
        mrbxtr
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
        blkdata['rval'] = blkdata['tblval'][:, :, :]
    return blkdata


def mrbcvt_encode(cmcids, rval):
    """
    Convert real values to table/BUFR values.

    tblval = mrbcvt_encode(cmcids, rval)
    tblval = mrbcvt_encode(blkdata)

    Args:
        cmcids : List of element names in the report in numeric BUFR codes.
                 See the code desc in the FM 94 BUFR man
        rval   : Real-valued table data
                 nint(rval) is used as tblval for mrbcvt_encode for not
                 converted values
        blkdata : (dict) Block data as returned by mrbxtr,
                           must contains 2 keys: 'cmcids', 'rval'
    Returns
        array, integer table data, converted rval to tblval
               missing values equielent to mrfopt("MISSING") will
               be properly encoded
               not convertable values will be copied from rval to tblval
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> funit  = rmn.burp_open(filename)
    >>> handle = rmn.mrfloc(funit)
    >>> rpt    = rmn.mrfget(handle, funit=funit)
    >>> params = rmn.mrbhdr(rpt)
    >>> for iblk in range(params['nblk']):
    ...     blkdata  = rmn.mrbxtr(rpt, iblk+1)
    ...     tblval0  = blkdata['tblval'].copy()
    ...     rval     = rmn.mrbcvt_decode(blkdata)
    ...     tblval   = rmn.mrbcvt_encode(blkdata['cmcids'], rval)
    ...     if not np.all(tblval == tblval0):
    ...        print("Problem encoding rval to tblval")
    >>> rmn.burp_close(funit)

    See Also:
        mrbcvt_decode
        mrfopt
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
            rval   = cmcids['rval']
            cmcids = cmcids['cmcids']
        except:
            raise KeyError('Provided blkdata should have these 2 keys: cmcids, rval')
    if isinstance(rval, _np.ndarray):
        if not rval.flags['F_CONTIGUOUS']:
            raise TypeError('Provided rval should be F_CONTIGUOUS')
        if len(rval.shape) != 3:
            raise TypeError('Provided rval should be an ndarray of rank 3')
    else:
        raise TypeError('Provided rval should be an ndarray of rank 3')
    (nele, nval, nt) = rval.shape
    dtype = _np.int32
    if isinstance(cmcids, _np.ndarray):
        if not cmcids.flags['F_CONTIGUOUS']:
            raise TypeError('Provided cmcids should be F_CONTIGUOUS')
        if dtype != cmcids.dtype:
            raise TypeError('Expecting cmcids of type {0}, got: {1}'
                            .format(repr(dtype), repr(cmcids.dtype)))
        if len(cmcids.shape) != 1:
            raise TypeError('cmcids should be a ndarray of rank 1, ' +
                            'got: shape= {}'.format(cmcids.shape))
        if cmcids.size != nele:
            raise TypeError('cmcids should be the size of ' +
                            'nele={}, got: size={}'
                            .format(nele, cmcids.size))
    else:
        raise TypeError('cmcids should be a ndarray of rank 1 (nele), got: {}'
                        .format(type(cmcids)))

    tblval = _np.round(rval).astype(_np.int32)

    istat = _rp.c_mrbcvt(cmcids, tblval, rval, nele, nval, nt,
                         _rbc.MRBCVT_ENCODE)

    if istat != 0:
        raise BurpError('c_mrbcvt', istat)

    ## try:
    ##     datyp = None
    ##     for k in _rbc.BURP_DATYP2NUMPY_LIST.keys():
    ##         ## print _rbc.BURP_DATYP2NUMPY_LIST[k] == rval.dtype, repr(_rbc.BURP_DATYP2NUMPY_LIST[k]), repr(rval.dtype)
    ##         if _rbc.BURP_DATYP2NUMPY_LIST[k] == rval.dtype:
    ##             datyp = k
    ##             break
    ##     if datyp is None:
    ##         rval_missing = _rbc.BURP_RVAL_MISSING0
    ##     else:
    ##         rval_missing = _rbc.BURP_RVAL_MISSING[datyp]
    ## except:
    ##     rval_missing = _rbc.BURP_RVAL_MISSING0  ##TODO: check: _rbc.BURP_TBLVAL_MISSING
    ## rval_missing = _rbc.BURP_RVAL_MISSING0  #TODO: if rval is actually int or char that may not be the proper missing val
    #### tblval[rval == rval_missing] = _rbc.BURP_TBLVAL_MISSING
    ## print (rval == rval_missing).ravel()
    ## print rval.ravel(), rval_missing, rval.dtype
    ## print tblval.ravel(), _rbc.BURP_TBLVAL_MISSING, tblval.dtype
    #TODO: tblval[_np.isnan(rval)] = _rbc.BURP_TBLVAL_MISSING

    return tblval


#TODO?: remove sup, nsup, xaux, nxaux from itf since not supported
def mrbini(funit, rpt, time=None, flgs=None, stnid=None, idtyp=None, ilat=None,
           ilon=None, idx=None, idy=None, ielev=None, drnd=None, date=None,
           oars=None, runn=None, sup=None, nsup=0, xaux=None, nxaux=0):
    """
    Writes report header.

    Similar to inverse mrbhdr operation.

    rpt = mrbini(funit, rpt, time, flgs, stnid, idtp, lat,
                 lon, dx, dy, elev, drnd, date, oars, runn)
    rpt = mrbini(funit, rptdict)

    Args:
        funit : (int)   File unit number
        rpt   : (array) vector to contain the report
                (int)   or max report size in file
        time  : (int)   Observation time/hour (HHMM)
        flgs  : (int)   Global flags
                        (24 bits, Bit 0 is the right most bit of the word)
                        See BURP_FLAGS_IDX_NAME for Bits/flags desc.
        stnid : (str)   Station ID
                        If it is a surface station, STNID = WMO number.
                        The name is aligned at left and filled with
                        spaces. In the case of regrouped data,
                        STNID contains blanks.
        idtyp : (int)   Report Type
        ilat  : (int)   Station latitude (1/100 of degrees)
                        with respect to the south pole. (0 to 1800)
                        (100*(latitude+90)) of a station or the
                        lower left corner of a box.
        ilon  : (int)   Station longitude (1/100 of degrees)
                        (0 to 36000) of a station or lower left corner
                        of a box.
        idx   : (int)   Width of a box for regrouped data
                        (delta lon, 1/10 of degrees)
        idy   : (int)   Height of a box for regrouped data
                        (delta lat, 1/10 of degrees)
        ielev : (int)   Station altitude (metres + 400.) (0 to 8191)
        drnd  : (int)   Reception delay: difference between the
                        reception time at CMC and the time of observation
                        (TIME). For the regrouped data, DRND indicates
                        the amount of data. DRND = 0 in other cases.
        date  : (int)   Report valid date (YYYYMMDD)
        oars  : (int)   Reserved for the Objective Analysis. (0-->65535)
        runn  : (int)   Operational pass identification.
        sup   : (array) supplementary primary keys array
                        (reserved for future expansion).
        nsup  : (int)   number of sup
        xaux  : (array) supplementary auxiliary keys array
                        (reserved for future expansion).
        nxaux : (int)   number of xaux
        rptdict : above args given as a dictionary (dict)
    Returns
        rpt : (array) vector containing the report
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> # Read and copy all blocks from one BURP rpt
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> ifname = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> ofname = 'newfile.brp'
    >>> ifunit = rmn.burp_open(ifname, rmn.BURP_MODE_READ)
    >>> ofunit = rmn.burp_open(ofname, rmn.BURP_MODE_CREATE)
    >>> ihandle= rmn.mrfloc(ifunit)
    >>> irpt   = rmn.mrfget(ihandle, funit=ifunit)
    >>> params = rmn.mrbhdr(irpt)
    >>> params['rpt'] = irpt.size
    >>> orpt = rmn.mrbini(ofunit, params)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(irpt, iblk+1)  # mrbxtr() calls mrbprm()
    ...     blkno = rmn.mrbadd(orpt, blkdata)
    >>> ohandle = 0
    >>> rmn.mrfput(ofunit, ohandle, orpt)
    >>> rmn.burp_close(ofunit)
    >>> rmn.burp_close(ifunit)
    >>> os.unlink(ofname)  # Remove test file

    See Also:
        mrbhdr
        mrbadd
        mrbput
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    time = _getCheckArg(int, time, rpt, 'time')
    flgs = _getCheckArg(int, flgs, rpt, 'flgs')
    stnid = _getCheckArg(str, stnid, rpt, 'stnid')
    idtyp = _getCheckArg(int, idtyp, rpt, 'idtyp')
    ilat = _getCheckArg(int, ilat, rpt, 'ilat')
    ilon = _getCheckArg(int, ilon, rpt, 'ilon')
    idx = _getCheckArg(int, idx, rpt, 'idx')
    idy  = _getCheckArg(int, idy, rpt, 'idy')
    ielev = _getCheckArg(int, ielev, rpt, 'ielev')
    drnd = _getCheckArg(int, drnd, rpt, 'drnd')
    date = _getCheckArg(int, date, rpt, 'date')
    oars = _getCheckArg(int, oars, rpt, 'oars')
    runn = _getCheckArg(int, runn, rpt, 'runn')
    ## sup = _getCheckArg(None, sup, rpt, 'sup')
    ## nsup = _getCheckArg(int, nsup, rpt, 'nsup')
    ## xaux = _getCheckArg(None, xaux, rpt, 'xaux')
    ## nxaux = _getCheckArg(int, nxaux, rpt, 'nxaux')
    rpt = _getCheckArg(None, rpt, rpt, 'rpt')

    sup, nsup, xaux, nxaux = (None, 0 , None, 0)
    if sup is None:
        sup = _np.empty((1, ), dtype=_np.int32)
    if xaux is None:
        xaux = _np.empty((1, ), dtype=_np.int32)

    if isinstance(rpt, _integer_types):
        nrpt = rpt
        ## nrpt *= 2  #TODO?: should we do this?
        rpt = _np.empty((nrpt,), dtype=_np.int32)
        rpt[0] = nrpt
    elif not isinstance(rpt, _np.ndarray):
        raise TypeError('rpt should be an ndarray')

    #TODO: if float values are given instead of int... convert
    ## 'lat'   : (float) Station latitude (degrees)
    ## 'lon'   : (float) Station longitude (degrees)
    ## 'dx'    : (float) Width of a box for regrouped data (degrees)
    ## 'dy'    : (float) Height of a box for regrouped data (degrees)
    ## 'elev'  : (float) Station altitude (metres)

    istat = _rp.c_mrbini(funit, rpt, time, flgs, _C_WCHAR2CHAR(stnid),
                         idtyp, ilat, ilon, idx, idy, ielev, drnd, date, oars,
                         runn, sup, nsup,xaux, nxaux)
    if istat != 0:
        raise BurpError('c_mrbini', istat)
    return rpt


#TODO? change cmcids for consistency (more explict name)
def mrbadd(rpt, nele, nval=None, nt=None, bfam=None, bdesc=None,
           btyp=None, nbit=None, datyp=None,
           cmcids=None, tblval=None):
    """
    Adds a block to a report.

    Similar to inverse mrbxtr/mrbprm operation.

    blkno = mrbadd(rpt, nele, nval, nt, bfam, bdesc, btyp, nbit,
                   datyp, cmcids, tblval)
    blkno = mrbadd(rpt, blkdata)

    Args:
        rpt    (array) : vector to contain the report to update
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
        blkdata        : above args given as a dictionary (dict)
    Returns
        blkno  (int)   : block number
                         Note: rpt is updated in place (not returned)
    Raises:
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> # Read and copy all blocks from one BURP rpt
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> ifname = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>> ofname = 'newfile.brp'
    >>> ifunit = rmn.burp_open(ifname, rmn.BURP_MODE_READ)
    >>> ofunit = rmn.burp_open(ofname, rmn.BURP_MODE_CREATE)
    >>> ihandle= rmn.mrfloc(ifunit)
    >>> irpt   = rmn.mrfget(ihandle, funit=ifunit)
    >>> params = rmn.mrbhdr(irpt)
    >>> params['rpt'] = irpt.size
    >>> orpt = rmn.mrbini(ofunit, params)
    >>> for iblk in range(params['nblk']):
    ...     blkdata = rmn.mrbxtr(irpt, iblk+1)  # mrbxtr() calls mrbprm()
    ...     blkno = rmn.mrbadd(orpt, blkdata)
    >>> ohandle = 0
    >>> rmn.mrfput(ofunit, ohandle, orpt)
    >>> rmn.burp_close(ofunit)
    >>> rmn.burp_close(ifunit)
    >>> os.unlink(ofname)  # Remove test file

    See Also:
        mrbhdr
        mrbini
        mrbxtr
        mrbput
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    nval = _getCheckArg(int, nval, nele, 'nval')
    nt = _getCheckArg(int, nt, nele, 'nt')
    bfam = _getCheckArg(int, bfam, nele, 'bfam')
    bdesc = _getCheckArg(int, bdesc, nele, 'bdesc')
    btyp = _getCheckArg(int, btyp, nele, 'btyp')
    nbit = _getCheckArg(int, nbit, nele, 'nbit')
    datyp = _getCheckArg(int, datyp, nele, 'datyp')
    cmcids = _getCheckArg(None, cmcids, nele, 'cmcids')
    tblval = _getCheckArg(None, tblval, nele, 'tblval')
    nele = _getCheckArg(int, nele, nele, 'nele')
    rpt = _getCheckArg(_np.ndarray, rpt, rpt, 'rpt')

    cmcids = _list2ftnf32(cmcids)
    tblval = _list2ftnf32(tblval)

    blkno = _ct.c_int()
    bit0 = _ct.c_int()

    istat = _rp.c_mrbadd(rpt, _ct.byref(blkno), nele, nval, nt, bfam, bdesc,
                         btyp, nbit, _ct.byref(bit0), datyp, cmcids, tblval)
    if istat != 0:
        raise BurpError('c_mrbadd', istat)
    return blkno


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

    See Also:
        mrbadd
        mrfdel
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    blkno = _getCheckArg(int, blkno, blkno, 'blkno')
    rpt = _getCheckArg(_np.ndarray, rpt, rpt, 'rpt')
    istat = _rp.c_mrbdel(rpt, blkno)
    if istat != 0:
        raise BurpError('c_mrbdel', istat)
    return rpt


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

    See Also:
        mrbdel
        mrbput
        burp_open
        burp_close
        rpnpy.librmn.burp_const
    """
    handle = _getCheckArg(int, handle, handle, 'handle')
    istat = _rp.c_mrfdel(handle)
    if istat != 0:
        raise BurpError('c_mrfdel', istat)
    return

##TODO: mrfrwd
##TODO: mrfapp

# =========================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
