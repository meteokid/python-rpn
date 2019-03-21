#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn.base contains python wrapper to
main librmn, base and primitives C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[librmn]]'s [[Librmn/FSTDfunctions|FSTD]] package.<br>
    You may want to refer to the [[Librmn/FSTDfunctions|FSTD]]
    documentation for more details.

See Also:
    rpnpy.librmn.fstd98
    rpnpy.librmn.interp
    rpnpy.librmn.grids
    rpnpy.librmn.const
"""

import ctypes as _ct
import numpy  as _np
import zlib   as _zl
from rpnpy.librmn import proto as _rp
from rpnpy.librmn import const as _rc
from rpnpy.librmn import RMNError
from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR as _C_MKSTR

## _C_MKSTR = lambda x: _ct.create_string_buffer(x)
## _C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer'

_C_TOINT = lambda x: (x if (type(x) != type(_ct.c_int())) else x.value)
_C_TOINT.__doc__ = 'lamda function to convert ctypes.c_int to python int'

_IS_LIST = lambda x: isinstance(x, (list, tuple))
_IS_LIST.__doc__ = 'lambda function to test if x is list or tuple'

class RMNBaseError(RMNError):
    """
    General librmn.base module error/exception

    To make your code handle errors in an elegant manner,
    you may want to catch that error with a 'try ... except' block.

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    ...     xg1234 = rmn.cigaxg('E', 0, 0, 0, 0)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem getting decoded grid values.")

    See also:
        rpnpy.librmn.RMNError
    """
    pass


#--- primitives -----------------------------------------------------

def get_funit(filename, filemode=_rc.FST_RW, iunit=0):
    """
    Get a semi-reserved file unit to open with another function

    funit = get_unit(filename, filemode, iunit=0)

    Args:
        filename : path/name of the file to open
        filemode : a string with the desired filemode (see librmn doc)
                   or one of these constants: FST_RW, FST_RW_OLD, FST_RO
        iunit    : forced unit number to conect to
                   if zero, will select a free unit
    Returns:
        int, Associated file unit number
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Notes:
        New function in version 2.1.b2

    See also:
        fnom
        fclos
    """
    iunit = 0 if iunit is None else iunit
    funit = fnom(filename, filemode, iunit)
    fclos(funit)  #TODO: too hacky... any way to reserve a unit w/o double open?
    return funit


def fclos(iunit):
    """
    Close file associated with unit through fnom

    Args:
        iunit   : unit number associated to the file
                  obtained with fnom or fstopenall
    Returns:
        0 on succes
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Examples:
    >>> import os, sys
    >>> import rpnpy.librmn.all as rmn
    >>> filename = 'myfstfile.fst'
    >>> try:
    ...     iunit = rmn.fnom(filename, rmn.FST_RW)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem opening the file: {0}".format(filename))
    >>> istat = rmn.fclos(iunit)
    >>> os.unlink(filename)  # Remove test file

    See also:
       fnom
       rpnpy.librmn.fstd98.fstopenall
       rpnpy.librmn.fstd98.fstcloseall
    """
    if not isinstance(iunit, _integer_types):
        raise TypeError("fcols: Expecting arg of type int, Got {0}"\
                        .format(type(iunit)))
    if iunit < 0:
        raise ValueError("fclos: must provide a valid iunit: {0}".format(iunit))
    istat = _rp.c_fclos(iunit)
    if istat < 0:
        raise RMNBaseError()
    return istat


def fnom(filename, filemode=_rc.FST_RW, iunit=0, legacy=False):
    """
    Open a file and make the connection with a unit number.

    Args:
        filename : path/name of the file to open
        filemode : a string with the desired filemode (see librmn doc)
                   or one of these constants: FST_RW, FST_RW_OLD, FST_RO
        iunit    : forced unit number to conect to
                   if zero, will select a free unit
        legacy   : fall back to legacy fnom mode for filenames if True
    Returns:
        int, Associated file unit number
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Examples:
    >>> import os, sys
    >>> import rpnpy.librmn.all as rmn
    >>> filename = 'myfstfile.fst'
    >>> try:
    ...     iunit = rmn.fnom(filename, rmn.FST_RW)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem opening the file: {0}".format(filename))
    >>> istat = rmn.fclos(iunit)
    >>> os.unlink(filename)  # Remove test file

    See also:
       fclos
       rpnpy.librmn.fstd98.isFST
       rpnpy.librmn.fstd98.fstopenall
       rpnpy.librmn.fstd98.fstcloseall
       rpnpy.librmn.const
    """
    if not isinstance(iunit, _integer_types):
        raise TypeError("fnom: Expecting arg of type int, Got {0}"\
                        .format(type(iunit)))
    ciunit = _ct.c_int(max(0, iunit))
    if not isinstance(iunit, _integer_types):
        raise TypeError("fnom: Expecting arg of type int, Got {0}"\
                        .format(type(iunit)))
    if not isinstance(filename, str):
        raise TypeError("fnom: Expecting filename arg of type str, Got {0}"\
                        .format(type(filename)))
    if filename.strip() == '':
        raise ValueError("fnom: must provide a valid filename")
    if not isinstance(filemode, str):
        raise TypeError("fnom: Expecting arg filemode of type str, Got {0}"\
                        .format(type(filemode)))
    # Prepend filename with '+' to tell librmn to preserve filename case.
    if not (legacy or filename.startswith('+')):
      filename = '+'+filename
    istat = _rp.c_fnom(_ct.byref(ciunit), _C_WCHAR2CHAR(filename),
                       _C_WCHAR2CHAR(filemode), 0)
    istat = _C_TOINT(istat)
    if istat < 0:
        raise RMNBaseError()
    return ciunit.value


def wkoffit(filename, legacy=False):
    """
    Return code type of file (int)

    Args:
        filename : path/name of the file to examine
        legacy   : fall back to legacy fnom mode for filenames if True
    Returns:
        int, file type code as follow:
          -3     FICHIER INEXISTANT
          -2     FICHIER VIDE
          -1     FICHIER INCONNU
           1     FICHIER STANDARD RANDOM 89
           2     FICHIER STANDARD SEQUENTIEL 89
           3     FICHIER STANDARD SEQUENTIEL FORTRAN 89
           4     FICHIER CCRN
           5     FICHIER CCRN-RPN
           6     FICHIER BURP
           7     FICHIER GRIB
           8     FICHIER BUFR
           9     FICHIER BLOK
          10     FICHIER FORTRAN
          11     FICHIER COMPRESS
          12     FICHIER GIF89
          13     FICHIER GIF87
          14     FICHIER IRIS
          15     FICHIER JPG
          16     FICHIER KMW
          17     FICHIER PBM
          18     FICHIER PCL
          19     FICHIER PCX
          20     FICHIER PDSVICAR
          21     FICHIER PM
          22     FICHIER PPM
          23     FICHIER PS
          24     FICHIER KMW_
          25     FICHIER RRBX
          26     FICHIER SUNRAS
          27     FICHIER TIFF
          28     FICHIER UTAHRLE
          29     FICHIER XBM
          30     FICHIER XWD
          31     FICHIER ASCII
          32     FICHIER BMP
          33     FICHIER STANDARD RANDOM 98
          34     FICHIER STANDARD SEQUENTIEL 98
          35     FICHIER NETCDF
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import os, os.path, sys
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> itype = rmn.wkoffit(filename)
    >>> if itype in rmn.WKOFFIT_TYPE_LIST_INV.keys():
    ...     print('# '+rmn.WKOFFIT_TYPE_LIST_INV[itype])
    # STANDARD RANDOM 98

    See also:
       rpnpy.librmn.fstd98.isFST
       rpnpy.librmn.const
    """
    if not isinstance(filename, str):
        raise TypeError("wkoffit: Expecting filename arg of type str, " +
                        "Got {0}".format(type(filename)))
    if filename.strip() == '':
        raise ValueError("wkoffit: must provide a valid filename")
    # Prepend filename with '+' to tell librmn to preserve filename case.
    if not (legacy or filename.startswith('+')):
      filename = '+'+filename
    return _rp.c_wkoffit(_C_WCHAR2CHAR(filename), len(filename))


def crc32(crc, buf):
    """
    Compute the Cyclic Redundancy Check (CRC)

    Args:
       crc : initial crc value (int)
       buf : list of number to compute updated crc (numpy.ndarray of uint32)
    Returns:
       crc : computed crc value (int)
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import sys
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>> buf = np.array([4,3,7,1,9], dtype=np.uint32)
    >>> try:
    ...     crc = rmn.crc32(0,buf)
    ... except:
    ...     sys.stderr.write("There was a problem computing CRC value.")
    """
    if not (buf.dtype == _np.uint32 and buf.flags['F_CONTIGUOUS']):
        buf = _np.asfortranarray(buf, dtype=_np.uint32)
    return _zl.crc32(buf, crc) & 0xffffffff

#--- base -----------------------------------------------------------


def cigaxg(grtyp, ig1, ig2=0, ig3=0, ig4=0):
    """
    Decode ig1, ig2, ig3, ig4 into real grid descriptors

    (xg1, xg2, xg3, xg4) = cigaxg(grtyp, ig1, ig2, ig3, ig4)
    (xg1, xg2, xg3, xg4) = cigaxg(grtyp, ig1234)

    Args:
        grtyp  : type of geographical projection (str)
        ig1..4 : 4 grid descriptors encoded values (4x int)
        ig1234 : 4 grid descriptors encoded values (tuple or list of 4x int)
    Returns:
        (float, float, float, float), Decoded grid parameters
        Meaning of xg1..4 values depends on the grid type,
        please refer to Librmn doc on grids for more details
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    ...     xg1234 = rmn.cigaxg('E', 0, 0, 0, 0)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem getting decoded grid values.")

    See also:
       cxgaig
       rpnpy.librmn.interp.ezgprm
       rpnpy.librmn.interp.ezgxprm
       rpnpy.librmn.grids.decodeIG2dict
       rpnpy.librmn.grids.decodeXG2dict
       rpnpy.librmn.grids.decodeGrid
       rpnpy.librmn.grids.encodeGrid
    """
    if not isinstance(grtyp, str):
        raise TypeError("cigaxg: Expecting grtyp arg of type str, Got {0}"\
                        .format(type(grtyp)))
    if grtyp.strip() == '':
        raise ValueError("cigaxg: must provide a valid grtyp")
    if _IS_LIST(ig1):
        (cig1, cig2, cig3, cig4) = (_ct.c_int(ig1[0]), _ct.c_int(ig1[1]),
                                    _ct.c_int(ig1[2]), _ct.c_int(ig1[3]))
    else:
        (cig1, cig2, cig3, cig4) = (_ct.c_int(ig1), _ct.c_int(ig2),
                                    _ct.c_int(ig3), _ct.c_int(ig4))
    (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(0.), _ct.c_float(0.),
                                _ct.c_float(0.), _ct.c_float(0.))
    _rp.f_cigaxg(_C_WCHAR2CHAR(grtyp),
                _ct.byref(cxg1),_ct.byref(cxg2),
                _ct.byref(cxg3),_ct.byref(cxg4),
                _ct.byref(cig1),_ct.byref(cig2),
                _ct.byref(cig3),_ct.byref(cig4))
    return (cxg1.value, cxg2.value, cxg3.value, cxg4.value)


def cxgaig(grtyp, xg1, xg2=0., xg3=0., xg4=0.):
    """
    Encode real grid descriptors into ig1, ig2, ig3, ig4

    (ig1, ig2, ig3, ig4) = cxgaig(grtyp, xg1, xg2, xg3, xg4)
    (ig1, ig2, ig3, ig4) = cxgaig(grtyp, xg1234)

    Args:
        grtyp  : type of geographical projection (str)
        xg1..4 : 4 grid descriptors values (4x float)
        xg1234 : 4 grid descriptors values (tuple or list of 4x float)
                 Meaning of xg1..4 values depends on the grid type,
                 please refer to Librmn doc on grids for more details
    Returns:
        (int, int, int, int), Encoded grid parameters
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    ...     ig1234 = rmn.cxgaig('L', -89.5, 180.0, 0.5, 0.5)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem getting encoded grid values.")

    See also:
       cigaxg
       rpnpy.librmn.interp.ezgprm
       rpnpy.librmn.interp.ezgxprm
       rpnpy.librmn.grids.decodeIG2dict
       rpnpy.librmn.grids.decodeXG2dict
       rpnpy.librmn.grids.decodeGrid
       rpnpy.librmn.grids.encodeGrid
    """
    if not isinstance(grtyp, str):
        raise TypeError("cigaxg: Expecting grtyp arg of type str, Got {0}"\
                        .format(type(grtyp)))
    if grtyp.strip() == '':
        raise ValueError("cigaxg: must provide a valid grtyp")
    if _IS_LIST(xg1):
        (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(xg1[0]), _ct.c_float(xg1[1]),
                                    _ct.c_float(xg1[2]), _ct.c_float(xg1[3]))
    else:
        (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(xg1), _ct.c_float(xg2),
                                    _ct.c_float(xg3), _ct.c_float(xg4))
    (cig1, cig2, cig3, cig4) = (_ct.c_int(0), _ct.c_int(0),
                                _ct.c_int(0), _ct.c_int(0))
    _rp.f_cxgaig(_C_WCHAR2CHAR(grtyp),
            _ct.byref(cig1), _ct.byref(cig2), _ct.byref(cig3), _ct.byref(cig4),
            _ct.byref(cxg1), _ct.byref(cxg2), _ct.byref(cxg3), _ct.byref(cxg4))
    return (cig1.value, cig2.value, cig3.value, cig4.value)


def incdatr(idate, nhours):
    """
    Increase idate by nhours

    date2 = incdatr(idate, nhours)

    Args:
        idate  : CMC encodec date (int)
        nhours : number of hours (float)
    Returns:
        int, CMC encodec date, idate+nhours
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> (yyyymmdd, hhmmsshh, nhours0) = (20150123, 0, 6.)
    >>> try:
    ...     idate1 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)
    ...     idate2 = rmn.incdatr(idate1, nhours0)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem computing increased date.")

    See also:
        newdate
        difdatr
        rpnpy.librmn.const
        rpnpy.rpndate
    """
    if not isinstance(idate, _integer_types):
        raise TypeError("incdatr: Expecting idate of type int, Got {0} : {1}"\
                        .format(type(idate), repr(idate)))
    if idate < 0:
        raise ValueError("incdatr: must provide a valid idate: {0}".format(idate))
    if isinstance(nhours, _integer_types):
        nhours = float(nhours)
    if not isinstance(nhours, float):
        raise TypeError("incdatr: Expecting nhours of type float, "+
                        "Got {0} : {1}".format(type(nhours), repr(nhours)))
    (cidateout, cidatein, cnhours) = (_ct.c_int(idate), _ct.c_int(idate),
                                      _ct.c_double(nhours))
    _rp.f_incdatr(_ct.byref(cidateout), _ct.byref(cidatein), _ct.byref(cnhours))
    if cidateout.value == 101010101:
        raise RMNBaseError()
    return cidateout.value


def difdatr(idate1, idate2):
    """
    Compute the diffence between dates in hours (nhours = idate1 - idate2)

    nhours = difdatr(idate1, idate2)

    Args:
        idate1 : CMC encodec date (int)
        idate2 : CMC encodec date (int)
    Returns:
        float, number of hours resulting from idate1 - idate2
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> (yyyymmdd, hhmmsshh, nhours0) = (20150123, 0, 6.)
    >>> try:
    ...     idate1 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)
    ...     idate2 = rmn.incdatr(idate1, nhours0)
    ...     nhours = rmn.difdatr(idate2, idate1)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem computing date diff.")

    See also:
        newdate
        incdatr
        rpnpy.librmn.const
        rpnpy.rpndate
    """
    if not (isinstance(idate1, _integer_types) and
            isinstance(idate2, _integer_types)):
        raise TypeError("difdatr: Expecting idate1, 2 of type int, " +
                        "Got {0}, {1}".format(type(idate1), type(idate2)))
    if idate1 < 0 or idate2 < 0:
        raise ValueError("difdatr: must provide a valid idates: {0}, {1}"\
                         .format(idate1, idate2))
    (cidate1, cidate2, cnhours) = (_ct.c_int(idate1), _ct.c_int(idate2),
                                   _ct.c_double())
    _rp.f_difdatr(_ct.byref(cidate1), _ct.byref(cidate2), _ct.byref(cnhours))
    if cnhours.value == 2.**30:
        raise RMNBaseError()
    return cnhours.value


def newdate_options_set(option):
    """
    Set option for newdate, incdatr, difdatr

    Args:
        option : 'option=value' to set (str)
                 possible values:
                    'year=gregorian'
                    'year=365_day'
                    'year=360_day'
    Returns:
        None
    Raises:
        TypeError if option not a string

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.newdate_options_set('year=gregorian')

    See also:
        newdate_options_get
        ignore_leapyear
        accept_leapyear
        newdate
    """
    cmd = 'set'
    _rp.f_newdate_options(_C_WCHAR2CHAR(option), _C_WCHAR2CHAR(cmd),
                          len(option), len(cmd))


def newdate_options_get(option):
    """
    Get option for newdate, incdatr, difdatr

    Args:
        option : option name (str)
                 possible values:
                    'year'
    Returns:
        option value (str)
    Raises:
        TypeError if option not a string

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> value = rmn.newdate_options_get('year')

    See also:
        newdate_options_set
        ignore_leapyear
        accept_leapyear
        newdate
    """
    cmd = 'get '
    optionv = _C_MKSTR(option.strip()+' '*32)
    loptionv = len(optionv.value)
    _rp.f_newdate_options(optionv, _C_WCHAR2CHAR(cmd), loptionv, len(cmd))
    if isinstance(optionv.value, bytes):
        return _C_CHAR2WCHAR(optionv.value).strip()
    else:
        return optionv.value.strip()


def ignore_leapyear():
    """
    Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr

    Equivalent to: NewDate_Options('year=365_day', 'set')

    Args:
        None
    Returns:
        None

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.ignore_leapyear()

    See also:
        accept_leapyear
        newdate_options_set
        newdate_options_get
        incdatr
        difdatr
    """
    _rp.f_ignore_leapyear()


def accept_leapyear():
    """
    Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr

    Equivalent to: NewDate_Options('year=gregorian', 'set')

    Args:
        None
    Returns:
        None

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.accept_leapyear()

    See also:
        ignore_leapyear
        newdate_options_set
        newdate_options_get
        incdatr
        difdatr
    """
    _rp.f_accept_leapyear()


def get_leapyear_status():
    """
    Get the leapyear status used in newdate, incdatr, difdatr

    Args:
        None
    Returns:
        True is leap year is used

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> isLeapYear = rmn.get_leapyear_status()

    See also:
        accept_leapyear
        ignore_leapyear
        newdate_options_set
        newdate_options_get
        incdatr
        difdatr
    """
    val = newdate_options_get('year')
    if val.strip() in ('365_day', '360_day'):
        return True
    return False


def newdate(imode, idate1, idate2=0):
    """
    Convert date format between: printable, CMC date-time stamp, true date

    Args:
        imode  : Conversion mode see below (int)
        idate1 : imode dependent, See Note below (int)
        idate2 : imode dependent, See Note below (int)
    Returns:
        The converted value(s), imode dependent, see note below
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNBaseError on any other error

    Details:
       Options details if
           outdate = newdate(imode, idate1, idate2)

       imode CAN TAKE THE FOLLOWING VALUES:
          -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7
       imode=1 : STAMP TO (TRUE_DATE AND RUN_NUMBER)
          (odate1, odate2) = newdate(imode, idate1)
          idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
          odate1 : THE TRUEDATE CORRESPONDING TO DAT2
          odate2 : RUN NUMBER OF THE DATE-TIME STAMP
       imode=-1 : (TRUE_DATE AND RUN_NUMBER) TO STAMP
          odate1 = newdate(imode, idate1, idate2)
          idate1 : TRUEDATE TO BE CONVERTED
          idate2 : RUN NUMBER OF THE DATE-TIME STAMP
          odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
       imode=2 : PRINTABLE TO TRUE_DATE
          odate1 = newdate(imode, idate1, idate2)
          idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
          odate1 : TRUE_DATE
       imode=-2 : TRUE_DATE TO PRINTABLE
          (odate1, odate2) = newdate(imode, idate1)
          idate1 : TRUE_DATE
          odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
       imode=3 : PRINTABLE TO STAMP
          odate1 = newdate(imode, idate1, idate2)
          idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
          odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
       imode=-3 : STAMP TO PRINTABLE
          (odate1, odate2) = newdate(imode, idate1)
          idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
          odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
       imode=4 : 14 word old style DATE array TO STAMP and array(14)
          odate1 = newdate(imode, idate1)
          idate1 : 14 word old style DATE array
          odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
       imode=-4 : STAMP TO 14 word old style DATE array
          odate1 = newdate(imode, idate1)
          idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
          odate1 : 14 word old style DATE array
       imode=5    PRINTABLE TO EXTENDED STAMP (year 0 to 10, 000)
          odate1 = newdate(imode, idate1, idate2)
          idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
          odate1 : EXTENDED DATE-TIME STAMP (NEW STYLE only)
       imode=-5   EXTENDED STAMP (year 0 to 10, 000) TO PRINTABLE
          (odate1, odate2) = newdate(imode, idate1)
          idate1 : EXTENDED DATE-TIME STAMP (NEW STYLE only)
          odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
       imode=6 :  EXTENDED STAMP TO EXTENDED TRUE_DATE (in hours)
          (odate1, odate2) = newdate(imode, idate1)
          idate2 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
          odate1 : THE TRUEDATE CORRESPONDING TO DAT2
          odate2 : RUN NUMBER, UNUSED (0)
       imode=-6 : EXTENDED TRUE_DATE (in hours) TO EXTENDED STAMP
          odate1 = newdate(imode, idate1, idate2)
          idate1 : TRUEDATE TO BE CONVERTED
          idate2 : RUN NUMBER, UNUSED
          odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
       imode=7  : PRINTABLE TO EXTENDED TRUE_DATE (in hours)
          odate1 = newdate(imode, idate1, idate2)
          idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
          odate1 : EXTENDED TRUE_DATE
       imode=-7 : EXTENDED TRUE_DATE (in hours) TO PRINTABLE
          (odate1, odate2) = newdate(imode, idate1)
          idate1 : EXTENDED TRUE_DATE
          odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
          odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)

    Notes:
        Old Style Date Array is composed of 14 elements:
        0 : Day of the week (1=Sunday, ..., 7=Saturday
        1 : Month (1=Jan, ..., 12=Dec)
        2 : Day of the Month
        3 : Year
        4 : Hour of the Day
        5 : Minutes  * 60 * 100
        ...
        13: CMC Date-Time Stamp

    Examples:
    >>> import sys
    >>> import rpnpy.librmn.all as rmn
    >>> (yyyymmdd, hhmmsshh) = (20150123, 0)
    >>> try:
    ...     idate1 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)
    ...     (yyyymmdd2, hhmmsshh2) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT, idate1)
    ... except rmn.RMNBaseError:
    ...     sys.stderr.write("There was a problem encoding/decoding the date.")

    See also:
        accept_leapyear
        ignore_leapyear
        get_leapyear_status
        newdate_options_set
        newdate_options_get
        incdatr
        difdatr
        rpnpy.librmn.const
        rpnpy.rpndate
    """
    if not isinstance(imode, _integer_types):
        raise TypeError("newdate: Expecting imode of type int, Got {0} : {1}"\
                        .format(type(imode), repr(imode)))
    if imode != 4:
        if not (isinstance(idate1, _integer_types) and
                isinstance(idate2, _integer_types)):
            raise TypeError("newdate: Expecting idate1, 2 of type int, " +
                            "Got {0}, {1}".format(type(idate1), type(idate2)))
    else:
        if not isinstance(idate1, (list, tuple)) or len(idate1) != 14:
            raise TypeError("newdate: Expecting idate1 of type=list, len=14, " +
                            "Got type={0}, len={1}".format(type(idate1), len(idate1)))

    if not isinstance(idate1, (list, tuple)):
        if idate1 < 0 or idate2 < 0:
            raise ValueError("newdate: must provide a valid idates: {0}, {1}"\
                             .format(idate1, idate2))
    cimode = _ct.c_int(imode)
    (cidate1, cidate2, cidate3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    if imode == 1:
        cidate2 = _ct.c_int(idate1)
    elif imode == -1:
        (cidate1, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == 2:
        (cidate2, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == -2:
        cidate1 = _ct.c_int(idate1)
    elif imode == 3:
        (cidate2, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == -3:
        cidate1 = _ct.c_int(idate1)
    elif imode == 4:
        cidate2 = _np.asfortranarray(idate1, dtype=_np.int32)
    elif imode == -4:
        cidate1 = _ct.c_int(idate1)
        cidate2 = _np.zeros((14,), dtype=_np.int32, order='F')
    elif imode == 5:
        (cidate2, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == -5:
        cidate1 = _ct.c_int(idate1)
    elif imode == 6:
        cidate2 = _ct.c_int(idate1)
    elif imode == -6:
        (cidate1, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == 7:
        (cidate2, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    elif imode == -7:
        cidate1 = _ct.c_int(idate1)
    else:
        raise ValueError("newdate: must provide a valid imode: {0}".format(imode))
    if imode in (4, -4):
        istat = _rp.f_newdate(_ct.byref(cidate1), cidate2,
                              _ct.byref(cidate3), cimode)
    else:
        istat = _rp.f_newdate(_ct.byref(cidate1), _ct.byref(cidate2),
                              _ct.byref(cidate3), _ct.byref(cimode))
    if istat == 1:
        raise RMNBaseError()
    if imode == 1:
        return (cidate1.value, cidate3.value)
    elif imode == -1:
        return cidate2.value
    elif imode == 2:
        return cidate1.value
    elif imode == -2:
        return (cidate2.value, cidate3.value)
    elif imode == 3:
        return cidate1.value
    elif imode == -3:
        return (cidate2.value, cidate3.value)
    elif imode == 4:
        return cidate1.value
    elif imode == -4:
        return list(cidate2)
    elif imode == 5:
        return cidate1.value
    elif imode == -5:
        return (cidate2.value, cidate3.value)
    elif imode == 6:
        return (cidate1.value, cidate3.value)
    elif imode == -6:
        return cidate2.value
    elif imode == 7:
        return cidate1.value
    elif imode == -7:
        return (cidate2.value, cidate3.value)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
