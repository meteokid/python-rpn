#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
Module librmn.base contains python wrapper to main librmn,
base and primitives C functions
 
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

import ctypes as _ct
import numpy  as _np
from . import proto as _rp
from . import const as _rc
from . import RMNError

C_MKSTR = _ct.create_string_buffer
C_TOINT = lambda x: (x if (type(x) != type(_ct.c_int())) else x.value)
IS_LIST = lambda x: type(x) in (list, tuple)

class RMNBaseError(RMNError):
    """General RMNBase module error/exception
    """
    pass

#--- primitives -----------------------------------------------------

def fclos(iunit):
    """Close file associated with unit through fnom

    fclos(iunit)

    Args:
        iunit   : unit number associated to the file
                  obtained with fnom
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types    
        ValueError on invalid input arg value
        RMNBaseError on any other error       
    """
    if not (type(iunit) == int):
        raise TypeError("fcols: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fclos: must provide a valid iunit: %d" % (iunit))
    istat = _rp.c_fclos(iunit)
    if istat < 0:
        raise RMNBaseError()
    return istat


def fnom(filename, filemode=_rc.FST_RW, iunit=0):
    """Open a file and make the connection with a unit number.

    iunit = fnom(filename)
    iunit = fnom(filename, filemode)
    iunit = fnom(filename, filemode, iunit)

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
    """
    if not (type(iunit) == int):
        raise TypeError("fnom: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    ciunit = _ct.c_int(max(0, iunit))
    if not (type(iunit) == int):
        raise TypeError("fnom: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if not (type(filename) == str):
        raise TypeError("fnom: Expecting filename arg of type str, Got %s" %
                        (type(filename)))
    if filename.strip() == '':
        raise ValueError("fnom: must provide a valid filename")
    if not (type(filemode) == str):
        raise TypeError("fnom: Expecting arg filemode of type str, Got %s" %
                        (type(filemode)))
    istat = _rp.c_fnom(_ct.byref(ciunit), filename, filemode, 0)
    istat = C_TOINT(istat)
    if istat < 0:
        raise RMNBaseError()
    return ciunit.value


def wkoffit(filename):
    """Return type of file (int)

    ftype = wkoffit(filename)

    Args:
        filename : path/name of the file to examine
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
    """
    if not (type(filename) == str):
        raise TypeError("wkoffit: Expecting filename arg of type str, " +
                        "Got %s" % (type(filename)))
    if filename.strip() == '':
        raise ValueError("wkoffit: must provide a valid filename")
    return _rp.c_wkoffit(filename, len(filename))


def crc32(crc, buf):
    """Compute the Cyclic Redundancy Check (CRC)

    crc = crc32(crc0, buf)

    Args:
       crc0 : initial crc value (int)
       buf  : list of number to compute updated crc (numpy.ndarray of uint32)
    Returns:
       crc : computed crc value (int)
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value    
    """
    if not (buf.dtype == _np.uint32 and buf.flags['F_CONTIGUOUS']):
        buf = _np.asfortranarray(buf, dtype=_np.uint32)
    return _rp.c_crc32(crc, buf, buf.size*4)

#--- base -----------------------------------------------------------


def cigaxg(grtyp, ig1, ig2=0, ig3=0, ig4=0):
    """Decode ig1, ig2, ig3, ig4 into real grid descriptors

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
        RMNBaseError on any other error
    """
    if not (type(grtyp) == str):
        raise TypeError("cigaxg: Expecting grtyp arg of type str, Got %s" %
(type(grtyp)))
    if grtyp.strip() == '':
        raise ValueError("cigaxg: must provide a valid grtyp")
    (cig1, cig2, cig3, cig4) = (_ct.c_int(ig1), _ct.c_int(ig2),
                                _ct.c_int(ig3), _ct.c_int(ig4))            
    if IS_LIST(ig1):
        (cig1, cig2, cig3, cig4) = (_ct.c_int(ig1[0]), _ct.c_int(ig1[1]),
                                    _ct.c_int(ig1[2]), _ct.c_int(ig1[3]))
    (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(0.), _ct.c_float(0.),
                                _ct.c_float(0.), _ct.c_float(0.))
    istat = _rp.f_cigaxg(grtyp,
                _ct.byref(cxg1), _ct.byref(cxg2),
                _ct.byref(cxg3), _ct.byref(cxg4),
                _ct.byref(cig1), _ct.byref(cig2),
                _ct.byref(cig3), _ct.byref(cig4))
    if istat < 0:
        raise RMNBaseError()
    return (cxg1.value, cxg2.value, cxg3.value, cxg4.value)


def cxgaig(grtyp, xg1, xg2=0., xg3=0., xg4=0.):
    """Encode real grid descriptors into ig1, ig2, ig3, ig4

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
        RMNBaseError on any other error
    """
    if not (type(grtyp) == str):
        raise TypeError("cigaxg: Expecting grtyp arg of type str, Got %s" %
                        (type(grtyp)))
    if grtyp.strip() == '':
        raise ValueError("cigaxg: must provide a valid grtyp")
    (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(xg1), _ct.c_float(xg2),
                                _ct.c_float(xg3), _ct.c_float(xg4))
    if IS_LIST(xg1):
        (cxg1, cxg2, cxg3, cxg4) = (_ct.c_float(xg1[0]), _ct.c_float(xg1[1]),
                                    _ct.c_float(xg1[2]), _ct.c_float(xg1[3]))
    (cig1, cig2, cig3, cig4) = (_ct.c_int(0), _ct.c_int(0),
                                _ct.c_int(0), _ct.c_int(0))
    istat = _rp.f_cxgaig(grtyp,
            _ct.byref(cig1), _ct.byref(cig2), _ct.byref(cig3), _ct.byref(cig4),
            _ct.byref(cxg1), _ct.byref(cxg2), _ct.byref(cxg3), _ct.byref(cxg4))
    if istat < 0:
        raise RMNBaseError()
    return (cig1.value, cig2.value, cig3.value, cig4.value)


def incdatr(idate, nhours):
    """Increate idate by nhours

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
    """
    if type(idate) != int:
        raise TypeError("incdatr: Expecting idate of type int, Got %s : %s" %
                        (type(idate), repr(idate)))
    if idate < 0:
        raise ValueError("incdatr: must provide a valid idate: %d" % (idate))
    if type(nhours) == int:
        nhours = float(nhours)
    if type(nhours) != float:
        raise TypeError("incdatr: Expecting nhours of type float, "+
                        "Got %s : %s" % (type(nhours), repr(nhours)))
    (cidateout, cidatein, cnhours) = (_ct.c_int(idate), _ct.c_int(idate),
                                      _ct.c_double(nhours))
    _rp.f_incdatr(_ct.byref(cidateout), _ct.byref(cidatein), _ct.byref(cnhours))
    if cidateout.value == 101010101:
        raise RMNBaseError()
    return cidateout.value


def difdatr(idate1, idate2):
    """Compute the diffence between dates in hours (nhours = idate1 - idate2)

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
    """
    if type(idate1) != int or type(idate2) != int:
        raise TypeError("difdatr: Expecting idate1, 2 of type int, " +
                        "Got %s, %s" % (type(idate1), type(idate2)))
    if idate1 < 0 or idate2 < 0:
        raise ValueError("difdatr: must provide a valid idates: %d, %d" %
                         (idate1, idate2))
    (cidate1, cidate2, cnhours) = (_ct.c_int(idate1), _ct.c_int(idate2),
                                   _ct.c_double())
    _rp.f_difdatr(_ct.byref(cidate1), _ct.byref(cidate2), _ct.byref(cnhours))
    if cnhours.value == 2.**30:
        raise RMNBaseError()
    return cnhours.value


def newdate_options_set(option):
    """Set option for newdate, incdatr, difdatr

    newdate_options_set('year=gregorian')
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
    """
    cmd = 'set'
    _rp.f_newdate_options(option, cmd, len(option), len(cmd))


def newdate_options_get(option):
    """Get option for newdate, incdatr, difdatr

    value = newdate_options_get('year')
    
    Args:
        option : option name (str)
                 possible values:
                    'year'
    Returns:
        option value (str)
    Raises:
        TypeError if option not a string
    """
    cmd = C_MKSTR('get ')
    optionv = C_MKSTR(option.strip()+' '*32)
    #optionv = C_MKSTR('year            ')
    _rp.f_newdate_options(optionv, cmd, len(optionv.value), len(cmd.value))
    return optionv.value.strip()


def ignore_leapyear():
    """Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr
    Equivalent to: NewDate_Options('year=365_day', 'set')

    ignore_leapyear()
    
    Args:
        None
    Returns:
        None
    """
    _rp.f_ignore_leapyear()


def accept_leapyear():
    """Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr
    Equivalent to: NewDate_Options('year=gregorian', 'set')

    accept_leapyear()
    
    Args:
        None
    Returns:
        None
    """
    _rp.f_accept_leapyear()


def get_leapyear_status():
    """Get the leapyear status used in newdate, incdatr, difdatr

    isLeapYear = get_leapyear_status()
    
    Args:
        None
    Returns:
        True is leap year is used
    """
    val = newdate_options_get('year')
    if val.strip() in ('365_day', '360_day'):
        return True
    return False    


def newdate(imode, idate1, idate2=0):
    """Convert date format between: printable, CMC date-time stamp, true date

    outdate = newdate(imode, idate1, idate2)
    
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
    """
    if type(imode) != int:
        raise TypeError("incdatr: Expecting imode of type int, Got %s : %s" %
                        (type(imode), repr(imode)))
    if type(idate1) != int or type(idate2) != int:
        raise TypeError("newdate: Expecting idate1, 2 of type int, " +
                        "Got %s, %s" % (type(idate1), type(idate2)))
    if idate1 < 0 or idate2 < 0:
        raise ValueError("newdate: must provide a valid idates: %d, %d" %
                         (idate1, idate2))
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
    #TODO: add support for imode == 4
    ## elif imode == 4:
    ##    (cidate2, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
    ## elif imode == -4:
    ##    (cidate1, cidate3) = (_ct.c_int(idate1), _ct.c_int(idate2))
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
        raise ValueError("incdatr: must provide a valid imode: %d" % (imode))
    istat = _rp.f_newdate(_ct.byref(cidate1), _ct.byref(cidate2),
                          _ct.byref(cidate3), _ct.byref(cimode))
    if istat == 1: #TODO: check this, should it be (istat < 0)
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
    #TODO: add support for imode == 4
    ## elif imode == 4:
    ##     return cidate1.value
    ## elif imode == -4:
    ##     return cidate2.value
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
