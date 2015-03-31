#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
 Module librmn.base contains python wrapper to main librmn's base and primitives C functions
 
 @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

import ctypes as _ct
from . import proto as _rp
from . import const as _rc

c_toint = lambda x: (x if (type(x) != type(_ct.c_int())) else x.value)
isListType = lambda x: type(x) in (type([]),type((1,)))

#--- primitives -----------------------------------------------------

def fclos(iunit):
    """Close file associated with unit through fnom
    
    iunit   : unit number associated to the file
              obtained with fnom
            
    return None on error int>=0 otherwise
    """
    istat = _rp.c_fclos(iunit)
    if istat < 0: return None
    return istat


def fnom(filename,filemode=_rc.FST_RW,iunit=0):
    """Open a file and make the connection with a unit number.
    
    filename : path/name of the file to open
    filemode : a string with the desired filemode (see librmn doc)
               or one of these constants: FST_RW, FST_RW_OLD, FST_RO
    iunit    : forced unit number to conect to
               if zero, will select a free unit

    return Associated file unit number
    return None on error
    """
    iunit2 = _ct.c_int(iunit)
    istat = _rp.c_fnom(_ct.byref(iunit2),filename,filemode,0)
    istat = c_toint(istat)
    if istat >= 0: return iunit2.value
    return None


def wkoffit(filename):
    """Return type of file (int)

    filename : path/name of the file to examine

    return file type code as follow:
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
    """
    return _rp.c_wkoffit(filename,len(filename))


#--- base -----------------------------------------------------------


def cigaxg(proj,ig1,ig2=0,ig3=0,ig4=0):
    """Encode real grid descriptors into ig1,ig2,ig3,ig4
    Args:
        proj   (str): (I)
        ig1..4 (int): encoded grid descriptor values
          or
        ig1    (int,int,int,int): tuple/list with encoded grid desc values
    Returns:
        (float,float,float,float), Decoded grid parameters value
        None on error
    """
    (cig1,cig2,cig3,cig4) = (_ct.c_int(ig1),_ct.c_int(ig2),_ct.c_int(ig3),_ct.c_int(ig4))            
    if isListType(ig1):
        (cig1,cig2,cig3,cig4) = (_ct.c_int(ig1[0]),_ct.c_int(ig1[1]),_ct.c_int(ig1[2]),_ct.c_int(ig1[3]))            
    (cxg1,cxg2,cxg3,cxg4) = (_ct.c_float(0.),_ct.c_float(0.),_ct.c_float(0.),_ct.c_float(0.))
    istat = _rp.f_cigaxg(proj,
                _ct.byref(cxg1),_ct.byref(cxg2),_ct.byref(cxg3),_ct.byref(cxg4),
                _ct.byref(cig1),_ct.byref(cig2),_ct.byref(cig3),_ct.byref(cig4))
    if istat < 0: return None
    return (cxg1.value,cxg2.value,cxg3.value,cxg4.value)


def cxgaig(proj,xg1,xg2=0.,xg3=0.,xg4=0.):
    """Decode ig1,ig2,ig3,ig4 into real grid descriptors 
    Args:
        proj   (str): (I)
        xg1..4 (float): encoded grid descriptor values
          or
        xg1    (float,...,float): tuple/list with real grid desc values
    Returns:
        (int,int,int,int), Encoded grid parameters
        None on error
    """
    (cxg1,cxg2,cxg3,cxg4) = (_ct.c_float(xg1),_ct.c_float(xg2),_ct.c_float(xg3),_ct.c_float(xg4))
    if isListType(xg1):
        (cxg1,cxg2,cxg3,cxg4) = (_ct.c_float(xg1[0]),_ct.c_float(xg1[1]),_ct.c_float(xg1[2]),_ct.c_float(xg1[3]))
    (cig1,cig2,cig3,cig4) = (_ct.c_int(0),_ct.c_int(0),_ct.c_int(0),_ct.c_int(0))
    istat = _rp.f_cxgaig(proj,
            _ct.byref(cig1),_ct.byref(cig2),_ct.byref(cig3),_ct.byref(cig4),
            _ct.byref(cxg1),_ct.byref(cxg2),_ct.byref(cxg3),_ct.byref(cxg4))
    if istat < 0: return None
    return (cig1.value,cig2.value,cig3.value,cig4.value)


def incdatr(idate,nhours):
    """Increate idate by nhours
    Args:
        idate  (int)  : CMC encodec date
        nhours (float): number of hours
    Returns:
        int, CMC encodec date, idate+nhours
        None on error
    """
    (cidateout,cidatein,cnhours) = (_ct.c_int(idate),_ct.c_int(idate),_ct.c_double(nhours))
    _rp.f_incdatr(_ct.byref(cidateout),_ct.byref(cidatein),_ct.byref(cnhours))
    if cidateout.value == 101010101: return None
    return cidateout.value


def difdatr(idate1,idate2):
    """Compute de diffence between dates in hours (nhours = idate1 - idate2)
    Args:
        idate1 (int)  : CMC encodec date
        idate2 (int)  : CMC encodec date
    Returns:
        float, number of hours resulting from idate1 - idate2
        None on error
    """
    (cidate1,cidate2,cnhours) = (_ct.c_int(idate1),_ct.c_int(idate2),_ct.c_double())
    _rp.f_difdatr(_ct.byref(cidate1),_ct.byref(cidate2),_ct.byref(cnhours))
    if cnhours.value == 2.**30: return None
    return cnhours.value


def newdate(imode,idate1,idate2=0):
    """Convert date format between: printable, CMC date-time stamp, true date
    Args:
        imode  (int)  : Conversion mode see below
        idate1 (int)  : See Note below
        idate2 (int)  : See Note below
    Returns:
        The converted value(s), imode dependent, see note below
        None on error
    imode CAN TAKE THE FOLLOWING VALUES:-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7
    imode=1 : STAMP TO (TRUE_DATE AND RUN_NUMBER)
        (odate1,odate2) = newdate(imode,idate1)
        idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
        odate1 : THE TRUEDATE CORRESPONDING TO DAT2
        odate2 : RUN NUMBER OF THE DATE-TIME STAMP
    imode=-1 : (TRUE_DATE AND RUN_NUMBER) TO STAMP
        odate1 = newdate(imode,idate1,idate2)
        idate1 : TRUEDATE TO BE CONVERTED
        idate2 : RUN NUMBER OF THE DATE-TIME STAMP
        odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
    imode=2 : PRINTABLE TO TRUE_DATE
        odate1 = newdate(imode,idate1,idate2)
        idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
        odate1 : TRUE_DATE
    imode=-2 : TRUE_DATE TO PRINTABLE
        (odate1,odate2) = newdate(imode,idate1)
        idate1 : TRUE_DATE
        odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
    imode=3 : PRINTABLE TO STAMP
        odate1 = newdate(imode,idate1,idate2)
        idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
        odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
    imode=-3 : STAMP TO PRINTABLE
        (odate1,odate2) = newdate(imode,idate1)
        idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
        odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
    imode=4 : 14 word old style DATE array TO STAMP and array(14)
        odate1 = newdate(imode,idate1)
        idate1 : 14 word old style DATE array
        odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
    imode=-4 : STAMP TO 14 word old style DATE array
        odate1 = newdate(imode,idate1)
        idate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
        odate1 : 14 word old style DATE array
    imode=5    PRINTABLE TO EXTENDED STAMP (year 0 to 10,000)
        odate1 = newdate(imode,idate1,idate2)
        idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
        odate1 : EXTENDED DATE-TIME STAMP (NEW STYLE only)
    imode=-5   EXTENDED STAMP (year 0 to 10,000) TO PRINTABLE
        (odate1,odate2) = newdate(imode,idate1)
        idate1 : EXTENDED DATE-TIME STAMP (NEW STYLE only)
        odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
    imode=6 :  EXTENDED STAMP TO EXTENDED TRUE_DATE (in hours)
        (odate1,odate2) = newdate(imode,idate1)
        idate2 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
        odate1 : THE TRUEDATE CORRESPONDING TO DAT2
        odate2 : RUN NUMBER, UNUSED (0)
    imode=-6 : EXTENDED TRUE_DATE (in hours) TO EXTENDED STAMP
        odate1 = newdate(imode,idate1,idate2)
        idate1 : TRUEDATE TO BE CONVERTED
        idate2 : RUN NUMBER, UNUSED
        odate1 : CMC DATE-TIME STAMP (OLD OR NEW STYLE)
    imode=7  : PRINTABLE TO EXTENDED TRUE_DATE (in hours)
        odate1 = newdate(imode,idate1,idate2)
        idate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        idate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
        odate1 : EXTENDED TRUE_DATE
    imode=-7 : EXTENDED TRUE_DATE (in hours) TO PRINTABLE
        (odate1,odate2) = newdate(imode,idate1)
        idate1 : EXTENDED TRUE_DATE
        odate1 : DATE OF THE PRINTABLE DATE (YYYYMMDD)
        odate2 : TIME OF THE PRINTABLE DATE (HHMMSSHH)
    """
    cimode = _ct.c_int(imode)
    (cidate1,cidate2,cidate3) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    if imode == 1:
       cidate2 = _ct.c_int(idate1)
    elif imode == -1:
       (cidate1,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == 2:
       (cidate2,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == -2:
       cidate1 = _ct.c_int(idate1)
    elif imode == 3:
       (cidate2,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == -3:
       cidate1 = _ct.c_int(idate1)
    elif imode == 4:
       (cidate2,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == -4:
       (cidate1,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == 5:
       (cidate2,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == -5:
       cidate1 = _ct.c_int(idate1)
    elif imode == 6:
       cidate2 = _ct.c_int(idate1)
    elif imode == -6:
       (cidate1,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == 7:
       (cidate2,cidate3) = (_ct.c_int(idate1),_ct.c_int(idate2))
    elif imode == -7:
       cidate1 = _ct.c_int(idate1)
    else:
       return None

    istat = _rp.f_newdate(_ct.byref(cidate1),_ct.byref(cidate2),_ct.byref(cidate3),_ct.byref(cimode))
    if istat == 1: return None

    if imode == 1:
       return (cidate1.value,cidate3.value)
    elif imode == -1:
       return cidate2.value
    elif imode == 2:
       return cidate1.value
    elif imode == -2:
       return (cidate2.value,cidate3.value)
    elif imode == 3:
       return cidate1.value
    elif imode == -3:
       return (cidate2.value,cidate3.value)
    elif imode == 4:
       return cidate1.value
    elif imode == -4:
       return cidate2.value
    elif imode == 5:
       return cidate1.value
    elif imode == -5:
       return (cidate2.value,cidate3.value)
    elif imode == 6:
       return (cidate1.value,cidate3.value)
    elif imode == -6:
       return cidate2.value
    elif imode == 7:
       return cidate1.value
    elif imode == -7:
       return (cidate2.value,cidate3.value)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
