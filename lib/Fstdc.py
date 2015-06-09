
"""Module Fstdc is a backward compatibility layer for rpnpy version 1.3 and older.
    This Module is deprecated in favor of rpnpy.librmn.all

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpnpy.version as rpn_version
import rpnpy.librmn.all as _rmn

KIND_ABOVE_SEA =  0
KIND_SIGMA     =  1
KIND_PRESSURE  =  2
KIND_ARBITRARY =  3
KIND_ABOVE_GND =  4
KIND_HYBRID    =  5
KIND_THETA     =  6
KIND_HOURS     = 10
KIND_SAMPLES   = 15
KIND_MTX_IND   = 17
KIND_M_PRES    = 21

LEVEL_KIND_MSL = KIND_ABOVE_SEA
LEVEL_KIND_SIG = KIND_SIGMA
LEVEL_KIND_PMB = KIND_PRESSURE
LEVEL_KIND_ANY = KIND_ARBITRARY
LEVEL_KIND_MGL = KIND_ABOVE_GND
LEVEL_KIND_HYB = KIND_HYBRID
LEVEL_KIND_TH  = KIND_THETA
LEVEL_KIND_MPR = KIND_M_PRES
TIME_KIND_HR   = KIND_HOURS

NEWDATE_PRINT2TRUE  =  2
NEWDATE_TRUE2PRINT  = -2
NEWDATE_PRINT2STAMP =  3
NEWDATE_STAMP2PRINT = -3

CONVIP_STYLE_DEFAULT =  1
CONVIP_STYLE_NEW     =  2
CONVIP_STYLE_OLD     =  3
CONVIP_IP2P_DEFAULT  = -1
CONVIP_IP2P_31BITS   =  0

FSTDC_FILE_RW     = "RND+R/W"
FSTDC_FILE_RW_OLD = "RND+R/W+OLD"
FSTDC_FILE_RO     = "RND+R/O"


class error(Exception):
    pass


class tooManyRecError(Exception):
    pass


def version():
    """(version,lastUpdate) = Fstdc.version()
    """
    return (rpn_version.__VERSION__,rpn_version.__LASTUPDATE__)


def fstouv(iunit,filename,options):
    """Interface to fstouv and fnom to open a RPN 2000 Standard File
        iunit = Fstdc.fstouv(iunit,filename,options)
        @param iunit unit number of the file handle, 0 for a new one (int)
        @param filename (string)
        @param option type of file and R/W options (string)
        @return File unit number (int), NULL on error
        @exception TypeError
        @exception Fstdc.error
    """
    iunit = _rmn.fnom(filename,options,iunit)
    _rmn.fstouv(iunit,options)
    return iunit


def fstfrm(iunit):
    """Close a RPN 2000 Standard File (Interface to fstfrm+fclos to close) 
        Fstdc.fstfrm(iunit)
        @param iunit file unit number handle returned by Fstdc_fstouv (int)
        @exception TypeError
        @exception Fstdc.error
    """
    _rmn.fstfrm(iunit)
    _rmn.fclos(iunit)


def level_to_ip1(level_list,kind):
    """Encode level value to ip1 (Interface to convip)
        myip1List = Fstdc.level_to_ip1(level_list,kind) 
        @param level_list list of level values (list of float)
        @param kind type of level (int)
        @return [(ip1new,ip1old),...] (list of tuple of int)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def ip1_to_level(ip1_list):
    """Decode ip1 to level type,value (Interface to convip)
        myLevelList = Fstdc.ip1_to_level(ip1_list)
        @param tuple/list of ip1 values to decode
        @return list of tuple (level,kind)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def cxgaig(grtyp,xg1,xg2,xg3,xg4):
    """Encode grid descriptors (Interface to cxgaig)
        (ig1,ig2,ig3,ig4) = Fstdc.cxgaig(grtyp,xg1,xg2,xg3,xg4) 
        @param ...TODO...
        @return (ig1,ig2,ig3,ig4)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def cigaxg(grtyp,ig1,ig2,ig3,ig4):
    """Decode grid descriptors (Interface to cigaxg)
        (xg1,xg2,xg3,xg4) = Fstdc.cigaxg(grtyp,ig1,ig2,ig3,ig4)
        @param ...TODO...
        @return (xg1,xg2,xg3,xg4)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def fstvoi(iunit,option):
    """Print a list view a RPN Standard File rec (Interface to fstvoi)
        Fstdc.fstvoi(iunit,option)
        @param iunit file unit number handle returned by Fstdc_fstouv (int)
        @param option 'STYLE_OLD' or 'STYLE_NEW' (sting)
        @return None
        @exception TypeError
    """
    raise error('Not implemented yet')

def fstluk(ihandle):
    """Read record data on file (Interface to fstluk)
        myRecDataDict = Fstdc.fstluk(ihandle)
        @param ihandle record handle (int) 
        @return record data (numpy.ndarray)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def fstinl(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev):
    """Find all records matching provided criterias (Interface to fstinl)
        Warning: list is limited to the first 50000 records in a file, subsequent matches raises Fstdc.tooManyRecError and are ignored.
        recList = Fstdc.fstinl(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev)
        param iunit file unit number handle returned by Fstdc_fstouv (int)
        @param nomvar seclect according to var name, blank==wildcard (string)
        @param typvar seclect according to var type, blank==wildcard (string)
        @param etiket seclect according to etiket, blank==wildcard (string)
        @param ip1 seclect according to ip1, -1==wildcard  (int)
        @param ip2 seclect according to ip2, -1==wildcard (int)
        @param ip3  seclect according to ip3, -1==wildcard (int)
        @param datev seclect according to date of validity, -1==wildcard (int)
        @returns python dict with handles+params of all matching records
        @exception TypeError
        @exception Fstdc.error
        @exception Fstdc.tooManyRecError
    """
    raise error('Not implemented yet')

def fstinf(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev,inhandle):
    """Find a record matching provided criterias (Interface to fstinf, dsfsui, fstinfx)
        recParamDict = Fstdc.fstinf(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev,inhandle)
        @param iunit file unit number handle returned by Fstdc_fstouv (int)
        @param nomvar seclect according to var name, blank==wildcard (string)
        @param typvar seclect according to var type, blank==wildcard (string)
        @param etiket seclect according to etiket, blank==wildcard (string)
        @param ip1 seclect according to ip1, -1==wildcard  (int)
        @param ip2 seclect according to ip2, -1==wildcard (int)
        @param ip3  seclect according to ip3, -1==wildcard (int)
        @param datev seclect according to date of validity, -1==wildcard (int)
        @param inhandle selcation criterion; inhandle=-2:search with criterion from start of file; inhandle=-1==fstsui, use previously provided criterion to find the next matching one; inhandle>=0 search with criterion from provided rec-handle (int)
        @returns python dict with record handle + record params keys/values
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def fsteff(ihandle):
    """Erase a record (Interface to fsteff)
        Fstdc.fsteff(ihandle)
        @param ihandle handle of the record to erase (int)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def fstecr(data,iunit,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits,datyp):
    """Write record data & meta(params) to file (Interface to fstecr), always append (no overwrite)
        Fstdc.fstecr(data,iunit,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits,datyp)
        @param data array to be written to file (numpy.ndarray)
        @param iunit file unit number handle returned by Fstdc_fstouv (int)
        @param ... 
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def fst_edit_dir(ihandle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp):
    """Rewrite the parameters of an rec on file, data part is unchanged (Interface to fst_edit_dir)
        Fstdc.fst_edit_dir(ihandle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp)
        @param ihandle record handle (int)
        @param ... 
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def gdllfxy(xin,yin,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Get (latitude,longitude) pairs cooresponding to (x,y) grid coordinates
        (lat,lon) = Fstdc.gdllfxy(xin,yin,
                (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param xin,yin: input coordinates (as float32 numpy array)
        @param (ni,nj) .. (i0,j0): Grid definition parameters
        @return (lat,lon): Computed latitude and longitude coordinates
    """
    raise error('Not implemented yet')

def gdxyfll(latin,lonin,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Get (x,y) pairs cooresponding to (lat,lon) grid coordinates
        (x,y) = Fstdc.gdxyfll(latin,lonin,
                (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param latin,lonin: input coordinates (as float32 numpy array)
        @param (ni,nj) .. (i0,j0): Grid definition parameters
        @return (x,y): Computed x and y coordinates (as floating point)
    """
    raise error('Not implemented yet')

def gdwdfuv(uuin,vvin,lat,lon,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Translate grid-directed winds to (magnitude,direction) pairs
        (UV, WD) = Fstdc.gdwdfuv(uuin,vvin,lat,lon,
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param uuin, vvin -- Grid-directed wind fields
        @param lat,lon -- Latitude and longitude coordinates of those winds
        @param ni ... j0 -- grid definition parameters
        @return (UV, WD): Wind modulus and direction as numpy.ndarray
    """
    raise error('Not implemented yet')

def gduvfwd(uvin,wdin,lat,lon,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Translate (magnitude,direction) winds to grid-directed
        (UV, WD) = Fstdc.gduvfwd(uvin,wdin,lat,lon,
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param uvin, wdin -- Grid-directed wind fields 
        @param lat,lon -- Latitude and longitude coordinates of those winds
        @param ni ... j0 -- grid definition parameters
        @return (UU, VV): Grid-directed winds as numpy.ndarray
    """
    raise error('Not implemented yet')

def gdllval(uuin,vvin,lat,lon,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Interpolate scalar or vector fields to scattered (lat,lon) points
        vararg = Fstdc.gdllval(uuin,vvin,lat,lon,
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param (uuin,vvin): Fields to interpolate from; if vvin is None then
                            perform scalar interpolation
        @param lat,lon:  Latitude and longitude coordinates for interpolation
        @param ni ... j0: grid definition parameters
        @return Zout or (UU,VV): scalar or tuple of grid-directed, vector-interpolated
                                 fields, as appropriate
    """
    raise error('Not implemented yet')

def gdxyval(uuin,vvin,x,y,nij,grtyp,grref_params,xyAxis,hasAxis,ij0):
    """Interpolate scalar or vector fields to scattered (x,y) points
        vararg = Fstdc.gdxyval(uuin,vvin,x,y,
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))
        @param (uuin,vvin): Fields to interpolate from; if vvin is None then
                            perform scalar interpolation
        @param x,y:  X and Y-coordinates for interpolation
        @param ni ... j0: grid definition parameters
        @return Zout or (UU,VV): scalar or tuple of grid-directed, vector-interpolated
                                 fields, as appropriate
    """
    raise error('Not implemented yet')

def ezinterp(arrayin,arrayin2,
             src_nij,src_grtyp,src_grref_params,src_xyAxis,src_hasAxis,src_ij0
             dst_nij,dst_grtyp,dst_grref_params,dst_xyAxis,dst_hasAxis,dst_ij0):
    """Interpolate from one grid to another
        newArray = Fstdc.ezinterp(arrayin,arrayin2,
        (niS,njS),grtypS,(grrefS,ig1S,ig2S,ig3S,ig4S),(xsS,ysS),hasSrcAxis,(i0S,j0S),
        (niD,njD),grtypD,(grrefD,ig1D,ig2D,ig3D,ig4D),(xsD,ysD),hasDstAxis,(i0D,j0D),
        isVect)
        @param ...TODO...
        @return interpolated data (numpy.ndarray)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def incdatr(date1,nhours):
    """Increase CMC datetime stamp by a N hours (Interface to incdatr)
        date2 = Fstdc.incdatr(date1,nhours)
        @param date1 original CMC datetime stamp(int)
        @param nhours number of hours to increase the date (double)
        @return Increase CMC datetime stamp (int)
        @exception TypeError
        @exception Fstdc.error
    """
    raise error('Not implemented yet')

def difdatr((date1,date2):
    """Compute differenc between 2 CMC datatime stamps (Interface to difdatr)
        nhours = Fstdc.difdatr(date1,date2)
        @param date1 CMC datatime stamp (int)
        @param date2 CMC datatime stamp (int)
        @return number of hours = date2-date1 (float)
        @exception TypeError
    """
    raise error('Not implemented yet')

def newdate(dat1,dat2,dat3,mode):
    """Convert data to/from printable format and CMC stamp (Interface to newdate)
        (fdat1,fdat2,fdat3) = Fstdc.newdate(dat1,dat2,dat3,mode)
        @param ...see newdate doc... 
        @return tuple with converted date values ...see newdate doc...
        @exception TypeError
        @exception Fstdc.error

1.1 ARGUMENTS
mode can take the following values:-3,-2,-1,1,2,3
mode=1 : stamp to (true_date and run_number)
   out  fdat1  the truedate corresponding to dat2       integer
    in  dat2   cmc date-time stamp (old or new style)   integer
   out  fdat3  run number of the date-time stamp        integer
    in  mode   set to 1                                 integer 
mode=-1 : (true_date and run_number) to stamp
    in  dat1   truedate to be converted                 integer
   out  fdat2  cmc date-time stamp (old or new style)   integer
    in  dat3   run number of the date-time stamp        integer
    in  mode   set to -1                                integer
mode=2 : printable to true_date
   out  fdat1  true_date                                integer
    in  dat2   date of the printable date (YYYYMMDD)    integer
    in  dat3   time of the printable date (HHMMSShh)    integer
    in  mode   set to 2                                 integer
mode=-2 : true_date to printable
    in  dat1   true_date                                integer
   out  fdat2  date of the printable date (YYYYMMDD)    integer
   out  fdat3  time of the printable date (HHMMSShh)    integer
    in  mode   set to -2                                integer
mode=3 : printable to stamp
   out  fdat1  cmc date-time stamp (old or new style)   integer
    in  dat2   date of the printable date (YYYYMMDD)    integer
    in  dat3   time of the printable date (HHMMSShh)    integer
    in  mode   set to 3                                 integer
mode=-3 : stamp to printable
    in  dat1   cmc date-time stamp (old or new style)   integer
   out  fdat2  date of the printable date (YYYYMMDD)    integer
   out  fdat3  time of the printable date (HHMMSShh)    integer
    in  mode   set to -3                                integer
mode=4 : 14 word old style DATE array to stamp and array(14)
   out  fdat1  CMC date-time stamp (old or new style)   integer
    in  dat2   14 word old style DATE array             integer
    in  dat3   unused                                   integer
    in  mode   set to 4                                 integer
mode=-4 : stamp to 14 word old style DATE array
    in  dat1   CMC date-time stamp (old or new style)   integer
   out  fdat2  14 word old style DATE array             integer
   out  fdat3  unused                                   integer
    in  mode   set to -4                                integer
    """
    raise error('Not implemented yet')

#  Functions not used in rpnstd.py nor rpn_helper.py
    
## {"fstsui",	(PyCFunction)Fstdc_fstsui,	METH_VARARGS,	Fstdc_fstsui__doc__},
## {"datematch",	(PyCFunction)Fstdc_datematch,	METH_VARARGS,	Fstdc_datematch__doc__},
## {"ConvertP2Ip",(PyCFunction)Fstdc_ConvertP2Ip,METH_VARARGS,	Fstdc_ConvertP2Ip__doc__},
## {"ConvertIp2P",(PyCFunction)Fstdc_ConvertIp2P,METH_VARARGS,	Fstdc_ConvertIp2P__doc__},
## {"EncodeIp",(PyCFunction)Fstdc_EncodeIp,METH_VARARGS,	Fstdc_EncodeIp__doc__},
## {"DecodeIp",(PyCFunction)Fstdc_DecodeIp,METH_VARARGS,	Fstdc_DecodeIp__doc__},
## {"ezgetlalo",	(PyCFunction)Fstdc_ezgetlalo,	METH_VARARGS,	Fstdc_ezgetlalo__doc__},
## {"ezgetopt", (PyCFunction) Fstdc_ezgetopt, METH_VARARGS, Fstdc_ezgetopt__doc__},
## {"ezsetopt", (PyCFunction) Fstdc_ezsetopt, METH_VARARGS, Fstdc_ezsetopt__doc__},
## {"ezgetval", (PyCFunction) Fstdc_ezgetval, METH_VARARGS, Fstdc_ezgetval__doc__},
## {"ezsetval", (PyCFunction) Fstdc_ezsetval, METH_VARARGS, Fstdc_ezsetval__doc__},

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
