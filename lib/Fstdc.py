#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module Fstdc is a backward compatibility layer for rpnpy version 1.3 and older.
This Module is deprecated in favor of rpnpy.librmn.all
"""
import ctypes as _ct
import numpy  as _np
## from scipy import interpolate
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
    """(version, lastUpdate) = Fstdc.version()
    """
    return (rpn_version.__VERSION__, rpn_version.__LASTUPDATE__)


def fstouv(iunit, filename, options):
    """Interface to fstouv and fnom to open a RPN 2000 Standard File
    iunit = Fstdc.fstouv(iunit, filename, options)
    @param iunit unit number of the file handle, 0 for a new one (int)
    @param filename (string)
    @param option type of file and R/W options (string)
    @return File unit number (int), NULL on error
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        iunit = _rmn.fnom(filename, options, iunit)
        _rmn.fstouv(iunit, options)
        return iunit
    except:
        raise error("Failed to open file: " + filename)


def fstfrm(iunit):
    """Close a RPN 2000 Standard File (Interface to fstfrm+fclos to close) 
    Fstdc.fstfrm(iunit)
    @param iunit file unit number handle returned by Fstdc_fstouv (int)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        _rmn.fstfrm(iunit)
        _rmn.fclos(iunit)
    except:
        raise error("Failed closing file")


def level_to_ip1(level_list, kind):
    """Encode level value to ip1 (Interface to convip)
    myip1List = Fstdc.level_to_ip1(level_list, kind) 
    @param level_list list of level values (list of float)
    @param kind type of level (int)
    @return [(ip1new, ip1old), ...] (list of tuple of int)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        return [(_rmn.convertIp(CONVIP_STYLE_NEW, lvl, kind),
                 _rmn.convertIp(CONVIP_STYLE_OLD, lvl, kind))
                 for lvl in level_list]
    except:
        raise error("")


def ip1_to_level(ip1_list):
    """Decode ip1 to level type, value (Interface to convip)
    myLevelList = Fstdc.ip1_to_level(ip1_list)
    @param tuple/list of ip1 values to decode
    @return list of tuple (level, kind)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        return [_rmn.convertIp(CONVIP_IP2P_DEFAULT, ip1) for ip1 in ip1_list]
    except:
        raise error("")


def cxgaig(grtyp, xg1, xg2, xg3, xg4):
    """Encode grid descriptors (Interface to cxgaig)
    (ig1, ig2, ig3, ig4) = Fstdc.cxgaig(grtyp, xg1, xg2, xg3, xg4) 
    @param ...TODO...
    @return (ig1, ig2, ig3, ig4)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        return _rmn.cxgaig(grtyp, xg1, xg2, xg3, xg4)
    except:
        raise error("")


def cigaxg(grtyp, ig1, ig2, ig3, ig4):
    """Decode grid descriptors (Interface to cigaxg)
    (xg1, xg2, xg3, xg4) = Fstdc.cigaxg(grtyp, ig1, ig2, ig3, ig4)
    @param ...TODO...
    @return (xg1, xg2, xg3, xg4)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        return _rmn.cigaxg(grtyp, ig1, ig2, ig3, ig4)
    except:
        raise error("")


def fstvoi(iunit, option):
    """Print a list view a RPN Standard File rec (Interface to fstvoi)
    Fstdc.fstvoi(iunit, option)
    @param iunit file unit number handle returned by Fstdc_fstouv (int)
    @param option 'STYLE_OLD' or 'STYLE_NEW' (sting)
    @return None
    @exception TypeError
    """
    try:
        _rmn.fstvoi(iunit, option)
    except:
        raise error("")


def fstluk(ihandle):
    """Read record data on file (Interface to fstluk)
    myRecDataDict = Fstdc.fstluk(ihandle)
    @param ihandle record handle (int) 
    @return record data (numpy.ndarray)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        return _rmn.fstluk(ihandle)['d']
    except:
        raise error("")


def fstinl(iunit, nomvar, typvar, etiket, ip1, ip2, ip3, datev):
    """Find all records matching provided criterias (Interface to fstinl)
    Warning: list is limited to the first 50000 records in a file, subsequent matches raises Fstdc.tooManyRecError and are ignored.
    recList = Fstdc.fstinl(iunit, nomvar, typvar, etiket, ip1, ip2, ip3, datev)
    param iunit file unit number handle returned by Fstdc_fstouv (int)
    @param nomvar select according to var name, blank==wildcard (string)
    @param typvar select according to var type, blank==wildcard (string)
    @param etiket select according to etiket, blank==wildcard (string)
    @param ip1 select according to ip1, -1==wildcard  (int)
    @param ip2 select according to ip2, -1==wildcard (int)
    @param ip3  select according to ip3, -1==wildcard (int)
    @param datev select according to date of validity, -1==wildcard (int)
    @returns python dict with handles+params of all matching records
    @exception TypeError
    @exception Fstdc.error
    @exception Fstdc.tooManyRecError
    """
    try:
        keylist = _rmn.fstinl(iunit, datev, etiket, ip1, ip2, ip3, typvar, nomvar)
    except:
        raise error('Problem getting record list')
    recParamList = []
    for k in keylist:
        recParams = _rmn.fstprm(k)
        recParams['handle'] = k
        recParams['nom']    = recParams['nomvar']
        recParams['type']   = recParams['typvar']
        recParams['datev']  = recParams['xtra1'] #TODO: Keep Fstdc original bug?
        recParamList.append(recParams)
    return recParamList


def fstinf(iunit, nomvar, typvar, etiket, ip1, ip2, ip3, datev, inhandle):
    """Find a record matching provided criterias (Interface to fstinf, dsfsui, fstinfx)
    recParamDict = Fstdc.fstinf(iunit, nomvar, typvar, etiket, ip1, ip2, ip3, datev, inhandle)
    @param iunit file unit number handle returned by Fstdc_fstouv (int)
    @param nomvar select according to var name, blank==wildcard (string)
    @param typvar select according to var type, blank==wildcard (string)
    @param etiket select according to etiket, blank==wildcard (string)
    @param ip1 select according to ip1, -1==wildcard  (int)
    @param ip2 select according to ip2, -1==wildcard (int)
    @param ip3  select according to ip3, -1==wildcard (int)
    @param datev select according to date of validity, -1==wildcard (int)
    @param inhandle selcation criterion; inhandle=-2:search with criterion from start of file; inhandle=-1==fstsui, use previously provided criterion to find the next matching one; inhandle>=0 search with criterion from provided rec-handle (int)
    @returns python dict with record handle + record params keys/values
    @exception TypeError
    @exception Fstdc.error
    """
    if inhandle < -1:
        mymatch = _rmn.fstinf(iunit, datev, etiket, ip1, ip2, ip3, typvar, nomvar)
    elif inhandle == -1:
        mymatch = _rmn.fstsui(iunit)
    else:
        mymatch = _rmn.fstinfx(inhandle, iunit, datev, etiket, ip1, ip2, ip3, typvar, nomvar)
    if not mymatch:
        raise error('No matching record')
    recParams = _rmn.fstprm(mymatch['key'])
    recParams['handle'] = recParams['key']
    recParams['nom']    = recParams['nomvar']
    recParams['type']   = recParams['typvar']
    recParams['datev']  = recParams['xtra1']  #TODO: Keep Fstdc original bug?
    return recParams


def fsteff(ihandle):
    """Erase a record (Interface to fsteff)
    Fstdc.fsteff(ihandle)
    @param ihandle handle of the record to erase (int)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        _rmn.fsteff(ihandle)
    except:
        raise error("")


def fstecr(data, iunit, nomvar, typvar, etiket, ip1, ip2, ip3, dateo, grtyp, ig1, ig2, ig3, ig4, deet, npas, nbits, datyp):
    """Write record data & meta(params) to file (Interface to fstecr), always append (no overwrite)
    Fstdc.fstecr(data, iunit, nomvar, typvar, etiket, ip1, ip2, ip3, dateo, grtyp, ig1, ig2, ig3, ig4, deet, npas, nbits, datyp)
    @param data array to be written to file (numpy.ndarray)
    @param iunit file unit number handle returned by Fstdc_fstouv (int)
    @param ... 
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        ni = data.shape[0]
        (nj, nk) = (1, 1)
        if len(data.shape) > 1:
            nj = data.shape[1]
        if len(data.shape) > 2:
            nk = data.shape[2]
    except:
        raise error("fstecr: cannot get data shape")
    mymeta = {
        'dateo' : dateo,
        'deet'  : deet,
        'npas'  : npas,
        'ni'    : ni,
        'nj'    : nj,
        'nk'    : nk,
        'nbits' : nbits,
        'datyp' : datyp,
        'ip1'   : ip1,
        'ip2'   : ip2,
        'ip3'   : ip3,
        'typvar': typvar,
        'nomvar': nomvar,
        'etiket': etiket,
        'grtyp' : grtyp,
        'ig1'   : ig1,
        'ig2'   : ig2,
        'ig3'   : ig3,
        'ig4'   : ig4
        }
    try:
        return _rmn.fstecr(iunit, data, mymeta, rewrite=False)
    except _rmn.FSTDError:
        raise error('Problem writing rec param/meta')


def fst_edit_dir(ihandle, date, deet, npas, ni, nj, nk, ip1, ip2, ip3, typvar, nomvar, etiket, grtyp, ig1, ig2, ig3, ig4, datyp):
    """Rewrite the parameters of an rec on file, data part is unchanged (Interface to fst_edit_dir)
    Fstdc.fst_edit_dir(ihandle, date, deet, npas, ni, nj, nk, ip1, ip2, ip3, typvar, nomvar, etiket, grtyp, ig1, ig2, ig3, ig4, datyp)
    @param ihandle record handle (int)
    @param ... 
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        _rmn.fst_edit_dir(ihandle, date, deet, npas, ni, nj, nk, ip1, ip2, ip3, typvar, nomvar, etiket, grtyp, ig1, ig2, ig3, ig4, datyp)
    except:
        raise error("fst_edit_dir: probleme updating rec meta")


def gdllfxy(xin, yin, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Get (latitude, longitude) pairs cooresponding to (x, y) grid coordinates
    (lat, lon) = Fstdc.gdllfxy(xin, yin,
        (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param xin, yin: input coordinates (as float32 numpy array)
    @param (ni, nj) .. (i0, j0): Grid definition parameters
    @return (lat, lon): Computed latitude and longitude coordinates
    """
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gdllfxy: Invalid Grid Desc")
    try:
        ll  = _rmn.gdllfxy(gid, xin, yin)
    except:
        raise error("gdllfxy: Problem computing lat, lon coor")
    return (ll['lat'], ['lon'])



def gdxyfll(latin, lonin, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Get (x, y) pairs cooresponding to (lat, lon) grid coordinates
    (x, y) = Fstdc.gdxyfll(latin, lonin,
        (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param latin, lonin: input coordinates (as float32 numpy array)
    @param (ni, nj) .. (i0, j0): Grid definition parameters
    @return (x, y): Computed x and y coordinates (as floating point)
    """
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gdxyfll: Invalid Grid Desc")
    try:
        xyc  = _rmn.gdxyfll(gid, latin, lonin)
    except:
        raise error("gdxyfll: Problem computing x, y coor")
    return (xyc['x'], xyc['y'])


def gdwdfuv(uuin, vvin, lat, lon, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Translate grid-directed winds to (magnitude, direction) pairs
    (UV, WD) = Fstdc.gdwdfuv(uuin, vvin, lat, lon,
        (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param uuin, vvin -- Grid-directed wind fields
    @param lat, lon -- Latitude and longitude coordinates of those winds
    @param ni ... j0 -- grid definition parameters
    @return (UV, WD): Wind modulus and direction as numpy.ndarray
    """
    raise error('gdwdfuv: Not implemented yet')
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gdwdfuv: Invalid Grid Desc")
  

def gduvfwd(uvin, wdin, lat, lon, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Translate (magnitude, direction) winds to grid-directed
    (UV, WD) = Fstdc.gduvfwd(uvin, wdin, lat, lon,
        (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param uvin, wdin -- Grid-directed wind fields 
    @param lat, lon -- Latitude and longitude coordinates of those winds
    @param ni ... j0 -- grid definition parameters
    @return (UU, VV): Grid-directed winds as numpy.ndarray
    """
    raise error('gduvfwd: Not implemented yet')
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gduvfwd: Invalid Grid Desc")


def gdllval(uuin, vvin, lat, lon, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Interpolate scalar or vector fields to scattered (lat, lon) points
    vararg = Fstdc.gdllval(uuin, vvin, lat, lon,
        (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param (uuin, vvin): Fields to interpolate from; if vvin is None then
                        perform scalar interpolation
    @param lat, lon:  Latitude and longitude coordinates for interpolation
    @param ni ... j0: grid definition parameters
    @return Zout or (UU, VV): scalar or tuple of grid-directed,
                             vector-interpolated fields, as appropriate
    """
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gdllval: Invalid Grid Desc")
    try:
        if vvin == None:
            return _rmn.gdllsval(gid, lat, lon, uuin)
        else:
            return _rmn.gdllvval(gid, lat, lon, uuin, vvin)
    except:
        raise error("gdllval: Problem interpolating")


def gdxyval(uuin, vvin, x, y, nij, grtyp, refparam, xyAxis, hasAxis, ij0):
    """Interpolate scalar or vector fields to scattered (x, y) points
    vararg = Fstdc.gdxyval(uuin, vvin, x, y,
    (ni, nj), grtyp, (grref, ig1, ig2, ig3, ig4), (xs, ys), hasAxis, (i0, j0))
    @param (uuin, vvin): Fields to interpolate from; if vvin is None then
                        perform scalar interpolation
    @param x, y:  X and Y-coordinates for interpolation
    @param ni ... j0: grid definition parameters
    @return Zout or (UU, VV): scalar or tuple of grid-directed,
                             vector-interpolated fields, as appropriate
    """
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("gdxyval: Invalid Grid Desc")
    try:
        if vvin == None:
            return _rmn.gdxysval(gid, x, y, uuin)
        else:
            return _rmn.gdxyvval(gid, x, y, uuin, vvin)
    except:
        raise error("gdxyval: Problem interpolating")


def ezinterp(arrayin, arrayin2,
             src_nij, src_grtyp, src_refparam, src_xyAxis, src_hasAxis, src_ij0,
             dst_nij, dst_grtyp, dst_refparam, dst_xyAxis, dst_hasAxis, dst_ij0,
             isVect):
    """Interpolate from one grid to another
    newArray = Fstdc.ezinterp(arrayin, arrayin2,
        (niS, njS), grtypS, (grrefS, ig1S, ig2S, ig3S, ig4S), (xsS, ysS), hasSrcAxis, (i0S, j0S),
        (niD, njD), grtypD, (grrefD, ig1D, ig2D, ig3D, ig4D), (xsD, ysD), hasDstAxis, (i0D, j0D),
        isVect)
    @param ...TODO...
    @return interpolated data (numpy.ndarray)
    @exception TypeError
    @exception Fstdc.error
    """
    try:
        src_gid = _getGridHandle(src_nij[0], src_nij[1], src_grtyp,
                                 src_refparam[0], src_refparam[1], src_refparam[2],
                                 src_refparam[3], src_refparam[4],
                                 src_ij0[0], src_ij0[1], src_xyAxis[0], src_xyAxis[1])
    except:
        raise error("ezgetlalo: Invalid Source Grid Desc")
    try:
        dst_gid = _getGridHandle(dst_nij[0], dst_nij[1], dst_grtyp,
                                 dst_refparam[0], dst_refparam[1], dst_refparam[2],
                                 dst_refparam[3], dst_refparam[4],
                                 dst_ij0[0], dst_ij0[1], dst_xyAxis[0], dst_xyAxis[1])
    except:
        raise error("ezgetlalo: Invalid Source Grid Desc")
    try:
        gridsetid = _rmn.ezdefset(dst_gid, src_gid)
    except:
        raise error("Problem defining a grid interpolation set")
    try:
        if isVect:
            dst_data = _rmn.ezuvint(dst_gid, src_gid, arrayin, arrayin2)
        else:
            dst_data = _rmn.ezsint(dst_gid, src_gid, arrayin)
        return dst_data
    except:
        raise error("Interpolation problem in ezscint")


def incdatr(date1, nhours):
    """Increase CMC datetime stamp by a N hours (Interface to incdatr)
    date2 = Fstdc.incdatr(date1, nhours)
    @param date1 original CMC datetime stamp(int)
    @param nhours number of hours to increase the date (double)
    @return Increase CMC datetime stamp (int)
    @exception TypeError
    @exception Fstdc.error
    """
    return _rmn.incdatr(date1, nhours)


def difdatr(date1, date2):
    """Compute differenc between 2 CMC datatime stamps (Interface to difdatr)
    nhours = Fstdc.difdatr(date1, date2)
    @param date1 CMC datatime stamp (int)
    @param date2 CMC datatime stamp (int)
    @return number of hours = date2-date1 (float)
    @exception TypeError
    """
    return _rmn.difdatr(date1, date2)


def newdate(dat1, dat2, dat3, mode):
    """Convert data to/from printable format and CMC stamp (Interface to newdate)
    (fdat1, fdat2, fdat3) = Fstdc.newdate(dat1, dat2, dat3, mode)
    @param ...see newdate doc... 
    @return tuple with converted date values ...see newdate doc...
    @exception TypeError
    @exception Fstdc.error

1.1 ARGUMENTS
mode can take the following values:-3, -2, -1, 1, 2, 3
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
    cimode = _ct.c_int(mode)
    (cidate1, cidate2, cidate3) = (_ct.c_int(dat1), _ct.c_int(dat2), _ct.c_int(dat3))
    istat = _rmn.f_newdate(_ct.byref(cidate1), _ct.byref(cidate2), _ct.byref(cidate3), _ct.byref(cimode))
    if istat < 0:
        raise error('Problem converting date with newdate (mode={0}) ({1}, {2}, {3})'.format(mode, dat1, dat2, dat3))
    return (cidate1.value, cidate2.value, cidate3.value)


#  Functions not used in rpnstd.py nor rpn_helper.py


def ConvertP2Ip(pvalue, pkind, istyle):
    """Encoding of P (real value, kind) into IP123 RPN-STD files tags
        ip123 = Fstdc.ConvertP2Ip(pvalue, pkind, istyle)
        @param  pvalue, value to encode, units depends on the kind (float)
        @param  pkind,  kind of pvalue (int)
        @param  istyle, CONVIP_STYLE_NEW/OLD/DEFAULT (int)
        @return IP encoded value (int)
        @exception TypeError
        @exception Fstdc.error
    """
    return _rmn.convertIp(istyle, pvalue, pkind)

    
def ConvertIp2P(ip123, imode):
    """Decoding of IP123 RPN-STD files tags into P (real values, kind)
        (pvalue, pkind) = Fstdc.ConvertIp2P(ip123, imode)
        @param  ip123, IP encoded value (int)
        @param  imode, CONVIP_IP2P_DEFAULT or CONVIP_IP2P_31BITS (int)
        @return pvalue, real decoded value, units depends on the kind (float)
        @return pkind, kind of pvalue (int)
        @exception TypeError
        @exception Fstdc.error
    """
    return _rmn.convertIp(imode, ip123)


def EncodeIp(pvalues):
    """Encoding of real level and time values+kind into the ip1+ip2+ip3 files tags triolets
        (ip1, ip2, ip3) = Fstdc.EncodeIp(pvalues)
        @param  pvalues, real level and time values/intervals, units depends on the kind
                pvalues must have has the format of list/tuple of tuples
                [(rp1.v1, rp1.v2, rp1.kind), (rp2.v1, rp2.v2, rp2.kind), (rp3.v1, rp3.v2, rp3.kind)]
                where v1, v2 are float, kind is an int (named constant KIND_*)
                RP1 must contain a level (or a pair of levels) in the atmosphere
                RP2 must contain  a time (or a pair of times)
                RP3 may contain anything, RP3.hi will be ignored (if RP1 or RP2 contains a pair, RP3 is ignored)
                If RP1 is not a level or RP2 is not a time, Fstdc.error is raised
                If RP1 and RP2 both contains a range , Fstdc.error is raised
        @return IP encoded values, tuple of int
        @exception TypeError
        @exception Fstdc.error
    """
    try:
        ip123 = _rmn.EncodeIp(_rmn.listToFLOATIP(pvalues[0]),
                              _rmn.listToFLOATIP(pvalues[1]),
                              _rmn.listToFLOATIP(pvalues[2]))
        if min(ip123) < 0:
            raise error("Proleme encoding provided values to ip123 in EncodeIp")
        return ip123
    except:
        raise error("Proleme encoding provided values to ip123 in EncodeIp")


def DecodeIp(ip123):
    """Decoding of ip1+ip2+ip3 files tags triolets into level+times values or interval
        pvalues = Fstdc.DecodeIp([ip1, ip2, ip3])
        @param  [ip1, ip2, ip3], tuple/list of int
        @return pvalues, real decoded level and time values/intervals, units depends on the kind
                pvalues has the format of list/tuple of tuples
                [(rp1.v1, rp1.v2, rp1.kind), (rp2.v1, rp2.v2, rp2.kind), (rp3.v1, rp3.v2, rp3.kind)]
                where v1, v2 are float, kind is an int (named constant KIND_*)
                RP1 will contain a level (or a pair of levels in atmospheric ascending order) in the atmosphere
                RP2 will contain a time (or a pair of times in ascending order)
                RP3.v1 will be the same as RP3.v2 (if RP1 or RP2 contains a pair, RP3 is ignored)
        @exception TypeError
        @exception Fstdc.error
    """
    try:
        (rp1, rp2, rp3) = _rmn.DecodeIp(ip123[0], ip123[1], ip123[2])
        return [(rp1.v1, rp1.v2, rp1.kind), (rp2.v1, rp2.v2, rp2.kind), (rp3.v1, rp3.v2, rp3.kind)]
    except:
        raise error("Proleme decoding ip123 in DecodeIp")


def _getGridHandle(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, i0, j0, xs, ys):
    ## if (_isGridValid(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, i0, j0, xs, ys)>=0) #TODO
    gdid = -1
    i0b  = 0
    j0b  = 0
    grtypZ = "Z"
    grtypY = "Y"
    grtyp = grtyp.upper().strip()
    if grtyp == 'Z' or grtyp == '#':
        if grtyp == '#':
            i0b = i0-1
            j0b = j0-1
        gdid = _rmn.ezgdef_fmem(ni, nj, grtypZ, grref, ig1, ig2, ig3, ig4, xs[i0b:, 0], ys[0, j0b:]);
    elif grtyp == 'Y':
        gdid = _rmn.ezgdef_fmem(ni, nj, grtypY, grref, ig1, ig2, ig3, ig4, xs[i0b:, j0b:], ys[i0b:, j0b:]);
    else:
        gdid = _rmn.ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4);
    return gdid


def ezgetlalo(nij, grtyp, refparam, xyAxis, hasAxis, ij0, doCorners):
    """Get Lat-Lon of grid points centers and corners
        (lat, lon, clat, clon) = Fstdc.ezgetlalo((niS, njS), grtypS, (grrefS, ig1S, ig2S, ig3S, ig4S), (xsS, ysS), hasSrcAxis, (i0S, j0S), doCorners)
        @param ...TODO...
        @return tuple of (numpy.ndarray) with center lat/lon (lat, lon) and optionally corners lat/lon (clat, clon)
        @exception TypeError
        @exception Fstdc.error
    """
    try:
        gid = _getGridHandle(nij[0], nij[1], grtyp,
                             refparam[0], refparam[1], refparam[2],
                             refparam[3], refparam[4],
                             ij0[0], ij0[1], xyAxis[0], xyAxis[1])
    except:
        raise error("ezgetlalo: Invalid Grid Desc")
    try:    
        gridLaLo = _rmn.gdll(gid);
    except:
        raise error("ezgetlalo: Problem computing lat, lon in ezscint")
    if not doCorners:
        return (gridLaLo['lat'], gridLaLo['lon'])
    xyc  = _rmn.gdxyfll(gid, gridLaLo['lat'], gridLaLo['lon'])
    xc1  = xyc['x']
    yc1  = xyc['y']
    nij2 = (4, gridLaLo['lat'].shape[0], gridLaLo['lat'].shape[1])
    xc4  = _np.empty(nij2, dtype=_np.float32, order='F')
    yc4  = _np.empty(nij2, dtype=_np.float32, order='F')
    ## x = _np.arange(float(nij2[1]))
    ## y = _np.arange(float(nij2[2]))
    ## fx = interpolate.interp2d(x, y, xc1, kind='linear', copy=False)
    ## fy = interpolate.interp2d(x, y, yc1, kind='linear', copy=False)
    dij_corners = (
        (-0.5, -0.5),  #SW
        (-0.5, 0.5),  #NW
        ( 0.5, 0.5),  #NE
        ( 0.5, -0.5)   #SE
        )
    for icorner in range(4):
        di = dij_corners[icorner][0]
        dj = dij_corners[icorner][1]
        ## xnew = x.copy('FORTRAN') + dij_corners[icorner][0]
        ## ynew = y.copy('FORTRAN') + dij_corners[icorner][1]
        ## xc4[icorner, :, :] = fx(xnew, ynew)
        ## yc4[icorner, :, :] = fy(xnew, ynew)
        xc4[icorner, :, :] = xc1[:, :] + dij_corners[icorner][0]
        yc4[icorner, :, :] = yc1[:, :] + dij_corners[icorner][1]
    llc = _rmn.gdllfxy(gid, xc4, yc4)
    return (gridLaLo['lat'], gridLaLo['lon'], llc['lat'], llc['lon'])

    
def fstsui(iunit):
    """Find next record matching criterions (Interface to fstsui)
        recParamDict = Fstdc.fstsui(iunit)
        @param iunit file unit number handle returned by Fstdc_fstouv (int)
        @returns python dict with record handle + record params keys/values
        @exception TypeError
        @exception Fstdc.error
    """
    try:
        mymatch = _rmn.fstsui(iunit)
    except:
        raise error('Problem getting record list')
    if not mymatch:
        raise error('No matching record')
    recParams = _rmn.fstprm(mymatch['key'])
    recParams['handle'] = recParams['key']
    recParams['nom']    = recParams['nomvar']
    recParams['type']   = recParams['typvar']
    recParams['datev']  = recParams['xtra1']  #TODO: Keep Fstdc original bug?
    return recParams


def datematch(indate, dateRangeStart, dateRangeEnd, delta):
    """Determine if date stamp match search crieterias
        doesmatch = Fstdc.datematch(indate, dateRangeStart, dateRangeEnd, delta)
        @param indate Date to be check against, CMC datetime stamp (int)
        @param dateRangeStart, CMC datetime stamp (int) 
        @param dateRangeEnd, CMC datetime stamp (int)
        @param delta (float)
        @return 1:if date match; 0 otherwise
        @exception TypeError
    """
    if dateRangeEnd != -1:
        if _rmn.difdatr(indate, dateRangeEnd) > 0.:
            return 0
    toler  = 0.00023 #tolerance d'erreur de 5 sec
    nhours = 0.
    if dateRangeStart != -1:
        nhours = _rmn.difdatr(indate, dateRangeStart)
        if nhours < 0.:
            return 0
    else:
        if dateRangeEnd == -1:
            return 1
        nhours = _rmn.difdatr(dateRangeEnd, indate)
    modulo = nhours % delta
    if modulo < toler or (delta - modulo) < toler:
        return 1
    else:
        return 0


def ezgetopt(option):
    """Get the string representation of one of the internal ezscint options
        opt_val = Fstrc.ezgetopt(option)
        @type option: A string
        @param option: The option to query 
        @rtype: A string
        @return: The string result of the query
    """
    try:
        return ezgetopt(option, str)
    except:
        raise error('Problem getting ezopt value')


def ezsetopt(option, value):
    """Set the string representation of one of the internal ezscint options
        opt_val = Fstrc.ezsetopt(option, value)
        @type option, value: A string
        @param option: The ezscint option 
        @param value: The value to set
    """
    try:
        _rmn.ezsetopt(option, value)
    except:
        raise error('Problem setting ezopt value')


def ezgetval(option):
    """Get an internal ezscint float or integer value by keyword
        opt_val = Fstdc.ezgetval(option)
        @type option: A string
        @param option: The keyword of the option to retrieve
        @return: The value of the option as returned by ezget[i]val
    """
    try:
        option = option.lower().strip()
        if option == "weight_number" or option == "missing_points_tolerance":
            return ezgetopt(option, int)
        else:
            return ezgetopt(option, float)
    except:
        raise error('Problem getting ezopt value')


def ezsetval(option, value):
    """Set an internal ezscint float or integer value by keyword
        opt_val = Fstdc.ezgetval(option, value)
        @type option: A string
        @param option: The keyword of the option to retrieve
        @type value: Float or integer, as appropriate for the option
        @param value: The value to set
    """
    try:
        _rmn.ezsetopt(option, value)
    except:
        raise error('Problem setting ezopt value')


if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
