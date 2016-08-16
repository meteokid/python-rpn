#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Michael Sitwell <michael.sitwell@canada.ca>
# Copyright: LGPL 2.1

"""
Python interface for BUPR files. Contains wrappers for ctypes functions in
proto_burp and the BurpError class.
"""

from rpnpy.librmn import proto_burp as _rp


class BurpError(Exception):
    """ Exception raised for BurpFile errors. """

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

    def __init__(self,fnc_name,ier):
        istat = abs(ier)
        self.msg = "Error occured while executing %s" % fnc_name
        if istat in BurpError.error_codes.keys():
            self.msg += " - %s (ISTAT=%i)" % (BurpError.error_codes[istat],istat)

    def __str__(self):
        return repr(self.msg)



def mrfopc(optnom,opvalc):
    """ Initializes character options. """
    istat = _rp.c_mrfopc(optnom,opvalc)
    if istat!=0:
        raise BurpError('c_mrfopc',istat)
    return istat


def mrfopn(iun,mode):
    """ Opens a BURP file. """
    nrep = _rp.c_mrfopn(iun,mode)
    return nrep


def mrfcls(iun):
    """ Closes a BURP file. """
    istat = _rp.c_mrfcls(iun)
    if istat!=0:
        raise BurpError('c_mrfcls',istat)
    return istat


def mrfnbr(iun):
    """ Returns number of reports in file. """
    nrep = _rp.c_mrfnbr(iun)
    return nrep


def mrfmxl(iun):
    """ Returns lenght of longest report in file. """
    nmax = _rp.c_mrfmxl(iun)
    return nmax


def mrfloc(iun,handle,stnid,idtyp,lati,long,date,temps,sup,nsup):
    """ Locate position of report in file. """
    handle = _rp.c_mrfloc(iun,handle,stnid,idtyp,lati,long,date,temps,sup,nsup)
    return handle


def mrfget(handle,buf):
    """ Put report pointed to by handle into buffer. """
    istat = _rp.c_mrfget(handle,buf)
    if istat!=0:
        raise BurpError('c_mrfget',istat)
    return istat


def mrfput(iun,handle,buf):
    """ Write a report. """
    istat = _rp.c_mrfput(iun,handle,buf)
    if istat!=0:
        raise BurpError('c_mrfput',istat)
    return istat


def mrbhdr(buf,temps,flgs,stnid,idtyp,lat,lon,dx,dy,elev,drnd,date,oars,runn,nblk,sup,nsup,xaux,nxaux):
    """ Returns report header information. """
    istat = _rp.c_mrbhdr(buf,temps,flgs,stnid,idtyp,lat,lon,dx,dy,elev,drnd,date,oars,runn,nblk,sup,nsup,xaux,nxaux)
    if istat!=0:
        raise BurpError('c_mrbhdr',istat)
    return istat


def mrbprm(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp):
    """ Returns block header information. """
    istat = _rp.c_mrbprm(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp)
    if istat!=0:
        raise BurpError('c_mrbprm',istat)
    return istat


def mrbxtr(buf,bkno,lstele,tblval):
    """ Extract block of data from the buffer. """
    istat = _rp.c_mrbxtr(buf,bkno,lstele,tblval)
    if istat!=0:
        raise BurpError('c_mrbxtr',istat)
    return istat


def mrbdcl(liste,dliste,nele):
    """ Convert CMC codes to BUFR codes. """
    istat = _rp.c_mrbdcl(liste,dliste,nele)
    if istat!=0:
        raise BurpError('c_mrbdcl',istat)
    return istat


def mrbcvt(liste,tblval,rval,nele,nval,nt,mode):
    """ Convert real values to table values. """
    istat = _rp.c_mrbcvt(liste,tblval,rval,nele,nval,nt,mode)
    if istat!=0:
        raise BurpError('c_mrbcvt',istat)
    return istat


def mrbini(iun,buf,temps,flgs,stnid,idtp,lati,lon,dx,dy,elev,drnd,date,oars,runn,sup,nsup,xaux,nxaux):
    """ Writes report header. """
    istat = _rp.c_mrbini(iun,buf,temps,flgs,stnid,idtp,lati,lon,dx,dy,elev,drnd,date,oars,runn,sup,nsup,xaux,nxaux)
    if istat!=0:
        raise BurpError('c_mrbini',istat)
    return istat


def mrbcol(dliste,liste,nele):
    """ Convert BUFR codes to CMC codes. """
    istat = _rp.c_mrbcol(dliste,liste,nele)
    if istat!=0:
        raise BurpError('c_mrbcol',istat)
    return istat


def mrbadd(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp,lstele,tblval):
    """ Adds a block to a report. """
    istat = _rp.c_mrbadd(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp,lstele,tblval)
    if istat!=0:
        raise BurpError('c_mrbadd',istat)
    return istat

# =========================================================================

if __name__ == "__main__":
    print("Python interface for BUPR files.")
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;

