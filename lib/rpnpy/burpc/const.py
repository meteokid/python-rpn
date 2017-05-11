#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc.const defines constants for the burpc module

Notes:
    This module is a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]]
    documentation for more details.

See Also:
    rpnpy.burpc.brpobj
    rpnpy.burpc.base
    rpnpy.burpc.proto

Details:
    See Source Code
"""
#import ctypes as _ct
import numpy  as _np
## import numpy.ctypeslib as _npc
import rpnpy.librmn.all as _rmn

##DETAILS_START
#== Constants Details ==
#<source lang="python">

## File access mode specifiers
BRP_FILE_READ   = 'r'
BRP_FILE_WRITE  = 'w'
BRP_FILE_APPEND = 'a'

BRP_FILEMODE2FST = {
    'r' : (_rmn.FST_RO,     _rmn.BURP_MODE_READ),
    'w' : (_rmn.FST_RW,     _rmn.BURP_MODE_CREATE),
    'a' : (_rmn.FST_RW_OLD, _rmn.BURP_MODE_APPEND)
    }
BRP_FILEMODE2FST_INV = dict([
    (v[1], k) for k, v in BRP_FILEMODE2FST.items()
    ])

## units conversion mode to be used with brp_convertblk,
## actually, it is to and from floating points with bufr integers
BRP_BUFR_to_MKSA  = 0
BRP_MKSA_to_BUFR  = 1
BRP_END_BURP_FILE = 0

## how data is stored in a data block
##  with STORE_INTEGER, calls to brp_convertblk can be omitted
##  with STORE_FLOAT or STORE_DOUBLE, brp_convertblk must be called
##  to write out data correctly
BRP_STORE_INTEGER = 'I'
BRP_STORE_FLOAT   = 'F'
BRP_STORE_DOUBLE  = 'D'
BRP_STORE_CHAR    = 'C'

## numpy dtype equivalent to BRP_STORE_TYPE
BRP_STORE_TYPE2NUMPY = {
    BRP_STORE_INTEGER : _np.int32,
    BRP_STORE_FLOAT   : _np.float32,
    BRP_STORE_DOUBLE  : _np.float64,
    BRP_STORE_CHAR    : _np.uint8   #TODO: check this
    }
## BRP_STORE_TYPE equivalent to numpy dtype
BRP_STORE_NUMPY2TYPE = dict([(v, k) for k, v in BRP_STORE_TYPE2NUMPY.items()])

## constants for specifying datyp
BRP_DATYP_BITSTREAM = 0
BRP_DATYP_UINTEGER  = 2
BRP_DATYP_CHAR      = 3
BRP_DATYP_INTEGER   = 4
BRP_DATYP_UCASECHAR = 5

BRP_BSTP_RESIDUS = 10
BRP_BFAM_RES_O_A = 12
BRP_BFAM_RES_O_I = 13
BRP_BFAM_RES_O_P = 14

## errors codes as returned by functions underneath BURP API
BRP_ERR_OPEN_FILE    = -1
BRP_ERR_CLOSE_FILE   = -2
BRP_ERR_TM_OPEN_FILE = -3
BRP_ERR_DAMAGED_FILE = -4
BRP_ERR_IUN          = -5
BRP_ERR_INVALID_FILE = -6
BRP_ERR_WRITE        = -7
BRP_ERR_NPG_REPT     = -8
BRP_ERR_HANDLE       = -9
BRP_ERR_SPEC_REC     = -10
BRP_ERR_DEL_REC      = -11
BRP_ERR_EXIST_REC    = -12
BRP_ERR_INIT_REC     = -13
BRP_ERR_UPDATE_REC   = -14
BRP_ERR_IDTYP        = -15
BRP_ERR_DATYP        = -16
BRP_ERR_DESC_KEY     = -17
BRP_ERR_ADDR_BITPOS  = -18
BRP_ERR_BUFFER_NSIZE = -19
BRP_ERR_OPTVAL_NAME  = -20
BRP_ERR_NOT_RPT_FILE = -30
BRP_ERR_OPEN_MODE    = -31
BRP_ERR_TM_SUP       = -32
BRP_ERR_BKNO         = -33
BRP_ERR_OPTNAME       = -34

## some usefull constants
BRP_VAR_NON_INIT = -1
BRP_ALL_STATION  = "*********"
BRP_STNID_STRLEN = 10

#</source>
##DETAILS_END


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
