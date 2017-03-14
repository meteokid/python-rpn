#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a4/intel13sp1u2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc is a ctypes import of burp_c's library (libburp_c_shared.so)
 
The burpc.proto python module includes ctypes prototypes for 
burp_c's libburp_c C functions.

Warning:
    Please use with caution.
    The functions in this module are actual C funtions and must thus be called
    as such with appropriate argument typing and dereferencing.
    It is highly advised in a python program to prefer the use of the
    python wrapper found in
    * rpnpy.burp.base

Notes:
    The functions described below are a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] documentation for more details.

See Also:
    rpnpy.burpc.base
    rpnpy.burpc.const

Details:
    See Source Code

##DETAILS_START
== Functions C Prototypes ==

<source lang="python">

 ## c_vgd_construct():
 ##    Returns a NOT fully initialized VGridDescriptor instance
 ##    Proto:
 ##       vgrid_descriptor* c_vgd_construct();
 ##    Args:
 ##       None
 ##    Returns:
 ##       POINTER(VGridDescriptor) : a pointer to a new VGridDescriptor object

</source>
##DETAILS_END
"""

import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc

from . import libburpc
from . import const as _cst


class BURP_RPT(_ct.Structure):
    """
    Python Class equivalenet of the burp_c's BURP_RPT C structure to hold
    the BURP report data

    typedef struct {
        int  *buffer;
        int  handle;
        int  nsize;
        int  temps;
        int  flgs;
        char stnid[10];
        int  idtype;
        int  lati;
        int  longi;
        int  dx;
        int  dy;
        int  elev;
        int  drnd;
        int  date;
        int  oars;
        int  runn;
        int  nblk;
        int  *sup;
        int  nsup;
        int  *xaux;
        int  nxaux;
        int  lngr;
        int  init_hdr; /* for internal use only */
    } BURP_RPT ;


    To get an instance of a pointer to this class you may use the
    provided functions:

    myBURP_RPTptr = rpnpy.burpc.propo.c_burp_rpt_construct()

    See Also:
       c_burp_rpt_construct
       c_burp_blk_construct
       BURP_BLK
   """
    _fields_ = [
        ("buffer", _ct.POINTER(_ct.c_int)),
        ("handle", _ct.c_int),
        ("nsize",  _ct.c_int),
        ("temps",  _ct.c_int),
        ("flgs",   _ct.c_int),
        ("stnid",  _ct.c_char * _cst.BRP_STNID_STRLEN),
        ("idtype", _ct.c_int),
        ("lati",   _ct.c_int),
        ("longi",  _ct.c_int),
        ("dx",     _ct.c_int),
        ("dy",     _ct.c_int),
        ("elev",   _ct.c_int),
        ("drnd",   _ct.c_int),
        ("date",   _ct.c_int),
        ("oars",   _ct.c_int),
        ("runn",   _ct.c_int),
        ("nblk",   _ct.c_int),
        ("*sup",   _ct.POINTER(_ct.c_int)),
        ("nsup",   _ct.c_int),
        ("*xaux",  _ct.POINTER(_ct.c_int)),
        ("nxaux",  _ct.c_int),
        ("lngr",   _ct.c_int),
        ("init_hdr", _ct.c_int)  ## for internal use only
        ]
    
    def __str__(self):
       return self.__class__.__name__ + str([x[0] + '=' + str(self.__getattribute__(x[0])) for x in self._fields_])
       ## s = self.__class__.__name__ + '('
       ## l = [y[0] for y in self._fields_]
       ## l.sort()
       ## for x in l:
       ##     s += x + '=' + str(self.__getattribute__(x)) + ', '
       ## s += ')'
       ## return s

    def __repr__(self):
       #return self.__class__.__name__ + str(self)
       return self.__class__.__name__ + repr([x[0] + '=' + repr(self.__getattribute__(x[0])) for x in self._fields_])


class BURP_BLK(_ct.Structure):
    """
    Python Class equivalenet of the burp_c's BURP_BLK C structure to hold
    the BURP block data

    typedef struct {
        int    bkno;
        int    nele;
        int    nval;
        int    nt;
        int    bfam;
        int    bdesc;
        int    btyp;
        int    bknat;
        int    bktyp;
        int    bkstp;
        int    nbit;
        int    bit0;
        int    datyp;
        char   store_type;
        int    *lstele;
        int    *dlstele;
        int    *tblval;
        float  *rval;
        double *drval;
        char   *charval;
        int    max_nval, max_nele, max_nt;
        int    max_len;
    } BURP_BLK ;


    To get an instance of a pointer to this class you may use the
    provided functions:

    myBURP_BLKptr = rpnpy.burpc.propo.c_burp_blk_construct()

    See Also:
       c_burp_rpt_construct
       c_burp_blk_construct
       BURP_RPT
   """
    _fields_ = [
        ("bkno",  _ct.c_int),
        ("nele",  _ct.c_int),
        ("nval",  _ct.c_int),
        ("nt",    _ct.c_int),
        ("bfam",  _ct.c_int),
        ("bdesc", _ct.c_int),
        ("btyp",  _ct.c_int),
        ("bknat", _ct.c_int),
        ("bktyp", _ct.c_int),
        ("bkstp", _ct.c_int),
        ("nbit",  _ct.c_int),
        ("bit0",  _ct.c_int),
        ("datyp", _ct.c_int),
        ("store_type", _ct.c_char),
        ("lstele",  _ct.POINTER(_ct.c_int)),
        ("dlstele", _ct.POINTER(_ct.c_int)),
        ("tblval",  _ct.POINTER(_ct.c_int)),
        ("rval",  _ct.POINTER(_ct.c_float)),
        ("drval",  _ct.POINTER(_ct.c_double)),
        ("charval",  _ct.c_char_p),
        ("max_nval", _ct.c_int),
        ("max_nele", _ct.c_int),
        ("max_nt", _ct.c_int),
        ("max_len", _ct.c_int),
        ]
    
    def __str__(self):
       return self.__class__.__name__ + str([x[0] + '=' + str(self.__getattribute__(x[0])) for x in self._fields_])
       ## s = self.__class__.__name__ + '('
       ## l = [y[0] for y in self._fields_]
       ## l.sort()
       ## for x in l:
       ##     s += x + '=' + str(self.__getattribute__(x)) + ', '
       ## s += ')'
       ## return s

    def __repr__(self):
       #return self.__class__.__name__ + str(self)
       return self.__class__.__name__ + repr([x[0] + '=' + repr(self.__getattribute__(x[0])) for x in self._fields_])


c_burp_rpt_construct = _ct.POINTER(BURP_RPT)

c_burp_blk_construct = _ct.POINTER(BURP_BLK)


## /*
##  * Macros for getting values of a Report
##  * acces to field in the BURP_RPT structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  * current definition do not cover all available field
##  * since they are not used by anyone yet.
##  */
## #define  RPT_HANDLE(rpt)        ((rpt)/**/->handle)
## #define  RPT_NSIZE(rpt)         ((rpt)/**/->nsize)
## #define  RPT_TEMPS(rpt)         ((rpt)/**/->temps)
## #define  RPT_FLGS(rpt)          ((rpt)/**/->flgs)
## #define  RPT_STNID(rpt)         ((rpt)/**/->stnid)
## #define  RPT_IDTYP(rpt)         ((rpt)/**/->idtype)
## #define  RPT_LATI(rpt)          ((rpt)/**/->lati)
## #define  RPT_LONG(rpt)          ((rpt)/**/->longi)
## #define  RPT_DX(rpt)            ((rpt)/**/->dx)
## #define  RPT_DY(rpt)            ((rpt)/**/->dy)
## #define  RPT_ELEV(rpt)          ((rpt)/**/->elev)
## #define  RPT_DRND(rpt)          ((rpt)/**/->drnd)
## #define  RPT_DATE(rpt)          ((rpt)/**/->date)
## #define  RPT_OARS(rpt)          ((rpt)/**/->oars)
## #define  RPT_RUNN(rpt)          ((rpt)/**/->runn)
## #define  RPT_NBLK(rpt)          ((rpt)/**/->nblk)
## #define  RPT_LNGR(rpt)          ((rpt)/**/->lngr)


## /*
##  * Macros for setting values of a Report
##  * acces to field in the BURP_RPT structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  * current definition do not cover all available field
##  * since they are not used by anyone yet.
##  */
## /* for internal use only */
## extern  void       brp_setstnid( BURP_RPT *rpt, const char *stnid );

## #define  RPT_SetHANDLE(rpt,val)    (rpt)/**/->handle=val
## #define  RPT_SetTEMPS(rpt,val)     (rpt)/**/->temps=val
## #define  RPT_SetFLGS(rpt,val)      (rpt)/**/->flgs=val
## #define  RPT_SetSTNID(rpt,val)     brp_setstnid(rpt,val)
## #define  RPT_SetIDTYP(rpt,val)     (rpt)/**/->idtype=val
## #define  RPT_SetLATI(rpt,val)      (rpt)/**/->lati=val
## #define  RPT_SetLONG(rpt,val)      (rpt)/**/->longi=val
## #define  RPT_SetDX(rpt,val)        (rpt)/**/->dx=val
## #define  RPT_SetDY(rpt,val)        (rpt)/**/->dy=val
## #define  RPT_SetELEV(rpt,val)      (rpt)/**/->elev=val
## #define  RPT_SetDRND(rpt,val)      (rpt)/**/->drnd=val
## #define  RPT_SetDATE(rpt,val)      (rpt)/**/->date=val
## #define  RPT_SetOARS(rpt,val)      (rpt)/**/->oars=val
## #define  RPT_SetRUNN(rpt,val)      (rpt)/**/->runn=val


## /*
##  * Macros for getting values of a Block
##  * acces to field in the BURP_BLK structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  */

## #define  BLK_BKNO(blk)             ((blk)/**/->bkno)
## #define  BLK_NELE(blk)             ((blk)/**/->nele)
## #define  BLK_NVAL(blk)             ((blk)/**/->nval)
## #define  BLK_NT(blk)               ((blk)/**/->nt)
## #define  BLK_BFAM(blk)             ((blk)/**/->bfam)
## #define  BLK_BDESC(blk)            ((blk)/**/->bdesc)
## #define  BLK_BTYP(blk)             ((blk)/**/->btyp)
## #define  BLK_BKNAT(blk)            ((blk)/**/->bknat)
## #define  BLK_BKTYP(blk)            ((blk)/**/->bktyp)
## #define  BLK_BKSTP(blk)            ((blk)/**/->bkstp)
## #define  BLK_NBIT(blk)             ((blk)/**/->nbit)
## #define  BLK_BIT0(blk)             ((blk)/**/->bit0)
## #define  BLK_DATYP(blk)            ((blk)/**/->datyp)
## #define  BLK_Data(blk)             ((blk)/**/->data)
## #define  BLK_DLSTELE(blk,e)        ((blk)->dlstele[e])
## #define  BLK_LSTELE(blk,e)         ((blk)->lstele[e])
## #define  BLK_TBLVAL(blk,e,v,t)     (blk)->tblval[(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
## #define  BLK_RVAL(blk,e,v,t)       (blk)->rval[  (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
## #define  BLK_DVAL(blk,e,v,t)       (blk)->drval[ (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
## #define  BLK_CVAL(blk,l,c)         ((blk)->charval[ (l)*((blk)->nt)+(c)  ])
## #define  BLK_STORE_TYPE(blk)       ((blk)->store_type)

## /*
##  * Macros for setting values of a Block
##  * acces to field in the BURP_BLK structure should be accessed
##  * through these Macros Only
##  * access without these macros would be at your own risk
##  */
## #define  BLK_SetNELE(blk,val)           (blk)->nele=val
## #define  BLK_SetNVAL(blk,val)           (blk)->nval=val
## #define  BLK_SetNT(blk,val)             (blk)->nt=val

## #define  BLK_SetBKNO(blk,val)           (blk)/**/->bkno=val
## #define  BLK_SetBFAM(blk,val)           (blk)/**/->bfam=val
## #define  BLK_SetBDESC(blk,val)          (blk)/**/->bdesc=val
## #define  BLK_SetBTYP(blk,val)           (blk)/**/->btyp=val
## #define  BLK_SetBKNAT(blk,val)          (blk)/**/->bknat=val
## #define  BLK_SetBKTYP(blk,val)          (blk)/**/->bktyp=val
## #define  BLK_SetBKSTP(blk,val)          (blk)/**/->bkstp=val
## #define  BLK_SetNBIT(blk,val)           (blk)/**/->nbit=val
## #define  BLK_SetDATYP(blk,val)          (blk)/**/->datyp=val
## #define  BLK_SetDVAL(blk,e,v,t,val)     (blk)->drval [(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
## #define  BLK_SetTBLVAL(blk,e,v,t,val)   (blk)->tblval[(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
## #define  BLK_SetRVAL(blk,e,v,t,val)     (blk)->rval[  (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
## #define  BLK_SetCVAL(blk,l,c,val)       (blk)->charval[ (l)*((blk)->nt)+(c)  ]=val;
## #define  BLK_SetLSTELE(blk,i,val)       (blk)->lstele[i]=val
## #define  BLK_SetDLSTELE(blk,i,val)      (blk)->dlstele[i]=val
## #define  BLK_SetSTORE_TYPE(blk,val)     (blk)->store_type=val

## /*
##  * allocators and constructors
##  */
## extern  BURP_BLK  *brp_newblk( void );
## extern  BURP_RPT  *brp_newrpt( void );
## extern  void       brp_allocrpt( BURP_RPT *rpt, int  nsize );
## extern  void       brp_allocblk( BURP_BLK *blk, int  nele, int nval, int nt );

## /*
##  *  find elements
##  */
## extern  int        brp_searchdlste( int  code, BURP_BLK *blk );
## /*
##  * destructors and deallocators
##  */
## extern  void       brp_freeblk( BURP_BLK *blk );
## extern  void       brp_freerpt( BURP_RPT *rpt );

## /* for internal use only */
## extern  void       brp_freebuf(BURP_RPT *rpt);
## extern  void       brp_freedata( BURP_BLK *blk );
## /*
##  * reinitializers
##  */
## extern  void       brp_clrblk( BURP_BLK  *blk );
## extern  void       brp_clrblkv(BURP_BLK  *bblk, float val);
## extern  void       brp_clrrpt( BURP_RPT *rpt );


## /* reset blk and rpt headers to default as initialised
##  * in brp_newblk and brp_newblk
##  */
## extern  void       brp_resetrpthdr( BURP_RPT *rpt );
## extern  void       brp_resetblkhdr( BURP_BLK *blk );
## /*
##  * converters
##  */
## extern  int        brp_encodeblk( BURP_BLK  *blk );
## extern  int        brp_safe_convertblk( BURP_BLK  *blk, int mode );
## extern  int        brp_convertblk( BURP_BLK  *blk, int mode );
## /*
##  * find report and block before reading them
##  */
## extern  int        brp_findblk( BURP_BLK  *blk, BURP_RPT  *rpt );
## extern  int        brp_findrpt( int iun, BURP_RPT *rpt );
## /*
##  * read in data
##  */
## extern  int        brp_getrpt( int iun, int handle, BURP_RPT  *rpt );
## extern  int        brp_safe_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);
## extern  int        brp_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);

## /* brp_readblk same as brp_getblk() but the BLK_RVAL(blk,e,v,t) values
##    are not available as conversion are not done. function to use when readig burp and
##    there is no need to work with real values
## */
## extern  int        brp_readblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt, int);

## /*
##  * read only header
##  */
## extern  int        brp_rdrpthdr(int handle, BURP_RPT *rpt);
## extern  int        brp_rdblkhdr(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);

## /* prepare a report for writing */
## extern  int        brp_initrpthdr( int iun, BURP_RPT *rpt );
## /* prepare a report for writing alias of brp_initrpthdr */
## extern  int        brp_putrpthdr( int iun, BURP_RPT *rpt );
## /* add new blocks into a report */
## extern  int        brp_putblk( BURP_RPT *rpt, BURP_BLK *blk );
## /* write out to a file */
## extern  int        brp_writerpt( int iun, BURP_RPT *rpt, int where );
## /* modify only the header of a report */
## extern  int        brp_updrpthdr( int iun, BURP_RPT *rpt );

## /*
##  * return the floating point constant used for missing values
##  */
## extern  float      brp_msngval(void);

## /*
##  * utilities
##  */
## /*  copy rpt header */
## extern void        brp_copyrpthdr( BURP_RPT * dest, const BURP_RPT *source);
## /*  copy the whole rpt  */
## extern void        brp_copyrpt( BURP_RPT * dest, const BURP_RPT *source);
## /*  resize the  rpt with newsize to add blocks  */
## extern void        brp_resizerpt( BURP_RPT * dest, int NewSize);
## /* duplicate block */
## extern void        brp_copyblk( BURP_BLK *dest, const BURP_BLK *source);
## /* resize  block */
## extern void        brp_resizeblk( BURP_BLK *source,int NEW_ele, int NEW_nval, int NEW_nt);
## extern void        brp_resizeblk_v2( BURP_BLK **source ,int nele, int nval, int nt);

## /*
##  * Opening files
##  */

## extern  int        brp_open(int  iun, const char *filename, char *op);
## extern  int        brp_close(int iun);

## /*
##  * deleting reports and blocks
##  */

## extern  int        brp_delblk(BURP_RPT *rpt, const BURP_BLK * blk);
## extern  int        brp_delrpt(BURP_RPT * rpt);

## /*
##  * burp rpn option functions
##  */
## extern  int        brp_SetOptFloat(char* opt, float val);
## extern  int        brp_SetOptChar (char* opt, char * val);
## extern  float      brp_msngval (void);
## /*
##  * burp rpn functions
##  */
## extern  int        c_mrfmxl ( int iun );
## extern  int        c_mrfnbr ( int iun );
## extern  int        c_mrbdel ( int *buf, int bkno);


if __name__ == "__main__":
    pass #print vgd version 
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
