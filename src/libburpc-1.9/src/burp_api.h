/*
 *
 *  file      :  burp_api.h
 *
 *  author    :  Vanh Souvanlasy
 *
 *  revision  : Hamid.Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  THIS FILE CONTAINS ALL THE DECLARATIONS
 *               NEEDED FOR THE BURP APPLICATION PROGRAMMING INTERFACE
 *
 *  Copyright (C) 2019  Government of Canada
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation, version
 *  2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA 02110-1301 USA
 *
 *  The Environment and Climate Change Canada open source licensing
 *  due diligence process form for libburpc was approved on December
 *  21, 2018.
 *
 */


#ifndef _burp_api_h_
#define _burp_api_h_
#include "declare.h"

 __BEGIN_DECLS

/****************************************************************************/

/*
 * units conversion mode to be used with brp_convertblk,
 * actually, it is to and from floating points with bufr integers
 */

#define  BUFR_to_MKSA           0
#define  MKSA_to_BUFR           1
#define  END_BURP_FILE          0

/*
 * how data is stored in a data block
 * with STORE_INTEGER, calls to brp_convertblk can be omitted
 * with STORE_FLOAT or STORE_DOUBLE, brp_convertblk must be called
 * to write out data correctly
 */
#define  STORE_INTEGER     'I'
#define  STORE_FLOAT       'F'
#define  STORE_DOUBLE      'D'
#define  STORE_CHAR        'C'

/*
 * constants for specifying datyp
 */
#define  DATYP_BITSTREAM    0
#define  DATYP_UINTEGER     2
#define  DATYP_CHAR         3
#define  DATYP_INTEGER      4
#define  DATYP_UCASECHAR    5

#define  BSTP_RESIDUS       10
#define  BFAM_RES_O_A       12
#define  BFAM_RES_O_I       13
#define  BFAM_RES_O_P       14

/*
** errors codes as returned by functions underneath BURP API
*/
#define  ERR_OPEN_FILE        -1
#define  ERR_CLOSE_FILE       -2
#define  ERR_TM_OPEN_FILE     -3
#define  ERR_DAMAGED_FILE     -4
#define  ERR_IUN              -5
#define  ERR_INVALID_FILE     -6
#define  ERR_WRITE            -7
#define  ERR_NPG_REPT         -8
#define  ERR_HANDLE           -9
#define  ERR_SPEC_REC         -10
#define  ERR_DEL_REC          -11
#define  ERR_EXIST_REC        -12
#define  ERR_INIT_REC         -13
#define  ERR_UPDATE_REC       -14
#define  ERR_IDTYP            -15
#define  ERR_DATYP            -16
#define  ERR_DESC_KEY         -17
#define  ERR_ADDR_BITPOS      -18
#define  ERR_BUFFER_NSIZE     -19
#define  ERR_OPTVAL_NAME      -20
#define  ERR_NOT_RPT_FILE     -30
#define  ERR_OPEN_MODE        -31
#define  ERR_TM_SUP           -32
#define  ERR_BKNO             -33
#define  ERR_OPTNAME          -34

/*
 * some usefull constants
 */
#define  VAR_NON_INIT  -1
#define  ALL_STATION     "*********"

/*
 * structure definition of a BURP Report
 */
typedef struct
{
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

typedef  struct
{
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


/*
 * Macros for getting values of a Report
 * acces to field in the BURP_RPT structure should be accessed
 * through these Macros Only
 * access without these macros would be at your own risk
 * current definition do not cover all available field
 * since they are not used by anyone yet.
 */
#define  RPT_HANDLE(rpt)        ((rpt)/**/->handle)
#define  RPT_NSIZE(rpt)         ((rpt)/**/->nsize)
#define  RPT_TEMPS(rpt)         ((rpt)/**/->temps)
#define  RPT_FLGS(rpt)          ((rpt)/**/->flgs)
#define  RPT_STNID(rpt)         ((rpt)/**/->stnid)
#define  RPT_IDTYP(rpt)         ((rpt)/**/->idtype)
#define  RPT_LATI(rpt)          ((rpt)/**/->lati)
#define  RPT_LONG(rpt)          ((rpt)/**/->longi)
#define  RPT_DX(rpt)            ((rpt)/**/->dx)
#define  RPT_DY(rpt)            ((rpt)/**/->dy)
#define  RPT_ELEV(rpt)          ((rpt)/**/->elev)
#define  RPT_DRND(rpt)          ((rpt)/**/->drnd)
#define  RPT_DATE(rpt)          ((rpt)/**/->date)
#define  RPT_OARS(rpt)          ((rpt)/**/->oars)
#define  RPT_RUNN(rpt)          ((rpt)/**/->runn)
#define  RPT_NBLK(rpt)          ((rpt)/**/->nblk)
#define  RPT_LNGR(rpt)          ((rpt)/**/->lngr)


/*
 * Macros for setting values of a Report
 * acces to field in the BURP_RPT structure should be accessed
 * through these Macros Only
 * access without these macros would be at your own risk
 * current definition do not cover all available field
 * since they are not used by anyone yet.
 */
/* for internal use only */
extern  void       brp_setstnid( BURP_RPT *rpt, const char *stnid );

#define  RPT_SetHANDLE(rpt,val)    (rpt)/**/->handle=val
#define  RPT_SetTEMPS(rpt,val)     (rpt)/**/->temps=val
#define  RPT_SetFLGS(rpt,val)      (rpt)/**/->flgs=val
#define  RPT_SetSTNID(rpt,val)     brp_setstnid(rpt,val)
#define  RPT_SetIDTYP(rpt,val)     (rpt)/**/->idtype=val
#define  RPT_SetLATI(rpt,val)      (rpt)/**/->lati=val
#define  RPT_SetLONG(rpt,val)      (rpt)/**/->longi=val
#define  RPT_SetDX(rpt,val)        (rpt)/**/->dx=val
#define  RPT_SetDY(rpt,val)        (rpt)/**/->dy=val
#define  RPT_SetELEV(rpt,val)      (rpt)/**/->elev=val
#define  RPT_SetDRND(rpt,val)      (rpt)/**/->drnd=val
#define  RPT_SetDATE(rpt,val)      (rpt)/**/->date=val
#define  RPT_SetOARS(rpt,val)      (rpt)/**/->oars=val
#define  RPT_SetRUNN(rpt,val)      (rpt)/**/->runn=val


/*
 * Macros for getting values of a Block
 * acces to field in the BURP_BLK structure should be accessed
 * through these Macros Only
 * access without these macros would be at your own risk
 */

#define  BLK_BKNO(blk)             ((blk)/**/->bkno)
#define  BLK_NELE(blk)             ((blk)/**/->nele)
#define  BLK_NVAL(blk)             ((blk)/**/->nval)
#define  BLK_NT(blk)               ((blk)/**/->nt)
#define  BLK_BFAM(blk)             ((blk)/**/->bfam)
#define  BLK_BDESC(blk)            ((blk)/**/->bdesc)
#define  BLK_BTYP(blk)             ((blk)/**/->btyp)
#define  BLK_BKNAT(blk)            ((blk)/**/->bknat)
#define  BLK_BKTYP(blk)            ((blk)/**/->bktyp)
#define  BLK_BKSTP(blk)            ((blk)/**/->bkstp)
#define  BLK_NBIT(blk)             ((blk)/**/->nbit)
#define  BLK_BIT0(blk)             ((blk)/**/->bit0)
#define  BLK_DATYP(blk)            ((blk)/**/->datyp)
#define  BLK_Data(blk)             ((blk)/**/->data)
#define  BLK_DLSTELE(blk,e)        ((blk)->dlstele[e])
#define  BLK_LSTELE(blk,e)         ((blk)->lstele[e])
#define  BLK_TBLVAL(blk,e,v,t)     (blk)->tblval[(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
#define  BLK_RVAL(blk,e,v,t)       (blk)->rval[  (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
#define  BLK_DVAL(blk,e,v,t)       (blk)->drval[ (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]
#define  BLK_CVAL(blk,l,c)         ((blk)->charval[ (l)*((blk)->nt)+(c)  ])
#define  BLK_STORE_TYPE(blk)       ((blk)->store_type)

/*
 * Macros for setting values of a Block
 * acces to field in the BURP_BLK structure should be accessed
 * through these Macros Only
 * access without these macros would be at your own risk
 */
#define  BLK_SetNELE(blk,val)           (blk)->nele=val
#define  BLK_SetNVAL(blk,val)           (blk)->nval=val
#define  BLK_SetNT(blk,val)             (blk)->nt=val

#define  BLK_SetBKNO(blk,val)           (blk)/**/->bkno=val
#define  BLK_SetBFAM(blk,val)           (blk)/**/->bfam=val
#define  BLK_SetBDESC(blk,val)          (blk)/**/->bdesc=val
#define  BLK_SetBTYP(blk,val)           (blk)/**/->btyp=val
#define  BLK_SetBKNAT(blk,val)          (blk)/**/->bknat=val
#define  BLK_SetBKTYP(blk,val)          (blk)/**/->bktyp=val
#define  BLK_SetBKSTP(blk,val)          (blk)/**/->bkstp=val
#define  BLK_SetNBIT(blk,val)           (blk)/**/->nbit=val
#define  BLK_SetDATYP(blk,val)          (blk)/**/->datyp=val
#define  BLK_SetDVAL(blk,e,v,t,val)     (blk)->drval [(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
#define  BLK_SetTBLVAL(blk,e,v,t,val)   (blk)->tblval[(e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
#define  BLK_SetRVAL(blk,e,v,t,val)     (blk)->rval[  (e) + ((blk)->nele)*((v)+((blk)->nval)*(t))]=val
#define  BLK_SetCVAL(blk,l,c,val)       (blk)->charval[ (l)*((blk)->nt)+(c)  ]=val;
#define  BLK_SetLSTELE(blk,i,val)       (blk)->lstele[i]=val
#define  BLK_SetDLSTELE(blk,i,val)      (blk)->dlstele[i]=val
#define  BLK_SetSTORE_TYPE(blk,val)     (blk)->store_type=val


/*
 * allocators and constructors
 */
extern  BURP_BLK  *brp_newblk( void );
extern  BURP_RPT  *brp_newrpt( void );
extern  void       brp_allocrpt( BURP_RPT *rpt, int  nsize );
extern  void       brp_allocblk( BURP_BLK *blk, int  nele, int nval, int nt );

/*
 *  find elements
 */
extern  int        brp_searchdlste( int  code, BURP_BLK *blk );
/*
 * destructors and deallocators
 */
extern  void       brp_freeblk( BURP_BLK *blk );
extern  void       brp_freerpt( BURP_RPT *rpt );

/* for internal use only */
extern  void       brp_freebuf(BURP_RPT *rpt);
extern  void       brp_freedata( BURP_BLK *blk );
/*
 * reinitializers
 */
extern  void       brp_clrblk( BURP_BLK  *blk );
extern  void       brp_clrblkv(BURP_BLK  *bblk, float val);
extern  void       brp_clrrpt( BURP_RPT *rpt );


/* reset blk and rpt headers to default as initialised
 * in brp_newblk and brp_newblk
 */
extern  void       brp_resetrpthdr( BURP_RPT *rpt );
extern  void       brp_resetblkhdr( BURP_BLK *blk );
/*
 * converters
 */
extern  int        brp_encodeblk( BURP_BLK  *blk );
extern  int        brp_safe_convertblk( BURP_BLK  *blk, int mode );
extern  int        brp_convertblk( BURP_BLK  *blk, int mode );
/*
 * find report and block before reading them
 */
extern  int        brp_findblk( BURP_BLK  *blk, BURP_RPT  *rpt );
extern  int        brp_findrpt( int iun, BURP_RPT *rpt );
/*
 * read in data
 */
extern  int        brp_getrpt( int iun, int handle, BURP_RPT  *rpt );
extern  int        brp_safe_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);
extern  int        brp_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);

/* brp_readblk same as brp_getblk() but the BLK_RVAL(blk,e,v,t) values
   are not available as conversion are not done. function to use when readig burp and
   there is no need to work with real values
*/
extern  int        brp_readblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt, int);

/*
 * read only header
 */
extern  int        brp_rdrpthdr(int handle, BURP_RPT *rpt);
extern  int        brp_rdblkhdr(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt);

/* prepare a report for writing */
extern  int        brp_initrpthdr( int iun, BURP_RPT *rpt );
/* prepare a report for writing alias of brp_initrpthdr */
extern  int        brp_putrpthdr( int iun, BURP_RPT *rpt );
/* add new blocks into a report */
extern  int        brp_putblk( BURP_RPT *rpt, BURP_BLK *blk );
/* write out to a file */
extern  int        brp_writerpt( int iun, BURP_RPT *rpt, int where );
/* modify only the header of a report */
extern  int        brp_updrpthdr( int iun, BURP_RPT *rpt );

/*
 * return the floating point constant used for missing values
 */
extern  float      brp_msngval(void);

/*
 * utilities
 */
/*  copy rpt header */
extern void        brp_copyrpthdr( BURP_RPT * dest, const BURP_RPT *source);
/*  copy the whole rpt  */
extern void        brp_copyrpt( BURP_RPT * dest, const BURP_RPT *source);
/*  resize the  rpt with newsize to add blocks  */
extern void        brp_resizerpt( BURP_RPT * dest, int NewSize);
/* duplicate block */
extern void        brp_copyblk( BURP_BLK *dest, const BURP_BLK *source);
/* resize  block */
extern void        brp_resizeblk( BURP_BLK *source,int NEW_ele, int NEW_nval, int NEW_nt);
extern void        brp_resizeblk_v2( BURP_BLK **source ,int nele, int nval, int nt);

/*
 * Opening files
 */

extern  int        brp_open(int  iun, const char *filename, char *op);
extern  int        brp_close(int iun);

/*
 * deleting reports and blocks
 */

extern  int        brp_delblk(BURP_RPT *rpt, const BURP_BLK * blk);
extern  int        brp_delrpt(BURP_RPT * rpt);

/*
 * burp rpn option functions
 */
extern  int        brp_SetOptFloat(char* opt, float val);
extern  int        brp_SetOptChar (char* opt, char * val);
extern  float      brp_msngval (void);
/*
 * burp rpn functions
 */
extern  int        c_mrfmxl ( int iun );
extern  int        c_mrfnbr ( int iun );
extern  int        c_mrbdel ( int *buf, int bkno);

/***************************************************************************
***************************************************************************/
 __END_DECLS
#endif /* _burp_api_h_ */


