
/*
 *
 *  file      :  burp_api.c
 *
 *  author    :  Vanh Souvanlasy
 *
 *  revision  :  Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  THIS FILE CONTAINS THE IMPLEMENTATION
 *               OF THE BURP APPLICATION PROGRAMMING INTERFACE WRAPPER
 *               THIS IS DESIGNED TO MAKE LIFE EASIER AND REMOVE THE BURDEN OF
 *               CALLING BURP SUBROUTINES DIRECTLY.
 *               NOT ALL ARE COVERED, ONLY THOSE THAT ARE MOSTLY USED ARE WRAPPED
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
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <syslog.h>
#include <errno.h>
#include <rmnlib.h>
#include "burp_api.h"


static int brp_getrpthdr( BURP_RPT *rpt );


/*
 *
 *  module    :  BRP_ALLOCRPT
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :  Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  allocate a buffer to read or write a record
 *
 */
void brp_allocrpt( BURP_RPT *rpt, int  nsize )
{
    int  *buf;

   if ( rpt == NULL )
   {
      fprintf(stderr,"rpt  pointer is NULL\n");
      return ;
   }

   if ( rpt->buffer != NULL )
   {
      if (RPT_NSIZE(rpt) >= nsize ) return;
      free( rpt->buffer );
   }
   if (nsize <= 0) nsize = 1000;
    buf = (int *) malloc ( nsize * sizeof(int) );
      buf[0] = nsize;
    rpt->buffer = buf;
    rpt->nsize = nsize;
}

/*
 *  module    :  BRP_ALLOCBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :  Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  allocate space of a data block
 */
void brp_allocblk( BURP_BLK *blk, int  nele, int nval, int nt )
{
        int  len ;
        int  i   ;
        if (blk == NULL) return;
        if (nele * nval * nt == 0 )
        {
            fprintf(stderr,"allooblk: une des dimensions du block est nulle!!\n");
            return;
        }
/*
 * reallocation dynamique de la memoire pour s'ajuster
 * a la dimension requis
 */
        BLK_SetNELE(blk, nele );
        BLK_SetNVAL(blk, nval );
        BLK_SetNT(blk, nt );

        len = nele * nval * nt;
        if (blk->max_nele < nele) {
                blk->max_nele = nele;
                if (blk->lstele != NULL) free( blk->lstele );
                blk->lstele = (int *) malloc( nele * sizeof(int) );
                for (i = 0; i != nele;++i)
                    (blk->lstele)[i] = 0;
                if (blk->dlstele != NULL) free( blk->dlstele );
                blk->dlstele = (int *) malloc( nele * sizeof(int) );
                for (i = 0; i != nele;++i)
                    (blk->dlstele)[i] = 0;
        }

        if (len <= blk->max_len) {
                nval = blk->max_len / (nele * nt);
                blk->max_nval = nval;
                blk->max_nt = nt;
                return;
        }
        blk->max_nval = nval;
        blk->max_nt = nt;

/*
** tous les tableaux sont lineaires, meme les tableau a plusieurs dimensions
** l'acces aux element de ce tableau doit etre fait de telle sorte a referer
** aux bons elements selon l'ordre fortran
** l'allocation est fait pour tous les types (entier, reel, double)
*/
        if (blk->tblval != NULL) free( blk->tblval );
        blk->tblval = (int *) malloc ( len * sizeof(int) );
        if (blk->rval != NULL) free( blk->rval );
        blk->rval = (float *) malloc ( len * sizeof(float) );
        blk->max_len = len;

    if (BLK_STORE_TYPE(blk)==STORE_DOUBLE) {
       len = nele*nval*nt;
       if (blk->drval == NULL)
         blk->drval = (double *) malloc ( len * sizeof(double) );
       else
         blk->drval = (double *) realloc ( blk->drval, len * sizeof(double) );
    }
        
    if (blk->charval == NULL)
      blk->charval = (char *) malloc ( len * sizeof(char) );
    else
      blk->charval = (char *) realloc ( blk->charval, len * sizeof(char) );        
        
}

/*
 *  module    :  BRP_CLRBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :  Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  clear the content of a block
 */
void brp_clrblk( BURP_BLK  *bblk )
{
  int i, cnt;
  float  msng_val;

  if (bblk == NULL) return;
  msng_val = brp_msngval();
  cnt = BLK_NELE(bblk) * BLK_NVAL(bblk) * BLK_NT(bblk);
  for ( i = 0 ; i < cnt ; i++ ) {
    (bblk)->rval[i] = msng_val;
    (bblk)->tblval[i] = -1;
  }
}

/*
 *  module    :  BRP_CLRBLKV
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  clear the content of a block with a value
 */
void brp_clrblkv( BURP_BLK  *bblk, float rval )
{
  int i, cnt;

  if (bblk == NULL) return;
  cnt = BLK_NELE(bblk) * BLK_NVAL(bblk) * BLK_NT(bblk);
  if (BLK_STORE_TYPE(bblk)==STORE_FLOAT) {
        for ( i = 0 ; i < cnt ; i++ )
        {
            (bblk)->rval[i] = rval;
            (bblk)->tblval[i] = -1;
        }

  } else if (BLK_STORE_TYPE(bblk)==STORE_INTEGER) {
        int ival=(int)rval;
        for ( i = 0 ; i < cnt ; i++ )
            (bblk)->tblval[i] = ival;
  }
}

/*
 *  module    :  BRP_clrrpt
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  clear the content of the record buffer
 */
void brp_clrrpt( BURP_RPT *rpt )
{
    int i;

    if ( rpt == NULL ) return;
    if ( rpt->buffer == NULL ) return;
/*
** la taille du tableau buffer
*/
    rpt->buffer[0] = RPT_NSIZE(rpt);

/*
** mettre a blanc le contenue du buffer
*/
    for ( i = 1 ; i < RPT_NSIZE(rpt) ; i++ )
        rpt->buffer[i] = 0;

    rpt->init_hdr = 1;
}

/*
 *  module    :  BRP_SAFE_CONVERTBLK
 *
 *  author    :  Chris Malek
 *
 *  revision  : 
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  convert between floating point and integer value
 *               of a block (and make sure output memory is initialized
 *               before conversion to avoid junk from memory creeping in)
 *
 */
int brp_safe_convertblk( BURP_BLK  *bb, int mode )
{
   if (bb == NULL) return(-1);

   if (mode == BUFR_to_MKSA)
   {
       memset(bb->rval, 0, BLK_NELE(bb)*BLK_NVAL(bb)*BLK_NT(bb)*sizeof(float));
   }
   else if (mode == MKSA_to_BUFR)
   {
       memset(bb->tblval, 0, BLK_NELE(bb)*BLK_NVAL(bb)*BLK_NT(bb)*sizeof(int));
   }
   else
   {
       return(-1);
   }

   brp_convertblk(bb, mode);
}

/*
 *  module    :  BRP_CONVERTBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  convert between floating point and integer value
 *               of a block
 *
 */
int brp_convertblk( BURP_BLK  *bb,int mode )
{
   int istat ;

   if (bb == NULL) return(-1);
   // only do conversion if this isn't a marker block
   // AND if it isn't a real or character block
   // (see brp_readblk for their conversion conditions as well)
   if ( (BLK_BKNAT(bb)%4) != 3 &&
            BLK_DATYP(bb) != 6 && 
            BLK_DATYP(bb) != 7 && 
            BLK_DATYP(bb) != 3 ) {
     istat = c_mrbcvt( bb->lstele, bb->tblval,
                bb->rval, BLK_NELE(bb),
                BLK_NVAL(bb), BLK_NT(bb), mode );
   }
   return ( istat );
}

/*
 *  module    :  BRP_ENCODEBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  encode the bufr code of a data block
 */
int brp_encodeblk( BURP_BLK  *bb )
{
   int istat;

   if (bb == NULL) return(-1);

   istat = c_mrbcol( bb->dlstele, bb->lstele, BLK_NELE(bb) );

   return( istat );
}

/*
 *  module    :  BRP_FINDBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  locate a data block within a report
 */
int  brp_findblk( BURP_BLK  *blk, BURP_RPT  *rpt )
{
    int    bkno;

    if ((blk == NULL)||(rpt == NULL)) return -1;
/*
** le BLK0 est pris a partir du BKNO contenu dans blk comme informations
** de recherche
*/
    if (BLK_BTYP(blk) != -1)
      bkno = c_mrbloc( rpt->buffer,
              BLK_BFAM(blk), BLK_BDESC(blk), BLK_BTYP(blk),
              BLK_BKNO(blk) );
    else
      bkno = c_mrblocx( rpt->buffer,
              BLK_BFAM(blk), BLK_BDESC(blk),
              BLK_BKNAT(blk), BLK_BKTYP(blk), BLK_BKSTP(blk),
              BLK_BKNO(blk) );

/*
** assigne la valeur de bkno trouve
*/
    BLK_SetBKNO( blk, bkno );

    return( bkno );
}

/*
 *  module    :  brp_findrpt
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  locate a report within a file
 *
 */
int  brp_findrpt( int iun, BURP_RPT *rpt )
{
    int  handle;

/*
** localise le premier enregistrement correspondant aux criteres de recherche
*/
    handle = c_mrfloc( iun, RPT_HANDLE(rpt),RPT_STNID(rpt),RPT_IDTYP(rpt),
                    RPT_LATI(rpt), RPT_LONG(rpt),
                    RPT_DATE(rpt), RPT_TEMPS(rpt),
                    rpt->sup, rpt->nsup );
/*
** affecte le pointeur de l'enregistrement pour retourner la valeur de ce
** pointeur
*/
    RPT_SetHANDLE(rpt, handle);

    return( handle );
}


/*
 *  module    :  BRP_FREEBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  free the memory used by the data of a block
 */
void  brp_freeblk( BURP_BLK *blk )
{

   if ( blk == NULL ) return;

  if ( blk->lstele ) {
    free( blk->lstele );
    blk->lstele = NULL;
  }
  if ( blk->dlstele ) {
    free( blk->dlstele );
    blk->dlstele = NULL;
  }
  if ( blk->tblval ) {
    free( blk->tblval );
    blk->tblval = NULL;
  }
  if ( blk->rval ) {
    free( blk->rval );
    blk->rval = NULL;
  }
   if ( blk->charval ) {
       free( blk->charval);
       blk->charval = NULL;
   }

  blk->max_nele = 0;
  blk->max_nval = 0;
  blk->max_nt = 0;
  blk->max_len = 0;
  BLK_SetNELE(blk, 0);
  BLK_SetNVAL(blk, 0);
  BLK_SetNT(blk, 0);
  if (blk->drval != NULL) {
    free(blk->drval);
    blk->drval = NULL;
  }
  free( blk );
}

/*
 *  module    :  BRP_FREERPT
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  destroy a report's memory allocation
 */
void brp_freerpt( BURP_RPT *rpt )
{

   if ( rpt == NULL ) return;

   if ( rpt->buffer != NULL )
   {
      free( rpt->buffer );
      rpt->buffer = NULL;
   }
   rpt->nsize = 0;
   free( rpt );
}

/*
 *  module    :  BRP_SAFE_GETBLK
 *
 *  author    :  Chris Malek
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read a block from a report
 *               and initialize memory of output block prior to conversion
 */
int  brp_safe_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt)
{
        int istat;
        istat = (brp_readblk( bkno, blk, rpt, 0 ));
        brp_safe_convertblk( blk, BUFR_to_MKSA );
        BLK_SetSTORE_TYPE(blk, STORE_FLOAT );
        return istat;
}


/*
 *  module    :  BRP_GETBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read a block from a report
 */
int  brp_getblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt)
{
        return (brp_readblk( bkno, blk, rpt, 1 ));
}

/*
 *  module    :  brp_rdblkhdr
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read header information of a block
 */
int  brp_rdblkhdr(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt)
{
        int istat;

        if ((rpt == NULL)||(blk == NULL)) return -1;
/*
** lire les informations du bloc
*/
        istat = c_mrbprm( rpt->buffer, bkno,
                 &(blk->nele), &(blk->nval), &(blk->nt),
                 &(blk->bfam), &(blk->bdesc),
                 &(blk->btyp), &(blk->nbit),
                 &(blk->bit0), &(blk->datyp) );
        c_mrbtyp( &(blk->bknat), &(blk->bktyp), &(blk->bkstp),
              BLK_BTYP(blk) );

        BLK_SetBKNO(blk, bkno);

        return istat;
}
/*
 *  module    :  BRP_READBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read a block from a report
 */
int  brp_readblk(int bkno, BURP_BLK  *blk, BURP_RPT  *rpt, int docvt)
{
        int istat;

        if ((rpt == NULL)) {
            fprintf(stderr,"attention rpt nul \n");
           return -1;
        }
        if ((blk == NULL)) {
            fprintf(stderr,"attention blk nul \n");
           return -1;
        }
/*
** lire les informations du bloc
*/
        istat = c_mrbprm( rpt->buffer, bkno,
                 &(blk->nele), &(blk->nval), &(blk->nt),
                 &(blk->bfam), &(blk->bdesc),
                 &(blk->btyp), &(blk->nbit),
                 &(blk->bit0), &(blk->datyp) );
        c_mrbtyp( &(blk->bknat), &(blk->bktyp), &(blk->bkstp),
              BLK_BTYP(blk) );

        if (istat < 0 ) return( istat );
/*
** allouer l'espace suffisante pour lire le bloc
*/
        if ( BLK_DATYP(blk) == DATYP_CHAR ) BLK_SetSTORE_TYPE(blk, STORE_CHAR );
        if ( BLK_DATYP(blk) == 7 )          BLK_SetSTORE_TYPE(blk, STORE_DOUBLE );
        
        if  (((BLK_NT(blk)) * (BLK_NVAL(blk)) *(BLK_NELE(blk))) != 0) {
            brp_allocblk( blk, BLK_NELE(blk), BLK_NVAL(blk), BLK_NT(blk));
        } else {
/*            fprintf(stderr,"une des dimensions du block est nulle!!\n");*/
            return(-1);
        }

/*        bb = BLK_Data(blk);*/

/*
** extraire le bloc du rapport
*/
        if ( BLK_DATYP(blk) == 7 ) {  /* REEL 8 */
                istat = c_mrbxtr( rpt->buffer, bkno,
                 blk->lstele, blk->drval );
                if (istat < 0 ) return( istat );
                BLK_SetBKNO(blk, bkno);
        } else if ( BLK_DATYP(blk) == 6 ) {  /* REEL 4 */
                istat = c_mrbxtr( rpt->buffer, bkno,
                 blk->lstele, blk->rval );
                if (istat < 0 ) return( istat );
                BLK_SetBKNO(blk, bkno);
        } else if ( BLK_DATYP(blk) == 3 ) {
                istat = c_mrbxtr( rpt->buffer, bkno,
                 blk->lstele, blk->charval );
                if (istat < 0 ) return( istat );
                BLK_SetBKNO(blk, bkno);
            
        } else {
                istat = c_mrbxtr( rpt->buffer, bkno,
                 blk->lstele, blk->tblval );
                if (istat < 0 ) return( istat );
                BLK_SetBKNO(blk, bkno);
/*
** convertir les donnees des valeurs entiere de BURP vers des valeurs reelles
** TBLVAL a RVAL si docvt et le blk n'est pas un marqueur
*/
                if (docvt && (BLK_BKNAT(blk)%4) != 3) {
                        c_mrbcvt( blk->lstele, blk->tblval,
                blk->rval, BLK_NELE(blk),
                BLK_NVAL(blk), BLK_NT(blk),
                BUFR_to_MKSA );
                        BLK_SetSTORE_TYPE(blk, STORE_FLOAT );
                }
        }

/*
** decode les noms des elements lues
*/
    istat = c_mrbdcl( blk->lstele, blk->dlstele, BLK_NELE(blk) );

        return( istat );
}

/*
 *  module    :  BRP_GETRPT
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read a report from a file
 */
int  brp_getrpt( int iun, int handle, BURP_RPT  *rpt )
{
        int istat;

        if (rpt == NULL) return -1;
        istat = brp_rdrpthdr( handle, rpt );

        if ( istat >= 0 ) {
                if ( (RPT_LNGR(rpt)+20) > RPT_NSIZE(rpt) ) {
                        brp_allocrpt( rpt, (c_mrfmxl(iun)+2000) );
                }
                istat = c_mrfget( handle,
                      rpt->buffer );
        }
        if ( istat >= 0 ) {
                istat = brp_getrpthdr( rpt );
                RPT_SetHANDLE(rpt,handle);
        }
        return( istat );
}

/*
 *  module    :  BRP_GETRPTHDR
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Handle Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read the header of a report (already read)
 */
static int brp_getrpthdr( BURP_RPT *rpt )
{
        int istat;

        if (rpt == NULL) return -1;

        if ( rpt->nsup > 0 ) {
                if (rpt->sup!=NULL) free( rpt->sup );
                rpt->sup = (int *)malloc(sizeof(int)*rpt->nsup );
        }
        if ( rpt->nxaux > 0 ) {
                if (rpt->xaux!=NULL) free( rpt->xaux );
                rpt->xaux = (int *)malloc(sizeof(int)*rpt->nxaux );
        }
/*
** ramasser l'entete du rapport lu
*/
        istat = c_mrbhdr( rpt->buffer, &(rpt->temps), &(rpt->flgs),
              rpt->stnid, &(rpt->idtype), &(rpt->lati),
              &(rpt->longi), &(rpt->dx), &(rpt->dy),
                          &(rpt->elev), &(rpt->drnd), &(rpt->date),
              &(rpt->oars), &(rpt->runn), &(rpt->nblk), rpt->sup,
              rpt->nsup, rpt->xaux, rpt->nxaux );

        return( istat );
}

/*
 *  module    :  BRP_MSNGVAL
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  return the value used by burp for marquing missing values
 */
float brp_msngval(void)
{
/*
** constante pour les valeurs manquantes
*/
   static float missing_value=0.0;

   if ( missing_value == 0.0 )
   {
      c_mrfgor( "MISSING", &missing_value );
   }
   return missing_value;
}
/*
 *  module    :  brp_msngval
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  return the value used by burp for marquing missing values
 */
int brp_SetMissing(char* opt, float val)
{
   int istat;

   istat = c_mrfopr( opt, val );
   return istat;
}

/*
 *  module    :  BRP_SetOptFloat
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  Set burp Option float
 */
int brp_SetOptFloat(char* opt, float val)
{
   int istat;

   istat = c_mrfopr( opt, val );
   return istat;
}
/*
 *  module    :  BRP_SetOptChar
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  Set burp Option float
 */
int brp_SetOptChar(char* opt, char * val)
{
   int istat;

   istat = c_mrfopc( opt, val );
   return istat;
}
/*
 *  module    :  brp_newblk
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  instantiate a new block entity
 *
 */
BURP_BLK *brp_newblk (void)
{
     BURP_BLK *blk;

     blk = (BURP_BLK *) malloc ( sizeof(BURP_BLK) );
     blk->max_nele = 0;
     blk->max_nval = 0;
     blk->max_nt = 0;
     blk->dlstele = NULL;
     blk->lstele = NULL;
     blk->tblval = NULL;
     blk->rval = NULL;
     blk->max_len = 0;
     blk->store_type = STORE_FLOAT;

     blk->drval = NULL;
     blk->charval = NULL;
     brp_resetblkhdr( blk );
     return( blk );
}

/*
 *  module    :  BRP_NEWRPT
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  instantiate a new report entity
 *
 */
BURP_RPT *brp_newrpt( void )
{
     BURP_RPT  *rpt;

     rpt = (BURP_RPT *) malloc( sizeof(BURP_RPT) );
     if (rpt == NULL) fprintf(stderr, "ERROR: Out of memory!\n");

     rpt->buffer = NULL;
     rpt->nsize = 0;
     rpt->nsup = 0;
     rpt->sup = NULL;
     rpt->nxaux = 0;
     rpt->xaux = NULL;
     brp_resetrpthdr( rpt );
     rpt->init_hdr = 1;
     return( rpt );
}

/*
 *  module    :  BRP_PUTRPTHDR
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  initialize the content of a report buffer with its header
 *
 */
int brp_putrpthdr( int   iun, BURP_RPT *rpt )
{
    int  istat;

        if (rpt == NULL) return -1;
/*
** initialiser l'entete du rapport
*/
    istat = c_mrbini( iun, rpt->buffer,
                   RPT_TEMPS(rpt), RPT_FLGS(rpt),
                   RPT_STNID(rpt), RPT_IDTYP(rpt), RPT_LATI(rpt),
                   RPT_LONG(rpt), RPT_DX(rpt), RPT_DY(rpt),
                               RPT_ELEV(rpt), RPT_DRND(rpt), RPT_DATE(rpt),
                   RPT_OARS(rpt), RPT_RUNN(rpt), rpt->sup,
                   rpt->nsup, rpt->xaux, rpt->nxaux );

    rpt->init_hdr = 0;

    return( istat );
}
/*
 *  module    :  BRP_INITRPTHDR
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  initialize the content of a report buffer with its header
 *
 */
int brp_initrpthdr( int   iun, BURP_RPT *rpt )
{
    int  istat;

        if (rpt == NULL) return -1;
/*
** initialiser l'entete du rapport
*/
    istat = c_mrbini( iun, rpt->buffer,
                   RPT_TEMPS(rpt), RPT_FLGS(rpt),
                   RPT_STNID(rpt), RPT_IDTYP(rpt), RPT_LATI(rpt),
                   RPT_LONG(rpt), RPT_DX(rpt), RPT_DY(rpt),
                               RPT_ELEV(rpt), RPT_DRND(rpt), RPT_DATE(rpt),
                   RPT_OARS(rpt), RPT_RUNN(rpt), rpt->sup,
                   rpt->nsup, rpt->xaux, rpt->nxaux );

    rpt->init_hdr = 0;

    return( istat );
}

/*
 *  module    :  BRP_PUTBLK
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  add a new block to a report for writing
 */
int brp_putblk( BURP_RPT *rpt, BURP_BLK *blk )
{
   int   istat;
   int bfam, bdesc, btyp, nbit, datyp;

   if (blk == NULL) return -1;
   if (rpt == NULL) return -1;


   bfam = BLK_BFAM(blk);
   bdesc = BLK_BDESC(blk);
   btyp = BLK_BTYP(blk);
   nbit = BLK_NBIT(blk);
   datyp = BLK_DATYP(blk);
   if (bfam < 0) bfam = 0;
   if (bdesc < 0) bdesc = 0;
   if (btyp < 0) btyp = 0;
   if (nbit < 0) nbit = 17;
   if (datyp < 0) datyp = DATYP_UINTEGER;

   if ( BLK_DATYP(blk) == 6 ) {
       istat = c_mrbadd( rpt->buffer, &(blk->bkno), BLK_NELE(blk),
                     BLK_NVAL(blk), BLK_NT(blk), bfam,
                     bdesc, btyp, nbit,
                     &(blk->bit0), datyp, blk->lstele,
                     blk->rval );
   } else if ( BLK_DATYP(blk) == 7 ) {
       istat = c_mrbadd( rpt->buffer, &(blk->bkno), BLK_NELE(blk),
                     BLK_NVAL(blk), BLK_NT(blk), bfam,
                     bdesc, btyp, nbit,
                     &(blk->bit0), datyp, blk->lstele,
                     blk->drval );
   } else if ( BLK_DATYP(blk) == 3 ) {
       istat = c_mrbadd( rpt->buffer, &(blk->bkno), BLK_NELE(blk),
                     BLK_NVAL(blk), BLK_NT(blk), bfam,
                     bdesc, btyp, nbit,
                     &(blk->bit0), datyp, blk->lstele,
                     blk->charval );
   } else {
       istat = c_mrbadd( rpt->buffer, &(blk->bkno), BLK_NELE(blk),
                     BLK_NVAL(blk), BLK_NT(blk), bfam,
                     bdesc, btyp, nbit,
                     &(blk->bit0), datyp, blk->lstele,
                     blk->tblval );
   }
   return( istat );
}


/*
 *  module    :  brp_rdrpthdr
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  read the header of a report without reading
 *               the report entirely
 */
int brp_rdrpthdr( int handle, BURP_RPT *rpt )
{
    int istat;

        if (rpt == NULL) return -1;

    if ( rpt->nsup > 0 ) {
      if (rpt->sup!=NULL) free( rpt->sup );
      rpt->sup = (int *)malloc(sizeof(int)*rpt->nsup);
    }
/*
** ramasser l'entete du rapport
*/
    istat = c_mrfprm( handle,
              rpt->stnid, &(rpt->idtype), &(rpt->lati),
              &(rpt->longi), &(rpt->dx), &(rpt->dy),
                          &(rpt->date), &(rpt->temps), &(rpt->flgs),
              rpt->sup, rpt->nsup, &(rpt->lngr) );

    return( istat );
}

/*
 *  module    :  brp_resetblkhdr
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  reinitialize a block's header for reading
 */
void brp_resetblkhdr( BURP_BLK *blk )
{
    if (blk == NULL) return;
    blk->nele = -1;
    blk->nval = -1;
    blk->nt = -1;
    BLK_SetBKNO(blk,0);
    BLK_SetBFAM(blk, -1);
    BLK_SetBDESC(blk, -1);
    BLK_SetBTYP(blk, -1);
    BLK_SetBKNAT(blk, -1);
    BLK_SetBKTYP(blk, -1);
    BLK_SetBKSTP(blk, -1);
    BLK_SetNBIT(blk, -1);
    blk->bit0 = -1;
    BLK_SetDATYP(blk, DATYP_INTEGER);
}
/*
 *  module    :  brp_copytblk
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a block's  from source to destination
 */
void brp_copyblk( BURP_BLK *dest, const BURP_BLK *source)
{
    int e,v,t;
    if (source == NULL)
    {
      fprintf(stderr," blk pointer is NULL, brp_copyblk not done!\n");
      return;
    }
    brp_allocblk( dest,BLK_NELE(source),BLK_NVAL(source), BLK_NT(source))  ;
    BLK_SetBKNO(dest, BLK_BKNO(source)) ;
    BLK_SetBFAM(dest, BLK_BFAM(source)) ;
    BLK_SetBDESC(dest,BLK_BDESC(source));
    BLK_SetBTYP(dest, BLK_BTYP(source)) ;
    BLK_SetBKNAT(dest,BLK_BKNAT(source));
    BLK_SetBKTYP(dest,BLK_BKTYP(source));
    BLK_SetBKSTP(dest,BLK_BKSTP(source));
    BLK_SetNBIT(dest, BLK_NBIT(source)) ;
    dest->bit0 = source->bit0;
    BLK_SetDATYP(dest, BLK_DATYP(source));
    BLK_SetSTORE_TYPE( dest, BLK_STORE_TYPE(source) );

    for (e = 0; e < BLK_NELE(source); e++) {
        if (source->dlstele != NULL)
            BLK_SetDLSTELE(dest,e,BLK_DLSTELE(source,e));
        if (source->lstele != NULL)
            BLK_SetLSTELE(dest,e,BLK_LSTELE(source,e));
    }
    for (e = 0; e < BLK_NELE(source); e++) {
        for (v = 0; v < BLK_NVAL(source); v++) {
            for (t = 0; t < BLK_NT(source); t++) {
                if (source->tblval != NULL)
                    BLK_SetTBLVAL(dest,e,v,t,BLK_TBLVAL(source,e,v,t));
                if (source->rval != NULL)
                    BLK_SetRVAL(dest,e,v,t,BLK_RVAL(source,e,v,t)) ;
                if (source->drval != NULL)
                    BLK_SetDVAL(dest,e,v,t,BLK_DVAL(source,e,v,t));
                if ( source->charval != NULL)
                    BLK_SetCVAL(dest,v,t, BLK_CVAL(source,v,t));                
            }
        }
    }
    

}
/*
 *  module    :  brp_resizeblk
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a block's header from source to destination
 */
void brp_resizeblk( BURP_BLK *dest,int nele, int nval, int nt)
{
    int e,v,t;
    int mele, mval, mnt;
    BURP_BLK * source;
    if (dest == NULL)
    {
      fprintf(stderr," blk pointer is NULL, resizeblk not done!\n");
        return;
    }
    source = brp_newblk();
    /* faire une copie dans et la mettre dans source */
    brp_copyblk(source,dest);
    /* reallouer des avec les nouvelles dimensions */
    brp_allocblk( dest, nele, nval, nt)  ;
    /* copier ce qui dans source et le mettre dans dest.
     * pour eviter ces manipulations utilser resize_v2
     */
    BLK_SetBKNO(dest, BLK_BKNO(source)) ;
    BLK_SetBFAM(dest, BLK_BFAM(source)) ;
    BLK_SetBDESC(dest,BLK_BDESC(source));
    BLK_SetBTYP(dest, BLK_BTYP(source)) ;
    BLK_SetBKNAT(dest,BLK_BKNAT(source));
    BLK_SetBKTYP(dest,BLK_BKTYP(source));
    BLK_SetBKSTP(dest,BLK_BKSTP(source));
    BLK_SetNBIT(dest, BLK_NBIT(source)) ;
    dest->bit0 = source->bit0;
    BLK_SetDATYP(dest, BLK_DATYP(source));
    BLK_SetSTORE_TYPE( dest, BLK_STORE_TYPE(source) );
    brp_clrblk( dest );

    mele = BLK_NELE(source) > BLK_NELE(dest) ? BLK_NELE(dest):BLK_NELE(source);
    mval = BLK_NVAL(source) > BLK_NVAL(dest) ? BLK_NVAL(dest):BLK_NVAL(source);
    mnt  = BLK_NT(source)   > BLK_NT(dest)   ? BLK_NT(dest):BLK_NT(source);
    for (e = 0; e != mele; e++) {
        if (source->dlstele != NULL)
            BLK_SetDLSTELE(dest,e,BLK_DLSTELE(source,e));
        if (source->lstele != NULL)
            BLK_SetLSTELE(dest,e,BLK_LSTELE(source,e));
    }
    for (e = 0; e != mele ; e++) {
        for (v = 0; v != mval; v++) {
            for (t = 0; t != mnt; t++) {
                if (source->tblval != NULL)
                    BLK_SetTBLVAL(dest,e,v,t,BLK_TBLVAL(source,e,v,t));
                if (source->rval != NULL)
                    BLK_SetRVAL(dest,e,v,t,BLK_RVAL(source,e,v,t)) ;
                if ((BLK_STORE_TYPE(source) == STORE_DOUBLE ) && source->drval != NULL )
                    BLK_SetDVAL(dest,e,v,t,BLK_DVAL(source,e,v,t));
                if ((BLK_STORE_TYPE(source) == STORE_CHAR ) && source->charval != NULL )
                    BLK_SetCVAL(dest,v,t,BLK_CVAL(source,v,t));
            }
        }
    }
    brp_freeblk(source);
}
/*
 *  module    :  brp_resizeblk_v2
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  resize block to new diemsions
 */
void brp_resizeblk_v2( BURP_BLK **inblk,int nele, int nval, int nt)
{
    int e,v,t;
    int mele, mval, mnt;
    BURP_BLK * dest;
    BURP_BLK * source = *inblk;
    if (source == NULL)
    {
      fprintf(stderr," blk pointer is NULL, resizeblk not done!\n");
        return;
    }
    dest = brp_newblk();
    brp_allocblk( dest, nele, nval, nt)  ;
    BLK_SetBKNO(dest, BLK_BKNO(source)) ;
    BLK_SetBFAM(dest, BLK_BFAM(source)) ;
    BLK_SetBDESC(dest,BLK_BDESC(source));
    BLK_SetBTYP(dest, BLK_BTYP(source)) ;
    BLK_SetBKNAT(dest,BLK_BKNAT(source));
    BLK_SetBKTYP(dest,BLK_BKTYP(source));
    BLK_SetBKSTP(dest,BLK_BKSTP(source));
    BLK_SetNBIT(dest, BLK_NBIT(source)) ;
    dest->bit0 = source->bit0;
    BLK_SetDATYP(dest, BLK_DATYP(source));
    BLK_SetSTORE_TYPE( dest, BLK_STORE_TYPE(source) );
    brp_clrblk( dest );

    mele = BLK_NELE(source) > BLK_NELE(dest) ? BLK_NELE(dest):BLK_NELE(source);
    mval = BLK_NVAL(source) > BLK_NVAL(dest) ? BLK_NVAL(dest):BLK_NVAL(source);
    mnt  = BLK_NT(source)   > BLK_NT(dest)   ? BLK_NT(dest):BLK_NT(source);
    for (e = 0; e != mele; e++) {
        if (source->dlstele != NULL)
            BLK_SetDLSTELE(dest,e,BLK_DLSTELE(source,e));
        if (source->lstele != NULL)
            BLK_SetLSTELE(dest,e,BLK_LSTELE(source,e));
    }
    for (e = 0; e != mele ; e++) {
        for (v = 0; v != mval; v++) {
            for (t = 0; t != mnt; t++) {
                if (source->tblval != NULL)
                    BLK_SetTBLVAL(dest,e,v,t,BLK_TBLVAL(source,e,v,t));
                if (source->rval != NULL)
                    BLK_SetRVAL(dest,e,v,t,BLK_RVAL(source,e,v,t)) ;
                if ((BLK_STORE_TYPE(source) == STORE_DOUBLE ) && source->drval != NULL ) {
                    BLK_SetDVAL(dest,e,v,t,BLK_DVAL(source,e,v,t));
                    }
            }
        }
    }
    *inblk = dest;
    brp_freeblk(source);
}
/*
 *  module    :  brp_delblk
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a block's header from source to destination
 */
int brp_delblk(BURP_RPT *rpt, const BURP_BLK *blk)
{
   int istat = -1;
   if ((rpt == NULL) && (blk == NULL))
   {
      fprintf(stderr,"rpt adn blk pointers are NULL\n");
      return istat;
   }
   if (rpt->buffer != NULL)
      istat = c_mrbdel(rpt->buffer,BLK_BKNO(blk));
   return istat;
}
/*
 *  module    :  brp_delrpt
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a block's header from source to destination
 */
int brp_delrpt(BURP_RPT *rpt)
{
    if (rpt == NULL)
    {
       fprintf(stderr,"rpt  pointer is NULL\n");
        return -1;
    }
     return(c_mrfdel(RPT_HANDLE(rpt))) ;
}

/*
 *  module    :  brp_resetrpthdr
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  : Hamid Benhocine
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  initialize a report's header for reading
 */
void brp_resetrpthdr( BURP_RPT *rpt )
{
   if (rpt == NULL) return;
    RPT_SetHANDLE(rpt,0);
    RPT_SetTEMPS(rpt, -1) ;
    RPT_SetFLGS(rpt, -1) ;
    RPT_SetSTNID(rpt,"*********");
    RPT_SetIDTYP(rpt, -1) ;
    RPT_SetLATI(rpt, -1) ;
    RPT_SetLONG(rpt, -1) ;
    RPT_SetDX(rpt, -1) ;
    RPT_SetDY(rpt, -1) ;
    RPT_SetELEV(rpt, -1) ;
    RPT_SetDRND(rpt, -1) ;
    RPT_SetDATE(rpt, -1) ;
    RPT_SetOARS(rpt, -1) ;
    RPT_SetRUNN(rpt, -1) ;
    rpt->nblk = 0;
    rpt->nsup = 0;
    rpt->nxaux = 0;
        if (rpt->xaux!=NULL) free( rpt->xaux );
        rpt->xaux = NULL;
        if (rpt->sup!=NULL) free( rpt->sup );
        rpt->sup = NULL;
}
/*
 *  module    :  brp_copytrpthdr
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a rpt's header from source to destination
 */
void brp_copyrpthdr( BURP_RPT * dest, const BURP_RPT *source )
{
   if (source == NULL)
   {
      fprintf(stderr,"rpt source  pointer is NULL\n");
      return ;
   }
   RPT_SetHANDLE(dest,RPT_HANDLE(source));
   RPT_SetTEMPS(dest, RPT_TEMPS(source)) ;
   RPT_SetFLGS(dest,  RPT_FLGS(source)) ;
   RPT_SetSTNID(dest, RPT_STNID(source));
   RPT_SetIDTYP(dest, RPT_IDTYP(source)) ;
   RPT_SetLATI(dest,  RPT_LATI(source)) ;
   RPT_SetLONG(dest,  RPT_LONG(source)) ;
   RPT_SetDX(dest,    RPT_DX(source)) ;
   RPT_SetDY(dest,    RPT_DY(source)) ;
   RPT_SetELEV(dest,  RPT_ELEV(source)) ;
   RPT_SetDRND(dest,  RPT_DRND(source)) ;
   RPT_SetDATE(dest,  RPT_DATE(source)) ;
   RPT_SetOARS(dest,  RPT_OARS(source)) ;
   RPT_SetRUNN(dest,  RPT_RUNN(source)) ;
   dest->nblk = source->nblk;
}
/*
 *  module    :  brp_copytrpt
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a rpt's header from source to destination
 */
void brp_copyrpt( BURP_RPT * dest, const BURP_RPT *source )
{
   int i;
   if (source == NULL)
   {
      fprintf(stderr,"rpt source  pointer is NULL\n");
      return ;
   }
   brp_resetrpthdr( dest );
   RPT_SetHANDLE(dest,RPT_HANDLE(source));
   RPT_SetTEMPS(dest, RPT_TEMPS(source)) ;
   RPT_SetFLGS(dest,  RPT_FLGS(source)) ;
   RPT_SetSTNID(dest, RPT_STNID(source));
   RPT_SetIDTYP(dest, RPT_IDTYP(source)) ;
   RPT_SetLATI(dest,  RPT_LATI(source)) ;
   RPT_SetLONG(dest,  RPT_LONG(source)) ;
   RPT_SetDX(dest,    RPT_DX(source)) ;
   RPT_SetDY(dest,    RPT_DY(source)) ;
   RPT_SetELEV(dest,  RPT_ELEV(source)) ;
   RPT_SetDRND(dest,  RPT_DRND(source)) ;
   RPT_SetDATE(dest,  RPT_DATE(source)) ;
   RPT_SetOARS(dest,  RPT_OARS(source)) ;
   RPT_SetRUNN(dest,  RPT_RUNN(source)) ;
   dest->nblk = source->nblk;
   brp_clrrpt(dest);
   brp_allocrpt( dest, RPT_NSIZE(source));

   for ( i = 1 ; i < RPT_NSIZE(source) ; i++ )
      dest->buffer[i] = source->buffer[i] ;

   dest->init_hdr = 1;
}
/*
 *  module    :  brp_resizetrpt
 *
 *  author    :  Hamid Benhocine
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  copy a rpt's header from source to destination
 */
void brp_resizerpt( BURP_RPT * dest, int NewSize )
{
    if (dest == NULL) return;
    brp_allocrpt( dest, NewSize );
    return;
}

/*
 *name:  brp_searchdlste()
 *  author    :  Souvanlasy Viengsvanh
 *  revsion   :  Hamid Benhocine
 *purpose:  retourner l'indice d'un element dans un vecteur d'elements non codes
 *
 *parameters:
 *
 *   code    : le code de BUFR decode recherche dans  DLSTELE
 *
 *   blk     : pointeur vers une structure contenant les informations sur
 *             un bloc de donnees,
 *             dont DLSTELE qui contient la liste des elements decodes
 *             et NELE le nombre d'element dans DLSTELE
 *
 *valeur retournee:
 *
 *    > 0 et < NELE   :  le code est trouve dans DLSTELE
 *
 *    -1   :   ce code n'existe pas dans DLSTELE
 */
int  brp_searchdlste( int  code, BURP_BLK *blk )
{
     int  i; /** compteur de boucle **/
      if (blk == NULL) return -1;

     for ( i = 0 ; i < BLK_NELE(blk) ; i++ )
     {
        if ( code == BLK_DLSTELE(blk,i) )
            return( i );
     }
     return( -1 );
}

/*
 *  module    :  BRP_SETSTNID
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  assign a value to the station id of a report
 */
void brp_setstnid( BURP_RPT *rpt, const char *stnid )
{
   int  len, i;

   if (rpt == NULL) return;
   len = strlen(stnid);

   if ( len > 9 ) len = 9;

   strncpy( rpt->stnid, stnid, len );
   if ( len < 9 )
   {
     for ( i = len ; i < 9 ; i++ )
       rpt->stnid[i] = ' ' ;
   }
   rpt->stnid[9] = '\0';
}

/*
 *  module    :  BRP_WRITERPT
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  write a report to a file
 */
int brp_writerpt( int iun, BURP_RPT *rpt, int handle )
{
    int istat;

        if (rpt == NULL) return -1;
/*
** le nouveau enregistrement sera ecrit a la fin du fichier
*/
   if (handle < 0) handle = 0;
   istat = c_mrfput( iun, handle, rpt->buffer );

   return( istat );
}

/*
 *  module    :  brp_updrpthdr
 *
 *  author    :  Souvanlasy Viengsvanh
 *
 *  revision  :
 *
 *  status    :
 *
 *  language  :  C
 *
 *  object    :  update only the header of a report in a file
 */
int brp_updrpthdr( int iun, BURP_RPT *rpt )
{
   int  istat;
   istat = c_mrbupd( iun, rpt->buffer,
                   RPT_TEMPS(rpt), RPT_FLGS(rpt),
                   RPT_STNID(rpt), RPT_IDTYP(rpt), RPT_LATI(rpt),
                   RPT_LONG(rpt), RPT_DX(rpt), RPT_DY(rpt),
                               RPT_ELEV(rpt), RPT_DRND(rpt), RPT_DATE(rpt),
                   RPT_OARS(rpt), RPT_RUNN(rpt), rpt->sup,
                   rpt->nsup, rpt->xaux, rpt->nxaux );
   return istat;
}
/*
 *----------------------------------------------------------------------
 *
 * Brp_Close --
 *
 *      This procedure is invoked by Brp_FileCmd to close a BURP file
 *
 * Author: Vanh Souvanlasy (July 1994)
 *
 * Results:
 *      A standard Tcl result.
 *
 * Side effects:
 *
 *----------------------------------------------------------------------
 */
 int brp_close(int iun)
{
   int       rtrn;

   rtrn = c_mrfcls( iun );
   rtrn = c_fclos( iun );
   return rtrn;
}
/*
 *----------------------------------------------------------------------
 *
 * Brp_Open --
 *
 *
 * Author: Vanh Souvanlasy
 * revision : Hamid Benhocine
 *
 * Results:
 *
 *
 * Side effects:
 *
 *----------------------------------------------------------------------
 */
 int brp_open(int  iun, const char *filename, char *op)
{
   char type[30], mode[30];
   int  istat;
   FILE *pf;


   if (strcmp(op,"r")==0) { /* READ ONLY */
      strcpy( type, "RND+OLD+R/O" );
      strcpy( mode, "READ" );
   } else
   if (strcmp(op,"w")==0) { /* WRITE ONLY */
      strcpy( type, "RND+R/W" );
      strcpy( mode, "CREATE" );
   } else
   if (strcmp(op,"a")==0) { /* APPEND */
      strcpy( type, "RND+APPEND" );
      strcpy( mode, "APPEND" );

      /* si fichier existe pas ouvrir en ecriture */
      pf = fopen(filename,"r");
      if (pf == (FILE *) NULL) {
        strcpy( type, "RND+R/W" );
        strcpy( mode, "CREATE" );
      }
   } else {
       return -1 ;
   }


   istat = c_fnom( iun , filename, type, 0 );
   if ( istat != 0 )
   {
       fprintf(stderr,"Unable to open file as %s : %s\n",
                type,  filename);
       c_fclos( iun );
       return istat;
   }

   istat = c_mrfopn( iun , mode );
   if ( istat < 0 )
   {
       istat = c_mrfcls( iun );
       istat = c_fclos( iun );
       fprintf(stderr,"Unable to open file as %s : %s\n",
                mode,  filename);
       c_fclos( iun );
       return istat;
   }
   return istat;

}
