/*************************************************************************
 * Hamid Benhocine                                                       *
 * Ce programme C utilise la libarairie burp C pour lire un fichier      *
 * burp et sort vers le stdout tout les enregistrements et leurs contenu *
 *************************************************************************/

#include <stdio.h>
/* pour utilser la librairie burp*/
#include "burp_api.h"
#define MIS_VAL   -99.99

int my_main ( int argc, char **argv )
{
   int istat;
   int i,j,k;
   int iun;
   BURP_BLK *bs, *br;
   BURP_RPT *rs, *rr;
   float toto;

   /* voir si qqchose en argument */
   if (!(argc -1)) {
      printf("Un fichier burp en argument, s.v.p\n");
      return 1;
   }
   /* niveau de tolerance erreur burp */
   istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );
   istat = brp_SetOptFloat ( "MISSING", MIS_VAL ) ; 

   /*  ... */
   bs = brp_newblk() ; /* bloc reference (ou on met les cles de recherche) */
   br = brp_newblk() ; /* bloc resultat */
   rs = brp_newrpt() ; /* rpt reference  (ou on met les cles de recherche) */
   rr = brp_newrpt() ; /* rpt resultat */
   iun = 10          ; /* unite pour ouverture de fichier */

   /* ouverture fichier burp passe en argument */
   istat = brp_open(iun,argv[1],"r");
   /* si erreur terminer programme */
   if (istat < 0) 
      return 1;
   /* si ok, istat represente nombre enregistrements */
   printf("Nombre Enreg = %d\n", istat );

   /* commencer recherche par debut fichier */
   RPT_SetHANDLE( rs, 0);
   while ( brp_findrpt( iun, rs ) >= 0 ) 
   {
      if (brp_getrpt(iun, RPT_HANDLE(rs), rr)  < 0) 
         continue;
      /* entete de rapport */
      printf ( "\n\n" ) ;
      printf ( "hhmm   =%8d " , RPT_TEMPS(rr)) ;
      printf ( "flgs   =%6d  ", RPT_FLGS(rr )) ;
      printf ( "codtyp =%6d  ", RPT_IDTYP(rr)) ;
      printf ( "stnids =%9s\n", RPT_STNID(rr)) ;
      printf ( "blat   =%8d " , RPT_LATI(rr)) ;
      printf ( "blon   =%6d  ", RPT_LONG(rr )) ;
      printf ( "dx     =%6d  ", RPT_DX(rr)) ;
      printf ( "dy     =%6d  ", RPT_DY(rr)) ;
      printf ( "stnhgt =%6d\n", RPT_ELEV(rr)) ;
      printf ( "yymmdd =%8d " , RPT_DATE(rr)) ;
      printf ( "oars   =%6d  ", RPT_OARS(rr)) ;
      printf ( "runn   =%6d  ", RPT_RUNN(rr)) ;
      printf ( "nblk   =%6d  ", RPT_NBLK(rr)) ;
      printf ( "dlay   =%6d\n", RPT_DRND(rr)) ;
      printf ( "\n" ) ;

      /* chercher tous les blocs */
      BLK_SetBKNO  (bs, 0);
      while ( brp_findblk( bs, rr ) >= 0 ) 
      {
         if (brp_getblk( BLK_BKNO(bs), br, rr ) < 0) 
/*         if (brp_readblk( BLK_BKNO(bs), br, rr,0 ) < 0) */
               continue; 
         /* entete de bloc */
         printf ( "\n" ) ;
         printf ( "blkno  =%6d  ", BLK_BKNO(br)    ) ;
         printf ( "nele   =%6d  ", BLK_NELE(br)    ) ;
         printf ( "nval   =%6d  ", BLK_NVAL(br)    ) ;
         printf ( "nt     =%6d  ", BLK_NT(br)      ) ;
         printf ( "bit0   =%6d\n", BLK_BIT0(br)    ) ;
         printf ( "bdesc  =%6d  ", BLK_BDESC(br)   ) ;
         printf ( "btyp   =%6d  ", BLK_BTYP(br)    ) ;
         printf ( "nbit   =%6d  ", BLK_NBIT(br)    ) ;
         printf ( "datyp  =%6d  ", BLK_DATYP(br)   ) ;
         printf ( "bfam   =%6d\n", BLK_BFAM(br)    ) ;

         for ( k = 0 ; k < BLK_NT(br) ; k++ ) 
         {
               if ( BLK_NT(br) != 1 ) 
                  printf (  "\n\nobservation %d/%d", k+1, BLK_NT(br) ) ;

               /* sortie des elements des blocs du fichier burp */
               printf (  "\nlstele =" ) ;
               for (  i = 0 ; i < BLK_NELE(br) ; i++ ) 
               {
                  printf (  "    %6.6d", BLK_DLSTELE(br,i) ) ;
                  if ( ((i+1)%7) == 0 && i != 0 && i+1 < BLK_NELE(br) )
                     printf (  "\n        " ) ;
               }
               /* sortie des valeurs des elements */
               for (  j = 0 ; j < BLK_NVAL(br) ; j++ ) 
               {
                  printf (  "\ntblval =" ) ;
                  for (  i = 0 ; i < BLK_NELE(br) ; i++ ) 
                  {
/*                     printf (  "%10d", BLK_TBLVAL(br,i,j,k) ) ;*/
                     printf (  "%15.6G", BLK_RVAL(br,i,j,k) ) ;
                     if ( ((i+1)%7) == 0 && i != 0 && i+1 < BLK_NELE(br) )
                           printf (  "\n        " ) ;
                  }
               }
         }
         if ( BLK_NT(br) != 1 ) 
               printf (  "\n" ) ;
         printf ( "\n" ) ;
      }
   }
/*               if ( conv && ((bknat%4)!=3) )*/
/*                  printf ( "%10.2f", tblvalf[i+nele*(j+nval*k)] ) ;*/
   /* fermeture de fichier burp */
   istat = brp_close(iun);
   /* liberer les ressources */
   brp_freeblk(br) ;
   brp_freeblk(bs) ;
   brp_freerpt(rr) ;
   brp_freerpt(rs) ;
   /* fin programme */
   return 0;
}


