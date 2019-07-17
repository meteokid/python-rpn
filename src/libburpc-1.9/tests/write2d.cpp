/***************************************************************************
 * Hamid Benhocine                                                       * *
 * Ce programme C++ utilise la libarairie burp C un creer un fichier       *
 * burp de 1 enregistrement de 4 blocs                                     *
 ***************************************************************************/

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
// pour utiliser la librairie burp
#include "burp_api.h"
using namespace std;

extern "C" int my_main ( int argc, char **argv )
{
    BURP_RPT  *rr;
    BURP_BLK  *br, *br2,*tmp;

    // voir si qqchose en argument
    if (!(argc -1)) {
        cout << "Un nom de fichier en argument, s.v.p" << endl;
        return 1;
    }
    //  ... Toujours initiliser les pointeurs
    rr        = brp_newrpt() ; // rpt
    br        = brp_newblk() ; // blk
    br2       = brp_newblk() ; // blk
    tmp       = brp_newblk() ; 
    int oun   = 20           ; // unite pour ouverture de fichier 
    int istat ;                // retour des fonctions burp

    // niveau de tolerance erreur burp 
    istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );

    //ouverture fichier burp passe en argument
    istat = brp_open(oun,argv[1],"a"); // en ecriture

    // si erreur terminer programme
    if (istat < 0) 
        return 1;

    // remplir entete enregistrement
    RPT_SetTEMPS( rr , 1200    ) ; 
    RPT_SetFLGS(  rr , 0       ) ;
    RPT_SetSTNID( rr , "74724" ) ;
    RPT_SetIDTYP( rr , 32      ) ;
    RPT_SetLATI(  rr , 14023   ) ;
    RPT_SetLONG(  rr , 27023   ) ;
    RPT_SetDX(    rr , 0       ) ;
    RPT_SetDY(    rr , 0       ) ;
    RPT_SetELEV(  rr , 0       ) ;
    RPT_SetDRND(  rr , 0       ) ;
    RPT_SetDATE(  rr , 20050317) ;
    RPT_SetOARS(  rr , 0       ) ;

    // allouer espace pour l'enregistrement
    // pour ajouter des blocs
    brp_allocrpt( rr, 10000 );

    // on peut reallouer + espace
    brp_resizerpt(rr, 20000 );

    // on peut voir quantite de memoire allouee
    cout << " rr apres resize:\t"<< RPT_NSIZE(rr) << endl;

    // on peut mettre le contenu du rapport a 0
    // cela n'affecte pas le header
    brp_clrrpt( rr );

    // ici c'est l'ecriture de l'enrgistrement dans le fichier
    brp_putrpthdr( oun, rr );

    // Section ajout de blocs dans l'enregistrement

    // Ici on indique que l'on desire remplir
    // le bloc br de valueurs reelles
    BLK_SetSTORE_TYPE( br, STORE_DOUBLE );

    // setter les params du bloc BFAM,BDESC et BTYP
    BLK_SetBFAM( br,  0 ) ;
    BLK_SetBDESC( br, 0 ) ;
    BLK_SetBTYP( br, 64 ) ;
    BLK_SetDATYP(br,7   ) ;  

    // allouer espace pour remplir le bloc
    // ici pour 2 elements et 1 valeur par element et
    // 1 groupe nelem X nval
    brp_allocblk( br, 2, 1, 1 );

    // Les indices en C commencent par 0
    // on remplit les elements: 7004 et 11001
    BLK_SetDLSTELE( br, 0, 7004 );
    BLK_SetDLSTELE( br, 1, 11001 );

    // Compacter les elements
    brp_encodeblk( br );

    // remplir les valeures pour chacun des elements
    BLK_SetDVAL( br, 0,0,0, 0.000096 ); // pour 7004
    BLK_SetDVAL( br, 1,0,0, 0.000007 ); // pour 11001

    // on a rempli les valeurs reelles
    // alors les convertir selon la table burp
    // en entiers qui seront enregistres dans le fichier burp
//#define  MKSA_to_BUFR		1
    if (brp_convertblk( br,MKSA_to_BUFR) < 0)
        return 1;

    // on met le bloc br dans l'enrgistrement rr
    istat =  brp_putblk( rr, br ) ;
    if (istat < 0) {
       cout << "istat - " << istat;
        return 1;
    }


    // on peut faire une copie du bloc br 
    // br2 est une copie de br
    // tous les attributs de br le seront pour br2
    brp_copyblk(br2,br);

    // on met le bloc br2 dans l'enrgistrement rr
    if (brp_putblk( rr, br2 ) < 0)
        return 1;
    
    // on peut redimensionner le bloc br
    // pour contenir 5 elements, 10 valeurs par element
    // et aussi 2 tuiles de 5 X 10
    // comme ici c'est une expansion du bloc
    // donc on retrouvera les elements et leurs
    // valeures (precedentes aux memes indices)
    brp_resizeblk(br,5,10,2);

    // on ajoute ce bloc a l'enregistrement
    brp_putblk( rr, br );

    // ici on fait une copie de br
    // tmp est une copie de br
    brp_copyblk(tmp,br);

    // redimensionner le bloc tmp
    // ici il s'agit d'une reduction
    // 3 elements, 2 valeures par element et 
    // 1 tuile de 3 x 2
    brp_resizeblk(tmp,3,2,1);

    // Ici on indique que l'on desire remplir
    // le bloc br de valueurs entieres 
     BLK_SetSTORE_TYPE( tmp, STORE_INTEGER);
 
     // setter l'element 3 a 11003
     BLK_SetDLSTELE( tmp, 2, 11003 );
     // et ses valeures entieres
     BLK_SetTBLVAL( tmp, 2,0,0, 15 );
     BLK_SetTBLVAL( tmp, 2,1,0, 30 );
 
     // compacter les elements du bloc
     brp_encodeblk( tmp );
 
     // ajouter le bloc tmp a l'enrgistrement rr
     brp_putblk( rr, tmp );

    // ajouter l'enrgistrement dans le fichier
    if (brp_writerpt( oun, rr, END_BURP_FILE ) < 0)
        return 1;

    // fermeture de fichier burp
    istat = brp_close(oun);
    if (istat < 0 )
        return 1;

    // liberer ressources
    brp_freerpt(rr);
    brp_freeblk(br);
    brp_freeblk(br2);
    brp_freeblk(tmp);

    // fin du programme
    return 0;
}



/* 
   --- execution apres compilation --
   --- visualisation du fichier cree a l'aide de liburp ----

   galilee% write2_Linux burp-file 
   rr apres resize:       20000

   galilee% liburp burp-file       

   19:03:57 30/03/05 debut de liburp v2.2

   hhmm   =    1200 flgs   =     0  codtyp =    32  stnids =74724    
   blat   =   14023 blon   = 27023  dx     =     0  dy     =     0  stnhgt =     0
   yymmdd =20050317 oars   =     0  runn   =     0  nblk   =     4  dlay   =     0

   blkno  =     1  nele   =     2  nval   =     1  nt     =     1  bfam   =     0
   bdesc  =     0  btyp   =    64  nbit   =    17  bit0   =     0  datyp  =     2

   lstele =    007004    011001
   tblval =         1        20

   blkno  =     2  nele   =     2  nval   =     1  nt     =     1  bfam   =     0
   bdesc  =     0  btyp   =    64  nbit   =    17  bit0   =     1  datyp  =     2

   lstele =    007004    011001
   tblval =         1        20

   blkno  =     3  nele   =     5  nval   =    10  nt     =     2  bfam   =     0
   bdesc  =     0  btyp   =    64  nbit   =    17  bit0   =     2  datyp  =     2


   observation 1/2
   lstele =    007004    011001    000000    000000    000000
   tblval =         1        20        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1

   observation 2/2
   lstele =    007004    011001    000000    000000    000000
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1
   tblval =        -1        -1        -1        -1        -1

   blkno  =     4  nele   =     3  nval   =     2  nt     =     1  bfam   =     0
   bdesc  =     0  btyp   =    64  nbit   =    17  bit0   =    30  datyp  =     2

   lstele =    007004    011001    011003
   tblval =         1        20        15
   tblval =        -1        -1        30
   19:03:58 30/03/05 fin   de liburp
 */
