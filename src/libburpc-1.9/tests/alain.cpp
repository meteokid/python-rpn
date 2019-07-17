/**************************************************************************
 * Hamid Benhocine                                                        *
 * Programme c++ utilsant la libraire burplib_c                           *
 * Objectif: pour chaque enregistrement du fichier burp  *
 * reecrire l'enregistrement dans le fichier, sauf q'un element           *
 * a ete ajoute pour le bloc de btyp 3072                                 *
 **************************************************************************/

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
// pour utilser la librairie burp
#include "burp_api.h"
using namespace std;

extern "C" int my_main ( int argc, char **argv )
{
    BURP_BLK *bs, *br;
    BURP_RPT *rs, *rr, *rout;

    // voir si qqchose en argument
    if (!(argc -1)) 
    {
        cout << "Un  fichier burp en argument, s.v.p" << endl;
        return 1;
    }
    //  ...
    // Intisialser les pointeurs
    bs      = brp_newblk(); // bloc reference (ou on met les cles de recherche)
    br      = brp_newblk(); // bloc resultat
    rs      = brp_newrpt(); // rpt reference  (ou on met les cles de recherche)
    rr      = brp_newrpt(); // rpt resultat
    rout    = brp_newrpt(); // sera utilsee pour un b=noveau rapport
    int iun = 10          ; // unite pour ouverture de fichier
    int istat             ; // retour des fonctions burp

    // niveau de tolerance erreur burp 
    istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );

    //ouverture fichier burp passe en argument
    istat = brp_open(iun,argv[1],"a");
    // si erreur terminer  passer au suivant
    if (istat < 0) 
        return 1;
    // si ok, istat represente nombre enregistrements
    cout << "Nombre Enreg = " << istat << endl;
    // vecteur pour contenir les adresses
    vector< int> addresses; 
    // chercher toutes les adresses des enregsitrements
    RPT_SetHANDLE( rs, 0 );
    while ( brp_findrpt( iun, rs ) >= 0 )
        addresses.push_back(RPT_HANDLE(rs));

    // si aucun enregistremnt trouve quitter
    if (addresses.empty()) return 0;

    // chercher la longueur du plus long enregsitrement
    int LenMax = c_mrfmxl(iun);
    // puisque le but du prog est de trouver un bloc
    // particulier et ajouter un element au bloc et des valeurs
    // pour cet element, on alloue de l'espace pour
    // l'enregistrement d'ecriture un espace suuffisant
    // pour ajouter envriron 10000 entiers par bloc
    brp_allocrpt( rout, LenMax + 10000 );

    typedef vector<int>::const_iterator iter_v;
    for (iter_v it = addresses.begin(); it !=addresses.end();++it) 
    {
        if ( brp_getrpt( iun, *it,rr ) < 0 )
            continue;
        brp_clrrpt(   rout );
        brp_copyrpthdr(rout,rr);
        brp_putrpthdr( iun, rout );
        // chercher tous les blocs
        BLK_SetBKNO  (bs, 0);
        while ( brp_findblk( bs, rr ) >= 0 ) 
        {
            if (brp_getblk( BLK_BKNO(bs), br, rr ) < 0)
                continue; 
            if (BLK_BTYP(br) == 3072 ) 
            {
                // ajouter un element au block
                brp_resizeblk(br,BLK_NELE(br)+1, BLK_NVAL(br), BLK_NT(br));
                BLK_SetDLSTELE(br,BLK_NELE(br)-1,62255);
                for ( int v = 0, t = 0; v != BLK_NVAL(br);++v)
                {
                    for (; t != BLK_NT(br);++t)
                        BLK_SetTBLVAL(br,BLK_NELE(br)-1,v,t,15);
                }
                brp_encodeblk(br);
            }
            brp_putblk(rout,br);
        }
        int overwrite = RPT_HANDLE(rr);
        // on ecrit l'enregistrement rout a la place de rr
        // si on voulait ajouter cet enregistrement sans detruire rr
        // alors faire brp_writerpt(iun, rout, END_BURP_FILE)
        brp_writerpt(iun,rout,overwrite);
    }
    // fermeture de fichier a burp 
   istat = brp_close(iun);

   // liberer les ressources
   brp_freeblk(bs);  
   brp_freeblk(br);
   brp_freerpt(rs);  
   brp_freerpt(rr);
   brp_freerpt(rout);
   // terminer programme
   return 0;
}
