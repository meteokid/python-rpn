/*************************************************************************
* Hamid Benhocine                                                       *
* Ce programme C++ utilise la libarairie burp C pour lire un fichier    *
* burp 1 en lecture. Cherche tous les enregistrements de codetype 32 et *
* de l'heure de validite de 2300. Pour tous ces enregistrements, on     *
* change l'heure de validite pour 2200 et on ecrit ces enrgistrements   *
* dans le fichier 2 qui est ouvert en ecriture. Les fichiers 1 et 2     *
* sont les arguments 1 et 2 du programme.                               *
* tester avec un fichier /data/ade/dbase/uprair/radiosonde/yyyymmjj00_  *
*************************************************************************/

#include <iostream>
#include <iomanip>
// pour utilser la librairie burp
#include "burp_api.h"
using namespace std;

extern "C" int my_main ( int argc, char **argv )
{
    BURP_RPT *rs, *rr;

    // voir si qqchose en argument
    if (!(argc -1)) {
        cout << "2 fichiers burp en argument, s.v.p  " << endl;
        return 1;
    }
    //  ...
    rs = brp_newrpt() ; // rpt reference  (ou on met les cles de recherche)
    rr = brp_newrpt() ; // rpt resultat
    int iun = 10      ; // unite pour ouverture de fichier lecture
    int oun = 20      ; // unite pour ouverture de fichier ecriure
    int istat         ; // retour des fonctions burp

    // niveau de tolerance erreur burp 
    istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );
    //ouverture fichier burp passe en argument
    istat = brp_open(iun,argv[1],"r"); // en lecture
    istat = brp_open(oun,argv[2],"w"); // en ecriture
    // si erreur terminer programme
    if (istat < 0) 
        return 1;

    // fixer les criteres de recherche
    RPT_SetHANDLE(rs, 0 ); // commencer du debut de fichier
    RPT_SetTEMPS(rs,2300); // heure
    RPT_SetIDTYP(rs,32)  ; // codetyp
    // chercher les enregistrements 
    while ( brp_findrpt( iun, rs ) >= 0 ) 
    {
        if (brp_getrpt(iun, RPT_HANDLE(rs), rr)  < 0) 
            continue;
        // on change l'heure de l'enregistrement 
        RPT_SetTEMPS(rr,2200);
        // on  met a jour entete enregistrement
        brp_updrpthdr(oun, rr );
        // on l'ecrit dans le fichier
        brp_writerpt( oun, rr , END_BURP_FILE);
    }
    // fermeture de fichiers burp
    istat = brp_close(iun);
    istat = brp_close(oun);

    // liberer ressources
    brp_freerpt(rs);
    brp_freerpt(rr);
    return 0;
}
