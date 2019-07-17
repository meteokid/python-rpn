/*************************************************************************
 * Hamid Benhocine                                                       *
 * Ce programme C++ utilise la libarairie burp C pour lire un fichier    *
 * burp et sort vers le stdout toutes les stations et le nombre          *
 * d'observations  pour chacune des stations et le total pour tout       *
 * le fichier burp                                                       *
 *************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
// pour utilser la librairie burp
#include "burp_api.h"
using namespace std;

extern "C" int my_main ( int argc, char **argv )
{
    BURP_BLK *bs, *br;
    BURP_RPT *rs, *rr;

    // voir si qqchose en argument
    if (!(argc -1)) {
        cout << "Un fichier burp en argument, s.v.p" << endl;
        return 1;
    }
    //  ...
    bs = brp_newblk() ; // bloc reference (ou on met les cles de recherche)
    br = brp_newblk() ; // bloc resultat
    rs = brp_newrpt() ; // rpt reference  (ou on met les cles de recherche)
    rr = brp_newrpt() ; // rpt resultat
    int iun = 10      ; // unite pour ouverture de fichier
    int istat         ; // retour des fonctions burp
    BLK_SetSTORE_TYPE( bs, STORE_FLOAT );
    cout <<" store type : " << BLK_STORE_TYPE(bs) << endl;

    // niveau de 
    istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );
    // ouverture fichier burp passe en argument
    istat = brp_open(iun,argv[1],"r");
    // si erreur terminer programme
    if (istat < 0) return 1;
    // si ok, istat represente nombre enregistrements
    cout << "Nombre Enreg = " << istat << endl;
    // creer un container de type map, dont la cle est la
    // chaine de caratere (la station) et la valeur est est un entier
    // qui represente le nombre d'observations pour cette station
    map<string, int> counters;

//    RPT_SetIDTYP (rs,181);
    while ( brp_findrpt( iun, rs ) >= 0 ) 
    {
        if (brp_getrpt(iun, RPT_HANDLE(rs), rr) >= 0) 
        {
            if (!(RPT_STNID(rr)[0] == '^')) 
            {
                // donnees non regroupees 
                // incremente nombre observations de 1
                ++counters[RPT_STNID(rr)];
                continue;
            }
        // si station commence par ^ alors on suppose que ce sont
        // des donnees regroupees, alors chercher
        // le block 3D. Si block 3D est trouve, alors le traiter, sinon
        // considerer l'enregsitrement non regroupe.
        BLK_SetBKNO  (bs, 0);
        bool Trouve = false;
        while ( brp_findblk( bs, rr ) >= 0 ) 
        {
//            if (brp_rdblkhdr( BLK_BKNO(bs), br, rr )>=0) 
            if (brp_getblk( BLK_BKNO(bs), br, rr )>=0) 
            {
                if ((BLK_BKNAT(br) &= 3) == 2) 
                {
                    // incremente nombre observations de NT
                    counters[RPT_STNID(rr)] += BLK_NT(br);
                    Trouve = true;
                    break;
                }
            }
        }
        if (!Trouve) 
        {
            // donnees non regroupees 
            // incremente nombre observations de 1
            ++counters[RPT_STNID(rr)];
        }
        }
    }
    // envoyer vers le stdout le resultats
    int i     = -1;
    int total =  0;
    for (map<string,int>::const_iterator iter = counters.begin(); iter !=counters.end();++iter) {
        cout << ++i << ")\t" << iter->first << "\t" << iter->second << endl;
        total += iter->second;
    }
    cout << "-----" << "\t" << "--------" << "\t" << "---------" << endl;
    cout << "     " << "\t" << "Total   " << "\t" <<  total      << endl;
    // fermeture de fichier burp
    istat = brp_close(iun);
    // liberation des ressources
    brp_freeblk(bs) ; 
    brp_freeblk(br) ;
    brp_freerpt(rs) ;
    brp_freerpt(rr) ;
    return 0;
}

/******************************************************************************************************
 *                                                                                                    *
 *                                                                                                    *
 * # Makefile   ----> utilser gmake (Gnu Make) ----------------                                       *
 * SHELL    = /bin/sh                                                                                 *
 * DIRLIB=/software/cmc/burplib_c-1.0                                                                 *
 *                                                                                                    *
 *                                                                                                    *
 * ifeq ($(ARCH),Linux)                                                                               *
 * CCP=g++                                                                                            *
 * else                                                                                               *
 * CCP=CC                                                                                             *
 * endif                                                                                              *
 *                                                                                                    *
 * obs: obs.cpp                                                                                       *
 * $(CCP)  $@.cpp -I$(DIRLIB)/include -I$(ARMNLIB)/include -L$(ARMNLIB)/lib/$(ARCH) -L$(DIRLIB)/lib \ *
 * -O3 -Wl,-rpath,$(ARMNLIB)/lib/$(ARCH):$(DIRLIB)/lib -o $@_$(ARCH) -lrmnshared  -lburp_c            *
 *                                                                                                    *
 *                                                                                                    *
 *                                                                                                    *
 ******************************************************************************************************/
