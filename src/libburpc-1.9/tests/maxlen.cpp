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

    bs = brp_newblk();

    // voir si qqchose en argument
    if (!(argc -1)) {
        cout << "Un fichier burp en argument, s.v.p" << endl;
        return 1;
    }
    //  ...
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
    cout << "max length = " << c_mrfmxl(iun) << endl;
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
