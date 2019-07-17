#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
// utiliser librairie burp
#include "burp_api.h"
using namespace std;

// lire un fichier burp et chercher toutes les stations
// et les mettre dans un vecteur. Puis faire une sortie vers le stout
// des sttions trouvees.

extern "C" int my_main ( int argc, char **argv )
{
    BURP_RPT *rs, *rr;

    // voir si qqchose en argument
    if (!(argc -1)) {
        cout << "Un fichier burp en argument, s.v.p" << endl;
        return 1;
    }
    //  ...
    rs = brp_newrpt(); // rpt reference, mettre les cles de recherche 
    rr = brp_newrpt(); // contiendra le rapport trouve
    int iun = 10;
    int istat;
    // option de fichier burp
    istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );
    istat = brp_open(iun,argv[1],"r");
    cout << "Nombre Enreg = " << istat << endl;
    vector<string> vec_one;
    vec_one.reserve(istat);

    while ( brp_findrpt( iun, rs ) >= 0 ) {
        if (brp_getrpt(iun, RPT_HANDLE(rs), rr) >= 0) {
            vec_one.push_back(RPT_STNID(rr));
        }
    }
    unsigned int indx =0;
    vector<string>::const_iterator iter;
    for ( iter = vec_one.begin(); iter !=vec_one.end();++iter) {
        cout << ++indx << "\t" << (*iter) << endl; //Write out vector item
    }
    istat = brp_close(iun);
    cout << "istat = " << istat << endl;
    return 0;
}
/*************************************************************************************
 *                                                                                   *
 *                                                                                   *
 * # Makefile   ----> utilser gmake (Gnu Make) ----------------                      *
 * SHELL    = /bin/sh                                                                *
 * DIRLIB=/software/cmc/burplib_c-1.0                                                *
 *                                                                                   *
 *                                                                                   *
 * ifeq ($(ARCH),Linux)                                                              *
 * CCP=g++                                                                           *
 * else                                                                              *
 * CCP=CC                                                                            *
 * endif                                                                             *
 *                                                                                   *
 * test: test.cpp                                                                    *
 * $(CCP) $@.cpp -I$(DIRLIB)/include -I$(ARMNLIB)/include -L$(ARMNLIB)/lib/$(ARCH) \ *
 * -L$(DIRLIB)/lib  -O3 -Wl,-rpath,$(ARMNLIB)/lib/$(ARCH):$(DIRLIB)/lib \            *
 * -o $@_$(ARCH) -lrmnshared  -lburp_c                                               *
 *                                                                                   *
 *                                                                                   *
 *************************************************************************************/
