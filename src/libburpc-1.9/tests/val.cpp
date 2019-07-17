/***************************************************************************
*  Hamid Benhocine                                                          *
*  Exemple C++ utilisant la libarairie burp C                               *
*  Ce programme traite les fichiers burp passes en                          *
*  arguments, cherche tous les elements uniques presents dans tous les blocs*
*  a l'exception des blocks marqueurs et fait une sortie vers le stdout     *
*  avec la description de chacun comme trouve dans la table burp            *
****************************************************************************/
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstring>
// pour utiliser la librairie burp
#include "burp_api.h"
using namespace std;
// convertit un entier en objet string 
std::string ToString(int );
bool parse_arguments ( int argc, char *argv[], int& idtyp , int& btyp, int& bfam,
                       std::string& burpin,int & elem, bool &);

extern "C" int my_main ( int argc, char **argv )
{
   BURP_BLK *bs, *br;
   BURP_RPT *rs, *rr;

   // voir si qqchose en argument
//   if (!(argc -1)) 
//   {
//      cout << "Un ou +eurs fichiers burp en argument, s.v.p" << endl;
//      return 1;
//   }
   bool debug = false;
   string burpin("none");
   int idtyp = -1;
   int btyp = -1;
   int bfam = -1;
   int elem =-1;

   bool no_erreur = parse_arguments(argc,argv,idtyp,btyp,bfam,burpin,elem,debug);
//   cout << "idtyp = " << idtyp << endl;
//   cout << "btyp = " << btyp << endl;
//   cout << "bfam = " << bfam << endl;
//   cout << "burpin = " << burpin << endl;
//   cout << "debug  = " << debug  << endl;
   if (! no_erreur)  {
      cerr << "usage= monitor [-idtyp ] [-bfam  ] [-btyp ] [-elem elem] -f fichier_burp " << no_erreur << endl;
      exit(1);
   }
   // voir si qqchose en argument
   if (burpin == "none") {
      cerr << "Un fichier burp en argument, s.v.p" << endl;
      exit(1);
   }
   //  ...
   bs = brp_newblk() ; // bloc reference (ou on met les cles de recherche)
   br = brp_newblk() ; // bloc resultat
   rs = brp_newrpt() ; // rpt reference  (ou on met les cles de recherche)
   rr = brp_newrpt() ; // rpt resultat
   int iun = 10      ; // unite pour ouverture de fichier
   int istat         ; // retour des fonctions burp

   // niveau de tolerance erreur burp 
   istat = brp_SetOptChar ( "MSGLVL",  "FATAL" );
   //creer un container de type map, dont la cle est la
   //chaine de caratere (la station) et la valeur est est un entier
   //qui represente le nombre d'observations pour cette station
   map< int, set <int> > counters;
//   map< int,int > counters;

   //ouverture fichier burp passe en argument
   istat = brp_open(iun,burpin.c_str(),"r");
   // si erreur terminer  passer au suivant
   if (istat < 0) exit(1);
   // cout << argv[i] << endl;
   // si ok, istat represente nombre enregistrements
   // cout << "Nombre Enreg = " << istat << endl;
   RPT_SetHANDLE( rs, 0 );
   RPT_SetIDTYP(rs,idtyp);
   int count = 0;
   while ( brp_findrpt( iun, rs ) >= 0 ) 
   {
//      cout << "okay " << ++count <<  endl;
//      continue; 

      if (brp_getrpt(iun, RPT_HANDLE(rs), rr)  < 0) 
            continue;
      // chercher tous les blocs
      BLK_SetBKNO  (bs, 0);
      BLK_SetBFAM  (bs, bfam);
      BLK_SetBTYP  (bs, btyp);
      string typeBlock;
      int e,k,j;
      while ( brp_findblk( bs, rr ) >= 0 ) 
      {
//            if (brp_getblk( BLK_BKNO(bs), br, rr ) < 0)
            if (brp_readblk( BLK_BKNO(bs), br, rr,0 ) < 0)
               continue; 
            e = brp_searchdlste( elem, br ); 
            if (e < 0) 
               continue;
            for ( k = 0 ; k != BLK_NT(br) ; ++k ) {

               /* sortie des valeurs des elements */
               for (  j = 0 ; j != BLK_NVAL(br) ; ++j ) {
                (counters[elem]).insert(BLK_TBLVAL(br,e,j,k));
               }
            }
      }
   }
//   return
   // fermeture de fichier burp
   istat = brp_close(iun);
   // si aucun element terminer
   if (counters.empty()) return 0;
   set<int>::const_iterator pos;
   map<int,set<int> >::const_iterator imap;
   for (imap = counters.begin(); imap !=counters.end();++imap)
   {
      cout << "Element " << imap->first << " a pour valeurs " << endl;
      for (pos = (imap->second).begin(); pos !=(imap->second).end();++pos)
      {
         cout << *pos  << endl;
      }

   }



   // liberer les ressources
   brp_freeblk(bs);  
   brp_freeblk(br);
   brp_freerpt(rs);  
   brp_freerpt(rr);

   // terminer programme
   return 0;
}
// convertit entier avec signe en objet string
std::string ToString(int value)
{
   bool negative;
   if (value < 0) { value = -value; negative = true; }
   else if (value == 0) return string("0");
   else negative = false;
   char b[100]; b[99] = (char)0; int i = 99;
   while (value > 0)
   {
      int nv = value / 10; int rm = value - nv * 10;
      b[--i] = (char)(rm + '0'); value = nv;
      if (i<=0) return ("*****");
   }
   if (negative) b[--i] = '-';
   return std::string(b+i);
}
bool parse_arguments ( int argc, char *argv[], int& idtyp,int& btyp, int& bfam,
                       std::string& burpin,int &elem, bool &debug)
{
   bool   no_erreur = true ;
   for (int i = 1; i < argc; i++) {
      if ( strcmp ( argv[i], "-bfam" ) == 0 ) {
         if ( ++i >= argc )
            no_erreur = false ;
         else
            bfam = atoi(argv[i]);
      } else if ( strcmp ( argv[i], "-btyp" ) == 0 ) {
         if ( ++i >= argc )
            no_erreur = false ;
         else
            btyp = atoi(argv[i]);
      } else if ( strcmp ( argv[i], "-elem" ) == 0 ) {
         if ( ++i >= argc )
            no_erreur = false ;
         else
            elem = atoi(argv[i]);
      } else if ( strcmp ( argv[i], "-idtyp" ) == 0 ) {
         if ( ++i >= argc )
            no_erreur = false ;
         else
            idtyp = atoi(argv[i]);
      } else if ( strcmp ( argv[i], "-f" ) == 0 ) {
         if ( ++i >= argc )
            no_erreur = false ;
         else
            burpin = argv[i];
      } else if ( strcmp ( argv[i], "-d" ) == 0 ) {
            debug = true;
      } else
         no_erreur = false ;
   }
   return ( no_erreur ) ;
}

