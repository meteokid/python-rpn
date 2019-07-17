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
#include <cstring> // for strcmp
using namespace std;
// convertit un entier en objet string 
std::string ToString(int );

extern "C" int my_main ( int argc, char **argv )
{
   bool debug = false;
   //creer un container de type map, dont la cle est la
   //chaine de caratere (la station) et la valeur est est un entier
   //qui represente le nombre d'observations pour cette station
   map< int, set <int> > counters;
//   map< int,int > counters;

   int e,k,j;
   for ( k = 0 ; k !=1 ; ++k ) {

      /* sortie des valeurs des elements */
      for (  j = 0 ; j !=10 ; ++j ) {
         (counters[k]).insert(j);
      }
   }
   // si aucun element terminer
   if (counters.empty()) return 0;
   set<int>::const_iterator pos;
   map<int,set<int> >::iterator imap;
   for (imap = counters.begin(); imap !=counters.end();++imap)
   {
      cout << "Element " << imap->first << " a pour valeurs " << endl;
      for (pos = (imap->second).begin(); pos !=(imap->second).end();++pos)
      {
         cout << *pos  << endl;
      }

   }



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

