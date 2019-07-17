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
                       std::string& burpin, bool &);

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

   bool no_erreur = parse_arguments(argc,argv,idtyp,btyp,bfam,burpin,debug);
//   cout << "idtyp = " << idtyp << endl;
//   cout << "btyp = " << btyp << endl;
//   cout << "bfam = " << bfam << endl;
//   cout << "burpin = " << burpin << endl;
//   cout << "debug  = " << debug  << endl;
   if (! no_erreur)  {
      cerr << "usage= monitor [-idtyp ] [-bfam  ] [-btyp ] -f fichier_burp " << no_erreur << endl;
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
   map< string, set <int> > counters;
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
      while ( brp_findblk( bs, rr ) >= 0 ) 
      {
//            if (brp_getblk( BLK_BKNO(bs), br, rr ) < 0)
            if (brp_readblk( BLK_BKNO(bs), br, rr,0 ) < 0)
               continue; 
            // sauter les blocs marqueurs
            if ( (BLK_BKNAT(br) &= 3) == 3)
               continue; 

            switch (BLK_BKNAT(br) &= 3  )
            {
               case 3 : 	
//                  cout << " mrq " <<  endl;
//                  typeBlock = "mrq  " + ToString(BLK_BKNO(br));
                  typeBlock = "mrq  " ;
                  break;

               case 2 : 	
//                  cout << " 3-d " <<  endl;
//                  typeBlock = "3-d  " + ToString(BLK_BKNO(br));
                  typeBlock = "3-d  " ;
                  break;

               case 1 : 	
//                  cout << " info " <<  endl;
//                  typeBlock = "info " + ToString(BLK_BKNO(br));
                  typeBlock = "info " ;
                  break;

               case 0 : 	
//                  cout << " data " <<  endl;
//                  typeBlock = "data " + ToString(BLK_BKNO(br));
                  typeBlock = "data " ;
                  break;

               default:	
                  break;
            }				/* -----  end switch  ----- */
            // sortie des elements des blocs du fichier burp 
             for ( int i = 0 ; i < BLK_NELE(br) ; i++ ) 
             {
 //               counters[BLK_DLSTELE(br,i)]= 0;
                (counters[typeBlock]).insert(BLK_DLSTELE(br,i));
             }
      }
   }
//   return
   // fermeture de fichier burp
   istat = brp_close(iun);
   // si aucun element terminer
   if (counters.empty()) return 0;
   set<int>::iterator pos;
   map<string,set<int> >::const_iterator imap;
   //for (imap = counters.begin(); imap !=counters.end();++imap)
   //{
//      cout << imap->first << endl;
      //for (pos = ((imap->second).begin()); pos !=(imap->second).end();++pos)
      //{
//         cout << *pos  << endl;
      //}

   //}


//    // construire un vecteur des elements de type string
//    vector <string> e;
//    string conv;
//    for (map<int,int>::const_iterator iter = counters.begin(); iter !=counters.end();++iter) 
//    {
//       // conversion de int a string
//       conv =  ToString(iter->first);
//       string stmp(6 - conv.size(),'0') ;
//       conv = stmp + conv;
//       e.push_back(conv);
//    }
// 
// 
//    // table burp des elements
//    ifstream In("/home/binops/afsi/sio/datafiles/constants/table_b_bufr");
// 
//    // cout << endl << "Read data from file" << endl;
//    typedef string::const_iterator is;
//    vector<string>::const_iterator resultIter;
//    string line;
//    // chercher la premiere ligne, nous n;avons pas besoin
//    getline (In, line);
//    while ( ! In.eof() ) 
//    {
//       getline (In, line);
//       is i = line.begin();
//       // ne rien faire si ligne commentaire
//       if (string(i,i+1) == "*") 
//          continue;
//       // regarder dans le vecteur elements si on trouve l'element
//       // qui est sur les 6 premiers caracteres de la ligne
//       resultIter = find(e.begin(), e.end(), string(i,i+6));
//       // si element triuve dans le container alors faire une sortie
//       if (resultIter != e.end()) 
//          cout << line << endl; 
//    }
//    // fermer le fstream
//    In.close();
    // print tous les elements
    // si on le desire
    // decommenter
 
    //for (resultIter = e.begin();  resultIter !=e.end();++resultIter) 
    //{
    //        cout << *resultIter << endl; 
    //}
 
    // construire un vecteur des elements de type string
//    vector <string> e;
//    string conv;
//    for (map<int,int>::const_iterator iter = counters.begin(); iter !=counters.end();++iter) 
//    {
//        conversion de int a string
//       conv =  ToString(iter->first);
//       string stmp(6 - conv.size(),'0') ;
//       conv = stmp + conv;
//       e.push_back(conv);
//    }
 
 
    // table burp des elements
    ifstream In("/home/binops/afsi/sio/datafiles/constants/table_b_bufr_e");
//    ifstream In("/home/binops/afsi/sio/datafiles/constants/table_b_bufr_f");
 
    // cout << endl << "Read data from file" << endl;
    typedef string::const_iterator is;
    set<int>::const_iterator resultIter;
    string line;
    // chercher la premiere ligne, nous n;avons pas besoin
    getline (In, line);
    while ( ! In.eof() ) 
    {
       getline (In, line);
       is i = line.begin();
       // ne rien faire si ligne commentaire
       if (string(i,i+1) == "*") 
          continue;
       // regarder dans le vecteur elements si on trouve l'element
       // qui est sur les 6 premiers caracteres de la ligne
         for (imap = counters.begin(); imap !=counters.end();++imap)
         {
//            cout << imap->first << string(i,i+6) << " " << atoi((string(i,i+6)).c_str()) << endl;
            resultIter = ((imap->second).find( atoi((string(i,i+6)).c_str())));
            if (resultIter != (imap->second).end()) 
               cout << imap->first << " --> " << line << endl; 

         }
    }
    // fermer le fstream
    In.close();

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
                       std::string& burpin, bool &debug)
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

