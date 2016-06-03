#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn is a ctypes import of librmnshared.so
 
The librmn.burp_proto python module includes ctypes prototypes for many
librmn burp C functions

 Warning:
    Please use with caution.
    The functions in this module are actual C funtions and
    must thus be called as such with appropriate argument typing and
    dereferencing.
    It is highly advised in a python program to prefer the use of the
    python wrapper found in
    * rpnpy.librmn.burp
    * rpnpy.librmn.const

 See Also:
    rpnpy.librmn.burp
    rpnpy.librmn.proto
    rpnpy.librmn.base
    rpnpy.librmn.fstd98
    rpnpy.librmn.interp
    rpnpy.librmn.grids
    rpnpy.librmn.const

 === EXTERNAL FUNCTIONS in c_burp ===

***S/P MRFOPC - INITIALISER UNE OPTION DE TYPE CARACTERE
*     FONCTION SERVANT A INITIALISER UNE OPTION DE TYPE CARACTERE
*     LA VALEUR DE L'OPTION EST CONSERVEE DANS LE COMMON XDFTLR
*     OPTNOM   ENTREE  NOM DE L'OPTION A INITIALISER
*              MSGLVL
*     OPVALC     "     VALEUR A DONNER A L'OPTION
*              TRIVIAL, INFORMATIF, WARNING, ERROR, FATAL, SYSTEM
int c_mrfopc(optnom,opvalc)
char optnom[],opvalc[];

***S/P MRFOPR - INITIALISER UNE OPTION DE TYPE REEL
*     FONCTION SERVANT A INITIALISER UNE OPTION DE TYPE REEL
*     LA VALEUR DE L'OPTION EST CONSERVEE DANS LE COMMON BURPUSR
*     OPTNOM   ENTREE  NOM DE L'OPTION A INITIALISER
*              MISSING
*     OPVALR     "     VALEUR A DONNER A L'OPTION

***S/P MRFOPN - OUVRIR UN FICHIER BURP
*     INITIALISER LES DESCRIPTEURS DE CLEFS ET CREER UN FICHIER
*     BURP OU OUVRIR UN FICHIER BURP DEJA EXISTANT.
*     MRFOPN RETOURNE LE NOMBRE D'ENREGISTREMENTS ACTIFS
*     CONTENUS DANS LE FICHIER.
*     IUN     ENTREE  NUMERO DU FICHIER A OUVRIR
*     MODE      "     MODE D'OUVERTURE (READ,CREATE,APPEND)
int c_mrfopn(iun,mode)
int iun;
char mode[];

***S/P MRFCLS - FERMER UN FICHIER BURP
*     FERMER UN FICHIER BURP
*     IUN    ENTREE     NUMERO D'UNITE ASSOCIE AU FICHIER
int c_mrfcls(iun)
int iun;

***S/P MRFNBR -  OBTENIR LE NOMBRE D'ENREGISTREMENTS DANS UN FICHIER
*     FONCTION RETOURNANT LE NOMBRE D'ENREGISTREMENTS ACTIFS CONTENUS 
*     DANS UN FICHIER RAPPORT
*     IUN      ENTREE  NUMERO DU FICHIER A OUVRIR
int c_mrfnbr(iun)
int iun;

*     RETOURNER LA LONGUEUR DE L'ENREGISTREMENT LE PLUS LONG
*     CONTENU DANS LE FICHIER IUN
*     IUN      ENTREE         NUMERO D'UNITE DU FICHIER
int c_mrfmxl(iun)
int iun;

***S/P MRFLOC - TROUVER UN RAPPORT DANS UN FICHIER BURP
*     TROUVER LE POINTEUR DU RAPPORT DECRIT PAR LES PARAMETRES STNID,
*     IDTYP,LAT,LON,DATE ET LE CONTENU DU TABLEAU SUP.  LA RECHERCHE SE
*     FAIT A PARTIR DE L'ENREGISTREMENT DONT LE POINTEUR EST HANDLE.
*     SI UN ELEMENT DE STNID = '*', CET ELEMENT SERA IGNORE POUR LA
*     RECHERCHE.  UNE VALEUR DE -1 POUR LES AUTRES ARGUMENTS A LE
*     MEME EFFET.  SI LA VALEUR DE HANDLE EST ZERO, ON AFFECTUE LA
*     RECHERCHE A PARTIR DU DEBUT DU FICHIER.
*     IUN     ENTREE   NUMERO D'UNITE DU FICHIER
*     HANDLE    "      POINTEUR A L'ENREGISTREMENT D'OU PART LA RECHERCHE
*     STNID     "      IDENTIFICATEUR DE LA STATION
*     IDTYP     "      TYPE DE RAPPORT
*     LAT       "      LATITUDE DE LA STATION
*     LON       "      LONGITUDE DE LA STATION
*     DATE      "      DATE DE VALIDITE DU RAPPORT
*     TEMPS     "      HEURE DE L'OBSERVATION
*     SUP       "      TABLEAU DE CLEFS DE RECHERCHES SUPPLEMENTAIRES
*     NSUP      "      NOMBRE DE CLEFS SUPPLEMENTAIRES
int c_mrfloc(iun,handle,stnid,idtyp,lat,lon,date,temps,sup,nsup)
int iun,handle,idtyp,lat,lon,date,temps,sup[],nsup;
char stnid[];

/***************************************************************************** 
 *                             C _ M R F G E T                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Read the report referenced by handle from the file.                     *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN   handle logical pointer to the record                                *
 *  OUT  buffer vector that contains the report                              * 
 *                                                                           * 
 *****************************************************************************/
int c_mrfget(int handle, void *buffer)


/***************************************************************************** 
 *                             C _ M R B H D R                               *
 *                                                                           * 
 *Object                                                                     * 
 *  Return the description parameters of the data block of order bkno.       *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN    buf    vector to contain the report                                * 
 *  IN    bkno   block number                                                *
 *  OUT   nele   number of elements                                          *
 *  OUT   nval   number of values per element                                *
 *  OUT   nt     number of NELE*NVAL values                                  *
 *  OUT   bfam   block family type (12 bits)                                 *
 *  OUT   bdesc  block descriptor (set to zero)                              *
 *  OUT   btyp   block type                                                  *
 *  OUT   nbit   number of bits kept per value                               *
 *  OUT   bit0   first bit of array values                                   *
 *  OUT   datyp  data compaction type                                        *
 *                                                                           *
 *****************************************************************************/
int c_mrbhdr(word *buf, int *temps, int *flgs, char *stnid, int *idtyp,
             int *lati, int *lon, int *dx, int *dy, int *elev,
             int *drcv, int *date, int *oars, int *run, int *nblk,
             word *sup, int nsup, word *xaux, int nxaux)


***S/P MRBPRML - EXTRAIRE LES PARAMETRES DESCRIPTEURS DE TOUS LES BLOCS
*     FONCTION SERVANT A RETOURNER DANS LE TABLEAU TBLPRM
*     LES PARAMETRES DESCRIPTEURS DES INBLOCS BLOCS A PARTIR 
*     DU BLOC SUIVANT LE BLOC NUMERO BKNO.
*     BUF        ENTREE    VECTEUR CONTENANT LE RAPPORT
*     INBKNO        "      NUMERO D'ORDRE DU PREMIER BLOC
*     NPRM          "      NOMBRE DE PARAMETRES A EXTRAIRE (DIM 1 DE TBLPRM)
*     INBLOCS       "      NOMBRE DE BLOCS DONT ON VEUT LES PARAMETRES
*     TBLPRM     SORTIE    TABLEAU CONTENANT LES PARAMETRES DES INBLOCS
*
*     STRUCTURE DE TBLPRM(NPRM,INBLOCS)
*     TBLPRM(1,I) - NUMERO DU BLOC I
*     TBLPRM(2,I) - NOMBRE D'ELEMENTS DANS LE BLOC I  (NELE)
*     TBLPRM(3,I) - NOMBRE DE VALEURS  PAR ELEMENT    (NVAL)
*     TBLPRM(4,I) - NOMBRE DE PAS DE TEMPS            (NT)
*     TBLPRM(5,I) - FAMILLE DU BLOC                   (BFAM) (12 bits)
*     TBLPRM(6,I) - DESCRIPTEUR DE BLOC               (BDESC) (mis a zero)
*     TBLPRM(7,I) - TYPE DU BLOC                      (BTYP)
*     TBLPRM(8,I) - NOMBRE DE BITS PAR ELEMENT        (NBIT)
*     TBLPRM(9,I) - NUMERO DU PREMIER BIT             (BIT0)
*     TBLPRM(10,I)- TYPE DE DONNEES POUR COMPACTION   (DATYP)
int c_mrbprml(buf,bkno,tblprm,nprm,inblocs)
int buf[],tblprm[],bkno,nprm,inblocs;

/***************************************************************************** 
 *                             C _ M R B P R M                               *
 *                                                                           * 
 *Object                                                                     * 
 *  Return the description parameters of the data block of order bkno.       *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN    buf    vector to contain the report                                * 
 *  IN    bkno   block number                                                *
 *  OUT   nele   number of elements                                          *
 *  OUT   nval   number of values per element                                *
 *  OUT   nt     number of NELE*NVAL values                                  *
 *  OUT   bfam   block family type (12 bits)                                 *
 *  OUT   bdesc  block descriptor (set to zero)                              *
 *  OUT   btyp   block type                                                  *
 *  OUT   nbit   number of bits kept per value                               *
 *  OUT   bit0   first bit of array values                                   *
 *  OUT   datyp  data compaction type                                        *
 *                                                                           *
 *****************************************************************************/
int c_mrbprm(word *buf,int  bkno, int *nele, int *nval, int *nt, int *bfam,
             int *bdesc, int *btyp, int *nbit, int *bit0, int *datyp)

/***************************************************************************** 
 *                             C _ M R B X T R                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Extract list of element and values from buffer.                         *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *    IN   buffer vector that contains the report                            * 
 *    IN   bkno   number of blocks in buf                                    *
 *   OUT   lstele list of nele meteorogical elements                         *
 *   OUT   tblval array of values to write (nele*nval*nt)                    *
 *                                                                           * 
 *****************************************************************************/
int c_mrbxtr(void *buffer, int bkno, word *lstele, word *tblval)


***S/P MRBDCL - DECODER LES ELEMENTS D'UNE LISTE
*     RETOURNE UNE LISTE D'ELEMENTS DECODES
*
*     POUR UN ELEMENT, ON RETOURNE SA VALEUR SOUS FORMAT DECIMAL  ABBCCC,
*                                                       (A,B,C DE 0 A 9)
*     OU A    PROVIENT DES BITS 14 ET 15 DE L'ELEMENT
*        BB       "    DES BITS 8 A 13 DE L'ELEMENT
*        CCC      "    DES BITS 0 A 7  DE L'ELEMENT
*
*ARGUMENT
*     CLISTE   ENTREE    LISTE DES ELEMENTS A CODER
*     LISTE    SORTIE    LISTE DES ELEMENTS CODES
*     NELE     ENTREE    NOMBRE D'ELEMENTS A CODER
int c_mrbdcl(cliste,liste,nele)
int liste[], cliste[], nele;


***S/PMRBCVT - FAIRE UNE CONVERSION D'UNITES
*     SOUS-PROGRAMME SERVANT A LA CONVERSION D'UNITES DE REEL A ENTIER
*     POSITIF OU L'INVERSE, SELON LA VALEUR DE MODE.  LA CONVERSION
*     SE FAIT EN CONSULTANT UN TABLEAU DUQUEL ON EXTRAIT UN FACTEUR
*     D'ECHELLE ET UNE VALEUR DE REFERENCE QUI SONT APPLIQUES A LA 
*     VARIABLE A CONVERTIR.
*     LE TABLEAU DE REFERENCE EST INITIALISE PAR LE SOUS-PROGRAMME
*     MRBSCT QUI EST APPELE LORS DE LA PREMIERE EXECUTION DE MRBCVT
*
*ARGUMENTS
*     LISTE    ENTREE    LISTE DES ELEMENTS A CONVERTIR
*     NELE       "       NOMBRE D'ELEMENTS (LONGUEUR DE LISTE)
*     NVAL       "       NOMBRE DE VALEURS PAR ELEMENT
*     NT         "       NOMBRE D'ENSEMBLES NELE*NVAL
*     MODE       "       MODE DE CONVERSION
*                        MODE = 0 DE TBLVAL(CODE BUFR) A RVAL
*                        MODE = 1 DE RVAL A TBLVAL(CODE BUFR)
*     TBLVAL   ENT/SRT   TABLEAU DE VALEURS EN CODE BUFR
*     RVAL               TABLEAU DE VALEURS REELLES
int c_mrbcvt(liste,tblval,rval,nele,nval,nt,mode)
int liste[], tblval[], nele, nval, nt, mode;
float rval[];

---write---

***S/P MRBINI - INITIALISER L'ENTETE D'UN RAPPORT
*     INITIALISER L'ENTETE D'UN RAPPORT.  AVANT DE METTRE QUOI QUE 
*     CE SOIT DANS UN RAPPORT, ON INITIALISE LES DIFFERENTES CLEFS
*     PRIMAIRES ET AUXILIAIRES.
*
*ARGUMENTS
*     IUN     ENTREE   NUMERO D'UNITE ASSOCIE AU FICHIER
*     TYPREC    "      TYPE D'ENREGISTREMENT
*     IDELT     "      DIFF DE TEMPS ENTRE T VALIDITE ET T SYNOPTIQUE
*     FLGS      "      MARQUEURS GLOBAUX
*     STNID     "      IDENTIFICATEUR DE LA STATION
*     IDTYP     "      TYPE DE RAPPORT
*     LATI      "      LATITUDE DE LA STATION EN CENTIDEGRES
*     LONG      "      LONGITUDE DE LA STATION EN CENTIDEGRES
*     DX        "      DIMENSION X D'UNE BOITE
*     DY        "      DIMENSION Y D'UNE BOITE
*     ELEV      "      ALTITUDE DE LA STATION EN METRES
*     IDRCV     "      DELAI DE RECEPTION
*     DATEin    "      DATE SYNOPTIQUE DE VALIDITE (AAMMJJHH)
*     OARS      "      RESERVE POUR ANALYSE OBJECTIVE
*     RUN       "      IDENTIFICATEUR DE LA PASSE OPERATIONNELLE
*     SUP       "      CLEFS PRIMAIRES SUPPLEMENTAIRES 
*                      (AUCUNE POUR LA VERSION 1990)
*     NSUP      "      NOMBRE DE CLEFS PRIMAIRES SUPPLEMENTAIRES 
*                      (DOIT ETRE ZERO POUR LA VERSION 1990)
*     XAUX      "      CLEFS AUXILIAIRES SUPPLEMENTAIRES (=0 VRSN 1990)
*     NXAUX     "      NOMBRE DE CLEFS AUXILIAIRES SUPPLEMENTAIRES(=0)
*     BUF       "      VECTEUR QUI CONTIENDRA LES ENREGISTREMENTS
int c_mrbini(iun,buf,temps,flgs,stnid,idtp,lati,longi,dx,dy,elev,drcv,date,
         oars,runn,sup,nsup,xaux,nxaux)


***S/P - MRBCOL - CODER LES ELEMENTS D'UNE LISTE
*     SOUS-PROGRAMME RETOURNANT UNE LISTE D'ELEMENTS CODES DE TELLE SORTE
*     QUE CHAQUE ELEMENT OCCUPE SEIZE BITS.
*     POUR UN ELEMENT AYANT LE FORMAT DECIMAL  ABBCCC, (A,B,C DE 0 A 9)
*     ON RETOURNE UN ENTIER CONTENANT A SUR DEUX BITS, BB SUR SIX BITS
*     ET CCC SUR HUIT BITS
*
*ARGUMENTS
*     LISTE   ENTREE  LISTE DES ELEMENTS A CODER
*     CLISTE  SORTIE  LISTE DES ELEMENTS CODES
*     NELE    ENTREE  NOMBRE D'ELEMENTS A CODER
*
      INTEGER I, IIBIT, VIBIT, VIIIBIT, IELEM
int c_mrbcol(liste,cliste,nele)
int liste[], cliste[], nele;


/***************************************************************************** 
 *                             C _ M R B A D D                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Add a data block at the end of the report.                              *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN/OUT buffer vector to contain the report                               * 
 *    OUT  bkno   number of blocks in buf                                    *
 *    IN   nele   number of meteorogical elements in block                   *
 *    IN   nval   number of data per elements                                *
 *    IN   nt     number of group of nele*nval values in block               *
 *    IN   bfam   block family (12 bits, bdesc no more used)                 *
 *    IN   bdesc  kept for backward compatibility                            *
 *    IN   btyp   block type                                                 *
 *    IN   nbit   number of bit to keep per values                           *
 *    OUT  bit0   position of first bit of the report                        *
 *    IN   datyp  data type for packing                                      *
 *    IN   lstele list of nele meteorogical elements                         *
 *    IN   tblval array of values to write (nele*nval*nt)                    *
 *                                                                           * 
 *****************************************************************************/
int c_mrbadd(void *buffer, int *bkno, int nele, int nval, int nt, int bfam
     int bdesc, int btyp, int nbit, int *bit0, int datyp,
     word *lstele, word *tblval)


/***************************************************************************** 
 *                             C _ M R F P U T                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Write a report to the file. If handle is not 0, record referenced by    *
 *   by handle is written at end of file. If handle is 0, a new record is    *
 *   written.  If hanlde is > 0, it will be forced to be negative to write   *
 *   at end of file.                                                         *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN   iun    unit number associated to the file                           *
 *  IN   handle logical pointer to the record                                *
 *  IN   buffer vector that contains the report                              * 
 *                                                                           * 
 *****************************************************************************/
int c_mrfput(int iun, int handle, void *buffer)

/***************************************************************************** 
 *                             C _ M R B D E L                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Delete a particular block of the report.                                *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN/OUT buffer   vector to contain the report                             * 
 *    IN   number   block number to be deleted                               *
 *                                                                           * 
 *****************************************************************************/
int c_mrbdel(void *buffer, int number)

/***************************************************************************** 
 *                             C _ M R B L E N                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Return the number of bits used in buf and the number of bits left.      *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *    IN   buffer   vector that contains the report                          * 
 *   OUT   lbits    number of bits used                                      *
 *   OUT   left     number of bits left                                      *
 *                                                                           * 
 *****************************************************************************/
int c_mrblen(void *buffer, int *lbits, int *left)

/***************************************************************************** 
 *                             C _ M R B L O C                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Search for a specific block in the buffer. Search starts at block       *
 *   blkno. If blkno = 0 search starts from the beginning.                   *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN   buffer vector to contain the report                                 * 
 *  IN   bfam   block family (12 bits, bdesc no more used)                   *
 *  IN   bdesc  kept for backward compatibility                              *
 *  IN   btyp   block type                                                   *
 *  in   bkno   number of blocks in buf                                      *
 *                                                                           * 
 *****************************************************************************/
int c_mrbloc(void *buffer, int bfam, int bdesc, int btyp, int blkno)

/***************************************************************************** 
 *                             C _ M R B R E P                               *
 *                                                                           * 
 *Object                                                                     * 
 *   Replace a data block by an other one with the same variables and        *
 *   dimensions.                                                             *
 *                                                                           *
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN/OUT buffer vector that contains the report                            * 
 *    IN   bkno   block number to be replaced                                *
 *    IN   tblval array of values to write (nele*nval*nt)                    *
 *                                                                           * 
 *****************************************************************************/
int c_mrbrep(void *buffer, int blkno, word *tblval)

/***************************************************************************** 
 *                            C _ M R F B F L                                *
 *                                                                           * 
 *Object                                                                     * 
 *   Return the length of the longer report in the file                      *
 *                                                                           * 
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN  iun     unit number associated to the file                           * 
 *                                                                           * 
 *****************************************************************************/
int c_mrfbfl(int iun)

/***************************************************************************** 
 *                           C _ M R F R W D                                 *
 *                                                                           * 
 *Object                                                                     * 
 *   Rewinds a BURP sequential file.                                         *
 *                                                                           * 
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN  iun     unit number associated to the file                           * 
 *                                                                           * 
 *****************************************************************************/
int c_mrfrwd(int iun)

/***************************************************************************** 
 *                            C _ M R F A P P                                *
 *                                                                           * 
 *Object                                                                     * 
 *   Position at the end of a sequential file for an append.                 *
 *                                                                           * 
 *Arguments                                                                  * 
 *                                                                           * 
 *  IN  iun     unit number associated to the file                           * 
 *                                                                           * 
 *****************************************************************************/
int c_mrfapp(int iun)

"""
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc

from . import librmn
from rpnpy.librmn import proto as _rp

librmn.c_mrfopc.argtypes = (_ct.c_char_p, _ct.c_char_p)
librmn.c_mrfopc.restype  = _ct.c_int
c_mrfopc = librmn.c_mrfopc

librmn.c_mrfopn.argtypes = (_ct.c_int, _ct.c_char_p)
librmn.c_mrfopn.restype  = _ct.c_int
c_mrfopn = librmn.c_mrfopn

librmn.c_mrfcls.argtypes = (_ct.c_int, )
librmn.c_mrfcls.restype  = _ct.c_int
c_mrfcls = librmn.c_mrfcls

librmn.c_mrfnbr.argtypes = (_ct.c_int, )
librmn.c_mrfnbr.restype  = _ct.c_int
c_mrfnbr = librmn.c_mrfnbr

librmn.c_mrfmxl.argtypes = (_ct.c_int, )
librmn.c_mrfmxl.restype  = _ct.c_int
c_mrfmxl = librmn.c_mrfmxl

librmn.c_mrfbfl.argtypes = (_ct.c_int, )
librmn.c_mrfbfl.restype  = _ct.c_int
c_mrfbfl = librmn.c_mrfbfl

librmn.c_mrfrwd.argtypes = (_ct.c_int, )
librmn.c_mrfrwd.restype  = _ct.c_int
c_mrfrwd = librmn.c_mrfrwd

librmn.c_mrfapp.argtypes = (_ct.c_int, )
librmn.c_mrfapp.restype  = _ct.c_int
c_mrfapp = librmn.c_mrfapp

librmn.c_mrfloc.argtypes = (_ct.c_int, _ct.c_int, _ct.c_char_p,
                            _ct.c_int, _ct.c_int, _ct.c_int,
                            _ct.c_int, _ct.c_int,
                            _npc.ndpointer(dtype=_np.int32), _ct.c_int)
librmn.c_mrfloc.restype  = _ct.c_int
c_mrfloc = librmn.c_mrfloc

librmn.c_mrfget.argtypes = (_ct.c_int, _npc.ndpointer(dtype=_np.int32))
librmn.c_mrfget.restype  = _ct.c_int
c_mrfget = librmn.c_mrfget

librmn.c_mrfput.argtypes = (_ct.c_int, _ct.c_int,
                            _npc.ndpointer(dtype=_np.int32))
librmn.c_mrfput.restype  = _ct.c_int
c_mrfput = librmn.c_mrfput


## int c_mrbhdr(word *buf, int *temps, int *flgs, char *stnid, int *idtyp,
##              int *lati, int *lon, int *dx, int *dy, int *elev,
##              int *drcv, int *date, int *oars, int *run, int *nblk,
##              word *sup, int nsup, word *xaux, int nxaux)
librmn.c_mrbhdr.argtypes = (
    ## word *buf, int *temps, int *flgs, char *stnid, int *idtyp,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_char_p, _ct.POINTER(_ct.c_int),
    ## int *lati, int *lon, int *dx, int *dy, int *elev,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), 
    ## int *drcv, int *date, int *oars, int *run, int *nblk,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int),     
    ## word *sup, int nsup, word *xaux, int nxaux
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int
    )
librmn.c_mrbhdr.restype  = _ct.c_int
c_mrbhdr = librmn.c_mrbhdr

## int c_mrbprm(word *buf,int  bkno, int *nele, int *nval, int *nt, int *bfam,
##              int *bdesc, int *btyp, int *nbit, int *bit0, int *datyp)
librmn.c_mrbprm.argtypes = (
    ## word *buf,int  bkno,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int,
    ## int *nele, int *nval, int *nt, int *bfam,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    ## int *bdesc, int *btyp, int *nbit, int *bit0, int *datyp
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int)
    )
librmn.c_mrbprm.restype  = _ct.c_int
c_mrbprm = librmn.c_mrbprm

## int c_mrbxtr(void *buffer, int bkno, word *lstele, word *tblval)
librmn.c_mrbxtr.argtypes = (
    ## void *buffer, int bkno,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int,
    ## word *lstele, word *tblval
    _npc.ndpointer(dtype=_np.int32),#float32),
    _npc.ndpointer(dtype=_np.int32)#float32)
    )
librmn.c_mrbxtr.restype  = _ct.c_int
c_mrbxtr = librmn.c_mrbxtr

## int c_mrbdcl(cliste,liste,nele)
## int liste[], cliste[], nele;
librmn.c_mrbdcl.argtypes = (
    ## int liste[], cliste[], nele;
    _npc.ndpointer(dtype=_np.int32), _npc.ndpointer(dtype=_np.int32), _ct.c_int,
    )
librmn.c_mrbdcl.restype  = _ct.c_int
c_mrbdcl = librmn.c_mrbdcl


## int c_mrbcvt(liste,tblval,rval,nele,nval,nt,mode)
## int liste[], tblval[], nele, nval, nt, mode;
## float rval[];
librmn.c_mrbcvt.argtypes = (
    ## int liste[], tblval[]
    _npc.ndpointer(dtype=_np.int32), _npc.ndpointer(dtype=_np.int32),
    ## float rval[];
    _npc.ndpointer(dtype=_np.float32), 
    ## int nele, nval, nt, mode;
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int
    )
librmn.c_mrbcvt.restype  = _ct.c_int
c_mrbcvt = librmn.c_mrbcvt

# =========================================================================

if __name__ == "__main__":
    print(str(_rp.c_fst_version()))
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
