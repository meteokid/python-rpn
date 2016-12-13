#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Author: Michael Sitwell <michael.sitwell@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn is a ctypes import of librmnshared.so

The librmn.proto_burp python module includes ctypes prototypes for many
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
    rpnpy.librmn.burpfile
    rpnpy.librmn.proto
    rpnpy.librmn.base
    rpnpy.librmn.fstd98
    rpnpy.librmn.interp
    rpnpy.librmn.grids
    rpnpy.librmn.const

 === EXTERNAL FUNCTIONS ===

    c_mrfgoc(option, value)
        Get a character option
        Proto:
            int c_mrfgoc(char *option, char *value)
        Args:
            option (str): (I) option name to be set
            value  (str): (O) option value
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfgor(option, value)
        Get a real value option
        Proto:
            int c_mrfgor(char *option, float value)
        Args:
            option (str)  : (I) option name to be set
            value  (float): (O) option value
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfopc(option, value)
        Set a character option
        Proto:
            int c_mrfopc(char *option, char *value)
        Args:
            option (str): (I) option name to be set
            value  (str): (I) option value
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfopr(option, value)
        Set a real value option
        Proto:
            int c_mrfopr(char *option, float value)
        Args:
            option (str)  : (I) option name to be set
            value  (float): (I) option value
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfopn(iun, mode)
        Open a BURP file
        Proto:
            int c_mrfopn(int iun, char *mode)
        Args:
            iun  (int): (I) file unit number
            mode (str): (I) file open mode, one of: ['READ', 'CREATE', 'APPEND']
        Returns:
            int, number of active records in the file

    c_mrfcls(iun)
        Close a previously opened BURP file
        Proto:
            int c_mrfcls(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfvoi(iun)
        Print the content of a previously opened BURP file
        Proto:
            int c_mrfvoi(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfnbr(iun)
        Return then number of active records in the file before opening it
        Proto:
            int c_mrfnbr(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, number of active records in the file

    c_mrfmxl(iun)
        Return the lenght of the longest record in the file
        Proto:
            int c_mrfmxl(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, lenght of the longest record in the file (units?)

    c_mrfbfl(iun)
        Return the length of the longest report in the file
        Proto:
            int c_mrfbfl(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, length of the longest report in the file (units?)

    c_mrfrwd(iun)
        Rewinds a BURP sequential file.
        Proto:
            int c_mrfrwd(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfapp(iun)
        Position at the end of a sequential file for an append.
        Proto:
            int c_mrfapp(int iun)
        Args:
            iun  (int): (I) file unit number
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfloc(iun, handle, stnid, idtyp, lat, lon, date, temps, sup, nsup)
        Search a report matching provided selection criterions
        Any agr with value = -1 or stnid = '*' acts as a wildcard
        Proto:
            int c_mrfloc(iun,handle,stnid,idtyp,lat,lon,date,temps,sup,nsup)
            int iun,handle,idtyp,lat,lon,date,temps,sup[],nsup;
            char stnid[];
        Args:
            iun    (int): (I) File unit number
            handle (int): (I) Start position for the search,
                              0 (zero) for beginning of file
            stnid  (str): (I) Station ID
            idtyp  (int): (I) Report Type
            lat    (int): (I) Station latitude (1/100 of degrees)
            lon    (int): (I) Station longitude (1/100 of degrees)
            date   (int): (I) Report valid date
            temps  (int): (I) Observation time/hour
            sup    (array): (I) Additional search keys (array of int)
            nsup   (int)  : (I) number of items in sup
        Returns:
            int, report handle

    c_mrfget(handle, buffer)
        Read the report referenced by handle from the file.
        Proto:
            int c_mrfget(int handle, void *buffer)
        Args:
            handle (int)   : (I) Report handle
            buffer (array) : (O) Report data
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfput(iun, handle, buffer)
        Write a report to the file.
        If handle != 0, record referenced by handle is written at end of file.
        If handle == 0, a new record is written.
        If hanlde >  0, it will be forced to be negative to write at end of file.
        Proto:
            int c_mrfput(int iun, int handle, void *buffer)
        Args:
            iun    (int)   : (I) File unit number
            handle (int)   : (I) Report handle
            buffer (array) : (I) Report data
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbhdr(buf, temps, flgs, stnid, idtyp, lati, lon, dx, dy, elev,
             drcv, date, oars, run, nblk, sup, nsup, xaux, nxaux)
        Return the description parameters of the data block of order bkno
        Proto:
            int c_mrbhdr(word *buf, int *temps, int *flgs, char *stnid,
                int *idtyp, int *lati, int *lon, int *dx, int *dy, int *elev,
                int *drcv, int *date, int *oars, int *run, int *nblk,
                word *sup, int nsup, word *xaux, int nxaux)
        Args:
            buf    (array) : (I) vector containing the report data
            temps  (int)   : (O) Observation time/hour
            flgs   (int)   : (O) Global flags
            stnid  (str)   : (O) Station ID
            idtyp  (int)   : (O) Report Type
            lati   (int)   : (O) Station latitude (1/100 of degrees)
            lon    (int)   : (O) Station longitude (1/100 of degrees)
            dx     (int)   : (O) dimension x d'une boite
            dy     (int)   : (O) dimension y d'une boite
            elev   (int)   : (O) Station altitude [metres]
            drcv   (int)   : (O) delai de reception
            date   (int)   : (O) Report valid date
            oars   (int)   : (O) reserve pour analyse objective
            run    (int)   : (O) identificateur de la passe operationnelle
            nblk   (int)   : (O) number of blocks
            sup    (array) : (O) clefs primaires supplementaires
            nsup   (int)   : (I) number of sup
            xaux   (array) : (O) clefs auxiliaires supplementaires
            nxaux  (int)   : (I) number of xaux
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbprm(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp)
        Return the description parameters of the data block of order bkno.
        Proto:
            int c_mrbprm(word *buf,int  bkno, int *nele, int *nval, int *nt,
                         int *bfam, int *bdesc, int *btyp, int *nbit,
                         int *bit0, int *datyp)
        Args:
            buf    (array) : (I) vector containing the report data
            bkno   (int)   : (I) block number
            nele   (int)   : (O) number of elements
            nval   (int)   : (O) number of values per element
            nt     (int)   : (O) number of NELE*NVAL values
            bfam   (int)   : (O) block family type (12 bits
            bdesc  (int)   : (O) block descriptor (set to zero)
            btyp   (int)   : (O) block type
            nbit   (int)   : (O) number of bits kept per value
            bit0   (int)   : (O) first bit of array values
            datyp  (int)   : (O) data compaction type
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbtyp(bknat, bktyp, bkstp, btyp)
        Convert btyp to/from bknat, bktyp, bkstp
        If btyp=0, Convert bknat, bktyp, bkstp to btyp
        If btyp>0, Convert btyp to bknat, bktyp, bkstp
        Proto:
            int c_mrbtyp(int *bknat, int *bktyp, int *bkstp, int btyp)
        Args:
            bknat  (int)   : (I/O) block type
            bktyp  (int)   : (I/O) block type, kind component
            bkstp  (int)   : (I/O) block type, Data-type component
            btyp   (int)   : (I)   block type, Sub data-type component
        Returns:
            int, encoded btyp (if input btype=0), zero otherwise

    c_mrbxtr(buf, bkno, lstele, tblval)
        Extract list of element and values from buffer.
        Proto:
            int c_mrbxtr(void *buf, int bkno, word *lstele, word *tblval)
        Args:
            buf    (array) : (I) vector containing the report data
            bkno   (int)   : (I) block number
            lstele (array) : (O) list of nele meteorogical elements (array of int) #TODO: CMC or BUFR codes?
            tblval (array) : (O) array of values to write (nele*nval*nt)
                                 (array of int or float)
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbdcl(cliste, liste, nele)
        Decode List of Elements from CMC to BUFR format
        pour un element, on retourne sa valeur sous format decimal
        abbccc, (a,b,c de 0 a 9)
        ou
        a    provient des bits 14 et 15 de l'element
        bb       "    des bits 8 a 13 de l'element
        ccc      "    des bits 0 a 7  de l'element
        Proto:
            int c_mrbdcl(int *cliste, int *liste, int nele)
        Args:
            cliste (array) : (I) Elements to be decoded, CMC format (array of int)
            liste  (array) : (O) Decoded elements, BUFR format (array of int)
            nele   (int)   : (I) Number of elemets to decode
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbdcv(elem)
        Convert Element from CMC to BUFR format
        Proto:
            int c_mrbdcv(int elem)
        Args:
            elem (int) : (I) Element to be converted, CMC format (int)
        Returns:
            int, converted element, BUFR format

    c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
        Perform a unit conversion to/from BUFR code to/from real values
        Proto:
            int c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
            int liste[], tblval[], nele, nval, nt, mode;
            float rval[];
        Args:
            liste  (array) : (I)   CMC codes of Elements to convert (array of int)
            tblval (array) : (I/O) Coded values (array of int or float)
            rval   (array) : (I/O) Real values (array of float)
            nele   (int)   : (I)   Number of elemets to convert
            nval   (int)   : (I)   Number of values per elemet
            nt     (int)   : (I)   Number of ensembles nele*nval
            mode   (int)   : (I)   Conversion mode,
                                   0 = RVAL to TBLVAL (BUFR format)
                                   1 = TBLVAL (BUFR format) to RVAL
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbini(iun, buf, temps, flgs, stnid, idtp, lati, longi, dx, dy,
             elev, drcv, date, oars, runn, sup, nsup, xaux, nxaux)
        Report header initialisation
        Proto:
            int c_mrbini(int iun, int *buf, int temps, int flgs, char *stnid,
                         int idtp, int lati, int longi, int dx, int dy,
                         int elev, int drcv, int date, int oars, int runn,
                         int *sup, int nsup, int *xaux, int nxaux)
        Args:
            iun    (int)   : (I) file unit number
            buf    (array) : (I) Report data
            temps  (int)   : (I) Observation time/hour
            flgs   (int)   : (I) Global flags
            stnid  (str)   : (I) Station ID
            idtyp  (int)   : (I) Report Type
            lati   (int)   : (I) Station latitude (1/100 of degrees)
            long   (int)   : (I) Station longitude (1/100 of degrees)
            dx     (int)   : (I) dimension x d'une boite
            dy     (int)   : (I) dimension y d'une boite
            elev   (int)   : (I) Station altitude [metres]
            drcv   (int)   : (I) delai de reception
            date   (int)   : (I) synoptique date of validity (aammjjhh)
            oars   (int)   : (I) reserve pour analyse objective
            runn   (int)   : (I) identificateur de la passe operationnelle
            sup    (array) : (I) clefs primaires supplementaires
                             (aucune pour la version 1990)
            nsup   (int)   : (I) number of sup
                             (must be = 0 for version 1990)
            xaux   (array) : (I) clefs auxiliaires supplementaires (=0 vrsn 1990)
            nxaux  (int)   : (I) number of xaux (=0)
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbcol(liste, cliste, nele)
        Encode elemets of a list
        Sous-programme retournant une liste d'elements codes de telle sorte
        que chaque element occupe seize bits.
        pour un element ayant le format decimal  abbccc, (a,b,c de 0 a 9)
        on retourne un entier contenant a sur deux bits, bb sur six bits
        et ccc sur huit bits
        Proto:
            int c_mrbcol(int *liste, int *cliste, int nele)
        Args:
            liste  (array) : (I) Elements to be encoded (array of int)
            cliste (array) : (O) Encoded Elements (array of int)
            nele   (int)   : (I) Number of elemets to encode
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbcov(delem)
        Convert Element from BUFR to CMC format
        Proto:
            int c_mrbcov(int delem)
        Args:
            delem (int) : (I) Element to be converted, BUFR format (int)
        Returns:
            int, converted element, CMC format

    c_mrbadd(buffer, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0,
             datyp, lstele,tblval)
        Add a data block at the end of the report.
        Proto:
            int c_mrbadd(void *buffer, int *bkno, int nele, int nval, int nt,
                         int bfam, int bdesc, int btyp, int nbit, int *bit0,
                         int datyp, word *lstele, word *tblval)
        Args:
            buffer (array) : (I/O) vector to contain the report
            bkno   (int)   : (O)   number of blocks in buf
            nele   (int)   : (I)   number of meteorogical elements in block
            nval   (int)   : (I)   number of data per elements
            nt     (int)   : (I)   number of group of nelenval values in block
            bfam   (int)   : (I)   block family (12 bits, bdesc no more used)
            bdesc  (int)   : (I)   kept for backward compatibility
            btyp   (int)   : (I)   block type
            nbit   (int)   : (I)   number of bit to keep per values
            bit0   (int)   : (O)   position of first bit of the report
            datyp  (int)   : (I)   data type for packing
            lstele (array) : (I)   list of nele meteorogical elements
            tblval (array) : (I)   array of values to write (nele*nval*nt)
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrbdel(buffer, bkno)
        Delete a particular block of the report.
        Proto:
            int c_mrbdel(void *buffer, int bkno)
        Args:
            buffer (array) : (I/O) report data
            bkno   (int)   : (O)   number of blocks in buf
        Returns:
            int, zero if successful, non-zero otherwise

    c_mrfdel(handle)
        Delete a particular report from a burp file.
        Proto:
            int c_mrfdel(int handle)
        Args:
            handle (int) : (I) Report handle
        Returns:
            int, zero if successful, non-zero otherwise

"""

##TODO: MRBPRML
## ***S/P MRBPRML - EXTRAIRE LES PARAMETRES DESCRIPTEURS DE TOUS LES BLOCS
## *     FONCTION SERVANT A RETOURNER DANS LE TABLEAU TBLPRM
## *     LES PARAMETRES DESCRIPTEURS DES INBLOCS BLOCS A PARTIR
## *     DU BLOC SUIVANT LE BLOC NUMERO BKNO.
## *     BUF        ENTREE    VECTEUR CONTENANT LE RAPPORT
## *     INBKNO        "      NUMERO D'ORDRE DU PREMIER BLOC
## *     NPRM          "      NOMBRE DE PARAMETRES A EXTRAIRE (DIM 1 DE TBLPRM)
## *     INBLOCS       "      NOMBRE DE BLOCS DONT ON VEUT LES PARAMETRES
## *     TBLPRM     SORTIE    TABLEAU CONTENANT LES PARAMETRES DES INBLOCS
## *
## *     STRUCTURE DE TBLPRM(NPRM,INBLOCS)
## *     TBLPRM(1,I) - NUMERO DU BLOC I
## *     TBLPRM(2,I) - NOMBRE D'ELEMENTS DANS LE BLOC I  (NELE)
## *     TBLPRM(3,I) - NOMBRE DE VALEURS  PAR ELEMENT    (NVAL)
## *     TBLPRM(4,I) - NOMBRE DE PAS DE TEMPS            (NT)
## *     TBLPRM(5,I) - FAMILLE DU BLOC                   (BFAM) (12 bits)
## *     TBLPRM(6,I) - DESCRIPTEUR DE BLOC               (BDESC) (mis a zero)
## *     TBLPRM(7,I) - TYPE DU BLOC                      (BTYP)
## *     TBLPRM(8,I) - NOMBRE DE BITS PAR ELEMENT        (NBIT)
## *     TBLPRM(9,I) - NUMERO DU PREMIER BIT             (BIT0)
## *     TBLPRM(10,I)- TYPE DE DONNEES POUR COMPACTION   (DATYP)
## int c_mrbprml(buf,bkno,tblprm,nprm,inblocs)
## int buf[],tblprm[],bkno,nprm,inblocs;


##TODO: c_mrblen
##  *   Return the number of bits used in buf and the number of bits left.
##  *    IN   buffer   vector that contains the report
##  *   OUT   lbits    number of bits used
##  *   OUT   left     number of bits left
## int c_mrblen(void *buffer, int *lbits, int *left)


##TODO: c_mrbloc
##  *   Search for a specific block in the buffer. Search starts at block
##  *   blkno. If blkno = 0 search starts from the beginning.
##  *  IN   buffer vector to contain the report
##  *  IN   bfam   block family (12 bits, bdesc no more used)
##  *  IN   bdesc  kept for backward compatibility 
##  *  IN   btyp   block type
##  *  in   bkno   number of blocks in buf
## int c_mrbloc(void *buffer, int bfam, int bdesc, int btyp, int blkno)


##TODO: c_mrblocx
## ***S/P MRBLOCX - TROUVER UN BLOC DANS UN RAPPORT
##       FUNCTION MRBLOCX( BUF,   BFAM, BDESC, BKNAT, BKTYP, BKSTP, BLKNO)
##       INTEGER  MRBLOCX, BUF(*),BFAM, BDESC, BKNAT, BKTYP, BKSTP, BLKNO
## *     FONCTION SERVANT A BATIR UNE CLEF DE RECHERCHE BTYP A
## *     PARTIR DE BKNAT, BKTYP ET BKSTP POUR L'APPEL SUBSEQUENT A MRBLOC
## *     POUR CHAQUE CLEF D'ENTREE, ON TRANSPOSE LA VALEUR DU BIT
## *     28, 29 OU 30 RESPECTIVEMENT DANS INBTYP (CES BITS NE SONT ALLUMES
## *     QUE SI LES CLEFS D'ENTREE SON MISE A -1)
## *     BUF     ENTREE  VECTEUR C ONTENANT LE RAPPORT
## *     BFAM       "    FAMILLE DU BLOC RECHERCHE
## *     BDESC      "    DESCRIPTION DU BLOC RECHERCHE
## *     BKNAT      "    PORTION NATURE DU BTYP DE BLOC RECHERCHE
## *     BKTYP      "    PORTION TYPE DU BTYP DE BLOC RECHERCHE
## *     BKSTP      "    PORTION SOUS-TYPE DU BTYP DE BLOC RECHERCHE
## *     BLKNO      "    BLOC D'OU PART LA RECHERCHE
## int c_mrblocx(buf,bfam,bdesc,bknat,bktyp,bkstp,blk0)
## int buf[],bfam,bdesc,bknat,bktyp,bkstp,blk0;
##   {
##   int lbfam,lbdesc,lbknat,lbktyp,lbkstp,lblk0;
##   lbfam = bfam; lbdesc = bdesc; lblk0 = blk0;
##   lbknat = bknat; lbktyp = bktyp; lbkstp = bkstp;
##   return(f77name(mrblocx)(buf,&lbfam,&lbdesc,&lbknat,&lbktyp,&lbkstp,&lblk0));
##   }


##TODO: c_mrbrep
##  *   Replace a data block by an other one with the same variables and
##  *   dimensions.
##  *  IN/OUT buffer vector that contains the report
##  *    IN   bkno   block number to be replaced
##  *    IN   tblval array of values to write (nele*nval*nt)
## int c_mrbrep(void *buffer, int blkno, word *tblval)


#TODO: c_mrbrpt
## ***S/P  MRBRPT - VERIFIER SI UN ELEMENT EST REPETITIF OU NON
##       FUNCTION MRBRPT( ELEMENT )
##       INTEGER  MRBRPT, ELEMENT
## *     FONCTION SERVANT A VERIFIER SI UN ELEMENT EST REPETITIF OU NON.
## *     LA FONCTION RETOURNE:
## *        1 - ELEMENT REPETITIF
## *        0 - ELEMENT NON REPETITIF
## *       <0 - CODE D'ELEMENT NON VALABLE
## *            (PLUS PETIT QUE UN OU PLUS GRAND QUE MAXREP)
## *     ELEMENT  ENTREE  CODE DE L'ELEMENT A VERIFIER
## int c_mrbrpt(int elem) {
##    int lelem;
##    lelem = elem;
##    return(f77name(mrbrpt)(&lelem));
##    }


#TODO: c_mrbsct
## ***S/P - MRBSCT - INITIALISER LE TABLEAU DE CONVERSION DE L'USAGER
##       ENTRY MRBSCT(TBLUSR, NELEUSR)
## *     SOUS-PROGRAMME SERVANT A AJOUTER AU TABLEAU DE
## *     CONVERSION STANDARD, UNE LISTE D'ELEMENTS QUE
## *     L'USAGER A DEFINIE LUI-MEME.
## *     SI LE TABLEAU STANDARD N'A PAS ETE INITIALISE,
## *     ON APPELLE QRBSCT POUR EN FAIRE L'INITIALISATION.
## *     ON AJOUTE LE TABLEAU DE L'USAGER A LA FIN.
## *     TBLBURP   ENTREE CONTIENT LES CODES D'ELEMENTS
## *     NELEUSR   ENTREE        - NOMBRE D'ELEMENTS
## int c_mrbsct(int *tablusr, int neleusr)


#TODO: c_mrbtbl
## ***S/P - MRBTBL - REMPLIR UN TABLEAU A PARTIR DE TABLEBURP
## *     SOUS-PROGRAMME SERVANT A REMPLIR LE TABLEAU TBLBURP
## *     A PARTIR DES DESCRIPTIONS D'ELEMENTS TROUVEES DANS
## *     LE FICHIER TABLEBURP.  POUR CHAQUE ELEMENT, 
## *     ON RETOURNE:
## *        - FACTEUR D'ECHELLE
## *        - VALEUR DE REFERENCE
## *        - SI L'ELEMENT EST CONVERTISSABLE OU NON
## *          0 - non convertissable
## *          1 - convertissable
## *     NELE      ENTREE        - NOMBRE D'ELEMENTS A TRAITER
## *     TBLBURP   ENTREE CONTIENT LES CODES D'ELEMENTS
## *        "      SORTIE CONTIENT LES PARAMETRES DE CHAQUE ELEMENT
## *     ARANGEMENT DE TBLBURP:
## *     ----------------------------------------------------------
## *     | code elem 16 bits | echelle | reference | convertissable |
## *     |                   |         |           |                |
## *              .               .          .             .
## int c_mrbtbl(int *tablusr,int nslots,int neleusr)


#TODO: c_mrbupd
## ***S/P MRBUPD - DONNER UNE VALEUR AUX CLEFS D'UN RAPPORT
##       FUNCTION MRBUPD(IUN, BUF, TEMPS, FLGS, STNID, IDTYP, LATI, LONG,
##      X                DX, DY, ELEV, DRCV, DATEin, OARS, RUN, SUP, NSUP,
##      X                XAUX, NXAUX)
##       IMPLICIT NONE
##       INTEGER  MRBUPD, NSUP, NXAUX, IUN, BUF(*), TEMPS, FLGS, IDTYP,
##      X         LATI,   LONG, ELEV,  DX,  DY,     DRCV,  DATEin, OARS,
##      X         RUN,    SUP(*), XAUX(*)
##       CHARACTER*(*) STNID
## *OBJET( MRBUPD )
## *     MISE A JOUR DE L'ENTETE D'UN RAPPORT. SEULES LES CLEFS QUI
## *     N'ONT PAS POUR VALEUR -1 SERONT MISE A JOUR.  IL EN VA DE MEME
## *     POUR CHAQUE CARACTERE DE STNID S'IL EST DIFFERENT DE '*'.
## *ARGUMENTS
## *     IUN     ENTREE  NUMERO D'UNITE ASSOCIE AU FICHIER
## *     TYPREC    "     TYPE D'ENREGISTREMENT
## *     TEMPS     "     DIFF DE TEMPS ENTRE T VALIDITE ET T SYNOPTIQUE
## *     FLGS      "     MARQUEURS GLOBAUX
## *     STNID     "     IDENTIFICATEUR DE LA STATION
## *     IDTYP     "     TYPE DE RAPPORT
## *     LATI      "     LATITUDE DE LA STATION EN CENTIDEGRES
## *     LONG      "     LONGITUDE DE LA STATION EN CENTIDEGRES
## *     DX        "     DIMENSION X D'UNE BOITE
## *     DY        "     DIMENSION Y D'UNE BOITE
## *     ELEV      "     ALTITUDE DE LA STATION EN METRES
## *     DRCV      "     DELAI DE RECEPTION
## *     DATE      "     DATE SYNOPTIQUE DE VALIDITE (AAMMJJHH)
## *     OARS      "     RESERVE POUR ANALYSE OBJECTIVE
## *     RUN       "     IDENTIFICATEUR DE LA PASSE OPERATIONNELLE
## *     SUP       "     CLEFS PRIMAIRES SUPPLEMENTAIRES (AUCUNE POUR
## *                     LA VERSION 1990)
## *     NSUP      "     NOMBRE DE CLEFS PRIMAIRES SUPPLEMENTAIRES (DOIT
## *                     ETRE ZERO POUR LA VERSION 1990)
## *     XAUX      "     CLEFS AUXILIAIRES SUPPLEMENTAIRES (=0 VRSN 1990)
## *     NXAUX     "     NOMBRE DE CLEFS AUXILIAIRES SUPPLEMENTAIRES(=0)
## *     BUF       "     VECTEUR QUI CONTIENDRA LES ENREGISTREMENTS
## int c_mrbupd(iun,buf,temps,flgs,stnid,idtp,lati,longi,dx,dy,elev,drcv,date,
##          oars,runn,sup,nsup,xaux,nxaux)
## int buf[],temps,flgs,idtp,lati,longi,elev,drcv,date,oars,runn,sup[],nsup;
## int dx, dy;
## int xaux[],nxaux;
## char stnid[];


#TODO: c_mrfprm
## ***S/P MRFPRM - OBTENIR LES PARAMETRES PRINCIPAUX D'UN RAPPORT
##       FUNCTION MRFPRM(HANDLE, STNID, IDTYP, LAT, LON, DX, DY, DATE,
##      X                TEMPS,  FLGS,  SUP,   NSUP,     LONENR)
##       IMPLICIT NONE
##       INTEGER  MRFPRM, NSUP, IDTYP, LAT, DX, DATE, SUP(*), LONENR,
##      X         HANDLE, FLGS, TEMPS, LON, DY 
##       CHARACTER*9 STNID
## *OBJET( MRFPRM )
## *     ALLER CHERCHER LES PARAMETRES PRINCIPAUX DE L'ENREGISTREMENT
## *     DONT LE POINTEUR EST HANDLE.
## *ARGUMENTS
## *     HANDLE   ENTREE  POINTEUR A L'ENREGISTREMENT
## *     NSUP        "    NOMBRE DE DESCRIPTEURS SUPPLEMENTAIRES 
## *                      (DOIT ETRE EGAL A ZERO POUR VERSION 1990)
## *     STNID    SORTIE  IDENTIFICATEUR DE LA STATION
## *     IDTYP       "    TYPE DE RAPPORT
## *     LAT         "    LATITUDE EN CENTIDEGRES PAR RAPPORT AU POLE SUD
## *     LON         "    LONGITUDE EN CENTIDEGRES (0-35999)
## *     DX          "    DIMENSION X D'UNE BOITE
## *     DY          "    DIMENSION Y D'UNE BOITE
## *     DATE        "    DATE DE VALIDITE (AAMMJJHH)
## *     TEMPS       "    HEURE DE L'OBSERVATION
## *     FLGS        "    MARQUEURS GLOBAUX
## *     SUP         "    LISTE DE DESCRIPTEURS SUPPLEMENTAIRES (AUCUN)
## *     LONENR      "    LONGUEUR DE L'ENREGISTREMENT EN MOTS HOTE
## int c_mrfprm(handle,stnid,idtyp,lat,lon,dx,dy,date,temps,flgs,sup,nsup,lng)
## int handle,*idtyp,*lat,*lon,*date,*temps,*flgs,sup[],nsup,*lng;
## int *dx, *dy;
## char stnid[10];

import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc

from . import librmn
from rpnpy.librmn import proto as _rp

librmn.c_mrfgoc.argtypes = (_ct.c_char_p, _ct.c_char_p)
librmn.c_mrfgoc.restype  = _ct.c_int
c_mrfgoc = librmn.c_mrfgoc

librmn.c_mrfgor.argtypes = (_ct.c_char_p, _ct.POINTER(_ct.c_float))
librmn.c_mrfgor.restype  = _ct.c_int
c_mrfgor = librmn.c_mrfgor

librmn.c_mrfopc.argtypes = (_ct.c_char_p, _ct.c_char_p)
librmn.c_mrfopc.restype  = _ct.c_int
c_mrfopc = librmn.c_mrfopc

librmn.c_mrfopr.argtypes = (_ct.c_char_p, _ct.c_float)
librmn.c_mrfopr.restype  = _ct.c_int
c_mrfopr = librmn.c_mrfopr

librmn.c_mrfopn.argtypes = (_ct.c_int, _ct.c_char_p)
librmn.c_mrfopn.restype  = _ct.c_int
c_mrfopn = librmn.c_mrfopn

librmn.c_mrfcls.argtypes = (_ct.c_int,)
librmn.c_mrfcls.restype  = _ct.c_int
c_mrfcls = librmn.c_mrfcls

librmn.c_mrfvoi.argtypes = (_ct.c_int,)
librmn.c_mrfvoi.restype  = _ct.c_int
c_mrfvoi = librmn.c_mrfvoi

librmn.c_mrfnbr.argtypes = (_ct.c_int,)
librmn.c_mrfnbr.restype  = _ct.c_int
c_mrfnbr = librmn.c_mrfnbr

librmn.c_mrfmxl.argtypes = (_ct.c_int,)
librmn.c_mrfmxl.restype  = _ct.c_int
c_mrfmxl = librmn.c_mrfmxl

librmn.c_mrfbfl.argtypes = (_ct.c_int,)
librmn.c_mrfbfl.restype  = _ct.c_int
c_mrfbfl = librmn.c_mrfbfl

librmn.c_mrfrwd.argtypes = (_ct.c_int,)
librmn.c_mrfrwd.restype  = _ct.c_int
c_mrfrwd = librmn.c_mrfrwd

librmn.c_mrfapp.argtypes = (_ct.c_int,)
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

librmn.c_mrbhdr.argtypes = (
    ## word *buf, int *temps, int *flgs, char *stnid, int *idtyp,
    _npc.ndpointer(dtype=_np.int32),
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
    _npc.ndpointer(dtype=_np.int32),
    _ct.c_int,
    _npc.ndpointer(dtype=_np.int32),
    _ct.c_int
    )
librmn.c_mrbhdr.restype  = _ct.c_int
c_mrbhdr = librmn.c_mrbhdr

librmn.c_mrbprm.argtypes = (
    ## word *buf,int  bkno,
    _npc.ndpointer(dtype=_np.int32),
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

librmn.c_mrbtyp.argtypes = (
    _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int),
    _ct.c_int
    )
librmn.c_mrbtyp.restype  = _ct.c_int
c_mrbtyp = librmn.c_mrbtyp

librmn.c_mrbxtr.restype  = _ct.c_int
def c_mrbxtr(buf, bkno, lstele, tblval):
    if not isinstance(tblval, _np.ndarray):
        raise TypeError("tblval, expecting type numpy.ndarray, got {0}".format(repr(type(tblval))))
    librmn.c_mrbxtr.argtypes = (
        _npc.ndpointer(dtype=_np.int32),   ## void *buffer,
        _ct.c_int,                         ## int bkno,
        _npc.ndpointer(dtype=_np.int32),   ## word *lstele
        _npc.ndpointer(dtype=tblval.dtype) ## word *tblval
        )
    return librmn.c_mrbxtr(buf, bkno, lstele, tblval)


librmn.c_mrbdcl.argtypes = (
    ## int liste[], cliste[], nele;
    _npc.ndpointer(dtype=_np.int32), _npc.ndpointer(dtype=_np.int32), _ct.c_int,
    )
librmn.c_mrbdcl.restype  = _ct.c_int
c_mrbdcl = librmn.c_mrbdcl


librmn.c_mrbdcv.argtypes = (_ct.c_int,)
librmn.c_mrbdcv.restype  = _ct.c_int
c_mrbdcv = librmn.c_mrbdcv


librmn.c_mrbcvt.restype  = _ct.c_int
## c_mrbcvt = librmn.c_mrbcvt
def c_mrbcvt(lstele, tblval, rval, nele, nval, nt, mode):
    if not isinstance(tblval, _np.ndarray):
        raise TypeError("tblval, expecting type numpy.ndarray, got {0}".format(repr(type(tblval))))
    if not isinstance(rval, _np.ndarray):
        raise TypeError("rval, expecting type numpy.ndarray, got {0}".format(repr(type(rval))))
    librmn.c_mrbcvt.argtypes = (
        _npc.ndpointer(dtype=_np.int32),    ## int lstele[]
        _npc.ndpointer(dtype=tblval.dtype), ## int tblval[]
        _npc.ndpointer(dtype=rval.dtype),   ## float rval[];
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int  ## int nele, nval, nt, mode;
        )
    return librmn.c_mrbcvt(lstele, tblval, rval, nele, nval, nt, mode)


librmn.c_mrbini.argtypes = (_ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _ct.c_int, _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int)
librmn.c_mrbini.restype = _ct.c_int
c_mrbini = librmn.c_mrbini


librmn.c_mrbcol.argtypes = (_npc.ndpointer(dtype=_np.int32),
    _npc.ndpointer(dtype=_np.int32), _ct.c_int)
librmn.c_mrbcol.restype = _ct.c_int
c_mrbcol = librmn.c_mrbcol


librmn.c_mrbcov.argtypes = (_ct.c_int,)
librmn.c_mrbcov.restype  = _ct.c_int
c_mrbcov = librmn.c_mrbcov


c_mrbadd_argtypes_int = (_npc.ndpointer(dtype=_np.int32), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _npc.ndpointer(dtype=_np.int32))
c_mrbadd_argtypes_float = (_npc.ndpointer(dtype=_np.int32), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _npc.ndpointer(dtype=_np.float32))
librmn.c_mrbadd.restype = _ct.c_int
def c_mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp, lstele, tblval):
    if tblval.dtype == _np.dtype('int32'):
        librmn.c_mrbadd.argtypes = c_mrbadd_argtypes_int
    elif tblval.dtype == _np.dtype('float32'):
        librmn.c_mrbadd.argtypes = c_mrbadd_argtypes_float
    return librmn.c_mrbadd(buf, bkno, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp, lstele, tblval)

librmn.c_mrbdel.argtypes = (_npc.ndpointer(dtype=_np.int32), _ct.c_int)
librmn.c_mrbdel.restype = _ct.c_int
c_mrbdel = librmn.c_mrbdel

librmn.c_mrfdel.argtypes = (_ct.c_int, )
librmn.c_mrfdel.restype = _ct.c_int
c_mrfdel = librmn.c_mrfdel

# =========================================================================

if __name__ == "__main__":
    print("ctypes prototypes for many librmn burp C functions")

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
