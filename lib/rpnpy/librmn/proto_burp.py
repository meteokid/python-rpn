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

    c_mrfopc(option, value)
        Set a character option
        Proto:
            int c_mrfopc(option, value)
            char option[], value[];
        Args:
            option (str): (I) option name to be set
            value  (str): (I) option value
        Returns:
           int, zero if the connection is successful, non-zero otherwise

    c_mrfopn(iun, mode)
        Open a BURP file
        Proto:
            int c_mrfopn(iun, mode)
            int iun;
            char mode[];
        Args:
            iun  (int): (I) file unit number
            mode (str): (I) file open mode, one of: ['READ', 'CREATE', 'APPEND']
        Returns:
            int, number of active records in the file

    c_mrfcls(iun)
        Close a previously opened BURP file
        Proto:
            int c_mrfcls(iun)
            int iun;
        Args:
            iun  (int): (I) file unit number
        Returns:
           int, zero if successful, non-zero otherwise
           
    c_mrfnbr(iun)
        Return then number of active records in the file before opening it
        Proto:
            int c_mrfnbr(iun)
            int iun;
        Args:
            iun  (int): (I) file unit number
        Returns:
           int, number of active records in the file

    c_mrfmxl(iun)
        Return the lenght of the longest record in the file
        Proto:
            int c_mrfmxl(iun)
            int iun;
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
            int, TODO

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
            int, TODO

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
            nsup   (int)   : (O) number of sup
            xaux   (array) : (O) clefs auxiliaires supplementaires
            nxaux  (int)   : (O) number of xaux
        Returns:
            int, TODO

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
            int, TODO

    c_mrbxtr(buf, bkno, lstele, tblval)
        Extract list of element and values from buffer. 
        Proto:
            int c_mrbxtr(void *buf, int bkno, word *lstele, word *tblval)
        Args:
            buf    (array) : (I) vector containing the report data
            bkno   (int)   : (I) block number
            lstele (array) : (O) list of nele meteorogical elements (array of int)
            tblval (array) : (O) array of values to write (nele*nval*nt)
                                 (array of int or float)
        Returns:
            int, TODO

    c_mrbdcl(cliste, liste, nele)
        Decode List of Elements
        pour un element, on retourne sa valeur sous format decimal
        abbccc, (a,b,c de 0 a 9)
        ou
        a    provient des bits 14 et 15 de l'element
        bb       "    des bits 8 a 13 de l'element
        ccc      "    des bits 0 a 7  de l'element
        Proto:
            int c_mrbdcl(cliste, liste, nele)
            int liste[], cliste[], nele;
        Args:
            cliste (array) : (I) Elements to be decoded (array of int)
            liste  (array) : (O) Decoded elements (array of int)
            nele   (int)   : (I) Number of elemets to decode
        Returns:
            int, TODO

    c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
        Perform a unit conversion to/from BUFR code to/from real values
        Proto:
            int c_mrbcvt(liste, tblval, rval, nele, nval, nt, mode)
            int liste[], tblval[], nele, nval, nt, mode;
            float rval[];
        Args:
            liste  (array) : (O)   Elements to convert (array of int)
            tblval (array) : (I/O) BURF code values (array of int or float)
            rval   (array) : (I/O) Real values (array of float)
            nele   (int)   : (I)   Number of elemets to convert
            nval   (int)   : (I)   Number of values per elemet
            nt     (int)   : (I)   Number of ensembles nele*nval
            mode   (int)   : (I)   Conversion mode,
                                   0 = RVAL to TBLVAL (BUFR Codes)
                                   1 = TBLVAL (BUFR Codes) to RVAL
        Returns:
            int, TODO

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
            int, TODO
         
    c_mrbcol(liste, cliste, nele)
        Encode elemets of a list
        Sous-programme retournant une liste d'elements codes de telle sorte
        que chaque element occupe seize bits.
        pour un element ayant le format decimal  abbccc, (a,b,c de 0 a 9)
        on retourne un entier contenant a sur deux bits, bb sur six bits
        et ccc sur huit bits
        Proto:
            int c_mrbcol(liste, cliste, nele)
            int liste[], cliste[], nele;
        Args:
            liste  (array) : (I) Elements to be encoded (array of int)
            cliste (array) : (O) Encoded Elements (array of int)
            nele   (int)   : (I) Number of elemets to encode
        Returns:
            int, TODO

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
            int, TODO


## ***S/P MRFOPR - INITIALISER UNE OPTION DE TYPE REEL
## *     FONCTION SERVANT A INITIALISER UNE OPTION DE TYPE REEL
## *     LA VALEUR DE L'OPTION EST CONSERVEE DANS LE COMMON BURPUSR
## *     OPTNOM   ENTREE  NOM DE L'OPTION A INITIALISER
## *              MISSING
## *     OPVALR     "     VALEUR A DONNER A L'OPTION

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

## /***************************************************************************** 
##  *                             C _ M R B D E L                               *
##  *                                                                           * 
##  *Object                                                                     * 
##  *   Delete a particular block of the report.                                *
##  *                                                                           *
##  *Arguments                                                                  * 
##  *                                                                           * 
##  *  IN/OUT buffer   vector to contain the report                             * 
##  *    IN   number   block number to be deleted                               *
##  *                                                                           * 
##  *****************************************************************************/
## int c_mrbdel(void *buffer, int number)

## /***************************************************************************** 
##  *                             C _ M R B L E N                               *
##  *                                                                           * 
##  *Object                                                                     * 
##  *   Return the number of bits used in buf and the number of bits left.      *
##  *                                                                           *
##  *Arguments                                                                  * 
##  *                                                                           * 
##  *    IN   buffer   vector that contains the report                          * 
##  *   OUT   lbits    number of bits used                                      *
##  *   OUT   left     number of bits left                                      *
##  *                                                                           * 
##  *****************************************************************************/
## int c_mrblen(void *buffer, int *lbits, int *left)

## /***************************************************************************** 
##  *                             C _ M R B L O C                               *
##  *                                                                           * 
##  *Object                                                                     * 
##  *   Search for a specific block in the buffer. Search starts at block       *
##  *   blkno. If blkno = 0 search starts from the beginning.                   *
##  *                                                                           *
##  *Arguments                                                                  * 
##  *                                                                           * 
##  *  IN   buffer vector to contain the report                                 * 
##  *  IN   bfam   block family (12 bits, bdesc no more used)                   *
##  *  IN   bdesc  kept for backward compatibility                              *
##  *  IN   btyp   block type                                                   *
##  *  in   bkno   number of blocks in buf                                      *
##  *                                                                           * 
##  *****************************************************************************/
## int c_mrbloc(void *buffer, int bfam, int bdesc, int btyp, int blkno)

## /***************************************************************************** 
##  *                             C _ M R B R E P                               *
##  *                                                                           * 
##  *Object                                                                     * 
##  *   Replace a data block by an other one with the same variables and        *
##  *   dimensions.                                                             *
##  *                                                                           *
##  *Arguments                                                                  * 
##  *                                                                           * 
##  *  IN/OUT buffer vector that contains the report                            * 
##  *    IN   bkno   block number to be replaced                                *
##  *    IN   tblval array of values to write (nele*nval*nt)                    *
##  *                                                                           * 
##  *****************************************************************************/
## int c_mrbrep(void *buffer, int blkno, word *tblval)

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

c_mrbxtr_argtypes_int = (
    ## void *buffer, int bkno,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int,
    ## word *lstele, word *tblval
    _npc.ndpointer(dtype=_np.int32),#float32),
    _npc.ndpointer(dtype=_np.int32)#float32)
    )
c_mrbxtr_argtypes_float = (
    ## void *buffer, int bkno,
    _npc.ndpointer(dtype=_np.int32),#float32),
    _ct.c_int,
    ## word *lstele, word *tblval
    _npc.ndpointer(dtype=_np.int32),#float32),
    _npc.ndpointer(dtype=_np.float32)
    )
librmn.c_mrbxtr.restype  = _ct.c_int
def c_mrbxtr(buf,bkno,lstele,tblval):
    if tblval.dtype == _np.dtype('int32'):
        librmn.c_mrbxtr.argtypes = c_mrbxtr_argtypes_int
    elif tblval.dtype == _np.dtype('float32'):
        librmn.c_mrbxtr.argtypes = c_mrbxtr_argtypes_float
    return librmn.c_mrbxtr(buf,bkno,lstele,tblval)


librmn.c_mrbdcl.argtypes = (
    ## int liste[], cliste[], nele;
    _npc.ndpointer(dtype=_np.int32), _npc.ndpointer(dtype=_np.int32), _ct.c_int,
    )
librmn.c_mrbdcl.restype  = _ct.c_int
c_mrbdcl = librmn.c_mrbdcl


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


librmn.c_mrbini.argtypes = ( _ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _ct.c_int, _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int )
librmn.c_mrbini.restype = _ct.c_int
c_mrbini = librmn.c_mrbini


librmn.c_mrbcol.argtypes = ( _npc.ndpointer(dtype=_np.int32), 
    _npc.ndpointer(dtype=_np.int32), _ct.c_int )
librmn.c_mrbcol.restype = _ct.c_int
c_mrbcol = librmn.c_mrbcol


c_mrbadd_argtypes_int = ( _npc.ndpointer(dtype=_np.int32), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _npc.ndpointer(dtype=_np.int32) )
c_mrbadd_argtypes_float = ( _npc.ndpointer(dtype=_np.int32), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.c_int, _npc.ndpointer(dtype=_np.int32),
    _npc.ndpointer(dtype=_np.float32) )
librmn.c_mrbadd.restype = _ct.c_int
def c_mrbadd(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp,lstele,tblval):
    if tblval.dtype == _np.dtype('int32'):
        librmn.c_mrbadd.argtypes = c_mrbadd_argtypes_int
    elif tblval.dtype == _np.dtype('float32'):
        librmn.c_mrbadd.argtypes = c_mrbadd_argtypes_float
    return librmn.c_mrbadd(buf,bkno,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp,lstele,tblval)

# =========================================================================

if __name__ == "__main__":
    print(str(_rp.c_fst_version()))
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
