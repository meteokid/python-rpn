#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
  Module librmn.fstd98 contains python wrapper to main librmn's fstd98, convip C functions along wiht helper functions

 @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

import os
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
import librmn.proto as _rp
import librmn.const as _rc
import librmn.base as _rb

c_mkstr = lambda x: _ct.create_string_buffer(x)
c_toint = lambda x: (x if (type(x) != type(_ct.c_int())) else x.value)
isListType = lambda x: type(x) in (type([]),type((1,)))

#---- helpers -------------------------------------------------------

def dtype_fst2numpy(datyp):
    """Return the numpy dtype datyp for the given fst datyp
            0: binary, transparent
            1: floating point
            2: unsigned integer
            3: character (R4A in an integer)
            4: signed integer
            5: IEEE floating point
            6: floating point (16 bit, made for compressor)
            7: character string
            8: complex IEEE
          130: compressed short integer  (128+2)
          133: compressed IEEE           (128+5)
          134: compressed floating point (128+6)
        +128 : second stage packer active
        +64  : missing value convention used
    """
    datyp = (datyp-128 if datyp>=128 else datyp)
    datyp = (datyp-64 if datyp>=64 else datyp)
    return _rc.FST_DATYP2NUMPY_LIST[datyp]


def dtype_numpy2fst(dtype,compress=True,missing=False):
    """Return the fst datyp for the given numpy dtype
       Optionally specify compression and missing value options.
    """
    datyp = 0 #default returned type: binary
    for k in _rc.FST_DATYP2NUMPY_LIST.keys():
        if _rc.FST_DATYP2NUMPY_LIST[k] == dtype:
            datyp = k
            break
    if compress: datyp += 128
    if missing:  datyp += 64
    return datyp


def isFST(filename):
    """Return True if file is of RPN STD RND type
    
    filename : path/name of the file to examine
    """
    return (_rb.wkoffit(filename) in (_rc.WKOFFIT_TYPE_LIST['STD_RND_89'],_rc.WKOFFIT_TYPE_LIST['STD_RND_98']))


def fstopenall(paths,filemode=_rc.FST_RO):
    """shortcut for fnom+fstouv+fstlnk
    
    paths    : path/name of the file to open
               if paths is a list, open+link all files
               if path is a dir, open+link all fst files in dir
    filemode : a string with the desired filemode (see librmn doc)
               or one of these constants: FST_RW, FST_RW_OLD, FST_RO
               
    return Associated file unit number
    """
    if type(paths) == type(''): paths = [paths]
    l = []
    for x in paths:
        if os.path.isdir(x):
            for (dirpath, dirnames, filenames) in os.walk(x):
                for f in filenames:
                    if isFST(os.path.join(x,f)):
                        l.append(os.path.join(x,f))
                break
            pass #TODO splice file list, discard non fst files
        else:
            l.append(x)
    if filemode != _rc.FST_RO and len(paths) > 1:
        return None #print error msg
    iunitlist = []
    for x in paths:
        i = _rb.fnom(x,filemode)
        if i:
            i2 = fstouv(i,filemode)
            if i2 != None: #TODO: else warning/ignore
                iunitlist.append(i)
    if len(iunitlist) == 0:
        return None #print error msg
    if len(iunitlist) == 1:
        return iunitlist[0]
    return fstlnk(unitList)


def fstcloseall(iunit):
    """shortcut for fclos+fstfrm
    
    iunit    : unit number associated to the file
               obtained with fnom or fstopenall
               
    return None on error int>=0 otherwise
    """
    #TODO: loop on all linked units
    istat = fstfrm(iunit)
    istat = _rb.fclos(iunit)
    return istat



def listToFLOATIP(rp1):
    """
    """
    if isinstance(rp1,_rp.FLOAT_IP):
        return rp1
    if not isListType(rp1):
        raise TypeError
    if not len(rp1) in (2,3):
        raise TypeError
    if len(rp1) == 2:
        return _rp.FLOAT_IP(rp1[0],rp1[0],rp1[1])
    return _rp.FLOAT_IP(rp1[0],rp1[1],rp1[2])

    
def FLOATIPtoList(rp1):
    """
    """
    if isinstance(rp1,_rp.FLOAT_IP):
        return (rp1.v1,rp1.v2,rp1.kind)
    return rp1
    
    
#--- fstd98 ---------------------------------------------------------

def fstecr(iunit,data,meta,rewrite=True):
    """Writes record to file
    """
    #TODO: check if file is open with write permission
    meta2 = _rc.FST_RDE_META_DEFAULT.copy() #.deepcopy()
    for k in meta.keys():
        if k != 'd' and meta[k] not in (' ',-1): meta2[k] = meta[k]
    irewrite = (1 if rewrite else 0)
    npak = -abs(meta['nbits'])
    _rp.c_fstecr.argtypes = (
        _npc.ndpointer(dtype=data.dtype), _npc.ndpointer(dtype=data.dtype),
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,_ct.c_char_p,
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int)
    istat = _rp.c_fstecr(data, data, npak, iunit,
                meta['dateo'], meta['deet'], meta['npas'],
                meta['ni'], meta['nj'], meta['nk'],
                meta['ip1'], meta['ip2'], meta['ip3'],
                meta['typvar'], meta['nomvar'], meta['etiket'], meta['grtyp'],
                meta['ig1'], meta['ig2'], meta['ig3'], meta['ig4'],
                meta['datyp'], irewrite)
    if istat < 0: return None
    return istat


def fst_edit_dir(key, dateo=-1, deet=-1, npas=-1, ni=-1, nj=-1, nk=-1,
                 ip1=-1, ip2=-1, ip3=-1,
                 typvar=' ', nomvar=' ', etiket=' ', grtyp=' ',
                 ig1=-1, ig2=-1, ig3=-1, ig4=-1, datyp=-1):
    """Edits the directory content of a RPN standard file
    
    key   : positioning information to the record,
            obtained with fstinf or fstinl, ...
    dateo : date time stamp
    deet  : length of a time step in seconds
    npas  : time step number
    ni    : first dimension of the data field
    nj    : second dimension of the data field
    nk    : third dimension of the data field
    nbits : number of bits kept for the elements of the field
    datyp : data type of the elements
    ip1   : vertical level
    ip2   : forecast hour
    ip3   : user defined identifier
    typvar: type of field (forecast, analysis, climatology)
    nomvar: variable name
    etiket: label
    grtyp : type of geographical projection
    ig1   : first grid descriptor
    ig2   : second grid descriptor
    ig3   : third grid descriptor
    ig4   : fourth grid descriptor
            
    return None on error int>=0 otherwise

    Only provided parameters with value different than default are updated
    """
    istat = _rp.c_fst_edit_dir(key,dateo, deet, npas, ni, nj, nk,
                 ip1, ip2, ip3, typvar, nomvar, etiket, grtyp,
                 ig1, ig2, ig3, ig4, datyp)
    if istat < 0: return None
    return istat


def fsteff(key):
    """Deletes the record associated to handle.
    
    key   : positioning information to the record,
            obtained with fstinf or fstinl, ...
            
    return None on error int>=0 otherwise
    """
    istat = _rp.c_fsteff(key)
    if istat < 0: return None
    return istat


def fstfrm(iunit):
    """Close a RPN standard file
    
    iunit    : unit number associated to the file
               obtained with fnom+fstouv

    return None on error int>=0 otherwise
    """
    istat = _rp.c_fstfrm(iunit)
    if istat < 0: return None
    return istat


def fstinf(iunit,datev=-1,etiket=' ',ip1=-1,ip2=-1,ip3=-1,typvar=' ',nomvar=' '):
    """Locate the next record that matches the research keys
        
    iunit   : unit number associated to the file
              obtained with fnom+fstouv
    datev   : valid date
    etiket  : label
    ip1     : vertical level
    ip2     : forecast hour
    ip3     : user defined identifier
    typvar  : type of field
    nomvar  : variable name

    return {
            'key'   : key,       # key/handle of the 1st matching record
            'shape' : (ni,nj,nk) # dimensions of the field
            }
    return None if no matching record

    Only provided parameters with value different than default
    are used as selction criteria
    """
    return fstinfx(-2,iunit,datev,etiket,ip1,ip2,ip3,typvar,nomvar)


def fstinfx(key,iunit,datev=-1,etiket=' ',ip1=-1,ip2=-1,ip3=-1,typvar=' ',nomvar=' '):
    """Locate the next record that matches the research keys
       The search begins at the position given by key/handle
       obtained with fstinf or fstinl, ...

    key     : record key/handle of the search start position
    iunit   : unit number associated to the file
              obtained with fnom+fstouv
    datev   : valid date
    etiket  : label
    ip1     : vertical level
    ip2     : forecast hour
    ip3     : user defined identifier
    typvar  : type of field
    nomvar  : variable name

    return {
            'key'   : key,       # key/handle of the 1st matching record
            'shape' : (ni,nj,nk) # dimensions of the field
            }
    return None if no matching record

    Only provided parameters with value different than default
    are used as selction criteria
    """
    (cni,cnj,cnk) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    key2 = _rp.c_fstinfx(key,iunit,_ct.byref(cni),_ct.byref(cnj),_ct.byref(cnk),datev,etiket,ip1,ip2,ip3,typvar,nomvar)
    ## key2 = c_toint(key2)
    if key2 < 0: return None
    ## fx = lambda x: (x.value if x.value>0 else 1)
    return {
        'key'   : key2 ,
        'shape' : (max(1,cni.value),max(1,cnj.value),max(1,cnk.value)),
        }


def fstinl(iunit,datev=-1,etiket=' ',ip1=-1,ip2=-1,ip3=-1,typvar=' ',nomvar=' ',nrecmax=-1):
    """Locate all the record matching the research keys
        
    iunit   : unit number associated to the file
              obtained with fnom+fstouv
    datev   : valid date
    etiket  : label
    ip1     : vertical level
    ip2     : forecast hour
    ip3     : user defined identifier
    typvar  : type of field
    nomvar  : variable name
    nrecmax : maximum number or record to find (-1 = all)

    return list of keys

    Only provided parameters with value different than default
    are used as selction criteria
    """
    if nrecmax <= 0: nrecmax = _rp.c_fstnbrv(iunit)
    creclist = _np.empty(nrecmax,dtype=_np.intc)
    print nrecmax,creclist,repr(creclist)
    (cni,cnj,cnk,cnfound) = (_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_fstinl(iunit,_ct.byref(cni),_ct.byref(cnj),_ct.byref(cnk),datev,etiket,ip1,ip2,ip3,typvar,nomvar,creclist,cnfound,nrecmax)
    if cnfound <= 0: return []
    return creclist[0:cnfound.value].tolist()


#TODO: fstlic
#TODO: fstlir
#TODO: fstlirx
#TODO: fstlis


def fstlnk(unitList):
    """Links a list of files together for search purpose

    unitList : list of previously opened (fnom+fstouv) file units

    return File unit for the grouped unit
    return None on error
    """
    if len(unitList)<1 or unitList[0]<=0: return None
    cunitList = nm.asarray(unitList, dtype=nm.intc)
    istat = _rp.c_xdflnk(cunitList,len(cunitList))
    if istat<0: return None
    return unitList[0]


def fstluk(key,dtype=None,rank=None):
    """Read the record at position given by key/handle
    
    key   : positioning information to the record,
            obtained with fstinf or fstinl, ...
    dtype : array type of the returned data
            Default is determined from records' datyp
            Could be any numpy.ndarray type
            See: http://docs.scipy.org/doc/numpy/user/basics.types.html
    rank  : try to return an array with the specified rank
    return {
            'd'   : data,       # record data as a numpy.ndarray
            ...                 # same params list as fstprm
            }
    return None on error
    """
    if type(key) != type(1):
       raise TypeError("fstluk: Expecting a key of type %s, Got %s : %s" % (type(1),type(key),repr(key)))
    params = fstprm(key)
    if params is None: return None
    (cni,cnj,cnk) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    if dtype is None: dtype = dtype_fst2numpy(params['datyp'])
    _rp.c_fstluk.argtypes = (_npc.ndpointer(dtype=dtype),_ct.c_int,
        _ct.POINTER(_ct.c_int),_ct.POINTER(_ct.c_int),_ct.POINTER(_ct.c_int))
    wantrank = 1 if rank is None else rank
    minrank = 3
    if params['shape'][2] <= 1:
        minrank = 1 if params['shape'][1] <= 1 else 2
    rank = max(1,max(minrank,wantrank))
    myshape = [1 for i in range(rank)]
    maxrank = min(rank,len(params['shape']))
    myshape[0:maxrank] = params['shape'][0:maxrank]
    params['shape'] = myshape
    #raise ValueError("fstluk (%d, %d, %d) r=%d, s=%s" % (wantrank, minrank, len(params['shape']),rank, repr(params['shape'][0:rank])))
    data = _np.empty(params['shape'],dtype=dtype,order='FORTRAN')
    istat = _rp.c_fstluk(data,key,cni,cnj,cnk)
    if istat < 0: return None
    params['d'] = data
    return params


#TODO: fstmsq
#TODO: fstnbr
#TODO: fstnbrv


def fstopt(optName,optValue,setOget=_rc.FSTOP_SET):
    """Set or print FST option.

    optName  : name of option to be set or printed
               or one of these constants:
               FSTOP_MSGLVL, FSTOP_TOLRNC, FSTOP_PRINTOPT, FSTOP_TURBOCOMP
    optValue : value to be set (int or string)
               or one of these constants:
               for optName=FSTOP_MSGLVL:
                  FSTOPI_MSG_DEBUG,   FSTOPI_MSG_INFO,  FSTOPI_MSG_WARNING,
                  FSTOPI_MSG_ERROR,   FSTOPI_MSG_FATAL, FSTOPI_MSG_SYSTEM,
                  FSTOPI_MSG_CATAST
               for optName=FSTOP_TOLRNC:
                  FSTOPI_TOL_NONE,    FSTOPI_TOL_DEBUG, FSTOPI_TOL_INFO,
                  FSTOPI_TOL_WARNING, FSTOPI_TOL_ERROR, FSTOPI_TOL_FATAL
               for optName=FSTOP_TURBOCOMP:
                  FSTOPS_TURBO_FAST, FSTOPS_TURBO_BEST
    setOget  : define mode, set or print/get
               one of these constants: FSTOP_SET, FSTOP_GET
               default: set mode
               
    return None on error int>=0 otherwise
    """
    if type(optValue) == type(''):
        istat = _rp.c_fstopc(optName,optValue,setOget)
    elif type(optValue) == type(1):
        istat = _rp.c_fstopi(optName,optValue,setOget)
    else:
        return None
    if istat >= 0: return istat
    return None


def fstouv(iunit,filemode=_rc.FST_RW):
    """Opens a RPN standard file
    
    iunit    : unit number associated to the file
               obtained with fnom
    filemode : a string with the desired filemode (see librmn doc)
               or one of these constants: FST_RW, FST_RW_OLD, FST_RO

    return None on error int>=0 otherwise
    """
    istat = _rp.c_fstouv(iunit,filemode)
    if istat < 0: return None
    return istat


def fstprm(key):
    """Get all the description informations of the record.
    
    key : positioning information to the record,
          obtained with fstinf or fstinl, ...
    
    return {
            'key'   : key,       # key/handle of the 1st matching record
            'shape' : (ni,nj,nk) # dimensions of the field
            'dateo' : date time stamp
            'deet'  : length of a time step in seconds
            'npas'  : time step number
            'ni'    : first dimension of the data field
            'nj'    : second dimension of the data field
            'nk'    : third dimension of the data field
            'nbits' : number of bits kept for the elements of the field
            'datyp' : data type of the elements
            'ip1'   : vertical level
            'ip2'   : forecast hour
            'ip3'   : user defined identifier
            'typvar': type of field (forecast, analysis, climatology)
            'nomvar': variable name
            'etiket': label
            'grtyp' : type of geographical projection
            'ig1'   : first grid descriptor
            'ig2'   : second grid descriptor
            'ig3'   : third grid descriptor
            'ig4'   : fourth grid descriptor
            'swa'   : starting word address
            'lng'   : record length
            'dltf'  : delete flag
            'ubc'   : unused bit count
            'xtra1' : extra parameter
            'xtra2' : extra parameter
            'xtra3' : extra parameter
            }
    return None on error
    """
    (cni,cnj,cnk)        = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cdateo,cdeet,cnpas) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cnbits,cdatyp,cip1,cip2,cip3) = (_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    (ctypvar,cnomvar,cetiket) = (c_mkstr(' '*_rc.FST_TYPVAR_LEN),c_mkstr(' '*_rc.FST_NOMVAR_LEN),c_mkstr(' '*_rc.FST_ETIKET_LEN))
    (cgrtyp,cig1,cig2,cig3,cig4) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cswa,clng,cdltf,cubc,cxtra1,cxtra2,cxtra3) = (_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_fstprm(
        key,_ct.byref(cdateo),_ct.byref(cdeet),_ct.byref(cnpas),
        _ct.byref(cni),_ct.byref(cnj),_ct.byref(cnk),
        _ct.byref(cnbits),_ct.byref(cdatyp),
        _ct.byref(cip1),_ct.byref(cip2),_ct.byref(cip3),
        ctypvar,cnomvar,cetiket,
        cgrtyp,_ct.byref(cig1),_ct.byref(cig2),_ct.byref(cig3),_ct.byref(cig4),
        _ct.byref(cswa),_ct.byref(clng),_ct.byref(cdltf),_ct.byref(cubc),
        _ct.byref(cxtra1),_ct.byref(cxtra2),_ct.byref(cxtra3))
    istat = c_toint(istat)
    if istat < 0: return None
    return {
        'key'   : key ,
        'shape' : (max(1,cni.value),max(1,cnj.value),max(1,cnk.value)),
        'dateo' : cdateo.value,
        'deet'  : cdeet.value,
        'npas'  : cnpas.value,
        'ni'    : cni.value,
        'nj'    : cnj.value,
        'nk'    : cnk.value,
        'nbits' : cnbits.value,
        'datyp' : cdatyp.value,
        'ip1'   : cip1.value,
        'ip2'   : cip2.value,
        'ip3'   : cip3.value,
        'typvar': ctypvar.value,
        'nomvar': cnomvar.value,
        'etiket': cetiket.value,
        'grtyp' : cgrtyp.value,
        'ig1'   : cig1.value,
        'ig2'   : cig2.value,
        'ig3'   : cig3.value,
        'ig4'   : cig4.value,
        'swa'   : cswa.value,
        'lng'   : clng.value,
        'dltf'  : cdltf.value,
        'ubc'   : cubc.value,
        'xtra1' : cxtra1.value,
        'xtra2' : cxtra2.value,
        'xtra3' : cxtra3.value
        }


#TODO: fstsui
#TODO: fstvoi
#TODO: fst_version

#TODO: ip1_all,ip2_all,ip3_all
#TODO: ip1_val,ip2_val,ip3_val
#TODO: ip_is_equal

#--- fstd98/convip_plus & convert_ip123 ---------------------------------

#TODO: review, test the 4 following and review docstring
    
def convertIp(mode,v,k=0):
    """Codage/Decodage P,kind <-> IP pour IP1, IP2, IP3
Note: successeur de convip

Proto:
    void ConvIp(int *ip, float *p, int *kind, int mode)

Args:
    ip   (int)  : (I/O) Valeur codee
    p    (float): (I/O) Valeur reelle
    kind (int)  : (I/O) Type de niveau
    mode (int)  : (I)   Mode de conversion

  kind:
    0, p est en hauteur (m) rel. au niveau de la mer (-20,000 -> 100,000)
    1, p est en sigma                                (0.0 -> 1.0)
    2, p est en pression (mb)                        (0 -> 1100)
    3, p est un code arbitraire                      (-4.8e8 -> 1.0e10)
    4, p est en hauteur (M) rel. au niveau du sol    (-20,000 -> 100,000)
    5, p est en coordonnee hybride                   (0.0 -> 1.0)
    6, p est en coordonnee theta                     (1 -> 200,000)
    10, p represente le temps en heure               (0.0 -> 1.0e10)
    15, reserve (entiers)                                   
    17, p represente l'indice x de la matrice de conversion (1.0 -> 1.0e10)
        (partage avec kind=1 a cause du range exclusif
    21, p est en metres-pression                     (0 -> 1,000,000) fact=1e4
        (partage avec kind=5 a cause du range exclusif)

  mode:
    -1, de IP -->  P
    0, forcer conversion pour ip a 31 bits
       (default = ip a 15 bits) (appel d'initialisation)
    +1, de P  --> IP
    +2, de P  --> IP en mode NEWSTYLE force a true
    +3, de P  --> IP en mode NEWSTYLE force a false
    """
    (cip,cp,ckind) = (_ct.c_int(),_ct.c_float(),_ct.c_int())
    if mode >0:
        (cp,ckind) = (_ct.c_float(v),_ct.c_int(k))
    else:
        cip = _ct.c_int(v)
    _rp.c_ConvertIp(_ct.byref(cip),_ct.byref(cp),_ct.byref(ckind),mode)
    if mode >0:
        return cip.value
    else:
        return (cp.value,ckind.value)


def convertIPtoPK(ip1,ip2,ip3):
    """Convert/decode ip1,ip2,ip3 to their kind + real value conterparts
    Proto:
        ConvertIPtoPK(RP1,kind1,RP2,kind2,RP3,kind3,IP1V,IP2V,IP3V) result(status)
    Args:
        integer(C_INT) :: status
        real(C_FLOAT),         intent(OUT) :: RP1,RP2,RP3
        integer(C_INT),        intent(OUT) :: kind1,kind2,kind3
        integer(C_INT), value, intent(IN)  :: IP1V,IP2V,IP3V

        INPUTS
            IP1V,IP2V,IP3V IP values to be decoded
        OUTPUTS
            RP1,kind1  result of IP1V decoding
            RP2,kind2  result of IP2V decoding
            RP3,kind3  result of IP3V decoding
    Returns:
        int, 0 if ok, >0 on guessed the value, 32 on warning
    Raises:
        ValueError when provided values cannot be converted
    """
    (cp1,ck1,cp2,ck2,cp3,ck3) = (_ct.c_float(),_ct.c_int(),_ct.c_float(),_ct.c_int(),_ct.c_float(),_ct.c_int())
    istat = _rp.c_ConvertIPtoPK(_ct.byref(cp1),_ct.byref(ck1),_ct.byref(cp2),_ct.byref(ck2),_ct.byref(cp3),_ct.byref(ck3),ip1,ip2,ip3)
    if istat == 64: raise ValueError
    return (listToFLOATIP((cp1.value,cp1.value,ck1.value)),
            listToFLOATIP((cp2.value,cp2.value,ck2.value)),
            listToFLOATIP((cp3.value,cp3.value,ck3.value)))
    

def convertPKtoIP(pk1,pk2,pk3):
    """Convert/encode kind + real value into ip1,ip2,ip3
    Proto:
        ConvertPKtoIP(IP1,IP2,IP3,P1,kkind1,P2,kkind2,P3,kkind3) result(status)
    Args:
        integer(C_INT) :: status
        integer(C_INT),        intent(OUT) :: IP1,IP2,IP3
        real(C_FLOAT),  value, intent(IN)  :: P1,P2,P3
        integer(C_INT), value, intent(IN)  :: kkind1,kkind2,kkind3

        INPUTS
            P1,kkind1 must be a level
            P2,kkind2 should be a time but a level is accepted (flagged as WARNING)
            P3,kkind3 may be anything
        OUTPUTS
            IP1,IP2,IP3 will contain the encoded values in case of success,
                and are undefined otherwise
    Returns:
        int, 0 if ok, >0 on guessed the value, 32 on warning
    Raises:
        ValueError when provided values cannot be converted
    """
    (cip1,cip2,cip3) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    pk1 = listToFLOATIP(pk1)
    pk2 = listToFLOATIP(pk2)
    pk3 = listToFLOATIP(pk3)

    istat = _rp.c_ConvertPKtoIP(_ct.byref(cip1),_ct.byref(cip2),_ct.byref(cip3),pk1.kind,pk1.v1,pk2.kind,pk2.v1,pk3.kind,pk3.v1)
    if istat == 64: raise ValueError
    return (cip1.value,cip2.value,cip3.value)


def EncodeIp(rp1,rp2,rp3):
    """Produce encode (ip1,ip2,ip3) triplet from (real value,kind) pairs
    Args:
       RP123 (FLOAT_IP, FLOAT_IP, RP123) :
          [0] a level (or a pair of levels) in the atmosphere
          [1] a time (or a pair of times)
          [2] may contain anything, RP3%hi will be ignored (if RP1 or RP2 contains a pair, RP3 is ignored)
    Returns:
       (IP1,IP2,IP3) : encoded values in case of success,
    Raises:
       ValueError when provided values cannot be converted
    """
    rp1 = listToFLOATIP(rp1)
    rp2 = listToFLOATIP(rp2)
    rp3 = listToFLOATIP(rp3)
    (cip1,cip2,cip3) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_EncodeIp(_ct.byref(cip1),_ct.byref(cip2),_ct.byref(cip3),
                       _ct.byref(rp1),_ct.byref(rp2),_ct.byref(rp3))
    if istat == 32: raise ValueError
    return (cip1.value,cip2.value,cip3.value)


def DecodeIp(ip1,ip2,ip3):
    """Produce decoded (real value,kind) pairs from (ip1,ip2,ip3) ecnoded triplet
    Args:
       ip123 (int,int,int): encoded value 'new style' (old style encoding accepted)
    Returns:
       (RP1,RP2,RP3) : (tuple o 3 FLOAT_IP) decoded values in case of success
       None on error
    Raises:
       ValueError when provided values cannot be converted
    """
    (rp1,rp2,rp3) = (_rp.FLOAT_IP(0.,0.,0),_rp.FLOAT_IP(0.,0.,0),_rp.FLOAT_IP(0.,0.,0))
    (cip1,cip2,cip3) = (_ct.c_int(ip1),_ct.c_int(ip2),_ct.c_int(ip3))
    istat = _rp.c_DecodeIp(_ct.byref(rp1),_ct.byref(rp2),_ct.byref(rp3),
                           cip1,cip2,cip3)
    if istat == 32: raise ValueError
    return (rp1,rp2,rp3)


def kindToString(kind):
    """Translate kind integer code to 2 character string,
gateway to Fortran kind_to_string
Proto:
    void KindToString(int kind, char *s1, char *s2)
Args:
    kind (int): (I) Valeur codee
    s1 (str): (O) first char
    s2 (str): (O) second char
    """
    (s1,s1) = (_ct.c_char_p,_ct.c_char_p) #TODO: string buffer?
    _rp.c_KindToString(kind,s1,s2)
    return s1[0]+s2[0]


# =========================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
