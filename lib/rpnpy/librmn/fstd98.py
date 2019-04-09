#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn.fstd98 contains python wrapper to main librmn's fstd98,
convip C functions along with helper functions
"""

import os
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from rpnpy.librmn import proto as _rp
from rpnpy.librmn import const as _rc
from rpnpy.librmn import base as _rb
from rpnpy.librmn import RMNError

#---- helpers -------------------------------------------------------

_C_MKSTR = _ct.create_string_buffer
_C_MKSTR.__doc__ = 'alias to ctypes.create_string_buffer'

_C_TOINT = lambda x: (x if (type(x) != type(_ct.c_int())) else x.value)
_C_TOINT.__doc__ = 'lamda function to convert ctypes.c_int to python int'

_IS_LIST = lambda x: isinstance(x, (list, tuple))
_IS_LIST.__doc__ = 'lambda function to test if x is list or tuple'

_linkedUnits = {}

class FSTDError(RMNError):
    """
    General librmn.fstd98 module error/exception

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    >>>    #... an fst98 operation ...
    >>> except(rmn.FSTDError):
    >>>    pass #ignore the error
    >>> #...
    >>> raise rmn.FSTDError()

    See Also:
       rpnpy.librmn.RMNError
    """
    pass


def dtype_fst2numpy(datyp, nbits=None):
    """
    Return the numpy dtype datyp for the given fst datyp

    Args:
        fst_datyp : RPN fst data type code (int)

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
    Returns:
        numpy.dtype
    Raises:
        TypeError on wrong input arg types
        FSTDError if no corresponding type found

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> fst_datyp   = rmn.FST_DATYP_LIST['float_IEEE_compressed']
    >>> numpy_dtype = rmn.dtype_fst2numpy(fst_datyp)

    See Also:
       dtype_numpy2fst
       rpnpy.librmn.const
       FSTDError
    """
    if not (type(datyp) == int):
        raise TypeError("dtype_fst2numpy: Expecting arg of type int, Got %s" %
                        (type(datyp)))
    datyp = (datyp-128 if datyp>=128 else datyp)
    datyp = (datyp-64 if datyp>=64 else datyp)
    try:
        if nbits == 64:
            return _rc.FST_DATYP2NUMPY_LIST64[datyp]
        else:
            return _rc.FST_DATYP2NUMPY_LIST[datyp]
    except:
        raise FSTDError()


def dtype_numpy2fst(npdtype, compress=True, missing=False):
    """
    Return the fst datyp for the given numpy dtype
    
    Optionally specify compression and missing value options.

    Args:
        numpy_dtype : numpy data type
        compress    : define fst data type with 2nd stage compression
        missing     : define fst data type with missing values
    Returns:
        int, fst data type
        0 if no corresponding data type found
    Raises:
        TypeError on wrong input arg types
        
    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> import numpy as np
    >>> numpy_dtype = np.float32
    >>> fst_datyp = rmn.dtype_numpy2fst(numpy_dtype)
    >>> fst_datyp = rmn.dtype_numpy2fst(numpy_dtype, compress=False)
    >>> fst_datyp = rmn.dtype_numpy2fst(numpy_dtype, missing=True)
    >>> fst_datyp = rmn.dtype_numpy2fst(numpy_dtype, compress=True, missing=True)

    See Also:
       dtype_fst2numpy
       rpnpy.librmn.const
       FSTDError
    """
    if not (type(npdtype) == _np.dtype or type(npdtype) == type):
        raise TypeError("dtype_numpy2fst: Expecting arg of type %s, Got %s" %
                        ('numpy.dtype', type(npdtype)))
    datyp = 0 #default returned type: binary
    for (i, dtype) in _rc.FST_DATYP2NUMPY_LIST.items() + \
        _rc.FST_DATYP2NUMPY_LIST64.items():
        if dtype == npdtype:
            datyp = i
            break
    #TODO: should we force nbits to 64 when 64 bits type?
    if compress:
        datyp += 128
    if missing:
        datyp += 64
    return datyp


def isFST(filename):
    """
    Return True if file is of RPN STD RND type
    
    Args:
        filename : path/name of the file to examine (str)
    Returns:
        True or False
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> isfst = rmn.isFST(filename)

    See Also:
       rpnpy.librmn.base.wkoffit
    """
    if not (type(filename) == str):
        raise TypeError("isFST: Expecting arg of type str, Got %s" %
                        (type(filename)))
    if filename.strip() == '':
        raise ValueError("isFST: must provide a valid filename")
    return _rb.wkoffit(filename) in \
        (_rc.WKOFFIT_TYPE_LIST['STANDARD RANDOM 89'],
         _rc.WKOFFIT_TYPE_LIST['STANDARD RANDOM 98'])
        

def fstopenall(paths, filemode=_rc.FST_RO, verbose=None):
    """
    Open all fstfiles found in path.
    Shortcut for fnom+fstouv+fstlnk

    Args:
        paths    : path/name of the file to open
                   if paths is a list, open+link all files
                   if path is a dir, open+link all fst files in dir
        filemode : a string with the desired filemode (see librmn doc)
                   or one of these constants: FST_RW, FST_RW_OLD, FST_RO
    Returns:
        int, file unit number associated with provided path
        None in ReadOnly mode if no FST file was found in path
    Raises:
        TypeError  on wrong input arg types    
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> funit1 = rmn.fstopenall(filename)
    >>> funit2 = rmn.fstopenall('newfile.fst', rmn.FST_RW)
    >>> #...
    >>> rmn.fstcloseall(funit1)
    >>> rmn.fstcloseall(funit2)
    >>> os.unlink('newfile.fst')  # Remove test file
    
    See Also:
       fstouv
       fstlnk
       fstcloseall
       rpnpy.librmn.base.fnom
       rpnpy.librmn.const
       FSTDError
    """
    #TODO: accepte pattern, use glob
    paths = [paths] if type(paths) == str else paths
    if not (type(paths) in (list, tuple)):
        raise TypeError("fstopenall: Expecting arg of type list, Got %s" %
                        (type(paths)))
    filelist = []
    for mypath in paths:
        if not (type(mypath) == str):
            raise TypeError("fstopenall: Expecting arg of type str, Got %s" %
                            (type(mypath)))
        if mypath.strip() == '':
            raise ValueError("fstopenall: must provide a valid path")
        if os.path.isdir(mypath):
            for paths_dirs_files in os.walk(mypath):
                for myfile in paths_dirs_files[2]:
                    if isFST(os.path.join(mypath, myfile)):
                        if verbose:
                            print "(fstopenall) Found FST file: %s" % \
                                (os.path.join(mypath, myfile))
                        filelist.append(os.path.join(mypath, myfile))
                    elif verbose:
                        print "(fstopenall) Ignoring non FST file: %s" % \
                            (os.path.join(mypath, myfile))
                break
        else:
            filelist.append(mypath)
    if filemode != _rc.FST_RO and len(paths) > 1:
        return None #TODO: print error msg
    iunitlist = []
    for myfile in filelist:
        funit = None
        try:
            if os.path.isfile(myfile):
                if isFST(myfile):
                    funit = _rb.fnom(myfile, filemode)
                elif verbose:
                    print "(fstopenall) Not a RPNSTD file: " + myfile
            elif filemode in (_rc.FST_RW_OLD, _rc.FST_RO):
                if verbose:
                    print "(fstopenall) File not found: " + myfile
            else:
                funit = _rb.fnom(myfile, filemode)
        except:
            pass
        if funit:
            try:
                fstouv(funit, filemode)
                iunitlist.append(funit)
                if verbose:
                    print "(fstopenall) Opening: " + myfile, funit
            except:
                if verbose:
                    print "(fstopenall) Problem Opening: " + myfile
        elif verbose:
            print "(fstopenall) Problem Opening: " + myfile
    if len(iunitlist) == 0:
        raise FSTDError("fstopenall: unable to open any file in path %s" %
                        (str(paths)))
    if len(iunitlist) == 1:
        return iunitlist[0]
    _linkedUnits[str(iunitlist[0])] = iunitlist
    return fstlnk(iunitlist)


def fstcloseall(iunit):
    """
    Close all files associated with provided file unit number.
    Shortcut for fclos+fstfrm

    Args:
        iunit    : unit number associated to the file
                   obtained with fnom or fstopenall
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
        
    Examples:
    >>> import rpnpy.librmn.all as rmn    
    >>> funit = rmn.fstopenall('mynewfile.fst', rmn.FST_RW)
    >>> #...
    >>> rmn.fstcloseall(funit)
    >>> os.unlink('mynewfile.fst')  # Remove test file

    See Also:
       fstfrm
       fstopenall
       rpnpy.librmn.base.fclos
       rpnpy.librmn.const
       FSTDError
    """
    if not (type(iunit) == int):
        raise TypeError("fstcloseall: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstcloseall: must provide a valid iunit: %d" %
                         (iunit))    
    try:
        iunitlist = _linkedUnits[str(iunit)]
    except KeyError:
        iunitlist = (iunit,)
    istat = 0
    for iunit in iunitlist:
        try:
            fstfrm(iunit)
            istat =_rb.fclos(iunit)
        except:
            istat = -1
    try:
        del _linkedUnits[str(iunitlist[0])]
    except:
        pass
    if istat >= 0:
        return
    raise FSTDError("fstcloseall: Unable to properly close unit %d" % (iunit))


def listToFLOATIP(rp1):
    """
    Encode values in FLOAT_IP type/struct
    
    floatip = listToFLOATIP(rp1)

    Args:
        rp1 : (value, kind) or (value1, value2, kind) (list or tuple)
              kind is one of FSTD ip accepted kind
    Returns:
        FLOAT_IP
    Raises:
        TypeError on wrong input arg types

    Examples:
    >>> import rpnpy.librmn.all as rmn    
    >>> pk1 = rmn.listToFLOATIP((500., 500., rmn.KIND_PRESSURE))
    >>> pk2 = rmn.listToFLOATIP((0.,     0., rmn.KIND_HOURS))
    >>> pk3 = rmn.listToFLOATIP((0.,     0., 0))
    >>> (ip1, ip2, ip3) = rmn.convertPKtoIP(pk1, pk2, pk3)

    See Also:
        FLOATIPtoList
        convertPKtoIP
        rpnpy.librmn.proto.FLOAT_IP
        rpnpy.librmn.const
    """
    if isinstance(rp1, _rp.FLOAT_IP):
        return rp1
    if not _IS_LIST(rp1):
        raise TypeError
    if not len(rp1) in (2, 3):
        raise TypeError()
    if len(rp1) == 2:
        return _rp.FLOAT_IP(rp1[0], rp1[0], rp1[1])
    return _rp.FLOAT_IP(rp1[0], rp1[1], rp1[2])

    
def FLOATIPtoList(rp1):
    """
    Decode values from FLOAT_IP type/struct
    
    (v1, v2, kind) = FLOATIPtoList(rp1)
    
    Args:
        rp1 : encoded FLOAT_IP
    Returns:
        v1 : level 1st value (float)
        v2 : level 2nd value (float)
             v2=v1 if not a range
        kind: level kind (int), one of FSTD ip accepted kind

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> pk1 = rmn.listToFLOATIP((500., 500., rmn.KIND_PRESSURE))
    >>> pk2 = rmn.listToFLOATIP((0.,     0., rmn.KIND_HOURS))
    >>> pk3 = rmn.listToFLOATIP((0.,     0., 0))
    >>> (ip1, ip2, ip3) = rmn.convertPKtoIP(pk1, pk2, pk3)
    >>> (rp1, rp2, rp3) = rmn.convertIPtoPK(ip1, ip2, ip3)
    >>> (v1, v2, kind)  = rmn.FLOATIPtoList(rp1)
    
    See Also:
        listToFLOATIP
        convertPKtoIP
        convertIPtoPK
        rpnpy.librmn.proto.FLOAT_IP
        rpnpy.librmn.const
    """
    if isinstance(rp1, _rp.FLOAT_IP):
        return (rp1.v1, rp1.v2, rp1.kind)
    return rp1
    
    
#--- fstd98 ---------------------------------------------------------

def fstecr(iunit, data, meta=None, rewrite=True):
    """
    Writes record to file previously opened with fnom+fstouv
    
    fstecr(iunit, data, meta)
    fstecr(iunit, data, meta, rewrite=True)
    fstecr(iunit, rec)
    
    Args:
        iunit : file unit number (int)
        data  : data to be written (numpy.ndarray, FORTRAN order)
        meta  : associated metadata (dict)
                Not specified meta params will be set to their default value
                as in FST_RDE_META_DEFAULT
                The list of known parameters is the same a the one returned
                by fstprm, see fstprm doc for details
                You can force meta['datyp'] but for the 
                sake of data/meta consistency, it is best not to specify
                meta['datyp'], data.dtype will be used instead
        rec   : data + meta in a dict
                Option to provide data+meta in a single dict
                where data = rec['d']
                Note: new option in version 2.0.rc1
        rewrite : force to overwrite any other fields with same meta 
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> funitIn  = rmn.fstopenall(filename)
    >>> funitOut = rmn.fstopenall('newfile.fst', rmn.FST_RW)
    >>> myrec = rmn.fstlir(funitIn, nomvar='P0')
    >>> 
    >>> # Write the record specifying data and meta separately
    >>> rmn.fstecr(funitOut, myrec['d'], myrec)
    >>> 
    >>> # Write the record specifying data and meta together
    >>> rmn.fstecr(funitOut, myrec)
    >>> 
    >>> # Properly close files, important when writing to avoid corrupted files
    >>> rmn.fstcloseall(funitOut)
    >>> rmn.fstcloseall(funitIn)

    See Also:
        fstopenall
        fstcloseall
        fstlir
        fstprm
        fstluk
        rpnpy.librmn.const
    """
    if not (type(iunit) == int):
        raise TypeError("fstecr: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstecr: must provide a valid iunit: %d" % (iunit))
    if isinstance(data, dict):
        meta0 = data
        data = meta0['d']
        if meta:
            meta0.update(meta)
        meta = meta0
    if not (type(data) == _np.ndarray and isinstance(meta, dict)):
        raise TypeError("fstecr: Expecting args of type %s, %s, Got %s, %s" %
                        ('numpy.ndarray', 'dict', type(data), type(meta)))
    if not data.flags['F_CONTIGUOUS']:
        raise TypeError("fstecr: Expecting data type " +
                        "numpy.ndarray with FORTRAN order")
    #TODO: check if file is open with write permission
    meta2 = _rc.FST_RDE_META_DEFAULT.copy()
    for k in _rc.FST_RDE_META_DEFAULT.keys():
        try:
            if k in meta.keys() and meta[k] not in ('', ' ', -1):
                meta2[k] = meta[k]
        except:
            sys.stderr.write("fstecr error, skipping copy of: %s\n" % str(k))
    datyp = dtype_numpy2fst(data.dtype)
    try:
        if meta['datyp'] >= 0:
            datyp = meta['datyp']
    except KeyError:
        pass
    irewrite = (1 if rewrite else 0)
    npak     = -abs(meta2['nbits'])
    _rp.c_fstecr.argtypes = (
        _npc.ndpointer(dtype=data.dtype), _npc.ndpointer(dtype=data.dtype),
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.c_char_p, _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int)
    #TODO: what if data not 32 bits? copy to 32bits field or modify nijk? 
    if not data.flags['F_CONTIGUOUS']:
        data = _np.asfortranarray(data, dtype=data.dtype)
    istat = _rp.c_fstecr(data, data, npak, iunit,
                meta2['dateo'], meta2['deet'], meta2['npas'],
                meta2['ni'],    meta2['nj'],   meta2['nk'],
                meta2['ip1'],   meta2['ip2'],  meta2['ip3'],
                meta2['typvar'], meta2['nomvar'], meta2['etiket'],
                meta2['grtyp'],
                meta2['ig1'],   meta2['ig2'],  meta2['ig3'], meta2['ig4'],
                datyp, irewrite)
    if istat >= 0:
        return
    raise FSTDError()


def fst_edit_dir(key, datev=-1, dateo=-1, deet=-1, npas=-1, ni=-1, nj=-1, nk=-1,
                 ip1=-1, ip2=-1, ip3=-1,
                 typvar=' ', nomvar=' ', etiket=' ', grtyp=' ',
                 ig1=-1, ig2=-1, ig3=-1, ig4=-1, datyp=-1, keep_dateo=False):
    """
    Edits the directory content of a RPN standard file
    Only provided parameters with value different than default are updated
    
    Note: by default datev is kept constant unless
          dateo is specified or
          keep_dateo=True

    fst_edit_dir(key, ... )
    fst_edit_dir(rec, ... )
    fst_edit_dir(keylist, ... )
  
    Args:
        key   : positioning information to the record,
                obtained with fstinf or fstinl, ...
        dateo : date of origin (date time stamp), cannot change dateo and datev
        datev : valid date     (date time stamp), cannot change dateo and datev
        deet  : length of a time step in seconds
                (datev constant unless keep_dateo)
        npas  : time step number (datev constant unless keep_dateo)
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
        keep_dateo : by default datev is kept constant unless
                     dateo is specified or
                     keep_dateo=True
                     (keep_dateo must be False is datev is provided)

        From verion 2.0.rc1, this function can be called with a rec meta
        or keylist instead of a simple key number (int):

        rec     (dict) : dictionary where key = rec['key']
        keylist (list) : list of keys for records to edit
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
    
    Examples:
    >>> import os, os.path, stat, shutil
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Copy a file locally to be able to edit it and set write permission
    >>> filename  = 'geophy.fst'
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename0 = os.path.join(ATM_MODEL_DFILES,'bcmk',filename)
    >>> shutil.copyfile(filename0, filename)
    >>> st = os.stat(filename)
    >>> os.chmod(filename, st.st_mode | stat.S_IWRITE)
    >>> 
    >>> # Open existing file in Rear/Write mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RW_OLD)
    >>> 
    >>> # Get the list of all records in file and change the etiket for them all
    >>> mykeylist = rmn.fstinl(funit)
    >>> 
    >>> # Iterate explicitely on list of records to change the etiket
    >>> for key in mykeylist: rmn.fst_edit_dir(key, etiket='MY_NEW_ETK')
    >>> 
    >>> # Could also be written as a one liner list comprenhension:
    >>> # [rmn.fst_edit_dir(key, etiket='MY_NEW_ETK') for key in rmn.fstinl(funit)]
    >>> 
    >>> # Iterate implicitely on list of records to change the etiket
    >>> rmn.fst_edit_dir(mykeylist, etiket='MY_NEW_ETK')
    >>> 
    >>> # Properly close files, important when editing to avoid corrupted files
    >>> rmn.fstcloseall(funit)

    See Also:
        fstopenall
        fstcloseall
        fstinl
        fstinf
        rpnpy.librmn.const
    """
    if datev != -1:
        if dateo != -1:
            raise FSTDError("fst_edit_dir: Cannot change dateo and datev " +
                            "simultaneously, try using npas or deet to " +
                            "change the other value")
        if keep_dateo:
            raise FSTDError("fst_edit_dir: Cannot change datev while " +
                            "keeping dateo unchanged, try using npas or " +
                            "deet to change datev instead")
    if isinstance(key, dict):
       key = key['key']
    
    if isinstance(key, (list, tuple)):
        for key2 in key:
            fst_edit_dir(key2, datev, dateo, deet, npas, ni, nj, nk,
                         ip1, ip2, ip3,
                         typvar, nomvar, etiket, grtyp,
                         ig1, ig2, ig3, ig4, datyp, keep_dateo)
        return
    
    if key < 0:
        raise ValueError("fst_edit_dir: must provide a valid record key: %d" % (key))
    if dateo != -1:
        recparams = fstprm(key)
        deet1 = recparams['deet'] if deet == -1 else deet
        npas1 = recparams['npas'] if npas == -1 else npas
        if deet1 == 0 or npas1 == 0 or dateo == 0:
            datev = dateo
        else:
            try: 
                datev = _rb.incdatr(dateo, deet1*npas1/3600.)
            except:
                raise FSTDError('fst_edit_dir: ' +
                                'error computing datev to set dateo')
    elif keep_dateo and (npas != -1 or deet != -1):
        recparams = fstprm(key)
        if recparams['dateo'] == 0:
            datev = 0
        else:
            deet1 = recparams['deet'] if deet == -1 else deet
            npas1 = recparams['npas'] if npas == -1 else npas
            try:
                datev = _rb.incdatr(recparams['dateo'], deet1*npas1/3600.)
            except:
                raise FSTDError('fst_edit_dir: ' +
                                'error computing datev to keep_dateo')
    istat = _rp.c_fst_edit_dir(key, datev, deet, npas, ni, nj, nk,
                 ip1, ip2, ip3, typvar, nomvar, etiket, grtyp,
                 ig1, ig2, ig3, ig4, datyp)
    if istat >= 0:
        return
    raise FSTDError()


def fsteff(key):
    """
    Deletes the record associated to handle.

    fsteff(key)
    fsteff(rec)
    fsteff(keylist)

    Args:
        key   : positioning information to the record,
                obtained with fstinf or fstinl, ...

        From verion 2.0.rc1, this function can be called with a rec meta
        or keylist instead of a simple key number (int):

        rec     (dict) : dictionary where key = rec['key']
        keylist (list) : list of keys for records to edit
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path, stat, shutil
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Copy a file locally to be able to edit it and set write permission
    >>> filename  = 'geophy.fst'
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename0 = os.path.join(ATM_MODEL_DFILES,'bcmk',filename)
    >>> shutil.copyfile(filename0, filename)
    >>> st = os.stat(filename)
    >>> os.chmod(filename, st.st_mode | stat.S_IWRITE)
    >>> 
    >>> # Open existing file in Rear/Write mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RW_OLD)
    >>> 
    >>> # Find the record name ME and erase it from the file
    >>> key = rmn.fstinf(funit, nomvar='ME')
    >>> rmn.fsteff(key['key'])
    >>> 
    >>> # Find the record name ME and erase it from the file,
    >>> # passing directly the dict returned by fstinf
    >>> key = rmn.fstinf(funit, nomvar='MG')
    >>> rmn.fsteff(key)
    >>> 
    >>> # Find all record named VF and erase them
    >>> keylist = rmn.fstinl(funit, nomvar='VF')
    >>> rmn.fsteff(keylist)

    >>> rmn.fstcloseall(funit)
    >>> os.unlink(filename)  #Remove test file

    See Also:
        fstopenall
        fstcloseall
        fstinl
        fstinf
        rpnpy.librmn.const
    """
    if isinstance(key, dict):
       key = key['key']
    
    if isinstance(key, (list, tuple)):
        for key2 in key:
            fsteff(key2)
        return
    
    if not (type(key) == int):
        raise TypeError("fsteff: Expecting arg of type int, Got %s" %
                        (type(key)))
    if key < 0:
        raise ValueError("fsteff: must provide a valid record key: %d" %
                         (key))
    istat = _rp.c_fsteff(key)
    if istat >= 0:
        return
    raise FSTDError()


def fstfrm(iunit):
    """
    Close a RPN standard file

    fstfrm(iunit)

    Args:
        iunit    : unit number associated to the file
                   obtained with fnom+fstouv
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myfstfile.fst', rmn.FST_RW)
    >>> istat = rmn.fstouv(funit, rmn.FST_RW)
    >>> #...
    >>> istat = rmn.fstfrm(funit)
    >>> istat = rmn.fclos(funit)
    >>> os.unlink('myfstfile.fst')  # Remove test file

    See Also:
        fstouv
        fstopenall
        fstcloseall
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
    """
    if not (type(iunit) == int):
        raise TypeError("fstfrm: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstfrm: must provide a valid iunit: %d" % (iunit))
    istat = _rp.c_fstfrm(iunit)
    if istat >= 0:
        return
    raise FSTDError()


def fstinf(iunit, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
           typvar=' ', nomvar=' '):
    """
    Locate the next record that matches the research keys

    Returns the key of the first record (only one) match the
    selection criteria.
    
    Only provided parameters with value different than default
    are used as selection criteria.
    
    Thus if you do not provide any other parameter that iunit, fstinf
    will return the key to the first record in the file.
    
    recmatch = fstinf(iunit, ... )
    
    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        datev   : valid date
        etiket  : label
        ip1     : vertical level
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
    Returns:
        None if no matching record, else:
        {
            'key'   : key,       # key/handle of the 1st matching record
            'shape' : (ni, nj, nk) # dimensions of the field
         }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find the record named P0 and read its metadata
    >>> key    = rmn.fstinf(funit, nomvar='P0')
    >>> p0meta = rmn.fstprm(key['key'])
    >>> 
    >>> rmn.fstcloseall(funit)

    See Also:
        fstinfx
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    return fstinfx(-2, iunit, datev, etiket, ip1, ip2, ip3, typvar, nomvar)


def fstinfx(key, iunit, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
            typvar=' ', nomvar=' '):
    """
    Locate the next record that matches the research keys
    
    Only provided parameters with value different than default
    are used as selection criteria
    The search begins at the position given by key/handle
    obtained with fstinf or fstinl, ...

    recmatch = fstinfx(key, iunit, ... )

    Args:
        key     : record key/handle of the search start position
                  (int or dict) if dict, must have key['key']
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        datev   : valid date
        etiket  : label
        ip1     : vertical level
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
    Returns:
        None if no matching record, else:
        {
            'key'   : key,       # key/handle of the 1st matching record
            'shape' : (ni, nj, nk) # dimensions of the field
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find the 1st record named P0 then the one follwoing it
    >>> # and read its metadata
    >>> key1   = rmn.fstinf(funit, nomvar='P0')
    >>> key2   = rmn.fstinfx(key1, funit, nomvar='P0')
    >>> p0meta = rmn.fstprm(key2)
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstinf
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    if isinstance(key, dict):
       key = key['key']
    
    if not (type(iunit) == int):
        raise TypeError("fstinfx: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstinfx: must provide a valid iunit: %d" % (iunit))
    if not (type(key) == int):
        raise TypeError("fstinfx: Expecting arg of type int, Got %s" %
                        (type(key)))
    (cni, cnj, cnk) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    key2 = _rp.c_fstinfx(key, iunit, _ct.byref(cni), _ct.byref(cnj),
                         _ct.byref(cnk), datev, etiket, ip1, ip2, ip3,
                         typvar, nomvar)
    ## key2 = _C_TOINT(key2)
    if key2 < 0:
        return None
    ## fx = lambda x: (x.value if x.value>0 else 1)
    return {
        'key'   : key2 ,
        'shape' : (max(1, cni.value), max(1, cnj.value), max(1, cnk.value)),
        }


def fstinl(iunit, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
           typvar=' ', nomvar=' ', nrecmax=-1):
    """
    Locate all the record matching the research keys
    
    Only provided parameters with value different than default
    are used as selection criteria
        
    recmatchlist = fstinl(iunit, ... )
    
    Args:
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
    Returns:
        list of matching records keys
        empty list ([]) if no matching record found
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find all records named VF and print the ip1 of the first 3
    >>> keylist = rmn.fstinl(funit, nomvar='VF')
    >>> for key in keylist[0:3]: print("# VF ip1=%s" % rmn.fstprm(key)['ip1'])
    # VF ip1=1199
    # VF ip1=1198
    # VF ip1=1197
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstinf
        fstinfx
        fstprm
        fstluk
        fstopenall
        fstcloseall    
    """
    if not (type(iunit) == int):
        raise TypeError("fstinl: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstinl: must provide a valid iunit: %d" % (iunit))
    if nrecmax <= 0:
        try:
            iunitlist = _linkedUnits[str(iunit)]
        except KeyError:
            iunitlist = (iunit,)
        nrecmax = 0
        for iunit1 in iunitlist:
            nrecmax += _rp.c_fstnbrv(iunit1)
    creclist = _np.empty(nrecmax, dtype=_np.intc)
    (cni, cnj, cnk) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    cnfound         = _ct.c_int()
    istat = _rp.c_fstinl(iunit, _ct.byref(cni), _ct.byref(cnj), _ct.byref(cnk),
                         datev, etiket, ip1, ip2, ip3, typvar, nomvar,
                         creclist, cnfound, nrecmax)
    ## if istat < 0:
    ##     raise FSTDError('fstinl: Problem searching record list')
    if cnfound.value <= 0:
        return []
    return creclist[0:cnfound.value].tolist()


## Note: fstlic not very usefull, provide better python implementation
## def fstlic(iunit, ni=-1, nj=-1, nk=-1, datev=-1, etiket=' ',
##            ip1=-1, ip2=-1, ip3=-1, typvar=' ', nomvar=' ',
##            ig1=-1, ig2=-1, ig3=-1, ig4=-1, grtyp=' ',
##            dtype=None, rank=None):
##     """Search for a record that matches the research keys and
##     check that the remaining parmeters match the record descriptors
##     
##     iunit   : unit number associated to the file
##               obtained with fnom+fstouv
##     ni, nj, nk: filter fields with field dims
##     datev   : valid date
##     etiket  : label
##     ip1     : vertical level
##     ip2     : forecast hour
##     ip3     : user defined identifier
##     typvar  : type of field
##     nomvar  : variable name
##     ig1, ig2, ig3, ig4: filter fields with field's ig1-4
##     grtyp   : grid type
##     dtype : array type of the returned data
##             Default is determined from records' datyp
##             Could be any numpy.ndarray type
##             See: http://docs.scipy.org/doc/numpy/user/basics.types.html
##     rank  : try to return an array with the specified rank
##    
##     return record data as a numpy.ndarray
##     return None on error    
##     """
##     if type(iunit) != int:
##        raise TypeError("fstluk: Expecting a iunit of type int, " +
##                        "Got %s : %s" % (type(iunit), repr(iunit)))


def fstlir(iunit, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
           typvar=' ', nomvar=' ', dtype=None, rank=None, dataArray=None):
    """
    Reads the next record that matches the research keys
    
    Only provided parameters with value different than default
    are used as selection criteria

    record = fstlir(iunit, ... )
    
    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        datev   : valid date
        etiket  : label
        ip1     : vertical level
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
        dtype   : array type of the returned data
                  Default is determined from records' datyp
                  Could be any numpy.ndarray type
                  See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank    : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data 
    Returns:
        None if no matching record, else:
        {
            'd'   : data,       # record data as a numpy.ndarray
            ...                 # same params list as fstprm
        }
     Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error       
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find and read p0 meta and data, then print its min,max,mean values
    >>> p0rec = rmn.fstlir(funit, nomvar='P0')
    >>> print("# P0 ip2=%s min=%f max=%f avg=%f" % \
              (p0rec['ip2'], p0rec['d'].min(), p0rec['d'].max(), p0rec['d'].mean()))
    # P0 ip2=0 min=530.641418 max=1039.641479 avg=966.500000
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstlis
        fstlirx
        fstinf
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    key = -2
    return fstlirx(key, iunit, datev, etiket, ip1, ip2, ip3,
                   typvar, nomvar, dtype, rank, dataArray)


def fstlirx(key, iunit, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
            typvar=' ', nomvar=' ', dtype=None, rank=None, dataArray=None):
    """
    Reads the next record that matches the research keys
    
    Only provided parameters with value different than default
    are used as selection criteria
    The search begins right after at the position given by record key/handle.

    record = fstlirx(key, iunit, ... )

    Args:
        key     : key/handle from where to start the search
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        datev   : valid date
        etiket  : label
        ip1     : vertical level
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
        dtype   : array type of the returned data
                  Default is determined from records' datyp
                  Could be any numpy.ndarray type
                  See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank    : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data 
    Returns:
        None if no matching record, else:
        {
            'd'   : data,       # record data as a numpy.ndarray
            ...                 # same params list as fstprm
        }
     Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error       
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find and read the 2nd p0 meta and data,
    >>> # then print its min,max,mean values
    >>> key1  = rmn.fstinf(funit, nomvar='P0')
    >>> p0rec = rmn.fstlirx(key1, funit, nomvar='P0')
    >>> print("# P0 ip2=%s min=%f max=%f avg=%f" % \
              (p0rec['ip2'], p0rec['d'].min(), p0rec['d'].max(), p0rec['d'].mean()))
    # P0 ip2=12 min=530.958008 max=1037.958008 avg=966.373600
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstlis
        fstlir
        fstinf
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    key2 = fstinfx(key, iunit, datev, etiket, ip1, ip2, ip3, typvar, nomvar)
    if (key2):
        return fstluk(key2['key'], dtype, rank, dataArray)
    return None


def fstlis(iunit, dtype=None, rank=None, dataArray=None):
    """
    Reads the next record that matches the research criterias

    record = fstlis(iunit, ... )

    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        dtype   : array type of the returned data
                  Default is determined from records' datyp
                  Could be any numpy.ndarray type
                  See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank    : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data 
    Returns:
        None if no matching record, else:
        {
            'd'   : data,       # record data as a numpy.ndarray
            ...                 # same params list as fstprm
        }
     Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error       
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find and read the 2nd p0 meta and data,
    >>> # then print its min,max,mean values
    >>> key1  = rmn.fstinf(funit, nomvar='P0')
    >>> p0rec = rmn.fstlis(funit)
    >>> print("# P0 ip2=%s min=%f max=%f avg=%f" % \
              (p0rec['ip2'], p0rec['d'].min(), p0rec['d'].max(), p0rec['d'].mean()))
    # P0 ip2=12 min=530.958008 max=1037.958008 avg=966.373600
    >>>    
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstlir
        fstlirx
        fstinf
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    key = fstsui(iunit)
    if (key):
        return fstluk(key['key'], dtype, rank, dataArray)
    return None


def fstlnk(unitList):
    """
    Links a list of files together for search purpose

    funit = fstlnk(unitList)

    Args:
        unitList : list of previously opened (fnom+fstouv) file units
                   (list or tuple)
    Returns:
        File unit for the grouped unit
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error       
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Open several files
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename1 = os.path.join(ATM_MODEL_DFILES,'bcmk','2009042700_000')
    >>> funit1 = rmn.fnom(filename1, rmn.FST_RO)
    >>> istat  = rmn.fstouv(funit1, rmn.FST_RO)
    >>> filename2 = os.path.join(ATM_MODEL_DFILES,'bcmk','2009042700_012')
    >>> funit2 = rmn.fnom(filename2, rmn.FST_RO)
    >>> istat  = rmn.fstouv(funit2, rmn.FST_RO)
    >>> 
    >>> # Link the file as one
    >>> funit = rmn.fstlnk((funit1, funit2))
    >>> 
    >>> # Use the linked files
    >>> for key in rmn.fstinl(funit, nomvar='P0'): print("# P0 ip2=%s" % rmn.fstprm(key)['ip2'])
    # P0 ip2=0
    # P0 ip2=12
    >>> 
    >>> # Close all linked files
    >>> istat = rmn.fstfrm(funit1)
    >>> istat = rmn.fclos(funit1)
    >>> istat = rmn.fstfrm(funit2)
    >>> istat = rmn.fclos(funit2)
    
    See Also:
        fstopenall
        fstcloseall
        fstouv
        fstfrm
        rpnpy.librmn.base.fclos
        rpnpy.librmn.const
    """
    nfilesmax = 999
    if type(unitList) == int:
        unitList = [unitList]
    if not (type(unitList) in (list, tuple)):
        raise TypeError("fstlnk: Expecting arg of type list, Got %s" %
                        (type(unitList)))
    if len(unitList)<1 or min(unitList)<=0:
        raise ValueError("fstlnk: must provide a valid iunit: %d" %
                         (min(unitList)))
    if len(unitList) > nfilesmax: #TODO: check this limit
        raise ValueError("fstlnk: Too many files (max %d): %d" %
                         (nfilesmax,len(unitList)))
    cunitList = _np.asfortranarray(unitList, dtype=_np.intc)
    ## istat = _rp.c_xdflnk(cunitList, len(cunitList))
    cnunits = _ct.c_int(len(cunitList))
    istat = _rp.f_fstlnk(cunitList, _ct.byref(cnunits))
    if istat >= 0:
        return unitList[0]
    raise FSTDError()


def fstluk(key, dtype=None, rank=None, dataArray=None):
    """
    Read the record at position given by key/handle

    record = fstluk(key)
    record = fstluk(key, dtype)
    record = fstluk(key, dtype, rank)

    Args:
        key   : positioning information to the record,
                obtained with fstinf or fstinl, ...
        dtype : array type of the returned data
                Default is determined from records' datyp
                Could be any numpy.ndarray type
                See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank  : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data 
    Returns:
        {
            'd'   : data,       # record data as a numpy.ndarray, FORTRAN order
            ...                 # same params list as fstprm
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error
        
    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find record named P0 and read it meta + data
    >>> # then print its min,max,mean values
    >>> key   = rmn.fstinf(funit, nomvar='P0')
    >>> p0rec = rmn.fstluk(key)
    >>> print("# P0 ip2=%s min=%f max=%f avg=%f" % \
              (p0rec['ip2'], p0rec['d'].min(), p0rec['d'].max(), p0rec['d'].mean()))
    # P0 ip2=0 min=530.641418 max=1039.641479 avg=966.500000
    >>> rmn.fstcloseall(funit)

    See Also:
        fstprm
        fstlir
        fstinf
        fstinl
        fstopenall
        fstcloseall
    """
    if isinstance(key, dict):
       key = key['key']
    if type(key) != int:
        raise TypeError("fstluk: Expecting a key of type int, Got %s : %s" %
                        (type(key), repr(key)))
    if key < 0:
        raise ValueError("fstluk: must provide a valid key: %d" % (key))
    params = fstprm(key)
    if params is None:
        raise FSTDError()
    (cni, cnj, cnk) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    if dtype is None:
        dtype = dtype_fst2numpy(params['datyp'], params['nbits'])
    _rp.c_fstluk.argtypes = (_npc.ndpointer(dtype=dtype), _ct.c_int,
                             _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                             _ct.POINTER(_ct.c_int))
    wantrank = 1 if rank is None else rank
    minrank = 3
    if params['shape'][2] <= 1:
        minrank = 1 if params['shape'][1] <= 1 else 2
    rank = max(1, max(minrank, wantrank))
    myshape = [1] * rank
    maxrank = min(rank, len(params['shape']))
    myshape[0:maxrank] = params['shape'][0:maxrank]
    params['shape'] = myshape
    if dataArray is None:
        data = _np.empty(params['shape'], dtype=dtype, order='FORTRAN')
    elif isinstance(dataArray,_np.ndarray):
        if not dataArray.flags['F_CONTIGUOUS']:
            raise TypeError('Provided dataArray should be F_CONTIGUOUS')
        if dtype != dataArray.dtype:
            raise TypeError('Expecting dataArray of type %s, got: %s' % (repr(dtype),repr(dataArray.dtype)))
        shape0 = [1,1,1] ; shape0[0:len(params['shape'])] = params['shape'][:]
        shape1 = [1,1,1] ; shape1[0:len(dataArray.shape)] = dataArray.shape[:]
        if shape0 != shape1:
            raise TypeError('Provided have wrong shape, expecting: %s, got: %s' %(repr(params['shape']),repr(dataArray.shape)))
        data = dataArray
    else:
        raise TypeError('Expecting dataArray of type ndarray, got: %s' % repr(type(dataArray)))
    istat = _rp.c_fstluk(data, key, _ct.byref(cni), _ct.byref(cnj),
                         _ct.byref(cnk))
    if istat < 0:
        raise FSTDError()  
    params['d'] = data
    return params


#TODO: fstmsq

def fstnbr(iunit):
    """
    Returns the number of records of the file associated with unit

    nrec = fstnbr(iunit)
    
    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
    Returns:
        int, nb of records
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk','2009042700_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Print number of records
    >>> # then print its min,max,mean values
    >>> nrec = rmn.fstnbr(funit)
    >>> print("# There are %d records in the file" % nrec)
    # There are 1083 records in the file
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstnbrv
        fstopenall
        fstcloseall
    """
    if not (type(iunit) == int):
        raise TypeError("fstnbr: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstnbr: must provide a valid iunit: %d" % (iunit))
    nrec = _rp.c_fstnbr(iunit)
    if nrec < 0:
        raise FSTDError()        
    return nrec


def fstnbrv(iunit):
    """
    Returns the number of valid records (excluding deleted records)
    
    nrec = fstnbrv(iunit)
    
    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
    Returns:
        int, nb of valid records
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk','2009042700_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Print number of records
    >>> # then print its min,max,mean values
    >>> nrec = rmn.fstnbrv(funit)
    >>> print("# There are %d valid records in the file" % nrec)
    # There are 1083 valid records in the file
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstnbr
        fstopenall
        fstcloseall

    """
    if not (type(iunit) == int):
        raise TypeError("fstnbrv: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstnbrv: must provide a valid iunit: %d" % (iunit))
    nrec = _rp.c_fstnbrv(iunit)
    if nrec < 0:
        raise FSTDError()        
    return nrec


def fstopt(optName, optValue, setOget=_rc.FSTOP_SET):
    """
    Set or print FST option.

    fstopt(optName, optValue)
    fstopt(optName, optValue, setOget)

    Args:
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
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> # Restric to the minimum the number of messages printed by librmn
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

    See Also:
       rpnpy.librmn.const
    """
    if type(optValue) == str:
        istat = _rp.c_fstopc(optName, optValue, setOget)
    elif type(optValue) == int:
        istat = _rp.c_fstopi(optName, optValue, setOget)
    else:
        raise TypeError("fstopt: cannot set optValue of type: %s %s" %
                        (type(optValue), repr(optValue)))
    if istat < 0:
        raise FSTDError()        
    return


def fstouv(iunit, filemode=_rc.FST_RW):
    """
    Opens a RPN standard file

    fstouv(iunit)
    fstouv(iunit, filemode)

    Args:
        iunit    : unit number associated to the file
                   obtained with fnom
        filemode : a string with the desired filemode (see librmn doc)
                   or one of these constants: FST_RW, FST_RW_OLD, FST_RO
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> funit = rmn.fnom('myfstfile.fst', rmn.FST_RW)
    >>> istat = rmn.fstouv(funit, rmn.FST_RW)
    >>> #...
    >>> istat = rmn.fstfrm(funit)
    >>> istat = rmn.fclos(funit)
    >>> os.unlink('myfstfile.fst')  # Remove test file
    
    See Also:
        fstfrm
        fstopenall
        fstcloseall
        rpnpy.librmn.base.fnom
        rpnpy.librmn.base.fclos
    """
    if not (type(iunit) == int):
        raise TypeError("fstinfx: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstinfx: must provide a valid iunit: %d" % (iunit))
    if not (type(filemode) == str):
        raise TypeError("fstinfx: Expecting arg filemode of type str, Got %s" %
                        (type(filemode)))
    istat = _rp.c_fstouv(iunit, filemode)
    if istat < 0:
        raise FSTDError()        
    return
 

def fstprm(key):
    """
    Get all the description informations of the record.

    params = fstprm(key)

    Args:
        key : positioning information to the record,
              obtained with fstinf or fstinl, ...
    Returns:
        {
            'key'   : key,       # key/handle of the record
            'shape' : (ni, nj, nk) # dimensions of the field
            'dateo' : date time stamp
            'datev' : date of validity (dateo+ deet * npas)
                      Will be set to '-1' if dateo invalid
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
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Print name, ip1, ip2 of first record in file
    >>> key  = rmn.fstinf(funit)
    >>> meta = rmn.fstprm(key['key'])
    >>> print("# %s ip1=%s ip2=%s" % (meta['nomvar'], meta['ip1'], meta['ip2']))
    # !!   ip1=0 ip2=0
    >>>
    >>> rmn.fstcloseall(funit)

    See Also:
        fstluk
        fstinf
        fstinl
        fstopenall
        fstcloseall
    """
    if isinstance(key, dict):
       key = key['key']
    if type(key) != int:
        raise TypeError("fstprm: Expecting a key of type int, Got %s : %s" %
                        (type(key), repr(key)))
    if key < 0:
        raise ValueError("fstprm: must provide a valid key: %d" % (key))
    (cni, cnj, cnk)        = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    (cdateo, cdeet, cnpas) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    (cnbits, cdatyp)       = (_ct.c_int(), _ct.c_int())
    (cip1, cip2, cip3)     = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    ctypvar                = _C_MKSTR(' '*_rc.FST_TYPVAR_LEN)
    cnomvar                = _C_MKSTR(' '*_rc.FST_NOMVAR_LEN)
    cetiket                = _C_MKSTR(' '*_rc.FST_ETIKET_LEN)
    cgrtyp                 = _C_MKSTR(' '*_rc.FST_GRTYP_LEN)
    (cig1, cig2, cig3, cig4)  = (_ct.c_int(), _ct.c_int(),
                                 _ct.c_int(), _ct.c_int())
    (cswa, clng, cdltf, cubc) = (_ct.c_int(), _ct.c_int(),
                                 _ct.c_int(), _ct.c_int())
    (cxtra1, cxtra2, cxtra3)  = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    istat = _rp.c_fstprm(
        key, _ct.byref(cdateo), _ct.byref(cdeet), _ct.byref(cnpas),
        _ct.byref(cni), _ct.byref(cnj), _ct.byref(cnk),
        _ct.byref(cnbits), _ct.byref(cdatyp),
        _ct.byref(cip1), _ct.byref(cip2), _ct.byref(cip3),
        ctypvar, cnomvar, cetiket,
        cgrtyp, _ct.byref(cig1), _ct.byref(cig2),
        _ct.byref(cig3), _ct.byref(cig4),
        _ct.byref(cswa), _ct.byref(clng), _ct.byref(cdltf), _ct.byref(cubc),
        _ct.byref(cxtra1), _ct.byref(cxtra2), _ct.byref(cxtra3))
    istat = _C_TOINT(istat)
    if istat < 0:
        raise FSTDError()
    datev = cdateo.value
    if cdateo.value != 0 and cdeet.value != 0 and cnpas.value != 0:
        try:
            datev = _rb.incdatr(cdateo.value, (cdeet.value*cnpas.value)/3600.)
        except:
            datev = -1
    return {
        'key'   : key ,
        'shape' : (max(1, cni.value), max(1, cnj.value), max(1, cnk.value)),
        'dateo' : cdateo.value,
        'datev' : datev,
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


def fstsui(iunit):
    """
    Finds the next record that matches the last search criterias

    recmatch = fstsui(iunit)

    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
    Returns:
        None if no more matching record, else
        {
            'key'   : key,       # key/handle of the next matching record
            'shape' : (ni, nj, nk) # dimensions of the field
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find the 1st record named P0 then the one follwoing it
    >>> # and read its metadata
    >>> key1 = rmn.fstinf(funit, nomvar='P0')
    >>> key2 = rmn.fstsui(funit)
    >>> meta = rmn.fstprm(key2)
    >>> print("# %s ip1=%s ip2=%s" % (meta['nomvar'], meta['ip1'], meta['ip2']))
    # P0   ip1=0 ip2=12
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstinf
        fstinfx
        fstinl
        fstprm
        fstluk
        fstopenall
        fstcloseall
    """
    if not (type(iunit) == int):
        raise TypeError("fstsui: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstsui: must provide a valid iunit: %d" % (iunit))
    (cni, cnj, cnk) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    key = _rp.c_fstsui(iunit, _ct.byref(cni), _ct.byref(cnj), _ct.byref(cnk))
    if key < 0:
        return None
    return {
        'key'   : key ,
        'shape' : (max(1, cni.value), max(1, cnj.value), max(1, cnk.value)),
        }


def fstvoi(iunit, options=' '):
    """
    Prints out the directory content of a RPN standard file

    fstvoi(iunit)
    fstvoi(iunit, options)

    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        options : printing options
                  a string with desired fields list, '+' separated
                  possible fields (keywords):
                  NONOMV, NOTYPV, NOETIQ,
                  NINJNK, DATEO, DATESTAMPO,
                  DATEV, LEVEL, IPALL, IP1,
                  NOIP23, NODEET, NONPAS, NODTY,
                  GRIDINFO
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk','2009042700_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Print meta of all record in file
    >>> rmn.fstvoi(funit)
    >>> rmn.fstvoi(funit,'DATEV+LEVEL+NOTYPV+NOETIQ+NOIP23+NODEET+NONPAS+NODTY')
    >>> 
    >>> rmn.fstcloseall(funit)
    
    See Also:
        fstopenall
        fstcloseall
    """
    if not (type(iunit) == int):
        raise TypeError("fstvoi: Expecting arg of type int, Got %s" %
                        (type(iunit)))
    if iunit < 0:
        raise ValueError("fstvoi: must provide a valid iunit: %d" % (iunit))
    if not (type(options) == str):
        raise TypeError("fstvoi: Expecting options arg of type str, Got %s" %
                        (type(options)))
    istat = _rp.c_fstvoi(iunit, options)
    if istat < 0:
        raise FSTDError()  
    return


def fst_version():
    """
    Returns package version number
    
    fstd_version = fst_version()
    
    Returns:
        int, fstd version number

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> print("# Using fst_version=%d" % rmn.fst_version())
    # Using fst_version=200001

    See Also:
        fstopt
    """
    return _rp.c_fst_version()


def ip1_all(level, kind):
    """
    Generates all possible coded ip1 values for a given level

    The next search (fstinf, fstinl, fstlir, ...) will look for
    all prossible alternative encoding

    ip1new = ip1_all(level, kind)

    Args:
        level (float): level value
        kind  (int)  : level kind
    Returns:
        int, ip1 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_p','anlp2015070706_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Look for TT at 500mb encoded old or new style
    >>> ip1new = rmn.ip1_all(500., rmn.LEVEL_KIND_PMB)
    >>> ttrec  = rmn.fstlir(funit, nomvar='TT', ip1=ip1new)
    >>> print("# Looked for TT with ip1=%s, found ip1=%s" % (ip1new,ttrec['ip1']))
    # Looked for TT with ip1=41394464, found ip1=500
    >>> 
    >>> rmn.fstcloseall(funit)

    See Also:
        ip1_val
        ip2_all
        ip3_all
        convertIp
        EncodeIp
        DecodeIp
        fstinf
        fstinl
        fstlir
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip1_all: Expecting arg of type float, Got %s" %
                        (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip1_all: Expecting arg of type int, Got %s" %
                         (type(kind)))
    if kind < 0:
        raise ValueError("ip1_all: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip1_all(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


def ip2_all(level, kind):
    """
    Generates all possible coded ip2 values for a given level

    The next search (fstinf, fstinl, fstlir, ...) will look for
    all prossible alternative encoding

    ip2new = ip2_all(level, kind)

    Args:
        level : float, level value
        kind  : int,   level kind
    Returns:
        int, ip2 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Look for TT at 500mb encoded old or new style
    >>> ip2new = rmn.ip2_all(0., rmn.TIME_KIND_HR)
    >>> ttrec  = rmn.fstlir(funit, nomvar='TT', ip2=ip2new)
    >>> print("# Looked for TT with ip2=%s, found ip2=%s" % (ip2new, ttrec['ip2']))
    # Looked for TT with ip2=183500800, found ip2=0
    >>> 
    >>> rmn.fstcloseall(funit)

    See Also:
        ip2_val
        ip1_all
        ip3_all
        convertIp
        EncodeIp
        DecodeIp
        fstinf
        fstinl
        fstlir
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip2_all: Expecting arg of type float, Got %s" %
                         (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip2_all: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("ip2_all: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip2_all(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


def ip3_all(level, kind):
    """
    Generates all possible coded ip3 values for a given level

    The next search (fstinf, fstinl, fstlir, ...) will look for
    all prossible alternative encoding

    ip3new = ip3_all(level, kind)

    Args:
        level : float, level value
        kind  : int,   level kind
    Returns:
        int, ip3 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_p','anlp2015070706_000')
    >>> 
    >>> # Open existing file in Rear Only mode
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Look for TT at 500mb encoded old or new style
    >>> ip3new = rmn.ip3_all(0., rmn.KIND_ARBITRARY)
    >>> ttrec  = rmn.fstlir(funit, nomvar='TT', ip3=ip3new)
    >>> print("# Looked for TT with ip3=%s, found ip3=%s" % (ip3new, ttrec['ip3']))
    # Looked for TT with ip3=66060288, found ip3=0
    >>> 
    >>> rmn.fstcloseall(funit)

    See Also:
        ip3_val
        ip1_all
        ip2_all
        convertIp
        EncodeIp
        DecodeIp
        fstinf
        fstinl
        fstlir
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip3_all: Expecting arg of type float, Got %s" %
                        (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip3_all: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("ip3_all: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip3_all(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


def ip1_val(level, kind):
    """
    Generates coded ip1 value for a given level (shorthand for convip)

    ip1new = ip1_val(level, kind)

    Args:
        level : float, level value
        kind  : int,   level kind
    Returns:
        int, ip1 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> ip1new = rmn.ip1_val(500., rmn.LEVEL_KIND_PMB)

    See Also:
        ip1_all
        ip2_val
        ip3_val
        convertIp
        EncodeIp
        DecodeIp
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip1_val: Expecting arg of type float, Got %s" %
                        (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip1_val: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("ip1_val: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip1_val(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


def ip2_val(level, kind):
    """
    Generates coded ip2 value for a given level (shorthand for convip)

    ip2new = ip2_val(level, kind)

    Args:
        level : float, level value
        kind  : int,   level kind
    Returns:
        int, ip2 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> ip2new = rmn.ip2_val(0., rmn.TIME_KIND_HR)

    See Also:
        ip2_all
        ip1_val
        ip3_val
        convertIp
        EncodeIp
        DecodeIp
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip2_val: Expecting arg of type float, Got %s" %
                        (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip2_val: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("ip2_val: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip2_val(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


def ip3_val(level, kind):
    """
    Generates coded ip3 value for a given level (shorthand for convip)

    ip3new = ip3_val(level, kind)

    Args:
        level : float, level value
        kind  : int,   level kind
    Returns:
        int, ip3 value newcode style
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> ip3new = rmn.ip3_all(0., rmn.KIND_ARBITRARY)

    See Also:
        ip3_all
        ip1_val
        ip2_val
        convertIp
        EncodeIp
        DecodeIp
        rpnpy.librmn.const
    """
    if type(level) == int:
        level = float(level)
    if not (type(level) == float):
        raise TypeError("ip3_val: Expecting arg of type float, Got %s" %
                        (type(level)))
    if not (type(kind) == int):
        raise TypeError("ip3_val: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("ip3_val: must provide a valid iunit: %d" % (kind))
    ip = _rp.c_ip3_val(level, kind)
    if ip < 0:
        raise FSTDError()
    return ip


#TODO: ip_is_equal
## def ip_is_equal(target, ip, ind):
##     """Compares different coded values of an ip for equality

##     target: must be first value in the table of coded value to compare with
##     ip    : current ip record value to compare
##     ind   : index (1, 2 or 3) representing ip1, ip2 or ip3 comparaisons

##     return ???
##     """
##     return _rp.ip_is_equal(target, ip, ind)


#--- fstd98/convip_plus & convert_ip123 ---------------------------------


def convertIp(mode, v, k=0):
    """
    Codage/Decodage P, kind <-> IP pour IP1, IP2, IP3
    
    ip        = convertIp(mode, p, kind) #if mode > 0
    (p, kind) = convertIp(mode, ip)      #if mode <= 0
    
    Args:
        ip   : Encoded value (int)
        p    : Real Value (float)
        kind : Level encoding kind/code (int)
        mode : Conversion mode (int)
        
        kind can take the following values
        0, p est en hauteur (m) rel. au niveau de la mer (-20, 000 -> 100, 000)
        1, p est en sigma                                (0.0 -> 1.0)
        2, p est en pression (mb)                        (0 -> 1100)
        3, p est un code arbitraire                      (-4.8e8 -> 1.0e10)
        4, p est en hauteur (M) rel. au niveau du sol    (-20, 000 -> 100, 000)
        5, p est en coordonnee hybride                   (0.0 -> 1.0)
        6, p est en coordonnee theta                     (1 -> 200, 000)
        10, p represente le temps en heure               (0.0 -> 1.0e10)
        15, reserve (entiers)                                   
        17, p represente l'indice x de la matrice de conversion (1.0 -> 1.0e10)
            (partage avec kind=1 a cause du range exclusif
        21, p est en metres-pression                     (0 -> 1, 000, 000)
                                                         fact=1e4
            (partage avec kind=5 a cause du range exclusif)
            
        mode can take the following values
        -1, de IP -->  P
        0, forcer conversion pour ip a 31 bits
        (default = ip a 15 bits) (appel d'initialisation)
        +1, de P  --> IP
        +2, de P  --> IP en mode NEWSTYLE force a true
        +3, de P  --> IP en mode NEWSTYLE force a false
    Returns:
        int, ip Encoded value, if mode > 0
        (float, int), (pvalue, kind) Decoded value, if mode <= 0
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error       

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> ip1 = rmn.convertIp(rmn.CONVIP_ENCODE_OLD, 500., rmn.KIND_PRESSURE)
    >>> ip1 = rmn.convertIp(rmn.CONVIP_ENCODE, 500., rmn.KIND_PRESSURE)
    >>> (val, kind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)

    See Also:
        ip1_val
        ip2_val
        ip3_val
        EncodeIp
        DecodeIp
        convertIPtoPK
        convertPKtoIP
        kindToString
        rpnpy.librmn.const
    """
    (cip, cp, ckind) = (_ct.c_int(), _ct.c_float(), _ct.c_int())
    if type(mode) != int:
        raise TypeError("convertIp: " +
                        "Expecting mode to be of type int, Got %s : %s" %
                        (type(mode), repr(mode)))
    if mode < -1 or mode > 3:
        raise ValueError("convertIp: must provide a valid mode: %d" % (mode))
    if mode > 0:
        if type(v) == int:
            v = float(v)
        if type(v) !=  float:
            raise TypeError("convertIp: Expecting value to be of type float, " +
                            "Got %s : %s" % (type(v), repr(v)))
        if type(k) !=  int:
            raise TypeError("convertIp: Expecting kind to be of type int, " +
                            "Got %s : %s" % (type(k), repr(k)))
        (cp, ckind) = (_ct.c_float(v), _ct.c_int(k))
    else:
        if type(v) !=  int:
            raise TypeError("convertIp: Expecting value to be of type int, " +
                            "Got %s : %s" % (type(v), repr(v)))
        cip = _ct.c_int(v)
    _rp.c_ConvertIp(_ct.byref(cip), _ct.byref(cp), _ct.byref(ckind), mode)
    if mode > 0:
        return cip.value
    else:
        return (cp.value, ckind.value)


def convertIPtoPK(ip1, ip2, ip3):
    """
    Convert/decode ip1, ip2, ip3 to their kind + real value conterparts

    (rp1, rp2, rp3) = convertIPtoPK(ip1, ip2, ip3)

    Args:
        ip1   : vertical level (int)
        ip2   : forecast hour (int)
        ip3   : user defined identifier (int)
    Returns:
        rp1    : decoded ip1 (FLOAT_IP)
                 a level (or a pair of levels) in the atmosphere
        rp2    : decoded ip2 (FLOAT_IP)
                 a time (or a pair of times)
        rp3    : decoded ip3 (FLOAT_IP)
                 may contain anything
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  when provided values cannot be converted

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Encode 500mb at 12h,
    >>> # these ip1, ip2, ip3 can be used as search keys int fstinf, fstlir, ...
    >>> pk1a = rmn.FLOAT_IP(500., 500., rmn.LEVEL_KIND_PMB)
    >>> pk2a = rmn.FLOAT_IP( 12.,  12., rmn.TIME_KIND_HR)
    >>> pk3a = rmn.FLOAT_IP(  0.,   0., rmn.KIND_ARBITRARY)
    >>> (ip1, ip2, ip3) = rmn.convertPKtoIP(pk1a, pk2a, pk3a)
    >>> 
    >>> # Decode and print
    >>> (pk1, pk2, pk3) = rmn.convertIPtoPK(ip1, ip2, ip3)
    >>> print("# Level v1=%f, v2=%f, type=%s" % \
              (pk1.v1, pk1.v2, rmn.kindToString(pk1.kind)))
    # Level v1=500.000000, v2=500.000000, type=mb
    >>> print("# Time v1=%f, v2=%f, type=%s" % \
              (pk2.v1, pk2.v2, rmn.kindToString(pk2.kind)))
    # Time v1=12.000000, v2=12.000000, type= H

    See Also:
        ip1_val
        ip2_val
        ip3_val
        EncodeIp
        DecodeIp
        convertIp
        convertPKtoIP
        kindToString
        rpnpy.librmn.const
    """
    if type(ip1) != int or type(ip2) != int or type(ip3) != int:
        raise TypeError("convertIPtoPK: Expecting ip123 to be of type int, " +
                        "Got %s, %s, %s" % (type(ip1), type(ip2), type(ip3)))
    if ip1 < 0 or ip2 < 0 or ip3 < 0:
        raise ValueError("convertIPtoPK: Expecting invalid ip123, " +
                         "Got %d, %d, %d" % (ip1, ip2, ip3))
    (cp1, cp2, cp3) = (_ct.c_float(), _ct.c_float(), _ct.c_float())
    (ck1, ck2, ck3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    istat = _rp.c_ConvertIPtoPK(_ct.byref(cp1), _ct.byref(ck1),
                                _ct.byref(cp2), _ct.byref(ck2),
                                _ct.byref(cp3), _ct.byref(ck3), ip1, ip2, ip3)
    if istat == 64:
        raise FSTDError()
    return (listToFLOATIP((cp1.value, cp1.value, ck1.value)),
            listToFLOATIP((cp2.value, cp2.value, ck2.value)),
            listToFLOATIP((cp3.value, cp3.value, ck3.value)))
    

def convertPKtoIP(pk1, pk2, pk3):
    """
    Convert/encode kind + real value into ip1, ip2, ip3

    (ip1, ip2, ip3) = convertPKtoIP(pk1, pk2, pk3)

    Args:
        pk1    : vertical level, real values & kind (FLOAT_IP)
                 a level (or a pair of levels) in the atmosphere
        pk2    : forecast hour, real values & kind (FLOAT_IP)
                 a time (or a pair of times)
        pk3    : user defined identifier, real values & kind (FLOAT_IP)
                 may contain anything, PK3%hi will be ignored
                 (if pk1 or pk2 contains a pair, pk3 is ignored)
    Returns:
        ip1   : encoded pk1, vertical level (int)
        ip2   : encoded pk2, forecast hour (int)
        ip3   : encoded pk3, user defined identifier (int)
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  when provided values cannot be converted

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Encode 500mb at 12h,
    >>> # these ip1, ip2, ip3 can be used as search keys int fstinf, fstlir, ...
    >>> pk1a = rmn.FLOAT_IP(500., 500., rmn.LEVEL_KIND_PMB)
    >>> pk2a = rmn.FLOAT_IP( 12.,  12., rmn.TIME_KIND_HR)
    >>> pk3a = rmn.FLOAT_IP(  0.,   0., rmn.KIND_ARBITRARY)
    >>> (ip1, ip2, ip3) = rmn.convertPKtoIP(pk1a, pk2a, pk3a)
    >>> 
    >>> # Decode and print
    >>> (pk1, pk2, pk3) = rmn.convertIPtoPK(ip1, ip2, ip3)
    >>> print("# Level v1=%f, v2=%f, type=%s" % \
              (pk1.v1, pk1.v2, rmn.kindToString(pk1.kind)))
    # Level v1=500.000000, v2=500.000000, type=mb
    >>> print("# Time v1=%f, v2=%f, type=%s" % \
              (pk2.v1, pk2.v2, rmn.kindToString(pk2.kind)))
    # Time v1=12.000000, v2=12.000000, type= H

    See Also:
        ip1_val
        ip2_val
        ip3_val
        EncodeIp
        DecodeIp
        convertIPtoPK
        convertIp
        kindToString
        rpnpy.librmn.const
    """
    (cip1, cip2, cip3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    pk1 = listToFLOATIP(pk1)
    pk2 = listToFLOATIP(pk2)
    pk3 = listToFLOATIP(pk3)
    istat = _rp.c_ConvertPKtoIP(_ct.byref(cip1), _ct.byref(cip2),
                                _ct.byref(cip3), pk1.kind, pk1.v1,
                                pk2.kind, pk2.v1, pk3.kind, pk3.v1)
    if istat == 64:
        raise FSTDError()
    return (cip1.value, cip2.value, cip3.value)


def EncodeIp(rp1, rp2, rp3):
    """
    Produce encoded (ip1, ip2, ip3) triplet from (real value, kind) pairs

    (ip1, ip2, ip3) = EncodeIp(rp1, rp2, rp3)

    Args:
        rp1    : vertical level, real values & kind (FLOAT_IP)
                 a level (or a pair of levels) in the atmosphere
        rp2    : forecast hour, real values & kind (FLOAT_IP)
                 a time (or a pair of times)
        rp3    : user defined identifier, real values & kind (FLOAT_IP)
                 may contain anything, RP3%hi will be ignored
                 (if rp1 or rp2 contains a pair, rp3 is ignored)
    Returns:
        ip1   : encoded rp1, vertical level (int)
        ip2   : encoded rp2, forecast hour (int)
        ip3   : encoded rp3, user defined identifier (int)
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  when provided values cannot be converted

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> 
    >>> # Encode 500mb at 12h,
    >>> # these ip1, ip2, ip3 can be used as search keys int fstinf, fstlir, ...
    >>> rp1a = rmn.FLOAT_IP(500., 500., rmn.LEVEL_KIND_PMB)
    >>> rp2a = rmn.FLOAT_IP( 12.,  12., rmn.TIME_KIND_HR)
    >>> rp3a = rmn.FLOAT_IP(  0.,   0., rmn.KIND_ARBITRARY)
    >>> (ip1, ip2, ip3) = rmn.EncodeIp(rp1a, rp2a, rp3a)
    >>> 
    >>> # Decode and print
    >>> (rp1, rp2, rp3) = rmn.DecodeIp(ip1, ip2, ip3)
    >>> print("# Level v1=%f, v2=%f, type=%s" % \
              (rp1.v1, rp1.v2, rmn.kindToString(rp1.kind)))
    # Level v1=500.000000, v2=500.000000, type=mb
    >>> print("# Time v1=%f, v2=%f, type=%s" % \
              (rp2.v1, rp2.v2, rmn.kindToString(rp2.kind)))
    # Time v1=12.000000, v2=12.000000, type= H

    See Also:
        DecodeIp
        ip1_val
        ip2_val
        ip3_val
        convertIp
        convertIPtoPK
        convertPKtoIP
        kindToString
        rpnpy.librmn.const
    """
    rp1 = listToFLOATIP(rp1)
    rp2 = listToFLOATIP(rp2)
    rp3 = listToFLOATIP(rp3)
    (cip1, cip2, cip3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    istat = _rp.c_EncodeIp(_ct.byref(cip1), _ct.byref(cip2), _ct.byref(cip3),
                       _ct.byref(rp1), _ct.byref(rp2), _ct.byref(rp3))
    if istat == 32:
        raise FSTDError()
    return (cip1.value, cip2.value, cip3.value)


def DecodeIp(ip1, ip2, ip3):
    """
    Produce decoded (real value, kind) pairs
    from (ip1, ip2, ip3) encoded triplet

    (rp1, rp2, rp3) = DecodeIp(ip1, ip2, ip3)
    
    Args:
        ip1   : vertical level (int)
        ip2   : forecast hour (int)
        ip3   : user defined identifier (int)
    Returns:
        rp1    : decoded ip1 (FLOAT_IP)
                 a level (or a pair of levels) in the atmosphere
        rp2    : decoded ip2 (FLOAT_IP)
                 a time (or a pair of times)
        rp3    : decoded ip3 (FLOAT_IP)
                 may contain anything
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  when provided values cannot be converted

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> (ip1, ip2, ip3) = (6441456, 176280768, 66060288)
    >>> (rp1, rp2, rp3) = rmn.DecodeIp(ip1, ip2, ip3)
    >>> print("# Level v1=%f, v2=%f, type=%s" % \
              (rp1.v1, rp1.v2, rmn.kindToString(rp1.kind)))
    # Level v1=1500.000000, v2=1500.000000, type= m

    See Also:
        EncodeIp
        ip1_val
        ip2_val
        ip3_val
        convertIp
        convertIPtoPK
        convertPKtoIP
        kindToString
        rpnpy.librmn.const
    """
    (rp1, rp2, rp3) = (_rp.FLOAT_IP(0., 0., 0), _rp.FLOAT_IP(0., 0., 0),
                       _rp.FLOAT_IP(0., 0., 0))
    (cip1, cip2, cip3) = (_ct.c_int(ip1), _ct.c_int(ip2), _ct.c_int(ip3))
    istat = _rp.c_DecodeIp(_ct.byref(rp1), _ct.byref(rp2), _ct.byref(rp3),
                           cip1, cip2, cip3)
    if istat == 32:
        raise FSTDError()
    return (rp1, rp2, rp3)


def kindToString(kind):
    """
    Translate kind integer code to 2 character string,

    kind_str = kindToString(kind)

    Args:
        kind : Level encoding kind/code (int)
    Returns:
        str, str repr of the kind code

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> print('# '+rmn.kindToString(rmn.LEVEL_KIND_PMB))
    # mb
    >>> print('# '+rmn.kindToString(rmn.TIME_KIND_HR))
    #  H

    See Also:
        ip1_val
        ip2_val
        ip3_val
        EncodeIp
        DecodeIp
        convertIp
        convertIPtoPK
        convertPKtoIP
        rpnpy.librmn.const
    """
    if not (type(kind) == int):
        raise TypeError("kindToString: Expecting arg of type int, Got %s" %
                        (type(kind)))
    if kind < 0:
        raise ValueError("kindToString: must provide a valid iunit: %d" %
                         (kind))
    (str1, str2) = (_C_MKSTR(' '), _C_MKSTR(' '))
    _rp.c_KindToString(kind, str1, str2)
    str12 = str1[0]+str2[0]
    if str12.strip() == '':
        raise FSTDError()
    return str12


# =========================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
