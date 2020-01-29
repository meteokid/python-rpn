#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn.interp contains python wrapper to
main librmn's interp (ezscint) C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[librmn]]'s [[Librmn/FSTDfunctions|FSTD]] package.<br>
    You may want to refer to the [[Librmn/FSTDfunctions|FSTD]]
    documentation for more details.

See Also:
    rpnpy.librmn.base
    rpnpy.librmn.fstd98
    rpnpy.librmn.grids
    rpnpy.librmn.const
"""

import ctypes as _ct
import numpy  as _np
from rpnpy.librmn import proto as _rp
from rpnpy.librmn  import const as _rc
from rpnpy.librmn  import RMNError
from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR as _C_MKSTR

#TODO: make sure caller can provide allocated array (recycle mem)

#---- helpers -------------------------------------------------------

def _getCheckArg(okTypes, value, valueDict, key):
    if isinstance(valueDict, dict) and (value is None or value is valueDict):
        if key in valueDict.keys():
            value = valueDict[key]
    if (okTypes is not None) and not isinstance(value, okTypes):
        raise EzscintError('For {0} type, Expecting {1}, Got {2}'.
                           format(key, repr(okTypes), type(value)))
    return value

_isftn = lambda x, t: x.dtype == t and x.flags['F_CONTIGUOUS']
_ftn   = lambda x, t: x if _isftnf32(x) else _np.asfortranarray(x, dtype=t)
_isftnf32 = lambda x: _isftn(x, _np.float32)
_ftnf32   = lambda x: _ftn(x, _np.float32)
_ftnOrEmpty = lambda x, s, t: \
    _np.empty(s, dtype=t, order='F') if x is None else _ftn(x, t)
_list2ftnf32 = lambda x: \
    x if isinstance(x, _np.ndarray) \
      else _np.asfortranarray(x, dtype=_np.float32)

class EzscintError(RMNError):
    """
    General librmn.interp module error/exception

    To make your code handle errors in an elegant manner,
    you may want to catch that error with a 'try ... except' block.

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> try:
    ...    rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
    ... except rmn.EzscintError:
    ...    pass #ignore the error
    ... finally:
    ...    print("# Whatever happens, error or not, print this.")
    # Whatever happens, error or not, print this.

    See also:
        rpnpy.librmn.RMNError
    """
    pass

#---- interp (ezscint) ----------------------------------------------

#---- Set Functions

def ezsetival(option, value):
    """
    Sets an integer numerical ezscint option
    
    ezsetival(option, value)
    
    Args:
        option : option name (string)
        value  : option value (float)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    See Also:
        ezsetval
        ezsetopt
        ezgetopt
        rpnpy.librmn.const
    """
    return ezsetopt(option, value)


def ezsetval(option, value):
    """
    Sets a floating point numerical ezscint option
    
    ezsetval(option, value)
    
    Args:
        option : option name (string)
        value  : option value (float)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.ezsetval(rmn.EZ_OPT_EXTRAP_VALUE, 999.)

    See Also:
        ezsetival
        ezsetopt
        ezgetopt
        rpnpy.librmn.const
    """
    return ezsetopt(option, value)


def ezsetopt(option, value):
    """
    Sets ezscint option: float, integer or string

    ezsetopt(option, value)

    Args:
        option : option name (string)
        value  : option value (int, float or string)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Notes:
        This function replaces the following C/Fortran functions:
        ezsetival, ezsetval, ezsetopt

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
    >>> rmn.ezsetopt(rmn.EZ_OPT_USE_1SUBGRID,  rmn.EZ_YES)

    See Also:
        ezsetval
        ezsetival
        ezgetopt
        rpnpy.librmn.const
    """
    if not isinstance(option, str):
        raise TypeError("ezsetopt: expecting args of type str, Got {0}".
                        format(type(option)))
    if isinstance(value, _integer_types):
        istat = _rp.c_ezsetival(_C_WCHAR2CHAR(option), value)
    elif isinstance(value, float):
        istat = _rp.c_ezsetval(_C_WCHAR2CHAR(option), value)
    elif isinstance(value, str):
        istat = _rp.c_ezsetopt(_C_WCHAR2CHAR(option), _C_WCHAR2CHAR(value))
    else:
        raise TypeError("ezsetopt: Not a supported type {0}".
                        format(type(value)))
    if istat >= 0:
        return None
    raise EzscintError()


def ezqkdef(ni, nj=None, grtyp=None, ig1=None, ig2=None, ig3=None, ig4=None,
            iunit=0):
    """
    Universal grid definition. Applicable to all cases.

    gdid = ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
    gdid = ezqkdef(gridParams)

    Args:
        ni, nj        : grid dims (int)
        grtyp        : grid type (str)
        ig1, ig2, ig3, ig4 : grid parameters, encoded (int)
        iunit        : File unit, optional (int)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, grid id
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> # Define a LatLon grid
    >>> (grtyp, lat0, lon0, dlat, dlon) = ('L', 45., 273., 0.5, 0.5)
    >>> (ig1, ig2, ig3, ig4) = rmn.cxgaig(grtyp, lat0, lon0, dlat, dlon)
    >>> gid = rmn.ezqkdef(90, 45, grtyp, ig1, ig2, ig3, ig4)

    See Also:
        ezgdef_fmem
        rpnpy.librmn.base.cxgaig
        rpnpy.librmn.grids
    """
    if isinstance(ni, dict):
        gridParams = ni
        try:
            (ni, nj) = gridParams['shape']
        except:
            (ni, nj) = (None, None)
        try:
            if not ni:
                ni = gridParams['ni']
            if not nj:
                nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            ig1 = gridParams['ig1']
            ig2 = gridParams['ig2']
            ig3 = gridParams['ig3']
            ig4 = gridParams['ig4']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
        try:
            iunit = gridParams['iunit']
        except:
            iunit = 0
    if (type(ni), type(nj), type(grtyp), type(ig1), type(ig2), type(ig3),
        type(ig4), type(iunit)) != (int, int, str, int, int, int, int, int):
        raise TypeError('ezqkdef: wrong input data type')
    if grtyp.strip() in ('', 'X'):
        raise EzscintError('ezqkdef: Grid type {0} Not supported'.format(grtyp))
    if iunit <= 0 and grtyp.strip() in ('Z', '#', 'Y', 'U'):
        raise EzscintError('ezqkdef: A valid opened file unit ({0}) is needed for Grid type {1}'.format(iunit, grtyp))
    gdid = _rp.c_ezqkdef(ni, nj, _C_WCHAR2CHAR(grtyp), ig1, ig2, ig3, ig4, iunit)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_fmem(ni, nj=None, grtyp=None, grref=None, ig1=None, ig2=None,
                ig3=None, ig4=None, ax=None, ay=None):
    """
    Generic grid definition except for 'U' grids (with necessary
    positional parameters taken from the calling arguments)

    gdid = ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
    gdid = ezgdef_fmem(gridParams)

    Args:
        ni, nj        : grid dims (int)
        grtyp, grref : grid type and grid ref type (str)
        ig1, ig2, ig3, ig4 : grid parameters, encoded (int)
        ax, ay       : grid axes (numpy.ndarray)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, grid id
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>> (ni,nj) = (90, 45)
    >>> gp = {
    ...     'shape' : (ni,nj),
    ...     'ni' : ni,
    ...     'nj' : nj,
    ...     'grtyp' : 'Z',
    ...     'grref' : 'E',
    ...     'xlat1' : 0.,
    ...     'xlon1' : 180.,
    ...     'xlat2' : 0.,
    ...     'xlon2' : 270.,
    ...     'dlat' : 0.25,
    ...     'dlon' : 0.25,
    ...     'lat0' : 45.,
    ...     'lon0' : 273.
    ...     }
    >>> ig1234 = rmn.cxgaig(gp['grref'], gp['xlat1'], gp['xlon1'],
    ...                     gp['xlat2'], gp['xlon2'])
    >>> gp['ax'] = np.empty((ni,1), dtype=np.float32, order='F')
    >>> gp['ay'] = np.empty((1,nj), dtype=np.float32, order='F')
    >>> for i in range(ni): gp['ax'][i,0] = gp['lon0']+float(i)*gp['dlon']
    >>> for j in range(nj): gp['ay'][0,j] = gp['lat0']+float(j)*gp['dlat']
    >>> gid = rmn.ezgdef_fmem(ni, nj, gp['grtyp'], gp['grref'],
    ...                       ig1234[0], ig1234[1], ig1234[2], ig1234[3],
    ...                       gp['ax'], gp['ay'])

    See Also:
        ezqkdef
        rpnpy.librmn.base.cxgaig
        rpnpy.librmn.grids
    """
    if isinstance(ni, dict):
        gridParams = ni
        try:
            (ni, nj) = gridParams['shape']
        except:
            (ni, nj) = (None, None)
        try:
            if not ni:
                ni = gridParams['ni']
            if not nj:
                nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            grref = gridParams['grref']
            ig1 = gridParams['ig1']
            ig2 = gridParams['ig2']
            ig3 = gridParams['ig3']
            ig4 = gridParams['ig4']
            ax = gridParams['ax']
            ay = gridParams['ay']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    if ((type(ni), type(nj), type(grtyp), type(grref), type(ig1), type(ig2),
        type(ig3), type(ig4), type(ax), type(ay)) !=
        (int, int, str, str, int, int, int, int, _np.ndarray, _np.ndarray)):
        raise TypeError('ezgdef_fmem: wrong input data type')
    if grtyp in ('Z', 'z', '#'):
        if ax.size != ni or ay.size != nj:
            raise EzscintError('ezgdef_fmem: size mismatch for provided ' +
                               'ax, ay compared to ni, nj')
    elif grtyp in ('Y', 'y'):
        if ax.shape != (ni, nj) or ay.shape != (ni, nj):
            raise EzscintError('ezgdef_fmem: size mismatch for provided ' +
                               'ax, ay compared to ni, nj')
    elif grtyp in ('U', 'u'):
        pass
        #TODO: check ni, nj ... ax, ay dims consis for U grids
    else:
        raise EzscintError('ezgdef_fmem: Unknown grid type: '+grtyp)
    ax = _ftnf32(ax)
    ay = _ftnf32(ay)
    gdid = _rp.c_ezgdef_fmem(ni, nj, _C_WCHAR2CHAR(grtyp), _C_WCHAR2CHAR(grref), ig1, ig2, ig3, ig4, ax, ay)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_supergrid(ni, nj, grtyp, grref, vercode, subgridid):
    """
    U grid definition
    (which associates to a list of concatenated subgrids in one record)

    gdid = ezgdef_supergrid(ni, nj, grtyp, grref, vercode, nsubgrids, subgridid)
    gdid = ezgdef_supergrid(gridParams)

    Args:
        ni, nj        : grid dims (int)
        grtyp, grref : grid type and grid ref type (str)
        vercode      :
        subgridid    : list of subgrid id (list, tuple or numpy.ndarray)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, super grid id
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> nj = 31 ; ni = (nj-1)*3 + 1
    >>> gp = {
    ...     'ni' : ni,
    ...     'nj' : nj,
    ...     'grtyp' : 'Z',
    ...     'grref' : 'E',
    ...     'xlat1' : 0.,
    ...     'xlon1' : 180.,
    ...     'xlat2' : 0.,
    ...     'xlon2' : 270.,
    ...     'dlat' : 0.25,
    ...     'dlon' : 0.25,
    ...     'lat0' : 45.,
    ...     'lon0' : 273.
    ...     }
    >>> yin = rmn.encodeGrid(gp)
    >>> (gp['xlat1'], gp['xlon1'], gp['xlat2'], gp['xlon2']) = (
    ...     rmn.yyg_yangrot_py(gp['xlat1'],gp['xlon1'],gp['xlat2'],gp['xlon2']))
    >>> yan = rmn.encodeGrid(gp)
    >>> yy_id = rmn.ezgdef_supergrid(ni, nj, 'U', 'F', 1, (yin['id'],yan['id']))


    See Also:
        ezget_nsubgrids
        ezget_subgridids
        rpnpy.librmn.grids.encodeGrid
        rpnpy.librmn.grids.yyg_yangrot_py
        rpnpy.librmn.grids.defGrid_YY
    """
    if isinstance(ni, dict):
        gridParams = ni
        try:
            (ni, nj) = gridParams['shape']
        except:
            (ni, nj) = (None, None)
        try:
            if not ni:
                ni = gridParams['ni']
            if not nj:
                nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            grref = gridParams['grref']
            vercode = gridParams['vercode']
            subgridid = gridParams['subgridid']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    csubgridid = _np.asfortranarray(subgridid, dtype=_np.intc)
    if (type(ni), type(nj), type(grtyp), type(grref), type(vercode),
        type(csubgridid)) != (int, int, str, str, int, _np.ndarray):
        raise TypeError('ezgdef_fmem: wrong input data type')
    nsubgrids = csubgridid.size
    gdid = _rp.c_ezgdef_supergrid(ni, nj, _C_WCHAR2CHAR(grtyp), _C_WCHAR2CHAR(grref), vercode, nsubgrids,
                                  csubgridid)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezdefset(gdidout, gdidin):
    """
    Defines a set of grids for interpolation

    gridsetid = ezdefset(gdidout, gdidin)

    Args:
        gdidout : output grid id (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        gdidin  : input  grid id (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
    Returns:
        int, grid set id
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn

    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Define a destination Grid
    >>> (lat0, lon0, dlat, dlon) = (35.,265.,0.25,0.25)
    >>> (ni, nj) = (200, 100)
    >>> outGrid  = rmn.defGrid_L(ni, nj, lat0, lon0, dlat, dlon)
    >>>
    >>> # Define the grid-set and interpolate data linearly
    >>> gridsetid = rmn.ezdefset(outGrid, inGrid)
    >>> rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
    >>> me2 = rmn.ezsint(outGrid['id'], inGrid['id'], meRec['d'])

    See Also:
        ezsetopt
        ezsint
        ezuvint
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
        rpnpy.librmn.const
    """
    gdidout = _getCheckArg(int, gdidout, gdidout, 'id')
    gdidin  = _getCheckArg(int, gdidin, gdidin, 'id')
    istat = _rp.c_ezdefset(gdidout, gdidin)
    if istat < 0:
        raise EzscintError()
    return istat


def gdsetmask(gdid, mask):
    """
    Associates a permanent mask with grid 'gdid'

    gdsetmask(gdid, mask)

    Args:
        gdid : grid id (int or dict)
               Dict with key 'id' is accepted from version 2.0.rc1
        mask : field mask (numpy.ndarray)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> mgRec  = rmn.fstlir(funit, nomvar='MG')
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Define a destination Grid
    >>> (lat0, lon0, dlat, dlon) = (35.,265.,0.25,0.25)
    >>> (ni, nj) = (200, 100)
    >>> outGrid  = rmn.defGrid_L(ni, nj, lat0, lon0, dlat, dlon)
    >>>
    >>> # Set a masks over land only for input and output grids
    >>> inMask = np.rint(mgRec['d']).astype(np.intc)
    >>> rmn.gdsetmask(inGrid['id'], inMask)
    >>> mg = rmn.ezsint(outGrid['id'], inGrid['id'], mgRec['d'])
    >>> outMask = np.rint(mg).astype(np.intc)
    >>> rmn.gdsetmask(outGrid['id'], outMask)

    See Also:
        gdgetmask
        ezsint
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    if not (isinstance(mask, _np.ndarray) and mask.dtype in (_np.intc, _np.int32)):
        raise TypeError("Expecting mask type=numpy,intc, Got {0}".format(mask.dtype))
    mask  = _ftn(mask, mask.dtype)
    istat = _rp.c_gdsetmask(gdid, mask)
    if istat < 0:
        raise EzscintError()
    return


#---- Query Functions


def ezgetival(option):
    """
    Gets an ezscint integer option value
    
    value = ezgetival(option)
    
    Args:
        option : option name (string)
    Returns:
        integer, option value
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
        
    See Also:
        ezgetval
        ezgetopt
        ezsetopt
        rpnpy.librmn.const
    """
    return ezgetopt(option, vtype=int)


def ezgetval(option):
    """
    Gets an ezscint float option value
    
    value = ezgetval(option)
    
    Args:
        option : option name (string)
    Returns:
        float, option value
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
        
    See Also:
        ezgetival
        ezgetopt
        ezsetopt
        rpnpy.librmn.const
    """
    return ezgetopt(option, vtype=float)


def ezgetopt(option, vtype=int):
    """
    Gets an ezscint option value

    value = ezgetopt(option)
    value = ezgetopt(option, vtype)

    Args:
        option : option name (string)
        vtype  : type of requested option (type.int, type.float or type.string)
                 default: int
    Returns:
        option value of the requested type
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Notes:
        This function replaces the following C/Fortran functions:
        ezgetival, ezgetval, ezgetopt

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> interp_degree = rmn.ezgetopt(rmn.EZ_OPT_INTERP_DEGREE, vtype=str)
    >>> print("# Field will be interpolated with type: {0}".format(interp_degree))
    # Field will be interpolated with type: linear

    See Also:
        ezgetival
        ezgetval
        ezsetopt
        rpnpy.librmn.const
    """
    if not isinstance(option, str):
        raise TypeError("ezgetopt: expecting args of type str, Got {0}".
                        format(type(option)))
    if vtype == int:
        cvalue = _ct.c_int()
        istat = _rp.c_ezgetival(_C_WCHAR2CHAR(option), cvalue)
    elif vtype == float:
        cvalue = _ct.c_float()
        istat = _rp.c_ezgetval(_C_WCHAR2CHAR(option), cvalue)
    elif vtype == str:
        cvalue = _C_MKSTR(' '*64)
        istat = _rp.c_ezgetopt(_C_WCHAR2CHAR(option), cvalue)
    else:
        raise TypeError("ezgetopt: Not a supported type {0}".format(repr(vtype)))
    if istat >= 0:
        if isinstance(cvalue.value, bytes):
            return _C_CHAR2WCHAR(cvalue.value)
        else:
            return cvalue.value
    raise EzscintError()


def ezget_nsubgrids(super_gdid):
    """
    Gets the number of subgrids from the 'U' (super) grid id

    nsubgrids = ezget_nsubgrids(super_gdid)

    Args:
        super_gdid : id of the super grid (int or dict)
                     Dict with key 'id' is accepted from version 2.0.rc1
    Returns:
        int, number of sub grids associated with super_gdid
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> yy = rmn.defGrid_YY(31)
    >>> n = rmn.ezget_nsubgrids(yy['id'])
    >>> print("# There are {0} subgrids in the YY grid".format(n))
    # There are 2 subgrids in the YY grid

    See Also:
        ezgdef_supergrid
        ezget_subgridids
        ezgprm
        ezgxprm
        ezgfstp
        gdgaxes
        rpnpy.librmn.grids.defGrid_YY
        rpnpy.librmn.grids.decodeGrid
    """
    super_gdid = _getCheckArg(int, super_gdid, super_gdid, 'id')
    nsubgrids = _rp.c_ezget_nsubgrids(super_gdid)
    if nsubgrids >= 0:
        return nsubgrids
    raise EzscintError()


def ezget_subgridids(super_gdid):
    """
    Gets the list of grid ids for the subgrids in the 'U' grid (super_gdid).

    subgridids = ezget_subgridids(super_gdid)

    Args:
        super_gdid : id of the super grid (int or dict)
                     Dict with key 'id' is accepted from version 2.0.rc1
    Returns:
        int, list of grid ids for the subgrids
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> yy = rmn.defGrid_YY(31)
    >>> idlist = rmn.ezget_subgridids(yy['id'])
    >>> print("# Found {} subrids".format(len(idlist)))
    # Found 2 subrids

    See Also:
        ezgdef_supergrid
        ezget_nsubgrids
        ezgprm
        ezgxprm
        ezgfstp
        gdgaxes
        rpnpy.librmn.grids.defGrid_YY
        rpnpy.librmn.grids.decodeGrid
    """
    super_gdid = _getCheckArg(int, super_gdid, super_gdid, 'id')
    nsubgrids  = ezget_nsubgrids(super_gdid)
    cgridlist  = _np.empty(nsubgrids, dtype=_np.intc, order='F')
    istat = _rp.c_ezget_subgridids(super_gdid, cgridlist)
    if istat >= 0:
        return cgridlist.tolist()
    raise EzscintError()


def ezgprm(gdid, doSubGrid=False):
    """
    Get grid parameters

    gridParams = ezgprm(gdid)

    Args:
        gdid      : id of the grid (int or dict)
                    Dict with key 'id' is accepted from version 2.0.rc1
        doSubGrid : recurse on subgrids if True
    Returns:
        {
            'id'    : grid id, same as input arg
            'shape' : (ni, nj) # dimensions of the grid
            'ni'    : first dimension of the grid
            'nj'    : second dimension of the grid
            'grtyp' : type of geographical projection
                      (one of 'A', 'B', 'E', 'G', 'L', 'N', 'S')
            'ig1'   : first grid descriptor
            'ig2'   : second grid descriptor
            'ig3'   : third grid descriptor
            'ig4'   : fourth grid descriptor
        }
        if doSubGrid, add these
        {
            'nsubgrids' : Number of subgrids
            'subgridid' : list of subgrids id
            'subgrid'   : list of subgrids details {'id', 'shape', ...}
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Define a LatLon grid
    >>> (grtyp, lat0, lon0, dlat, dlon) = ('L', 45., 273., 0.5, 0.5)
    >>> (ig1, ig2, ig3, ig4) = rmn.cxgaig(grtyp, lat0, lon0, dlat, dlon)
    >>> gid = rmn.ezqkdef(90, 45, grtyp, ig1, ig2, ig3, ig4)
    >>>
    >>> # Get grid info from any grid id
    >>> params = rmn.ezgprm(gid)
    >>> print("# Grid type={grtyp} of size={ni}, {nj}".format(**params))
    # Grid type=L of size=90, 45

    See Also:
        ezgxprm
        ezgfstp
        gdgaxes
        ezget_nsubgrids
        ezget_subgridids
        ezqkdef
        rpnpy.librmn.base.cxgaig
        rpnpy.librmn.grids.decodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    (cni, cnj) = (_ct.c_int(), _ct.c_int())
    (cgrtyp, cig1, cig2, cig3, cig4) = (_C_MKSTR(' '*_rc.FST_GRTYP_LEN),
                                        _ct.c_int(), _ct.c_int(), _ct.c_int(),
                                        _ct.c_int())
    istat = _rp.c_ezgprm(gdid, cgrtyp, cni, cnj, cig1, cig2, cig3, cig4)
    if istat < 0:
        raise EzscintError()
    params = {
            'id'    : gdid,
            'shape' : (max(1, cni.value), max(1, cnj.value)),
            'ni'    : cni.value,
            'nj'    : cnj.value,
            'grtyp' : _C_CHAR2WCHAR(cgrtyp.value),
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value
            }
    if doSubGrid:
        params['nsubgrids'] = ezget_nsubgrids(gdid)
        params['subgridid'] = ezget_subgridids(gdid)
        params['subgrid'] = []
        if params['nsubgrids'] > 0:
            for gid2 in params['subgridid']:
                params['subgrid'].append(ezgprm(gid2))
    return params


#TODO: merge ezgprm et ezgxprm et gdgaxes (conditional axes)?
def ezgxprm(gdid, doSubGrid=False):
    """
    Get extended grid parameters

    gridParams = ezgxprm(gdid)

    Args:
        gdid      : id of the grid (int or dict)
                    Dict with key 'id' is accepted from version 2.0.rc1
        doSubGrid : recurse on subgrids if True
    Returns:
        {
            'id'    : grid id, same as input arg
            'shape'  : (ni, nj) # dimensions of the grid
            'ni'     : first dimension of the grid
            'nj'     : second dimension of the grid
            'grtyp'  : type of geographical projection
                       (one of 'Z', '#', 'Y', 'U')
            'ig1'    : first grid descriptor
            'ig2'    : second grid descriptor
            'ig3'    : third grid descriptor
            'ig4'    : fourth grid descriptor
            'grref'  : grid ref type (one of 'A', 'B', 'E', 'G', 'L', 'N', 'S')
            'ig1ref' : first grid descriptor of grid ref
            'ig2ref' : second grid descriptor of grid ref
            'ig3ref' : third grid descriptor of grid ref
            'ig4ref' : fourth grid descriptor of grid ref
        }
        For grtyp not in ('Z', '#', 'Y', 'U'), grref=' ', ig1..4ref=0
        if doSubGrid, add these
        {
            'nsubgrids' : Number of subgrids
            'subgridid' : list of subgrids id
            'subgrid'   : list of subgrids details {'id', 'shape', ...}
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Define a Z/E grid
    >>> (ni,nj) = (90, 45)
    >>> gp = {
    ...     'shape' : (ni,nj),
    ...     'ni' : ni,
    ...     'nj' : nj,
    ...     'grtyp' : 'Z',
    ...     'grref' : 'E',
    ...     'xlat1' : 0.,
    ...     'xlon1' : 180.,
    ...     'xlat2' : 0.,
    ...     'xlon2' : 270.,
    ...     'dlat' : 0.25,
    ...     'dlon' : 0.25,
    ...     'lat0' : 45.,
    ...     'lon0' : 273.
    ...     }
    >>> ig1234 = rmn.cxgaig(gp['grref'], gp['xlat1'], gp['xlon1'],
    ...                     gp['xlat2'], gp['xlon2'])
    >>> gp['ax'] = np.empty((ni,1), dtype=np.float32, order='F')
    >>> gp['ay'] = np.empty((1,nj), dtype=np.float32, order='F')
    >>> for i in range(ni): gp['ax'][i,0] = gp['lon0']+float(i)*gp['dlon']
    >>> for j in range(nj): gp['ay'][0,j] = gp['lat0']+float(j)*gp['dlat']
    >>> gid = rmn.ezgdef_fmem(ni, nj, gp['grtyp'], gp['grref'],
    ...                       ig1234[0], ig1234[1], ig1234[2], ig1234[3],
    ...                       gp['ax'], gp['ay'])
    >>>
    >>> # Get grid info
    >>> params = rmn.ezgxprm(gid)
    >>> print("# Grid type={grtyp}/{grref} of size={ni}, {nj}".format(**params))
    # Grid type=Z/E of size=90, 45

    See Also:
        ezgprm
        ezgfstp
        gdgaxes
        ezget_nsubgrids
        ezget_subgridids
        ezgdef_fmem
        rpnpy.librmn.base.cxgaig
        rpnpy.librmn.grids.decodeGrid
        rpnpy.librmn.grids.readGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    (cni, cnj) = (_ct.c_int(), _ct.c_int())
    cgrtyp = _C_MKSTR(' '*_rc.FST_GRTYP_LEN)
    (cig1, cig2, cig3, cig4) = (_ct.c_int(), _ct.c_int(),
                                _ct.c_int(), _ct.c_int())
    cgrref = _C_MKSTR(' '*_rc.FST_GRTYP_LEN)
    (cig1ref, cig2ref, cig3ref, cig4ref) = (_ct.c_int(), _ct.c_int(),
                                            _ct.c_int(), _ct.c_int())
    istat = _rp.c_ezgxprm(gdid, cni, cnj, cgrtyp, cig1, cig2, cig3, cig4,
                          cgrref, cig1ref, cig2ref, cig3ref, cig4ref)
    if istat < 0:
        raise EzscintError()
    params = {
            'id'    : gdid,
            'shape' : (max(1, cni.value), max(1, cnj.value)),
            'ni'    : cni.value,
            'nj'    : cnj.value,
            'grtyp' : _C_CHAR2WCHAR(cgrtyp.value),
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value,
            'grref' : _C_CHAR2WCHAR(cgrref.value),
            'ig1ref'   : cig1ref.value,
            'ig2ref'   : cig2ref.value,
            'ig3ref'   : cig3ref.value,
            'ig4ref'   : cig4ref.value
            }
    #TODO: ezgxprm: be more explicit on the ref values: tags, i0, j0, ...
    if doSubGrid:
        params['nsubgrids'] = ezget_nsubgrids(gdid)
        params['subgridid'] = ezget_subgridids(gdid)
        params['subgrid'] = []
        if params['nsubgrids'] > 0:
            for gid2 in params['subgridid']:
                params['subgrid'].append(ezgxprm(gid2))
    return params


def ezgfstp(gdid, doSubGrid=False):
    """
    Get the standard file attributes of the positional records

    recParams = ezgfstp(gdid)
    recParams = ezgfstp(gdid, doSubGrid=True)

    Args:
        gdid : grid id (int or dict)
               Dict with key 'id' is accepted from version 2.0.rc1
        doSubGrid : recurse on subgrids if True
    Returns:
        {
            'id'    : grid id, same as input arg
            'typvarx': x-axe type of field (forecast, analysis, climatology)
            'nomvarx': x-axe variable name
            'etikx'  : x-axe label
            'typvary': y-axe type of field (forecast, analysis, climatology)
            'nomvary': y-axe variable name
            'etiky'  : y-axe label
            'ip1'    : grid tag 1
            'ip2'    : grid tag 2
            'ip3'    : grid tag 3
            'dateo'  : date time stamp
            'deet'   : length of a time step in seconds
            'npas'   : time step number
            'nbits' : number of bits kept for the elements of the field
        }
        if doSubGrid, add these
        {
            'nsubgrids' : Number of subgrids
            'subgridid' : list of subgrids id
            'subgrid'   : list of subgrids details {'id', 'typvarx', ...}
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> myfile = os.path.join(ATM_MODEL_DFILES, 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> meGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Get standard file attributes of the positional records
    >>> params = rmn.ezgfstp(meGrid['id'])
    >>> print("# {0} grid axes are in {nomvarx} and {nomvary} records".format(meRec['nomvar'], **params))
    # ME   grid axes are in >>   and ^^   records

    See Also:
        ezgprm
        ezgxprm
        gdgaxes
        ezget_nsubgrids
        ezget_subgridids
        rpnpy.librmn.grids.decodeGrid
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    (ctypvarx, cnomvarx, cetikx) = (_C_MKSTR(' '*_rc.FST_TYPVAR_LEN),
                                    _C_MKSTR(' '*_rc.FST_NOMVAR_LEN),
                                    _C_MKSTR(' '*_rc.FST_ETIKET_LEN))
    (ctypvary, cnomvary, cetiky) = (_C_MKSTR(' '*_rc.FST_TYPVAR_LEN),
                                    _C_MKSTR(' '*_rc.FST_NOMVAR_LEN),
                                    _C_MKSTR(' '*_rc.FST_ETIKET_LEN))
    (cip1, cip2, cip3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    (cdateo, cdeet, cnpas, cnbits) = (_ct.c_int(), _ct.c_int(),
                                      _ct.c_int(), _ct.c_int())
    istat = _rp.c_ezgfstp(gdid, cnomvarx, ctypvarx, cetikx, cnomvary,
                          ctypvary, cetiky, cip1, cip2, cip3, cdateo,
                          cdeet, cnpas, cnbits)
    if istat < 0:
        raise EzscintError()
    params = {
            'id'    : gdid,
            'typvarx': _C_CHAR2WCHAR(ctypvarx.value),
            'nomvarx': _C_CHAR2WCHAR(cnomvarx.value),
            'etikx'  : _C_CHAR2WCHAR(cetikx.value),
            'typvary': _C_CHAR2WCHAR(ctypvary.value),
            'nomvary': _C_CHAR2WCHAR(cnomvary.value),
            'etiky ' : _C_CHAR2WCHAR(cetiky.value),
            'ip1'   : cip1.value,
            'ip2'   : cip2.value,
            'ip3'   : cip3.value,
            'dateo' : cdateo.value,
            'deet'  : cdeet.value,
            'npas'  : cnpas.value,
            'nbits' : cnbits.value
            }
    if doSubGrid:
        params['nsubgrids'] = ezget_nsubgrids(gdid)
        params['subgridid'] = ezget_subgridids(gdid)
        params['subgrid'] = []
        if params['nsubgrids'] > 0:
            for gid2 in params['subgridid']:
                params['subgrid'].append(ezgfstp(gid2))
    return params


def gdgaxes(gdid, ax=None, ay=None):
    """
    Gets the deformation axes of the Z, Y, #, U grids

    gridAxes = gdgaxes(gdid)
    gridAxes = gdgaxes(gdid, ax, ay)
    gridAxes = gdgaxes(gdid, gridAxes)
    gridAxes = gdgaxes(griddict)

    Args:
        gdid     : id of the grid (int)
        ax, ay   : (optional) 2 pre-allocated grid axes arrays (numpy.ndarray)
        gridAxes : (optional) gridAxes['ax'] and gridAxes['ay'] are
                   2 pre-allocated grid axes arrays (numpy.ndarray)
        griddict : dict with minimally key 'id' as id of the grid
                   optionnaly keys 'ax' and 'ay' can be provided as
                   2 pre-allocated grid axes arrays (numpy.ndarray)
    Returns:
        {
            'id' : grid id, same as input arg
            'ax' : x grid axe data (numpy.ndarray)
                   same as gridAxes['subgrid'][0]['ax']
            'ay' : y grid axe data (numpy.ndarray)
                   same as gridAxes['subgrid'][0]['ay']
            'nsubgrids' : Number of subgrids
            'subgridid' : list of subgrids id
            'subgrid'   : list of subgrids details {'id', 'ax', 'ay'}
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Get grid for ME record in file
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> rec    = rmn.fstlir(funit, nomvar='ME')
    >>> rec['iunit'] = funit
    >>> gridid = rmn.ezqkdef(rec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Get axes values for the grid
    >>> axes = rmn.gdgaxes(gridid)
    >>> print("# Got grid axes of shape: {0}, {1}"
    ...       .format(str(axes['ax'].shape), str(axes['ay'].shape)))
    # Got grid axes of shape: (201, 1), (1, 100)

    See Also:
        ezgprm
        ezgxprm
        ezgfstp
        ezget_nsubgrids
        ezget_subgridids
        ezqkdef
        rpnpy.librmn.grids.decodeGrid
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
    """
    ax = _getCheckArg(None, ax, gdid, 'ax')
    ay = _getCheckArg(None, ay, gdid, 'ay')
    ay = _getCheckArg(None, ay, ax, 'ay')
    ax = _getCheckArg(None, ax, ax, 'ax')
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        axes = []
        subgridid = ezget_subgridids(gdid)
        for id in subgridid:
            axes.append(gdgaxes(id, ax, ay))
            ax, ay = None, None
        if not len(axes):
            raise EzscintError()
        return {
                'id' : gdid,
                'ax' : axes[0]['ax'],
                'ay' : axes[0]['ay'],
                'nsubgrids' : nsubgrids,
                'subgridid' : subgridid,
                'subgrid'   : axes
                }
    gridParams = ezgxprm(gdid)
    axshape = None
    ayshape = None
    if gridParams['grtyp'].lower() == 'y':
        axshape = gridParams['shape']
        ayshape = gridParams['shape']
    elif gridParams['grtyp'].lower() in ('z', '#'):
        axshape = (gridParams['shape'][0], 1)
        ayshape = (1, gridParams['shape'][1])
    else:
        raise EzscintError("gdgaxes: grtyp/grref = {grtyp}/{grref} not supported".format(**gridParams))
    ax = _ftnOrEmpty(ax, axshape, _np.float32)
    ay = _ftnOrEmpty(ay, ayshape, _np.float32)
    if not (isinstance(ax, _np.ndarray) and isinstance(ay, _np.ndarray)):
        raise TypeError("gdgaxes: Expecting ax, ay as 2 numpy.ndarray, " +
                        "Got {0}, {1}".format(type(ax), type(ay)))
    if ax.shape != axshape or ay.shape != ayshape:
        raise TypeError("gdgaxes: provided ax, ay have the wrong shape")
    istat = _rp.c_gdgaxes(gdid, ax, ay)
    if istat >= 0:
        return {
            'id' : gdid,
            'ax' : ax,
            'ay' : ay,
            'nsubgrids' : 0,
            'subgridid' : [gdid],
            'subgrid'   : [{
                'id' : gdid,
                'ax' : ax,
                'ay' : ay
                }]
            }
    raise EzscintError()


def gdll(gdid, lat=None, lon=None):
    """
    Gets the latitude/longitude position of grid 'gdid'

    gridLatLon = gdll(gdid)
    gridLatLon = gdll(gdid, lat, lon)
    gridLatLon = gdll(gdid, gridLatLon)
    gridLatLon = gdll(griddict)

    Args:
        gdid       : id of the grid (int)
        lat, lon   : (optional) 2 pre-allocated lat, lon arrays (numpy.ndarray)
        gridLatLon : (optional) gridLatLon['lat'], gridLatLon['lon'] are
                     2 pre-allocated lat, lon arrays (numpy.ndarray)
        griddict   : dict with minimally key 'id' as id of the grid
                     optionnaly keys 'lat' and 'lon' can be provided as
                     2 pre-allocated lat, lon arrays (numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : latitude  data (numpy.ndarray)
                    same as gridLatLon['subgrid'][0]['lat']
            'lon' : longitude data (numpy.ndarray)
                    same as gridLatLon['subgrid'][0]['lon']
            'nsubgrids' : Number of subgrids
            'subgridid' : list of subgrids id
            'subgrid'   : list of subgrids {'id', 'lat', 'lon'}
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> grid = rmn.defGrid_G(90, 45)
    >>> lalo = rmn.gdll(grid['id'])
    >>> (i, j) = (45, 20)
    >>> print("# Lat, Lon of point {0}, {1} is: {2:4.1f}, {3:5.1f}"
    ...       .format(i, j, lalo['lat'][i,j], lalo['lon'][i,j]))
    # Lat, Lon of point 45, 20 is: -7.9, 180.0

    See Also:
        gdxyfll
        gdllfxy
        rpnpy.librmn.grids
    """
    lat = _getCheckArg(None, lat, gdid, 'lat')
    lon = _getCheckArg(None, lon, gdid, 'lon')
    lon = _getCheckArg(None, lon, lat, 'lon')
    lat = _getCheckArg(None, lat, lat, 'lat')
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        latlon = []
        subgridid = ezget_subgridids(gdid)
        for id in subgridid:
            latlon.append(gdll(id, lat, lon))
            lat, lon = None, None
        if not len(latlon):
            raise EzscintError()
        return {
                'id' : gdid,
                'lat' : latlon[0]['lat'],
                'lon' : latlon[0]['lon'],
                'nsubgrids' : nsubgrids,
                'subgridid' : subgridid,
                'subgrid'   : latlon
                }
    gridParams = ezgxprm(gdid)
    lat = _ftnOrEmpty(lat, gridParams['shape'], _np.float32)
    lon = _ftnOrEmpty(lon, gridParams['shape'], _np.float32)
    if not (isinstance(lat, _np.ndarray) and isinstance(lon, _np.ndarray)):
        raise TypeError("gdll: Expecting lat, lon as 2 numpy.ndarray," +
                        "Got {0}, {1}".format(type(lat), type(lon)))
    if lat.shape != gridParams['shape'] or lon.shape != gridParams['shape']:
        raise TypeError("gdll: provided lat, lon have the wrong shape")
    istat = _rp.c_gdll(gdid, lat, lon)
    if istat >= 0:
        return {
            'id'  : gdid,
            'lat' : lat,
            'lon' : lon,
            'nsubgrids' : 0,
            'subgridid' : [gdid],
            'subgrid'   : [{
                'id'  : gdid,
                'lat' : lat,
                'lon' : lon,
                }]
            }
    raise EzscintError()


def gdxyfll(gdid, lat=None, lon=None):
    """
    Returns the x-y positions of lat lon points on grid 'gdid'

    Note that provided grid points coor. are considered
    to be Fortran indexing, from 1 to ni and from 1 to nj
    While numpy/C indexing starts from 0

    pointXY = gdxyfll(gdid, lat, lon)
    pointXY = gdxyfll(griddict, lat, lon)
    pointXY = gdxyfll(gdid, gridLaLo)
    pointXY = gdxyfll(griddict)

    Args:
        gdid     : id of the grid (int)
        lat, lon : list of points lat, lon (list, tuple or numpy.ndarray)
        gridLaLo : (optional) gridLaLo['lat'], gridLaLo['lon'] are
                   2 pre-allocated lat, lon arrays (numpy.ndarray)
        griddict : dict with keys
                   'id'  : id of the grid (int)
                   'lat' : (optional) list of points lon (list, tuple or numpy.ndarray)
                   'lon' : (optional) list of points lon (list, tuple or numpy.ndarray)
    Returns:
        {
            'id' : grid id, same as input arg
            'x'  : list of points x-coor (numpy.ndarray)
            'y'  : list of points y-coor (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> grid = rmn.defGrid_G(90, 45)
    >>> (la, lo) = (45., 273.)
    >>> xy = rmn.gdxyfll(grid['id'], [la], [lo])
    >>> print("# x, y pos at lat={0}, lon={1} is: {2:5.2f}, {3:5.2f}"
    ...       .format(la, lo, xy['x'][0], xy['y'][0]))
    # x, y pos at lat=45.0, lon=273.0 is: 69.25, 34.38

    See Also:
        gdllfxy
        gdll
        rpnpy.librmn.grids
    """
    #TODO: what about multi-grids? multi values of x,y for each lat,lon pair?
    lat = _getCheckArg(None, lat, gdid, 'lat')
    lon = _getCheckArg(None, lon, gdid, 'lon')
    lon = _getCheckArg(None, lon, lat, 'lon')
    lat = _getCheckArg(None, lat, lat, 'lat')
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.size != clon.size:
        raise TypeError("gdxyfll: provided lat, lon should have the same size")
    cx = _np.empty(clat.shape, dtype=_np.float32, order='F')
    cy = _np.empty(clat.shape, dtype=_np.float32, order='F')
    istat = _rp.c_gdxyfll(gdid, cx, cy, clat, clon, clat.size)
    if istat >= 0:
        return {
            'id'  : gdid,
            'lat' : clat,
            'lon' : clon,
            'x'   : cx,
            'y'   : cy
            }
    raise EzscintError()


def gdllfxy(gdid, xpts=None, ypts=None):
    """
    Returns the lat-lon coordinates of data
    located at positions x-y on grid GDID

    Note that provided grid points coor. are considered
    to be Fortran indexing, from 1 to ni and from 1 to nj
    While numpy/C indexing starts from 0

    pointLL = gdllfxy(gdid, xpts, ypts)
    pointLL = gdllfxy(gdid, xypts)
    pointLL = gdllfxy(griddict, xpts, ypts)
    pointLL = gdllfxy(griddict)

    Args:
        gdid       : id of the grid (int)
        xpts, ypts : list of points x, y coor (list, tuple or numpy.ndarray)
        xypts      : xypts['xpts'], xypts['ypts'] are
                     2 pre-allocated xpts, ypts arrays (numpy.ndarray)
        griddict   : dict with keys
                     'id'   : id of the grid (int)
                     'xpts' : (optional) list of points x coor (list, tuple or numpy.ndarray)
                     'ypts' : (optional) list of points y coor (list, tuple or numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : list of points lat-coor (numpy.ndarray)
            'lon' : list of points lon-coor (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> grid = rmn.defGrid_G(90, 45)
    >>> (i, j) = (69, 34)
    >>> lalo = rmn.gdllfxy(grid['id'], [i], [j])
    >>> print("# Lat, Lon of point {0}, {1} is: {2:4.1f}, {3:5.1f}"
    ...       .format(i, j, lalo['lat'][0], lalo['lon'][0]))
    # Lat, Lon of point 69, 34 is: 43.5, 272.0

    See Also:
        gdll
        gdxyfll
        rpnpy.librmn.grids
    """
    #TODO: what about multi-grids? multi values of lat,lon for each x,y pair?
    xpts = _getCheckArg(None, xpts, gdid, 'xpts')
    ypts = _getCheckArg(None, ypts, gdid, 'ypts')
    ypts = _getCheckArg(None, ypts, xpts, 'ypts')
    xpts = _getCheckArg(None, xpts, xpts, 'xpts')
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    cx = _list2ftnf32(xpts)
    cy = _list2ftnf32(ypts)
    if not (isinstance(cx, _np.ndarray) and isinstance(cy, _np.ndarray)):
        raise TypeError("xpts and ypts must be arrays")
    if cx.size != cy.size:
        raise TypeError(
            "gdllfxy: provided xpts, ypts should have the same size")
    clat = _np.empty(cx.shape, dtype=_np.float32, order='F')
    clon = _np.empty(cx.shape, dtype=_np.float32, order='F')
    istat = _rp.c_gdllfxy(gdid, clat, clon, cx, cy, cx.size)
    if istat >= 0:
        return {
            'id'  : gdid,
            'lat' : clat,
            'lon' : clon,
            'x'   : cx,
            'y'   : cy
            }
    raise EzscintError()


def gdgetmask(gdid, mask=None):
    """
    Returns the mask associated with grid 'gdid'

    Note that if no mask was previouly set on this grid,
    gdgetmask will raise an EzscintError

    mask = gdgetmask(gdid)
    mask = gdgetmask(gdid, mask)
    mask = gdgetmask(griddict, mask)
    mask = gdgetmask(griddict)

    Args:
        gdid     : id of the grid (int)
        mask     : mask array (numpy.ndarray)
        griddict : dict with keys
                   'id'   : id of the grid (int)
                   'mask' : (optional) mask array (numpy.ndarray)
    Returns:
        mask array (numpy.ndarray)
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> mgRec  = rmn.fstlir(funit, nomvar='MG')
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Set a mask
    >>> mask = np.rint(mgRec['d']).astype(np.intc)
    >>> rmn.gdsetmask(inGrid['id'], mask)
    >>> # ...
    >>>
    >>> # Get the mask back
    >>> mask2 = rmn.gdgetmask(inGrid['id'])

    See Also:
        gdsetmask
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    gridParams = ezgxprm(gdid)
    mask = _ftnOrEmpty(mask, gridParams['shape'], _np.int32)
    if not (isinstance(mask, _np.ndarray) and
            mask.shape == gridParams['shape'] and
            mask.dtype in (_np.intc, _np.int32)):
        raise TypeError("Wrong mask type,shape numpy.ndarray: {0}, {1}"\
                        .format(type(mask), repr(gridParams['shape'])))
    istat = _rp.c_gdgetmask(gdid, mask)
    if istat < 0:
        raise EzscintError('gdgetmask: Problem getting the mask for grid id={0}'.format(gdid))
    return mask


#TODO:    c_gdxpncf(gdid, i1, i2, j1, j2)
#TODO:    c_gdgxpndaxes(gdid, ax, ay)


#---- Interpolation Functions

#TODO: ezsint, when given dict for grids, return dict then (would need new fn)?
def ezsint(gdidout, gdidin, zin, zout=None):
    """
    Scalar horizontal interpolation

    zout = ezsint(gdidout, gdidin, zin)
    zout = ezsint(gdidout, gdidin, zin, zout)

    Args:
        gdidout : output grid id (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        gdidid  : grid id describing zin grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        zin     : data to interpolate (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        zout    : optional, interp.result array (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
    Returns:
        numpy.ndarray, interpolation result
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Define a destination Grid
    >>> (ni, nj, lat0, lon0, dlat, dlon) = (200, 100, 35.,265.,0.25,0.25)
    >>> outGrid  = rmn.defGrid_L(ni, nj, lat0, lon0, dlat, dlon)
    >>>
    >>> # Interpolate ME linearly
    >>> rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
    >>> me = rmn.ezsint(outGrid['id'], inGrid['id'], meRec['d'])

    See Also:
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdidout = _getCheckArg(int, gdidout, gdidout, 'id')
    gdidin  = _getCheckArg(int, gdidin, gdidin, 'id')
    zin     = _getCheckArg(_np.ndarray, zin, zin, 'd')
    zout    = _getCheckArg(None, zout, zout, 'd')
    gridsetid = ezdefset(gdidout, gdidin)
    gridParams = ezgxprm(gdidin)
    zin  = _ftnf32(zin)
    if zin.shape != gridParams['shape']:
        raise TypeError("Provided zin array have inconsistent " +
                        "shape compared to the input grid")
    dshape = ezgprm(gdidout)['shape']
    zout = _ftnOrEmpty(zout, dshape, zin.dtype)
    if not (isinstance(zout, _np.ndarray) and zout.shape == dshape):
        raise TypeError("Wrong type,shape for zout: {0}, {1}"\
                        .format(type(zout), repr(dshape)))
    istat = _rp.c_ezsint(zout, zin)
    if istat >= 0:
        return zout
    raise EzscintError()


#TODO: ezuvint, when given dict for grids, return dict then (would need new fn)?
def ezuvint(gdidout, gdidin, uuin, vvin, uuout=None, vvout=None):
    """
    Vectorial horizontal interpolation

    (uuout, vvout) = ezuvint(gdidout, gdidin, uuin, vvin)
    (uuout, vvout) = ezuvint(gdidout, gdidin, uuin, vvin, uuout, vvout)

    Args:
        gdidout : output grid id (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        gdidid  : grid id describing uuin grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        uuin    : data x-part to interpolate (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        vvin    : data y-part to interpolate (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        uuout   : interp.result array x-part (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        vvout   : interp.result array y-part (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
    Returns:
        interpolation result (numpy.ndarray, numpy.ndarray)
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Define a destination Grid
    >>> (ni, nj, lat0, lon0, dlat, dlon) = (200, 100, 35.,265.,0.25,0.25)
    >>> outGrid  = rmn.defGrid_L(ni, nj, lat0, lon0, dlat, dlon)
    >>>
    >>> # Interpolate U/V vectorially
    >>> (uu, vv) = rmn.ezuvint(outGrid['id'], inGrid['id'], uuRec['d'], vvRec['d'])

    See Also:
        ezsint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdidout = _getCheckArg(int, gdidout, gdidout, 'id')
    gdidin  = _getCheckArg(int, gdidin, gdidin, 'id')
    uuin    = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin    = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    uuout   = _getCheckArg(None, uuout, uuout, 'd')
    vvout   = _getCheckArg(None, vvout, vvout, 'd')
    gridsetid = ezdefset(gdidout, gdidin)
    gridParams = ezgxprm(gdidin)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape'] or vvin.shape != gridParams['shape']:
        raise TypeError("ezuvint: Provided uuin, vvin array have " +
                        "inconsistent shape compared to the input grid")
    dshape = ezgprm(gdidout)['shape']
    uuout = _ftnOrEmpty(uuout, dshape, uuin.dtype)
    vvout = _ftnOrEmpty(vvout, dshape, uuin.dtype)
    if not (isinstance(uuout, _np.ndarray) and
            isinstance(vvout, _np.ndarray)):
        raise TypeError("ezuvint: Expecting uuout, vvout of type " +
                        "numpy.ndarray, Got {0}".format(type(uuout)))
    if uuout.shape != dshape or vvout.shape != dshape:
        raise TypeError("ezuvint: Provided uuout, vvout array have " +
                        "inconsistent shape compered to the output grid")
    istat = _rp.c_ezuvint(uuout, vvout, uuin, vvin)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


def gdllsval(gdid, lat, lon, zin, zout=None):
    """
    Scalar interpolation to points located at lat-lon coordinates

    zout = gdllsval(gdid, lat, lon, zin)
    zout = gdllsval(gdid, lat, lon, zin, zout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        lat     : list of resquested points lat (list or numpy.ndarray)
        lon     : list of resquested points lon (list or numpy.ndarray)
        zin     : data to interpolate, on grid gdid (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        zout    : optional, interp.result array,
                  same shape a lat, lon (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate ME to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> meValues = rmn.gdllsval(inGrid['id'], la,lo, meRec['d'])

    See Also:
        gdxysval
        gdllvval
        gdxyvval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    zin  = _getCheckArg(_np.ndarray, zin, zin, 'd')
    gridParams = ezgxprm(gdid)
    zin  = _ftnf32(zin)
    if zin.shape != gridParams['shape']:
        raise TypeError("gdllsval: Provided zin array have inconsistent " +
                        "shape compered to the input grid")
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.shape != clon.shape:
        raise TypeError("Provided lat, lon arrays have inconsistent shapes")
    dshape = clat.shape
    zout = _ftnOrEmpty(zout, dshape, zin.dtype)
    if not (isinstance(zout, _np.ndarray) and zout.shape == dshape):
        raise TypeError("Wrong type,shape for zout: {0}, {1}"\
                        .format(type(zout), repr(dshape)))
    istat = _rp.c_gdllsval(gdid, zout, zin, clat, clon, clat.size)
    if istat >= 0:
        return zout
    raise EzscintError()


def gdxysval(gdid, xpts, ypts, zin, zout=None):
    """
    Scalar intepolation to points located at x-y coordinates

    Note that provided grid points coor. are considered
    to be Fortran indexing, from 1 to ni and from 1 to nj
    While numpy/C indexing starts from 0

    zout = gdxysval(gdid, xpts, ypts, zin)
    zout = gdxysval(gdid, xpts, ypts, zin, zout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        xpts    : list of resquested points x-coor (list or numpy.ndarray)
        ypts    : list of resquested points y-coor (list or numpy.ndarray)
        zin     : data to interpolate, on grid gdid (numpy.ndarray or dict)
                  Dict with key 'd' is accepted from version 2.0.rc1
        zout    : optional, interp.result array, same shape a xpts, ypts
                  (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a xpts, ypts
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', 'geophy.fst')
    >>> funit  = rmn.fstopenall(myfile)
    >>> meRec  = rmn.fstlir(funit, nomvar='ME')
    >>> inGrid = rmn.readGrid(funit, meRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate ME to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> xy = rmn.gdxyfll(inGrid['id'], la, lo)
    >>> meValues = rmn.gdxysval(inGrid['id'], xy['x'], xy['y'], meRec['d'])

    See Also:
        gdllsval
        gdllvval
        gdxyvval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    zin  = _getCheckArg(_np.ndarray, zin, zin, 'd')
    gridParams = ezgxprm(gdid)
    zin  = _ftnf32(zin)
    if zin.shape != gridParams['shape']:
        raise TypeError("gdxysval: Provided zin array have inconsistent " +
                        "shape compered to the input grid")
    cx = _list2ftnf32(xpts)
    cy = _list2ftnf32(ypts)
    if not (isinstance(cx, _np.ndarray) and isinstance(cy, _np.ndarray)):
        raise TypeError("xpts and ypts must be arrays")
    if cx.size != cy.size:
        raise TypeError(
            "provided xpts, ypts should have the same size")
    dshape = cx.shape
    zout = _ftnOrEmpty(zout, dshape, zin.dtype)
    if not (isinstance(zout, _np.ndarray) and zout.shape == dshape):
        raise TypeError("Wrong type,shape for zout: {0}, {1}"\
                        .format(type(zout), repr(dshape)))
    istat = _rp.c_gdxysval(gdid, zout, zin, cx, cy, cx.size)
    if istat >= 0:
        return zout
    raise EzscintError()


def gdllvval(gdid, lat, lon, uuin, vvin, uuout=None, vvout=None):
    """
    Vectorial interpolation to points located at lat-lon coordinates

    (uuout, vvout) = gdllsval(gdid, lat, lon, uuin, vvin)
    (uuout, vvout) = gdllsval(gdid, lat, lon, uuin, vvin, uuout, vvout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        lat     : list of resquested points lat (list or numpy.ndarray)
        lon     : list of resquested points lon (list or numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        uuout, vvout : optional, interp.result array, same shape a lat, lon
                       (numpy.ndarray)
    Returns:
        (uuout, vvout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> (uu, vv) = rmn.gdllvval(inGrid['id'], la, lo, uuRec['d'], vvRec['d'])

    See Also:
        gdllsval
        gdxysval
        gdxyvval
        gdllwdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    uuin = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    gridParams = ezgxprm(gdid)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.shape != clon.shape:
        raise TypeError("Provided lat, lon arrays have inconsistent shapes")
    dshape = clat.shape
    uuout = _ftnOrEmpty(uuout, dshape, uuin.dtype)
    vvout = _ftnOrEmpty(vvout, dshape, uuin.dtype)
    if not (isinstance(uuout, _np.ndarray) and uuout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(uuout), repr(dshape)))
    if not (isinstance(vvout, _np.ndarray) and vvout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(vvout), repr(dshape)))
    istat = _rp.c_gdllvval(gdid, uuout, vvout, uuin, vvin, clat,
                           clon, clat.size)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


def gdxyvval(gdid, xpts, ypts, uuin, vvin, uuout=None, vvout=None):
    """
    Vectorial intepolation to points located at x-y coordinates

    Note that provided grid points coor. are considered
    to be Fortran indexing, from 1 to ni and from 1 to nj
    While numpy/C indexing starts from 0

    (uuout, vvout) = gdxysval(gdid, xpts, ypts, uuin, vvin)
    (uuout, vvout) = gdxysval(gdid, xpts, ypts, uuin, vvin, uuout, vvout)

    Args:
        gdid     : id of the grid(int or dict)
                   Dict with key 'id' is accepted from version 2.0.rc1
        xpts     : list of resquested points x-coor (list or numpy.ndarray)
        ypts     : list of resquested points y-coor (list or numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        uuout, vvout : optional, interp.result array, same shape a xpts, ypts
                       (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a xpts, ypts
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> xy = rmn.gdxyfll(inGrid['id'], la, lo)
    >>> (uu, vv) = rmn.gdxyvval(inGrid['id'], xy['x'], xy['y'], uuRec['d'], vvRec['d'])

    See Also:
        gdllsval
        gdxysval
        gdllvval
        gdllwdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    uuin = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    gridParams = ezgxprm(gdid)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    cx = _list2ftnf32(xpts)
    cy = _list2ftnf32(ypts)
    if not (isinstance(cx, _np.ndarray) and isinstance(cy, _np.ndarray)):
        raise TypeError("xpts and ypts must be arrays")
    if cx.size != cy.size:
        raise TypeError(
            "provided xpts, ypts should have the same size")
    dshape = cx.shape
    uuout = _ftnOrEmpty(uuout, dshape, uuin.dtype)
    vvout = _ftnOrEmpty(vvout, dshape, uuin.dtype)
    if not (isinstance(uuout, _np.ndarray) and uuout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(uuout), repr(dshape)))
    if not (isinstance(vvout, _np.ndarray) and vvout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(vvout), repr(dshape)))
    istat = _rp.c_gdxyvval(gdid, uuout, vvout, uuin, vvin,
                           cx, cy, cx.size)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()

#TODO:    c_ezsint_mdm(zout, mask_out, zin, mask_in)
#TODO:    c_ezuvint_mdm(uuout, vvout, mask_out, uuin, vvin, mask_in)
#TODO:    c_ezsint_mask(mask_out, mask_in)

#TODO: need fixing before making public
def _gdwdfuv(gdid, lat, lon, uuin, vvin, wsout=None, wdout=None):
    """
    Converts, on grid 'gdid', the grid winds at grid points speed/direction.
    The lat/lon coordinates of each point have to be present.

    (ws, wd) = gdwdfuv(gdid, lat, lon, uuin, vvin)
    (ws, wd) = gdwdfuv(gdid, lat, lon, uuin, vvin, wsout, wdout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        lat     : list of resquested points lat (list or numpy.ndarray)
        lon     : list of resquested points lon (list or numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        wsout, wdout : optional, interp.result array, same shape a lat, lon
                       wind-speed and wind-direction (numpy.ndarray)
    Returns:
        (wsout, wdout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> ni, nj = uuRec['d'].shape
    >>> destPoints = ((ni//4, nj//2), (ni//2, nj//2))
    >>> xx = [float(x[0]) for x in destPoints]
    >>> yy = [float(x[1]) for x in destPoints]
    >>> ll = gdllfxy(inGrid['id'], xx, yy)
    >>> # (ws, wd) = rmn.gdwdfuv(inGrid['id'], ll['lat'], ll['lon'], uuRec['d'], vvRec['d'])


    See Also:
        gduvfwd
        gdllsval
        gdxysval
        gdllvval
        gdxyvval
        gdllwdval
        gdxywdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    uuin = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    gridParams = ezgxprm(gdid)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.shape != clon.shape:
        raise TypeError("Provided lat, lon arrays have inconsistent shapes")
    dshape = clat.shape
    wsout = _ftnOrEmpty(wsout, dshape, uuin.dtype)
    wdout = _ftnOrEmpty(wdout, dshape, uuin.dtype)
    if not (isinstance(wsout, _np.ndarray) and wsout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wsout), repr(dshape)))
    if not (isinstance(wdout, _np.ndarray) and wdout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wdout), repr(dshape)))
    istat = _rp.c_gdwdfuv(gdid, wsout, wdout, uuin, vvin, clat,
                          clon, clat.size)
    if istat >= 0:
        return (wsout, wdout)
    raise EzscintError()


#TODO: need fixing before making public
def _gduvfwd(gdid, lat, lon, wsin, wdin, uuout=None, vvout=None):
    """
    Converts, on grid 'gdid', the direction/speed values at grid points
    to grid coordinates.
    The lat/lon coordinates of each point have to be present.

    (uuout, vvout) = gdwdfuv(gdid, lat, lon, wsin, wdin)
    (uuout, vvout) = gdwdfuv(gdid, lat, lon, wsin, wdin, uuout, vvout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        lat     : list of resquested points lat (list or numpy.ndarray)
        lon     : list of resquested points lon (list or numpy.ndarray)
        wsin, wdin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        uuout, vvout : optional, interp.result array, same shape a lat, lon
                       (numpy.ndarray)
    Returns:
        (uuout, vvout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import numpy as np
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> lalo = rmn.gdll(inGrid['id'])
    >>> # (ws, wd) = rmn.gdwdfuv(inGrid['id'], lalo['lat'], lalo['lon'], uuRec['d'], vvRec['d'])
    >>> # (uu, vv) = rmn.gduvfwd(inGrid['id'], lalo['lat'], lalo['lon'], ws, wd)

    See Also:
        gduvfwd
        gdllsval
        gdxysval
        gdllvval
        gdxyvval
        gdllwdval
        gdxywdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    wsin = _getCheckArg(_np.ndarray, wsin, wsin, 'd')
    wdin = _getCheckArg(_np.ndarray, wdin, wdin, 'd')
    gridParams = ezgxprm(gdid)
    wsin  = _ftnf32(wsin)
    wdin  = _ftnf32(wdin)
    if wsin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided wsin array have inconsistent " +
                        "shape compered to the input grid")
    if wdin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided wdin array have inconsistent " +
                        "shape compered to the input grid")
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.shape != clon.shape:
        raise TypeError("Provided lat, lon arrays have inconsistent shapes")
    dshape = clat.shape
    uuout = _ftnOrEmpty(uuout, dshape, wsin.dtype)
    vvout = _ftnOrEmpty(vvout, dshape, wsin.dtype)
    if not (isinstance(uuout, _np.ndarray) and uuout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(uuout), repr(dshape)))
    if not (isinstance(vvout, _np.ndarray) and vvout.shape == dshape):
        raise TypeError("Wrong type,shape for uuout: {0}, {1}"\
                        .format(type(vvout), repr(dshape)))
    istat = _rp.c_gdwdfuv(gdid, uuout, vvout, wsin, wdin, clat,
                          clon, clat.size)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


def gdllwdval(gdid, lat, lon, uuin, vvin, wsout=None, wdout=None):
    """
    Vectorial interpolation of points located at lat-lon coordinates,
    returned as speed and direction (WS and WD).

    (ws, wd) = gdllwdval(gdid, lat, lon, uuin, vvin)
    (ws, wd) = gdllwdval(gdid, lat, lon, uuin, vvin, wsout, wdout)

    Args:
        gdid    : id of the grid (int or dict)
                  Dict with key 'id' is accepted from version 2.0.rc1
        lat     : list of resquested points lat (list or numpy.ndarray)
        lon     : list of resquested points lon (list or numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        wsout, wdout : optional, interp.result array, same shape a lat, lon
                       wind-speed and wind-direction (numpy.ndarray)
    Returns:
        (wsout, wdout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> (ws, wd) = rmn.gdllwdval(inGrid['id'], la, lo, uuRec['d'], vvRec['d'])
    >>> print("# (ws,wd) pt1:({}, {}) pt2:({}, {})"
    ...       .format(int(round(ws[0])), int(round(wd[0])),
    ...               int(round(ws[1])), int(round(wd[1]))))
    # (ws,wd) pt1:(8, 119) pt2:(12, 125)

    See Also:
        gdllsval
        gdxysval
        gdllvval
        gdxyvval
        gdxywdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    uuin = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    gridParams = ezgxprm(gdid)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    clat = _list2ftnf32(lat)
    clon = _list2ftnf32(lon)
    if not (isinstance(clat, _np.ndarray) and isinstance(clon, _np.ndarray)):
        raise TypeError("lat and lon must be arrays: {0}, {1}".
                        format(type(clat), type(clon)))
    if clat.shape != clon.shape:
        raise TypeError("Provided lat, lon arrays have inconsistent shapes")
    dshape = clat.shape
    wsout = _ftnOrEmpty(wsout, dshape, uuin.dtype)
    wdout = _ftnOrEmpty(wdout, dshape, uuin.dtype)
    if not (isinstance(wsout, _np.ndarray) and wsout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wsout), repr(dshape)))
    if not (isinstance(wdout, _np.ndarray) and wdout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wdout), repr(dshape)))
    istat = _rp.c_gdllwdval(gdid, wsout, wdout, uuin, vvin, clat,
                            clon, clat.size)
    if istat >= 0:
        return (wsout, wdout)
    raise EzscintError()


def gdxywdval(gdid, xpts, ypts, uuin, vvin, wsout=None, wdout=None):
    """
    Vectorial intepolation to points located at x-y coordinates
    returned as speed and direction (WS and WD).

    Note that provided grid points coor. are considered
    to be Fortran indexing, from 1 to ni and from 1 to nj
    While numpy/C indexing starts from 0

    (wsout, wdout) = gdxywdval(gdid, xpts, ypts, uuin, vvin)
    (wsout, wdout) = gdxywdval(gdid, xpts, ypts, uuin, vvin, wsout, wdout)

    Args:
        gdid     : id of the grid(int or dict)
                   Dict with key 'id' is accepted from version 2.0.rc1
        xpts     : list of resquested points x-coor (list or numpy.ndarray)
        ypts     : list of resquested points y-coor (list or numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray or dict)
                       Dict with key 'd' is accepted from version 2.0.rc1
        wsout, wdout : optional, interp.result array, same shape a xpts, ypts
                       wind-speed and wind-direction (numpy.ndarray)
    Returns:
        (wsout, wdout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Read source data and define its grid
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk', '2009042700_000')
    >>> funit  = rmn.fstopenall(myfile)
    >>> uuRec  = rmn.fstlir(funit, nomvar='UU', ip1=93423264)
    >>> vvRec  = rmn.fstlir(funit, nomvar='VV', ip1=uuRec['ip1'], ip2=uuRec['ip2'])
    >>> inGrid = rmn.readGrid(funit, uuRec)
    >>> rmn.fstcloseall(funit)
    >>>
    >>> # Interpolate UV vectorially to a specific set of points
    >>> destPoints = ((35.5, 10.), (36., 10.5))
    >>> la = [x[0] for x in destPoints]
    >>> lo = [x[1] for x in destPoints]
    >>> xy = rmn.gdxyfll(inGrid['id'], la, lo)
    >>> (ws, wd) = rmn.gdxywdval(inGrid['id'], xy['x'], xy['y'], uuRec['d'], vvRec['d'])
    >>> print("# (ws,wd) pt1:({}, {}) pt2:({}, {})"
    ...       .format(int(round(ws[0])), int(round(wd[0])),
    ...               int(round(ws[1])), int(round(wd[1]))))
    # (ws,wd) pt1:(8, 119) pt2:(12, 125)

    See Also:
        gdllsval
        gdxysval
        gdllvval
        gdxyvval
        gdllwdval
        ezsint
        ezuvint
        ezsetopt
        rpnpy.librmn.const
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(int, gdid, gdid, 'id')
    uuin = _getCheckArg(_np.ndarray, uuin, uuin, 'd')
    vvin = _getCheckArg(_np.ndarray, vvin, vvin, 'd')
    gridParams = ezgxprm(gdid)
    uuin  = _ftnf32(uuin)
    vvin  = _ftnf32(vvin)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    cx = _list2ftnf32(xpts)
    cy = _list2ftnf32(ypts)
    if not (isinstance(cx, _np.ndarray) and isinstance(cy, _np.ndarray)):
        raise TypeError("xpts and ypts must be arrays")
    if cx.size != cy.size:
        raise TypeError(
            "provided xpts, ypts should have the same size")
    dshape = cx.shape
    wsout = _ftnOrEmpty(wsout, dshape, uuin.dtype)
    wdout = _ftnOrEmpty(wdout, dshape, uuin.dtype)
    if not (isinstance(wsout, _np.ndarray) and wsout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wsout), repr(dshape)))
    if not (isinstance(wdout, _np.ndarray) and wdout.shape == dshape):
        raise TypeError("Wrong type,shape for wsout: {0}, {1}"\
                        .format(type(wdout), repr(dshape)))
    istat = _rp.c_gdxywdval(gdid, wsout, wdout, uuin, vvin,
                            cx, cy, cx.size)
    if istat >= 0:
        return (wsout, wdout)
    raise EzscintError()


#---- Other Functions

def gdrls(gdid):
    """
    Frees a previously allocated grid

    gdrls(gdid)

    Args:
        gdid : grid id to free/release (int, list or dict)
               Dict with key 'id' is accepted from version 2.0.rc1
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>>
    >>> # Define a Grid
    >>> (ni, nj, lat0, lon0, dlat, dlon) = (200, 100, 35.,265.,0.25,0.25)
    >>> grid  = rmn.defGrid_L(ni, nj, lat0, lon0, dlat, dlon)
    >>>
    >>> # Release memory associated with grid info
    >>> rmn.gdrls(grid)

    See Also:
        ezqkdef
        ezgdef_fmem
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.defGrid_L
        rpnpy.librmn.grids.encodeGrid
    """
    gdid = _getCheckArg(None, gdid, gdid, 'id')
    if not isinstance(gdid, (list, tuple)):
        gdid = [gdid]
    for id1 in gdid:
        if not isinstance(id1, _integer_types):
            raise TypeError("gdrls: Expecting gdid of type int, Got {0}"\
                            .format(type(id1)))
        istat = _rp.c_gdrls(id1)
        if istat < 0:
            raise EzscintError()
    return None


def ezcalcdist(lat1, lon1, lat2, lon2):
    """
    This function computes the distance between 2 latlon points
    on the sphere (double precision).

    Source of the formula :
        http://mathworld.wolfram.com/GreatCircle.html

    dist = ezcalcdist(lat1, lon1, lat2, lon2)

    Args:
        lat1, lon1 : latitude, longitude of the first point [deg]
        lat2, lon2 : latitude, longitude of second point [deg]
    Returns:
        float, distance on the sphere between the points [m]
    Raises:
        TypeError on wrong input arg types

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> dist = rmn.ezcalcdist(45, 270, 45, 271)
    >>> print("dist = {}".format(int(dist)))
    dist = 78626

    See Also:
        ezcalcarea
    """
    (clat1, clon1, clat2, clon2) = (_ct.c_float(lat1), _ct.c_float(lon1),
                                    _ct.c_float(lat2), _ct.c_float(lon2))
    dist = 0
    cdist = _ct.c_double(dist)
    _rp.c_ez_calcdist2(_ct.byref(cdist), clat1, clon1, clat2, clon2)
    return cdist.value


def ezcalcarea(lats, lons):
    """
    This function computes the area on the sphere of
    the solid polygon formed by 2 or 4 latlon points.

    Source of the formula :
        http://mathworld.wolfram.com/GreatCircle.html
        http://mathworld.wolfram.com/SphericalTrigonometry.html
        http://mathworld.wolfram.com/SphericalTriangle.html

    area = ezcalcarea(lats, lons)

    Args:
        lats : latitudes of 2/4 points defining the rectangle/polygone [deg]
        lons : longitude of 2/4 points defining the rectangle/polygone [deg]
    Returns:
        float, area on the sphere [m^2]
    Raises:
        TypeError   on wrong input arg types
        ValueError  on wrong number of inputs values

    Examples:
    >>> import rpnpy.librmn.all as rmn
    >>> area = rmn.ezcalcarea((45,46),(270,271))
    >>> print("area = {}".format(int(area)))
    area = 8666027008
    >>> area = rmn.ezcalcarea((45,45,46,46),(270,271,270,271))
    >>> print("area = {}".format(int(area)))
    area = 8666027008

    See Also:
        ezcalcdist
    """
    if len(lats) != len(lons):
        raise ValueError("ezcalcarea: should provide same number of lats and lons")
    area = 0
    if len(lats) == 2:
        (clat1, clon1, clat2, clon2) = (_ct.c_float(lats[0]),
                                        _ct.c_float(lons[0]),
                                        _ct.c_float(lats[1]),
                                        _ct.c_float(lons[1]))
        carea = _ct.c_float(area)
        _rp.c_ez_calcarea_rect(_ct.byref(carea), clat1, clon1, clat2, clon2)
    elif len(lats) == 4:
        clats = _ftnf32(_list2ftnf32(lats))
        clons = _ftnf32(_list2ftnf32(lons))
        carea = _ct.c_double(area)
        _rp.c_ez_calcarea2(_ct.byref(carea), clats, clons)
    else:
        raise ValueError("ezcalcarea: should provide 2 or 4 lats,lons pairs")
    return carea.value


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
