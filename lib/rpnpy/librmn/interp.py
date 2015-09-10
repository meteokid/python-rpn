#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
Module librmn.interp contains python wrapper to
main librmn's interp (ezscint) C functions
 
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

import ctypes as _ct
import numpy  as _np
from . import proto as _rp
from . import const as _rc
from . import RMNError

#TODO: make sure caller can provide allocated array (recycle mem)

#---- helpers -------------------------------------------------------

C_MKSTR = _ct.create_string_buffer

class EzscintError(RMNError):
    """General EzscintError module error/exception
    """
    pass

#---- interp (ezscint) ----------------------------------------------

#---- Set Functions

def ezsetopt(option, value):
    """Sets a floating point numerical option from the package
    
    ezsetopt(option, value)
    
    Args:
        option : option name (string)
        value  : option value (int, float or string)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(option) == str):
        raise TypeError("ezsetopt: expecting args of type str, Got %s" %
                        (type(option)))
    if type(value) == int:
        istat = _rp.c_ezsetival(option, value)
    elif type(value) == float:
        istat = _rp.c_ezsetval(option, value)
    elif type(value) == str:
        istat = _rp.c_ezsetopt(option, value)
    else:
        raise TypeError("ezsetopt: Not a supported type %s" % (type(value)))
    if istat >= 0:
        return None
    raise EzscintError()

#TODO: should we merge ezqkdef et ezgdef_fmem?
def ezqkdef(ni, nj=None, grtyp=None, ig1=None, ig2=None, ig3=None, ig4=None,
            iunit=0):
    """Universal grid definition. Applicable to all cases.
    
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
    """
    if type(ni) == dict:
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
    gdid = _rp.c_ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_fmem(ni, nj=None, grtyp=None, grref=None, ig1=None, ig2=None,
                ig3=None, ig4=None, ax=None, ay=None):
    """Generic grid definition except for 'U' grids (with necessary
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
    """
    if type(ni) == dict:
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
    if not (ax.dtype == _np.float32 and ax.flags['F_CONTIGUOUS']):
        ax = _np.asfortranarray(ax, dtype=_np.float32)    
    if not (ay.dtype == _np.float32 and ay.flags['F_CONTIGUOUS']):
        ay = _np.asfortranarray(ay, dtype=_np.float32)    
    gdid = _rp.c_ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_supergrid(ni, nj, grtyp, grref, vercode, subgridid):
    """U grid definition
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
    """
    if type(ni) == dict:
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
    gdid = _rp.c_ezgdef_supergrid(ni, nj, grtyp, grref, vercode, nsubgrids,
                                  csubgridid)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezdefset(gdidout, gdidin):
    """Defines a set of grids for interpolation
    
    gridsetid = ezdefset(gdidout, gdidin)

    Args:
        gdidout : output grid id (int) 
        gdidin  : input  grid id (int) 
    Returns:
        int, grid set id
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdidout) == int and type(gdidin) == int):
        raise TypeError("ezdefset: Expecting a grid ids of type int, " +
                        "Got %s, %s" % (type(gdidout), type(gdidin)))
    istat = _rp.c_ezdefset(gdidout, gdidin)
    if istat < 0:
        raise EzscintError()
    return istat


def gdsetmask(gdid, mask):
    """Associates a permanent mask with grid 'gdid'
    
    gdsetmask(gdid, mask)
    
    Args:
        gdid : grid id (int)
        mask : field mask (numpy.ndarray)
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int and type(mask) == _np.ndarray):
        raise TypeError("gdsetmask: Expecting args of type int, _np.ndarray " +
                        "Got %s, %s" % (type(gdid), type(mask)))
    if not mask.dtype in (_np.intc, _np.int32):
        raise TypeError("gdsetmask: Expecting mask arg of type numpy, " +
                        "intc Got %s, %s" % (type(mask.dtype)))
    if not mask.flags['F_CONTIGUOUS']:
        mask = _np.asfortranarray(mask, dtype=mask.dtype)
    istat = _rp.c_gdsetmask(gdid, mask)
    if istat < 0:
        raise EzscintError()
    return istat


#---- Query Functions


def ezgetopt(option, vtype=int):
    """Gets an option value from the package
    
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
    """
    if not (type(option) == str):
        raise TypeError("ezgetopt: expecting args of type str, Got %s" %
                        (type(option)))
    if vtype == int:
        cvalue = _ct.c_int()
        istat = _rp.c_ezgetival(option, cvalue)
    elif vtype == float:
        cvalue = _ct.c_float()
        istat = _rp.c_ezgetval(option, cvalue)
    elif vtype == str:
        cvalue = C_MKSTR(' '*64)
        istat = _rp.c_ezgetopt(option, cvalue)
    else:
        raise TypeError("ezgetopt: Not a supported type %s" % (repr(vtype)))
    if istat >= 0:
        return cvalue.value
    raise EzscintError()


def ezget_nsubgrids(super_gdid):
    """Gets the number of subgrids from the 'U' (super) grid id
    
    nsubgrids = ezget_nsubgrids(super_gdid)
    
    Args:
        super_gdid (int): id of the super grid
    Returns:
        int, number of sub grids associated with super_gdid
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(super_gdid) == int):
        raise TypeError("ezgetival: expecting args of type int, Got %s" %
                        (type(super_gdid)))
    nsubgrids = _rp.c_ezget_nsubgrids(super_gdid)
    if nsubgrids >= 0:
        return nsubgrids
    raise EzscintError()


def ezget_subgridids(super_gdid):
    """Gets the list of grid ids for the subgrids in the 'U' grid (super_gdid).
    
    subgridids = ezget_subgridids(super_gdid)
    Args:
        super_gdid (int): id of the super grid
    Returns:
        int, list of grid ids for the subgrids
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    nsubgrids = ezget_nsubgrids(super_gdid)
    cgridlist = _np.empty(nsubgrids, dtype=_np.intc, order='FORTRAN')
    istat = _rp.c_ezget_subgridids(super_gdid, cgridlist)
    if istat >= 0:
        return cgridlist.tolist()
    raise EzscintError()


def ezgprm(gdid, doSubGrid=False):
    """Get grid parameters
    
    gridParams = ezgprm(gdid)
    
    Args:
        gdid      : id of the grid (int)
        doSubGrid : recurse
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
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgprm: expecting args of type int, Got %s" %
                        (type(gdid)))
    (cni, cnj) = (_ct.c_int(), _ct.c_int())
    (cgrtyp, cig1, cig2, cig3, cig4) = (C_MKSTR(' '*_rc.FST_GRTYP_LEN),
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
            'grtyp' : cgrtyp.value,
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value
            }
    if doSubGrid:
        params['nsubgrids'] = ezget_nsubgrids(gdid)
        params['subgridid'] = ezget_subgridids(gdid)
        params['subgrid'] = [gdid]
        if params['nsubgrids'] > 1:
            params['subgrid'] = []
            for gid2 in params['subgridid']:
                params['subgrid'].append(ezgprm(gid2))
    return params
            
#TODO: merge ezgprm et ezgxprm et gdgaxes (conditional axes)?
def ezgxprm(gdid, doSubGrid=False):
    """Get extended grid parameters
    
    gridParams = ezgxprm(gdid)
    
    Args:
        gdid (int): id of the grid
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
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgxprm: expecting args of type int, Got %s" %
                        (type(gdid)))
    (cni, cnj) = (_ct.c_int(), _ct.c_int())
    cgrtyp = C_MKSTR(' '*_rc.FST_GRTYP_LEN)
    (cig1, cig2, cig3, cig4) = (_ct.c_int(), _ct.c_int(),
                                _ct.c_int(), _ct.c_int())
    cgrref = C_MKSTR(' '*_rc.FST_GRTYP_LEN)
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
            'grtyp' : cgrtyp.value,
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value,
            'grref' : cgrref.value,
            'ig1ref'   : cig1ref.value,
            'ig2ref'   : cig2ref.value,
            'ig3ref'   : cig3ref.value,
            'ig4ref'   : cig4ref.value
            }
    #TODO: ezgxprm: be more explicit on the ref values: tags, i0, j0, ...
    if doSubGrid:
        params['nsubgrids'] = ezget_nsubgrids(gdid)
        params['subgridid'] = ezget_subgridids(gdid)
        params['subgrid'] = [gdid]
        if params['nsubgrids'] > 1:
            params['subgrid'] = []
            for gid2 in params['subgridid']:
                params['subgrid'].append(ezgxprm(gid2))
    return params


def ezgfstp(gdid):
    """Get the standard file attributes of the positional records

    recParams = ezgfstp(gdid)
    
    Args:
        gdid (int): id of the grid
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
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgfstp: expecting args of type int, Got %s" %
                        (type(gdid)))
    (ctypvarx, cnomvarx, cetikx) = (C_MKSTR(' '*_rc.FST_TYPVAR_LEN),
                                    C_MKSTR(' '*_rc.FST_NOMVAR_LEN),
                                    C_MKSTR(' '*_rc.FST_ETIKET_LEN))
    (ctypvary, cnomvary, cetiky) = (C_MKSTR(' '*_rc.FST_TYPVAR_LEN),
                                    C_MKSTR(' '*_rc.FST_NOMVAR_LEN),
                                    C_MKSTR(' '*_rc.FST_ETIKET_LEN))
    (cip1, cip2, cip3) = (_ct.c_int(), _ct.c_int(), _ct.c_int())
    (cdateo, cdeet, cnpas, cnbits) = (_ct.c_int(), _ct.c_int(),
                                      _ct.c_int(), _ct.c_int())
    istat = _rp.c_ezgfstp(gdid, cnomvarx, ctypvarx, cetikx, cnomvary,
                          ctypvary, cetiky, cip1, cip2, cip3, cdateo,
                          cdeet, cnpas, cnbits)
    if istat >= 0:
        return {
            'id'    : gdid,
            'typvarx': ctypvarx.value,
            'nomvarx': cnomvarx.value,
            'etikx'  : cetikx.value,
            'typvary': ctypvary.value,
            'nomvary': cnomvary.value,
            'etiky ' : cetiky.value,
            'ip1'   : cip1.value,
            'ip2'   : cip2.value,
            'ip3'   : cip3.value,
            'dateo' : cdateo.value,
            'deet'  : cdeet.value,
            'npas'  : cnpas.value,
            'nbits' : cnbits.value
            }
    raise EzscintError()


def gdgaxes(gdid, ax=None, ay=None):
    """Gets the deformation axes of the Z, Y, # grids
    
    gridAxes = gdgaxes(gdid)
    gridAxes = gdgaxes(gdid, gridAxes)
    gridAxes = gdgaxes(gdid, ax, ay)
    Args:
        gdid     : id of the grid (int)
        gridAxes : grid axes dictionary, same as return gridAxes (dict)
        ax, ay   : 2 pre-allocated grid axes arrays (numpy.ndarray)
    Returns:
        {
            'id' : grid id, same as input arg
            'ax' : x grid axe data (numpy.ndarray)
            'ay' : y grid axe data (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdgaxes: expecting args of type int, Got %s" %
                        (type(gdid)))
    if type(ax) == dict:
        gridAxes = ax
        try:
            ax = gridAxes['ax']
            ay = gridAxes['ay']
        except:
            raise TypeError("gdgaxes: Expecting gridAxes = {'ax':ax, 'ay':ay}")
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        raise EzscintError("gdgaxes: supergrids not supported yet, " +
                           " loop through individual grids instead")
        #TODO: automate the process (loop) for super grids
    gridParams = ezgxprm(gdid)
    axshape = None
    ayshape = None
    if gridParams['grtyp'].lower() == 'y':
        axshape = gridParams['shape']
        ayshape = gridParams['shape']
    elif gridParams['grtyp'].lower() in ('z', '#'):
        axshape = (gridParams['shape'][0], 1)
        ayshape = (1, gridParams['shape'][1])
    #elif gridParams['grtyp'].lower() == 'u': #TODO add support of U/F-grids
    else:
        raise EzscintError("gdgaxes: grtyp/grref = %s/%s not supported" %
                           (gridParams['grtyp'], gridParams['grref']))
    if ax is None:
        ax = _np.empty(axshape, dtype=_np.float32, order='FORTRAN')
        ay = _np.empty(ayshape, dtype=_np.float32, order='FORTRAN')
    elif not(type(ax) == _np.ndarray and type(ay) == _np.ndarray):
        raise TypeError("gdgaxes: Expecting ax, ay as 2 numpy.ndarray, " +
                        "Got %s, %s" % (type(ax), type(ay)))
    if not (ax.dtype == _np.float32 and ax.flags['F_CONTIGUOUS']):
        ax = _np.asfortranarray(ax, dtype=_np.float32)
    if not (ay.dtype == _np.float32 and ay.flags['F_CONTIGUOUS']):
        ay = _np.asfortranarray(ay, dtype=_np.float32)
    if ax.shape != axshape or ay.shape != ayshape:
        raise TypeError("gdgaxes: provided ax, ay have the wrong shape")
    istat = _rp.c_gdgaxes(gdid, ax, ay)
    if istat >= 0:
        return {
            'id' : gdid,
            'ax' : ax,
            'ay' : ay
            }
    raise EzscintError()


def gdll(gdid, lat=None, lon=None):
    """Gets the latitude/longitude position of grid 'gdid'
    
    gridLatLon = gdll(gdid)
    gridLatLon = gdll(gdid, gridLatLon)
    gridLatLon = gdll(gdid, lat, lon)
    Args:
        gdid       : id of the grid (int)
        gridLatLon : grid lat, lon dictionary, same as return gridLatLon (dict)
        lat, lon    : 2 pre-allocated lat, lon arrays (numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : latitude  data (numpy.ndarray)
            'lon' : longitude data (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdll: expecting args of type int, Got %s" %
                        (type(gdid)))
    if type(lat) == dict:
        gridLatLon = lat
        try:
            lat = gridLatLon['lat']
            lon = gridLatLon['lon']
        except:
            raise TypeError(
                "gdll: Expecting gridLatLon = {'lat':lat, 'lon':lon}")
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        raise EzscintError("gdll: supergrids not supported yet, " +
                           "loop through individual grids instead")
        #TODO: automate the process (loop) for super grids
    gridParams = ezgxprm(gdid)
    if lat is None:
        lat = _np.empty(gridParams['shape'], dtype=_np.float32, order='FORTRAN')
        lon = _np.empty(gridParams['shape'], dtype=_np.float32, order='FORTRAN')
    elif not(type(lat) == _np.ndarray and type(lon) == _np.ndarray):
        raise TypeError("gdll: Expecting lat, lon as 2 numpy.ndarray," +
                        "Got %s, %s" % (type(lat), type(lon)))
    if not (lat.dtype == _np.float32 and lat.flags['F_CONTIGUOUS']):
        lat = _np.asfortranarray(lat, dtype=_np.float32)
    if not (lon.dtype == _np.float32 and lon.flags['F_CONTIGUOUS']):
        lon = _np.asfortranarray(lon, dtype=_np.float32)
    if lat.shape != gridParams['shape'] or lon.shape != gridParams['shape']:
        raise TypeError("gdll: provided lat, lon have the wrong shape")
    istat = _rp.c_gdll(gdid, lat, lon)
    if istat >= 0:
        return {
            'id'  : gdid,
            'lat' : lat,
            'lon' : lon
            }
    raise EzscintError()


def gdxyfll(gdid, lat, lon):
    """Returns the x-y positions of lat lon points on grid 'gdid'
    
    pointXY = gdxyfll(gdid, lat, lon)
    
    Args:
        gdid     : id of the grid (int)
        lat, lon  : list of points lat, lon (list, tuple or numpy.ndarray)
    Returns:
        {
            'id' : grid id, same as input arg
            'x'  : list of points x-coor (numpy.ndarray)
            'y'  : list of points y-coor (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdxyfll: expecting args of type int, Got %s" %
                        (type(gdid)))
    (clat, clon) = lat, lon
    if type(lat) in (list, tuple):
        clat = _np.asfortranarray(lat, dtype=_np.float32)
    elif type(lat) == _np.ndarray:
        if not (lat.dtype == _np.float32 and lat.flags['F_CONTIGUOUS']):
            clat = _np.asfortranarray(lat, dtype=_np.float32)
    else:
        raise TypeError("y: provided lat must be arrays")
    if type(lon) in (list, tuple):
        clon = _np.asfortranarray(lon, dtype=_np.float32)
    elif type(lon) == _np.ndarray:
        if not (lon.dtype == _np.float32 and lon.flags['F_CONTIGUOUS']):
            clon = _np.asfortranarray(lon, dtype=_np.float32)
    else:
        raise TypeError("y: provided lon must be arrays")
    if clat.size != clon.size:
        raise TypeError("gdxyfll: provided lat, lon should have the same size")
    cx = _np.empty(clat.shape, dtype=_np.float32, order='FORTRAN')
    cy = _np.empty(clat.shape, dtype=_np.float32, order='FORTRAN')
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


def gdllfxy(gdid, xpts, ypts):
    """Returns the lat-lon coordinates of data
    located at positions x-y on grid GDID
    
     pointLL = gdllfxy(gdid, xpts, ypts)
     
    Args:
        gdid       : id of the grid (int)
        xpts, ypts : list of points x, y coor (list, tuple or numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : list of points lat-coor (numpy.ndarray)
            'lon' : list of points lon-coor (numpy.ndarray)
        }
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdllfxy: expecting args of type int, Got %s" %
                        (type(gdid)))
    (cx, cy) = (xpts, ypts)
    if type(cx) in (list, tuple):
        cx = _np.asfortranarray(xpts, dtype=_np.float32)
    elif type(cx) == _np.ndarray:
        if not (cx.dtype == _np.float32 and cx.flags['F_CONTIGUOUS']):
            cx = _np.asfortranarray(xpts, dtype=_np.float32)
    else:
        raise TypeError("y: provided xpts must be arrays")
    if type(cy) in (list, tuple):
        cy = _np.asfortranarray(ypts, dtype=_np.float32)
    elif type(cy) == _np.ndarray:
        if not (cy.dtype == _np.float32 and cy.flags['F_CONTIGUOUS']):
            cy = _np.asfortranarray(ypts, dtype=_np.float32)
    else:
        raise TypeError("y: provided ypts must be arrays")
    if cx.size != cy.size:
        raise TypeError(
            "gdllfxy: provided xpts, ypts should have the same size")
    clat = _np.empty(cx.shape, dtype=_np.float32, order='FORTRAN')
    clon = _np.empty(cx.shape, dtype=_np.float32, order='FORTRAN')
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
    """Returns the mask associated with grid 'gdid'

    mask = gdgetmask(gdid)
    mask = gdgetmask(gdid, mask)
    
    Args:
        gdid : id of the grid (int)
        mask : mask array (numpy.ndarray)
    Returns:
        mask array (numpy.ndarray)
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdgetmask: expecting args of type int, Got %s" %
                        (type(gdid)))
    gridParams = ezgxprm(gdid)
    if mask:
        if type(mask) != _np.ndarray:
            raise TypeError("gdgetmask: Expecting mask array of type " +
                            "numpy.ndarray, Got %s" % (type(mask)))
        if mask.shape != gridParams['shape']:
            raise TypeError("gdgetmask: Provided mask array have " +
                            "inconsistent shape compered to the grid")
        if not mask.dtype in (_np.intc, _np.int32):
            raise TypeError("gdsetmask: Expecting mask arg of type numpy, " +
                            "intc Got %s, %s" % (type(mask.dtype)))
        if not mask.flags['F_CONTIGUOUS']:
            mask = _np.asfortranarray(mask, dtype=mask.dtype)
    else:
        mask = _np.empty(gridParams['shape'], dtype=_np.intc)
    istat = _rp.c_gdgetmask(gdid, mask)
    if istat < 0:
        raise EzscintError()
    return mask


#TODO:    c_gdxpncf(gdid, i1, i2, j1, j2)
#TODO:    c_gdgxpndaxes(gdid, ax, ay)


#---- Interpolation Functions


def ezsint(gdidout, gdidin, zin, zout=None):
    """Scalar horizontal interpolation

    zout = ezsint(gdidout, gdidin, zin)
    zout = ezsint(gdidout, gdidin, zin, zout)

    Args:
        gdidout : output grid id
        gdidid  : grid id describing zin grid
        zin     : data to interpolate (numpy.ndarray)
        zout    : optional, interp.result array (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    gridsetid = ezdefset(gdidout, gdidin)
    if not (type(zin) == _np.ndarray):
        raise TypeError("ezsint: expecting args of type numpy.ndarray, " +
                        "Got %s" % (type(zin)))
    gridParams = ezgxprm(gdidin)
    if zin.shape != gridParams['shape']:
        raise TypeError("ezsint: Provided zin array have inconsistent " +
                        "shape compered to the input grid")
    dshape = ezgprm(gdidout)['shape']
    if not (zin.dtype == _np.float32 and zin.flags['F_CONTIGUOUS']):
        zin = _np.asfortranarray(zin, dtype=_np.float32)    
    if zout:
        if not (type(zout) == _np.ndarray):
            raise TypeError("ezsint: Expecting zout of type numpy.ndarray, " +
                            "Got %s" % (type(zout)))
        if zout.shape != dshape:
            raise TypeError("ezsint: Provided zout array have inconsistent " +
                            "shape compered to the output grid")
        if not (zout.dtype == _np.float32 and zout.flags['F_CONTIGUOUS']):
            zout = _np.asfortranarray(zout, dtype=_np.float32)    
    else:
        zout = _np.empty(dshape, dtype=zin.dtype, order='FORTRAN')
    istat = _rp.c_ezsint(zout, zin)
    if istat >= 0:
        return zout
    raise EzscintError()

    
def ezuvint(gdidout, gdidin, uuin, vvin, uuout=None, vvout=None):
    """Vectorial horizontal interpolation

    (uuout, vvout) = ezuvint(gdidout, gdidin, uuin, vvin)
    (uuout, vvout) = ezuvint(gdidout, gdidin, uuin, vvin, uuout, vvout)

    Args:
        gdidout : output grid id
        gdidid  : grid id describing uuin grid
        uuin     : data x-part to interpolate (numpy.ndarray)
        vvin     : data y-part to interpolate (numpy.ndarray)
        uuout    : interp.result array x-part (numpy.ndarray)
        vvout    : interp.result array y-part (numpy.ndarray)
    Returns:
        interpolation result (numpy.ndarray, numpy.ndarray)
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    gridsetid = ezdefset(gdidout, gdidin)
    if not (type(uuin) == _np.ndarray and type(vvin) == _np.ndarray):
        raise TypeError("ezuvint: expecting args of type numpy.ndarray, " +
                        "Got %s, %s" % (type(uuin), type(vvin)))
    gridParams = ezgxprm(gdidin)
    if uuin.shape != gridParams['shape'] or vvin.shape != gridParams['shape']:
        raise TypeError("ezuvint: Provided uuin, vvin array have " +
                        "inconsistent shape compered to the input grid")
    if not (uuin.dtype == _np.float32 and uuin.flags['F_CONTIGUOUS']):
        uuin = _np.asfortranarray(uuin, dtype=_np.float32)    
    if not (vvin.dtype == _np.float32 and vvin.flags['F_CONTIGUOUS']):
        vvin = _np.asfortranarray(vvin, dtype=_np.float32)      
    dshape = ezgprm(gdidout)['shape']
    if uuout and vvout:
        if not (type(uuout) == _np.ndarray and type(vvout) == _np.ndarray):
            raise TypeError("ezuvint: Expecting uuout, vvout of type " +
                            "numpy.ndarray, Got %s" % (type(uuout)))
        if uuout.shape != dshape or vvout.shape != dshape:
            raise TypeError("ezuvint: Provided uuout, vvout array have " +
                            "inconsistent shape compered to the output grid")
        if not (uuout.dtype == _np.float32 and uuout.flags['F_CONTIGUOUS']):
            uuout = _np.asfortranarray(uuout, dtype=_np.float32)      
        if not (vvout.dtype == _np.float32 and vvout.flags['F_CONTIGUOUS']):
            vvout = _np.asfortranarray(vvout, dtype=_np.float32)      
    else:
        uuout = _np.empty(dshape, dtype=uuin.dtype, order='FORTRAN')
        vvout = _np.empty(dshape, dtype=uuin.dtype, order='FORTRAN')
    istat = _rp.c_ezuvint(uuout, vvout, uuin, vvin)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


def gdllsval(gdid, lat, lon, zin, zout=None):
    """Scalar interpolation to points located at lat-lon coordinates

    zout = gdllsval(gdid, lat, lon, zin)
    zout = gdllsval(gdid, lat, lon, zin, zout)

    Args:
        gdid    : id of the grid (int)
        lat     : list of resquested points lat  (numpy.ndarray)
        lon     : list of resquested points lon  (numpy.ndarray)
        zin     : data to interpolate, on grid gdid (numpy.ndarray)
        zout    : optional, interp.result array,
                  same shape a lat, lon (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a lat, lon 
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not isinstance(zin, _np.ndarray):
        raise TypeError("gdllsval: expecting zin arg of type numpy.ndarray, " +
                        "Got %s" % (type(zin)))
    gridParams = ezgxprm(gdid)
    if zin.shape != gridParams['shape']:
        raise TypeError("gdllsval: Provided zin array have inconsistent " +
                        "shape compered to the input grid")
    if not isinstance(lat, _np.ndarray):
        raise TypeError("gdllsval: expecting lat arg of type numpy.ndarray, " +
                        "Got %s" % (type(lat)))
    if not isinstance(lon, _np.ndarray):
        raise TypeError("gdllsval: expecting lon arg of type numpy.ndarray, " +
                        "Got %s" % (type(lon)))
    if lat.shape != lon.shape:
        raise TypeError("gdllsval: Provided lat, lon arrays have " +
                        "inconsistent shapes")
    dshape = lat.shape
    if not (lat.dtype == _np.float32 and lat.flags['F_CONTIGUOUS']):
        lat = _np.asfortranarray(lat, dtype=_np.float32)
    if not (lon.dtype == _np.float32 and lon.flags['F_CONTIGUOUS']):
        lon = _np.asfortranarray(lon, dtype=_np.float32)
    if zout:
        if not (type(zout) == _np.ndarray):
            raise TypeError("gdllsval: Expecting zout of type numpy.ndarray, " +
                            "Got %s" % (type(zout)))
        if zout.shape != dshape:
            raise TypeError("gdllsval: Provided zout array have " +
                            "inconsistent shape compered to lat, lon arrays")
        if not (zout.dtype == _np.float32 and zout.flags['F_CONTIGUOUS']):
            zout = _np.asfortranarray(zout, dtype=_np.float32)
    else:
        zout = _np.empty(dshape, dtype=zin.dtype, order='FORTRAN')
    istat = _rp.c_gdllsval(gdid, zout, zin, lat, lon, lat.size)
    if istat >= 0:
        return zout
    raise EzscintError()


def gdxysval(gdid, xpts, ypts, zin, zout=None):
    """Scalar intepolation to points located at x-y coordinates

    zout = gdxysval(gdid, xpts, ypts, zin)
    zout = gdxysval(gdid, xpts, ypts, zin, zout)

    Args:
        gdid    : id of the grid (int)
        xpts     : list of resquested points x-coor  (numpy.ndarray)
        ypts     : list of resquested points y-coor  (numpy.ndarray)
        zin     : data to interpolate, on grid gdid (numpy.ndarray)
        zout    : optional, interp.result array, same shape a xpts, ypts
                  (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a xpts, ypts 
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not isinstance(zin, _np.ndarray):
        raise TypeError("gdxysval: expecting zin arg of type numpy.ndarray, " +
                        "Got %s" % (type(zin)))
    gridParams = ezgxprm(gdid)
    if zin.shape != gridParams['shape']:
        raise TypeError("gdxysval: Provided zin array have inconsistent " +
                        "shape compered to the input grid")
    if not isinstance(xpts, _np.ndarray):
        raise TypeError("gdxysval: expecting xpts arg of type numpy.ndarray, " +
                        "Got %s" % (type(xpts)))
    if not isinstance(ypts, _np.ndarray):
        raise TypeError("gdxysval: expecting ypts arg of type numpy.ndarray, " +
                        "Got %s" % (type(ypts)))
    if xpts.shape != ypts.shape:
        raise TypeError("gdxysval: Provided xpts, ypts arrays have " +
                        "inconsistent shapes")
    dshape = xpts.shape
    if not (xpts.dtype == _np.float32 and xpts.flags['F_CONTIGUOUS']):
        xpts = _np.asfortranarray(xpts, dtype=_np.float32)
    if not (ypts.dtype == _np.float32 and ypts.flags['F_CONTIGUOUS']):
        ypts = _np.asfortranarray(ypts, dtype=_np.float32)
    if not (zin.dtype == _np.float32 and zin.flags['F_CONTIGUOUS']):
        zin = _np.asfortranarray(zin, dtype=_np.float32)
    if zout:
        if not (type(zout) == _np.ndarray):
            raise TypeError("gdxysval: Expecting zout of type numpy.ndarray, " +
                            "Got %s" % (type(zout)))
        if zout.shape != dshape:
            raise TypeError("gdxysval: Provided zout array have inconsistent " +
                            "shape compered to xpts, ypts arrays")
        if not (zout.dtype == _np.float32 and zout.flags['F_CONTIGUOUS']):
            zout = _np.asfortranarray(zout, dtype=_np.float32)
    else:
        zout = _np.empty(dshape, dtype=zin.dtype, order='FORTRAN')
    istat = _rp.c_gdxysval(gdid, zout, zin, xpts, ypts, xpts.size)
    if istat >= 0:
        return zout
    raise EzscintError()


def gdllvval(gdid, lat, lon, uuin, vvin, uuout=None, vvout=None):
    """Vectorial interpolation to points located at lat-lon coordinates

    (uuout, vvout) = gdllsval(gdid, lat, lon, uuin, vvin)
    (uuout, vvout) = gdllsval(gdid, lat, lon, uuin, vvin, uuout, vvout)

    Args:
        gdid    : id of the grid (int)
        lat     : list of resquested points lat  (numpy.ndarray)
        lon     : list of resquested points lon  (numpy.ndarray)
        uuin, vvin   : data to interpolate, on grid gdid (numpy.ndarray)
        uuout, vvout : optional, interp.result array, same shape a lat, lon
                       (numpy.ndarray)
    Returns:
        (uuout, vvout), tuple of 2 numpy.ndarray, interpolation result,
        same shape a lat, lon 
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not isinstance(uuin, _np.ndarray):
        raise TypeError("gdllvval: expecting uuin arg of type numpy.ndarray, " +
                        "Got %s" % (type(uuin)))
    if not isinstance(vvin, _np.ndarray):
        raise TypeError("gdllvval: expecting vvin arg of type numpy.ndarray," +
                        "Got %s" % (type(vvin)))
    gridParams = ezgxprm(gdid)
    if uuin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("gdllvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    if not isinstance(lat, _np.ndarray):
        raise TypeError("gdllvval: expecting lat arg of type numpy.ndarray, " +
                        "Got %s" % (type(lat)))
    if not isinstance(lon, _np.ndarray):
        raise TypeError("gdllvval: expecting lon arg of type numpy.ndarray, " +
                        "Got %s" % (type(lon)))
    if lat.shape != lon.shape:
        raise TypeError("gdllvval: Provided lat, lon arrays have " +
                        "inconsistent shapes")
    if not (lat.dtype == _np.float32 and lat.flags['F_CONTIGUOUS']):
        lat = _np.asfortranarray(lat, dtype=_np.float32)
    if not (lon.dtype == _np.float32 and lon.flags['F_CONTIGUOUS']):
        lon = _np.asfortranarray(lon, dtype=_np.float32)
    if not (uuin.dtype == _np.float32 and uuin.flags['F_CONTIGUOUS']):
        uuin = _np.asfortranarray(uuin, dtype=_np.float32)
    if not (vvin.dtype == _np.float32 and vvin.flags['F_CONTIGUOUS']):
        vvin = _np.asfortranarray(vvin, dtype=_np.float32)
    dshape = lat.shape
    if uuout:
        if not (type(uuout) == _np.ndarray):
            raise TypeError("gdllvval: Expecting uuout of type " +
                            "numpy.ndarray, Got %s" % (type(uuout)))
        if uuout.shape != dshape:
            raise TypeError("gdllvval: Provided uuout array have " +
                            "inconsistent shape compered to lat, lon arrays")
        if not (uuout.dtype == _np.float32 and uuout.flags['F_CONTIGUOUS']):
            uuout = _np.asfortranarray(uuout, dtype=_np.float32)      
    else:
        uuout = _np.empty(dshape, dtype=uuin.dtype, order='FORTRAN')
    if vvout:
        if not (type(vvout) == _np.ndarray):
            raise TypeError("gdllvval: Expecting vvout of type " +
                            "numpy.ndarray, Got %s" % (type(vvout)))
        if vvout.shape != dshape:
            raise TypeError("gdllvval: Provided vvout array have " +
                            "inconsistent shape compered to lat, lon arrays")
        if not (vvout.dtype == _np.float32 and vvout.flags['F_CONTIGUOUS']):
            vvout = _np.asfortranarray(vvout, dtype=_np.float32)      
    else:
        vvout = _np.empty(dshape, dtype=uuin.dtype, order='FORTRAN')
    istat = _rp.c_gdllvval(gdid, uuout, vvout, uuin, vvin, lat, lon, lat.size)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


def gdxyvval(gdid, xpts, ypts, uuin, vvin, uuout=None, vvout=None):
    """Vectorial intepolation to points located at x-y coordinates

    (uuout, vvout) = gdxysval(gdid, xpts, ypts, uuin, vvin)
    (uuout, vvout) = gdxysval(gdid, xpts, ypts, uuin, vvin, uuout, vvout)

    Args:
        gdid    : id of the grid (int)
        xpts     : list of resquested points x-coor  (numpy.ndarray)
        ypts     : list of resquested points y-coor  (numpy.ndarray)
        uuin     : data to interpolate, on grid gdid (numpy.ndarray)
        uuout    : optional, interp.result array, same shape a xpts, ypts
                   (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result, same shape a xpts, ypts 
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not isinstance(uuin, _np.ndarray):
        raise TypeError("dgxyvval: expecting uuin arg of type numpy.ndarray, " +
                        "Got %s" % (type(uuin)))
    if not isinstance(vvin, _np.ndarray):
        raise TypeError("dgxyvval: expecting vvin arg of type numpy.ndarray, " +
                        "Got %s" % (type(vvin)))
    gridParams = ezgxprm(gdid)
    if uuin.shape != gridParams['shape']:
        raise TypeError("dgxyvval: Provided uuin array have inconsistent " +
                        "shape compered to the input grid")
    if vvin.shape != gridParams['shape']:
        raise TypeError("dgxyvval: Provided vvin array have inconsistent " +
                        "shape compered to the input grid")
    if not isinstance(xpts, _np.ndarray):
        raise TypeError("dgxyvval: expecting xpts arg of type " +
                        "numpy.ndarray, Got %s" % (type(xpts)))
    if not isinstance(ypts, _np.ndarray):
        raise TypeError("dgxyvval: expecting ypts arg of type " +
                        "numpy.ndarray, Got %s" % (type(ypts)))
    if xpts.shape != ypts.shape:
        raise TypeError("dgxyvval: Provided xpts, ypts arrays have " +
                        "inconsistent shapes")
    if not (xpts.dtype == _np.float32 and xpts.flags['F_CONTIGUOUS']):
        xpts = _np.asfortranarray(xpts, dtype=_np.float32)
    if not (ypts.dtype == _np.float32 and ypts.flags['F_CONTIGUOUS']):
        ypts = _np.asfortranarray(ypts, dtype=_np.float32)
    if not (uuin.dtype == _np.float32 and uuin.flags['F_CONTIGUOUS']):
        uuin = _np.asfortranarray(uuin, dtype=_np.float32)
    if not (vvin.dtype == _np.float32 and vvin.flags['F_CONTIGUOUS']):
        vvin = _np.asfortranarray(vvin, dtype=_np.float32)
    dshape = xpts.shape
    if uuout:
        if not (type(uuout) == _np.ndarray):
            raise TypeError("dgxyvval: Expecting uuout of type " +
                            "numpy.ndarray, Got %s" % (type(uuout)))
        if uuout.shape != dshape:
            raise TypeError("dgxyvval: Provided uuout array have " +
                            "inconsistent shape compered to xpts, ypts arrays")
        if not (uuout.dtype == _np.float32 and uuout.flags['F_CONTIGUOUS']):
            uuout = _np.asfortranarray(uuout, dtype=_np.float32)      
    else:
        uuout = _np.empty(dshape, dtype=uuin.dtype, order='FORTRAN')
    if vvout:
        if not (type(vvout) == _np.ndarray):
            raise TypeError("dgxyvval: Expecting vvout of type " +
                            "numpy.ndarray, Got %s" % (type(vvout)))
        if vvout.shape != dshape:
            raise TypeError("dgxyvval: Provided vvout array have " +
                            "inconsistent shape compered to xpts, ypts arrays")
        if not (vvout.dtype == _np.float32 and vvout.flags['F_CONTIGUOUS']):
            vvout = _np.asfortranarray(vvout, dtype=_np.float32)      
    else:
        vvout = _np.empty(dshape, dtype=vvin.dtype, order='FORTRAN')
    istat = _rp.c_gdxyvval(gdid, uuout, vvout, uuin, vvin,
                           xpts, ypts, xpts.size)
    if istat >= 0:
        return (uuout, vvout)
    raise EzscintError()


#TODO:    c_gdllwdval(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
#TODO:    c_gdxywdval(gdin, uuout, vvout, uuin, vvin, x, y, n)

#TODO:    c_ezsint_mdm(zout, mask_out, zin, mask_in)
#TODO:    c_ezuvint_mdm(uuout, vvout, mask_out, uuin, vvin, mask_in)
#TODO:    c_ezsint_mask(mask_out, mask_in)

#---- Other Functions

def gdrls(gdid):
    """Frees a previously allocated grid
    
    gdrls(gdid)

    Args:
        gdid : grid id to free/release
    Returns:
        None
    Raises:
        TypeError    on wrong input arg types
        EzscintError on any other error
    """
    if not isinstance(gdid, (list, tuple)):
        gdid = [gdid]
    for id1 in gdid:
        if not (type(id1) == int):
            raise TypeError("gdrls: Expecting gdid of type int, Got %s" %
                            (type(id1)))
        istat = _rp.c_gdrls(id1)
        if istat < 0:
            raise EzscintError()
    return None
    
#TODO:    c_gduvfwd(gdid, uuout, vvout, spdin, wdin, lat, lon, n)
#TODO:    c_gdwdfuv(gdid, spdout, wdout, uuin, vvin, lat, lon, n)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
