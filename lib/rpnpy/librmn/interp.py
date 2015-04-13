#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
 Module librmn.intero contains python wrapper to main librmn's interp (ezscint) C functions
 
 @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from . import proto as _rp
from . import const as _rc
from . import base as _rb

#TODO: make sure caller can provide allocated array (recycle mem)

#---- helpers -------------------------------------------------------

c_mkstr = lambda x: _ct.create_string_buffer(x)

class EzscintError(Exception):
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(option) == str):
        raise TypeError("ezgetopt: Expecting data of type str, Got %s" % (type(option)))
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
def ezqkdef(ni, nj=None, grtyp=None, ig1=None, ig2=None, ig3=None, ig4=None, iunit=None):
    """Universal grid definition. Applicable to all cases.
    
    gdid = ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
    gdid = ezqkdef(gridParams)
    
    Args:
        ni,nj        : grid dims (int)
        grtyp        : grid type (str)
        ig1,ig2,ig3,ig4 : grid parameters, encoded (int)
        iunit        : File unit (int)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, grid id
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if type(ni) == dict:
        gridParams = ni
        try:
            (ni,nj) = gridParams['shape']
        except:
            (ni,nj) = (None,None)
        try:
            if not ni: ni = gridParams['ni']
            if not nj: nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            ig1 = gridParams['ig1']
            ig2 = gridParams['ig2']
            ig3 = gridParams['ig3']
            ig4 = gridParams['ig4']
            iunit = gridParams['iunit']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    if (type(ni),type(nj),type(grtyp),type(ig1),type(ig2),type(ig3),type(ig4),type(iunit)) != (int,int,str,int,int,int,int,int):
        raise TypeError('ezqkdef: wrong input data type')
    gdid = _rp.c_ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_fmem(ni, nj=None, grtyp=None, grref=None, ig1=None, ig2=None, ig3=None, ig4=None, ax=None, ay=None):
    """Generic grid definition except for 'U' grids (with necessary
    positional parameters taken from the calling arguments)
    
    gdid = ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
    gdid = ezgdef_fmem(gridParams)
    
    Args:
        ni,nj        : grid dims (int)
        grtyp, grref : grid type and grid ref type (str)
        ig1,ig2,ig3,ig4 : grid parameters, encoded (int)
        ax, ay       : grid axes (numpy.ndarray)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, grid id
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if type(ni) == dict:
        gridParams = ni
        try:
            (ni,nj) = gridParams['shape']
        except:
            (ni,nj) = (None,None)
        try:
            if not ni: ni = gridParams['ni']
            if not nj: nj = gridParams['nj']
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
    if (type(ni),type(nj),type(grtyp),type(grref),type(ig1),type(ig2),type(ig3),type(ig4),type(ax),type(ay)) != (int,int,str,str,int,int,int,int,_np.ndarray,_np.ndarray):
        raise TypeError('ezgdef_fmem: wrong input data type')
    #TODO: check ni,nj ... ax,ay dims consis
    gdid = _rp.c_ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezgdef_supergrid(ni, nj, grtyp, grref, vercode,subgridid):
    """U grid definition (which associates to a list of concatenated subgrids in one record)

    gdid = ezgdef_supergrid(ni, nj, grtyp, grref, vercode,nsubgrids,subgridid)
    gdid = ezgdef_supergrid(gridParams)
    
    Args:
        ni,nj        : grid dims (int)
        grtyp, grref : grid type and grid ref type (str)
        vercode      : 
        subgridid    : list of subgrid id (list, tuple or numpy.ndarray)
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, super grid id
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if type(ni) == dict:
        gridParams = ni
        try:
            (ni,nj) = gridParams['shape']
        except:
            (ni,nj) = (None,None)
        try:
            if not ni: ni = gridParams['ni']
            if not nj: nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            grref = gridParams['grref']
            vercode = gridParams['vercode']
            subgridid = gridParams['subgridid']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    csubgridid = subgridid
    if type(subgridid) in (list,tuple): csubgridid = _np.array(subgridid)
    if (type(ni),type(nj),type(grtyp),type(grref),type(vercode),type(csubgridid)) != (int,int,str,str,int,_np.ndarray):
        raise TypeError('ezgdef_fmem: wrong input data type')
    nsubgrids = subgridid.size
    gdid = _rp.c_ezgdef_supergrid(ni, nj, grtyp, grref, vercode,nsubgrids,csubgridid)
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdidout) == int and type(gdidin) == int):
        raise TypeError("ezdefset: Expecting a grid ids of type int, Got %s, %s" % (type(gdidout),type(gdidin)))
    istat = _rp.c_ezdefset(gdidout, gdidin)
    if istat >= 0:
        return istat
    raise EzscintError()


def gdsetmask(gdid, mask):
    """Associates a permanent mask with grid 'gdid'
    
    gdsetmask(gdid, mask)
    
    Args:
        gdid : grid id (int)
        mask : field mask (numpy.ndarray)
    Returns:
        None
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int and type(mask) == _np.ndarray):
        raise TypeError("gdsetmask: Expecting args of type int, _np.ndarray Got %s, %s" % (type(gdid),type(mask)))
    istat = _rp.c_gdsetmask(gdid, mask)
    if istat >= 0:
        return istat
    raise EzscintError()


#---- Query Functions


def ezgetopt(option,vtype=int):
    """Gets an option value from the package
    
    value = ezgetopt(option)
    value = ezgetopt(option,vtype)
    
    Args:
        option : option name (string)
        vtype  : type of requested option (type.int, type.float or type.string)
                 default: string
                 Returns:
    Returns:
        option value of the requested type
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(option) == str):
        raise TypeError("ezgetopt: Expecting data of type str, Got %s" % (type(option)))
    if vtype == int:
        cvalue = _ct.c_int()
        istat = _rp.c_ezgetival(option, cvalue)
    elif vtype == float:
        cvalue = _ct.c_float()
        istat = _rp.c_ezgetval(option, cvalue)
    elif vtype == str:
        cvalue = c_mkstr(' '*64)
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(super_gdid) == int):
        raise TypeError("ezgetival: Expecting data of type int, Got %s" % (type(super_gdid)))
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    nsubgrids = ezget_nsubgrids(super_gdid)
    cgridlist = _np.empty(nsubgrids,dtype=_np.intc)
    istat = _rp.c_ezget_subgridids(super_gdid, cgridlist)
    if istat >= 0:
        return cgridlist.tolist()
    raise EzscintError()


def ezgprm(gdid):
    """Get grid parameters
    
    gridParams = ezgprm(gdid)
    
    Args:
        gdid (int): id of the grid
    Returns:
        {
            'id'    : grid id, same as input arg
            'shape' : (ni,nj) # dimensions of the grid
            'ni'    : first dimension of the grid
            'nj'    : second dimension of the grid
            'grtyp' : type of geographical projection (one of 'A', 'B', 'E', 'G', 'L', 'N', 'S')
            'ig1'   : first grid descriptor
            'ig2'   : second grid descriptor
            'ig3'   : third grid descriptor
            'ig4'   : fourth grid descriptor
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgprm: Expecting data of type int, Got %s" % (type(gdid)))
    (cni,cnj) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrtyp,cig1,cig2,cig3,cig4) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgprm(gdid, cgrtyp, cni, cnj, cig1, cig2, cig3, cig4)
    if istat >= 0:
        return {
            'id'    : gdid,
            'shape' : (max(1,cni.value),max(1,cnj.value)),
            'ni'    : cni.value,
            'nj'    : cnj.value,
            'grtyp' : cgrtyp.value,
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value
            }
    raise EzscintError()
    
#TODO: merge ezgprm et ezgxprm et gdgaxes (conditional axes)?
def ezgxprm(gdid):
    """Get extended grid parameters
    
    gridParams = ezgxprm(gdid)
    
    Args:
        gdid (int): id of the grid
    Returns:
        {
            'id'    : grid id, same as input arg
            'shape'  : (ni,nj) # dimensions of the grid
            'ni'     : first dimension of the grid
            'nj'     : second dimension of the grid
            'grtyp'  : type of geographical projection (one of 'Z','#','Y','U')
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
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgxprm: Expecting data of type int, Got %s" % (type(gdid)))
    (cni,cnj) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrtyp,cig1,cig2,cig3,cig4) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrref,cig1ref,cig2ref,cig3ref,cig4ref) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgprm(gdid, cgrtyp, cni, cnj, cig1, cig2, cig3, cig4, cgrref,cig1ref,cig2ref,cig3ref,cig4ref)
    if istat >= 0:
        return {
            'id'    : gdid,
            'shape' : (max(1,cni.value),max(1,cnj.value)),
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
    #TODO: ezgxprm: be more explicit on the ref values: tags, i0,j0,...
    raise EzscintError()


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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("ezgfstp: Expecting data of type int, Got %s" % (type(gdid)))
    (ctypvarx,cnomvarx,cetikx) = (c_mkstr(' '*_rc.FST_TYPVAR_LEN),c_mkstr(' '*_rc.FST_NOMVAR_LEN),c_mkstr(' '*_rc.FST_ETIKET_LEN))
    (ctypvary,cnomvary,cetiky) = (c_mkstr(' '*_rc.FST_TYPVAR_LEN),c_mkstr(' '*_rc.FST_NOMVAR_LEN),c_mkstr(' '*_rc.FST_ETIKET_LEN))
    (cip1,cip2,cip3) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cdateo,cdeet,cnpas,cnbits) = (_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgfstp(gdid, cnomvarx, ctypvarx, cetikx, cnomvary, ctypvary, cetiky, cip1, cip2, cip3, cdateo, cdeet, cnpas, cnbits)
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


def gdgaxes(gdid,ax=None,ay=None):
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdgaxes: Expecting data of type int, Got %s" % (type(gdid)))
    if type(ax) == dict:
        gridAxes = ax
        try:
            ax = gridAxes['ax']
            ay = gridAxes['ay']
        except:
            raise TypeError("gdgaxes: Expecting gridAxes = {'ax':ax,'ay':ay}")
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        raise EzscintError("gdgaxes: supergrids not supported yet, loop through individual grids instead") #TODO: automate the process (loop) for super grids
    gridParams = ezgxprm(gdid)
    axshape = None
    ayshape = None
    if gridParams['grtyp'].lower() == 'y':
        axshape = gridParams['shape']
        ayshape = gridParams['shape']
    elif gridParams['grtyp'].lower() in ('z','#'):
        axshape = (gridParams['shape'][0],1)
        ayshape = (1,gridParams['shape'][1])
    #elif gridParams['grtyp'].lower() == 'u': #TODO add support of U/F-grids
    else:
        raise EzscintError("gdgaxes: grtyp/grref = %s/%s not supported" % (gridParams['grtyp'],gridParams['grref']))
    if ax is None:
        ax = _np.empty(axshape,dtype=_np.float32,order='FORTRAN')
        ay = _np.empty(ayshape,dtype=_np.float32,order='FORTRAN')
    elif not(type(ax) == _np.ndarray and type(ay) == _np.ndarray):
        raise TypeError("gdgaxes: Expecting ax,ay as 2 numpy.ndarray, Got %s, %s" % (type(ax),type(ay)))
    if ax.shape != axshape or ay.shape != ayshape:
        raise TypeError("gdgaxes: provided ax, ay have the wrong shape")
    istat = _rp.c_gdgaxes(gdid,ax,ay)
    if istat >= 0:
        return {
            'id' : gdid,
            'ax' : ax,
            'ay' : ay
            }
    raise EzscintError()


def gdll(gdid,lat=None,lon=None):
    """Gets the latitude/longitude position of grid 'gdid'
    
    gridLatLon = gdll(gdid)
    gridLatLon = gdll(gdid, gridLatLon)
    gridLatLon = gdll(gdid, lat,lon)
    Args:
        gdid       : id of the grid (int)
        gridLatLon : grid lat,lon dictionary, same as return gridLatLon (dict)
        lat,lon    : 2 pre-allocated lat,lon arrays (numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : latitude  data (numpy.ndarray)
            'lon' : longitude data (numpy.ndarray)
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdll: Expecting data of type int, Got %s" % (type(gdid)))
    if type(lat) == dict:
        gridLatLon = lat
        try:
            lat = gridAxes['lat']
            lon = gridAxes['lon']
        except:
            raise TypeError("gdll: Expecting gridLatLon = {'lat':lat,'lon':lon}")
    nsubgrids = ezget_nsubgrids(gdid)
    if nsubgrids > 1:
        raise EzscintError("gdll: supergrids not supported yet, loop through individual grids instead") #TODO: automate the process (loop) for super grids
    gridParams = ezgxprm(gdid)
    if lat is None:
        lat = _np.empty(gridParams['shape'],dtype=_np.float32,order='FORTRAN')
        lon = _np.empty(gridParams['shape'],dtype=_np.float32,order='FORTRAN')
    elif not(type(lat) == _np.ndarray and type(lon) == _np.ndarray):
        raise TypeError("gdll: Expecting lat,lon as 2 numpy.ndarray, Got %s, %s" % (type(lat),type(lon)))
    if lat.shape != gridParams['shape'] or lon.shape != gridParams['shape']:
        raise TypeError("gdll: provided lat,lon have the wrong shape")
    istat = _rp.c_gdll(gdidlat,lon)
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
        lat,lon  : list of points lat, lon (list, tuple or numpy.ndarray)
    Returns:
        {
            'id' : grid id, same as input arg
            'x'  : list of points x-coor (numpy.ndarray)
            'y'  : list of points y-coor (numpy.ndarray)
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdxyfll: Expecting data of type int, Got %s" % (type(gdid)))
    (clat,clon) = lat,lon
    if type(lat) in (list,tuple): clat = _np.array(lat,dtype=_np.float32)
    if type(lon) in (list,tuple): clon = _np.array(lon,dtype=_np.float32)
    if clat.size != clon.size:
        raise TypeError("gdxyfll: provided lat,lon should have the same size")
    (cx,cy) = (_np.empty(clat.size,dtype=_np.float32),_np.empty(clat.size,dtype=_np.float32))
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
    """Returns the lat-lon coordinates of data located at positions x-y on grid GDID
    
     pointLL = gdllfxy(gdid, xpts, ypts)
     
    Args:
        gdid       : id of the grid (int)
        xpts, ypts : list of points x,y coor (list, tuple or numpy.ndarray)
    Returns:
        {
            'id'  : grid id, same as input arg
            'lat' : list of points lat-coor (numpy.ndarray)
            'lon' : list of points lon-coor (numpy.ndarray)
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdllfxy: Expecting data of type int, Got %s" % (type(gdid)))
    (cx,cy) = (xpts,ypts)
    if type(cx) in (list,tuple): cx = _np.array(xpts,dtype=_np.float32)
    if type(cy) in (list,tuple): cy = _np.array(ypts,dtype=_np.float32)
    if cx.size != cy.size:
        raise TypeError("gdllfxy: provided xpts,ypts should have the same size")
    (clat,clon) = (_np.empty(cx.size,dtype=_np.float32),_np.empty(cx.size,dtype=_np.float32))
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdgetmask: Expecting data of type int, Got %s" % (type(gdid)))
    gridParams = ezgxprm(gdid)
    if mask:
        if type(mask) != _np.ndarray:
            raise TypeError("gdgetmask: Expecting mask array of type numpy.ndarray, Got %s" % (type(mask)))
        if mask.shape != gridParams['shape']:
            raise TypeError("gdgetmask: Provided mask array have inconsistent shape compered to the grid")
    else:
        mask = _np.empty(gridParams['shape'],dtype=_np.float32)
    istat = _rp.c_gdgetmask(gdid, mask)
    if istat >= 0:
        return mask
    #TODO: should we return gridParams as well (also for other similar functions above)
    raise EzscintError()
    
#TODO:    c_gdxpncf(gdid, i1, i2, j1, j2)
#TODO:    c_gdgxpndaxes(gdid, ax, ay)


#---- Interpolation Functions


def ezsint(gdidout,gdidin,zin,zout=None):
    """Scalar horizontal interpolation

    zout = ezsint(gdidout,gdidin,zin)
    zout = ezsint(gdidout,gdidin,zin,zout)

    Args:
        gdidout : output grid id
        gdidid  : grid id describing zin grid
        zin     : data to interpolate (numpy.ndarray)
        zout    : interp.result array (numpy.ndarray)
    Returns:
        numpy.ndarray, interpolation result
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    gridsetid = ezdefset(gdidout, gdidin)
    if not (type(zin) == numpy.ndarray):
        raise TypeError("ezsint: Expecting data of type numpy.ndarray, Got %s" % (type(zin)))
    gridParams = ezgxprm(gdidin)
    if zin.shape != gridParams['shape']:
        raise TypeError("ezsint: Provided zin array have inconsistent shape compered to the input grid")
    dshape = ezgprm(gdidout)['shape']
    if zout:
        if not (type(zout) == numpy.ndarray):
            raise TypeError("ezsint: Expecting zout of type numpy.ndarray, Got %s" % (type(zout)))
        if zout.shape != dshape:
            raise TypeError("ezsint: Provided zout array have inconsistent shape compered to the output grid")
    else:
        zout = _np.empty(dshape,dtype=zin.dtype,order='FORTRAN')
    istat = _rp.c_ezsint(zout, zin)
    if istat >= 0:
        return zout
    raise EzscintError()


def ezuvint(gdidout,gdidin,uuin,vvin,uuout=None,vvout=None):
    """Vectorial horizontal interpolation

    (uuout,vvout) = ezuvint(gdidout,gdidin,uuin,vvin)
    (uuout,vvout) = ezuvint(gdidout,gdidin,uuin,vvin,uuout,vvout)

    Args:
        gdidout : output grid id
        gdidid  : grid id describing uuin grid
        uuin     : data x-part to interpolate (numpy.ndarray)
        vvin     : data y-part to interpolate (numpy.ndarray)
        uuout    : interp.result array x-part (numpy.ndarray)
        vvout    : interp.result array y-part (numpy.ndarray)
    Returns:
        interpolation result (numpy.ndarray,numpy.ndarray)
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    gridsetid = ezdefset(gdidout, gdidin)
    if not (type(uuin) == numpy.ndarray and type(vvin) == numpy.ndarray):
        raise TypeError("ezuvint: Expecting data of type numpy.ndarray, Got %s, %s" % (type(uuin),type(vvin)))
    gridParams = ezgxprm(gdidin)
    if uuin.shape != gridParams['shape'] or vvin.shape != gridParams['shape']:
        raise TypeError("ezuvint: Provided uuin,vvin array have inconsistent shape compered to the input grid")
    dshape = ezgprm(gdidout)['shape']
    if uuout and vvout:
        if not (type(uuout) == numpy.ndarray and type(vvout) == numpy.ndarray):
            raise TypeError("ezuvint: Expecting uuout,vvout of type numpy.ndarray, Got %s" % (type(uuout)))
        if uuout.shape != dshape or vvout.shape != dshape:
            raise TypeError("ezuvint: Provided uuout,vvout array have inconsistent shape compered to the output grid")
    else:
        uuout = _np.empty(dshape,dtype=uuin.dtype,order='FORTRAN')
        vvout = _np.empty(dshape,dtype=uuin.dtype,order='FORTRAN')
    istat = _rp.c_ezuvint(uuout, vvout, uuin, vvin)
    if istat >= 0:
        return (uuout,vvout)
    raise EzscintError()


#TODO:    c_gdllsval(gdid, zout, zin, lat, lon, n)
#TODO:    c_gdxysval(gdid, zout, zin, x, y, n)
#TODO:    c_gdllvval(gdid, uuout, vvout, uuin, vvin, lat, lon, n)
#TODO:    c_gdxyvval(gdid, uuout, vvout, uuin, vvin, x, y, n)
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
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == int):
        raise TypeError("gdrls: Expecting gdid of type int, Got %s" % (type(gdid)))
    istat = _rp.c_gdrls(gdid)
    if istat >= 0:
        return istat
    raise EzscintError()

#TODO:    c_gduvfwd(gdid, uuout, vvout, spdin, wdin, lat, lon, n)
#TODO:    c_gdwdfuv(gdid, spdout, wdout, uuin, vvin, lat, lon, n)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
