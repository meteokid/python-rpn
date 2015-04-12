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
    if not (type(option) == type("")):
        raise TypeError("ezgetopt: Expecting data of type %s, Got %s" % (type(""),type(option)))
    if type(value) == type(1):
        istat = _rp.c_ezsetival(option, value)
    elif type(value) == type(1.0):
        istat = _rp.c_ezsetval(option, value)
    elif type(value) == type(" "):
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
    if type(ni) == type({}):
        gridParams = ni
        try:
            #TODO: accept shape instead of ni,nj
            ni = gridParams['ni']
            nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            ig1 = gridParams['ig1']
            ig2 = gridParams['ig2']
            ig3 = gridParams['ig3']
            ig4 = gridParams['ig4']
            iunit = gridParams['iunit']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    if (type(ni),type(nj),type(grtyp),type(ig1),type(ig2),type(ig3),type(ig4),type(iunit)) != (type(1),type(1),type(" "),type(1),type(2),type(3),type(4),type(5)):
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
    if type(ni) == type({}):
        gridParams = ni
        try:
            #TODO: accept shape instead of ni,nj
            ni = gridParams['ni']
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
    if (type(ni),type(nj),type(grtyp),type(grref),type(ig1),type(ig2),type(ig3),type(ig4),type(ax),type(ay)) != (type(1),type(1),type(" "),type(" "),type(1),type(2),type(3),type(4),_np.ndarray,_np.ndarray):
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
        subgridid    : 
        gridParams   : key=value pairs for each grid params (dict)
    Returns:
        int, super grid id
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if type(ni) == type({}):
        gridParams = ni
        try:
            #TODO: accept shape instead of ni,nj
            ni = gridParams['ni']
            nj = gridParams['nj']
            grtyp = gridParams['grtyp']
            grref = gridParams['grref']
            vercode = gridParams['vercode']
            subgridid = gridParams['subgridid']
        except:
            raise TypeError('ezgdef_fmem: provided incomplete grid description')
    if (type(ni),type(nj),type(grtyp),type(grref),type(vercode),type(subgridid)) != (type(1),type(1),type(" "),type(" "),type(1),_np.ndarray):
        raise TypeError('ezgdef_fmem: wrong input data type')
    nsubgrids = subgridid.size #TODO check this
    gdid = _rp.c_ezgdef_supergrid(ni, nj, grtyp, grref, vercode,nsubgrids,subgridid)
    if gdid >= 0:
        return gdid
    raise EzscintError()


def ezdefset(gdidout, gdidin):
    """Defines a set of grids for interpolation
    
    gdid = ezdefset(gdidout, gdidin) #TODO: check this, does it return gdid

    Args:
        gdidout : output grid id (int) 
        gdidin  : input  grid id (int) 
    Returns:
        int, grid set id
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdidout) == type(1) and type(gdidin) == type(1)):
        raise TypeError("ezdefset: Expecting a grid ids of type %s, Got %s, %s" % (type(1),type(gdidout),type(gdidin)))
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
    if not (type(gdid) == type(1) and type(mask) == _np.ndarray):
        raise TypeError("gdsetmask: Expecting args of type %s,%s Got %s, %s" % (type(1),repr(_np.ndarray),type(gdid),type(mask)))
    istat = _rp.c_gdsetmask(gdid, mask)
    if istat >= 0:
        return istat
    raise EzscintError()


#---- Query Functions


def ezgetopt(option,vtype=type(1)):
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
    if not (type(option) == type("")):
        raise TypeError("ezgetopt: Expecting data of type %s, Got %s" % (type(""),type(option)))
    if vtype == type(1):
        cvalue = _ct.c_int()
        istat = _rp.c_ezgetival(option, cvalue)
    elif vtype == type(1.0):
        cvalue = _ct.c_float()
        istat = _rp.c_ezgetval(option, cvalue)
    elif vtype == type(" "):
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
    if not (type(super_gdid) == type(1)):
        raise TypeError("ezgetival: Expecting data of type %s, Got %s" % (type(1),type(super_gdid)))
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
            'shape' : (ni,nj) # dimensions of the grid
            'ni'    : first dimension of the grid
            'nj'    : second dimension of the grid
            'grtyp' : type of geographical projection
            'ig1'   : first grid descriptor
            'ig2'   : second grid descriptor
            'ig3'   : third grid descriptor
            'ig4'   : fourth grid descriptor
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == type(1)):
        raise TypeError("ezgprm: Expecting data of type %s, Got %s" % (type(1),type(gdid)))
    (cni,cnj) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrtyp,cig1,cig2,cig3,cig4) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgprm(gdid, cgrtyp, cni, cnj, cig1, cig2, cig3, cig4)
    if istat >= 0:
        return {
            'shape' : (max(1,cni.value),max(1,cnj.value)),
            'ni'    : cni.value,
            'nj'    : cnj.value,
            'grtyp' : cgrtyp.value,
            'ig1'   : cig1.value,
            'ig2'   : cig2.value,
            'ig3'   : cig3.value,
            'ig4'   : cig4.value,
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
            'shape'  : (ni,nj) # dimensions of the grid
            'ni'     : first dimension of the grid
            'nj'     : second dimension of the grid
            'grtyp'  : type of geographical projection
            'ig1'    : first grid descriptor
            'ig2'    : second grid descriptor
            'ig3'    : third grid descriptor
            'ig4'    : fourth grid descriptor
            'grref'  : grid ref type
            'ig1ref' : first grid descriptor of grid ref
            'ig2ref' : second grid descriptor of grid ref
            'ig3ref' : third grid descriptor of grid ref
            'ig4ref' : fourth grid descriptor of grid ref
        }
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    if not (type(gdid) == type(1)):
        raise TypeError("ezgxprm: Expecting data of type %s, Got %s" % (type(1),type(gdid)))
    (cni,cnj) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrtyp,cig1,cig2,cig3,cig4) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cgrref,cig1ref,cig2ref,cig3ref,cig4ref) = (c_mkstr(' '*_rc.FST_GRTYP_LEN),_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgprm(gdid, cgrtyp, cni, cnj, cig1, cig2, cig3, cig4, cgrref,cig1ref,cig2ref,cig3ref,cig4ref)
    if istat >= 0:
        return {
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
            'ig4ref'   : cig4ref.value,
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
    if not (type(gdid) == type(1)):
        raise TypeError("ezgfstp: Expecting data of type %s, Got %s" % (type(1),type(gdid)))
    (ctypvarx,cnomvarx,cetikx) = (c_mkstr(' '*_rc.FST_TYPVAR_LEN),c_mkstr(' '*_rc.FST_NOMVAR_LEN),c_mkstr(' '*_rc.FST_ETIKET_LEN))
    (ctypvary,cnomvary,cetiky) = (c_mkstr(' '*_rc.FST_TYPVAR_LEN),c_mkstr(' '*_rc.FST_NOMVAR_LEN),c_mkstr(' '*_rc.FST_ETIKET_LEN))
    (cip1,cip2,cip3) = (_ct.c_int(),_ct.c_int(),_ct.c_int())
    (cdateo,cdeet,cnpas,cnbits) = (_ct.c_int(),_ct.c_int(),_ct.c_int(),_ct.c_int())
    istat = _rp.c_ezgfstp(gdid, cnomvarx, ctypvarx, cetikx, cnomvary, ctypvary, cetiky, cip1, cip2, cip3, cdateo, cdeet, cnpas, cnbits)
    if istat >= 0:
        return {
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
        'nbits' : cnbits.value,
            }
    raise EzscintError()


#TODO:    c_gdgaxes(gdid, ax, ay)

#TODO:    c_gdll(gdid, lat, lon)
#TODO:    c_gdxyfll(gdid, x, y, lat, lon, n)
#TODO:    c_gdllfxy(gdid, lat, lon, x, y, n)

#TODO:    c_gdgetmask(gdid, mask)

#TODO:    c_gdxpncf(gdid, i1, i2, j1, j2)
#TODO:    c_gdgxpndaxes(gdid, ax, ay)

#---- Interpolation Functions

def ezsint(zin,gdidin,gdidout):
    """ Scalar interpolation on previsouly defined gridset (ezdefset)

    zout = ezsint(zin,gdidin,gdidout)
    
    Args:
        zin     : data to interpolate (numpy.ndarray)
        gdidid  : grid id describing zin grid
        gdidout : output grid id
    Returns:
        numpy.ndarray, interpolation result
    Raises:
        TypeError on wrong input arg types
        EzscintError on any other error
    """
    gridsetid = ezdefset(gdidout, gdidin)
    if not (type(zin) == numpy.ndarray):
        raise TypeError("ezsint: Expecting data of type %s, Got %s" % (repr(numpy.ndarray),type(zin)))
    #TODO: check that zin and gdidin have consistent shape
    dshape = ezgprm(gdidout)['shape']
    zout = _np.empty(dshape,dtype=zin.dtype,order='FORTRAN')
    istat = _rp.c_ezsint(zout, zin)
    if istat >= 0:
        return zout
    raise EzscintError()


#TODO:    c_ezuvint(uuout, vvout, uuin, vvin)
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

#TODO:    c_gdrls(gdid)

#TODO:    c_gduvfwd(gdid, uuout, vvout, spdin, wdin, lat, lon, n)
#TODO:    c_gdwdfuv(gdid, spdout, wdout, uuin, vvin, lat, lon, n)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
