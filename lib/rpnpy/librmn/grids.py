#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2

"""
 Librmn Fstd grid helper functions

 @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
from . import RMNError
from . import const as _rc
from . import base as _rb
from . import interp as _ri

#TODO: general defGrid fn
#TODO: general decodeGrid fn
#TODO: Z grids family with '#' variante
#TODO: Y grids family
#TODO: U/F grid

def defGrid_L(ni,nj=None,lat0=None,lon0=None,dlat=None,dlon=None,setGridId=True):
    """Defines an FSTD LatLon (cylindrical equidistant) Grid (LAM)

    gridParams = defGrid_L(ni,nj,lat0,lon0,dlat,dlon,setGridId)
    gridParams = defGrid_L(ni,nj,lat0,lon0,dlat,dlon)
    gridParams = defGrid_L(params,setGridId)
    gridParams = defGrid_L(params)

    Args:
        ni, nj     : grid dims (int)
        lat0, lon0 : lat, lon of SW grid corner in degrees (float)
        dlat, dlon : grid resolution/spacing along lat, lon axes in degrees (float)
        setGridId  : Flag for creation of gid, ezscint grid id (True or False)
        params     : above parameters given as a dictionary (dict)
    Returns:
        {
            'ni'    : grid dim along the x-axis (int)
            'nj'    : grid dim along the y-axis (int)
            'grtyp' : grid type (str)
            'lat0'  : lat of SW grid corner in degrees (float)
            'lon0'  : lon of SW grid corner in degrees (float)
            'dlat'  : grid resolution/spacing along lat axe in degrees (float)
            'dlon'  : grid resolution/spacing along lon axe in degrees (float)
            'ig1'   : grid parameters, encoded (int)
            'ig2'   : grid parameters, encoded (int)
            'ig3'   : grid parameters, encoded (int)
            'ig4'   : grid parameters, encoded (int)
            'gid'   : ezscint grid-id if setGridId==True, -1 otherwise (int)
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNError   on any other error
    """
    params = {
        'ni'   : ni,
        'nj'   : nj,
        'lat0' : lat0,
        'lon0' : lon0,
        'dlat' : dlat,
        'dlon' : dlon
        }
    if isinstance(ni,dict):
        params.update(ni)
    for k in ('ni','nj'):
        v = params[k]
        if not isinstance(v,int):
            raise TypeError('defGrid_L: wrong input data type for %s, expecting int, Got (%s)' % (k,type(v)))
        if v <= 0:
            raise ValueError('defGrid_L: grid dims must be >= 0, got %s=%d' % (k,v))
    for k in ('lat0','lon0','dlat','dlon'):
        v = params[k]
        if isinstance(v,int): v = float(v)
        if not isinstance(v,float):
            raise TypeError('defGrid_L: wrong input data type for %s, expecting float, Got (%s)' % (k,type(v)))
        params[k] = v
    params['grtyp'] = 'L'
    ig1234 = _rb.cxgaig(params['grtyp'],params['lat0'],params['lon0'],
                        params['dlat'],params['dlon'])
    params['ig1'] = ig1234[0]
    params['ig2'] = ig1234[1]
    params['ig3'] = ig1234[2]
    params['ig4'] = ig1234[3]
    params['gid'] = _ri.ezqkdef(params) if setGridId else -1
    return params


def defGrid_E(ni,nj=None,xlat1=None,xlon1=None,xlat2=None,xlon2=None,setGridId=True):
    """Defines an FSTD Global, rotated, LatLon (cylindrical equidistant) Grid

    gridParams = defGrid_E(ni,nj,xlat1,xlon1,xlat2,xlon2,setGridId)
    gridParams = defGrid_E(ni,nj,xlat1,xlon1,xlat2,xlon2)
    gridParams = defGrid_E(params,setGridId)
    gridParams = defGrid_E(params)

    Args:
        ni, nj      : grid dims (int)
        xlat1,xlon1 : lat, lon of the grid center in degrees (float)
                      This defines, in rotated coor., (rlat,rlon) = (0.,180.)
                      on the rotated equator
                      The grid is defined, in rotated coor on
                      rlat: -90. to +90. degrees
                      rlon:   0. to 360. degrees
        xlat2,xlon2 : lat, lon of a 2nd ref. point in degrees (float)
                      This point is considered to be on the rotated equator,
                      east of xlat1,xlon1 (it thus defines the rotation)
        setGridId   : Flag for creation of gid, ezscint grid id (True or False)
        params      : above parameters given as a dictionary (dict)
    Returns:
        {
            'ni'    : grid dim along the x-axis (int)
            'nj'    : grid dim along the y-axis (int)
            'grtyp' : grid type (str)
            'xlat1' : lat of grid center in degrees (float)
            'xlon1' : lon of grid center in degrees (float)
            'xlat2' : lat of a 2nd ref. point in degrees (float)
            'xlon2' : lon of a 2nd ref. point in degrees (float)
            'ig1'   : grid parameters, encoded (int)
            'ig2'   : grid parameters, encoded (int)
            'ig3'   : grid parameters, encoded (int)
            'ig4'   : grid parameters, encoded (int)
            'gid'   : ezscint grid-id if setGridId==True, -1 otherwise (int)
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNError   on any other error
    """
    params = {
        'ni'    : ni,
        'nj'    : nj,
        'xlat1' : xlat1,
        'xlon1' : xlon1,
        'xlat2' : xlat2,
        'xlon2' : xlon2
        }
    if isinstance(ni,dict):
        params.update(ni)
    for k in ('ni','nj'):
        v = params[k]
        if not isinstance(v,int):
            raise TypeError('defGrid_E: wrong input data type for %s, expecting int, Got (%s)' % (k,type(v)))
        if v <= 0:
            raise ValueError('defGrid_E: grid dims must be >= 0, got %s=%d' % (k,v))
    for k in ('xlat1','xlon1','xlat2','xlon2'):
        try:
            v = params[k]
        except:
            raise TypeError('defGrid_E: provided incomplete grid description, missing: %s' % k)
        if isinstance(v,int): v = float(v)
        if not isinstance(v,float):
            raise TypeError('defGrid_E: wrong input data type for %s, expecting float, Got (%s)' % (k,type(v)))
        params[k] = v
    params['grtyp'] = 'E'
    ig1234 = _rb.cxgaig(params['grtyp'],params['xlat1'],params['xlon1'],
                        params['xlat2'],params['xlon2'])
    params['ig1'] = ig1234[0]
    params['ig2'] = ig1234[1]
    params['ig3'] = ig1234[2]
    params['ig4'] = ig1234[3]
    params['gid'] = _ri.ezqkdef(params) if setGridId else -1
    return params


def defGrid_G(ni,nj=None,glb=True,north=True,inverted=False,setGridId=True):
    """Provide grid parameters to define an FSTD Gaussian Grid

    gridParams = gridParams_G(ni,nj,lat0,lon0,dlat,dlon,setGridId)
    gridParams = gridParams_G(ni,nj,lat0,lon0,dlat,dlon)
    gridParams = gridParams_G(params,setGridId)
    gridParams = gridParams_G(params)

    Args:
        ni, nj     : grid dims (int)
        glb        : True for Global grid coverage,
                     False for Hemispheric
        north      : (used only if glb==False)
                     True for northern hemisphere,
                     False for Southern
        inverted   : False, South -> North (pt (1,1) at grid bottom)
                     True, North -> South (pt (1,1) at grid top)
        setGridId  : Flag for creation of gid, ezscint grid id (True or False)
        params     : above parameters given as a dictionary (dict)
    Returns:
        {
            'ni'    : grid dim along the x-axis (int)
            'nj'    : grid dim along the y-axis (int)
            'grtyp' : grid type (str)
            'glb'   : True for Global grid coverage, False for Hemispheric
            'north' : (used only if glb==False) True for northern hemisphere, False for Southern
            'inverted' : False, South -> North (pt (1,1) at grid bottom)
                         True,  North -> South (pt (1,1) at grid top)
            'ig1'   : grid parameters, encoded (int)
            'ig2'   : grid parameters, encoded (int)
            'ig3'   : grid parameters, encoded (int)
            'ig4'   : grid parameters, encoded (int)
            'gid'   : ezscint grid-id if setGridId==True, -1 otherwise (int)
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNError   on any other error
    """
    params = {
        'ni'   : ni,
        'nj'   : nj,
        'glb'   : glb,
        'north' : north,
        'inverted' : inverted
        }
    if isinstance(ni,dict):
        params.update(ni)
    for k in ('ni','nj'):
        v = params[k]
        if not isinstance(v,int):
            raise TypeError('defGrid_G: wrong input data type for %s, expecting int, Got (%s)' % (k,type(v)))
        if v <= 0:
            raise ValueError('defGrid_G: grid dims must be >= 0, got %s=%d' % (k,v))
    params['grtyp'] = 'G'
    params['ig1'] = 0
    if not params['glb']:
        params['ig1'] = 1 if params['north'] else 2
    params['ig2'] = 1 if params['inverted'] else 0
    params['ig3'] = 0
    params['ig4'] = 0
    params['gid'] = _ri.ezqkdef(params) if setGridId else -1
    return params


def defGrid_PS(ni,nj=None,north=True,pi=None,pj=None,d60=None,dgrw=0.,setGridId=True):
    """Define a Polar stereographic grid for the northern or southern hemisphere

    gridParams = defGrid_PS(ni,nj,north,pi,pj,d60,dgrw,setGridId)
    gridParams = defGrid_PS(ni,nj,north,pi,pj,d60,dgrw)
    gridParams = defGrid_PS(params,setGridId)
    gridParams = defGrid_PS(params)

    Args:
        ni, nj    : grid dims (int)
        pi        : Horizontal position of the pole, (float
                    in grid points, from bottom left corner (1,1).
        pj        : Vertical position of the pole, (float
                    in grid points, from bottom left corner (1,1).
        d60       : grid length, in meters, at 60deg. of latitude. (float)
        dgrw      : angle (between 0 and 360, +ve counterclockwise)
                    between the Greenwich meridian and the horizontal
                    axis of the grid. (float)
        north     : True for northern hemisphere,
                    False for Southern
        setGridId : Flag for creation of gid, ezscint grid id (True or False)
        params    : above parameters given as a dictionary (dict)
    Returns:
        {
            'ni'    : grid dim along the x-axis (int)
            'nj'    : grid dim along the y-axis (int)
            'grtyp' : grid type (str)
            'pi'    : Horizontal position of the pole, (float
                      in grid points, from bottom left corner (1,1).
            'pj'    : Vertical position of the pole, (float
                      in grid points, from bottom left corner (1,1).
            'd60'   : grid length, in meters, at 60deg. of latitude. (float)
            'dgrw'  : angle (between 0 and 360, +ve counterclockwise)
                      between the Greenwich meridian and the horizontal
                      axis of the grid. (float)
            'north' : True for northern hemisphere,
                      False for Southern
            'ig1'   : grid parameters, encoded (int)
            'ig2'   : grid parameters, encoded (int)
            'ig3'   : grid parameters, encoded (int)
            'ig4'   : grid parameters, encoded (int)
            'gid'   : ezscint grid-id if setGridId==True, -1 otherwise (int)
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNError   on any other error
    """
    params = {
        'ni'   : ni,
        'nj'   : nj,
        'north' : north,
        'pi'    : pi, 
        'pj'    : pj,
        'd60'   : d60,
        'dgrw'  : dgrw
        }
    if isinstance(ni,dict):
        params.update(ni)
    for k in ('ni','nj'):
        v = params[k]
        if not isinstance(v,int):
            raise TypeError('defGrid_PS: wrong input data type for %s, expecting int, Got (%s)' % (k,type(v)))
        if v <= 0:
            raise ValueError('defGrid_PS: grid dims must be >= 0, got %s=%d' % (k,v))
    for k in ('pi','pj','d60','dgrw'):
        try:
            v = params[k]
        except:
            raise TypeError('defGrid_PS: provided incomplete grid description, missing: %s' % k)
        if isinstance(v,int): v = float(v)
        if not isinstance(v,float):
            raise TypeError('defGrid_PS: wrong input data type for %s, expecting float, Got (%s)' % (k,type(v)))
        params[k] = v
    params['grtyp'] = 'N' if params['north'] else ['S']
    ig1234 = _rb.cxgaig(params['grtyp'],params['pi'],params['pj'],
                        params['d60'],params['dgrw'])
    params['ig1'] = ig1234[0]
    params['ig2'] = ig1234[1]
    params['ig3'] = ig1234[2]
    params['ig4'] = ig1234[3]
    params['gid'] = _ri.ezqkdef(params) if setGridId else -1
    return params


# =========================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
