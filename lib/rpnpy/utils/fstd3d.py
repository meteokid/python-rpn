#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Utility functions to create, read and write 3d RPNSTD fields
along with proper metadata
"""

import numpy  as _np

import rpnpy.librmn.all as _rmn
import rpnpy.vgd.all as _vgd
import rpnpy.utils.tdpack_consts as _cst

#TODO: fst_read_3d_sample points
#TODO: fst_write_3d
#TODO: fst_new_3d

def get_levels_press(fileId, vGrid, shape, ip1list,
                     datev=-1, ip2=-1, ip3=-1, typvar=' ', etiket=' ',
                     verbose=False):
    """
    Read the reference surface field and computer the pressure cube

    press = get_levels_press(fileId, vGrid, shape, ip1list)
    press = get_levels_press(fileId, vGrid, shape, ip1list,
                             datev, ip2, ip3, typvar, etiket)

    Args:
        fileId  : unit number associated to the file
                  obtained with fnom+fstouv
        vGrid   : vertical grid descriptor
        shape   : shape of the field
        ip1list : vertical levels ip lists
        datev   : valid date
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        etiket  : label
        verbose : Print some info when true
    Returns:
        {
            'rfld' : rfld,  # 2d field reference value
            'phPa' : phPa   # 3d pressure values
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>>
    >>> # Open existing file in Rear Only mode
    >>> fileId = rmn.fstopenall(filename, rmn.FST_RO)
    >>>
    >>> # Get the pressure cube
    >>> ipkeys  = fstd3d.get_levels_keys(fileId, 'TT', thermoMom='VIPT')
    >>> ip1list = [ip1 for ip1,key in ipkeys['ip1keys']]
    >>> shape   = rmn.fstinf(fileId, nomvar='TT')['shape'][0:2]
    >>> press   = fstd3d.get_levels_press(fileId, ipkeys['vgrid'], shape, ip1list)
    >>> print('# {} {} {}'.format(shape, press['rfld'].shape, press['phPa'].shape))
    # (200, 100) (200, 100) (200, 100, 80)
    >>> rmn.fstcloseall(fileId)

    See Also:
        get_levels_keys
        fst_read_3d
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.vgd.base.vgd_levels
    """
    rfldName = _vgd.vgd_get(vGrid, 'RFLD')
    rfld     = _np.empty(shape, dtype=_np.float32, order='F')
    rfld[:]  = 1000. * _cst.MB2PA
    if rfldName:
        r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2,
                          ip3=ip3, typvar=typvar, etiket=etiket)
        if r2d is None:
            r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2,
                             typvar=typvar, etiket=etiket)
        if r2d is None:
            r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2,
                             etiket=etiket)
        if r2d is None:
            r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2)
        if r2d is None:
            r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev)
        if r2d is None:
            r2d = _rmn.fstlir(fileId, nomvar=rfldName)
        if not r2d is None:
            if verbose:
                print("Read {nomvar} ip1={ip1} ip2={ip2} ip3={ip3} typv={typvar} etk={etiket}".format(**r2d))
            ## g = _rmn.readGrid(fileId, r2d)
            ## if len(xpts) > 0:
            ##     v1 = _rmn.gdxysval(g['id'], xpts, ypts, r2d['d'])
            ##     rfld[0:len(xy)] = v1[:]
            ## if len(lats) > 0:
            ##     v1 = _rmn.gdllsval(g['id'], lats, lons, r2d['d'])
            ##     rfld[len(xy):len(xy)+len(ll)] = v1[:]
            rfld[:, :] = r2d['d'][:, :] * _cst.MB2PA
    phPa = _vgd.vgd_levels(vGrid, rfld, ip1list)
    phPa[:, :, :] /= _cst.MB2PA
    return {
        'rfld' : rfld,
        'phPa' : phPa
        }


def get_levels_keys(fileId, nomvar, datev=-1, ip2=-1, ip3=-1,
                    typvar=' ', etiket=' ',
                    vGrid=None, thermoMom='VIPT', verbose=False):
    """
    Get from file the list of ip1 and fstd-record-key matching provided filters

    ipkeys = get_levels_keys(fileId, nomvar)

    Args:
        fileId  : unit number associated to the file
                  obtained with fnom+fstouv
        nomvar  : variable name
        datev   : valid date
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        etiket  : label
        vGrid   : vertical grid descriptor
        thermoMom : 'VIPT' to get Thermo levels, 'VIPT' for momentum levels
        verbose : Print some info when true
    Returns:
        {
        'nomvar' : nomvar,  # variable name
        'datev'  : datev,   # valid date
        'ip2'    : ip2,     # forecast hour
        'ip3'    : ip3,     # user defined identifier
        'typvar' : typvar,  # type of field
        'etiket' : etiket,  # label
        'vgrid'  : vGrid,   # vertical grid descriptor as returned by vgd_read
        'ip1keys': vipkeys  # list of ip1 and corresponding FSTD rec key as
                            # ((ip1,key1), (ip1b, key2), ...)
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>>
    >>> # Open existing file in Rear Only mode
    >>> fileId = rmn.fstopenall(filename, rmn.FST_RO)
    >>>
    >>> # Find ip1, key for all TT in file
    >>> ipkeys = fstd3d.get_levels_keys(fileId, 'TT', thermoMom='VIPT', verbose=True)
    Getting vertical grid description
    Found 158 VIPT levels of type hyb
    >>> print('# Found {} levels for TT'.format(len(ipkeys['ip1keys'])))
    # Found 80 levels for TT
    >>> rmn.fstcloseall(fileId)

    See Also:
        get_levels_press
        fst_read_3d
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.vgd.base.vgd_read
    """
    #TODO: try to get the sorted ip1 list w/o vgrid, because vgrid doesn;t support 2 different vertical coor in the same file (or list of linked files)

    # Get the vgrid definition present in the file
    if vGrid is None:
        if verbose:
            print("Getting vertical grid description")
        _vgd.vgd_put_opt('ALLOW_SIGMA', _vgd.VGD_ALLOW_SIGMA)
        vGrid = _vgd.vgd_read(fileId)

    vip = _vgd.vgd_get(vGrid, thermoMom)
    if verbose:
        vkind = _vgd.vgd_get(vGrid, 'KIND')
        vver  = _vgd.vgd_get(vGrid, 'VERS')
        vtype = _vgd.VGD_KIND_VER_INV[(vkind, vver)]
        print("Found %d %s levels of type %s" % (len(vip), thermoMom, vtype))

    # Trim the list of ip1 to actual levels in files for nomvar
    # since the vgrid in the file is a super set of all levels
    # and get their "key"
    vipkeys = []
    for ip1 in vip:
        (lval, lkind) = _rmn.convertIp(_rmn.CONVIP_DECODE, ip1)
        key = _rmn.fstinf(fileId, nomvar=nomvar, datev=datev, ip2=ip2, ip3=ip3,
                         ip1=_rmn.ip1_all(lval, lkind),
                         typvar=typvar, etiket=etiket)
        if key is not None:
            vipkeys.append((ip1, key['key']))
            if (datev == -1 or ip2 == -1 or ip3 == -1 or typvar.strip() == ''
                or etiket.strip() == ''):
                meta   = _rmn.fstprm(key)
                datev  = meta['datev']
                ip2    = meta['ip2']
                ip3    = meta['ip3']
                typvar = meta['typvar']
                etiket = meta['etiket']
    return {
        'nomvar' : nomvar,
        'datev'  : datev,
        'ip2'    : ip2,
        'ip3'    : ip3,
        'typvar' : typvar,
        'etiket' : etiket,
        'vgrid'  : vGrid,
        'ip1keys': vipkeys
        }


def fst_read_3d(fileId, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
                typvar=' ', nomvar=' ', getPress=False,
                dtype=None, rank=None, dataArray=None, verbose=False):
    """
    Reads the records matching the research keys into a 3D array
    along with horizontal and vertical grid info

    Only provided parameters with value different than default
    are used as selection criteria

    field3d = fst_read_3d(fileId, ... )

    Args:
        fileId  : unit number associated to the file
                  obtained with fnom+fstouv or fstopenall
        datev   : valid date
        etiket  : label
        ip1     : vertical levels lists
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
        getPress: if true, get the ref. surface field and compute pressure cube
        dtype   : array type of the returned data
                  Default is determined from records' datyp
                  Could be any numpy.ndarray type
                  See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank    : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data
        verbose :  Print some info when true
    Returns:
        None if no matching record, else:
        {
            'd'    : data,       # 3d field data (numpy.ndarray), Fortran order
            'hgrid': hGridInfo,  # horizontal grid info as returned by readGrid
            'vgrid': vGridInfo,  # vertical grid info as returned by vgd_read
            'ip1s' : ip1List     # List of ip1 of each level (tuple of int)
            ...                  # same params list as fstprm (less ip1)
            'rfld' : rfld,       # (if getPress) 2d reference field value
            'phPa' : phPa        # (if getPress) 3d pressure values
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d

    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>>
    >>> # Open existing file in Rear Only mode
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> fileId = rmn.fstopenall(filename, rmn.FST_RO)

    >>> # Find and read J1 meta and data, then print its min,max,mean values
    >>> j1 = fstd3d.fst_read_3d(fileId, nomvar='J1')
    >>> print("# J1 min={:4.1f} max={:3.1f} avg={:4.1f}"
    ...       .format(j1['d'].min(), j1['d'].max(), j1['d'].mean()))
    # J1 min= 0.0 max=92.0 avg=18.8

    >>> # Find and read specific levels J1 meta and data, then print its min,max,mean values
    >>> ip1s = (1199, 1198, 1197, 1196, 1195)
    >>> j2 = fstd3d.fst_read_3d(fileId, nomvar='J2', ip1=ip1s)
    >>> print("# J2 min={:4.1f} max={:3.1f} avg={:4.1f}"
    ...       .format(j2['d'].min(), j2['d'].max(), j2['d'].mean()))
    # J2 min= 0.0 max=94.4 avg= 8.8

    >>> # Find and read TT meta and data, then print its min,max,mean values
    >>> tt3d = fstd3d.fst_read_3d(fileId, nomvar='TT')
    >>> print("# TT ip2={0} min={1:4.1f} max={2:3.1f} avg={3:4.1f}"
    ...       .format(tt3d['ip2'], tt3d['d'].min(), tt3d['d'].max(), tt3d['d'].mean()))
    # TT ip2=0 min=-88.4 max=40.3 avg=-36.3

    >>> rmn.fstcloseall(fileId)

    See Also:
        get_levels_keys
        get_levels_press
        fst_write_3d
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_levels
    """
    # Get the list of ip1 on thermo, momentum levels in this file
    vGrid = None
    tlevels = get_levels_keys(fileId, nomvar, datev, ip2, ip3, typvar, etiket,
                              vGrid=vGrid, thermoMom='VIPT', verbose=verbose)
    vGrid = tlevels['vgrid']
    mlevels = get_levels_keys(fileId, tlevels['nomvar'], tlevels['datev'],
                              tlevels['ip2'], tlevels['ip3'],
                              tlevels['typvar'], tlevels['etiket'],
                              vGrid=vGrid, thermoMom='VIPM', verbose=verbose)

    ip1keys = None
    if ip1 in (-1, None) and len(tlevels['ip1keys']) and len(mlevels['ip1keys']):
        ip1keys = tlevels['ip1keys']
        if len(mlevels['ip1keys']) > len(tlevels['ip1keys']):
            if verbose:
                print("(fst_read_3d) Using Momentum level list")
                ip1keys = mlevels['ip1keys']
        elif verbose:
            print("(fst_read_3d) Using Thermo level list")

    if ip1 == -1 and (ip1keys is None or len(ip1keys) == 0):
        keys = _rmn.fstinl(fileId, datev, etiket, ip1, ip2, ip3, typvar, nomvar)
        ip1keys = [(_rmn.fstprm(k)['ip1'], k) for k in keys]
        vGrid = None
        if verbose:
            print("(fst_read_3d) Using Arbitrary list of levels: {}".format([i for i,k in ip1keys]))
    elif isinstance(ip1, (list, tuple)):
        if not len(ip1):
            return None
        tip1list = [i for i,k in tlevels['ip1keys']]
        mip1list = [i for i,k in mlevels['ip1keys']]
        if all([i in tip1list for i in ip1]):
            ip1keys = [(i,k) for i,k in tlevels['ip1keys'] if i in ip1]
        elif all([i in mip1list for i in ip1]):
            ip1keys = [(i,k) for i,k in mlevels['ip1keys'] if i in ip1]
        else:
            if any([a < 0 for a in ip1]):
                _rmn.RMNError("Cannot Provide CatchAll (-1) in ip1 list")
            vGrid = None
            ip1keys = [(i,None) for i in ip1]
    elif ip1 not in (-1, None):
        raise TypeError('Provided ip1 should be a list or an int')

    if verbose or ip1keys is None or len(ip1keys) == 0:
        print("(fst_read_3d) Found {0} records for {1} ip1={2} ip2={3} ip3={4} datev={5} typvar={6} etiket={7}"
              .format(len(ip1keys), nomvar, ip1, ip2, ip3, datev, typvar, etiket))

    if len(ip1keys) == 0:
        return None

    # Read all 2d records and copy to 3d array
    r3d = None
    r2d = {'d' : None}
    k = 0
    for ip1, key in ip1keys:
        if key is None:
            r2d = _rmn.fstlir(fileId, datev, etiket, ip1, ip2, ip3, typvar,
                              nomvar, dtype, rank, dataArray=r2d['d'])
            if r2d is None:
                raise _rmn.RMNError("No record matching {} ip1={} ip2={} ip3={} datev={} typvar={} etiket={}"
              .format(nomvar, ip1, ip2, ip3, datev, typvar, etiket))
        else:
            r2d = _rmn.fstluk(key, dataArray=r2d['d'])
        if verbose:
            print("Read {nomvar} ip1={ip1} ip2={ip2} ip3={ip3} typv={typvar} etk={etiket}".format(**r2d))
        if r3d is None:
            r3d = r2d.copy()
            rshape = list(r2d['d'].shape[0:2]) + [len(ip1keys)]
            if dataArray is None:
                r3d['d'] = _np.empty(rshape, dtype=r2d['d'].dtype, order='FORTRAN')
            else:
                if isinstance(dataArray,_np.ndarray) and dataArray.shape == rshape:
                    r3d['d'] = dataArray
                else:
                    raise TypeError('Provided dataArray is not the right type or shape')
            r3d['hgrid'] = _rmn.readGrid(fileId, r2d)
            if verbose:
                print("Read the horizontal grid descriptors for {nomvar}".format(**r2d))
        if r2d['d'].shape[0:2] != r3d['d'].shape[0:2]:
            raise _rmn.RMNError("Wrong shape for input data.")
        r3d['d'][:, :, k] = r2d['d'][:, :]
        k += 1

    ip1list = [x[0] for x in ip1keys]

    r3d.update({
        'shape' : r3d['d'].shape,
        'ni'    : r3d['d'].shape[0],
        'nj'    : r3d['d'].shape[1],
        'nk'    : r3d['d'].shape[2],
        'ip1'   : -1,
        'ip1s'  : ip1list,
        'vgrid' : vGrid
         })

    # Read RFLD and compute pressure on levels
    if getPress and vGrid:
        press = get_levels_press(fileId, vGrid, r3d['d'].shape[0:2], ip1list,
                                 tlevels['datev'], tlevels['ip2'],
                                 tlevels['ip3'], tlevels['typvar'], tlevels['etiket'],
                                 verbose=False)
        r3d.update({
            'rfld' : press['rfld'],
            'phPa' : press['phPa']
            })

    return r3d


## def fst_write_3d(fileId, r3d, verbose=False):
##     """
##     #TODO: doc
##     """
##     #TODO: mom or thermo?
##     pass

#TODO: example should provide 'rfld' to completely define the vgrid
def fst_new_3d(params, hgrid, ip1s=None, vgrid=None,
               dtype=None, dataArray=None, verbose=False):
    """
    Create an RPNStd 3D record with provided data and param,
    along with horizontal and vertical grid info

    field3d = fst_new_3d(params, hgrid, ip1s, vgrid, dataArray=dataArray)
    field3d = fst_new_3d(params, hgrid, ip1s, dataArray=dataArray)
    field3d = fst_new_3d(params, hgrid, ip1s, vgrid, dtype=dtype)
    field3d = fst_new_3d(params, hgrid, ip1s, dtype=dtype)

    Args:
        params    : Field metadata would normally include these keys
                    'nomvar', 'dateo', 'deet', 'npas', 'ip2', 'etiket' [dict]
        hgrid     : horizontal grid description [dict]
                    result of rmn.encodeGrid()
        ip1s      : List of ip1 defining fields levels (list) or 
                    if vgrid is provided one can provided "VIPM" or "VIPT" (str)
        vgrid     : (optional) vgrid descriptor created with any
                    vgrid descriptor generating functions.
        dtype     : (optional) numpy.dtype for array creation
                    (if params[''datyp'] nor dataArray are provided)
        dataArray : (optional) allocated array [numpy.ndarray]
        verbose   : (optional) Print some info when true
    Returns:
        {
            'd'    : data,       # 3d field data (numpy.ndarray), Fortran order
            'hgrid': hGridInfo,  # horizontal grid info as returned by readGrid
            'vgrid': vGridInfo,  # vertical grid info as returned by vgd_read
            'ip1s' : ip1List     # List of ip1 of each level (tuple of int)
            ...                  # same params list as fstprm (less ip1)
         }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value

    Examples:
    >>> import os, os.path
    >>> import numpy  as np
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> import rpnpy.vgd.all as vgd
    >>> from rpnpy.rpndate import RPNDate

    >>> params0 = {
    ...     'grtyp' : 'Z',
    ...     'grref' : 'E',
    ...     'ni'    : 90,
    ...     'nj'    : 45,
    ...     'lat0'  : 10.,
    ...     'lon0'  : 11.,
    ...     'dlat'  : 1.,
    ...     'dlon'  : 0.5,
    ...     'xlat1' : 0.,
    ...     'xlon1' : 180.,
    ...     'xlat2' : 1.,
    ...     'xlon2' : 270.
    ...     }
    >>> hgrid = rmn.encodeGrid(params0)

    >>> lvls  = (0.013,   0.027,    0.051,    0.075,
    ...          0.101,   0.127,    0.155,    0.185,    0.219,
    ...          0.258,   0.302,    0.351,    0.405,    0.460,
    ...          0.516,   0.574,    0.631,    0.688,    0.744,
    ...          0.796,   0.842,    0.884,    0.922,    0.955,
    ...          0.980,   0.995)
    >>> rcoef1 = 0.
    >>> rcoef2 = 1.
    >>> pref   = 100000.
    >>> dhm    = 10.
    >>> dht    = 2.
    >>> try:
    ...     vgrid = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")

    >>> params = rmn.FST_RDE_META_DEFAULT.copy()
    >>> params['nomvar'] = 'tt'
    >>> params['dateo'] = RPNDate(20190625, 0).stamp
    >>> params['deet'] = 1800
    >>> params['npas'] = 6
    >>> params['etiket'] = 'myetk'

    >>> # Create new 3d field for TT on vgrid Thermo levels
    >>> field3d = fstd3d.fst_new_3d(params, hgrid, ip1s='VIPT',vgrid=vgrid)

    >>> field3d['d'] = 0.

    See Also:
        fst_read_3d
        fst_write_3d
        rpnpy.librmn.fstd98.const
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.grids.encodeGrid
        rpnpy.vgd.base.vgd_new_hybmd
        rpnpy.rpndate
    """
    r3d = _rmn.FST_RDE_META_DEFAULT.copy()
    datyp = None
    if 'datyp' in params.keys():
        datyp = params['datyp']
        if dtype is None:
            dtype = _rmn.dtype_fst2numpy(datyp)
        elif dtype != _rmn.dtype_fst2numpy(datyp):
            raise ValueError("fst_new_3d: inconsistent dtype and datyp")
    r3d.update(params)
    if datyp is not None:
        r3d['datyp'] = datyp
    for k in r3d.keys():
        if k in hgrid.keys():
            r3d[k] = hgrid[k]
    (ni, nj) = (hgrid['ni'], hgrid['nj'])
    #TODO: check if vgrid is None or vgridtype
    if isinstance(ip1s, str):
        if vgrid is None:
            raise ValueError("fst_new_3d: with ip1s='{}'".format(ip1s) +
                             ", vgrid should be provided")
        if ip1s.upper() == 'VIPM':
            ip1s = _vgd.vgd_get(vgrid,'VIPM')
        elif ip1s.upper() == 'VIPT':
            ip1s = _vgd.vgd_get(vgrid,'VIPT')
        else:
            raise ValueError("fst_new_3d: ip1s unkown value '{}'"
                             .format(ip1s))
    elif isinstance(ip1s, (int, long)):
        ip1s = [ip1s]
    elif not isinstance(ip1s, (list, tuple)):
        raise TypeError("fst_new_3d: ip1s should by of type str, int, or list")
    nk = len(ip1s)

    if dataArray is None:
        if dtype is None:
            dtype = _rmn.dtype_fst2numpy(params['datyp'], params['nbits'])
        dataArray = _np.zeros((ni,nj,nk), dtype=dtype, order='FORTRAN')
    if not isinstance(dataArray, _np.ndarray):
        raise TypeError("fst_new_3d: Expecting dataArray of type {}, Got {}"\
                        .format('numpy.ndarray', type(dataArray)))
    elif not dataArray.flags['F_CONTIGUOUS']:
        raise TypeError("fst_new_3d: Expecting dataArray type " +
                        "numpy.ndarray with FORTRAN order")
    if dtype is not None and dataArray.dtype != dtype:
        raise TypeError("fst_new_3d: Inconsistency in provided dtype and dataArray")
    if dataArray.shape != (ni,nj,nk):
        raise ValueError("fst_new_3d: Inconsistency shape between vgrid/hgrid and dataArray")

    #TODO: allow providing datyp in params
    #TODO: should we compute datev if not provided or check consistency with dateo+npas*deet; or should we accept dateve instead of dateo?

    if datyp is None:
        datyp = _rmn.dtype_numpy2fst(dataArray.dtype)
    r3d.update({
        'd' : dataArray,
        'datyp' : datyp,
        'shape' : dataArray.shape,
        'ni'    : dataArray.shape[0],
        'nj'    : dataArray.shape[1],
        'nk'    : dataArray.shape[2],
        'ip1'   : -1,
        'ip1s'  : ip1s,
        'hgrid' : hgrid,
        'vgrid' : None
         })
    if vgrid:
        r3d['vgrid'] = _vgd.vgd_copy(vgrid)
    return r3d


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
