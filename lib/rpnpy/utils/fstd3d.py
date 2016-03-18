#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""

"""

import numpy  as _np

import rpnpy.librmn.all as _rmn
import rpnpy.vgd.all as _vgd
import rpnpy.utils.thermoconsts as _cst


#TODO: fst_read_3d_sample points
#TODO: fst_write_3d


def get_levels_press(fileId, vGrid, shape, ip1list,
                     datev=-1, ip2=-1, ip3=-1, typvar=' ', etiket=' ',
                     verbose=False):
    """
    """
    rfldName = _vgd.vgd_get(vGrid, 'RFLD')
    rfld     = _np.empty(shape, dtype=_np.float32, order='F')
    rfld[:]  = 1000. * _cst.MB2PA
    if rfldName:
        r2d = _rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2, ip3=ip3,
                         typvar=typvar, etiket=etiket)
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
            rfld[:,:] = r2d['d'][:,:] * _cst.MB2PA
    phPa = _vgd.vgd_levels(vGrid, rfld, ip1list)
    phPa[:,:,:] /= _cst.MB2PA
    return {
        'rfld' : rfld,
        'phPa' : phPa
        }


def get_levels_keys(fileId, nomvar, datev=-1, ip2=-1, ip3=-1,
                    typvar=' ', etiket=' ',
                    vGrid=None, thermoMom='VIPT', verbose=False):
    """
    """
    #TODO: try to get the sorted ip1 list w/o vgrid, because vgrid doesn;t support 2 different vertical coor in the same file (or list of linked files)
    
    # Get the vgrid definition present in the file
    if vGrid is None:
        if verbose:
            print("Getting vertical grid description")
        _vgd.vgd_put_opt('ALLOW_SIGMA', _vgd.VGD_ALLOW_SIGMA)
        vGrid = _vgd.vgd_read(fileId)
        
    vip  = _vgd.vgd_get(vGrid, thermoMom)
    if verbose:
        vkind    = _vgd.vgd_get(vGrid, 'KIND')
        vver     = _vgd.vgd_get(vGrid, 'VERS')
        vtype    = _vgd.VGD_KIND_VER_INV[(vkind,vver)]
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
            if datev == -1 or ip2 == -1 or ip3 == -1 or typvar.strip() == '' or etiket.strip() == '':
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
        'v'      : vGrid,
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
    
    field3d = fst_read_3d(iunit, ... )
    
    Args:
        iunit   : unit number associated to the file
                  obtained with fnom+fstouv
        datev   : valid date
        etiket  : label
        ip1     : vertical levels lists
        ip2     : forecast hour
        ip3     : user defined identifier
        typvar  : type of field
        nomvar  : variable name
        getPress:
        dtype   : array type of the returned data
                  Default is determined from records' datyp
                  Could be any numpy.ndarray type
                  See: http://docs.scipy.org/doc/numpy/user/basics.types.html
        rank    : try to return an array with the specified rank
        dataArray (ndarray): (optional) allocated array where to put the data
        verbose : 
    Returns:
        None if no matching record, else:
        {
            'd'    : data,       # 3d field data (numpy.ndarray), Fortran order
            'g'    : hGridInfo,  # horizontal grid info as returned by readGrid
            'v'    : vGridInfo,  # vertical grid info as returned by vgd_read
            'ip1s' : ip1List     # List of ip1 of each level (tuple of int)
            ...                  # same params list as fstprm (less ip1)
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
    >>> funit = rmn.fstopenall(filename, rmn.FST_RO)
    >>> 
    >>> # Find and read p0 meta and data, then print its min,max,mean values
    >>> tt3d = fstd3d.fst_read_3d(funit, nomvar='TT')
    >>> print("# TT ip2={0} min={1} max={2} avg={3}"\
              .format(tt3d['ip2'], tt3d['d'].min(), tt3d['d'].max(), tt3d['d'].mean()))
    # TT ip2=0 min=530.641418 max=1039.641479 avg=966.500000
    >>> rmn.fstcloseall(funit)
    
    See Also:
    get_levels_keys
    fst_write_3d
    rpnpy.librmn.fstd98.fstlir
    rpnpy.librmn.fstd98.fstprm
    rpnpy.librmn.fstd98.fstluk
    rpnpy.librmn.fstd98.fstopenall
    rpnpy.librmn.fstd98.fstcloseall
    rpnpy.librmn.grids.readGrid
    rpnpy.vgd.base.vgd_read
    """
    # Get the list of ip1 on thermo, momentum levels in this file
    #TODO: if ip1 is provided get the keys for these ip1
    vGrid = None
    tlevels = get_levels_keys(fileId, nomvar, datev, ip2, ip3, typvar, etiket,
                              vGrid=vGrid, thermoMom='VIPT', verbose=verbose)
    vGrid = tlevels['v']
    mlevels = get_levels_keys(fileId, tlevels['nomvar'], tlevels['datev'], tlevels['ip2'],
                              tlevels['ip3'], tlevels['typvar'], tlevels['etiket'],
                              vGrid=vGrid, thermoMom='VIPM', verbose=verbose)

    ip1keys = tlevels['ip1keys']
    if len(mlevels['ip1keys']) > len(tlevels['ip1keys']):
        if verbose: print("(fst_read_3d) Using Momentum level list")
        ip1keys = mlevels['ip1keys']
    elif verbose: print("(fst_read_3d) Using Thermo level list")

    if verbose or len(ip1keys) == 0:
        print("(fst_read_3d) Found {0} records for {1} ip2={2} ip3={3} datev={4} typvar={5} etiket={6}"\
              .format(len(ip1keys), nomvar, ip2, ip3, datev, typvar, etiket))
        
    if len(ip1keys) == 0:
        return None
    
    # Read all 2d records and copy to 3d array
    r3d = None
    r2d = {'d' : None}
    k = 0
    for ip1, key in ip1keys:
        r2d = _rmn.fstluk(key, dataArray=r2d['d'])
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
            r3d['g'] = _rmn.readGrid(fileId, r2d)
            print("Read the horizontal grid descriptors for {nomvar}".format(**r2d))
        if r2d['d'].shape[0:2] != r3d['d'].shape[0:2]:
            raise _rmn.RMNError("Wrong shape for input data.")
        r3d['d'][:,:,k] = r2d['d'][:,:]
        k += 1

    ip1list = [x[0] for x in ip1keys]
            
    r3d.update({
        'shape' : r3d['d'].shape,
        'ni'    : r3d['d'].shape[0],
        'nj'    : r3d['d'].shape[1],
        'nk'    : r3d['d'].shape[2],
        'ip1'   : -1,
        'ip1s'  : ip1list,
        'v'     : vGrid
         })

    # Read RFLD and compute pressure on levels
    if getPress:
        press = get_levels_press(fileId, vGrid, r3d['d'].shape[0:2], ip1list,
                                 tlevels['datev'], tlevels['ip2'],
                                 tlevels['ip3'], tlevels['typvar'], tlevels['etiket'],
                                 verbose=False)
        r3d.update({
            'rfld' : press['rfld'],
            'phPa' : press['phPa']
            })

    return r3d
