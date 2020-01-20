#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from rpnpy import integer_types as _integer_types

#TODO: fst_read_3d_sample points

def sort_ip1(ip1s):
    """
    Sort unique a list of ip1 according to the value type
    Args:
        ip1s : list of ip1 to be sorted [list of int]
    Returns:
        list of int, sorted ip1s, double values are removed
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        RMNError   on any other error

    Examples:
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> ip1s = [1195, 1196, 1197, 1198, 1199, 1199]
    >>> ip1s = fstd3d.sort_ip1(ip1s)
    >>> print("# {}".format(repr(ip1s)))
    # [1199, 1198, 1197, 1196, 1195]

    See Also:
        get_levels_keys
        rpnpy.librmn.fstd98.convertIp
    """
    if isinstance(ip1s, (list, tuple)):
        if not all([(isinstance(i, _integer_types) and i >= 0) for i in ip1s]):
            raise ValueError('All ip1s should be int >= 0, got {}'.format(repr(ip1s)))
    else:
        raise TypeError('ip1s should be a list of int, got {} {}'
                        .format(repr(ip1s), type(ip1s)))

    if len(ip1s) < 2:
        return ip1s

    # Remove duplicate ip1
    ip1s = sorted(list(set(ip1s)))

    # Check sort order
    n2 = (len(ip1s)-1)//2
    lvlType = _rmn.convertIp(_rmn.CONVIP_DECODE, ip1s[n2])[1]
    reverse = True
    if lvlType in (_rmn.LEVEL_KIND_MSL, _rmn.LEVEL_KIND_ANY,
                   _rmn.LEVEL_KIND_MGL):
        reverse = False

    # Get real values
    lvls = [(i, _rmn.convertIp(_rmn.CONVIP_DECODE, i)[0]) for i in ip1s]

    # Sort
    ip1s = [i for i,l in sorted(lvls, key=lambda l: l[1], reverse=reverse)]

    return ip1s


#TODO: should also get rfls if needed
#TODO: skip phPa if not a press coor
def get_levels_press(fileId, vptr, shape, ip1list,
                     datev=-1, ip2=-1, ip3=-1, typvar=' ', etiket=' ',
                     verbose=False):
    """
    Read the reference surface field and computer the pressure cube

    press = get_levels_press(fileId, vptr, shape, ip1list)
    press = get_levels_press(fileId, vptr, shape, ip1list,
                             datev, ip2, ip3, typvar, etiket)

    Args:
        fileId  : unit number associated to the file
                  obtained with fnom+fstouv
        vptr    : vertical grid descriptor created with any
                  vgrid descriptor generating functions or vgd_read.
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
    >>>
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
    >>> press   = fstd3d.get_levels_press(fileId, ipkeys['vptr'], shape, ip1list)
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
    rfldName = _vgd.vgd_get(vptr, 'RFLD')
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
    #TODO: use vgd_leves2 with rfls
    phPa = _vgd.vgd_levels(vptr, rfld, ip1list)
    phPa[:, :, :] /= _cst.MB2PA
    return {
        'rfld' : rfld,
        'phPa' : phPa
        }


#TODO: get arbitrary list of levels as well
def get_levels_keys(fileId, nomvar, datev=-1, ip2=-1, ip3=-1,
                    typvar=' ', etiket=' ',
                    vptr=None, thermoMom='VIPT', verbose=False):
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
        vptr    : vertical grid descriptor created with any
                  vgrid descriptor generating functions or vgd_read.
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
        'vptr'   : vptr,    # vertical grid descriptor as returned by vgd_read
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
    if vptr is None:
        if verbose:
            print("Getting vertical grid description")
        _vgd.vgd_put_opt('ALLOW_SIGMA', _vgd.VGD_ALLOW_SIGMA)
        vptr = _vgd.vgd_read(fileId)

    vip = _vgd.vgd_get(vptr, thermoMom)
    if verbose:
        vkind = _vgd.vgd_get(vptr, 'KIND')
        vver  = _vgd.vgd_get(vptr, 'VERS')
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
        'vptr'   : vptr,
        'vgrid'  : vptr,  # kept for backward compat with v2.1.b3
        'ip1keys': vipkeys
        }


def vgrid_new(ip1s, vptr=None, rfld=None, rlfs=None, rfldError=True):
    """
    Pack vertical grid parameters in a dict from provided params
    and check consistency

    Args:
        ip1s  : List of ip1 [list of int] or
                'VIPM' or 'VIPT' to get list from vgd_get(vptr, 'VIP?') [str]
        vptr  : vgrid descriptor created with any
                vgrid descriptor generating functions.
        rfld  : Reference field for the vert.coor., data and meta [dict]
        rfls  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
        rfldError : Raise error on missing RFLD/S when needed
    Returns:
        {
            'ip1s'  : List of ip1 [list of int]
            'vptr'  : vgrid descriptor created with any
                      vgrid descriptor generating functions.
            'rfld'  : Reference field for the vert.coor., data and meta [dict]
            'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error

    Examples:
    >>> import numpy  as np
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> import rpnpy.vgd.all as vgd
    >>> from rpnpy.rpndate import RPNDate
    >>>
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
    ...     v = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>>
    >>> vgrid = fstd3d.vgrid_new('VIPT', v, rfldError=False)
    >>> print("# nblevels={}".format(len(vgrid['ip1s'])))
    # nblevels=28
    """
    if isinstance(ip1s, _integer_types):
        ip1s = [ip1s]
    if isinstance(ip1s, (list, tuple)):
        if not all([(isinstance(i, _integer_types) and i >= 0) for i in ip1s]):
            raise ValueError('All ip1s should be int >= 0')
    elif isinstance(ip1s, str):
        if ip1s.strip().upper() in ('VIPM', 'VIPT') and vptr is None:
            raise TypeError('With ip1s=VIPM or VIPT, vptr should be provided')
    else:
        raise ValueError('All ip1s should be int >= 0 or VIPM or VIPT')

    if not (vptr is None or isinstance(vptr, type(_vgd.c_vgd_construct()))):
            raise TypeError('vptr should be None or an instance of c_vgd_construct()')

    v = {
        'ip1s'  : ip1s,
        'vptr'  : vptr,
        'rfld'  : None,
        'rfls'  : None
        }

    if not vptr:
        return v

    if isinstance(ip1s, str):
        ip1s = _vgd.vgd_get(vptr, ip1s)
        v['ip1s'] = ip1s
    else:
        ip1sm = _vgd.vgd_get(vptr, 'VIPM')
        ip1st = _vgd.vgd_get(vptr, 'VIPT')
        if not(all([(i in ip1sm) for i in ip1s]) or
               all([(i in ip1st) for i in ip1s])):
            raise ValueError('Provided ip1s are not a subset of vptr')

    rfldname = _vgd.vgd_get(vptr, 'RFLD', defaultOnFail=True)
    if rfldname:
        if rfld is None:
            if rfldError:
                raise TypeError('rfld should provided')
        elif not isinstance(rfld, dict):
            raise TypeError('rfld should be a dict')
        elif ('nomvar' not in rfld.keys() or
              rfld['nomvar'].upper().strip() != rfldname.upper().strip()):
            raise ValueError('wrong field for rlfd')
        if rfld:
            v['rfld'] = rfld

    rflsname = _vgd.vgd_get(vptr, 'RFLS', defaultOnFail=True)
    if rflsname:
        if rfls is None:
            if rfldError:
                raise TypeError('rfls should provided')
        elif not isinstance(rfls, dict):
            raise TypeError('rfls should be a dict')
        elif ('nomvar' not in rfls.keys() or
              rfls['nomvar'].upper().strip() != rflsname.upper().strip()):
            raise ValueError('wrong field for rlfd')
        if rfls:
            v['rfls'] = rfls

    return v


def vgrid_write(fileId, vgrid, writeRfld=False, verbose=False):
    """
    Write the vertical grid parameters
    Write the vertical ref. surface fields as well if requested

    Args:
        fileId : unit number associated to the file
                   obtained with fnom+fstouv or fstopenall
        vgrid  : dict as returned by readVGrid() or vgrid_new()
            {
            'ip1s'  : List of ip1 [list of int]
            'vptr'  : vertical grid info as returned by vgd_read
            'rfld'  : Reference field for the vert.coor., data and meta [dict]
            'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
            }
        writeRfld : write vgrid['rfld'] and vgrid['rfls'] if any
        verbose   : Print some info when true
    Returns:
        None
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error
        RMNError   on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>>
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk')
    >>> fileId  = rmn.fstopenall(myfile)
    >>> vgridTT = fstd3d.vgrid_read(fileId, nomvar='TT', ip2=0)
    >>> rmn.fstcloseall(fileId)
    >>>
    >>> TMPDIR = os.getenv('TMPDIR')
    >>> myfile = os.path.join(TMPDIR, 'vgrid_writeTest.fst')
    >>> fileId = rmn.fstopenall(myfile, rmn.FST_RW)
    >>> fstd3d.vgrid_write(fileId, vgridTT, writeRfld=True)
    >>> rmn.fstcloseall(fileId)
    >>> os.unlink(myfile)

    See Also:
        vgrid_new
        vgrid_write
        fst_read_3d
        get_levels_press
        get_levels_keys
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
    """
    if not isinstance(vgrid, dict):
        raise TypeError('Provided vgrid should be a dict')
    if 'vgrid' in vgrid.keys():
        raise ValueError('Provided vgrid should be a dict with "vgrid" key')
    if vgrid['vptr'] is None:
        if verbose:
            print("vgrid_write: Nothing to be done.")
        return

    if writeRfld:
        for rfl in ('RFLD', 'RFLS'):
            rfldname = _vgd.vgd_get(vgrid['vptr'], rfl, defaultOnFail=True)
            if rfldname and rfl.lower() in vgrid.keys():
                rfldict = vgrid[rfl.lower()]
                _rmn.fstecr(fileId, rfldict['d'], rfldict, rewrite=True)
                if 'hgrid' in rfldict.keys():
                    _rmn.writeGrid(fileId, rfldict['hgrid'])
                if verbose:
                    print("vgrid_write: wrote {}={}"
                          .format(rfl, vgrid[rfl.lower()]['nomvar']))

    _vgd.vgd_write(vgrid['vptr'], fileId)
    if verbose:
        print("vgrid_write: wrote vgrid")


def vgrid_read(fileId, datev=-1, etiket=' ', ip1=-1, ip2=-1, ip3=-1,
              typvar=' ', nomvar=' ', verbose=False):
    """
    Read the vertical grid parameters from provided selection params
    Read surface ref fields from file if need be

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
        verbose : Print some info when true
    Returns:
        None if no matching record, else:
        {
            'ip1s'  : List of ip1 [list of int]
            'keys'  : List of keys associated with ip1s [list of int]
            'vptr'  : vertical grid info as returned by vgd_read
            'rfld'  : Reference field for the vert.coor., data and meta [dict]
            'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error
        RMNError   on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>>
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
    >>> myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk')
    >>> fileId  = rmn.fstopenall(myfile)
    >>>
    >>> vgridTT = fstd3d.vgrid_read(fileId, nomvar='TT', ip2=0)
    >>> print("# nblevels={}, rfld={}".format(len(vgridTT['ip1s']), vgridTT['rfld']['nomvar'].strip()))
    # nblevels=80, rfld=P0
    >>>
    >>> vgridJ1 = fstd3d.vgrid_read(fileId, nomvar='J1')
    >>> print("# nblevels={}".format(len(vgridJ1['ip1s'])))
    # nblevels=5
    >>>
    >>> rmn.fstcloseall(fileId)

    See Also:
        vgrid_new
        vgrid_write
        fst_read_3d
        get_levels_press
        get_levels_keys
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
    """
    # Get the list of ip1 on thermo, momentum levels in this file
    vptr = None
    tlevels = get_levels_keys(fileId, nomvar, datev, ip2, ip3, typvar, etiket,
                              vptr=vptr, thermoMom='VIPT', verbose=verbose)
    vptr = tlevels['vptr']
    mlevels = get_levels_keys(fileId, tlevels['nomvar'], tlevels['datev'],
                              tlevels['ip2'], tlevels['ip3'],
                              tlevels['typvar'], tlevels['etiket'],
                              vptr=vptr, thermoMom='VIPM', verbose=verbose)

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
        ip1keysdict = dict([(_rmn.fstprm(k)['ip1'], k) for k in keys])
        ip1keys = [(i, ip1keysdict[i]) for i in
                   sort_ip1(list(ip1keysdict.keys()))]
        vptr = None
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
            vptr = None
            ip1keys = [(i,None) for i in ip1]
    elif ip1 not in (-1, None):
        raise TypeError('Provided ip1 should be a list or an int')

    if verbose or ip1keys is None or len(ip1keys) == 0:
        print("(fst_read_3d) Found {0} records for {1} ip1={2} ip2={3} ip3={4} datev={5} typvar={6} etiket={7}"
              .format(len(ip1keys), nomvar, ip1, ip2, ip3, datev, typvar, etiket))

    if len(ip1keys) == 0:
        return None
    ip1s = [i for i,k in ip1keys]

    rlfds = {
        'RFLD' : None,
        'RFLS' : None
        }
    if vptr:
        for rfl in ('RFLD', 'RFLS'):
            rfldname = _vgd.vgd_get(vptr, rfl, defaultOnFail=True)
            if rfldname:
                r = _rmn.fstlir(fileId, datev=datev, ip2=ip2, nomvar=rfldname)
                if not r and datev != -1:
                    r = _rmn.fstlir(fileId, datev=datev, nomvar=rfldname)
                if not r and ip2 != -1:
                    r = _rmn.fstlir(fileId, ip2=ip2, nomvar=rfldname)
                if not r and not (datev == -1 and ip2 == -1):
                    r = _rmn.fstlir(fileId, nomvar=rfldname)
                if r:
                    r['hgrid'] = _rmn.readGrid(fileId, r)
                    rlfds[rfl] = r
    v = vgrid_new(ip1s, vptr=vptr, rfld=rlfds['RFLD'], rlfs=rlfds['RFLS'])
    v['keys'] = [k for i,k in ip1keys]
    return v


#TODO: accept datev as a dict for all other params
#TODO: skip phPa if not a press coor
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
        dataArray: (optional) allocated array where to put the data [ndarray]
        verbose : Print some info when true
    Returns:
        None if no matching record, else:
        {
            'd'    : 3d field data (numpy.ndarray), Fortran order
            'hgrid': horizontal grid info as returned by readGrid
            'vgrid': vertical grid info as returned by vgrid_read
                {
                'ip1s'  : List of ip1 [list of int]
                'keys'  : List of keys associated with ip1s [list of int]
                'vptr'  : vertical grid info as returned by vgd_read
                'rfld'  : Reference field for the vert.coor., data and meta [dict]
                'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
                }
            ...    : ...same params list as fstprm (less ip1)...
            'phPa' : (if getPress) 3d pressure values
        }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error
        FSTDError  on any other error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>>
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>>
    >>> # Open existing file in Rear Only mode
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> fileId = rmn.fstopenall(filename, rmn.FST_RO)
    >>>
    >>> # Find and read J1 meta and data, then print its min,max,mean values
    >>> j1 = fstd3d.fst_read_3d(fileId, nomvar='J1')
    >>> print("# J1 min={:4.1f} max={:3.1f} avg={:4.1f}"
    ...       .format(j1['d'].min(), j1['d'].max(), j1['d'].mean()))
    # J1 min= 0.0 max=92.0 avg=18.8
    >>>
    >>> # Find and read specific levels J1 meta and data, then print its min,max,mean values
    >>> ip1s = (1199, 1198, 1197, 1196, 1195)
    >>> j2 = fstd3d.fst_read_3d(fileId, nomvar='J2', ip1=ip1s)
    >>> print("# J2 min={:4.1f} max={:3.1f} avg={:4.1f}"
    ...       .format(j2['d'].min(), j2['d'].max(), j2['d'].mean()))
    # J2 min= 0.0 max=94.4 avg= 8.8
    >>>
    >>> # Find and read TT meta and data, then print its min,max,mean values
    >>> tt3d = fstd3d.fst_read_3d(fileId, nomvar='TT')
    >>> print("# TT ip2={0} min={1:4.1f} max={2:3.1f} avg={3:4.1f}"
    ...       .format(tt3d['ip2'], tt3d['d'].min(), tt3d['d'].max(), tt3d['d'].mean()))
    # TT ip2=0 min=-88.4 max=40.3 avg=-36.3
    >>>
    >>> rmn.fstcloseall(fileId)

    See Also:
        get_levels_keys
        get_levels_press
        fst_write_3d
        vgrid_read
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_levels
    """
    vGrid = vgrid_read(fileId, datev, etiket, ip1, ip2, ip3,
                       typvar, nomvar, verbose)
    if vGrid is None:
        return None

    ip1keys = list(zip(vGrid['ip1s'], vGrid['keys']))

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
                r3d['d'] = _np.empty(rshape, dtype=r2d['d'].dtype, order='F')
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
        'vgrid' : vGrid
         })

    # For backward compat with v2.1.b3
    r3d['g'] = r3d['hgrid']
    r3d['v'] = r3d['vgrid']['vptr']
    r3d['ip1s'] = r3d['vgrid']['ip1s']

    # Read RFLD and compute pressure on levels
    if getPress and r3d['vgrid'] and r3d['vgrid']['vptr']:
        press = get_levels_press(fileId, r3d['vgrid']['vptr'],
                                 r3d['d'].shape[0:2], ip1list,
                                 r3d['datev'], r3d['ip2'],
                                 r3d['ip3'], r3d['typvar'],
                                 r3d['etiket'], verbose=False)
        r3d.update({
            'rfld' : press['rfld'], # For backward compat with v2.1.b3
            'phPa' : press['phPa']
            })

    return r3d


def fst_write_3d(fileId, rec3d, verbose=False):
    """
    Write a RPNStd/FST list of records for each level of rec3d
    along with its horizontal and vertical description.

    field3d = fst_write_3d(fileId, r3d)

    Args:
        fileId : unit number associated to the file
                 obtained with fnom+fstouv or fstopenall
        rec3d  : dict with 3D record data and meta
           {
              d     : data [numpy.ndarray]
              hgrid : horizontal grid description [dict]
                      result of rmn.encodeGrid()
              vgrid : dict as returned by readVGrid() or vgrid_new()
                  {
                  'ip1s' : List of ip1 [list of int]
                  'vptr' : vertical grid info as returned by vgd_read
                  'rfld' : Reference field for the vert.coor., data and meta [dict]
                  'rfls' : Reference field (SLEVE) for the vert.coor., data and meta [dict]
                  }
              ...    : ...same params list as fstprm (less ip1), fst_read_3d...
           }
        verbose: (optional) Print some info when true
    Returns:
        {
            'd'    : 3d field data (numpy.ndarray), Fortran order
            'hgrid': horizontal grid info as returned by readGrid
            'vgrid': vertical grid info as returned by vgrid_read
                {
                'ip1s'  : List of ip1 [list of int]
                'keys'  : List of keys associated with ip1s [list of int]
                'vptr'  : vertical grid info as returned by vgd_read
                'rfld'  : Reference field for the vert.coor., data and meta [dict]
                'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
                }
            ...    : ...same params list as fstprm (less ip1)...
         }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error

    Examples:
    >>> import os, os.path
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>>
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk')
    >>> rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    >>> fileId = rmn.fstopenall(filename, rmn.FST_RO)
    >>> tt3d = fstd3d.fst_read_3d(fileId, nomvar='TT')
    >>> rmn.fstcloseall(fileId)
    >>>
    >>> TMPDIR = os.getenv('TMPDIR')
    >>> myfile = os.path.join(TMPDIR, 'fst_write_3d.fst')
    >>> fileId = rmn.fstopenall(myfile, rmn.FST_RW)
    >>> fstd3d.fst_write_3d(fileId, tt3d)
    >>> rmn.fstcloseall(fileId)
    >>> os.unlink(myfile)

    See Also:
        get_levels_keys
        get_levels_press
        fst_read_3d
        vgrid_read
        vgrid_new
        vgrid_write
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstecr
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_levels
    """
    if not isinstance(rec3d, dict):
        raise TypeError('rec3d shoud be a dict')
    if not 'vgrid' in rec3d.keys():
        raise ValueError('rec3d should have a "vgrid" key')
    if not 'ip1s' in rec3d['vgrid'].keys():
        raise ValueError('rec3d["vgrid"] should have a "ip1s" key')
    if not 'hgrid' in rec3d.keys():
        raise ValueError('')
        raise ValueError('rec3d should have a "hgrid" key')
    if not 'd' in rec3d.keys():
        raise ValueError('rec3d should have a "d" key with data')
    if not isinstance(rec3d['d'], _np.ndarray):
        raise TypeError('rec3d["d"] should be of type numpy.ndarray')
    if len(rec3d['d'].shape) != 3 or any([i < 1 for i in rec3d['d'].shape]):
        raise ValueError('rec3d["d"] should be of rank 3')
    vgrid_write(fileId, rec3d['vgrid'], writeRfld=True, verbose=verbose)
    _rmn.writeGrid(fileId, rec3d['hgrid'])
    for k, ip1 in enumerate(rec3d['vgrid']['ip1s']):
        rec3d['ip1'] = ip1
        d = rec3d['d'][:,:,k]
        _rmn.fstecr(fileId, d, rec3d, rewrite=True)


#TODO: params could include all other args
def fst_new_3d(params, hgrid, vgrid,
               dtype=None, dataArray=None, verbose=False):
    """
    Create an RPNStd 3D record with provided data and param,
    along with horizontal and vertical grid info

    field3d = fst_new_3d(params, hgrid, vgrid, dataArray=dataArray)
    field3d = fst_new_3d(params, hgrid, dataArray=dataArray)
    field3d = fst_new_3d(params, hgrid, vgrid, dtype=dtype)
    field3d = fst_new_3d(params, hgrid, dtype=dtype)

    Args:
        params    : Field metadata would normally include these keys
                    'nomvar', 'dateo', 'deet', 'npas', 'ip2', 'etiket' [dict]
        hgrid     : horizontal grid description [dict]
                    result of rmn.encodeGrid()
        vgrid     : dict as returned by readVGrid() or vgrid_new()
            {
            'ip1s' : List of ip1 [list of int]
            'vptr' : vertical grid info as returned by vgd_read
            'rfld' : Reference field for the vert.coor., data and meta [dict]
            'rfls' : Reference field (SLEVE) for the vert.coor., data and meta [dict]
            }
        dtype     : (optional) numpy.dtype for array creation
                    (if params[''datyp'] nor dataArray are provided)
        dataArray : (optional) allocated array [numpy.ndarray]
        verbose   : (optional) Print some info when true
    Returns:
        {
            'd'    : 3d field data (numpy.ndarray), Fortran order
            'hgrid': horizontal grid info as returned by readGrid
            'vgrid': vertical grid info as returned by vgrid_read
                {
                'ip1s'  : List of ip1 [list of int]
                'keys'  : List of keys associated with ip1s [list of int]
                'vptr'  : vertical grid info as returned by vgd_read
                'rfld'  : Reference field for the vert.coor., data and meta [dict]
                'rfls'  : Reference field (SLEVE) for the vert.coor., data and meta [dict]
                }
            ...    : ...same params list as fstprm (less ip1)...
         }
    Raises:
        TypeError  on wrong input arg types
        ValueError on invalid input arg value
        VGDError   on any vgrid error

    Examples:
    >>> import os, os.path
    >>> import numpy  as np
    >>> import rpnpy.librmn.all as rmn
    >>> import rpnpy.utils.fstd3d as fstd3d
    >>> import rpnpy.vgd.all as vgd
    >>> from rpnpy.rpndate import RPNDate
    >>>
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
    >>>
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
    ...     vptr = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
    ... except vgd.VGDError:
    ...     sys.stderr.write("There was a problem creating the VGridDescriptor")
    >>> vgrid = fstd3d.vgrid_new('VIPT', vptr, rfldError=False)
    >>>
    >>> params = rmn.FST_RDE_META_DEFAULT.copy()
    >>> params['nomvar'] = 'tt'
    >>> params['dateo'] = RPNDate(20190625, 0).stamp
    >>> params['deet'] = 1800
    >>> params['npas'] = 6
    >>> params['etiket'] = 'myetk'
    >>>
    >>> # Create new 3d field for TT on vgrid Thermo levels
    >>> field3d = fstd3d.fst_new_3d(params, hgrid, vgrid=vgrid)
    >>>
    >>> field3d['d'] = 0.

    See Also:
        fst_read_3d
        fst_write_3d
        vgrid_new
        vgrid_read
        vgrid_write
        rpnpy.librmn.fstd98.const
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.grids.encodeGrid
        rpnpy.vgd.base.vgd_new_hybmd
        rpnpy.rpndate
    """
    rec3d = _rmn.FST_RDE_META_DEFAULT.copy()
    datyp = None
    if 'datyp' in params.keys():
        datyp = params['datyp']
        if dtype is None:
            dtype = _rmn.dtype_fst2numpy(datyp)
        elif dtype != _rmn.dtype_fst2numpy(datyp):
            raise ValueError("fst_new_3d: inconsistent dtype and datyp")
    rec3d.update(params)
    if datyp is not None:
        rec3d['datyp'] = datyp
    for k in rec3d.keys():
        if k in hgrid.keys():
            rec3d[k] = hgrid[k]
    (ni, nj) = (hgrid['ni'], hgrid['nj'])
    nk = len(vgrid['ip1s'])

    if dataArray is None:
        if dtype is None:
            dtype = _rmn.dtype_fst2numpy(params['datyp'], params['nbits'])
        dataArray = _np.zeros((ni,nj,nk), dtype=dtype, order='F')
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
    rec3d.update({
        'd' : dataArray,
        'datyp' : datyp,
        'shape' : dataArray.shape,
        'ni'    : dataArray.shape[0],
        'nj'    : dataArray.shape[1],
        'nk'    : dataArray.shape[2],
        'ip1'   : -1,
        'hgrid' : hgrid,
        'vgrid' : vgrid
         })
    if vgrid['vptr']:
        rec3d['vgrid']['vptr'] = _vgd.vgd_copy(vgrid['vptr'])
    return rec3d


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
