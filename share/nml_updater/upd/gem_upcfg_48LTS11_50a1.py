#!/usr/bin/env python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
"""

from rpnpy.ftnnml import *

def main(cfg, fromVersion=None, toVersion=None, verbose=False, debug=False):
    """
    """
    if verbose or debug:
        print("Updating from gem 4.8-LTS.11 to 5.0.a1 (%s, %s)" % (fromVersion, toVersion))

    # Convert schm_psadj
    # rm: gem_cfgs / schm_psadj
    # New: gem_cfgs  /schm_psadj_l
    nml1 = 'gem_cfgs'
    var1 = 'schm_psadj'
    var2 = 'schm_psadj_l'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            val = '.true.' if int(val) > 0 else '.false.'
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal(val))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass

    # Convert vtopo to str
    # rm: gem_cfgs / vtopo_ndt
    # rm: gem_cfgs / vtopo_start
    # New: gem_cfgs / vtopo_length_s
    # New: gem_cfgs / vtopo_start_s
    nml1 = 'gem_cfgs'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        var1 = 'vtopo_ndt'
        var2 = 'vtopo_length_s'
        units = 'p'
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal("'"+str(val)+units+"'"))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass
        var1 = 'vtopo_start'
        var2 = 'vtopo_start_s'
        units = 'p'
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal("'"+str(val)+units+"'"))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass

    # Convert fcst_alarm_s
    # New: step / fcst_alarm_s
    # rm: step / step_alarm
    nml1 = 'step'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        var1 = 'step_alarm'
        var2 = 'fcst_alarm_s'
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            units = 'p'
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal("'"+str(val)+units+"'"))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass

    # Convert grid iref
    # rm: grid / grd_iref
    # rm: grid / grd_jref
    # rm: grdc / grdc_iref
    # rm: grdc / grdc_jref
    nml1 = 'grid'
    try:
        nml = cfg.get(nml1)
    except:
        raise
        nml = None
    if nml:
        try:
            val = nml.get('Grd_typ_S').get('v').toStr()
        except:
            val = None
        if val.lower().strip(' \'"') == 'lu':
            try:
                val_iref = nml.get('Grd_iref').get('v').toStr()
                val_jref = nml.get('Grd_jref').get('v').toStr()
                val_dx = nml.get('Grd_dx').get('v').toStr()
                val_dy = nml.get('Grd_dy').get('v').toStr()
                val_latr = nml.get('Grd_latr').get('v').toStr()
                val_lonr = nml.get('Grd_lonr').get('v').toStr()
                val_ni = nml.get('Grd_ni').get('v').toStr()
                val_nj = nml.get('Grd_nj').get('v').toStr()
            except:
                val_iref = None
                val_jref = None
                val_dx = None
                val_dy = None
                val_latr = None
                val_lonr = None
                val_ni = None
                val_nj = None
                raise
            if val_iref is not None:
                di = int(val_ni) // 2 - int(val_iref)
                dj = int(val_nj) // 2 - int(val_jref)
                dlat = dj * float(val_dy)
                dlon = di * float(val_dx)
                val_latr = float(val_latr) + float(dlat)
                val_lonr = float(val_lonr) + float(dlon)
                nml.rm('Grd_latr')
                nml.rm('Grd_lonr')
                nml.add(
                    FtnNmlKeyVal('Grd_latr', FtnNmlVal(val_latr))
                    )
                nml.add(
                    FtnNmlKeyVal('Grd_lonr', FtnNmlVal(val_lonr))
                    )
            try:
                nml.rm('Grd_iref')
            except:
                raise
                pass
            try:
                nml.rm('Grd_jref')
            except:
                raise
                pass
            if verbose or debug:
                sys.stdout.write('Convert: %s / Grd_iref, Grd_jref\n'
                                 % nml1)

    nml1 = 'grdc'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        try:
            val_iref = nml.get('Grdc_iref').get('v').toStr()
            val_jref = nml.get('Grdc_jref').get('v').toStr()
            val_dx = nml.get('Grdc_dx').get('v').toStr()
            val_dy = nml.get('Grdc_dy').get('v').toStr()
            val_latr = nml.get('Grdc_latr').get('v').toStr()
            val_lonr = nml.get('Grdc_lonr').get('v').toStr()
            val_ni = nml.get('Grdc_ni').get('v').toStr()
            val_nj = nml.get('Grdc_nj').get('v').toStr()
        except:
            val_iref = None
            val_jref = None
            val_dx = None
            val_dy = None
            val_latr = None
            val_lonr = None
            val_ni = None
            val_nj = None
        if val_iref is not None:
            dlat = int(val_ni) // 2 - int(val_iref)
            dlon = int(val_nj) // 2 - int(val_jref)
            val_latr = float(val_latr) + float(dlat)
            val_lonr = float(val_lonr) + float(dlon)
            nml.rm('Grdc_latr')
            nml.rm('Grdc_lonr')
            nml.add(
                FtnNmlKeyVal('Grdc_latr', FtnNmlVal(val_latr))
                )
            nml.add(
                FtnNmlKeyVal('Grdc_lonr', FtnNmlVal(val_lonr))
                )
        try:
            nml.rm('Grdc_iref')
        except:
            pass
        try:
            nml.rm('Grdc_jref')
        except:
            pass
        if verbose or debug:
            sys.stdout.write('Convert: %s / Grdc_iref, Grdc_jref\n'
                             % nml1)

    return cfg

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
