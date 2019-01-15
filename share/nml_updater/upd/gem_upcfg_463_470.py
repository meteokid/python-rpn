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
        print("Updating from gem 4.6.3 to 4.7.0 (%s, %s)" % (fromVersion, toVersion))
    # mv:  gem_cfgs / init_dfnp      > gem_cfgs / init_dflength_s
    # mv:  gem_cfgs / init_dfpl_8    > gem_cfgs / init_dfpl_s
    # mv:  gem_cfgs / out3_postfreq  > gem_cfgs / out3_postfreq_s
    # mv: step / step_rsti           > step / fcst_rstrt_s
    # mv: step / step_bkup           > step / fcst_bkup_s
    # mv: step / step_gstat          > step / fcst_gstat_s
    # mv: step / step_spinphy        > step / fcst_spinphy_s
    # mv: step / step_nesdt          > step / fcst_nesdt_s
    # mv: physics_cfgs / kntrad      > physics_cfgs / kntrad_s
    toconvert = (
        ('gem_cfgs' , ('init_dfnp','init_dflength_S','p')),
        ('gem_cfgs' , ('init_dfpl_8','init_dfpl_S','s')),
        ('gem_cfgs' , ('out3_postfreq','out3_postfreq_S','m')),
        ('step' , ('Step_rsti','Fcst_rstrt_S','p')),
        ('step' , ('Step_bkup','Fcst_bkup_S','p')),
        ('step' , ('Step_gstat','Fcst_gstat_S','p')),
        ('step' , ('Step_spinphy','Fcst_spinphy_S','p')),
        ('step' , ('Step_nesdt','Fcst_nesdt_S','s')),
        ('physics_cfgs' , ('kntrad','kntrad_S','p'))
    )
    for (nml1,items) in toconvert:
        nml2 = nml1
        (var1,var2,unit) = items
        value = None
        try:
            value = str(cfg.get(nml1).get(var1).get('v')).toStr()
            #print nml1,var1,value,type(values)
            cfg.get(nml1).rm(var1)
        except:
            if verbose or debug:
                sys.stdout.write('Ignore - not present: %s/%s => %s/%s\n'
                                 % (nml1,var1,nml2,var2))
            continue
        if value:
            try:
                ## try:
                ##     value = str(int(float(value)))
                ## except:
                ##     pass
                cfg.get(nml1).add(
                    FtnNmlKeyVal(var2, FtnNmlVal("'"+str(value)+unit+"'"))
                    )
                if verbose or debug:
                    sys.stdout.write('Convert: %s/%s => %s/%s\n'
                                     % (nml1,var1,nml2,var2))
            except:
                raise
                sys.stderr.write('Ignore1 - set error:  %s/%s => %s/%s\n'
                                 % (nml1,var1,nml2,var2))


    # mv:  physics_cfgs / veg_rs_x2  > surface_cfgs / veg_rs_mult
    (nml1,var1,nml2,var2) = ('physics_cfgs','veg_rs_x2',
                             'surface_cfgs','veg_rs_mult')
    value = None
    try:
        value = str(cfg.get(nml1).get(var1).get('v')).toStr()
        cfg.get(nml1).rm(var1)
    except:
        if verbose or debug:
            sys.stdout.write('Ignore - not present: %s/%s => %s/%s\n'
                             % (nml1,var1,nml2,var2))
    if value:
        try:
            if value.lower() in ('.true.', '.t.', 'true', 't'):
                value = 2.
            else:
                value = 1.
            cfg.get(nml2).add(
                FtnNmlKeyVal(var2, FtnNmlVal(str(value)))
            )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s/%s\n'
                                 % (nml1,var1,nml2,var2))
        except:
            sys.stderr.write('Ignore2 - set error: %s/%s => %s/%s\n'
                             % (nml1,var1,nml2,var2))

    return cfg

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
