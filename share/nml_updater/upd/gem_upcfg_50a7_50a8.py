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
        print("Updating from gem 5.0.a7 to 5.0.a8 (%s, %s)" % (fromVersion, toVersion))

    # Special Cases
    # rm: gem_cfgs / hzd_smago_theta_nobase_l = F
    # New: gem_cfgs / hzd_smago_theta_base_l = T
    # rm: gem_cfgs / hzd_smago_bot_lev = 0.7000000
    # rm: gem_cfgs / hzd_smago_top_lev = 0.4000000
    # New: gem_cfgs / hzd_smago_lev = 0.7000000    ,  0.4000000
    # rm: gem_cfgs / hzd_smago_lnr = 0.0000000E+00
    # rm: gem_cfgs / hzd_smago_min_lnr = -1.000000
    # New: gem_cfgs / hzd_smago_lnr = -1.000000    ,  0.0000000E+00,  -1.000000

    nml1 = 'gem_cfgs'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None

    if not nml:
        sys.stdout.write('Warning: could not get the &{} Namelist - no update doe\n'.format(nml1))
        return cfg

    # rm: gem_cfgs / hzd_smago_theta_nobase_l = F
    # New: gem_cfgs / hzd_smago_theta_base_l = T
    var1 = 'hzd_smago_theta_nobase_l'
    var2 = 'hzd_smago_theta_base_l'
    try:
        val = nml.get(var1).get('v').toStr().upper()
        nml.rm(var1)
        if val in ('F', '.F.', 'FALSE', '.FALSE.'):
            val = '.TRUE.'
        else:
            val = '.FALSE.'
        nml.add(
            FtnNmlKeyVal(var2, FtnNmlVal(val))
            )
        if verbose or debug:
            sys.stdout.write('Convert: {}/{} => {}\n'
                             .format(nml1, var1, var2))
    except:
        pass

    # rm: gem_cfgs / hzd_smago_bot_lev = 0.7000000
    # rm: gem_cfgs / hzd_smago_top_lev = 0.4000000
    # New: gem_cfgs / hzd_smago_lev = 0.7000000    ,  0.4000000
    var1a = 'hzd_smago_bot_lev'
    var1b = 'hzd_smago_top_lev'
    var2 = 'hzd_smago_lev'
    found = False
    try:
        val1a = float(nml.get(var1a).get('v').toStr())
        found = True
        nml.rm(var1a)
    except:
        val1a = 0.7
    try:
        val1b = float(nml.get(var1b).get('v').toStr())
        found = True
        nml.rm(var1b)
    except:
        val1b = 0.4
    if found:
        nml.add(
            FtnNmlKeyVal(var2, FtnNmlVal('{}, {}'.format(val1a, val1b)))
            )
        if verbose or debug:
            sys.stdout.write('Convert: {}/{} + {} => {}\n'
                             .format(nml1, var1a, var1b, var2))

    # rm: gem_cfgs / hzd_smago_lnr = 0.0000000E+00
    # rm: gem_cfgs / hzd_smago_min_lnr = -1.000000
    # New: gem_cfgs / hzd_smago_lnr = -1.000000    ,  0.0000000E+00,  -1.000000
    var1a = 'hzd_smago_min_lnr'
    var1b = 'hzd_smago_lnr'
    var2  = 'hzd_smago_lnr'
    found = False
    try:
        val1a = float(nml.get(var1a).get('v').toStr())
        found = True
        nml.rm(var1a)
    except:
        val1a = 0.
    try:
        val1b = float(nml.get(var1b).get('v').toStr())
        found = True
        nml.rm(var1b)
    except:
        val1b = -1.
    if found:
        nml.add(
            FtnNmlKeyVal(var2, FtnNmlVal('{}, {}, {}'
                                         .format(val1a, val1b, val1b)))
            )
        if verbose or debug:
            sys.stdout.write('Convert: {}/{} + {}  => {}\n'
                             .format(nml1, var1a, var1b, var2))

    return cfg

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
