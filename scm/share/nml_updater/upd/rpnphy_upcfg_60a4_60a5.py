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
        print("Updating from gem 5.0.a4 to 5.0.a5 (%s, %s)" % (fromVersion, toVersion))
    # Special Cases
    # Convert schm_psadj
    # rm: gem_cfgs / schm_psadj_l
    # New: gem_cfgs / schm_psadj
    nml1 = 'gem_cfgs'
    var1 = 'schm_psadj_l'
    var2 = 'schm_psadj'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            val = 1 if val.lower() in ('.t.', '.true.') else 0
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal(val))
                )
            sys.stdout.write('WARNING: Convert: %s/%s => %s ; Check its value\n'
                             % (nml1, var1, var2))
        except:
            pass

    # Convert vtopo to str
    # rm: mtn_cfgs / vtopo_ndt
    # rm: mtn_cfgs / vtopo_start
    # New: mtn_cfgs / vtopo_length_s
    # New: mtn_cfgs / vtopo_start_s
    nml1 = 'mtn_cfgs'
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

    # Convert z0trdps300 to z0ttype
    # rm: surface_cfgs / z0trdps300
    # New: surface_cfgs / z0ttype
    nml1 = 'surface_cfgs'
    var1 = 'z0trdps300'
    var2 = 'z0ttype'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        try:
            val = nml.get(var1).get('v').toStr()
            nml.rm(var1)
            val = 'DEACU12' if val.lower() in ('.t.', '.true.') else 'MOMENTUM'
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal(val))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass

    return cfg

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
