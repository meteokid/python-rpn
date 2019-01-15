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
        print("Updating from gem 4.7.3 to 4.8.rc5 (%s, %s)" % (fromVersion, toVersion))

    # mv: gem_cfgs / out3_closestep = 0 > gem_cfgs / out3_close_interval_s
    # mv: gem_cfgs / out3_postfreq_s > gem_cfgs / out3_postproc_fact
    # rm: gem_cfgs / out3_unit_s
    closestep  = None
    postfreq_s = None
    unit_s     = None
    dt         = None
    nml1 = 'gem_cfgs'
    try:
        dt = float(cfg.get('step').get('step_dt').get('v').toStr())
    except:
        pass
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        try:
            closestep  = nml.get('out3_closestep').get('v').toStr()
            nml.rm('out3_closestep')
        except:
            pass
        try:
            postfreq_s = nml.get('out3_postfreq_s').get('v').toStr()
            nml.rm('out3_postfreq_s')
        except:
            pass
        try:
            unit_s     = nml.get('out3_unit_s').get('v').toStr()
            nml.rm('out3_unit_s')
        except:
            pass
        if not closestep is None:
            closestep_sec = None
            if dt is None:
                float(closestep) * dt
            var1 = 'out3_closestep'
            var2 = 'out3_close_interval_s'
            units = 'p'
            if not unit_s is None:
                if unit_s.lower().strip() != 'p':
                    if dt is None:
                        closestep = None
                        sys.stderr.write('Error, missing dt, cannot convert: %s/%s\n'
                                         % (nml1, 'out3_unit_s'))
                    else:
                        units = unit_s.lower().strip()
                        if units == 's':
                            closestep = closestep_sec
                        elif units == 'm':
                            closestep = closestep_sec / 60.
                        elif units == 'h':
                            closestep = closestep_sec / 3600.
                        elif units == 'd':
                            closestep = closestep_sec / 86400.
            if not closestep is None:
                cfg.get(nml1).add(
                    FtnNmlKeyVal(var2, FtnNmlVal("'"+str(closestep)+units+"'"))
                    )
                if verbose or debug:
                    sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        if not postfreq_s is None:
            var1 = 'out3_postfreq_s'
            var2 = 'out3_postproc_fact'
            value = int(postfreq_s.strip()[0:-1])
            units = postfreq_s.strip().upper()[-1]
            if units == 'p':
                value /= int(closestep)
            elif dt is None:
                value = None
            else:
                if units == 's':
                    value = float(value) / closestep_sec
                elif units == 'm':
                    value = float(value) * 60. / closestep_sec
                elif units == 'h':
                    value = float(value) * 3600. / closestep_sec
                elif units == 'd':
                    value = float(value) * 86400 / closestep_sec
                else:
                    try:
                        value = float(postfreq_s.strip()) * dt
                    except:
                        value = None
            if not value is None:
                cfg.get(nml1).add(
                    FtnNmlKeyVal(var2, FtnNmlVal(str(value)))
                    )
                if verbose or debug:
                    sys.stdout.write('Convert: %s/%s => %s\n'
                                     % (nml1, var1, var2))
            else:
                sys.stderr.write('Error, cannot convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))

    # Set: physics_cfgs / stcond = 'MY_DM*' > 'MP_MY2_OLD'
    try:
        stcondv = cfg.get('physics_cfgs').get('stcond')
        stcond =  stcondv.get('v').toStr()
        if stcond.strip().upper()[0:5] == 'MY_DM':
            stcondv.data = 'MP_MY2_OLD'
            if verbose or debug:
                sys.stdout.write('Convert: physics_cfgs/stcond = "MY_DM" => "MP_MY2_OLD"\n')
    except:
        pass

    return cfg

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
