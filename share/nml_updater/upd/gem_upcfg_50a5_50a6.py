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
        print("Updating from gem 5.0.a5 to 5.0.a6 (%s, %s)" % (fromVersion, toVersion))

    # Special Cases
    # New: convection_cfgs / deep_timeconv
    # New: convection_cfgs / deep_timerefresh
    # New: convection_cfgs / shal_timeconv
    # rm: convection_cfgs / kfctimea
    # rm: convection_cfgs / kfctimec
    # rm: convection_cfgs / bkf_lrefresh
    # rm: convection_cfgs / bkf_lsettadj
    # rm: convection_cfgs / bkf_xtadjd
    # rm: convection_cfgs / bkf_xtadjs
    nml1 = 'convection_cfgs'
    try:
        nml = cfg.get(nml1)
    except:
        nml = None
    if nml:
        # bkf_xtadjs (int) > shal_timeconv (str)
        var1 = 'bkf_xtadjs'
        var2 = 'shal_timeconv'
        units = 's'
        try:
            val = int(nml.get(var1).get('v').toStr())
            nml.rm(var1)
            nml.add(
                FtnNmlKeyVal(var2, FtnNmlVal("'"+str(val)+units+"'"))
                )
            if verbose or debug:
                sys.stdout.write('Convert: %s/%s => %s\n'
                                 % (nml1, var1, var2))
        except:
            pass
        try:
            val = nml.get('deep').get('v').toStr()
            bkf_deep = val.upper() == 'BECHTOLD'
        except:
            bkf_deep = None
        if bkf_deep:
            ## bkf_lrefresh : deep_timerefresh = '1p' if T else None
            ## bkf_lsettadj : deep_timeconv = 'advective' if T else bkf_xtadjd
            ## bkf_xtadjd   : if bkf_lsettadj in (None, False)
            ##                bkf_xtadjd (int) > deep_timeconv (str)
            try:
                var1 = 'bkf_lrefresh'
                val = nml.get(var1).get('v').toStr()
                nml.rm(var1)
                bkf_lrefresh = val.lower() in ('.t.', '.true.')
            except:
                bkf_lrefresh = False
            try:
                var1 = 'bkf_lsettadj'
                val = nml.get(var1).get('v').toStr()
                nml.rm(var1)
                bkf_lsettadj = val.lower() in ('.t.', '.true.')
            except:
                bkf_lsettadj = False
            try:
                var1 = 'bkf_xtadjd'
                val = nml.get(var1).get('v').toStr()
                nml.rm(var1)
                bkf_xtadjd = int(val)
            except:
                bkf_xtadjd = None
            if bkf_lrefresh:
                nml.add(
                    FtnNmlKeyVal('deep_timerefresh', FtnNmlVal("1p"))
                    )
                if verbose or debug:
                    var1 = 'bkf_lrefresh'
                    var2 = 'deep_timerefresh'
                    sys.stdout.write('Convert: %s/%s => %s\n'
                                     % (nml1, var1, var2))
            if bkf_lsettadj:
                nml.add(
                    FtnNmlKeyVal('deep_timeconv', FtnNmlVal("advective"))
                    )
                if verbose or debug:
                    var1 = 'bkf_lsettadj'
                    var2 = 'deep_timeconv'
                    sys.stdout.write('Convert: %s/%s => %s\n'
                                     % (nml1, var1, var2))
            elif bkf_xtadjd is not None:
                nml.add(
                    FtnNmlKeyVal('deep_timeconv',
                                 FtnNmlVal(str(bkf_xtadjd)+'s'))
                    )
                if verbose or debug:
                    var1 = 'bkf_xtadjd'
                    var2 = 'deep_timeconv'
                    sys.stdout.write('Convert: %s/%s => %s\n'
                                     % (nml1, var1, var2))
        else:
            ## kfctimea (int) > deep_timerefresh (str)
            ## kfctimec (int) > deep_timeconv (str)
            var1 = 'kfctimea'
            var2 = 'deep_timerefresh'
            try:
                val = int(float(nml.get(var1).get('v').toStr()))
                nml.rm(var1)
            except:
                val = None
            try:
                if val is not None:
                    nml.add(
                        FtnNmlKeyVal(var2,
                                     FtnNmlVal(str(val)+'s'))
                        )
                    if verbose or debug:
                        sys.stdout.write('Convert: %s/%s => %s\n'
                                         % (nml1, var1, var2))
            except:
                pass
            var1 = 'kfctimec'
            var2 = 'deep_timeconv'
            try:
                val = int(float(nml.get(var1).get('v').toStr()))
                nml.rm(var1)
            except:
                val = None
            try:
                if val is not None:
                    nml.add(
                        FtnNmlKeyVal(var2,
                                     FtnNmlVal(str(val)+'s'))
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
