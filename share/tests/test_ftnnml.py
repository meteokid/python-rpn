#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.base"""

from rpnpy.ftnnml import *
import unittest

#--- primitives -----------------------------------------------------

## class ftnnmlTests(unittest.TestCase):

##     def testNeedSomeTests(self):
##         self.assertTrue(False)

def not_a_test():
    # Read the namelist file
    cfg = FtnNmlFile('gem_settings.nml')
    # Print list of present namelists name
    print cfg.keys()
    # Get the gem_cfgs FtnNmlSection object
    gemcfgs = cfg.get('gem_cfgs')
    # Print list of present vars name in gem_cfgs
    print gemcfgs.keys()
    # Print all present vars name in gem_cfgs with values
    for vname in gemcfgs.keys():
        print vname,gemcfgs.get(vname).get('v')
    # Check if a var is set
    try:
        gemcfgs.get('hyb')
    except KeyError:
        print "Key %s is not found in namelist %s of file %s" % ('hyb', 'gem_cfgs','gem_settings.nml')
    # Delete a var in a namlist
    gemcfgs.rm('hyb')
    # Delete a namelist from the file
    cfg.rm('grid')
    # Rename a namlist
    cfg.get('grid_YU').rename('grid')
    # Rename a namelist var
    cfg.get('grid').get('Grd_typ_S').rename('Grd_typ')
    # Change the value of a namelist var
    cfg.get('grid').get('grd_ni').get('v').set(333)
    # Add a new namelist var
    cfg.get('grid').add(FtnNmlKeyVal('grd_nj2',FtnNmlVal(999)))
    # Add new namelist
    cfg.add(FtnNmlSection('new_cfgs'))
    # Move namlist var into a new namelist
    newnml = FtnNmlSection('new_cfgs2')
    mykeyval = cfg.get('gem_cfgs').get('Schm_hydro_L')
    newnml.add(mykeyval)
    cfg.add(newnml)
    cfg.get('gem_cfgs').rm('Schm_hydro_L')
    # Write the resulting namelist file to disk
    cfg.write('gem_settings.nml2',clean=True)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
