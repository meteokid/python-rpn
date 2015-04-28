#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import os
import rpnpy.librmn.all as rmn
import unittest
## import ctypes as ct
import numpy as np

class Librmn_grids_Test(unittest.TestCase):

    epsilon = 0.0005

    def test_degGrid_L(self):
        params = rmn.defGrid_L(90,45,0.,180.,1.,0.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_E(self):
        params = rmn.defGrid_E(90,45,0.,180.,1.,270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
            
    def test_degGrid_ZE(self):
        params = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,0.,180.,1.,270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                self.assertFalse(ok)
            else:
                self.assertEqual(params[k],params2[k])
            
    def test_degGrid_diezeE(self):
        params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,0.,180.,1.,270.,lni=90,lnj=45,i0=1,j0=1)
        params2 = rmn.decodeGrid(params['id'])
    ##     for k in params.keys():
    ##         if isinstance(params[k],np.ndarray):
    ##             ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
    ##             self.assertFalse(ok)
    ##         else:
    ##             self.assertEqual(params[k],params2[k],'p[%s] : %s != %s' % (k,str(params[k]),str(params2[k])))
        
    def test_degGrid_G_glb(self):
        params = rmn.defGrid_G(90,45,glb=True,north=True,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_G_glb_inv(self):
        params = rmn.defGrid_G(90,45,glb=True,north=True,inverted=True)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_G_N(self):
        params = rmn.defGrid_G(90,45,glb=False,north=True,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_G_S(self):
        params = rmn.defGrid_G(90,45,glb=False,north=False,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_PS_N(self):
        params = rmn.defGrid_PS(90,45,north=True,pi=45,pj=30,d60=5000.,dgrw=270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_degGrid_PS_S(self):
        params = rmn.defGrid_PS(90,45,north=False,pi=45,pj=30,d60=5000.,dgrw=270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])

    def test_degGrid_YY(self):
        params = rmn.defGrid_YY(31,1.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if k == 'subgrid':
                pass
            elif isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                self.assertFalse(ok)
            elif isinstance(params[k],float):
                self.assertTrue(abs(params[k]-params2[k]) < self.epsilon)
            else:
                self.assertEqual(params[k],params2[k])
        for i in (0,1):
            p0 = params['subgrid'][i]
            p2 = params2['subgrid'][i]
            for k in p0.keys():
                if isinstance(p0[k],np.ndarray):
                    ok = np.any(np.abs(p0[k]-p2[k]) > self.epsilon)
                    self.assertFalse(ok)
                elif isinstance(p0[k],float):
                    self.assertTrue(abs(p0[k]-p2[k]) < self.epsilon)
                else:
                    self.assertEqual(p0[k],p2[k])
            


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
