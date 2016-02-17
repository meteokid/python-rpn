#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import os, os.path
import rpnpy.librmn.all as rmn
import unittest
## import ctypes as ct
import numpy as np

class Librmn_grids_Test(unittest.TestCase):

    epsilon = 0.0005

    def test_readGrid(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/geophy.fst')

        funit = rmn.fstopenall(myfile)
        rec   = rmn.fstlir(funit, nomvar='ME')
        grid  = rmn.readGrid(funit,rec)
        self.assertEqual(grid['grref'],'E')
        self.assertEqual(grid['grtyp'],'Z')
        self.assertEqual(grid['ig1'],2002)
        self.assertEqual(grid['ig2'],1000)
        self.assertEqual(grid['ig3'],0)
        self.assertEqual(grid['ig4'],0)
        self.assertEqual(grid['ig1ref'],900)
        self.assertEqual(grid['ig2ref'],0)
        self.assertEqual(grid['ig3ref'],43200)
        self.assertEqual(grid['ig4ref'],43200)
        self.assertEqual(grid['ni'],201)
        self.assertEqual(grid['nj'],100)
        self.assertEqual(grid['xg1'],0.)
        self.assertEqual(grid['xg2'],180.)
        self.assertEqual(grid['xg3'],0.)
        self.assertEqual(grid['xg4'],270.)
        self.assertEqual(grid['xlat1'],0.)
        self.assertEqual(grid['xlon1'],180.)
        self.assertEqual(grid['xlat2'],0.)
        self.assertEqual(grid['xlon2'],270.)
        self.assertEqual(grid['tag1'],2002)
        self.assertEqual(grid['tag2'],1000)
        self.assertEqual(grid['tag3'],0)
            
    def test_degGrid_L(self):
        params = rmn.defGrid_L(90,45,0.,180.,1.,0.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        params3 = rmn.decodeGrid(params)
        for k in params.keys():
            self.assertEqual(params[k],params3[k])
                    
    def test_degGrid_E(self):
        params = rmn.defGrid_E(90,45,0.,180.,1.,270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])

    def test_degGrid_ZL(self):
        params = rmn.defGrid_ZL(90,45,10.,11.,1.,0.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                self.assertFalse(ok)
            else:
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
        params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,0.,180.,1.,270.,lni=10,lnj=15,i0=12,j0=13)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            #Note: gdef_fmem ceash with #... faked as Z grid until fix in librmn
            if k not in ('i0','j0','lni','lnj','grtyp','ig3','ig4','lshape'):
                if isinstance(params[k],np.ndarray):
                    ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                    ## if ok: print k,params[k],params2[k]
                    self.assertFalse(ok)
                else:
                    ## if params[k] != params2[k]: print k,params[k],params2[k]
                    self.assertEqual(params[k],params2[k])
        params2 = rmn.decodeGrid(params)
        for k in params.keys():
            #Note: gdef_fmem ceash with #... faked as Z grid until fix in librmn
            if k not in ('i0','j0','lni','lnj','grtyp','lshape'):
                if isinstance(params[k],np.ndarray):
                    ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                    ## if ok: print k,params[k],params2[k]
                    self.assertFalse(ok)
                else:
                    ## if params[k] != params2[k]: print k,params[k],params2[k]
                    self.assertEqual(params[k],params2[k],'degGrid_diezeE: %s, expected: %s,  got: %s' % (k,str(params[k]),str(params2[k])))
           ## try:
           ##      i = type(params[k])
           ##      j = type(params2[k])
           ##  except:
           ##      print 'test_degGrid_dE',k,k in params.keys(),k in params2.keys() , params['grtyp'], params['grref'], params2['grtyp'], params2['grref']
    ##         if isinstance(params[k],np.ndarray):
    ##             ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
    ##             self.assertFalse(ok)
    ##         else:
    ##             self.assertEqual(params[k],params2[k],'p[%s] : %s != %s' % (k,str(params[k]),str(params2[k])))

    def test_degGrid_diezeL(self):
        params = rmn.defGrid_diezeL(90,45,11.,12.,1.,0.5,lni=90,lnj=45,i0=1,j0=1)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            #Note: gdef_fmem ceash with #... faked as Z grid until fix in librmn
            if k not in ('i0','j0','lni','lnj','grtyp','ig3','ig4','lshape'):
                if isinstance(params[k],np.ndarray):
                    ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                    ## if ok: print k,params[k],params2[k]
                    self.assertFalse(ok)
                else:
                    ## if params[k] != params2[k]: print k,params[k],params2[k]
                    self.assertEqual(params[k],params2[k])
           ## try:
           ##      i = type(params[k])
           ##      j = type(params2[k])
           ##  except:
           ##      print 'test_degGrid_dE',k,k in params.keys(),k in params2.keys() , params['grtyp'], params['grref'], params2['grtyp'], params2['grref']
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
            
    def test_yyg_yangrot(self):
        (xlat1,xlon1,xlat2,xlon2) = (0.,180.,0.,270.)
        (xlat1b,xlon1b,xlat2b,xlon2b) = rmn.yyg_yangrot_py(xlat1,xlon1,xlat2,xlon2)
        (xlat1c,xlon1c,xlat2c,xlon2c) = rmn.yyg_yangrot_py(xlat1b,xlon1b,xlat2b,xlon2b)
        self.assertEqual((xlat1,xlon1,xlat2,xlon2),(xlat1c,xlon1c,xlat2c,xlon2c))


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
