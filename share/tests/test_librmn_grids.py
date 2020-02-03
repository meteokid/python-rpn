#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import os, os.path
import rpnpy.librmn.all as rmn
from rpnpy import range as _range
import unittest
## import ctypes as ct
import numpy as np

class Librmn_grids_Test(unittest.TestCase):

    epsilon = 0.0005
    fname = '__rpnstd__testfile__.fst'

    def erase_testfile(self):
        try:
            os.unlink(self.fname)
        except:
            pass

    def test_readGrid(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
        funit = rmn.fstopenall(myfile)
        rec   = rmn.fstlir(funit, nomvar='P0')
        grid  = rmn.readGrid(funit,rec)
        self.assertEqual(grid['grtyp'],'G')
        self.assertEqual(grid['ig1'],0)
        self.assertEqual(grid['ig2'],0)
        self.assertEqual(grid['ig3'],0)
        self.assertEqual(grid['ig4'],0)
        self.assertEqual(grid['xg1'],0.)
        self.assertEqual(grid['xg2'],0.)
        self.assertEqual(grid['xg3'],0.)
        self.assertEqual(grid['xg4'],0.)
        self.assertEqual(grid['ni'],200)
        self.assertEqual(grid['nj'],100)
        self.assertEqual(grid['shape'],(200,100))
        self.assertEqual(grid['glb'],True)
        self.assertEqual(grid['north'],True)
        self.assertEqual(grid['inverted'],False)

    def test_readGridRef(self):
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

    def test_writeGrid(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        file0  = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/geophy.fst')
        funit  = rmn.fstopenall(file0)
        rec    = rmn.fstlir(funit, nomvar='ME')
        grid0  = rmn.readGrid(funit, rec)
        rmn.fstcloseall(funit)
        grid1  = rmn.defGrid_L(180,60,0.,180.,1.,0.5)
        grid2  = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,0.,180.,1.,270.)
        grid3  = rmn.defGrid_YY(31,5,0.,180.,1.,270.)
        
        self.erase_testfile()
        myfile = self.fname
        funit  = rmn.fstopenall(myfile, rmn.FST_RW)
        rmn.fstecr(funit,rec['d'],rec)
        rmn.writeGrid(funit, grid0)
        rmn.writeGrid(funit, grid1)
        rmn.writeGrid(funit, grid2)
        rmn.writeGrid(funit, grid3)
        rmn.fstcloseall(funit)
        
        funit  = rmn.fstopenall(myfile, rmn.FST_RO)
        rec    = rmn.fstlir(funit, nomvar='ME')
        grid0b = rmn.readGrid(funit, rec)
        rmn.fstcloseall(funit)
        self.erase_testfile()
        for k in grid0.keys():
            if isinstance(grid0[k],np.ndarray):
                ok = np.any(np.abs(grid0b[k]-grid0[k]) > self.epsilon)
                self.assertFalse(ok, 'For k=%s, grid0b - grid0 = %s' % (k,str(np.abs(grid0b[k]-grid0[k]))))
            else:
                self.assertEqual(grid0b[k],grid0[k], 'For k=%s, expected:%s, got:%s' % (k, str(grid0[k]), str(grid0b[k])))
       

    def test_defGrid_L(self):
        params = rmn.defGrid_L(90,45,0.,180.,1.,0.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k], float):
                self.assertEqual(int((params[k]-params2[k])*1000.), 0, '{}: {} != {}'.format(k,params[k],params2[k]))
            elif k != 'id':
                self.assertEqual(params[k],params2[k], '{}: {} != {}'.format(k,params[k],params2[k]))
        params3 = rmn.decodeGrid(params)
        for k in params.keys():
            if isinstance(params[k], float):
                self.assertEqual(int((params[k]-params3[k])*1000.), 0, '{}: {} != {}'.format(k,params[k],params3[k]))
            elif k != 'id':
                self.assertEqual(params[k],params3[k], '{}: {} != {}'.format(k,params[k],params3[k]))
                    
    def test_defGrid_E(self):
        params = rmn.defGrid_E(90,45,0.,180.,1.,270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])

    def test_defGrid_ZL(self):
        params = rmn.defGrid_ZL(90,45,10.,11.,1.,0.5)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                ## print k,not ok,np.max(np.abs(params[k]-params2[k])),np.max(params[k])
                self.assertFalse(ok)
            else:
                ## print k,params[k] == params2[k],params[k],params2[k]
                self.assertEqual(params[k],params2[k])

                
    def test_defGrid_ZL2(self):
        params = rmn.defGrid_ZL(90,45,10.,11.,1.,0.5)
        params2 = rmn.decodeGrid(params)
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                ## print k,not ok,np.max(np.abs(params[k]-params2[k])),np.max(params[k])
                self.assertFalse(ok)
            else:
                ## print k,params[k] == params2[k],params[k],params2[k]
                self.assertEqual(params[k],params2[k])

    def test_defGrid_ZE(self):
        ## params = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,0.,180.,1.,270.)
        params = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,35.,230.,0.,320.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                #print k,not ok,np.max(np.abs(params[k]-params2[k])),np.min(params[k]),np.max(params[k])
                self.assertFalse(ok,(k,np.max(np.abs(params[k]-params2[k]))))
            elif isinstance(params[k],float):
                ok = abs(params[k]-params2[k]) > self.epsilon
                self.assertFalse(ok,(k,params[k]-params2[k]))
            else:
                #print k,params[k] == params2[k],params[k],params2[k]
                self.assertEqual(params[k],params2[k])

                
    def test_defGrid_ZE2(self):
        ## params = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,0.,180.,1.,270.)
        params = rmn.defGrid_ZE(90,45,10.,11.,1.,0.5,35.,230.,0.,320.)
        params2 = rmn.decodeGrid(params)
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                #print k,not ok,np.max(np.abs(params[k]-params2[k])),np.min(params[k]),np.max(params[k])
                self.assertFalse(ok,(k,np.max(np.abs(params[k]-params2[k]))))
            elif isinstance(params[k],float):
                ok = abs(params[k]-params2[k]) > self.epsilon
                self.assertFalse(ok,(k,params[k]-params2[k]))
            else:
                #print k,params[k] == params2[k],params[k],params2[k]
                self.assertEqual(params[k],params2[k])


    def test_defGrid_diezeE(self):
        ## params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,0.,180.,1.,270.,lni=10,lnj=15,i0=12,j0=13)
        params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,35.,230.,0.,320.,lni=10,lnj=15,i0=12,j0=13)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            #Note: gdef_fmem ceash with #... faked as Z grid until fix in librmn
            if k not in ('i0','j0','lni','lnj','grtyp','ig3','ig4','lshape'):
                if isinstance(params[k],np.ndarray):
                    ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                    ## if ok: print k,params[k],params2[k]
                    self.assertFalse(ok,(k,np.max(np.abs(params[k]-params2[k]))))
                elif isinstance(params[k],float):
                    ok = abs(params[k]-params2[k]) > self.epsilon
                    self.assertFalse(ok,(k,params[k]-params2[k]))
                else:
                    ## if params[k] != params2[k]: print k,params[k],params2[k]
                    self.assertEqual(params[k],params2[k])


    def test_defGrid_diezeE2(self):
        ## params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,0.,180.,1.,270.,lni=10,lnj=15,i0=12,j0=13)
        params = rmn.defGrid_diezeE(90,45,11.,12.,1.,0.5,35.,230.,0.,320.,lni=10,lnj=15,i0=12,j0=13)
        params2 = rmn.decodeGrid(params)
        for k in params.keys():
            #Note: gdef_fmem ceash with #... faked as Z grid until fix in librmn
            if k not in ('i0','j0','lni','lnj','grtyp','lshape'):
                if isinstance(params[k],np.ndarray):
                    ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                    ## if ok: print k,params[k],params2[k]
                    self.assertFalse(ok)
                elif isinstance(params[k],float):
                    ok = (abs(params[k]-params2[k]) > self.epsilon)
                    self.assertFalse(ok,'defGrid_diezeE: %s, expected: %s,  got: %s' % (k,str(params[k]),str(params2[k])))
                else:
                    ## if params[k] != params2[k]: print k,params[k],params2[k]
                    self.assertEqual(params[k],params2[k],'defGrid_diezeE: %s, expected: %s,  got: %s' % (k,str(params[k]),str(params2[k])))
           ## try:
           ##      i = type(params[k])
           ##      j = type(params2[k])
           ##  except:
           ##      print 'test_defGrid_dE',k,k in params.keys(),k in params2.keys() , params['grtyp'], params['grref'], params2['grtyp'], params2['grref']
    ##         if isinstance(params[k],np.ndarray):
    ##             ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
    ##             self.assertFalse(ok)
    ##         else:
    ##             self.assertEqual(params[k],params2[k],'p[%s] : %s != %s' % (k,str(params[k]),str(params2[k])))

    def test_defGrid_diezeL(self):
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
           ##      print 'test_defGrid_dE',k,k in params.keys(),k in params2.keys() , params['grtyp'], params['grref'], params2['grtyp'], params2['grref']
    ##         if isinstance(params[k],np.ndarray):
    ##             ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
    ##             self.assertFalse(ok)
    ##         else:
    ##             self.assertEqual(params[k],params2[k],'p[%s] : %s != %s' % (k,str(params[k]),str(params2[k])))

    def test_defGrid_YL(self):
        params0 = { \
            'ax'    : ( 45.,  46.5),\
            'ay'    : (273., 273. )\
            }
        params = rmn.defGrid_YL(params0)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                ## print k,not ok,np.max(np.abs(params[k]-params2[k])),np.max(params[k])
                self.assertFalse(ok, k)
            elif isinstance(params[k], float):
                ok = abs(params[k]-params2[k]) > self.epsilon
                self.assertFalse(ok, k)
            elif k != 'id':
                self.assertEqual(params[k],params2[k], k)

                
    def test_defGrid_YL2(self):
        params0 = { \
            'ax'    : ( 45.,  46.5),\
            'ay'    : (273., 273. )\
            }
        params = rmn.defGrid_YL(params0)
        params2 = rmn.decodeGrid(params)
        for k in params.keys():
            if isinstance(params[k],np.ndarray):
                ok = np.any(np.abs(params[k]-params2[k]) > self.epsilon)
                ## print k,not ok,np.max(np.abs(params[k]-params2[k])),np.max(params[k])
                self.assertFalse(ok, k)
            elif isinstance(params[k], float):
                ok = abs(params[k]-params2[k]) > self.epsilon
                self.assertFalse(ok, k)
            elif k != 'id':
                self.assertEqual(params[k],params2[k],k)

    def test_defGrid_G_glb(self):
        params = rmn.defGrid_G(90,45,glb=True,north=True,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_defGrid_G_glb_inv(self):
        params = rmn.defGrid_G(90,45,glb=True,north=True,inverted=True)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_defGrid_G_N(self):
        params = rmn.defGrid_G(90,45,glb=False,north=True,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_defGrid_G_S(self):
        params = rmn.defGrid_G(90,45,glb=False,north=False,inverted=False)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_defGrid_PS_N(self):
        params = rmn.defGrid_PS(90,45,north=True,pi=45,pj=30,d60=5000.,dgrw=270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])
        
    def test_defGrid_PS_S(self):
        params = rmn.defGrid_PS(90,45,north=False,pi=45,pj=30,d60=5000.,dgrw=270.)
        params2 = rmn.decodeGrid(params['id'])
        for k in params.keys():
            self.assertEqual(params[k],params2[k])


    def test_defGrid_YY(self):
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
                self.assertEqual(params[k],params2[k], 'For k=%s, expected:%s, got:%s' % (k, str(params[k]), str(params2[k])))
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
                    self.assertEqual(p0[k],p2[k],'sugridid=%d, For k=%s, expected:%s, got:%s' % (i, k, str(p0[k]), str(p2[k])))
            
    def test_yyg_yangrot(self):
        (xlat1,xlon1,xlat2,xlon2) = (0.,180.,0.,270.)
        (xlat1b,xlon1b,xlat2b,xlon2b) = rmn.yyg_yangrot_py(xlat1,xlon1,xlat2,xlon2)
        (xlat1c,xlon1c,xlat2c,xlon2c) = rmn.yyg_yangrot_py(xlat1b,xlon1b,xlat2b,xlon2b)
        self.assertEqual((xlat1,xlon1,xlat2,xlon2),(xlat1c,xlon1c,xlat2c,xlon2c))

    def test_ll2rll_norot(self):
        (xlat1, xlon1, xlat2, xlon2) = (0.,180.,0.,270.)
        ok = True
        for j in _range(178):
            for i in _range(358):
                (lat0,lon0) = (float(j+1-90), float(i+1))
                (rlat, rlon) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat0, lon0)
                (lat1, lon1) = rmn.egrid_rll2ll(xlat1, xlon1, xlat2, xlon2, rlat, rlon)
                (rlat1, rlon1) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat1, lon1)
                dlon  = abs(lon1-lon0)
                if dlon > 180.: dlon =- 360.
                drlon = abs(rlon1-rlon)
                if drlon > 180.: drlon =- 360.
                if False in (abs(lat1-lat0)<self.epsilon, dlon<self.epsilon,
                             abs(rlat1-rlat)<self.epsilon, drlon<self.epsilon):
                    print('n',i,j,abs(lat1-lat0), dlon, \
                        abs(rlat1-rlat), drlon)
                    ok = False
        self.assertTrue(ok)

                    
    def test_ll2rll_rot(self):
        from math import cos, radians
        (xlat1, xlon1, xlat2, xlon2) = (35.,230.,0.,320.)
        ok = True
        for j in _range(178):
            for i in _range(358):
                (lat0,lon0) = (float(j+1-90), float(i+1))
                (rlat, rlon) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat0, lon0)
                (lat1, lon1) = rmn.egrid_rll2ll(xlat1, xlon1, xlat2, xlon2, rlat, rlon)
                (rlat1, rlon1) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat1, lon1)
                dlon  = abs(lon1-lon0)
                if dlon > 180.: dlon =- 360.
                drlon = abs(rlon1-rlon)
                if drlon > 180.: drlon =- 360.
                if False in (abs(lat1-lat0)<self.epsilon, dlon*cos(radians(lat0))<self.epsilon,
                             abs(rlat1-rlat)<self.epsilon, drlon*cos(radians(rlat))<self.epsilon):
                    print('r',i,j,abs(lat1-lat0), dlon, \
                        abs(rlat1-rlat), drlon)
                    ok = False
        self.assertTrue(ok)

                    
    def test_ll2rll_rot2(self):
        from math import cos, radians
        (xlat1, xlon1, xlat2, xlon2) = (0.,180.,1.,270.)
        ok = True
        for j in _range(178):
            for i in _range(358):
                (lat0,lon0) = (float(j+1-90), float(i+1))
                (rlat, rlon) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat0, lon0)
                (lat1, lon1) = rmn.egrid_rll2ll(xlat1, xlon1, xlat2, xlon2, rlat, rlon)
                (rlat1, rlon1) = rmn.egrid_ll2rll(xlat1, xlon1, xlat2, xlon2, lat1, lon1)
                dlon  = abs(lon1-lon0)
                if dlon > 180.: dlon =- 360.
                drlon = abs(rlon1-rlon)
                if drlon > 180.: drlon =- 360.
                if False in (abs(lat1-lat0)<self.epsilon, dlon*cos(radians(lat0))<self.epsilon,
                             abs(rlat1-rlat)<self.epsilon, drlon*cos(radians(rlat))<self.epsilon):
                    print('r2',i,j,abs(lat1-lat0), dlon, \
                        abs(rlat1-rlat), drlon)
                    ok = False
        self.assertTrue(ok)

              
if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
