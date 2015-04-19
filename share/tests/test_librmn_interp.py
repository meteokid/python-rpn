#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import rpnpy.librmn.all as rmn
import unittest
## import ctypes as ct
import numpy as np

class Librmn_interp_Test(unittest.TestCase):

    epsilon = 0.0005
    
    def setIG_L(self,gp):
        ig1234 = rmn.cxgaig(gp['grtyp'],gp['lat0'],gp['lon0'],
                            gp['dlat'],gp['dlon'])
        gp['ig1'] = ig1234[0]
        gp['ig2'] = ig1234[1]
        gp['ig3'] = ig1234[2]
        gp['ig4'] = ig1234[3]
        return gp
        
    def getGridParams_L(self):
        gp = {
            'shape' : (90,180),
            'ni' : 90,
            'nj' : 180,
            'grtyp' : 'L',
            'dlat' : 0.5,
            'dlon' : 0.5,
            'lat0' : 45.,
            'lon0' : 273.
            }
        return self.setIG_L(gp)
    
    def getGridParams_L2x(self):
        gp = {
            'shape' : (45,90),
            'ni' : 45,
            'nj' : 90,
            'grtyp' : 'L',
            'dlat' : 0.5,
            'dlon' : 0.5,
            'lat0' : 45.,
            'lon0' : 273.25
            }
        return self.setIG_L(gp)

    def test_ezsetopt_ezgetopt(self):
        otplist = [
            (rmn.EZ_OPT_WEIGHT_NUMBER,2), #int
            (rmn.EZ_OPT_EXTRAP_VALUE,99.), #float
            (rmn.EZ_OPT_EXTRAP_DEGREE.lower(),rmn.EZ_EXTRAP_VALUE.lower()) #str
            ]
        for (o,v) in otplist:
            rmn.ezsetopt(o,v)
            v1 = rmn.ezgetopt(o,vtype=type(v))
            self.assertEqual(v1,v)
    
    def test_ezqkdef_ezgprm(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gprm = rmn.ezgprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k])
            
    def test_ezqkdef_ezgxprm(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp['ni'],gp['nj'],gp['grtyp'],
                           gp['ig1'],gp['ig2'],gp['ig3'],gp['ig4'])
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gp['grref'] = ' '
        gp['ig1ref'] = 0
        gp['ig2ref'] = 0
        gp['ig3ref'] = 0
        gp['ig4ref'] = 0
        gprm = rmn.ezgxprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k])
        rmn.gdrls(gid1)
        
    def test_ezqkdef_1subgrid(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        ng = rmn.ezget_nsubgrids(gid1)
        self.assertEqual(ng,1)
        subgid = rmn.ezget_subgridids(gid1)
        self.assertEqual(subgid,[gid1])
        rmn.gdrls(gid1)

    def test_ezqkdef_gdll(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        gll = rmn.gdll(gid1)
        self.assertEqual(gp['shape'],gll['lat'].shape)
        self.assertEqual(gp['shape'],gll['lon'].shape)
        self.assertEqual(gp['lat0'],gll['lat'][0,0])
        self.assertEqual(gp['lon0'],gll['lon'][0,0])
        rmn.gdrls(gid1)

    def test_ezsint(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        gp2 = self.getGridParams_L2x()
        gid2 = rmn.ezqkdef(gp2)
        self.assertTrue(gid2>=0)
        setid = rmn.ezdefset(gid2, gid1)
        self.assertTrue(setid>=0)
        zin = np.empty(gp1['shape'],dtype=np.float32,order='FORTRAN')
        for x in xrange(gp1['ni']):
            zin[:,x] = x
        zout = rmn.ezsint(gid2,gid1,zin)
        self.assertEqual(gp2['shape'],zout.shape)
        for j in xrange(gp2['nj']):
            for i in xrange(gp2['ni']):
                self.assertTrue(abs((zin[i,j]+zin[i+1,j])/2.-zout[i,j]) < self.epsilon)
        #rmn.gdrls([gid1,gid2]) #TODO: Makes the test crash

    def test_ezuvint(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        gp2 = self.getGridParams_L2x()
        gid2 = rmn.ezqkdef(gp2)
        self.assertTrue(gid2>=0)
        setid = rmn.ezdefset(gid2, gid1)
        self.assertTrue(setid>=0)
        uuin = np.empty(gp1['shape'],dtype=np.float32,order='FORTRAN')
        vvin = np.empty(gp1['shape'],dtype=np.float32,order='FORTRAN')
        for x in xrange(gp1['ni']):
            uuin[:,x] = x
        vvin = uuin*3.
        (uuout,vvout) = rmn.ezuvint(gid2,gid1,uuin,vvin)
        self.assertEqual(gp2['shape'],uuout.shape)
        self.assertEqual(gp2['shape'],vvout.shape)
        for j in xrange(gp2['nj']):
            for i in xrange(gp2['ni']):
                self.assertTrue(abs((uuin[i,j]+uuin[i+1,j])/2.-uuout[i,j]) < self.epsilon,'uvint, u: abs(%f-%f)=%f' % (((uuin[i,j]+uuin[i+1,j])/2),uuout[i,j],(uuin[i,j]+uuin[i+1,j])/2.-uuout[i,j]))

                self.assertTrue(abs((vvin[i,j]+vvin[i+1,j])/2.-vvout[i,j]) < self.epsilon,'uvint, v: abs(%f-%f)=%f' % (((vvin[i,j]+vvin[i+1,j])/2),vvout[i,j],(vvin[i,j]+vvin[i+1,j])/2.-vvout[i,j]))
                ## self.assertEqual((uuin[i,j]+uuin[i+1,j])/2.,uuout[i,j])
                ## self.assertEqual((vvin[i,j]+vvin[i+1,j])/2.,vvout[i,j])
        #rmn.gdrls([gid1,gid2]) #TODO: Makes the test crash

    ## la = np.array(
    ##     [[-89.5, -89. , -88.5],
    ##     [-89.5, -89. , -88.5],
    ##     [-89.5, -89. , -88.5]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    ## lo = np.array(
    ##     [[ 180. ,  180. ,  180. ],
    ##     [ 180.5,  180.5,  180.5],
    ##     [ 181. ,  181. ,  181. ]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    ## cla = np.array(
    ##     [[[-89.75, -89.25, -88.75],
    ##     [-89.75, -89.25, -88.75],
    ##     [-89.75, -89.25, -88.75]],
    ##     [[-89.25, -88.75, -88.25],
    ##     [-89.25, -88.75, -88.25],
    ##     [-89.25, -88.75, -88.25]],
    ##     [[-89.25, -88.75, -88.25],
    ##     [-89.25, -88.75, -88.25],
    ##     [-89.25, -88.75, -88.25]],
    ##     [[-89.75, -89.25, -88.75],
    ##     [-89.75, -89.25, -88.75],
    ##     [-89.75, -89.25, -88.75]]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    ## clo = np.array(
    ##     [[[ 179.75,  179.75,  179.75],
    ##     [ 180.25,  180.25,  180.25],
    ##     [ 180.75,  180.75,  180.75]],
    ##     [[ 179.75,  179.75,  179.75],
    ##     [ 180.25,  180.25,  180.25],
    ##     [ 180.75,  180.75,  180.75]],
    ##     [[ 180.25,  180.25,  180.25],
    ##     [ 180.75,  180.75,  180.75],
    ##     [ 181.25,  181.25,  181.25]],
    ##     [[ 180.25,  180.25,  180.25],
    ##     [ 180.75,  180.75,  180.75],
    ##     [ 181.25,  181.25,  181.25]]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')



##     def test_Librmn_ezgetlalo_KnownValues(self):
##         """Librmn_ezgetlalo should give known result with known input"""
##         (ni,nj) = self.la.shape
##         grtyp='L'
##         grref='L'
##         (ig1,ig2,ig3,ig4) =  rmn.cxgaig(grtyp,-89.5,180.0,0.5,0.5)
##         hasAxes = 0
##         doCorners = 0
##         (i0,j0) = (0,0)
##         (la2,lo2) = rmn.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(None,None),hasAxes,(i0,j0),doCorners)
##         if np.any(self.la!=la2):
##             print self.la
##             print la2
##         if np.any(self.lo!=lo2):
##             print self.lo
##             print lo2
##         self.assertFalse(np.any(self.la!=la2))
##         self.assertFalse(np.any(self.lo!=lo2))

##     def test_Librmn_ezgetlalo_KnownValues2(self):
##         """Librmn_ezgetlalo corners should give known result with known input"""
##         (ni,nj) = self.la.shape
##         grtyp='L'
##         grref='L'
##         (ig1,ig2,ig3,ig4) =  rmn.cxgaig(grtyp,-89.5,180.0,0.5,0.5)
##         hasAxes = 0
##         doCorners = 1
##         (i0,j0) = (0,0)
##         (la2,lo2,cla2,clo2) = rmn.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(None,None),hasAxes,(i0,j0),doCorners)
##         if np.any(self.la!=la2):
##             print self.la
##             print la2
##         if np.any(self.lo!=lo2):
##             print self.lo
##             print lo2
##         self.assertFalse(np.any(self.la!=la2))
##         self.assertFalse(np.any(self.lo!=lo2))
##         for ic in range(0,4):
##             if np.any(self.cla[ic,...]!=cla2[ic,...]):
##                 print ic,'cla'
##                 print self.cla[ic,...]
##                 print cla2[ic,...]
##             self.assertFalse(np.any(self.cla[ic,...]!=cla2[ic,...]))
##             if np.any(self.clo[ic,...]!=clo2[ic,...]):
##                 print ic,'clo'
##                 print self.clo[ic,...]
##                 print clo2[ic,...]
##             self.assertFalse(np.any(self.clo[ic,...]!=clo2[ic,...]))

##     def test_Librmn_ezgetlalo_Z_KnownValues(self):
##         """Librmn_ezgetlalo with Z grid should give known result with known input"""
##         (ni,nj) = self.la.shape
##         grtyp='Z'
##         grref='L'
##         (ig1,ig2,ig3,ig4) =  rmn.cxgaig(grref,0.,0.,1.,1.)
##         xaxis = self.lo[:,0].reshape((self.lo.shape[0],1)).copy('FORTRAN')
##         yaxis = self.la[0,:].reshape((1,self.la.shape[1])).copy('FORTRAN')
##         hasAxes = 1
##         doCorners = 0
##         (i0,j0) = (0,0)
##         (la2,lo2) = rmn.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xaxis,yaxis),hasAxes,(i0,j0),doCorners)
##         if np.any(self.la!=la2):
##             print self.la
##             print la2
##         if np.any(self.lo!=lo2):
##             print self.lo
##             print lo2
##         self.assertFalse(np.any(self.la!=la2))
##         self.assertFalse(np.any(self.lo!=lo2))

##     def test_Librmn_ezgetlalo_Dieze_KnownValues(self):
##         """Librmn_ezgetlalo with #-grid should give known result with known input"""
##         (ni,nj) = self.la.shape
##         grtyp='#'
##         grref='L'
##         (ig1,ig2,ig3,ig4) =  rmn.cxgaig(grref,0.,0.,1.,1.)
##         xaxis = self.lo[:,0].reshape((self.lo.shape[0],1)).copy('FORTRAN')
##         yaxis = self.la[0,:].reshape((1,self.la.shape[1])).copy('FORTRAN')
##         hasAxes = 1
##         doCorners = 0
##         (i0,j0) = (2,2)
##         (la2,lo2) = rmn.ezgetlalo((ni-1,nj-1),grtyp,(grref,ig1,ig2,ig3,ig4),(xaxis,yaxis),hasAxes,(i0,j0),doCorners)
##         if np.any(self.la[1:,1:]!=la2):
##             print self.la[1:,1:]
##             print la2
##         if np.any(self.lo[1:,1:]!=lo2):
##             print self.lo[1:,1:]
##             print lo2
##         self.assertFalse(np.any(self.la[1:,1:]!=la2))
##         self.assertFalse(np.any(self.lo[1:,1:]!=lo2))

## class LibrmnInterpTests(unittest.TestCase):

##     epsilon = 1.e-5

##     def gridL(self,dlalo=0.5,nij=10):
##         """provide grid and rec values for other tests"""
##         grtyp='L'
##         grref=grtyp
##         la0 = 0.-dlalo*(nij/2.)
##         lo0 = 180.-dlalo*(nij/2.)
##         ig14 = (ig1,ig2,ig3,ig4) =  rmn.cxgaig(grtyp,la0,lo0,dlalo,dlalo)
##         axes = (None,None)
##         hasAxes = 0
##         ij0 = (1,1)
##         doCorners = 0
##         (la,lo) = rmn.ezgetlalo((nij,nij),grtyp,(grref,ig1,ig2,ig3,ig4),axes,hasAxes,ij0,doCorners)
##         return (grtyp,ig14,(nij,nij),la,lo)

##     def test_Librmn_exinterp_KnownValues(self):
##         """Librmn_exinterp should give known result with known input"""
##         (g1_grtyp,g1_ig14,g1_shape,la1,lo1) = self.gridL(0.5,6)
##         (g2_grtyp,g2_ig14,g2_shape,la2,lo2) = self.gridL(0.25,8)
##         axes = (None,None)
##         ij0  = (1,1)
##         g1ig14 = list(g1_ig14)
##         g1ig14.insert(0,g1_grtyp)
##         g2ig14 = list(g2_ig14)
##         g2ig14.insert(0,g2_grtyp)
##         la2b = rmn.ezinterp(la1,None,
##             g1_shape,g1_grtyp,g1ig14,axes,0,ij0,
##             g2_shape,g2_grtyp,g2ig14,axes,0,ij0,
##             0)
##         if np.any(np.abs(la2-la2b)>self.epsilon):
##                 print 'g1:'+repr((g1_grtyp,g1_ig14,g1_shape))
##                 print 'g2:'+repr((g2_grtyp,g2_ig14,g2_shape))
##                 print 'la2:',la2
##                 print 'la2b:',la2b
##         self.assertFalse(np.any(np.abs(la2-la2b)>self.epsilon))


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
