#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import librmn.all as rmn
import unittest
## import ctypes as ct
## import numpy as np

class Librmn_interp_Test(unittest.TestCase):

    def test_interp(self):
        """Need to write some ezscint tests"""
        a = rmn.isFST('myfile')
        self.assertTrue(a,'Need to write some ezscint tests')
    

## class Librmn_ezgetlaloKnownValues(unittest.TestCase):

##     la = np.array(
##         [[-89.5, -89. , -88.5],
##         [-89.5, -89. , -88.5],
##         [-89.5, -89. , -88.5]]
##         ,dtype=np.dtype('float32'),order='FORTRAN')
##     lo = np.array(
##         [[ 180. ,  180. ,  180. ],
##         [ 180.5,  180.5,  180.5],
##         [ 181. ,  181. ,  181. ]]
##         ,dtype=np.dtype('float32'),order='FORTRAN')
##     cla = np.array(
##         [[[-89.75, -89.25, -88.75],
##         [-89.75, -89.25, -88.75],
##         [-89.75, -89.25, -88.75]],
##         [[-89.25, -88.75, -88.25],
##         [-89.25, -88.75, -88.25],
##         [-89.25, -88.75, -88.25]],
##         [[-89.25, -88.75, -88.25],
##         [-89.25, -88.75, -88.25],
##         [-89.25, -88.75, -88.25]],
##         [[-89.75, -89.25, -88.75],
##         [-89.75, -89.25, -88.75],
##         [-89.75, -89.25, -88.75]]]
##         ,dtype=np.dtype('float32'),order='FORTRAN')
##     clo = np.array(
##         [[[ 179.75,  179.75,  179.75],
##         [ 180.25,  180.25,  180.25],
##         [ 180.75,  180.75,  180.75]],
##         [[ 179.75,  179.75,  179.75],
##         [ 180.25,  180.25,  180.25],
##         [ 180.75,  180.75,  180.75]],
##         [[ 180.25,  180.25,  180.25],
##         [ 180.75,  180.75,  180.75],
##         [ 181.25,  181.25,  181.25]],
##         [[ 180.25,  180.25,  180.25],
##         [ 180.75,  180.75,  180.75],
##         [ 181.25,  181.25,  181.25]]]
##         ,dtype=np.dtype('float32'),order='FORTRAN')


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
