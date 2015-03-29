#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.fstd98"""

import os
import librmn.all as rmn
import unittest
## import ctypes as ct
import numpy as np

class Librmn_fstd98_Test(unittest.TestCase):

    ## lad = np.array(
    ##     [[-89.5, -89. , -88.5],
    ##     [-89.5, -89. , -88.5],
    ##     [-89.5, -89. , -88.5],
    ##     [-89.5, -89. , -88.5]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    ## lod = np.array(
    ##     [[ 180. ,  180. ,  180. ],
    ##     [ 180.5,  180.5,  180.5],
    ##     [ 181. ,  181. ,  181. ],
    ##     [ 181.5,  181.5,  181.5]]
    ##     ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    lad = np.array(
        [
            [1.1, 2.1 , 3.1, 4.1, 5.1],
            [1.2, 2.2 , 3.2, 4.2, 5.2],
            [1.3, 2.3 , 3.3, 4.3, 5.3],
            [1.4, 2.4 , 3.4, 4.4, 5.4]
        ]
        ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    lod = np.array(
        [
            [-1.1, -2.1 , -3.1],
            [-1.2, -2.2 , -3.2],
            [-1.3, -2.3 , -3.3],
            [-1.4, -2.4 , -3.4],
            [-1.5, -2.5 , -3.5]
        ]
        ,dtype=np.dtype('float32')).T #,order='FORTRAN')
    grtyp='L'
    xg14 = (-89.5,180.0,0.5,0.5)
    fname = '__rpnstd__testfile__.fst'
    la = None
    lo = None

    def erase_testfile(self):
        import os
        try:
            os.unlink(self.fname)
        except:
            pass

    def create_basefile(self):
        """create a basic test file for RPNFile tests"""
        self.erase_testfile()
        funit = rmn.fstopenall(self.fname,rmn.FST_RW)
        (ig1,ig2,ig3,ig4) = rmn.cxgaig(self.grtyp,self.xg14[0],self.xg14[1],self.xg14[2],self.xg14[3])
        self.la = rmn.FST_RDE_META_DEFAULT.copy()
        self.la.update(
            {'nomvar' : 'LA',
             'typvar' : 'C',
             'ni' : self.lad.shape[0],
             'nj' : self.lad.shape[1],
             'nk' : 1,
             'grtyp' : self.grtyp,
             'ig1' : ig1,
             'ig2' : ig2,
             'ig3' : ig3,
             'ig4' : ig4,
             'd'   : self.lad}
            )
        rmn.fstecr(funit,self.la['d'],self.la)
        self.lo = rmn.FST_RDE_META_DEFAULT.copy()
        self.lo.update(
            {'nomvar' : 'LO',
             'typvar' : 'C',
             'ni' : self.lod.shape[0],
             'nj' : self.lod.shape[1],
             'nk' : 1,
             'grtyp' : self.grtyp,
             'ig1' : ig1,
             'ig2' : ig2,
             'ig3' : ig3,
             'ig4' : ig4,
             'd'   : self.lod}
            )
        rmn.fstecr(funit,self.lo['d'],self.lo)
        rmn.fstcloseall(funit)
        return (self.la,self.lo)

    def test_fstinf_fstluk(self):
        """fstinf, fstluk should give known result with known input"""
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        a = rmn.isFST(self.fname)
        self.assertTrue(a)
        funit = rmn.fstopenall(self.fname,rmn.FST_RW)
        nrec = rmn.c_fstnbrv(funit)
        self.assertEqual(nrec,2,' c_fstnbrv found %d/2 rec ' % nrec)
        keylist = rmn.fstinl(funit)
        self.assertEqual(len(keylist),2,'fstinl found %d/2 rec: %s' % (len(keylist),repr(keylist)))
        kla = rmn.fstinf(funit,nomvar='LA')['key']
        la2 = rmn.fstluk(kla)#,rank=2)
        klo = rmn.fstinf(funit,nomvar='LO')['key']
        lo2 = rmn.fstluk(klo)#,rank=2)
        istat = rmn.fst_edit_dir(klo,nomvar='QW')        
        istat = rmn.fsteff(kla)
        rmn.fstcloseall(funit)

        self.assertEqual(la2['nomvar'].strip(),la['nomvar'].strip())
        self.assertEqual(lo2['nomvar'].strip(),lo['nomvar'].strip())
        
        self.assertEqual(la2['d'].shape,la['d'].shape)
        self.assertEqual(lo2['d'].shape,lo['d'].shape)

        epsilon = 0.05
        if np.any(np.fabs(la2['d'] - la['d']) > epsilon):
                print 'la2:',la2['d']
                print 'la :',la['d']
                print np.fabs(la2['d'] - la['d'])
        self.assertFalse(np.any(np.fabs(la2['d'] - la['d']) > epsilon))
        if np.any(np.fabs(lo2['d'] - lo['d']) > epsilon):
                print 'lo2:',lo2['d']
                print 'lo :',lo['d']
                print np.fabs(lo2['d'] - lo['d'])
        self.assertFalse(np.any(np.fabs(la2['d'] - la['d']) > epsilon))


        funit = rmn.fstopenall(self.fname,rmn.FST_RO)
        kla = rmn.fstinf(funit,nomvar='LA')
        self.assertEqual(kla,None,'LA found after delete: '+repr(kla))
        klo = rmn.fstinf(funit,nomvar='QW')['key']
        self.assertNotEqual(klo,None,'QW not found after rename: '+repr(klo))
        rmn.fstcloseall(funit)
        
        self.erase_testfile()
        

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
