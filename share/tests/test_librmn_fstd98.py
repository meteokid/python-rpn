#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.fstd98"""

import os
import rpnpy.librmn.all as rmn
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
    epsilon = 0.05

    def erase_testfile(self):
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


    def test_fstvoi_fstversion(self):
        """fstvoi, fst_version should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        funit = rmn.fstopenall(self.fname,rmn.FST_RW)
        ## ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        ## myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
        ## funit = rmn.fstopenall(myfile,rmn.FST_RO)
        rmn.fstvoi(funit)
        rmn.fstvoi(funit,'NINJNK+GRIDINFO')
        rmn.fstcloseall(funit)
        self.erase_testfile()

        a = rmn.fst_version()
        self.assertEqual(a,200001)

    def test_openall_closeall_loop(self):
        """Test if close all on linked file actually close them all"""
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        mydir = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/')
        for i in range(1000):
            funit = rmn.fstopenall(mydir)
            rmn.fstcloseall(funit)
            
    def test_openall_closeall_list(self):
        """Test if close all on linked file actually close them all"""
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        mydir1 = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/')
        mydir2 = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk_p/')
        funit1 = rmn.fstopenall(mydir1)
        funit2 = rmn.fstopenall(mydir2)
        rmn.fstcloseall((funit1,funit2))        

    def test_isfst_openall_fstnbr(self):
        """isfst_openall_fstnbr should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ## rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST,rmn.FSTOP_GET)

        HOME = os.getenv('HOME')
        a = rmn.isFST(os.path.join(HOME.strip(),'.profile'))
        self.assertFalse(a,'isFST should return false on non FST files')

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
        a = rmn.isFST(myfile)
        self.assertTrue(a,'isFST should return true on FST files')
    
        funit = rmn.fstopenall(myfile,rmn.FST_RO)
        self.assertTrue(funit>0,'fstopenall should return a valid file unit')

        nrec = rmn.c_fstnbrv(funit)
        self.assertEqual(nrec,1083,' c_fstnbrv found %d/1083 rec ' % nrec)
        
        nrec = rmn.fstnbrv(funit)
        self.assertEqual(nrec,1083,' fstnbrv found %d/1083 rec ' % nrec)
       
        rmn.fstcloseall(funit)


    def test_isfst_openall_dir_fstnbr(self):
        """isfst_openall__dir_fstnbr should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ## rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST,rmn.FSTOP_GET)

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        mydir = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/')    
        funit = rmn.fstopenall(mydir,rmn.FILE_MODE_RO)
        self.assertTrue(funit>0,'fstopenall should return a valid file unit')

        reclist = rmn.fstinl(funit,nomvar='VF')
        self.assertEqual(len(reclist),26,' fstinl of VF found %d/26 rec ' % len(reclist))

        reclist2 = rmn.fstinl(funit)
        self.assertEqual(len(reclist2),2788,' fstinl found %d/2788 rec ' % len(reclist))
        
        #Note: c_fstnbrv on linked files returns only nrec on the first file
        #      python's fstnbrv interface add results for all linked files
        nrec = rmn.fstnbrv(funit)
        self.assertEqual(nrec,2788,' fstnbrv found %d/2788 rec ' % nrec)

        #Note: c_fstnbr on linked files returns only nrec on the first file
        #      python's fstnbr interface add results for all linked files
        nrec = rmn.fstnbr(funit)
        self.assertEqual(nrec,2788,' fstnbr found %d/2788 rec ' % nrec)

        rmn.fstcloseall(funit)


    def test_fstsui_fstprm_fstlir(self):
        """fstsui_fstprm_fstlir should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
        funit = rmn.fstopenall(myfile,rmn.FST_RO)

        k = rmn.fstinf(funit)
        a = rmn.fstprm(k)
        self.assertEqual(a['nomvar'].strip(),'P0','fstinf/fstprm wrong rec, Got %s expected P0' % (a['nomvar']))
        k = rmn.fstsui(funit)['key']
        a = rmn.fstprm(k)
        self.assertEqual(a['nomvar'].strip(),'TT','fstsui/fstprm wrong rec, Got %s expected TT' % (a['nomvar']))

        k = rmn.fstinf(funit,nomvar='MX')['key']
        a = rmn.fstlir(funit)
        self.assertEqual(a['nomvar'].strip(),'P0','fstlir wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),530)
        self.assertEqual(int(np.amax(a['d'])),1039)
  
        k = rmn.fstinf(funit,nomvar='MX')['key']
        a = rmn.fstlirx(k,funit)
        self.assertEqual(a['nomvar'].strip(),'LA','fstlirx wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),-88)
        self.assertEqual(int(np.amax(a['d'])),88)

        a = rmn.fstlis(funit)
        self.assertEqual(a['nomvar'].strip(),'LO','fstlirx wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),-180)
        self.assertEqual(int(np.amax(a['d'])),178)

        rmn.fstcloseall(funit)

    def test_fstlir_fstlirx_fstlir_witharray(self):
        """fstlir_fstlirx_fstlir_witharray should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
        funit = rmn.fstopenall(myfile,rmn.FST_RO)

        k = rmn.fstinf(funit)['key']
        a = rmn.fstprm(k)
        self.assertEqual(a['nomvar'].strip(),'P0','fstinf/fstprm wrong rec, Got %s expected P0' % (a['nomvar']))
        k = rmn.fstsui(funit)['key']
        a = rmn.fstprm(k)
        self.assertEqual(a['nomvar'].strip(),'TT','fstsui/fstprm wrong rec, Got %s expected TT' % (a['nomvar']))

        k = rmn.fstinf(funit,nomvar='MX')['key']
        a = rmn.fstlir(funit)
        a = rmn.fstlir(funit,dataArray=a['d'])
        self.assertEqual(a['nomvar'].strip(),'P0','fstlir wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),530)
        self.assertEqual(int(np.amax(a['d'])),1039)
  
        k = rmn.fstinf(funit,nomvar='MX')['key']
        a = rmn.fstlirx(k,funit,dataArray=a['d'])
        self.assertEqual(a['nomvar'].strip(),'LA','fstlirx wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),-88)
        self.assertEqual(int(np.amax(a['d'])),88)

        a = rmn.fstlis(funit,dataArray=a['d'])
        self.assertEqual(a['nomvar'].strip(),'LO','fstlis wrong rec, Got %s expected P0' % (a['nomvar']))
        self.assertEqual(int(np.amin(a['d'])),-180)
        self.assertEqual(int(np.amax(a['d'])),178)

        rmn.fstcloseall(funit)

    def test_fstecr_fstinf_fstluk(self):
        """fstinf, fstluk should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        a = rmn.isFST(self.fname)
        self.assertTrue(a)
        funit = rmn.fstopenall(self.fname,rmn.FST_RW)
        nrec = rmn.c_fstnbrv(funit)
        keylist = rmn.fstinl(funit)
        kla = rmn.fstinf(funit,nomvar='LA')['key']
        la2prm = rmn.fstprm(kla)#,rank=2)
        la2 = rmn.fstluk(kla)#,rank=2)
        klo = rmn.fstinf(funit,nomvar='LO')
        lo2 = rmn.fstluk(klo)#,rank=2)
        rmn.fstcloseall(funit)
        self.erase_testfile()

        self.assertEqual(nrec,2,' c_fstnbrv found %d/2 rec ' % nrec)
        self.assertEqual(len(keylist),2,'fstinl found %d/2 rec: %s' % (len(keylist),repr(keylist)))

        self.assertEqual(la2['nomvar'].strip(),la['nomvar'].strip())
        self.assertEqual(lo2['nomvar'].strip(),lo['nomvar'].strip())
        
        self.assertEqual(la2['d'].shape,la['d'].shape)
        self.assertEqual(lo2['d'].shape,lo['d'].shape)

        if np.any(np.fabs(la2['d'] - la['d']) > self.epsilon):
                print('la2:',la2['d'])
                print('la :',la['d'])
                print(np.fabs(la2['d'] - la['d']))
        self.assertFalse(np.any(np.fabs(la2['d'] - la['d']) > self.epsilon))
        if np.any(np.fabs(lo2['d'] - lo['d']) > self.epsilon):
                print('lo2:',lo2['d'])
                print('lo :',lo['d'])
                print(np.fabs(lo2['d'] - lo['d']))
        self.assertFalse(np.any(np.fabs(la2['d'] - la['d']) > self.epsilon))


    def test_fstecr_fstluk_order(self):
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        fname = '__rpnstd__testfile2__.fst'
        try:
            os.unlink(fname)
        except:
            pass
        funit = rmn.fstopenall(fname,rmn.FST_RW)
        (ig1,ig2,ig3,ig4) = rmn.cxgaig(self.grtyp,self.xg14[0],self.xg14[1],self.xg14[2],self.xg14[3])
        (ni,nj) = (90,45)
        la = rmn.FST_RDE_META_DEFAULT.copy()
        la.update(
            {'nomvar' : 'LA',
             'typvar' : 'C',
             'ni' : ni,
             'nj' : nj,
             'nk' : 1,
             'grtyp' : self.grtyp,
             'ig1' : ig1,
             'ig2' : ig2,
             'ig3' : ig3,
             'ig4' : ig4
             }
            )
        lo = la.copy()
        lo['nomvar'] = 'LO'
        #Note: For the order to be ok in the FSTD file, order='FORTRAN' is mandatory
        la['d'] = np.empty((ni,nj),dtype=np.float32,order='FORTRAN')
        lo['d'] = np.empty((ni,nj),dtype=np.float32,order='FORTRAN')
        for j in range(nj):
            for i in range(ni):
                lo['d'][i,j] = 100.+float(i)        
                la['d'][i,j] = float(j)
        rmn.fstecr(funit,la['d'],la)
        rmn.fstecr(funit,lo)
        rmn.fstcloseall(funit)
        funit = rmn.fstopenall(fname,rmn.FST_RW)
        kla = rmn.fstinf(funit,nomvar='LA')['key']
        la2 = rmn.fstluk(kla)#,rank=2)
        klo = rmn.fstinf(funit,nomvar='LO')['key']
        lo2 = rmn.fstluk(klo)#,rank=2)
        rmn.fstcloseall(funit)
        try:
            os.unlink(fname)
        except:
            pass
        self.assertTrue(np.isfortran(la2['d']))
        self.assertTrue(np.isfortran(lo2['d']))
        self.assertFalse(np.any(np.fabs(la2['d'] - la['d']) > self.epsilon))
        self.assertFalse(np.any(np.fabs(lo2['d'] - lo['d']) > self.epsilon))


    def test_fsteditdir_fsteff(self):
        """fst_edit_dir should give known result with known input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        funit = rmn.fstopenall(self.fname,rmn.FST_RW)
        kla = rmn.fstinf(funit,nomvar='LA')['key']
        klo = rmn.fstinf(funit,nomvar='LO')['key']
        istat = rmn.fst_edit_dir(klo,nomvar='QW')        
        istat = rmn.fsteff(kla)
        rmn.fstcloseall(funit)

        funit = rmn.fstopenall(self.fname,rmn.FST_RO)
        kla = rmn.fstinf(funit,nomvar='LA')
        klo = rmn.fstinf(funit,nomvar='QW')['key']
        rmn.fstcloseall(funit)
        self.erase_testfile()
        self.assertEqual(kla,None,'LA found after delete: '+repr(kla))
        self.assertNotEqual(klo,None,'QW not found after rename: '+repr(klo))

        
    def test_fsteditdir_list_rec(self):
        """fst_edit_dir accept list and dict as input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        funit   = rmn.fstopenall(self.fname,rmn.FST_RW)
        keylist = rmn.fstinl(funit)
        istat   = rmn.fst_edit_dir(keylist, etiket='MY_NEW_ETK')
        klo     = rmn.fstinf(funit,nomvar='LO')
        istat   = rmn.fst_edit_dir(klo, nomvar='QW')
        rmn.fstcloseall(funit)

        funit = rmn.fstopenall(self.fname,rmn.FST_RO)
        la = rmn.fstlir(funit,nomvar='LA')
        lo = rmn.fstlir(funit,nomvar='QW')
        rmn.fstcloseall(funit)
        self.erase_testfile()
        self.assertNotEqual(lo,None,'QW not found after rename: '+repr(klo))
        self.assertNotEqual(la['etiket'],'MY_NEW_ETK')
        self.assertNotEqual(lo['etiket'],'MY_NEW_ETK')


    def test_fsteff_list_rec(self):
        """fsteff accept list and dict as input"""
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        funit   = rmn.fstopenall(self.fname,rmn.FST_RW)
        keylist = rmn.fstinl(funit)
        rmn.fsteff(keylist)
        rmn.fstcloseall(funit)
        funit = rmn.fstopenall(self.fname,rmn.FST_RO)
        kla = rmn.fstinf(funit,nomvar='LA')
        klo = rmn.fstinf(funit,nomvar='LO')
        rmn.fstcloseall(funit)
        self.erase_testfile()
        self.assertEqual(kla,None,'LA found after delete: '+repr(kla))
        self.assertEqual(klo,None,'LO found after delete: '+repr(klo))

        (la,lo) = self.create_basefile() #wrote 2 recs in that order: la, lo
        funit   = rmn.fstopenall(self.fname,rmn.FST_RW)
        klo     = rmn.fstinf(funit,nomvar='LO')
        rmn.fsteff(klo)
        rmn.fstcloseall(funit)
        funit = rmn.fstopenall(self.fname,rmn.FST_RO)
        klo = rmn.fstinf(funit,nomvar='LO')
        rmn.fstcloseall(funit)
        self.erase_testfile()
        self.assertEqual(klo,None,'LO found after delete: '+repr(klo))


    ## def test_fstluk_f16_datyp134(self):
    ##     """fstluk of f16 fields (datyp=134) should give known result with known input"""
    ##     self.assertEqual(0,1,'Need to update test with a new FST file')
    ##     rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    ##     CMCGRIDF = os.getenv('CMCGRIDF')
    ##     myfile = os.path.join(CMCGRIDF.strip(),'prog','gsloce','2015070706_042')
    ##     funit = rmn.fstopenall(myfile,rmn.FST_RO)
    ##     k = rmn.fstinf(funit, nomvar='UUW', typvar='P@')
    ##     r = rmn.fstluk(k['key'])
    ##     self.assertEqual(r['nomvar'],'UUW ')
    ##     rmn.fstcloseall(funit)


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
