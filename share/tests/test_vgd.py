#!/usr/bin/env python

import os
import datetime
import unittest
import ctypes as _ct
import numpy as np
import rpnpy.vgd.all as vgd
import rpnpy.librmn.all as rmn

C_MKSTR = _ct.create_string_buffer

class VGDReadTests(unittest.TestCase):

    def testConstruct(self):
        vgd0ptr = vgd.c_vgd_construct()
        self.assertEqual(vgd0ptr[0].rcoef1,-9999.)
        
    def testNewRead(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vgd0ptr[0].kind,vgd.VGD_HYB_KIND)
        self.assertEqual(vgd0ptr[0].version,vgd.VGD_HYB_VER)

    def testNewReadGetInt(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        vkind = _ct.c_int(0)
        vvers = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, 'KIND', _ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, 'VERS', _ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYB_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYB_VER)
        ok = vgd.c_vgd_get_int(vgd0ptr, 'SCRAP', _ct.byref(vkind), quiet)
        self.assertEqual(ok,vgd.VGD_ERROR)

    def testNewReadGetFloat(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        #print vgd0ptr[0].rcoef1,vgd0ptr[0].rcoef2
        v1 = _ct.c_float(0)
        v2 = _ct.c_float(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_float(vgd0ptr, 'RC_1', _ct.byref(v1), quiet)
        ok = vgd.c_vgd_get_float(vgd0ptr, 'RC_2', _ct.byref(v2), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(int(v1.value*100),160)
        self.assertEqual(int(v2.value),vgd.VGD_MISSING)

    def testNewReadGetDouble(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        #print vgd0ptr[0].pref_8,vgd0ptr[0].ptop_8
        v1 = _ct.c_double(0)
        v2 = _ct.c_double(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_double(vgd0ptr, 'PREF', _ct.byref(v1), quiet)
        ok = vgd.c_vgd_get_double(vgd0ptr, 'PTOP', _ct.byref(v2), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(int(v1.value*100.),8000000)
        self.assertEqual(int(v2.value*100.),1000)

    def testNewReadGetChar(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        #print vgd0ptr[0].ref_name
        v1 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', v1, quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(v1.value.strip(),'P0')

    def testNewReadGetInt1D(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].ip1_m[0]
        v1 = _ct.POINTER(_ct.c_int)()
        nv = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(v1), _ct.byref(nv), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(nv.value,158)
        self.assertEqual(v1[0:3],[97642568, 97690568, 97738568])

    def testNewReadGetFloat1D(self):
        self.assertEqual('MISSING_TEST: GetFloat1D','')

    def testNewReadGetDouble1D(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].a_m_8[0]
        v1 = _ct.POINTER(_ct.c_double)()
        nv = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_double_1d(vgd0ptr, 'CA_M', _ct.byref(v1), _ct.byref(nv), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(nv.value,158)
        self.assertEqual(int(v1[0]*100.),1000)
        self.assertEqual(int(v1[1]*100.),1383)
        self.assertEqual(int(v1[2]*100.),1765)

    def testNewReadGetDouble3D(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].a_m_8[0]
        v1 = _ct.POINTER(_ct.c_double)()
        ni = _ct.c_int(0)
        nj = _ct.c_int(0)
        nk = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_double_3d(vgd0ptr, 'VTBL', _ct.byref(v1), _ct.byref(ni), _ct.byref(nj), _ct.byref(nk), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x*100.) for x in v1[0:9]],
                         [500, 100, 300, 1000, 8000000, 160, 0, 0, 0])
        
    def testNewReadPutChar(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        v1 = C_MKSTR('PRES')
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_put_char(vgd0ptr, 'RFLD', v1)
        self.assertEqual(ok,vgd.VGD_OK)
        v2 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', v2, quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(v2.value.strip(),'PRES')

    def testNewReadPutInt(self):
        self.assertEqual('MISSING_TEST: PutInt','')
        ## ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        ## fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        ## fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        ## vgd0ptr = vgd.c_vgd_construct()
        ## ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        ## rmn.fstcloseall(fileId)
        ## v1 = _ct.c_int(6)
        ## quiet = _ct.c_int(0)
        ## ok = vgd.c_vgd_put_int(vgd0ptr, 'DIPM', v1)
        ## self.assertEqual(ok,vgd.VGD_OK)
        ## v2 = _ct.c_int(0)
        ## ok = vgd.c_vgd_get_int(vgd0ptr, 'DIPM', v2, quiet)
        ## self.assertEqual(ok,vgd.VGD_OK)
        ## self.assertEqual(v2.value,6)

    def testNewReadPutDouble(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        #print vgd0ptr[0].pref_8,vgd0ptr[0].ptop_8
        v1 = _ct.c_double(70000.)
        ok = vgd.c_vgd_put_double(vgd0ptr, 'PREF', v1)
        self.assertEqual(ok,vgd.VGD_OK)
        v2 = _ct.c_double(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_double(vgd0ptr, 'PREF', _ct.byref(v2), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(int(v2.value*100.),7000000)

    def testFree(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        vgd.c_vgd_free(vgd0ptr)
        v1 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', v1, quiet)
        self.assertEqual(ok,vgd.VGD_ERROR)

    def testCmp(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        vgd1ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd1ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
        self.assertEqual(ok,vgd.VGD_OK)
        v1 = C_MKSTR('PRES')
        ok = vgd.c_vgd_put_char(vgd0ptr, 'RFLD', v1)
        ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
        self.assertNotEqual(ok,vgd.VGD_OK)

    fname = '__rpnstd__testfile__.fst'
    def erase_testfile(self):
        try:
            os.unlink(self.fname)
        except:
            pass

    def testWriteDesc(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)

        self.erase_testfile()
        fileName = self.fname
        fileId = rmn.fstopenall(fileName, rmn.FST_RW)
        ok = vgd.c_vgd_write_desc(vgd0ptr,fileId)
        rmn.fstcloseall(fileId)
        self.assertEqual(ok,vgd.VGD_OK)

        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd1ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd1ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        self.erase_testfile()
        ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
        self.assertEqual(ok,vgd.VGD_OK)

    def testNewGen(self):
        self.assertEqual('MISSING_TEST: ','')
  ## real, dimension(57) :: hyb= &
  ##    (/0.0134575, 0.0203980, 0.0333528, 0.0472815, 0.0605295, 0.0720790, &
  ##      0.0815451, 0.0889716, 0.0946203, 0.0990605, 0.1033873, 0.1081924, &
  ##      0.1135445, 0.1195212, 0.1262188, 0.1337473, 0.1422414, 0.1518590, &
  ##      0.1627942, 0.1752782, 0.1895965, 0.2058610, 0.2229843, 0.2409671, &
  ##      0.2598105, 0.2795097, 0.3000605, 0.3214531, 0.3436766, 0.3667171, &
  ##      0.3905587, 0.4151826, 0.4405679, 0.4666930, 0.4935319, 0.5210579, &
  ##      0.5492443, 0.5780612, 0.6074771, 0.6374610, 0.6679783, 0.6989974, &
  ##      0.7299818, 0.7591944, 0.7866292, 0.8123021, 0.8362498, 0.8585219, &
  ##      0.8791828, 0.8983018, 0.9159565, 0.9322280, 0.9471967, 0.9609448, &
  ##      0.9735557, 0.9851275, 0.9950425/)
  ## real :: rcoef1=0.,rcoef2=1.
  ## real*8 :: ptop=805d0,pref=100000d0
  ## stat = vgd_new(vgd,kind=5,version=2,hyb=hyb,rcoef1=rcoef1,rcoef2=rcoef2,ptop_8=ptop,pref_8=pref)


    def testNewBuildVert(self):
        self.assertEqual('MISSING_TEST: ','')
       ##  type(vgrid_descriptor) :: vgd
       ##  integer, parameter :: nk=9 ! including diag level
       ##  integer :: stat,ip1
       ##  integer, dimension(:), pointer :: ip1_m,ip1_t
       ##  real*8, dimension(:), pointer :: a_m_8,b_m_8,a_t_8,b_t_8
       ##  real :: height=-1
       ##  logical :: OK=.true.
       ##  nullify(ip1_m,ip1_t,a_m_8,b_m_8,a_t_8,b_t_8)
       ##  allocate(ip1_m(nk),ip1_t(nk),a_m_8(nk),b_m_8(nk),a_t_8(nk),b_t_8(nk))
       ##  ip1_m=(/97618238,96758972,95798406,94560550,94831790,95102940,95299540,93423264,75597472/)
       ##  a_m_8=(/2.30926271551059,5.66981194184163,8.23745285281583,9.84538165280926,10.7362879740149,11.1997204664634,11.4378785724517,11.51293,11.5116748020711/)
       ##  b_m_8=(/0.000000000000000E+000,1.154429569962798E-003,0.157422392639441,0.591052504380263,0.856321652104870,0.955780377300956,0.991250207889939,1.00000000000000,1.00000000000000/)
       ##  ip1_t=(/97698159,96939212,95939513,94597899,94877531,95139482,95323042,93423264,76746048/)
       ##  a_t_8=(/2.89364884405945,6.15320066567627,8.55467550398551,10.0259661797048,10.8310952652232,11.2484934057893,11.4628969443959,11.51293,11.5126753323904/)
       ##  b_t_8=(/5.767296480554498E-009,7.010292926951782E-003,0.227561997481228,0.648350006620964,0.878891216792279,0.963738779730914,0.994233214440677,1.00000000000000,1.00000000000000/)
       ##  stat=vgd_new(vgd,kind=5,version=5,nk=nk-2,ip1=1,ip2=2,&
       ##       pref_8=100000.d0,&
       ##       rcoef1=1.,rcoef2=10.,&
       ##       a_m_8=a_m_8,b_m_8=b_m_8,&
       ##       a_t_8=a_t_8,b_t_8=b_t_8, &
       ## ip1_m=ip1_m,ip1_t=ip1_t)

    def testNewFromTable(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)

        v1 = _ct.POINTER(_ct.c_double)()
        ni = _ct.c_int(0)
        nj = _ct.c_int(0)
        nk = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_double_3d(vgd0ptr, 'VTBL', _ct.byref(v1), _ct.byref(ni), _ct.byref(nj), _ct.byref(nk), quiet)

        vgd1ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_from_table(vgd1ptr, v1, ni, nj, nk)
        self.assertEqual(ok,vgd.VGD_OK)
        ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
        self.assertEqual(ok,vgd.VGD_OK)
        
    def testLevels_prof(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)

        MB2PA = 100.
        p0_stn_mb = 1013.
        p0_stn = np.empty((1,), dtype=np.float32, order='FORTRAN')
        p0_stn[0] = p0_stn_mb * MB2PA

        prof = np.empty((nip1.value,), dtype=np.float32, order='FORTRAN')

        ni = 1 ; nj = 1 ; in_log = 0
        ok = vgd.c_vgd_levels(vgd0ptr, ni, nj, nip1, ip1list, prof, p0_stn, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in prof[0:5]*10000.],
                         [100000, 138426, 176879, 241410, 305984])


    def testLevels8_prof(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)

        MB2PA = 100.
        p0_stn_mb = 1013.
        p0_stn = np.empty((1,), dtype=np.float64, order='FORTRAN')
        p0_stn[0] = p0_stn_mb * MB2PA

        prof8 = np.empty((nip1.value,), dtype=np.float64, order='FORTRAN')

        ni = 1 ; nj = 1 ; in_log = 0
        ok = vgd.c_vgd_levels_8(vgd0ptr, ni, nj, nip1, ip1list, prof8, p0_stn, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in prof8[0:5]*10000.],
                         [100000, 138426, 176879, 241410, 305984])


    def testLevels_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=rfld_name.value.strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA
        
        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)
        
        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels = np.empty((ni, nj, nip1.value), dtype=np.float32, order='FORTRAN')
        ok = vgd.c_vgd_levels(vgd0ptr, ni, nj, nip1, ip1list, levels, rfld, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testLevels8_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=rfld_name.value.strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA
        
        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)
        
        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels8 = np.empty((ni, nj, nip1.value), dtype=np.float64, order='FORTRAN')
        rfld8 = np.empty((ni, nj), dtype=np.float64, order='FORTRAN')
        rfld8[:,:] = rfld[:,:]
        ok = vgd.c_vgd_levels_8(vgd0ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels8[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testDiag_withref_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=rfld_name.value.strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA
        
        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)
        
        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels = np.empty((ni, nj, nip1.value), dtype=np.float32, order='FORTRAN')
        ok = vgd.c_vgd_diag_withref(vgd0ptr, ni, nj, nip1, ip1list, levels, rfld, in_log, vgd.VGD_DIAG_DPIS);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testDiag_withref8_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, 'RFLD', rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=rfld_name.value.strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA
        
        rmn.fstcloseall(fileId)

        ip1list = _ct.POINTER(_ct.c_int)()
        nip1 = _ct.c_int(0)
        quiet = _ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, 'VIPM', _ct.byref(ip1list), _ct.byref(nip1), quiet)
        
        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels8 = np.empty((ni, nj, nip1.value), dtype=np.float64, order='FORTRAN')
        rfld8 = np.empty((ni, nj), dtype=np.float64, order='FORTRAN')
        rfld8[:,:] = rfld[:,:]
        ok = vgd.c_vgd_diag_withref_8(vgd0ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log, vgd.VGD_DIAG_DPIS);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels8[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])
        

if __name__ == "__main__":
    ## print vgd.VGD_LIBPATH
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
