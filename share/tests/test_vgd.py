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
        self.assertEqual('MISSING_TEST: GetDouble3D','')

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

    def testNewBuildVert(self):
        self.assertEqual('MISSING_TEST: ','')
    def testNewTableShape(self):
        self.assertEqual('MISSING_TEST: ','')
    def testNewFromTable(self):
        self.assertEqual('MISSING_TEST: ','')
    def testWriteDesc(self):
        self.assertEqual('MISSING_TEST: ','')
    def testLevels(self):
        self.assertEqual('MISSING_TEST: ','')
    def testLevels8(self):
        self.assertEqual('MISSING_TEST: ','')
        

if __name__ == "__main__":
    print vgd.VGD_LIBPATH
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
