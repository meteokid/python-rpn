#!/usr/bin/env python

#from __future__ import division

import os
import datetime
import unittest
import ctypes as ct
import numpy as np
import rpnpy.vgd.all as vgd
import rpnpy.librmn.all as rmn

from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR

class VGDProtoTests(unittest.TestCase):

    def _newReadBcmk(self, vcode_name=None):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        if vcode_name is None:
            fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk_toctoc',
                                    '2009042700_000')
        else:
            fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk_vgrid',
                                    vcode_name.strip())
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except Exception as e:
            print(e)
            raise RuntimeError("Invalid file name: {}".format(fileName))
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        return vgd0ptr

    def testGetPutOptInt(self):
        quiet = ct.c_int(0)
        v1 = ct.c_int(0)
        ok = vgd.c_vgd_getopt_int(_C_WCHAR2CHAR('ALLOW_SIGMA'), ct.byref(v1), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(v1.value,vgd.VGD_DISALLOW_SIGMA)
        ok = vgd.c_vgd_putopt_int(_C_WCHAR2CHAR('ALLOW_SIGMA'), vgd.VGD_ALLOW_SIGMA)
        self.assertEqual(ok,vgd.VGD_OK)
        ok = vgd.c_vgd_getopt_int(_C_WCHAR2CHAR('ALLOW_SIGMA'), ct.byref(v1), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(v1.value,vgd.VGD_ALLOW_SIGMA)
        ok = vgd.c_vgd_putopt_int(_C_WCHAR2CHAR('ALLOW_SIGMA'), vgd.VGD_DISALLOW_SIGMA)

    ## def testConstruct(self):
    ##     vgd0ptr = vgd.c_vgd_construct()
    ##     self.assertEqual(vgd0ptr[0].rcoef1,-9999.)

    def testNewRead(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)
        rmn.fstcloseall(fileId)
        self.assertEqual(ok,vgd.VGD_OK)
        ## self.assertEqual(vgd0ptr[0].kind,vgd.VGD_HYB_KIND)
        ## self.assertEqual(vgd0ptr[0].version,vgd.VGD_HYB_VER)

    def testNewReadGetInt(self):
        vgd0ptr = self._newReadBcmk()
        vkind = ct.c_int(0)
        vvers = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('KIND'), ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('VERS'), ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYB_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYB_VER)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('SCRAP'), ct.byref(vkind), quiet)
        self.assertEqual(ok,vgd.VGD_ERROR)

    def testNewReadGetFloat(self):
        vgd0ptr = self._newReadBcmk()
        #print vgd0ptr[0].rcoef1,vgd0ptr[0].rcoef2
        v1 = ct.c_float(0)
        v2 = ct.c_float(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_float(vgd0ptr, _C_WCHAR2CHAR('RC_2'), ct.byref(v2), quiet)
        ok = vgd.c_vgd_get_float(vgd0ptr, _C_WCHAR2CHAR('RC_1'), ct.byref(v1), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(int(v1.value*100),160)
        self.assertEqual(int(v2.value),vgd.VGD_MISSING)

    def testNewReadGetDouble(self):
        vgd0ptr = self._newReadBcmk()
        #print vgd0ptr[0].pref_8,vgd0ptr[0].ptop_8
        v1 = ct.c_double(0)
        v2 = ct.c_double(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_double(vgd0ptr, _C_WCHAR2CHAR('PREF'), ct.byref(v1), quiet)
        ok = vgd.c_vgd_get_double(vgd0ptr, _C_WCHAR2CHAR('PTOP'), ct.byref(v2), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(int(v1.value*100.),8000000)
        self.assertEqual(int(v2.value*100.),1000)

    def testNewReadGetChar(self):
        vgd0ptr = self._newReadBcmk()
        #print vgd0ptr[0].ref_name
        v1 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), v1, quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(_C_CHAR2WCHAR(v1.value).strip(),'P0')

    def testNewReadGetInt1D(self):
        vgd0ptr = self._newReadBcmk()
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].ip1_m[0]
        v1 = ct.POINTER(ct.c_int)()
        nv = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(v1), ct.byref(nv), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(nv.value,158)
        self.assertEqual(v1[0:3],[97642568, 97690568, 97738568])

    def testNewReadGetFloat1D(self):
        vgd0ptr = self._newReadBcmk()
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].ip1_m[0]
        v1 = ct.POINTER(ct.c_float)()
        nv = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_float_1d(vgd0ptr, _C_WCHAR2CHAR('VCDM'), ct.byref(v1), ct.byref(nv), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(nv.value,158)
        self.assertEqual([int(x*10000000) for x in v1[0:3]],[1250, 1729, 2209])

    def testNewReadGetDouble1D(self):
        vgd0ptr = self._newReadBcmk()
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].a_m_8[0]
        v1 = ct.POINTER(ct.c_double)()
        nv = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_double_1d(vgd0ptr, _C_WCHAR2CHAR('CA_M'), ct.byref(v1), ct.byref(nv), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(nv.value,158)
        self.assertEqual(int(v1[0]*100.),1000)
        self.assertEqual(int(v1[1]*100.),1383)
        self.assertEqual(int(v1[2]*100.),1765)

    def testNewReadGetDouble3D(self):
        vgd0ptr = self._newReadBcmk()
        ## print vgd0ptr[0].nl_m, vgd0ptr[0].nl_t
        ## print vgd0ptr[0].a_m_8[0]
        v1 = ct.POINTER(ct.c_double)()
        ni = ct.c_int(0)
        nj = ct.c_int(0)
        nk = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_double_3d(vgd0ptr, _C_WCHAR2CHAR('VTBL'), ct.byref(v1), ct.byref(ni), ct.byref(nj), ct.byref(nk), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x*100.) for x in v1[0:9]],
                         [500, 100, 300, 1000, 8000000, 160, 0, 0, 0])

    def testNewReadPutChar(self):
        vgd0ptr = self._newReadBcmk()
        v1 = C_MKSTR('PRES')
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_put_char(vgd0ptr, _C_WCHAR2CHAR('ETIK'), v1)
        self.assertEqual(ok,vgd.VGD_OK)
        v2 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('ETIK'), v2, quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(_C_CHAR2WCHAR(v2.value).strip(),'PRES')

    def testNewReadPutInt(self):
        vgd0ptr = self._newReadBcmk()
        v1 = ct.c_int(6)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_put_int(vgd0ptr, _C_WCHAR2CHAR('IG_1'), v1)
        self.assertEqual(ok,vgd.VGD_OK)
        v2 = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('IG_1'), v2, quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(v1.value,v2.value)

    ## def testNewReadPutDouble(self): #removed from vgd 6.2.1
    ##     vgd0ptr = self._newReadBcmk()
    ##     #print vgd0ptr[0].pref_8,vgd0ptr[0].ptop_8
    ##     v1 = ct.c_double(70000.)
    ##     ok = vgd.c_vgd_put_double(vgd0ptr, _C_WCHAR2CHAR('PREF'), v1)
    ##     self.assertEqual(ok,vgd.VGD_OK)
    ##     v2 = ct.c_double(0)
    ##     quiet = ct.c_int(0)
    ##     ok = vgd.c_vgd_get_double(vgd0ptr, _C_WCHAR2CHAR('PREF'), ct.byref(v2), quiet)
    ##     self.assertEqual(ok,vgd.VGD_OK)
    ##     self.assertEqual(int(v2.value*100.),7000000)

    def testFree(self):
        vgd0ptr = self._newReadBcmk()
        vgd.c_vgd_free(vgd0ptr)
        v1 = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), v1, quiet)
        self.assertEqual(ok,vgd.VGD_ERROR)

    def testCmp(self):
        vgd0ptr = self._newReadBcmk()
        vgd1ptr = self._newReadBcmk()
        ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
        self.assertEqual(ok,vgd.VGD_OK)
        #TODO: find a way to change the vgd to make it different
    ##     v1 = C_MKSTR('PRES')
    ##     ok = vgd.c_vgd_put_char(vgd0ptr, _C_WCHAR2CHAR('ETIK'), v1)
    ##     v1 = ct.c_int(6)
    ##     ok = vgd.c_vgd_put_int(vgd0ptr, _C_WCHAR2CHAR('DIPT'), v1)
    ##     ok = vgd.c_vgd_vgdcmp(vgd0ptr,vgd1ptr)
    ##     self.assertNotEqual(ok,vgd.VGD_OK)

    fname = '__rpnstd__testfile__.fst'
    def erase_testfile(self):
        try:
            os.unlink(self.fname)
        except:
            pass

    def testWriteDesc(self):
        vgd0ptr = self._newReadBcmk()
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
        hyb = (0.0134575, 0.0203980, 0.0333528, 0.0472815, 0.0605295, 0.0720790,
               0.0815451, 0.0889716, 0.0946203, 0.0990605, 0.1033873, 0.1081924,
               0.1135445, 0.1195212, 0.1262188, 0.1337473, 0.1422414, 0.1518590,
               0.1627942, 0.1752782, 0.1895965, 0.2058610, 0.2229843, 0.2409671,
               0.2598105, 0.2795097, 0.3000605, 0.3214531, 0.3436766, 0.3667171,
               0.3905587, 0.4151826, 0.4405679, 0.4666930, 0.4935319, 0.5210579,
               0.5492443, 0.5780612, 0.6074771, 0.6374610, 0.6679783, 0.6989974,
               0.7299818, 0.7591944, 0.7866292, 0.8123021, 0.8362498, 0.8585219,
               0.8791828, 0.8983018, 0.9159565, 0.9322280, 0.9471967, 0.9609448,
               0.9735557, 0.9851275, 0.9950425)
        nhyb = len(hyb)
        chyb = np.asarray(hyb, dtype=np.float32)
        (rcoef1, rcoef2) = (ct.c_float(0.), ct.c_float(1.))
        ptop  = ct.c_double(805.)
        pref  = ct.c_double(100000.)
        p_ptop_out = ct.POINTER(ct.c_double)()
        (kind, version) = (vgd.VGD_HYBS_KIND, vgd.VGD_HYBS_VER)
        (ip1, ip2) = (0, 0)
        dhm = ct.c_float(10.)
        dht = ct.c_float(2.)
        (p_dhm, p_dht) = (None, None) #(ct.pointer(dhm), ct.pointer(dht))
        #TODO: why: (Cvgd) ERROR: dhm,dht is not a required constructor entry
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_gen(vgd0ptr,
                               kind, version,
                               chyb, nhyb,
                               ct.byref(rcoef1), ct.byref(rcoef2),
                               ct.byref(ptop),   ct.byref(pref),
                               p_ptop_out,
                               ip1, ip2, p_dhm, p_dht)
        self.assertEqual(ok,vgd.VGD_OK)

        vkind = ct.c_int(0)
        vvers = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('KIND'), ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('VERS'), ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYBS_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYBS_VER)

    def testNewGen2(self):
        hyb = (30968.,  24944., 20493., 16765., 13525., 10814.,  8026., 5477.,
               3488., 1842., 880., 0.)
        nhyb = len(hyb)
        chyb = np.asarray(hyb, dtype=np.float32)
        (rcoef1, rcoef2, rcoef3, rcoef4) = (ct.c_float(0.), ct.c_float(5.),
                                            ct.c_float(0.), ct.c_float(100.))
        p_ptop  = ct.POINTER(ct.c_double)()
        p_pref  = ct.POINTER(ct.c_double)()
        p_ptop_out = ct.POINTER(ct.c_double)()
        (kind, version) = (vgd.VGD_HYBHLS_KIND, vgd.VGD_HYBHLS_VER)
        (ip1, ip2, avg) = (0, 0, 0)
        dhm = ct.c_float(10.)
        dht = ct.c_float(2.)
        dhw = ct.c_float(10.)
        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_gen2(vgd0ptr,
                                kind, version,
                                chyb, nhyb,
                                ct.byref(rcoef1), ct.byref(rcoef2),
                                ct.byref(rcoef3), ct.byref(rcoef4),
                                p_ptop, p_pref, p_ptop_out,
                                ip1, ip2, ct.byref(dhm), ct.byref(dht),
                                ct.byref(dhw), avg)
        self.assertEqual(ok,vgd.VGD_OK)

        vkind = ct.c_int(0)
        vvers = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('KIND'), ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('VERS'), ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYBHLS_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYBHLS_VER)

    def testNewBuildVert(self):
        vgd0ptr = vgd.c_vgd_construct()
        (kind, version) = (vgd.VGD_HYBS_KIND, vgd.VGD_HYBS_VER)
        (ip1, ip2) = (0, 0)
        ptop  = ct.c_double(805.)
        pref  = ct.c_double(100000.)
        (rcoef1, rcoef2) = (ct.c_float(1.), ct.c_float(10.))

        ip1_m =(97618238, 96758972, 95798406, 94560550, 94831790, 95102940,
                95299540, 93423264, 75597472)
        nk = len(ip1_m) - 2  # why -2!!!
        cip1_m = np.asarray(ip1_m, dtype=np.int32)

        a_m_8 = (2.30926271551059, 5.66981194184163, 8.23745285281583,
                 9.84538165280926, 10.7362879740149, 11.1997204664634,
                 11.4378785724517, 11.51293, 11.5116748020711)
        ca_m_8 = np.asarray(a_m_8, dtype=np.float64)
        b_m_8 = (0., 1.154429569962798E-003, 0.157422392639441,
                 0.591052504380263, 0.856321652104870, 0.955780377300956,
                 0.991250207889939, 1., 1.)
        cb_m_8 = np.asarray(b_m_8, dtype=np.float64)
        ip1_t = (97698159, 96939212, 95939513, 94597899, 94877531,
                 95139482, 95323042, 93423264, 76746048)
        cip1_t = np.asarray(ip1_t, dtype=np.int32)
        a_t_8 = (2.89364884405945, 6.15320066567627, 8.55467550398551,
                 10.0259661797048, 10.8310952652232, 11.2484934057893,
                 11.4628969443959, 11.51293, 11.5126753323904)
        ca_t_8 = np.asarray(a_t_8, dtype=np.float64)
        b_t_8 = (5.767296480554498E-009, 7.010292926951782E-003,
                 0.227561997481228, 0.648350006620964, 0.878891216792279,
                 0.963738779730914, 0.994233214440677, 1. ,1.)
        cb_t_8 = np.asarray(b_t_8, dtype=np.float64)

        (nl_m, nl_t) = (len(a_m_8), len(a_t_8))

        ok = vgd.c_vgd_new_build_vert(vgd0ptr,
                                      kind, version,
                                      nk, ip1, ip2,
                                      ct.byref(ptop),   ct.byref(pref),
                                      ct.byref(rcoef1), ct.byref(rcoef2),
                                      ca_m_8, cb_m_8, ca_t_8,
                                      cb_t_8, cip1_m, cip1_t,
                                      nl_m, nl_t)
        self.assertEqual(ok,vgd.VGD_OK)

        vkind = ct.c_int(0)
        vvers = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('KIND'), ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('VERS'), ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYBS_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYBS_VER)

    def testNewBuildVert2(self):
        vgd0ptr = vgd.c_vgd_construct()
        (kind, version) = (vgd.VGD_HYBHLS_KIND, vgd.VGD_HYBHLS_VER)
        (ip1, ip2) = (0, 0)
        p_ptop = ct.POINTER(ct.c_double)()
        p_pref = ct.POINTER(ct.c_double)()
        p_ptop_out = ct.POINTER(ct.c_double)()
        (rcoef1, rcoef2) = (ct.c_float(0.), ct.c_float(1.))
        (rcoef3, rcoef4) = (ct.c_float(0.), ct.c_float(5.))

        ip1_m =(85095624, 85065817, 86890841, 86530977, 86332098, 86167510,
                87911659, 93423364, 75597472)
        nk = len(ip1_m) - 2  # why -2!!!
        cip1_m = np.asarray(ip1_m, dtype=np.int32)

        a_m_8 = (16096.822266, 13116.121094, 9076.089844, 5477.454102,
                 3488.660400, 1842.784424, 879.851318, 0.000000, 10.000000)
        ca_m_8 = np.asarray(a_m_8, dtype=np.float64)
        b_m_8 = (0.000000, 0.001038, 0.096399, 0.492782, 0.767428, 0.932772,
                 0.984755, 1.000000, 1.000000)
        cb_m_8 = np.asarray(b_m_8, dtype=np.float64)
        c_m_8 = (0.000000, 0.252011, 0.529947, 0.375240, 0.181007, 0.053405,
                 0.012177, 0.000000, 0.000000)
        cc_m_8 = np.asarray(c_m_8, dtype=np.float64)
        ip1_t = (85095624, 85065817, 86890841, 86530977, 86332098, 86167510,
                 87911659, 93423364, 76696048)
        cip1_t = np.asarray(ip1_t, dtype=np.int32)
        a_t_8 = (16096.822266, 13116.121094, 9076.089844, 5477.454102,
                 3488.660400, 1842.784424, 879.851318, 0.000000, 1.500000)
        ca_t_8 = np.asarray(a_t_8, dtype=np.float64)
        b_t_8 = (0.000000, 0.001038, 0.096399, 0.492782, 0.767428, 0.932772,
                 0.984755, 1.000000, 1.000000)
        cb_t_8 = np.asarray(b_t_8, dtype=np.float64)
        c_t_8 = (0.000000, 0.252011, 0.529947, 0.375240, 0.181007, 0.053405,
                 0.012177, 0.000000, 0.000000)
        cc_t_8 = np.asarray(c_t_8, dtype=np.float64)
        ip1_w = (85080721, 85045617, 86710909, 86431538, 86249804, 86119364,
                 93423364, 93423364, 82837504)
        cip1_w = np.asarray(ip1_w, dtype=np.int32)
        a_w_8 = (14606.471680, 11096.105469, 7276.771973, 4483.057251,
                 2665.722412, 1361.317871, 0.000000, 0.000000, 0.000000)
        ca_w_8 = np.asarray(a_w_8, dtype=np.float64)
        b_w_8 = (0.000519, 0.048718, 0.294591, 0.630105, 0.850100, 0.958764,
                 1.000000, 1.000000, 1.000000)
        cb_w_8 = np.asarray(b_w_8, dtype=np.float64)
        c_w_8 = (0.126005, 0.390979, 0.452594, 0.278124, 0.117206, 0.032791,
                 0.000000, 0.000000, 0.000000)
        cc_w_8 = np.asarray(c_w_8, dtype=np.float64)

        (nl_m, nl_t, nl_w) = (len(a_m_8), len(a_t_8), len(a_w_8))

        ok = vgd.c_vgd_new_build_vert2(vgd0ptr,
                                      kind, version,
                                      nk, ip1, ip2,
                                      p_ptop,   p_pref,
                                      ct.byref(rcoef1), ct.byref(rcoef2),
                                      ct.byref(rcoef3), ct.byref(rcoef4),
                                      ca_m_8, cb_m_8, cc_m_8,
                                      ca_t_8, cb_t_8, cc_t_8,
                                      ca_w_8, cb_w_8, cc_w_8,
                                      cip1_m, cip1_t, cip1_w,
                                      nl_m, nl_t, nl_w)
        self.assertEqual(ok,vgd.VGD_OK)

        vkind = ct.c_int(0)
        vvers = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('KIND'), ct.byref(vkind), quiet)
        ok = vgd.c_vgd_get_int(vgd0ptr, _C_WCHAR2CHAR('VERS'), ct.byref(vvers), quiet)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual(vkind.value,vgd.VGD_HYBHLS_KIND)
        self.assertEqual(vvers.value,vgd.VGD_HYBHLS_VER)

    def testNewFromTable(self):
        vgd0ptr = self._newReadBcmk()
        v1 = ct.POINTER(ct.c_double)()
        ni = ct.c_int(0)
        nj = ct.c_int(0)
        nk = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_double_3d(vgd0ptr, _C_WCHAR2CHAR('VTBL'), ct.byref(v1), ct.byref(ni), ct.byref(nj), ct.byref(nk), quiet)

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

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        MB2PA = 100.
        p0_stn_mb = 1013.
        p0_stn = np.empty((1,), dtype=np.float32, order='F')
        p0_stn[0] = p0_stn_mb * MB2PA

        prof = np.empty((nip1.value,), dtype=np.float32, order='F')

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

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        MB2PA = 100.
        p0_stn_mb = 1013.
        p0_stn = np.empty((1,), dtype=np.float64, order='F')
        p0_stn[0] = p0_stn_mb * MB2PA

        prof8 = np.empty((nip1.value,), dtype=np.float64, order='F')

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
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=_C_CHAR2WCHAR(rfld_name.value).strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA

        rmn.fstcloseall(fileId)

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        (ni, nj, in_log) = (rfld.shape[0], rfld.shape[1], 0)
        levels = np.empty((ni, nj, nip1.value), dtype=np.float32, order='F')
        ok = vgd.c_vgd_levels(vgd0ptr, ni, nj, nip1, ip1list, levels, rfld, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels[ni//2,nj//2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testLevels8_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)

        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=_C_CHAR2WCHAR(rfld_name.value).strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA

        rmn.fstcloseall(fileId)

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels8 = np.empty((ni, nj, nip1.value), dtype=np.float64, order='F')
        rfld8 = np.empty((ni, nj), dtype=np.float64, order='F')
        rfld8[:,:] = rfld[:,:]
        ok = vgd.c_vgd_levels_8(vgd0ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels8[ni//2,nj//2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testDiag_withref_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)

        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=_C_CHAR2WCHAR(rfld_name.value).strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA

        rmn.fstcloseall(fileId)

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        (ni,nj) = rfld.shape[0:2] ; in_log = 0
        levels = np.empty((ni, nj, nip1.value), dtype=np.float32, order='F')
        ok = vgd.c_vgd_diag_withref(vgd0ptr, ni, nj, nip1, ip1list, levels, rfld, in_log, vgd.VGD_DIAG_DPIS);
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels[ni//2,nj//2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])


    def testDiag_withref8_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)

        vgd0ptr = vgd.c_vgd_construct()
        ok = vgd.c_vgd_new_read(vgd0ptr,fileId,-1,-1,-1,-1)

        rfld_name = C_MKSTR(' '*vgd.VGD_MAXSTR_NOMVAR)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_char(vgd0ptr, _C_WCHAR2CHAR('RFLD'), rfld_name, quiet)

        rfld = rmn.fstlir(fileId, nomvar=_C_CHAR2WCHAR(rfld_name.value).strip())['d']
        MB2PA = 100.
        rfld = rfld * MB2PA

        rmn.fstcloseall(fileId)

        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'), ct.byref(ip1list), ct.byref(nip1), quiet)

        ni = rfld.shape[0] ; nj = rfld.shape[1] ; in_log = 0
        levels8 = np.empty((ni, nj, nip1.value), dtype=np.float64, order='F')
        rfld8 = np.empty((ni, nj), dtype=np.float64, order='F')
        rfld8[:,:] = rfld[:,:]
        ok = vgd.c_vgd_diag_withref_8(vgd0ptr, ni, nj, nip1, ip1list, levels8, rfld8, in_log, vgd.VGD_DIAG_DPIS)
        self.assertEqual(ok,vgd.VGD_OK)
        self.assertEqual([int(x) for x in levels8[ni//2,nj//2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])

    def testStda76_temp(self):
        vgd0ptr = self._newReadBcmk()
        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'),
                                  ct.byref(ip1list), ct.byref(nip1), quiet)
        temp = np.empty(nip1.value, dtype=np.float32, order='F')
        ok = vgd.c_vgd_stda76_temp(vgd0ptr, ip1list, nip1, temp)
        self.assertEqual(ok,vgd.VGD_OK)
        #TODO: assert temp values

    def testStda76_pres(self):
        vgd0ptr = self._newReadBcmk(vcode_name="21002_SLEVE")
        ip1list = ct.POINTER(ct.c_int)()
        nip1 = ct.c_int(0)
        quiet = ct.c_int(0)
        ok = vgd.c_vgd_get_int_1d(vgd0ptr, _C_WCHAR2CHAR('VIPM'),
                                  ct.byref(ip1list), ct.byref(nip1), quiet)
        pres = np.empty(nip1.value, dtype=np.float32, order='F')
        (p_sfc_temp, p_sfc_pres) = (None, None)
        ok = vgd.c_vgd_stda76_pres(vgd0ptr, ip1list, nip1, pres, p_sfc_temp,
                                   p_sfc_pres)
        self.assertEqual(ok,vgd.VGD_OK)
        #TODO: assert pres values
        sfc_temp = ct.c_float(273.15)
        ok = vgd.c_vgd_stda76_pres(vgd0ptr, ip1list, nip1, pres, sfc_temp,
                                   p_sfc_pres)
        self.assertEqual(ok,vgd.VGD_OK)
        sfc_pres = ct.c_float(100000.)
        ok = vgd.c_vgd_stda76_pres(vgd0ptr, ip1list, nip1, pres, p_sfc_temp,
                                   sfc_pres)
        self.assertEqual(ok,vgd.VGD_OK)
        #TODO: assert pres values

    def testStda76_hgts_from_pres_list(self):
        # Value obtained from vgrid test c_standard_atmosphere_hgts_from_pres
        pres = (105000., 95005.25, 5813.071777, 20104.253906, 10.)
        sol  = (-301.530579, 539.898010, 19620.611328, 11751.479492,
                64949.402344)
        cpres = np.asarray(pres, dtype=np.float32)
        csol = np.asarray(sol, dtype=np.float32)
        chgts = np.empty(cpres.size, dtype=np.float32)
        ok = vgd.c_vgd_stda76_hgts_from_pres_list(chgts, cpres, cpres.size)
        self.assertEqual(ok,vgd.VGD_OK)
        # TODO put a python loop
        self.assertAlmostEqual(chgts[0], csol[0], places=6, msg=None,
                               delta=None)
        self.assertAlmostEqual(chgts[1], csol[1], places=6, msg=None,
                               delta=None)
        self.assertAlmostEqual(chgts[2], csol[2], places=6, msg=None,
                               delta=None)
        self.assertAlmostEqual(chgts[3], csol[3], places=6, msg=None,
                               delta=None)
        self.assertAlmostEqual(chgts[4], csol[4], places=6, msg=None,
                               delta=None)

    def testStda76_pres_from_hgts_list(self):
        # Value obtained from vgrid test c_standard_atmosphere_hgts_from_pres
        sol  = (105000., 95005.25, 5813.071777, 20104.253906, 10.)
        hgts = (-301.530579, 539.898010, 19620.611328, 11751.479492,
                64949.402344)
        chgts = np.asarray(hgts, dtype=np.float32)
        csol = np.asarray(sol, dtype=np.float32)
        cpres = np.empty(chgts.size, dtype=np.float32)
        ok = vgd.c_vgd_stda76_pres_from_hgts_list(cpres, chgts, chgts.size)
        self.assertEqual(ok,vgd.VGD_OK)
        # TODO put a python loop
        self.assertAlmostEqual(cpres[0], csol[0], places=1, msg=None,
                               delta=None)
        self.assertAlmostEqual(cpres[1], csol[1], places=1, msg=None,
                               delta=None)
        self.assertAlmostEqual(cpres[2], csol[2], places=1, msg=None,
                               delta=None)
        self.assertAlmostEqual(cpres[3], csol[3], places=1, msg=None,
                               delta=None)
        self.assertAlmostEqual(cpres[4], csol[4], places=1, msg=None,
                               delta=None)

    def testPrint_desc(self):
        my_vgd = self._newReadBcmk(vcode_name="21002_SLEVE")
        sout = ct.c_int(-1)
        convip = ct.c_int(1)
        ok = vgd.c_vgd_print_desc(my_vgd, sout, convip)
        self.assertEqual(ok,vgd.VGD_OK)

if __name__ == "__main__":
    ## print vgd.VGD_LIBPATH
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
