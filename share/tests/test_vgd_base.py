#!/usr/bin/env python

import os
import datetime
import unittest
import ctypes as ct
import numpy as np
import rpnpy.vgd.all as vgd
import rpnpy.librmn.all as rmn

class VGDBaseTests(unittest.TestCase):

    def _newReadBcmk(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.vgd_read(fileId)
        rmn.fstcloseall(fileId)
        return vgd0ptr

    def testGetPutOptInt(self):
        v1 = vgd.vgd_get_opt('ALLOW_SIGMA')
        self.assertEqual(v1,vgd.VGD_DISALLOW_SIGMA)
        vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
        v2 = vgd.vgd_get_opt('ALLOW_SIGMA')
        self.assertEqual(v2,vgd.VGD_ALLOW_SIGMA)
        
    def testNewRead(self):
        vgd0ptr = self._newReadBcmk()
        self.assertEqual(vgd0ptr[0].kind,vgd.VGD_HYB_KIND)
        self.assertEqual(vgd0ptr[0].version,vgd.VGD_HYB_VER)

    def testNewReadGetInt(self):
        vgd0ptr = self._newReadBcmk()
       
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')
        self.assertEqual(vkind,vgd.VGD_HYB_KIND)
        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual(vvers,vgd.VGD_HYB_VER)

        try:
            scrap = vgd.vgd_get(vgd0ptr, 'SCRAP')
        except KeyError:
            pass
        except:
            self.assertEqual(0,1,'vgd_get of Unknown key should raise a KeyError')

    def testNewReadGetFloat(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'RC_1')
        self.assertEqual(int(v1*100),160)

        v2 = vgd.vgd_get(vgd0ptr, 'RC_2')
        self.assertEqual(int(v2),vgd.VGD_MISSING)

    def testNewReadGetDouble(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'PREF')
        self.assertEqual(int(v1*100.),8000000)
        
        v2 = vgd.vgd_get(vgd0ptr, 'PTOP')
        self.assertEqual(int(v2*100.),1000)

    def testNewReadGetChar(self):
        vgd0ptr = self._newReadBcmk()
       
        v1 = vgd.vgd_get(vgd0ptr, 'RFLD')
        self.assertEqual(v1.strip(),'P0')

    def testNewReadGetInt1D(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'VIPM')
        self.assertEqual(len(v1),158)
        self.assertEqual(v1[0:3],[97642568, 97690568, 97738568])

    def testNewReadGetFloat1D(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'VCDM')
        self.assertEqual(len(v1),158)
        self.assertEqual([int(x*10000000) for x in v1[0:3]],[1250, 1729, 2209])

    def testNewReadGetDouble1D(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'CA_M')
        self.assertEqual(len(v1),158)
        self.assertEqual(int(v1[0]*100.),1000)
        self.assertEqual(int(v1[1]*100.),1383)
        self.assertEqual(int(v1[2]*100.),1765)

    def testNewReadGetDouble3D(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = vgd.vgd_get(vgd0ptr, 'VTBL')
        self.assertEqual([int(x*100.) for x in v1[:,0:3,0].T.flatten()],
                         [500, 100, 300, 1000, 8000000, 160, 0, 0, 0])
        
    def testNewReadPutChar(self):
        vgd0ptr = self._newReadBcmk()

        v1 = 'PRES'
        vgd.vgd_put(vgd0ptr, 'RFLD', v1)
        v2 = vgd.vgd_get(vgd0ptr, 'RFLD')
        self.assertEqual(v2.strip(),v1)

    def testNewReadPutInt(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = 6
        vgd.vgd_put(vgd0ptr, 'IG_1', v1)
        v2 = vgd.vgd_get(vgd0ptr, 'IG_1')
        self.assertEqual(v1,v2)

    def testNewReadPutDouble(self):
        vgd0ptr = self._newReadBcmk()
        
        v1 = 70000.
        vgd.vgd_put(vgd0ptr, 'PREF', v1)
        v2 = vgd.vgd_get(vgd0ptr, 'PREF')
        self.assertEqual(int(v2*100.),int(v1*100.))

    def testFree(self):
        vgd0ptr = self._newReadBcmk()
        
        vgd.vgd_free(vgd0ptr)
        try:
            v1 = vgd.vgd_get(vgd0ptr, 'RFLD')
        except vgd.VGDError:
            pass
        except:
            self.assertEqual(0,1,'vgd_get of freed vgd should raise a VGDError')

    def testCmp(self):
        vgd0ptr = self._newReadBcmk()
        vgd1ptr = self._newReadBcmk()
        
        ok = vgd.vgd_cmp(vgd0ptr,vgd1ptr)
        self.assertTrue(ok)
        
        vgd.vgd_put(vgd0ptr, 'RFLD', 'PRES')
        ok = vgd.vgd_cmp(vgd0ptr,vgd1ptr)
        self.assertFalse(ok)

    fname = '__rpnstd__testfile__.fst'
    def _erase_testfile(self):
        try:
            os.unlink(self.fname)
        except:
            pass

    def testWriteDesc(self):
        vgd0ptr = self._newReadBcmk()

        self._erase_testfile()
        fileName = self.fname
        fileId = rmn.fstopenall(fileName, rmn.FST_RW)
        vgd.vgd_write(vgd0ptr,fileId)
        rmn.fstcloseall(fileId)

        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd1ptr = vgd.vgd_read(fileId)
        rmn.fstcloseall(fileId)
        self._erase_testfile()
        ok = vgd.vgd_cmp(vgd0ptr,vgd1ptr)
        self.assertTrue(ok)

    hyblist = (0.0134575, 0.0203980, 0.0333528, 0.0472815, 0.0605295, 0.0720790,
               0.0815451, 0.0889716, 0.0946203, 0.0990605, 0.1033873, 0.1081924,
               0.1135445, 0.1195212, 0.1262188, 0.1337473, 0.1422414, 0.1518590,
               0.1627942, 0.1752782, 0.1895965, 0.2058610, 0.2229843, 0.2409671,
               0.2598105, 0.2795097, 0.3000605, 0.3214531, 0.3436766, 0.3667171,
               0.3905587, 0.4151826, 0.4405679, 0.4666930, 0.4935319, 0.5210579,
               0.5492443, 0.5780612, 0.6074771, 0.6374610, 0.6679783, 0.6989974,
               0.7299818, 0.7591944, 0.7866292, 0.8123021, 0.8362498, 0.8585219,
               0.8791828, 0.8983018, 0.9159565, 0.9322280, 0.9471967, 0.9609448,
               0.9735557, 0.9851275, 0.9950425)
    MB2PA = 100.
  
    def testNewSigm(self):
        sigma = (0.011000, 0.027000, 0.051000, 0.075000, 0.101000, 0.127000,
                 0.155000, 0.185000, 0.219000, 0.258000, 0.302000, 0.351000,
                 0.405000, 0.460000, 0.516000, 0.574000, 0.631000, 0.688000,
                 0.744000, 0.796000, 0.842000, 0.884000, 0.922000, 0.955000,
                 0.980000, 0.993000, 1.000000)
        vgd0ptr = vgd.vgd_new_sigm(sigma)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['sigm'])
        
    def testNewPres(self):
        # pres = [x*self.MB2PA for x in (500.,850.,1000.)]
        pres = (500.,850.,1000.)
        vgd0ptr = vgd.vgd_new_pres(pres)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['pres'])

    def testNewEta(self):
        hyb = (0.000,   0.011,    0.027,    0.051,    0.075,
               0.101,   0.127,    0.155,    0.185,    0.219,
               0.258,   0.302,    0.351,    0.405,    0.460,
               0.516,   0.574,    0.631,    0.688,    0.744,
               0.796,   0.842,    0.884,    0.922,    0.955,
               0.980,   0.993,    1.000)
        ptop = 10. * self.MB2PA
        vgd0ptr = vgd.vgd_new_eta(hyb, ptop)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['eta'])

    ## def testNewHybN(self):
    ##     rcoef1 = 0.
    ##     ptop = 8.05 * self.MB2PA
    ##     pref = 1000. * self.MB2PA
    ##     vgd0ptr = vgd.vgd_new_hybn(self.hyblist, rcoef1, ptop, pref)
    ##     vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
    ##     vvers = vgd.vgd_get(vgd0ptr, 'VERS')
    ##     self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hybn'])

    def testNewHyb(self):
        hyb =(0.0125000,0.0233625,0.0391625,0.0628625,0.0865625,
              0.1122375,0.1379125,0.1655625,0.1951875,0.2287625,
              0.2672750,0.3107250,0.3591125,0.4124375,0.4667500,
              0.5220500,0.5793250,0.6356125,0.6919000,0.7472000,
              0.7985500,0.8439750,0.8854500,0.9229750,0.9555625,
              0.9802499,0.9930875,1.0000000)
        rcoef1 = 1.6
        ptop = 10. * self.MB2PA
        pref = 800. * self.MB2PA
        vgd0ptr = vgd.vgd_new_hyb(hyb, rcoef1, ptop, pref)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hyb'])

    def testNewHybS(self):
        rcoef1 = 0.
        rcoef2 = 10.
        ptop = 8.05 * self.MB2PA
        pref = 1000. * self.MB2PA
        vgd0ptr = vgd.vgd_new_hybs(self.hyblist, rcoef1, rcoef2, ptop, pref)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hybs'])

    def testNewHybT(self):
        rcoef1 = 0.
        rcoef2 = 10.
        ptop = 10. * self.MB2PA
        pref = 1000. * self.MB2PA
        vgd0ptr = vgd.vgd_new_hybt(self.hyblist, rcoef1, rcoef2, ptop, pref)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hybt'])

    def testNewHybM(self):
        rcoef1 = 0.
        rcoef2 = 1.
        pref = 1000. * self.MB2PA
        ptop = -1. # -2.
        vgd0ptr = vgd.vgd_new_hybm(self.hyblist, rcoef1, rcoef2, ptop, pref)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hybm'])

    def testNewHybMD(self):
        rcoef1 = 0.
        rcoef2 = 10.
        pref = 1000. * self.MB2PA
        dhm = 10.
        dht = 2.
        vgd0ptr = vgd.vgd_new_hybmd(self.hyblist, rcoef1, rcoef2, pref,
                                dhm, dht)
        vkind = vgd.vgd_get(vgd0ptr, 'KIND')        
        vvers = vgd.vgd_get(vgd0ptr, 'VERS')
        self.assertEqual((vkind,vvers),vgd.VGD_KIND_VER['hybmd'])



    ## def testNewBuildVert(self):
    ##     vgd0ptr = vgd.c_vgd_construct()
    ##     (kind, version) = (vgd.VGD_HYBS_KIND, vgd.VGD_HYBS_VER)
    ##     (ip1, ip2) = (0, 0)
    ##     ptop  = ct.c_double(805.)
    ##     pref  = ct.c_double(100000.)
    ##     (rcoef1, rcoef2) = (ct.c_float(1.), ct.c_float(10.))

    ##     ip1_m =(97618238, 96758972, 95798406, 94560550, 94831790, 95102940,
    ##             95299540, 93423264, 75597472)
    ##     nk = len(ip1_m) - 2 #why -2!!!
    ##     cip1_m = np.asarray(ip1_m, dtype=np.int32)

    ##     a_m_8 = (2.30926271551059, 5.66981194184163, 8.23745285281583,
    ##              9.84538165280926, 10.7362879740149, 11.1997204664634,
    ##              11.4378785724517, 11.51293, 11.5116748020711)
    ##     ca_m_8 = np.asarray(a_m_8, dtype=np.float64)
    ##     b_m_8 = (0., 1.154429569962798E-003, 0.157422392639441,
    ##              0.591052504380263, 0.856321652104870, 0.955780377300956,
    ##              0.991250207889939, 1., 1.)
    ##     cb_m_8 = np.asarray(b_m_8, dtype=np.float64)
    ##     ip1_t = (97698159, 96939212, 95939513, 94597899, 94877531,
    ##              95139482, 95323042, 93423264, 76746048)
    ##     cip1_t = np.asarray(ip1_t, dtype=np.int32)
    ##     a_t_8 = (2.89364884405945, 6.15320066567627, 8.55467550398551,
    ##              10.0259661797048, 10.8310952652232, 11.2484934057893,
    ##              11.4628969443959, 11.51293, 11.5126753323904)
    ##     ca_t_8 = np.asarray(a_t_8, dtype=np.float64)
    ##     b_t_8 = (5.767296480554498E-009, 7.010292926951782E-003,
    ##              0.227561997481228, 0.648350006620964, 0.878891216792279,
    ##              0.963738779730914, 0.994233214440677, 1. ,1.)
    ##     cb_t_8 = np.asarray(b_t_8, dtype=np.float64)
        
    ##     (nl_m, nl_t) = (len(a_m_8), len(a_t_8))
        
    ##     ok = vgd.c_vgd_new_build_vert(vgd0ptr,
    ##                                   kind, version,
    ##                                   nk, ip1, ip2, 
    ##                                   ct.byref(ptop),   ct.byref(pref),
    ##                                   ct.byref(rcoef1), ct.byref(rcoef2),
    ##                                   ca_m_8, cb_m_8, ca_t_8,
    ##                                   cb_t_8, cip1_m, cip1_t,
    ##                                   nl_m, nl_t)
    ##     self.assertEqual(ok,vgd.VGD_OK)
        
    ##     vkind = ct.c_int(0)
    ##     vvers = ct.c_int(0)
    ##     quiet = ct.c_int(0)
    ##     ok = vgd.c_vgd_get_int(vgd0ptr, 'KIND', ct.byref(vkind), quiet)
    ##     ok = vgd.c_vgd_get_int(vgd0ptr, 'VERS', ct.byref(vvers), quiet)
    ##     self.assertEqual(ok,vgd.VGD_OK)
    ##     self.assertEqual(vkind.value,vgd.VGD_HYBS_KIND)
    ##     self.assertEqual(vvers.value,vgd.VGD_HYBS_VER)

    def testNewFromTable(self):
        vgd0ptr = self._newReadBcmk()
        vgdtbl  = vgd.vgd_tolist(vgd0ptr)
        vgd1ptr = vgd.vgd_fromlist(vgdtbl)
        ok = vgd.vgd_cmp(vgd0ptr,vgd1ptr)
        self.assertTrue(ok)
        
    def testLevels_prof(self):
        vgd0ptr = self._newReadBcmk()
        MB2PA = 100.
        p0_stn_mb = 1013.  * MB2PA
        prof = vgd.vgd_levels(vgd0ptr, p0_stn_mb)
        self.assertEqual([int(x) for x in prof[0:5]*10000.],
                         [100000, 138426, 176879, 241410, 305984])
        self.assertEqual(len(prof.shape),1)
        self.assertEqual(prof.dtype,np.float32)

    def testLevels8_prof(self):
        vgd0ptr = self._newReadBcmk()
        MB2PA = 100.
        p0_stn_mb = 1013.  * MB2PA
        prof = vgd.vgd_levels(vgd0ptr, p0_stn_mb, double_precision=True)
        self.assertEqual([int(x) for x in prof[0:5]*10000.],
                         [100000, 138426, 176879, 241410, 305984])
        self.assertEqual(len(prof.shape),1)
        self.assertEqual(prof.dtype,np.float64)

    ## def testLevels_3d(self):
    def testDiag_withref_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.vgd_read(fileId)
        levels8 = vgd.vgd_levels(vgd0ptr, fileId)
        rmn.fstcloseall(fileId)
        (ni,nj,nk) = levels8.shape
        self.assertEqual([int(x) for x in levels8[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])
        self.assertEqual(len(levels8.shape),3)
        self.assertEqual(levels8.dtype,np.float32)

    ## def testLevels8_3d(self):
    def testDiag_withref8_3d(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES,'bcmk_toctoc','2009042700_000')
        fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        vgd0ptr = vgd.vgd_read(fileId)
        levels8 = vgd.vgd_levels(vgd0ptr, fileId, double_precision=True)
        rmn.fstcloseall(fileId)
        (ni,nj,nk) = levels8.shape
        self.assertEqual([int(x) for x in levels8[ni/2,nj/2,0:5]*10000.],
                         [100000, 138425, 176878, 241408, 305980])
        self.assertEqual(len(levels8.shape),3)
        self.assertEqual(levels8.dtype,np.float64)
        

if __name__ == "__main__":
    ## print vgd.VGD_LIBPATH
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
