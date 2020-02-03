#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for librmn.base

See: https://wiki.cmc.ec.gc.ca/wiki/Exemples_d%27utilisation_des_programmes_BURP
"""

import os
import sys
import rpnpy.librmn.all as rmn
import rpnpy.utils.all as rutils
import unittest
import ctypes as _ct
import numpy as _np

if sys.version_info < (3, ):
    range = xrange
else:
    long = int

#--- primitives -----------------------------------------------------

#TODO: mrfopn return None
#TODO: mrfopn takes BURP_MODE_READ, BURP_MODE_CREATE, BURP_MODE_APPEND as mode
#TODO: replace fnom + mrfopn with burp_open, mrdcls bu burp_close

class RpnPyLibrmnBurp(unittest.TestCase):

    burptestfile = 'bcmk_burp/2007021900.brp'
    #(path, itype, iunit)
    knownValues = (
        (burptestfile, rmn.WKOFFIT_TYPE_LIST['BURP'], 999), 
        )

    def getFN(self, name):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        return os.path.join(ATM_MODEL_DFILES.strip(), name)
        

    def testmrfoptKnownValues(self):
        """mrfopt should give known result with known input"""
        for k in (rmn.BURPOP_MSG_TRIVIAL, rmn.BURPOP_MSG_INFO,
                  rmn.BURPOP_MSG_WARNING, rmn.BURPOP_MSG_ERROR,
                  rmn.BURPOP_MSG_FATAL, rmn.BURPOP_MSG_SYSTEM):
            optValue = rmn.mrfopt(rmn.BURPOP_MSGLVL, k)
            self.assertEqual(optValue[0:6], k[0:6])
            optValue = rmn.mrfopt(rmn.BURPOP_MSGLVL)
            self.assertEqual(optValue[0:5], k[0:5])

        optValue0 = 1.0000000150474662e+30
        optValue = rmn.mrfopt(rmn.BURPOP_MISSING)
        self.assertEqual(optValue, optValue0)
        
        optValue0 = 99.
        optValue = rmn.mrfopt(rmn.BURPOP_MISSING, optValue0)
        self.assertEqual(optValue, optValue0)
        optValue = rmn.mrfopt(rmn.BURPOP_MISSING)
        self.assertEqual(optValue, optValue0)

    def test_flags_decode(self):
        flgsdict = rmn.flags_decode(1024)
        self.assertEqual(flgsdict['flgs'], 1024)
        self.assertEqual(flgsdict['flgsl'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(flgsdict['flgsd'], 'data observed')

    def test_mrbtyp_decode(self):
        # From official btyp tool
        ## btyp 0     : 00:00:0:000000:0000 uni,data,sfc,observations (ADE), valeur observee
        ## btyp 6144  : 00:11:0:000000:0000 uni,mrqr,sfc,observations (ADE), valeur observee
        ## btyp 8192  : 01:00:0:000000:0000 multi,data,sfc,observations (ADE), valeur observee
        ## btyp 14336 : 01:11:0:000000:0000 multi,mrqr,sfc,observations (ADE), valeur observee
        btypdict = rmn.mrbtyp_decode(0)
        self.assertEqual(btypdict['bknat'], 0)
        self.assertEqual(btypdict['bknat_multi'], 0)
        self.assertEqual(btypdict['bknat_kind'], 0)
        self.assertEqual(btypdict['bknat_kindd'], 'data')
        self.assertEqual(btypdict['bktyp'], 0)
        self.assertEqual(btypdict['bktyp_alt'], 0)
        self.assertEqual(btypdict['bktyp_kind'], 0)
        self.assertEqual(btypdict['bktyp_kindd'], 'observations (ADE)')
        self.assertEqual(btypdict['bkstp'], 0)
        self.assertEqual(btypdict['bkstpd'], 'observed value')
        btypdict = rmn.mrbtyp_decode(6144)
        self.assertEqual(btypdict['bknat'], 3)
        self.assertEqual(btypdict['bknat_multi'], 0)
        self.assertEqual(btypdict['bknat_kind'], 3)
        self.assertEqual(btypdict['bktyp_kindd'], 'observations (ADE)')
        self.assertEqual(btypdict['bktyp'], 0)
        self.assertEqual(btypdict['bktyp_alt'], 0)
        self.assertEqual(btypdict['bktyp_kind'], 0)
        self.assertEqual(btypdict['bknat_kindd'], 'flags')
        self.assertEqual(btypdict['bkstp'], 0)
        self.assertEqual(btypdict['bkstpd'], 'observed value')
        btypdict = rmn.mrbtyp_decode(8192)
        self.assertEqual(btypdict['bknat'], 4)
        self.assertEqual(btypdict['bknat_multi'], 1)
        self.assertEqual(btypdict['bknat_kind'], 0)
        self.assertEqual(btypdict['bknat_kindd'], 'data')
        self.assertEqual(btypdict['bktyp'], 0)
        self.assertEqual(btypdict['bktyp_alt'], 0)
        self.assertEqual(btypdict['bktyp_kind'], 0)
        self.assertEqual(btypdict['bktyp_kindd'], 'observations (ADE)')
        self.assertEqual(btypdict['bkstp'], 0)
        self.assertEqual(btypdict['bkstpd'], 'observed value')
        btypdict = rmn.mrbtyp_decode(14336)
        self.assertEqual(btypdict['bknat'], 7)
        self.assertEqual(btypdict['bknat_multi'], 1)
        self.assertEqual(btypdict['bknat_kind'], 3)
        self.assertEqual(btypdict['bknat_kindd'], 'flags')
        self.assertEqual(btypdict['bktyp'], 0)
        self.assertEqual(btypdict['bktyp_alt'], 0)
        self.assertEqual(btypdict['bktyp_kind'], 0)
        self.assertEqual(btypdict['bktyp_kindd'], 'observations (ADE)')
        self.assertEqual(btypdict['bkstp'], 0)
        self.assertEqual(btypdict['bkstpd'], 'observed value')

    def testWkoffitKnownValues(self):
        """wkoffit should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            funit = rmn.wkoffit(self.getFN(mypath))
            self.assertEqual(funit, itype,
                             mypath+':'+repr(funit)+' != '+repr(itype))

    def testisburpKnownValues(self):
        """isburp should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            isburp = rmn.isBURP(self.getFN(mypath))
            self.assertTrue(isburp, 'isBRUP should return True for '+mypath)

    def testfnomfclosKnownValues(self):
        """fnom fclos should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            funit = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            rmn.fclos(funit)
            self.assertTrue(funit > 900 and funit <= 999,
                             mypath+':'+repr(funit)+' != '+repr(iunit))

    def testmrfnbrKnownValues(self):
        """mrfnbr mrfmxl mrfbfl should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            funit   = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            nbrp    = rmn.mrfnbr(funit)
            maxlen  = rmn.mrfmxl(funit)
            maxlen2 = rmn.mrfbfl(funit)
            ## https://wiki.cmc.ec.gc.ca/wiki/Probl%C3%A8me_avec_les_fonctions_de_manipulation_de_fichiers_BURP_dans_RMNLIB
            maxlen = max(64, maxlen)+10
            rmn.fclos(funit)
            self.assertEqual(nbrp, 47544)
            self.assertEqual(maxlen,  6208+10)
            self.assertEqual(maxlen2, 6208+10)

    def testmrfopnclsKnownValues(self):
        """mrfopn mrfcls should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            nbrp   = rmn.mrfopn(funit, rmn.BURP_MODE_READ)
            rmn.mrfcls(funit)
            rmn.fclos(funit)
            self.assertEqual(nbrp, 47544)

    def testburpopencloseKnownValues(self):
        """mrfopn mrfcls should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit = rmn.burp_open(self.getFN(mypath), rmn.BURP_MODE_READ)
            rmn.burp_close(funit)
            # self.assertEqual(nbrp, 47544)

    def testmrflocKnownValues(self):
        """mrfloc should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            self.assertNotEqual(handle, 0)
            (stnid, idtyp, lat, lon, date, time, sup) = \
                ('*********', -1, -1, -1, -1, -1, None)
            handle = 0
            nbrp2 = 0
            for irep in range(nbrp):
                handle = rmn.mrfloc(funit, handle, stnid, idtyp, lat, lon,
                                    date, time, sup)
                ## sys.stderr.write(repr(handle)+'\n')
                self.assertNotEqual(handle, 0)
                nbrp2 += 1
            handle = 0
            sup = []
            for irep in range(nbrp):
                handle = rmn.mrfloc(funit, handle, stnid, idtyp, lat, lon,
                                    date, time, sup)
                self.assertNotEqual(handle, 0)
            rmn.burp_close(funit)
            self.assertEqual(nbrp2, nbrp)

    def testmrfgetKnownValues(self):
        """mrfget should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit = rmn.burp_open(self.getFN(mypath))
            nbrp  = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            self.assertEqual(buf.size, 12416)
            #TODO: self.assertEqual(buf, ???)
            ## sys.stderr.write(repr(handle)+"("+repr(maxlen)+') rmn.mrfget None size='+repr(buf.size)+'\n')

            buf = maxlen
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            self.assertEqual(buf.size, 12436)
            #TODO: self.assertEqual(buf, ???)
            ## sys.stderr.write(repr(handle)+"("+repr(maxlen)+') rmn.mrfget maxlen size='+repr(buf.size)+'\n')

            buf = _np.empty((maxlen, ), dtype=_np.int32)
            buf[0] = maxlen
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            self.assertEqual(buf.size, 6218)
            #TODO: self.assertEqual(buf, ???)
            ## sys.stderr.write(repr(handle)+"("+repr(maxlen)+') rmn.mrfget empty size='+repr(buf.size)+'\n')

            handle = 0
            for irep in range(nbrp):
                handle = rmn.mrfloc(funit, handle)
                buf = rmn.mrfget(handle, buf, funit)
                ## print handle, buf.shape, buf[0:10]
                self.assertEqual(buf.size, 6218)
            
            rmn.burp_close(funit)

    def testmrbhdrKnownValues(self):
        """mrbhdr should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            ## for k,v in rmn.BURP_FLAGS_IDX_NAME.items():
            ##     print k, params['flgsl'][k], v
            params0 = {'datemm': 2, 'dy': 0.0, 'nxaux': 0,
                       'lat': 64.19999999999999, 'xaux': None,
                       'flgsl': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        'idtypd': 'TEMP + PILOT + SYNOP', 'lon': 276.63,
                        'nsup': 0, 'datedd': 19, 'timemm': 0, 'drnd': 0,
                        'flgs': 72706,
                        'flgsd': 'surface wind used, data observed, data derived, residues, TEMP part B',
                        'sup': None, 'nblk': 12, 'ilon': 27663, 'oars': 518,
                        'dx': 0.0, 'stnid': '71915    ', 'date': 20070219,
                        'ilat': 15420, 'ielev': 457, 'idx': 0, 'idy': 0,
                        'idtyp': 138, 'elev': 57.0, 'time': 0, 'dateyy': 2007,
                        'timehh': 0, 'runn': 8}
            for k in params.keys():
                self.assertEqual(params0[k], params[k],
                                 'For {0}, expected {1}, got {2}'
                                 .format(k, params0[k], params[k]))
            rmn.burp_close(funit)

    def testmrbprmKnownValues(self):
        """mrbprm should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkparams = rmn.mrbprm(buf, iblk+1)
            blkparams0 = {'datypd': 'uint', 'nele': 10, 'nbit': 20,
                          'datyp': 2, 'nval': 17,   'bdesc': 0, 'btyp': 9326,
                          'bfam': 10, 'nt': 1, 'bit0': 288,
                          'bktyp': 70, 'bkstp': 14, 'bknat': 4,
                          'bknat_multi': 1, 'bknat_kind': 0,
                          'bknat_kindd': 'data', 'bktyp_alt': 1,
                          'bktyp_kind': 6, 'bkno': iblk+1,
                          'bktyp_kindd': 'data seen by OA at altitude, global model',
                          'bkstpd': "statistiques d'erreur d'observation"}
            ## print 1,blkparams
            for k in blkparams.keys():
                self.assertEqual(blkparams0[k], blkparams[k],
                                 'For {0}, expected {1}, got {2}'
                                 .format(k, blkparams0[k], blkparams[k]))
            rmn.burp_close(funit)

    def testmrbtypKnownValues(self):
        """mrbtyp should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkparams = rmn.mrbprm(buf, iblk+1)
                blktypdict = rmn.mrbtyp_decode(blkparams['btyp'])
                btyp1 = rmn.mrbtyp_encode(blktypdict)
                btyp2 = rmn.mrbtyp_encode(blktypdict['bknat'], blktypdict['bktyp'], blktypdict['bkstp'])
                bknat = rmn.mrbtyp_encode_bknat(blkparams['bknat_multi'],
                                                blkparams['bknat_kind'])
                bktyp = rmn.mrbtyp_encode_bktyp(blkparams['bktyp_alt'],
                                                blkparams['bktyp_kind'])
                btyp3 = rmn.mrbtyp_encode(bknat, bktyp, blkparams['bkstp'])
                self.assertEqual(btyp1, blkparams['btyp'])
                self.assertEqual(btyp2, blkparams['btyp'])
                self.assertEqual(blkparams['bknat'], bknat)
                self.assertEqual(blkparams['bktyp'], bktyp)
                self.assertEqual(btyp3, blkparams['btyp'])

            rmn.burp_close(funit)

    ## def testmrbprm2KnownValues(self):
    ##     """mrbprm should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
    ##         funit  = rmn.burp_open(self.getFN(mypath))
    ##         nbrp   = rmn.mrfnbr(funit)
    ##         maxlen = max(64, rmn.mrfmxl(funit))+10

    ##         handle = 0
    ##         buf = None
    ##         for irep in range(nbrp):
    ##             handle = rmn.mrfloc(funit, handle)
    ##             buf = rmn.mrfget(handle, buf, funit)
    ##             rparams = rmn.mrbhdr(buf)
    ##             for iblk in range(rparams['nblk']):
    ##                 bparams = rmn.mrbprm(buf, iblk+1)
    ##                 ## print irep, handle, iblk+1, bparams['datyp']
    ##                 print bparams['datyp']
    ##         rmn.burp_close(funit)

    def testmrbxtrKnownValues(self):
        """mrbprm should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            ## blkdata = {
            ##     'cmcids' : None,
            ##     'tblval' : None
            ##     }
            for iblk in range(params['nblk']):
                blkparams = rmn.mrbprm(buf, iblk+1)
                ## blkdata   = rmn.mrbxtr(buf, iblk+1, blkdata['cmcids'], blkdata['tblval'])
                blkdata   = rmn.mrbxtr(buf, iblk+1)
                for k in blkparams.keys():
                    self.assertEqual(blkparams[k], blkdata[k],
                                     'For {0}, expected {1}, got {2}'
                                     .format(k, blkparams[k], blkdata[k]))
            lstele0 = _np.array([1796, 2817, 2818, 3073, 3264, 2754, 2049,
                                 2819, 2820, 3538], dtype=_np.int32)
            tblval0 = _np.array([10000, -1, -1, -1, -1, 405, -1, -1, -1,
                                 1029000], dtype=_np.int32)
            self.assertFalse(_np.any(lstele0 - blkdata['cmcids'] != 0))
            self.assertEqual((blkparams['nele'], blkparams['nval'],
                              blkparams['nt']), blkdata['tblval'].shape)
            self.assertFalse(_np.any(tblval0 -
                                     blkdata['tblval'][0:blkdata['nele'],0,0]
                                     != 0))
            rmn.burp_close(funit)

    def testmrbdclcovKnownValues(self):
        """mrbdcl/cov should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkparams  = rmn.mrbprm(buf, iblk+1)
                blkdata    = rmn.mrbxtr(buf, iblk+1)
                lstelebufr = rmn.mrbdcl(blkdata['cmcids'])
            rmn.burp_close(funit)
            lstelebufr0 = _np.array([7004, 11001, 11002, 12001, 12192, 10194,
                                     8001, 11003, 11004, 13210],
                                     dtype=_np.int32)
            self.assertFalse(_np.any(lstelebufr0 - lstelebufr != 0))
            lstelecmc = rmn.mrbcol(lstelebufr)
            self.assertFalse(_np.any(lstelecmc - blkdata['cmcids'] != 0))
            v1 = rmn.mrbdcl(blkdata['cmcids'][0])
            v2 = rmn.mrbcol(v1)
            self.assertEqual(lstelebufr0[0], v1)
            self.assertEqual(blkdata['cmcids'][0], v2)


    def testmrbcvtdictKnownValues(self):
        """mrbcvt_dict should give known result with known input"""
        d  = rmn.mrbcvt_dict(297)
        d0 = {
            'e_error'   : 0,
            'e_cmcid'   : 297,
            'e_bufrid'  : 1041,
            'e_bufrid_F': 0,
            'e_bufrid_X': 1,
            'e_bufrid_Y': 41,
            'e_cvt'     : 1,
            'e_desc'    : 'ABSOL. PLATFORM VELOCITY, FIRST COMPONENT',
            'e_units'   : 'M/S',
            'e_scale'   : 5,
            'e_bias'    : -1073741824,
            'e_nbits'   : 31,
            'e_multi'   : 0
            }
        ## print 'mrbcvt_dict',d
        for k in d.keys():
            self.assertEqual(d0[k], d[k],
                             'For {0}, expected {1}, got {2}'
                             .format(k, d0[k], d[k]))
        d  = rmn.mrbcvt_dict(2591)
        d0 = {
            'e_error'   : 0,
            'e_cmcid'   : 2591,
            'e_bufrid'  : 10031,
            'e_bufrid_F': 0,
            'e_bufrid_X': 10,
            'e_bufrid_Y': 31,
            'e_cvt'     : 1,
            'e_desc'    : "IN DIR. N. POLE, DIST. FM EARTH'S CENTRE",
            'e_units'   : 'M',
            'e_scale'   : 2,
            'e_bias'    : -1073741824,
            'e_nbits'   : 31,
            'e_multi'   : 0
            }
        ## print 'mrbcvt_dict',d
        for k in d.keys():
            self.assertEqual(d0[k], d[k],
                             'For {0}, expected {1}, got {2}'
                             .format(k, d0[k], d[k]))
        d  = rmn.mrbcvt_dict_bufr(10031)
        for k in d.keys():
            self.assertEqual(d0[k], d[k],
                             'For {0}, expected {1}, got {2}'
                             .format(k, d0[k], d[k]))
        bufridlist = rmn.mrbcvt_dict_find_id('.*ground\s+temperature.*')
        self.assertEqual(bufridlist[0], 12120)

    def testmrbcvtdecodeKnownValues(self):
        """mrbcvt should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkparams = rmn.mrbprm(buf, iblk+1)
                blkdata   = rmn.mrbxtr(buf, iblk+1)
                rval      = rmn.mrbcvt_decode(blkdata['cmcids'],
                                              blkdata['tblval'],
                                              blkparams['datyp'])
                rval      = rmn.mrbcvt_decode(blkdata,
                                              datyp=blkparams['datyp'])
                #TODO: check results
            rmn.burp_close(funit)

    def _fnmrbcvtencodeDecodeSanity(self, id):
        bufrid = _np.asfortranarray(id, dtype=_np.int32)
        cmcid0  = rmn.mrbcol(bufrid)
        values0 = [20., 10., 2., 1., 0., -1., -2., -10., -20.]
        cmcid = _np.asfortranarray(cmcid0, dtype=_np.int32)
        values = _np.reshape(_np.asfortranarray(
            values0, dtype=_np.float32),
            (1,len(values0),1), order='F')
        tblvals = rmn.mrbcvt_encode(cmcid, values.copy(order='F'))
        values1 = rmn.mrbcvt_decode(cmcid, tblvals.copy(order='F'))
        tblval1 = rmn.mrbcvt_encode(cmcid, values1.copy(order='F'))
        self.assertTrue(_np.all(tblval1 == tblvals))
        self.assertTrue(_np.all(values1 == values))

    def testmrbcvtencodeDecodeSanity(self):
        """mrbcvt encode/decode sycle should give back same result"""
        self._fnmrbcvtencodeDecodeSanity(2005)
        ## self._fnmrbcvtencodeDecodeSanity(7004)  # Not enough precision for number < 10.
        self._fnmrbcvtencodeDecodeSanity(8001)

    def testmrbcvtencodeKnownValues(self):
        """mrbcvt should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkparams = rmn.mrbprm(buf, iblk+1)
                blkdata   = rmn.mrbxtr(buf, iblk+1)
                blkdatatbl0= _np.array(blkdata['tblval'], copy=True)
                #Note: need to take a copy of blkdata['tblval'] for comparison
                #      after re-encoding since mrbcvt modify it for
                #      negative values
                rval      = rmn.mrbcvt_decode(blkdata['cmcids'],
                                              blkdata['tblval'],
                                              blkparams['datyp'])
                tblval    = rmn.mrbcvt_encode(blkdata['cmcids'], rval)
                for iele in range(blkdata['cmcids'].size):
                    e_cmcids = blkdata['cmcids'][iele]
                    e_cmcids = rmn.mrbcvt_dict_bufr(e_cmcids, raise_error=False)['e_bufrid']
                    e_rval = rval[iele,:,:]
                    e_tblval0 = blkdatatbl0[iele,:,:]
                    e_tblval1 = tblval[iele,:,:]
                    if not _np.all(e_tblval0 == e_tblval1):
                        ## print 'b1',repr(blkparams)
                        print('b2',repr(blkdata))
                        print('b3',repr(rmn.mrbcvt_dict(e_cmcids)))
                        print('b4',repr(rmn.mrbcvt_dict_bufr(e_cmcids, raise_error=False)))
                        print('rf',repr(e_rval.ravel()), blkparams['datyp'])
                        print('rv',repr(_np.round(e_rval).astype(_np.int32).ravel()), blkparams['datyp'])
                        print('t0',repr(e_tblval0.ravel()))
                        print('t1',repr(e_tblval1.ravel()))
                    self.assertTrue(_np.all(e_tblval0 == e_tblval1),
                                    "{}, {}: id={}, \nrval:{}, \nexp:{}, \ngot:{}"
                                    .format(iblk, iele, e_cmcids, 
                                            e_rval.ravel(), e_tblval0.ravel(),
                                            e_tblval1.ravel()))
            rmn.burp_close(funit)

    def testmrfvoiKnownValues(self):
        """mrfvoi should give known result with known input"""
        RPNPY_NOLONGTEST = os.getenv('RPNPY_NOLONGTEST', None)
        if RPNPY_NOLONGTEST:
            return
        for mypath, itype, iunit in self.knownValues:
            funit = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            rmn.mrfvoi(funit)
            rmn.fclos(funit)

    def testburpfilereadKnownValues(self):
        """mrbprm should give known result with known input"""
        RPNPY_NOLONGTEST = os.getenv('RPNPY_NOLONGTEST', None)
        if RPNPY_NOLONGTEST:
            return
        for mypath, itype, iunit in self.knownValues:
            bfile = rutils.BurpFile(self.getFN(mypath), 'r')
            #TODO: check results
            
    ## def testburpfileReadWrite(self):
    ##     """mrbprm should give known result with known input"""
    ##     RPNPY_NOLONGTEST = os.getenv('RPNPY_NOLONGTEST', None)
    ##     if RPNPY_NOLONGTEST:
    ##         return
    ##     for mypath, itype, iunit in self.knownValues:
    ##         bfile0 = rutils.BurpFile(self.getFN(mypath), 'r')
    ##         fnameo = '__rpnpy_burfile__testfile__.fst'
    ##         bfile1 = rutils.BurpFile(fnameo, 'w')
    ##         rutils.copy_burp(bfile0, bfile1)
    ##         bfile1.write_burpfile()
    ##         bfile2 = rutils.BurpFile(fnameo, 'r')
    ##         #TODO: compare bfile0 and bfile2
    ##         os.unlink(fnameo)

    def testmrbxtrdclcvtKnownValues(self):
        """mrbxtrdclcvt should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
            funit  = rmn.burp_open(self.getFN(mypath))
            nbrp   = rmn.mrfnbr(funit)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf    = rmn.mrfget(handle, buf, funit)
            params = rmn.mrbhdr(buf)
            for iblk in range(params['nblk']):
                blkdata = rmn.mrb_prm_xtr_dcl_cvt(buf, iblk+1)
                #TODO: check results
            rmn.burp_close(funit)


    def testmrbcvtdictsetKnownValues(self):
        """mrbcvt_dict_path_set should give known result with known input"""
        rpnpypath = os.getenv('rpnpy', None)
        mypath = os.path.join(rpnpypath, 'share', 'table_b_bufr_f')
        rmn.mrbcvt_dict_path_set(mypath)
        d  = rmn.mrbcvt_dict(297)
        rmn.mrbcvt_dict_path_set()  # Reset
        d0 = {
            'e_error'   : 0,
            'e_cmcid'   : 297,
            'e_bufrid'  : 1041,
            'e_bufrid_F': 0,
            'e_bufrid_X': 1,
            'e_bufrid_Y': 41,
            'e_cvt'     : 1,
            'e_desc'    : 'VITESSE ABS. PLATEFORME, 1ERE COMPOSANTE',
            'e_units'   : 'M/S',
            'e_scale'   : 5,
            'e_bias'    : -1073741824,
            'e_nbits'   : 31,
            'e_multi'   : 0
            }
        ## print 'mrbcvt_dict',d
        for k in d.keys():
            self.assertEqual(d0[k], d[k],
                             'For {0}, expected {1}, got {2}'
                             .format(k, d0[k], d[k]))


    def testmrbcvtdictseterror(self):
        """mrbcvt_dict_path_set should rais an error on non existing file"""
        hasError = True
        try:
            rmn.mrbcvt_dict_path_set('__no_such_file__')
            hasError = False
        except IOError:
            pass
        rmn.mrbcvt_dict_path_set()  # Reset
        self.assertEqual(hasError, True)


    def testmrbcvtdictraiseerror(self):
        rpnpypath = os.getenv('rpnpy', None)
        mypath = os.path.join(rpnpypath, 'share', 'table_b_bufr_e_err')
        rmn.mrbcvt_dict_path_set(mypath, raiseError=False)
        d  = rmn.mrbcvt_dict(297)
        d0 = {
            'e_error'   : 0,
            'e_cmcid'   : 297,
            'e_bufrid'  : 1041,
            'e_bufrid_F': 0,
            'e_bufrid_X': 1,
            'e_bufrid_Y': 41,
            'e_cvt'     : 1,
            'e_desc'    : 'ABSOL. PLATFORM VELOCITY, FIRST COMPONENT',
            'e_units'   : 'M/S',
            'e_scale'   : 5,
            'e_bias'    : -1073741824,
            'e_nbits'   : 31,
            'e_multi'   : 0
            }
        ## print 'mrbcvt_dict',d
        for k in d.keys():
            self.assertEqual(d0[k], d[k],
                             'For {0}, expected {1}, got {2}'
                             .format(k, d0[k], d[k]))
        rmn.mrbcvt_dict_path_set(mypath, raiseError=True)
        hasError = True
        try:
            d  = rmn.mrbcvt_dict(297)
            hasError = False
        except:
            pass
        rmn.mrbcvt_dict_path_set()  # Reset
        self.assertEqual(hasError, True)


    def testmrbcvtdictget(self):
        cvt = rmn.mrbcvt_dict_get()


    ## def testmrbxtrcvtKnownValues(self):
    ##     """fnomfclos should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         ier    = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
    ##         funit  = rmn.burp_open(self.getFN(mypath))
    ##         nbrp   = rmn.mrfnbr(funit)
    ##         maxlen = max(64, rmn.c_mrfmxl(funit))+10

    ##         (stnid, idtyp, lat, lon, date, time, nsup, nxaux) = \
    ##             ('*********', -1, -1, -1, -1, -1, 0, 0)
    ##         sup  = _np.empty((1, ), dtype=_np.int32)
    ##         xaux = _np.empty((1, ), dtype=_np.int32)
    ##         buf  = _np.empty((maxlen, ), dtype=_np.int32)
    ##         buf[0] = maxlen
    ##         handle = 0
            
    ##         itime = _ct.c_int(0)
    ##         iflgs = _ct.c_int(0)
    ##         stnids = ''
    ##         idburp = _ct.c_int(0)
    ##         ilat  = _ct.c_int(0)
    ##         ilon  = _ct.c_int(0)
    ##         idx   = _ct.c_int(0)
    ##         idy   = _ct.c_int(0)
    ##         ialt  = _ct.c_int(0)
    ##         idelay = _ct.c_int(0)
    ##         idate = _ct.c_int(0)
    ##         irs   = _ct.c_int(0)
    ##         irunn = _ct.c_int(0)
    ##         nblk  = _ct.c_int(0)

    ##         nele  = _ct.c_int(0)
    ##         nval  = _ct.c_int(0)
    ##         nt    = _ct.c_int(0)
    ##         bfam  = _ct.c_int(0)
    ##         bdesc = _ct.c_int(0)
    ##         btyp  = _ct.c_int(0)
    ##         nbit  = _ct.c_int(0)
    ##         bit0  = _ct.c_int(0)
    ##         datyp = _ct.c_int(0)

    ##         MRBCVT_DECODE = 0
    ##         MRBCVT_ENCODE = 1

    ##         for irep in range(nbrp):
    ##             handle = rmn.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, time, sup, nsup)
    ##             ier = rmn.c_mrfget(handle, buf)
    ##             ier = rmn.c_mrbhdr(buf, 
    ##                     itime, iflgs, stnids, idburp, 
    ##                     ilat, ilon, idx, idy, ialt, 
    ##                     idelay, idate, irs, irunn, nblk, 
    ##                     sup, nsup, xaux, nxaux)
    ##             ## print irep, handle, itime, iflgs, stnids, idburp, ilat, ilon, idx, idy, ialt, idelay, idate, irs, irunn, nblk

    ##             for iblk in range(nblk.value):
    ##                 ier = rmn.c_mrbprm(buf, iblk, 
    ##                              nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp)
    ##                 lstele = _np.empty((nele.value, ), dtype=_np.int32)
    ##                 tblval = _np.empty((nele.value, nval.value, nt.value), dtype=_np.int32)
    ##                 pval   = _np.empty((nele.value, nval.value, nt.value), dtype=_np.float32)
    ##                 ## nmax = nele.value*nval.value*nt.value#*2
    ##                 ## tblval = _np.empty((nmax, ), dtype=_np.int32)
    ##                 ier = rmn.c_mrbxtr(buf, iblk, lstele, tblval)
    ##                 codes = _np.empty((nele.value, ), dtype=_np.int32)
    ##                 ier = rmn.c_mrbdcl(lstele, codes, nele)
    ##                 ## print irep, iblk, ier, codes
    ##                 #self.assertEqual(ier, 0)

    ##                 if datyp.value in (2, 4):
    ##                     pval[:, :, :] = tblval[:, :, :]
    ##                     ier = rmn.c_mrbcvt(lstele, tblval, pval, nele, nval, nt, MRBCVT_DECODE)
    ##                 elif datyp.value == 6:
    ##                     pval[:, :, :] = tblval[:, :, :] #??transfer???
    ##                     ## pval(j)=transfer(tblval(j), z4val)
    ##                 else:
    ##                     pass #raise
    ##                 #print irep, iblk, ier, datyp.value, pval
    ##        rmn.burp_close(funit)

    ## def testmrfdelKnownValues(self):
    ##     """mrfdel should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         #TODO: Copy test file
    ##         rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.BURPOP_MSG_FATAL)
    ##         funit  = rmn.burp_open(self.getFN(mypath), rmn.BURP_MODE_APPEND)
    ##         handle = 0
    ##         handle = rmn.mrfloc(funit, handle)
    ##         rmn.mrfdel(handle)
    ##         #TODO: check if record is deleted
    ##         rmn.burp_close(funit)
    ##         #TODO: rm test file


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
