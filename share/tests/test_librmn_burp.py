#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for librmn.base

See: https://wiki.cmc.ec.gc.ca/wiki/Exemples_d%27utilisation_des_programmes_BURP
"""

import os
import sys
import rpnpy.librmn.all as rmn
import unittest
import ctypes as _ct
import numpy as _np

if sys.version_info > (3, ):
    long = int

#--- primitives -----------------------------------------------------

class RpnPyLibrmnBurp(unittest.TestCase):

    burptestfile = 'bcmk_burp/2007021900.brp'
    #(path, itype, iunit)
    knownValues = (
        (burptestfile, rmn.WKOFFIT_TYPE_LIST['BURP'], 999), 
        )

    def getFN(self, name):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        return os.path.join(ATM_MODEL_DFILES.strip(), name)
        

    def testWkoffitKnownValues(self):
        """wkoffit should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            funit = rmn.wkoffit(self.getFN(mypath))
            self.assertEqual(funit, itype, mypath+':'+repr(funit)+' != '+repr(itype))

    def testfnomfclosKnownValues(self):
        """fnom fclos should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            funit = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            rmn.fclos(funit)
            self.assertEqual(funit, iunit, mypath+':'+repr(funit)+' != '+repr(iunit))

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
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
            funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            nbrp   = rmn.mrfopn(funit, rmn.FST_RO)
            rmn.mrfcls(funit)
            rmn.fclos(funit)
            self.assertEqual(nbrp, 47544)

    def testmrflocKnownValues(self):
        """mrfloc should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
            funit = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            nbrp  = rmn.mrfopn(funit, rmn.FST_RO)
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            self.assertNotEqual(handle, 0)
            (stnid, idtyp, lat, lon, date, temps, sup) = \
                ('*********', -1, -1, -1, -1, -1, None)
            handle = 0
            nbrp2 = 0
            for irep in xrange(nbrp):
                handle = rmn.mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup)
                ## sys.stderr.write(repr(handle)+'\n')
                self.assertNotEqual(handle, 0)
                nbrp2 += 1
            handle = 0
            sup = []
            for irep in xrange(nbrp):
                handle = rmn.mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup)
                self.assertNotEqual(handle, 0)
            ## handle = 0
            ## sup = _np.empty((1,), dtype=_np.int32)
            ## sup[0] = 0
            ## for irep in xrange(nbrp):
            ##     handle = rmn.mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup)
            ##     self.assertNotEqual(handle, 0)
            rmn.mrfcls(funit)
            rmn.fclos(funit)
            self.assertEqual(nbrp2, nbrp)

    def testmrfgetKnownValues(self):
        """mrfget should give known result with known input"""
        for mypath, itype, iunit in self.knownValues:
            rmn.mrfopt(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
            funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
            nbrp   = rmn.mrfopn(funit, rmn.FST_RO)
            maxlen = max(64, rmn.mrfmxl(funit))+10

            ## (stnid, idtyp, lat, lon, date, temps) = \
            ##     ('*********', -1, -1, -1, -1, -1)

            buf = None
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            #TODO: self.assertEqual(buf, ???)

            buf = maxlen
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            #TODO: self.assertEqual(buf, ???)

            buf = _np.empty((maxlen, ), dtype=_np.int32)
            buf[0] = maxlen
            handle = 0
            handle = rmn.mrfloc(funit, handle)
            buf = rmn.mrfget(handle, buf, funit)
            #TODO: self.assertEqual(buf, ???)

            ## for irep in xrange(nbrp):
            ##     handle = rmn.mrfloc(funit, handle)
            ##     buf = rmn.mrfget(handle, buf, funit)
            ##     ## print handle, ier, buf, buf.shape
            ##     self.assertEqual(ier, 0)
            
            rmn.mrfcls(funit)
            rmn.fclos(funit)


    ## def testmrbhdrKnownValues(self):
    ##     """fnomfclos should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         ier    = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
    ##         funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
    ##         nbrp   = rmn.c_mrfopn(funit, rmn.FST_RO)
    ##         maxlen = max(64, rmn.c_mrfmxl(funit))+10

    ##         (stnid, idtyp, lat, lon, date, temps, nsup, nxaux) = \
    ##             ('*********', -1, -1, -1, -1, -1, 0, 0)
    ##         sup  = _np.empty((1, ), dtype=_np.int32)
    ##         xaux = _np.empty((1, ), dtype=_np.int32)
    ##         buf = _np.empty((maxlen, ), dtype=_np.int32)
    ##         buf[0] = maxlen
    ##         handle = 0
            
    ##         itime = _ct.c_int(0)
    ##         iflgs = _ct.c_int(0)
    ##         stnids = ''
    ##         idburp = _ct.c_int(0)
    ##         ilat = _ct.c_int(0)
    ##         ilon = _ct.c_int(0)
    ##         idx = _ct.c_int(0)
    ##         idy = _ct.c_int(0)
    ##         ialt = _ct.c_int(0)
    ##         idelay = _ct.c_int(0)
    ##         idate = _ct.c_int(0)
    ##         irs = _ct.c_int(0)
    ##         irunn = _ct.c_int(0)
    ##         nblk = _ct.c_int(0)
    ##         for irep in xrange(nbrp):
    ##             handle = rmn.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup, nsup)
    ##             ier = rmn.c_mrfget(handle, buf)
    ##             ier = rmn.c_mrbhdr(buf, 
    ##                     itime, iflgs, stnids, idburp, 
    ##                     ilat, ilon, idx, idy, ialt, 
    ##                     idelay, idate, irs, irunn, nblk, 
    ##                     sup, nsup, xaux, nxaux)
    ##             ## print irep, handle, itime, iflgs, stnids, idburp, ilat, ilon, idx, idy, ialt, idelay, idate, irs, irunn, nblk
    ##             self.assertEqual(ier, 0)
                
    ##         ier    = rmn.c_mrfcls(funit)
    ##         ier    = rmn.fclos(funit)


    ## def testmrbprmKnownValues(self):
    ##     """fnomfclos should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         ier    = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
    ##         funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
    ##         nbrp   = rmn.c_mrfopn(funit, rmn.FST_RO)
    ##         maxlen = max(64, rmn.c_mrfmxl(funit))+10

    ##         (stnid, idtyp, lat, lon, date, temps, nsup, nxaux) = \
    ##             ('*********', -1, -1, -1, -1, -1, 0, 0)
    ##         sup  = _np.empty((1, ), dtype=_np.int32)
    ##         xaux = _np.empty((1, ), dtype=_np.int32)
    ##         buf = _np.empty((maxlen, ), dtype=_np.int32)
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

    ##         for irep in xrange(nbrp):
    ##             handle = rmn.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup, nsup)
    ##             ier = rmn.c_mrfget(handle, buf)
    ##             ier = rmn.c_mrbhdr(buf, 
    ##                     itime, iflgs, stnids, idburp, 
    ##                     ilat, ilon, idx, idy, ialt, 
    ##                     idelay, idate, irs, irunn, nblk, 
    ##                     sup, nsup, xaux, nxaux)
    ##             ## print irep, handle, itime, iflgs, stnids, idburp, ilat, ilon, idx, idy, ialt, idelay, idate, irs, irunn, nblk

    ##             for iblk in xrange(nblk.value):
    ##                 ier = rmn.c_mrbprm(buf, iblk, 
    ##                              nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp)
    ##                 ## print irep, iblk, nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp
    ##                 self.assertEqual(ier, 0)
                
    ##         ier    = rmn.c_mrfcls(funit)
    ##         ier    = rmn.fclos(funit)

    ## def testmrbxtrdclKnownValues(self):
    ##     """fnomfclos should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         ier    = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
    ##         funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
    ##         nbrp   = rmn.c_mrfopn(funit, rmn.FST_RO)
    ##         maxlen = max(64, rmn.c_mrfmxl(funit))+10

    ##         (stnid, idtyp, lat, lon, date, temps, nsup, nxaux) = \
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

    ##         for irep in xrange(nbrp):
    ##             handle = rmn.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup, nsup)
    ##             ier = rmn.c_mrfget(handle, buf)
    ##             ier = rmn.c_mrbhdr(buf, 
    ##                     itime, iflgs, stnids, idburp, 
    ##                     ilat, ilon, idx, idy, ialt, 
    ##                     idelay, idate, irs, irunn, nblk, 
    ##                     sup, nsup, xaux, nxaux)
    ##             ## print irep, handle, itime, iflgs, stnids, idburp, ilat, ilon, idx, idy, ialt, idelay, idate, irs, irunn, nblk

    ##             for iblk in xrange(nblk.value):
    ##                 ier = rmn.c_mrbprm(buf, iblk, 
    ##                              nele, nval, nt, bfam, bdesc, btyp, nbit, bit0, datyp)
    ##                 lstele = _np.empty((nele.value, ), dtype=_np.int32)
    ##                 ## tblval = _np.empty((nele.value, nval.value, nt.value), dtype=_np.int32)
    ##                 nmax = nele.value*nval.value*nt.value#*2
    ##                 tblval = _np.empty((nmax, ), dtype=_np.int32)
    ##                 ier = rmn.c_mrbxtr(buf, iblk, lstele, tblval)
    ##                 #print irep, iblk, ier, nele.value, nval.value, nt.value, lstele, tblval
    ##                 codes = _np.empty((nele.value, ), dtype=_np.int32)
    ##                 ier = rmn.c_mrbdcl(lstele, codes, nele)
    ##                 #print irep, iblk, ier, codes
    ##                 #self.assertEqual(ier, 0)
                
    ##         ier    = rmn.c_mrfcls(funit)
    ##         ier    = rmn.fclos(funit)


    ## def testmrbxtrcvtKnownValues(self):
    ##     """fnomfclos should give known result with known input"""
    ##     for mypath, itype, iunit in self.knownValues:
    ##         ier    = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
    ##         funit  = rmn.fnom(self.getFN(mypath), rmn.FST_RO)
    ##         nbrp   = rmn.c_mrfopn(funit, rmn.FST_RO)
    ##         maxlen = max(64, rmn.c_mrfmxl(funit))+10

    ##         (stnid, idtyp, lat, lon, date, temps, nsup, nxaux) = \
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

    ##         for irep in xrange(nbrp):
    ##             handle = rmn.c_mrfloc(funit, handle, stnid, idtyp, lat, lon, date, temps, sup, nsup)
    ##             ier = rmn.c_mrfget(handle, buf)
    ##             ier = rmn.c_mrbhdr(buf, 
    ##                     itime, iflgs, stnids, idburp, 
    ##                     ilat, ilon, idx, idy, ialt, 
    ##                     idelay, idate, irs, irunn, nblk, 
    ##                     sup, nsup, xaux, nxaux)
    ##             ## print irep, handle, itime, iflgs, stnids, idburp, ilat, ilon, idx, idy, ialt, idelay, idate, irs, irunn, nblk

    ##             for iblk in xrange(nblk.value):
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
    ##         ier    = rmn.c_mrfcls(funit)
    ##         ier    = rmn.fclos(funit)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
