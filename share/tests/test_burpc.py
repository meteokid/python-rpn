#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for burpc

See: http://iweb.cmc.ec.gc.ca/~afsdcvs/burplib_c/
"""

import os
import sys
import rpnpy.librmn.all as rmn
import rpnpy.burpc.all as brp
import unittest
## import ctypes as ct
import numpy as np

if sys.version_info > (3, ):
    long = int

#--- primitives -----------------------------------------------------

class RpnPyBurpc(unittest.TestCase):

    burptestfile = 'bcmk_burp/2007021900.brp'
    #(path, itype, iunit)
    knownValues = (
        (burptestfile, rmn.WKOFFIT_TYPE_LIST['BURP'], 999), 
        )

    def getFN(self, name):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        return os.path.join(ATM_MODEL_DFILES.strip(), name)
        

    def test_brp_opt_Error(self):
        """brp_opt should check for proper keys and types"""
        try:
            optValue = brp.brp_opt('No such Key')
            self.assertTrue(False, 'brp_opt should raise KeyError with "No such Key"')
        except KeyError:
            pass
        try:
            optValue = brp.brp_opt(rmn.BURPOP_MSGLVL)
            self.assertTrue(False, 'brp_opt should raise KeyError when getting msglvl')
        except KeyError:
            pass
        try:
            optValue = brp.brp_opt(rmn.BURPOP_MSGLVL, 1)
            self.assertTrue(False, 'brp_opt should raise TypeError when setting msglvl with int')
        except TypeError:
            pass
        
    def test_brp_opt_set(self):
        """brp_opt should give known result with known input"""
        for k in (rmn.BURPOP_MSG_TRIVIAL, rmn.BURPOP_MSG_INFO,
                  rmn.BURPOP_MSG_WARNING, rmn.BURPOP_MSG_ERROR,
                  rmn.BURPOP_MSG_FATAL, rmn.BURPOP_MSG_SYSTEM):
            optValue = brp.brp_opt(rmn.BURPOP_MSGLVL, k)
            self.assertEqual(optValue[0:6], k[0:6])
                   
    def test_brp_opt_get_set_missing(self):
        """brp_opt BURPOP_MISSING should give known result with known input"""
        optValue0 = 1.0000000150474662e+30
        optValue = brp.brp_opt(rmn.BURPOP_MISSING)
        self.assertEqual(optValue, optValue0)

        optValue0 = 99.
        optValue = brp.brp_opt(rmn.BURPOP_MISSING, optValue0)
        self.assertEqual(optValue, optValue0)
        
    ## def test_brp_opt_set_get_missing(self):
    ##     """(known bug) brp_opt BURPOP_MISSING should give known result with known input """ #TODO: apparently c_brp_SetOptFloat(BURPOP_MISSING, value) is not working
    ##     optValue0 = 99.
    ##     optValue = brp.brp_opt(rmn.BURPOP_MISSING, optValue0)
    ##     optValue = brp.brp_opt(rmn.BURPOP_MISSING)
    ##     self.assertEqual(optValue, optValue0)


    def test_brp_open_ValueError(self):
        """brp_open filemode ValueError"""
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        try:
            brp.brp_open(mypath, 'z')
            raise Error('brp_open should raise ValueError on wrong filemode')
        except ValueError:
            pass

    def test_brp_open_ReadOnly(self):
        """brp_open  ReadOnly Error"""
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        try:
            brp.brp_open(mypath, 'w')
            self.assertTrue(False, 'brp_open should raise BurpcError wrong permission')
        except brp.BurpcError:
            pass
        
    def test_brp_open_no_such_file(self):
        """brp_open no_such_file"""
        try:
            brp.brp_open('__no_such_file__', 'r')
            self.assertTrue(False, 'brp_open should raise BurpcError file not found')
        except brp.BurpcError:
            pass

    def test_brp_open_close(self):
        """brp_open  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        funit = brp.brp_open(mypath)
        brp.brp_close(funit)
        funit = brp.brp_open(mypath, 'r')
        brp.brp_close(funit)
        funit = brp.brp_open(mypath, rmn.BURP_MODE_READ)
        brp.brp_close(funit)
        funit0 = 10
        funit = brp.brp_open(mypath, rmn.BURP_MODE_READ, funit=funit0)
        brp.brp_close(funit)
        self.assertEqual(funit, funit0)
        (funit, nrec) = brp.brp_open(mypath, getnbr=True)
        brp.brp_close(funit)
        self.assertEqual(nrec, 47544)


    def test_brp_BurpcFile(self):
        """brp_BurpcFile_open  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        del bfile
        bfile = brp.BurpcFile(mypath, 'r')
        del bfile
        bfile = brp.BurpcFile(mypath, rmn.BURP_MODE_READ)
        del bfile
        bfile0 = 10
        bfile = brp.BurpcFile(mypath, rmn.BURP_MODE_READ, funit=bfile0)
        self.assertEqual(bfile.funit, bfile0)
        self.assertEqual(len(bfile), 47544)
        del bfile

    def test_brp_BurpcFile_with(self):
        """brp_BurpcFile_with  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            self.assertEqual(len(bfile), 47544)
            
    def test_brp_BurpcFile_iter(self):
        """brp_BurpcFile_iter  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            i = 0
            for rpt in bfile:
                i += 1
            self.assertEqual(len(bfile), i)        
        
    def test_BURP_RPT(self):
        """brp_BURP_RPT  """
        rpt = brp.BurpcRpt()
        rpt.stnid = '012345678'
        self.assertEqual(rpt.stnid, '012345678')
        del rpt
        
    def test_BURP_RPT_error(self):
        """brp_BURP_RPT  error"""
        try:
            rpt = brp.BurpcRpt(1)
            self.assertTrue(False, 'BURP_RPT should raise TypeError when init with int')
        except TypeError:
            pass
        
    def test_BURP_RPT_keyerror(self):
        """brp_BURP_RPT  keyerror"""
        rpt = brp.BurpcRpt()
        try:
            a = rpt.no_such_attr
            self.assertTrue(False, 'BURP_RPT.attr should raise AttrError when init with int')
        except AttributeError:
            pass
        #TODO: should we prevent setting a unknown param?
        ## try:
        ##     rpt.no_such_attr = 1
        ##     self.assertTrue(False, 'BURP_RPT.attr should raise AttrError when init with int')
        ## except AttributeError:
        ##     pass

    def test_BURP_RPT_dict(self):
        """brp_BURP_RPT  dict"""
        rpt = brp.BurpcRpt({
            'stnid' : '012345678',
            'date'  : 20170101
            })
        self.assertEqual(rpt.stnid, '012345678')
        self.assertEqual(rpt.date, 20170101)
        
    def test_BURP_RPT_rpt(self):
        """brp_BURP_RPT  rpt"""
        rpt0 = brp.BurpcRpt()
        rpt0.stnid = '012345678'
        rpt0.date = 20170101
        rpt = brp.BurpcRpt(rpt0)
        self.assertEqual(rpt.stnid, '012345678')
        self.assertEqual(rpt.date, 20170101)
        self.assertEqual(rpt['stnid'], '012345678')
        self.assertEqual(rpt['date'], 20170101)
        # Should have taken a copy
        rpt0.stnid = 'abcdefghi'
        rpt0.date = 20000101
        self.assertEqual(rpt.stnid, '012345678')
        self.assertEqual(rpt.date, 20170101)        

    def test_BURP_RPT_update_error(self):
        """brp_BURP_RPT  update error"""
        rpt = brp.BurpcRpt()
        try:
            rpt.update(1)
            self.assertTrue(False, 'BURP_RPT_update should raise TypeError when init with int')
        except TypeError:
            pass

    def test_BURP_RPT_update_dict(self):
        """brp_BURP_RPT  update_dict"""
        rpt = brp.BurpcRpt()
        rpt.update({
            'stnid' : '012345678',
            'date'  : 20170101
            })
        self.assertEqual(rpt.stnid, '012345678')
        self.assertEqual(rpt.date, 20170101)

    def test_BURP_RPT_update_rpt(self):
        """brp_BURP_RPT  update_rpt"""
        rpt  = brp.BurpcRpt()
        rpt2 = brp.BurpcRpt()
        rpt2.stnid = '012345678'
        rpt2.date = 20170101
        rpt.update(rpt2)
        self.assertEqual(rpt.stnid, '012345678')
        self.assertEqual(rpt.date, 20170101)

    def test_brp_find_rpt0(self):
        """brp_find_rpt  funit"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        self.assertEqual(rpt.handle, 1)

    def test_brp_find_rpt1(self):
        """brp_find_rpt  bfile"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        self.assertEqual(rpt.handle, 1)
        
    def test_brp_find_rpt1b(self):
        """with BurpcFile getrpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
        self.assertEqual(rpt.handle, 1)

    def test_brp_find_rpt2(self):
        """brp_find_rpt 2nd """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_findrpt(bfile, rpt)
        self.assertEqual(rpt.handle, 1025)
        
    def test_brp_find_rpt2b(self):
        """with BurpcFile getrpt 2nd"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            rpt = bfile.getrpt(rpt.handle)
        self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt2c(self):
        """with BurpcFile getrpt 2nd + recycle mem"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            rpt = bfile.getrpt(rpt.handle, rpt)
        self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt2d(self):
        """brp_BurpcFile_item  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            rpt = bfile[rpt.handle]
        self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt_not_found(self):
        """brp_find_rpt  not_found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.BurpcRpt()
        rpt.stnid = '123456789'
        rpt = brp.brp_findrpt(bfile, rpt)
        self.assertEqual(rpt, None)

    def test_brp_find_rpt_not_found2(self):
        """with BurpcFile getrpt not_found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = brp.BurpcRpt()
            rpt.stnid = '123456789'
            rpt = bfile.getrpt(rpt)
        self.assertEqual(rpt, None)

    def test_brp_find_rpt_not_found2d(self):
        """with BurpcFile getrpt not_found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = brp.BurpcRpt({'stnid' : '123456789'})
            rpt = bfile.getrpt(rpt)
        self.assertEqual(rpt, None)

    def test_brp_find_rpt_handle(self):
        """brp_find_rpt  handle"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile, 1)
        self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt_stnid(self):
        """brp_find_rpt stnid"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.BurpcRpt()
        rpt.stnid = 'S********'
        rpt = brp.brp_findrpt(bfile, rpt)
        self.assertEqual(rpt.handle, 1227777)

    def test_brp_find_rpt_stnid2(self):
        """brp_find_rpt stnid"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = brp.BurpcRpt()
            rpt.stnid = 'S********'
            rpt = bfile.getrpt(rpt)
        self.assertEqual(rpt.handle, 1227777)

    def test_brp_find_rpt_stnid2d(self):
        """brp_BurpcFile_item  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = brp.BurpcRpt({'stnid' : 'S********'})
            rpt = bfile[rpt]
        self.assertEqual(rpt.handle, 1227777)

    def test_brp_get_rpt1(self):
        """brp_get_rpt handle"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_getrpt(bfile, rpt.handle)
        self.assertEqual(rpt.handle, 1)
        self.assertEqual(rpt.stnid, '71915    ')
        self.assertEqual(rpt.date, 20070219)

    def test_brp_get_rpt2(self):
        """brp_get_rpt handle + rpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_getrpt(bfile, rpt.handle, rpt)
        self.assertEqual(rpt.handle, 1)
        self.assertEqual(rpt.stnid, '71915    ')
        self.assertEqual(rpt.date, 20070219)
        
    def test_brp_get_rpt3(self):
        """brp_get_rpt from rpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_getrpt(bfile, rpt)
        self.assertEqual(rpt.handle, 1)
        self.assertEqual(rpt.stnid, '71915    ')
        self.assertEqual(rpt.date, 20070219)

    def test_BURP_BLK(self):
        """brp_BURP_BLK  """
        blk = brp.BurpcBlk()
        blk.btyp = 8
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk['btyp'], 8)
        del blk

    def test_BURP_BLK_keyerror(self):
        """brp_BURP_BLK  keyerror"""
        blk = brp.BurpcBlk()
        try:
            a = blk.no_such_attr
            self.assertTrue(False, 'BURP_BLK.attr should raise AttrError when init with int')
        except AttributeError:
            pass
        #TODO: should we prevent setting a unknown param?
        ## try:
        ##     blk.no_such_attr = 1
        ##     self.assertTrue(False, 'BURP_BLK.attr should raise AttrError when init with int')
        ## except AttributeError:
        ##     pass
        
    def test_BURP_BLK_error(self):
        """brp_BURP_BLK  error"""
        try:
            blk = brp.BurpcBlk(1)
            self.assertTrue(False, 'BURP_BLK should raise TypeError when init with int')
        except TypeError:
            pass

    def test_BURP_BLK_dict(self):
        """brp_BURP_BLK  dict"""
        blk = brp.BurpcBlk({
            'btyp'  : 8,
            'datyp' : 1
            })
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk.datyp, 1)
        
    def test_BURP_BLK_blk(self):
        """brp_BURP_BLK  blk"""
        blk0 = brp.BurpcBlk()
        blk0.btyp = 8
        blk0.datyp = 1
        blk = brp.BurpcBlk(blk0)
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk.datyp, 1)
        # Should have taken a copy
        blk0.btyp = 3
        blk0.datyp = 4
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk.datyp, 1)

    def test_BURP_BLK_update_error(self):
        """brp_BURP_BLK  update error"""
        blk = brp.BurpcBlk()
        try:
            blk.update(1)
            self.assertTrue(False, 'BURP_BLK_update should raise TypeError when init with int')
        except TypeError:
            pass

    def test_BURP_BLK_update_dict(self):
        """brp_BURP_BLK  update_dict"""
        blk = brp.BurpcBlk()
        blk.update({
            'btyp'  : 8,
            'datyp' : 1
            })
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk.datyp, 1)

    def test_BURP_BLK_update_blk(self):
        """brp_BURP_BLK  update_blk"""
        blk  = brp.BurpcBlk()
        blk2 = brp.BurpcBlk()
        blk2.btyp = 8
        blk2.datyp = 1
        blk.update(blk2)
        self.assertEqual(blk.btyp, 8)
        self.assertEqual(blk.datyp, 1)

    def test_brp_find_blk0(self):
        """brp_find_blk bkno"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        self.assertEqual(rpt.nblk, 12)
        blk = brp.brp_findblk(None, rpt)
        self.assertEqual(blk.bkno, 1)
        self.assertEqual(blk.datyp, 4)
        blk = brp.brp_findblk(blk.bkno, rpt)
        self.assertEqual(blk.bkno, 2)
        self.assertEqual(blk.datyp, 4)
        blk = brp.brp_findblk(blk, rpt)
        self.assertEqual(blk.bkno, 3)
        self.assertEqual(blk.datyp, 4)
        blk = brp.brp_findblk(6, rpt)
        self.assertEqual(blk.bkno, 7)
        self.assertEqual(blk.datyp, 4)

    def test_brp_find_blk0b(self):
        """BurpcRpt getblk"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            self.assertEqual(rpt.nblk, 12)
            blk = rpt.getblk()
            self.assertEqual(blk.bkno, 1)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.getblk(None)
            self.assertEqual(blk.bkno, 1)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.getblk(blk.bkno)
            self.assertEqual(blk.bkno, 2)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.getblk(6)
            self.assertEqual(blk.bkno, 7)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.getblk(6, blk)
            self.assertEqual(blk.bkno, 7)
            self.assertEqual(blk.datyp, 4)
        
    def test_brp_find_blk_err1(self):
        """brp_find_blk bkno not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        self.assertEqual(rpt.nblk, 12)
        blk = brp.brp_findblk(12, rpt)
        self.assertEqual(blk, None)

    def test_brp_find_blk_err1b(self):
        """BurpcRpt getblk bkno not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            self.assertEqual(rpt.nblk, 12)
            blk = rpt.getblk(12)
            self.assertEqual(blk, None)

    def test_brp_find_blk_err2(self):
        """brp_find_blk bkno search keys not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.BurpcBlk({'bkno':0, 'btyp':999})
        blk = brp.brp_findblk(blk, rpt)
        self.assertEqual(blk, None)

    def test_brp_find_blk_err2b(self):
        """BurpcRpt getblk search keys not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':999})
            blk = rpt.getblk(blk)
            self.assertEqual(blk, None)
            blk = rpt.getblk({'bkno':0, 'btyp':999})
            self.assertEqual(blk, None)

    def test_brp_find_blk_err2d(self):
        """BurpcRpt getblk search keys not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile.getrpt()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':999})
            blk = rpt[blk]
            self.assertEqual(blk, None)
            blk = rpt[{'bkno':0, 'btyp':999}]
            self.assertEqual(blk, None)

    def test_brp_find_get_blk1(self):
        """brp_find_blk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
        blk = brp.brp_findblk(blk, rpt)
        blk = brp.brp_getblk(blk.bkno, blk, rpt)
        self.assertEqual(blk.bkno, 6)
        self.assertEqual(blk.datyp, 2)
        self.assertEqual(blk.btyp, 15456)

    def test_brp_find_blk_blk1b(self):
        """BurpcRpt getblk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
            blk = rpt.getblk(blk)
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)
            blk = rpt.getblk({'bkno':0, 'btyp':15456})
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)

    def test_brp_find_blk_blk1d(self):
        """BurpcRpt getblk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile.getrpt()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
            blk = rpt[blk]
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)
            blk = rpt[{'bkno':0, 'btyp':15456}]
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)

    def test_brp_get_blk_iter(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            i = 0
            for blk in rpt:
                i += 1
                self.assertEqual(blk.bkno, i)
                self.assertTrue(i <= rpt.nblk)
            self.assertEqual(blk.bkno, rpt.nblk)

    def test_brp_get_blk_data(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.brp_findblk(None, rpt)
        blk = brp.brp_getblk(blk.bkno, blk, rpt)
        self.assertEqual(blk.lstele.shape, (8, ))
        self.assertEqual(blk.lstele[0], 2564)
        self.assertEqual(blk.lstele[2], 2828)
        self.assertEqual(blk.tblval.shape, (8, 1, 1))
        self.assertEqual(blk.tblval[0,0,0], 10)
        self.assertEqual(blk.tblval[2,0,0], -1)

    def test_brp_get_blk_datab(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.getrpt()
            blk = rpt.getblk()
            self.assertEqual(blk.lstele.shape, (8, ))
            self.assertEqual(blk.lstele[0], 2564)
            self.assertEqual(blk.lstele[2], 2828)
            self.assertEqual(blk.tblval.shape, (8, 1, 1))
            self.assertEqual(blk.tblval[0,0,0], 10)
            self.assertEqual(blk.tblval[2,0,0], -1)

    def test_brp_rpt_derived(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            rpt2 = rpt.derived_attr()
            a = {
                'idtypd' : 'TEMP + PILOT + SYNOP' ,
                'datemm' : 2 ,
                'dy' : 0.0 ,
                'nxaux' : 0 ,
                'lat' : 64.19999999999999 ,
                'flgsl' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] ,
                'lon' : 276.63 ,
                'flgsd' : 'surface wind used, data observed, data derived, residues, TEMP part B' ,
                'nsup' : 0 ,
                'datedd' : 19 ,
                'timemm' : 0 ,
                'drnd' : 0 ,
                'flgs' : 72706 ,
                'xaux' : None ,
                'sup' : None ,
                'nblk' : 12 ,
                'ilon' : 27663 ,
                'oars' : 518 ,
                'dx' : 0.0 ,
                'stnid' : '71915    ' ,
                'date' : 20070219 ,
                'ilat' : 15420 ,
                'ielev' : 457 ,
                'idx' : 0 ,
                'idy' : 0 ,
                'idtyp' : 138 ,
                'elev' : 57.0 ,
                'time' : 0 ,
                'dateyy' : 2007 ,
                'timehh' : 0 ,
                'runn' : 8
                }
            for k in a.keys():
                self.assertEqual(rpt2[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(rpt2[k])))
            a.update({
                'rdx' : 0.0 ,
                'rdy' : 0.0 ,
                'relev' : 57.0 ,
                })
            del a['dx'], a['dy'], a['elev']
            for k in a.keys():
                self.assertEqual(rpt[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(rpt[k])))
            self.assertEqual(rpt.rdx, a['rdx'])
            self.assertEqual(rpt.rdy, a['rdy'])
            self.assertEqual(rpt.relev, a['relev'])
            self.assertEqual(rpt.ilon, a['ilon'])
            self.assertEqual(rpt.ilon, rpt.longi)

    def test_brp_rpt_derived_integrity(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            rpt2 = rpt.derived_attr()
            self.assertEqual(rpt2['ilon'], rpt.ilon)
            rpt2['ilon'] = 1234556789
            self.assertNotEqual(rpt2['ilon'], rpt.ilon)

    def test_brp_blk_derived(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            blk = rpt[1]
            blk2 = blk.derived_attr()
            ## for k,v in blk2.items():
            ##     print "{} : {},".format(repr(k),repr(v))
            a = {
                'bktyp' : 6,
                'nele' : 8,
                'nbit' : 14,
                'bktyp_kindd' : 'data seen by OA at altitude, global model',
                'bknat_kind' : 0,
                'bfam' : 14,
                'nval' : 1,
                'btyp' : 106,
                'bkstpd' : 'statistiques de diff\xc3\xa9rences (r\xc3\xa9sidus)',
                'bknat' : 0,
                'bktyp_alt' : 0,
                'bknat_kindd' : 'data',
                'bknat_multi' : 0,
                'bdesc' : 0,
                'bkstp' : 10,
                'bkno' : 1,
                'datypd' : 'int',
                'bktyp_kind' : 6,
                'datyp' : 4,
                'nt' : 1,
                'bit0' : 0                
                }
            for k in a.keys():
                self.assertEqual(blk2[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(blk2[k])))
            for k in a.keys():
                self.assertEqual(blk[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(blk[k])))
            self.assertEqual(blk.bfam, a['bfam'])
            self.assertEqual(blk.bknat_kindd, a['bknat_kindd'])

    def test_brp_blk_derived_integrity(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            blk = rpt[1]
            blk2 = blk.derived_attr()
            self.assertEqual(blk.bknat_kind, blk2['bknat_kind'])
            blk2['bknat_kind'] = 1234556789
            self.assertNotEqual(blk.bknat_kind, blk2['bknat_kind'])


    def test_brp_blk_getele(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            blk = rpt[1]
            ## for k,v in rmn.mrbcvt_dict(blk.lstele[0], raise_error=False).items():
            ##     print "{} : {},".format(repr(k),repr(v))
            ele = blk.getelem(0)
            ## for k,v in ele.items():
            ##     print "{} : {},".format(repr(k),repr(v))
            a = {
                'e_ele_no' : 0,
                'e_scale' : -1,
                'e_multi' : 0,
                'e_cmcid' : 2564,
                'e_bias' : 0,
                'e_error' : 0,
                'e_bufrid_F' : 0,
                'e_units' : 'PA',
                'e_desc' : 'PRESSURE',
                'e_bufrid' : 10004,
                'e_nbits' : 14,
                'e_bufrid_Y' : 4,
                'e_bufrid_X' : 10,
                'e_cvt' : 1,
                'bktyp' : 6,
                'nele' : 8,
                'nbit' : 14,
                'store_type' : 'F',
                'e_bias' : 0,
                'e_rval' : np.array([[ 100.]], dtype=np.float32),
                'bfam' : 14,
                'e_charval' : None,
                'e_desc' : 'PRESSURE',
                'e_val' : np.array([[ 100.]], dtype=np.float32),
                'max_len' : 8,
                'e_cmcid' : 2564,
                'e_tblval' : np.array([[10]], dtype=np.int32),
                'e_drval' : None,
                'nval' : 1,
                'btyp' : 106,
                'e_cvt' : 1,
                'bknat' : 0,
                'e_multi' : 0,
                'e_error' : 0,
                'e_scale' : -1,
                'bdesc' : 0,
                'e_bufrid_F' : 0,
                'e_units' : 'PA',
                'bkstp' : 10,
                'bkno' : 1,
                'e_nbits' : 14,
                'max_nval' : 1,
                'max_nele' : 8,
                'datyp' : 4,
                'e_bufrid_Y' : 4,
                'e_bufrid_X' : 10,
                'e_bufrid' : 10004,
                'bit0' : 0,
                'nt' : 1,
                'max_nt' : 1
                }
            for k in a.keys():
                self.assertEqual(ele[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(ele[k])))
            ele = blk[0]
            for k in a.keys():
                self.assertEqual(ele[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(ele[k])))           
            try:
                ele = blk[-1]
                self.assertTrue(False, 'BurpcBlk.getelem() out of range should raise an error')
            except IndexError:
                pass
            try:
                ele = blk[blk.nele]
                self.assertTrue(False, 'BurpcBlk.getelem() out of range should raise an error')
            except IndexError:
                pass
            print
            print blk.lstele
            ele = blk[1]
            self.assertEqual(ele['e_cmcid'], blk.lstele[1])

    def test_brp_blk_getele_iter(self):
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            blk = rpt[1]
            n = 0
            for ele in blk:
                self.assertEqual(n, ele['e_ele_no'])
                self.assertEqual(ele['e_cmcid'], blk.lstele[n])
                self.assertEqual(ele['e_tblval'], blk.tblval[n,:,:])
                n += 1
            self.assertEqual(n, blk.nele)


    #TODO: tests for writing burp


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
