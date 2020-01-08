#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for burpc

See: http://iweb.cmc.ec.gc.ca/~afsdcvs/burplib_c/
"""

import os
import sys
import rpnpy.librmn.all as rmn
import rpnpy.burpc.all as brp
from rpnpy import range as _range
import unittest
import ctypes as ct
import numpy as np

if sys.version_info > (3, ):
    long = int

_GETPTR = lambda o, t: o.getptr() if (isinstance(o, t)) else o
_RPTPTR = lambda rpt: _GETPTR(rpt, brp.BurpcRpt)
_BLKPTR = lambda blk: _GETPTR(blk, brp.BurpcBlk)

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

    def test_brp_msngval(self):
        optValue0 = 1.0000000150474662e+30
        optValue = brp.brp_msngval()
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

    def test_brp_newrpt(self):
        rpt = brp.brp_newrpt()
        self.assertTrue(isinstance(rpt, ct.POINTER(brp.BURP_RPT)))
        brp.RPT_SetSTNID(rpt, '012345678')
        self.assertEqual(brp.RPT_STNID(rpt), '012345678')

    def test_brp_newblk(self):
        blk = brp.brp_newblk()
        self.assertTrue(isinstance(blk, ct.POINTER(brp.BURP_BLK)))
        brp.BLK_SetBTYP(blk, 8)
        self.assertEqual(brp.BLK_BTYP(blk), 8)

    def test_brp_allocrpt(self):
        rpt = brp.brp_newrpt()
        brp.brp_allocrpt(rpt, 999)
        self.assertEqual(brp.RPT_NSIZE(rpt), 999)

    def test_brp_resizerpt(self):
        rpt = brp.brp_newrpt()
        brp.brp_allocrpt(rpt, 99)
        self.assertEqual(brp.RPT_NSIZE(rpt), 99)
        brp.brp_resizerpt(rpt, 888)
        self.assertEqual(brp.RPT_NSIZE(rpt), 888)

    def test_brp_allocblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 3,2,1)
        self.assertEqual((brp.BLK_NELE(blk), brp.BLK_NVAL(blk), brp.BLK_NT(blk)), (3,2,1))

    def test_brp_resizeblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 3,2,1)
        self.assertEqual((brp.BLK_NELE(blk), brp.BLK_NVAL(blk), brp.BLK_NT(blk)), (3,2,1))
        brp.brp_resizeblk(blk, 7,5,3)
        self.assertEqual((brp.BLK_NELE(blk), brp.BLK_NVAL(blk), brp.BLK_NT(blk)), (7,5,3))

    def test_brp_free(self):
        rpt = brp.brp_newrpt()
        brp.brp_allocrpt(rpt, 99)
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 3,2,1)
        brp.brp_free(blk, rpt)
        self.assertEqual(brp.RPT_NSIZE(rpt), 0)
        self.assertEqual((brp.BLK_NELE(blk), brp.BLK_NVAL(blk), brp.BLK_NT(blk)), (0,0,0))

    def test_brp_clrrpt(self):
        rpt = brp.brp_newrpt()
        brp.brp_allocrpt(rpt, 99)
        brp.brp_clrrpt(rpt)
        #TODO: self.assert

    def test_brp_clrblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 3,2,1)
        brp.brp_clrblk(blk)
        #TODO: self.assert

    def test_brp_clrblkv(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 3,2,1)
        brp.brp_clrblkv(blk, 9.)
        #TODO: self.assert

    def test_brp_resetrpthdr(self):
        rpt = brp.brp_newrpt()
        brp.brp_resetrpthdr(rpt)
        #TODO: self.assert

    def test_brp_resetblkhdr(self):
        blk = brp.brp_newblk()
        brp.brp_resetblkhdr(blk)
        #TODO: self.assert

    def test_brp_encodeblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 1,1,1)
        brp.brp_clrblk(blk)
        brp.BLK_SetDLSTELE(blk, 0, 10004)
        brp.brp_encodeblk(blk)
        self.assertEqual(brp.BLK_LSTELE(blk, 0), 2564)

    def test_brp_convertblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 1,1,1)
        brp.brp_clrblk(blk)
        brp.BLK_SetDLSTELE(blk, 0, 10004)
        brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
        brp.brp_encodeblk(blk)
        brp.brp_convertblk(blk)
        self.assertEqual(brp.BLK_TBLVAL(blk, 0, 0, 0), 10)

    def test_brp_safe_convertblk(self):
        blk = brp.brp_newblk()
        brp.brp_allocblk(blk, 1,1,1)
        brp.brp_clrblk(blk)
        brp.BLK_SetDLSTELE(blk, 0, 10004)
        brp.BLK_SetRVAL(blk, 0, 0, 0, 100.)
        brp.brp_encodeblk(blk)
        brp.brp_safe_convertblk(blk)
        self.assertEqual(brp.BLK_TBLVAL(blk, 0, 0, 0), 10)

    ## def test_brp_initrpthdr(self): #TODO
    ## def test_brp_putrpthdr(self): #TODO
    ## def test_brp_updrpthdr(self): #TODO
    ## def test_brp_writerpt(self): #TODO
    ## def test_brp_putblk(self): #TODO
    ## def test_brp_delblk(self): #TODO
    ## def test_brp_delrpt(self): #TODO

    def test_brp_copyrpthdr(self):
        rpt = brp.brp_newrpt()
        brp.RPT_SetSTNID(rpt, '012345678')
        self.assertEqual(brp.RPT_STNID(rpt), '012345678')
        rpt2 = brp.brp_newrpt()
        brp.brp_copyrpthdr(rpt2, rpt)
        brp.RPT_SetSTNID(rpt2, '987654321')
        self.assertEqual(brp.RPT_STNID(rpt),  '012345678')
        self.assertEqual(brp.RPT_STNID(rpt2), '987654321')

    def test_brp_copyrpt(self):
        rpt = brp.brp_newrpt()
        brp.RPT_SetSTNID(rpt, '012345678')
        self.assertEqual(brp.RPT_STNID(rpt), '012345678')
        rpt2 = brp.brp_newrpt()
        brp.brp_copyrpt(rpt2, rpt)
        brp.RPT_SetSTNID(rpt2, '987654321')
        self.assertEqual(brp.RPT_STNID(rpt),  '012345678')
        self.assertEqual(brp.RPT_STNID(rpt2), '987654321')

    def test_brp_copyblk(self):
        blk = brp.brp_newblk()
        brp.BLK_SetBTYP(blk, 8)
        self.assertEqual(brp.BLK_BTYP(blk), 8)
        blk2 = brp.brp_newblk()
        brp.brp_copyblk(blk2, blk)
        brp.BLK_SetBTYP(blk2, 9)
        self.assertEqual(brp.BLK_BTYP(blk),  8)
        self.assertEqual(brp.BLK_BTYP(blk2), 9)
 
    def test_brp_searchdlste(self):
        """test_brp_searchdlste"""
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        filename = os.path.join(ATM_MODEL_DFILES, 'bcmk_burp', '2007021900.brp')
        funit = brp.brp_open(filename, brp.BRP_FILE_READ)
        rpt = brp.brp_findrpt(funit)
        rpt = brp.brp_getrpt(funit, rpt)
        blk = brp.brp_findblk(None, rpt)
        blk = brp.brp_getblk(brp.BLK_BKNO(blk), rpt=rpt)
        idx = brp.brp_searchdlste(11011, blk)
        brp.brp_close(funit)
        self.assertEqual(idx, 1)

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

    def test_brp_BurpcFile_iter_break(self):
        """brp_BurpcFile_iter  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            i = 0
            for rpt in bfile:
                i += 1
                if i == 10:
                    break
            i = 0
            for rpt in bfile:
                i += 1
            self.assertEqual(len(bfile), i)

    def test_brp_BurpcFile_iter2(self):
        """brp_BurpcFile_iter  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            for i in _range(len(bfile)):
                rpt = bfile[i]
                i += 1
            self.assertEqual(len(bfile), i)

    def test_brp_BurpcFile_indexError(self):
        """brp_BurpcFile_iter  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            try:
                rpt = bfile[-1]
                self.assertTrue(False, 'BurpFile[-1] should raise index error')
            except IndexError:
                pass
            try:
                rpt = bfile[len(bfile)]
                self.assertTrue(False, 'BurpFile[len(bfile)] should raise index error')
            except IndexError:
                pass

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
            self.assertTrue(False, 'BURP_RPT.attr should raise AttrError')
        except AttributeError:
            pass
        try:
            a = rpt['no_such_key']
            self.assertTrue(False, 'BURP_RPT["no_such_key"] should raise KeyError')
        except KeyError:
            pass
        #TODO: should we prevent setting a unknown param?
        ## try:
        ##     rpt.no_such_attr = 1
        ##     self.assertTrue(False, 'BURP_RPT.attr should raise AttrError setting an unknown param')
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
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)

    def test_brp_find_rpt1(self):
        """brp_find_rpt  bfile"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)

    def test_brp_find_rpt1b(self):
        """with BurpcFile getrpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
        self.assertEqual(rpt.handle, 1)

    def test_brp_find_rpt2(self):
        """brp_find_rpt 2nd """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_findrpt(bfile, rpt)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1025)

    def test_brp_find_rpt2b(self):
        """with BurpcFile getrpt 2nd"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            rpt = bfile.get({'handle' : rpt.handle})
            self.assertEqual(rpt.handle, 1025)
            rpt = bfile.get(1)
            self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt2c(self):
        """with BurpcFile getrpt 2nd + recycle mem"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            rpt = bfile.get({'handle' : rpt.handle}, rpt)
            self.assertEqual(rpt.handle, 1025)
            rpt = bfile.get(1, rpt)
            self.assertEqual(rpt.handle, 1025)

    def test_brp_find_rpt2d(self):
        """brp_BurpcFile_item  """
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            rpt = bfile[{'handle' : rpt.handle}]
            self.assertEqual(rpt.handle, 1025)
            rpt = bfile[1]
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
            rpt = bfile.get(rpt)
        self.assertEqual(rpt, None)

    def test_brp_find_rpt_not_found2d(self):
        """with BurpcFile getrpt not_found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = brp.BurpcRpt({'stnid' : '123456789'})
            rpt = bfile.get(rpt)
        self.assertEqual(rpt, None)

    def test_brp_find_rpt_handle(self):
        """brp_find_rpt  handle"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile, 1)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1025)

    def test_brp_find_rpt_stnid(self):
        """brp_find_rpt stnid"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.BurpcRpt()
        rpt.stnid = 'S********'
        rpt = brp.brp_findrpt(bfile, rpt)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1227777)

    def test_brp_find_rpt_stnid2(self):
        """brp_find_rpt stnid"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = brp.BurpcRpt()
            rpt.stnid = 'S********'
            rpt = bfile.get(rpt)
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
        rpt = brp.brp_getrpt(bfile, brp.RPT_HANDLE(rpt))
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)
        self.assertEqual(brp.RPT_STNID(rpt), '71915    ')
        self.assertEqual(brp.RPT_DATE(rpt), 20070219)

    def test_brp_get_rpt1hdr(self):
        """brp_get_rpt handle"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_rdrpthdr(brp.RPT_HANDLE(rpt), rpt)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)
        self.assertEqual(brp.RPT_STNID(rpt), '71915    ')
        self.assertEqual(brp.RPT_DATE(rpt), 20070219)

    def test_brp_get_rpt2(self):
        """brp_get_rpt handle + rpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_getrpt(bfile, brp.RPT_HANDLE(rpt), rpt)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)
        self.assertEqual(brp.RPT_STNID(rpt), '71915    ')
        self.assertEqual(brp.RPT_DATE(rpt), 20070219)

    def test_brp_get_rpt3(self):
        """brp_get_rpt from rpt"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile)
        rpt = brp.brp_getrpt(bfile, rpt)
        self.assertEqual(brp.RPT_HANDLE(rpt), 1)
        self.assertEqual(brp.RPT_STNID(rpt), '71915    ')
        self.assertEqual(brp.RPT_DATE(rpt), 20070219)

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
        self.assertEqual(brp.RPT_NBLK(rpt), 12)
        blk = brp.brp_findblk(None, rpt)
        self.assertEqual(brp.BLK_BKNO(blk), 1)
        self.assertEqual(brp.BLK_DATYP(blk), 4)
        blk = brp.brp_findblk(brp.BLK_BKNO(blk), rpt)
        self.assertEqual(brp.BLK_BKNO(blk), 2)
        self.assertEqual(brp.BLK_DATYP(blk), 4)
        blk = brp.brp_findblk(blk, rpt)
        self.assertEqual(brp.BLK_BKNO(blk), 3)
        self.assertEqual(brp.BLK_DATYP(blk), 4)
        blk = brp.brp_findblk(6, rpt)
        self.assertEqual(brp.BLK_BKNO(blk), 7)
        self.assertEqual(brp.BLK_DATYP(blk), 4)

    def test_brp_find_blk0b(self):
        """BurpcRpt getblk"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            self.assertEqual(rpt.nblk, 12)
            blk = rpt.get()
            self.assertEqual(blk.bkno, 1)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.get(None)
            self.assertEqual(blk.bkno, 1)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.get(3)
            self.assertEqual(blk.bkno, 4)
            self.assertEqual(blk.datyp, 2)
            blk = rpt[6]
            self.assertEqual(blk.bkno, 7)
            self.assertEqual(blk.datyp, 4)
            blk = rpt.get(6, blk)
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
        self.assertEqual(brp.RPT_NBLK(rpt), 12)
        blk = brp.brp_findblk(12, rpt)
        self.assertEqual(blk, None)

    def test_brp_find_blk_err1b(self):
        """BurpcRpt getblk bkno not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            self.assertEqual(rpt.nblk, 12)
            blk = rpt[0]
            blk = rpt[11]
            try:
                blk = rpt[12]
                self.assertTrue(False, "gpt.get should raise IndexError")
            except:
                pass
            try:
                blk = rpt[-1]
                self.assertTrue(False, "gpt.get should raise IndexError")
            except:
                pass

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
            rpt = bfile.get()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':999})
            blk = rpt.get(blk)
            self.assertEqual(blk, None)
            blk = rpt.get({'bkno':0, 'btyp':999})
            self.assertEqual(blk, None)

    def test_brp_find_blk_err2d(self):
        """BurpcRpt getblk search keys not found"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile.get()
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

    def test_brp_find_get_blk2(self):
        """brp_find_blk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = bfile.get(0)
        blk = rpt.get({'bkno':0, 'btyp':15456})
        ## blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
        ## blk = brp.brp_findblk(blk, rpt)
        ## blk = brp.brp_getblk(blk.bkno, blk, rpt)
        self.assertEqual(blk.bkno, 6)
        self.assertEqual(blk.datyp, 2)
        self.assertEqual(blk.btyp, 15456)

    def test_brp_find_get_blk1safe(self):
        """brp_find_blk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
        blk = brp.brp_findblk(blk, rpt)
        blk = brp.brp_safe_getblk(blk.bkno, blk, rpt)
        self.assertEqual(blk.bkno, 6)
        self.assertEqual(blk.datyp, 2)
        self.assertEqual(blk.btyp, 15456)

    def test_brp_find_get_blk1read(self):
        """brp_find_blk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
        blk = brp.brp_findblk(blk, rpt)
        blk = brp.brp_readblk(blk.bkno, blk, rpt)
        self.assertEqual(blk.bkno, 6)
        self.assertEqual(blk.datyp, 2)
        self.assertEqual(blk.btyp, 15456)

    def test_brp_find_get_blk1hdr(self):
        """brp_find_blk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
        blk = brp.brp_findblk(blk, rpt)
        blk = brp.brp_rdblkhdr(blk.bkno, blk, rpt)
        self.assertEqual(blk.bkno, 6)
        self.assertEqual(blk.datyp, 2)
        self.assertEqual(blk.btyp, 15456)

    def test_brp_find_blk_blk1b(self):
        """BurpcRpt getblk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
            blk = rpt.get(blk)
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)
            blk = rpt.get({'bkno':0, 'btyp':15456})
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)

    def test_brp_find_blk_blk1d(self):
        """BurpcRpt getblk search keys"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile.get()
            blk = brp.BurpcBlk({'bkno':0, 'btyp':15456})
            blk = rpt[blk]
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)
            blk = rpt[{'bkno':0, 'btyp':15456}]
            self.assertEqual(blk.bkno, 6)
            self.assertEqual(blk.datyp, 2)
            self.assertEqual(blk.btyp, 15456)

    ## def test_min_rpt_buf_size(self):
    ##     brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    ##     mypath = self.getFN(self.knownValues[0][0])
    ##     ## print '\nFile:', mypath
    ##     with brp.BurpcFile(mypath) as bfile:
    ##         i=0
    ##         for rpt in bfile:
    ##             m = 0
    ##             l = 0
    ##             s = 0
    ##             x = 0
    ##             for blk in rpt:
    ##                 max_len = blk.getptr()[0].max_len
    ##                 m = max(m, max_len)
    ##                 s += max_len
    ##                 l += blk.nele*blk.nval*blk.nt
    ##                 x += rmn.LBLK(blk.nele, blk.nval, blk.nt, blk.nbit)
    ##             ##     print 'b', blk.bkno, blk.nbit, blk.bit0,\
    ##             ##         ':nijk', blk.nele*blk.nval*blk.nt,\
    ##             ##         ':maxl', max_len,\
    ##             ##         ':LBLK', rmn.LBLK(blk.nele, blk.nval, blk.nt, blk.nbit)
    ##             ## print 'r', rpt.stnid, rpt.lngr, rpt.nsize, ':n', rmn.LRPT(x)
    ##             ## i += 1
    ##             ## if i == 2: break
    ##             self.assertEqual(rmn.LRPT(x), rpt.lngr + 10)

    def test_brp_get_blk_iter(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            i = 0
            for blk in rpt:
                i += 1
                self.assertEqual(blk.bkno, i)
                self.assertTrue(i <= rpt.nblk)
            self.assertEqual(blk.bkno, rpt.nblk)
            self.assertEqual(i, rpt.nblk)

    def test_brp_get_blk_iter_break(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            i = 0
            for blk in rpt:
                i += 1
                if i == 2:
                    break
            i = 0
            for blk in rpt:
                i += 1
                self.assertEqual(blk.bkno, i)
                self.assertTrue(i <= rpt.nblk)
            self.assertEqual(blk.bkno, rpt.nblk)
            self.assertEqual(i, rpt.nblk)

    def test_brp_get_blk_iter2(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            for i in _range(rpt.nblk):
                blk = rpt[i]
                self.assertEqual(blk.bkno, i+1)
                self.assertTrue(i <= rpt.nblk)
            self.assertEqual(blk.bkno, rpt.nblk)

    def test_brp_get_blk_cmp(self):
        """Report iter on block cmp proto and obj"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)

        x = []
        with brp.BurpcFile(mypath) as bfile:
            bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
            rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
            while brp.c_brp_findrpt(bfile.funit, rs) >= 0:
                if brp.c_brp_getrpt(bfile.funit, brp.RPT_HANDLE(rs), rr) < 0:
                    continue
                if  brp.RPT_STNID(rr) == '>>POSTALT':
                    continue
                brp.BLK_SetBKNO(bs, 0)
                while brp.c_brp_findblk(bs, rr) >= 0:
                    if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                        continue
                    x1 = (brp.RPT_STNID(rr), brp.BLK_BKNO(br),
                          brp.BLK_NELE(br), brp.BLK_NVAL(br), brp.BLK_NT(br),
                          brp.BLK_NELE(br),
                          brp.BLK_NELE(br) * brp.BLK_NVAL(br) * brp.BLK_NT(br))
                    x.append(x1)

        with brp.BurpcFile(mypath) as bfile:
            i = 0
            for rr in bfile:
                if  rr.stnid == '>>POSTALT':
                    continue
                for br in rr:
                    x1 = (rr.stnid, br.bkno, br.nele, br.nval, br.nt,
                          br.dlstele.size, br.tblval.size)
                    self.assertEqual(x[i], x1)
                    i += 1


    def test_brp_get_blk_data(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        bfile = brp.BurpcFile(mypath)
        rpt = brp.brp_findrpt(bfile.funit)
        rpt = brp.brp_getrpt(bfile, rpt)
        blk = brp.brp_findblk(None, rpt)
        blk = brp.brp_getblk(brp.BLK_BKNO(blk), blk, rpt)
        self.assertEqual(brp.BLK_NELE(blk), 8)
        self.assertEqual(brp.BLK_LSTELE(blk, 0), 2564)
        self.assertEqual(brp.BLK_LSTELE(blk, 2), 2828)
        self.assertEqual(brp.BLK_NVAL(blk), 1)
        self.assertEqual(brp.BLK_NT(blk), 1)
        self.assertEqual(brp.BLK_TBLVAL(blk, 0,0,0), 10)
        self.assertEqual(brp.BLK_TBLVAL(blk, 2,0,0), -1)

    def test_brp_get_blk_datab(self):
        """Report iter on block"""
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        mypath = self.getFN(mypath)
        with brp.BurpcFile(mypath) as bfile:
            rpt = bfile.get()
            blk = rpt.get()
            self.assertEqual(blk.lstele.shape, (8, ))
            self.assertEqual(blk.lstele[0], 2564)
            self.assertEqual(blk.lstele[2], 2828)
            self.assertEqual(blk.tblval.shape, (8, 1, 1))
            self.assertEqual(blk.tblval[0,0,0], 10)
            self.assertEqual(blk.tblval[2,0,0], -1)

    def skipped_test_brp_rpt_derived(self): #derived_attr now priv #TODO: todict
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

    def skipped_test_brp_rpt_derived_integrity(self): #derived_attr now priv #TODO: todict
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            rpt2 = rpt.derived_attr()
            self.assertEqual(rpt2['ilon'], rpt.ilon)
            rpt2['ilon'] = 1234556789
            self.assertNotEqual(rpt2['ilon'], rpt.ilon)

    def skipped_test_brp_blk_derived(self): #derived_attr now priv #TODO: todict
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        mypath, itype, iunit = self.knownValues[0]
        with brp.BurpcFile(self.getFN(mypath)) as bfile:
            rpt = bfile[0]
            blk = rpt[0]
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
                'bkstpd' : "statistiques de différences (résidus)",
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

    def skipped_test_brp_blk_derived_integrity(self): #derived_attr now priv #TODO: todict
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
            blk = rpt[0]
            ## for k,v in rmn.mrbcvt_dict(blk.lstele[0], raise_error=False).items():
            ##     print "{} : {},".format(repr(k),repr(v))
            ele = blk.get(0)
            ## for k,v in ele.items():
            ##     print "{} : {},".format(repr(k),repr(v))
            a = {
                'store_type': 'F',
                'shape': (1, 1),
                'e_drval': None,
                'e_rval': np.array([[ 100.]], dtype=np.float32),
                'e_bias': 0,
                'e_error': 0,
                'e_charval': None,
                'e_bufrid_F': 0,
                'e_desc': 'PRESSURE',
                'e_nbits': 14,
                'e_bufrid_Y': 4,
                'e_bufrid_X': 10,
                'nt': 1,
                'e_ival': None,
                'e_tblval': np.array([[ 10.]], dtype=np.int32),
                'e_scale': -1,
                'ptrkey': 'e_rval',
                'nval': 1,
                'e_multi': 0,
                'e_cmcid': 2564,
                'e_units': 'PA',
                'e_bufrid': 10004,
                'e_cvt': 1}
            ok = True
            for k in a.keys():
                try:
                    self.assertEqual(ele[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(ele[k])))
                except AssertionError:
                    ok = False
                    print('Should be equal {}: expected={}, got={}'.
                          format(k, repr(a[k]), repr(ele[k])))
            self.assertTrue(ok)
            ele = blk[0]
            ok = True
            for k in a.keys():
                try:
                    self.assertEqual(ele[k], a[k], 'Should be equal {}: expected={}, got={}'.format(k, repr(a[k]), repr(ele[k])))
                except AssertionError:
                    ok = False
                    print('Should be equal {}: expected={}, got={}'.
                          format(k, repr(a[k]), repr(ele[k])))
            self.assertTrue(ok)
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
            ## print
            ## print blk.lstele
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
                self.assertEqual(ele['e_cmcid'], blk.lstele[n])
                self.assertEqual(ele['ptrkey'], 'e_rval')
                self.assertTrue(np.all(ele['e_rval'][:,:]==blk.rval[n,:,:]))
                n += 1
            self.assertEqual(n, blk.nele)

    def test_BurpcEle_init_args(self):
        e = brp.BurpcEle(7004, [10])
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.e_tblval[0], 10)

    def test_BurpcEle_init_dict(self):
        e = brp.BurpcEle({'e_bufrid' : 7004, 'e_tblval' : [10]})
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.store_type, 'I')
        self.assertEqual(e.e_tblval[0], 10)
        self.assertEqual(e.ptrkey, 'e_ival')

    def test_BurpcEle_init_dict2(self):
        e = brp.BurpcEle({'e_bufrid' : 7004,
                          'store_type' : 'I',
                          'e_tblval' : [10, 2]})
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.store_type, 'I')
        self.assertEqual(e.e_tblval[0, 0], 10)
        self.assertEqual(e.e_tblval[1, 0], 2)
        self.assertEqual(e.shape, (2, 1))
        self.assertEqual(e.nval, 2)
        self.assertEqual(e.nt, 1)

    def test_BurpcEle_init_dict3(self):
        e = brp.BurpcEle({'e_bufrid' : 7004,
                          'e_tblval' : [10, 2]})
        e.reshape((1,2))
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.store_type, 'I')
        self.assertEqual(e.e_ival[0, 0], 100)
        self.assertEqual(e.e_ival[0, 1],  20)
        self.assertEqual(e.e_tblval[0, 0], 10)
        self.assertEqual(e.e_tblval[0, 1],  2)
        self.assertEqual(e.shape, (1, 2))
        self.assertEqual(e.nval, 1)
        self.assertEqual(e.nt, 2)

    def test_BurpcEle_init_dict4(self):
        e = brp.BurpcEle({'e_bufrid' : 7004,
                          'e_ival' : [100, 20]})
        e.reshape((1,2))
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.store_type, 'I')
        self.assertEqual(e.e_ival[0, 0], 100)
        self.assertEqual(e.e_ival[0, 1],  20)
        self.assertEqual(e.e_tblval[0, 0], 10)
        self.assertEqual(e.e_tblval[0, 1],  2)
        self.assertEqual(e.shape, (1, 2))
        self.assertEqual(e.nval, 1)
        self.assertEqual(e.nt, 2)

    def test_BurpcEle_init_derived(self):
        e = brp.BurpcEle({'e_cmcid' : 1796, 'e_rval' : [10.]})
        self.assertEqual(e.e_bufrid, 7004)
        ## self.assertEqual(e.e_tblval[0], 10)
        self.assertEqual(e.e_rval[0], 10.)
        self.assertEqual(e.store_type, 'F')
        self.assertEqual(e.ptrkey, 'e_rval')

    def test_BurpcEle_init_derived2(self):
        e = brp.BurpcEle({'e_cmcid' : 1796, 'e_drval' : [10.]})
        self.assertEqual(e.e_bufrid, 7004)
        ## self.assertEqual(e.e_tblval[0], 10)
        self.assertEqual(e.e_drval[0], 10.)
        self.assertEqual(e.store_type, 'D')
        self.assertEqual(e.ptrkey, 'e_drval')

    def test_BurpcEle_init_copy(self):
        e  = brp.BurpcEle({'e_bufrid' : 7004, 'e_tblval' : [10]})
        e2 = brp.BurpcEle(e)
        self.assertEqual(e2.e_bufrid, e.e_bufrid)
        self.assertEqual(e2.e_tblval[0], e.e_tblval[0])

    def test_BurpcEle_init_copy_inegrity(self):
        e  = brp.BurpcEle({'e_bufrid' : 7004, 'e_tblval' : [10]})
        e2 = brp.BurpcEle(e)
        e2.e_bufrid = 1234
        e2.e_tblval[0] = 9876
        self.assertNotEqual(e2.e_bufrid, e.e_bufrid)
        self.assertNotEqual(e2.e_tblval[0], e.e_tblval[0])


    #TODO: test report copy & _inegrity
    #TODO: test block copy & _inegrity

    def test_BurpcEle_init_err(self):
        try:
            e = brp.BurpcEle()
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle(7004)
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle('7004')
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle({'e_bufrid' : 7004})
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle({'bufrid' : 7004, 'e_tblval' : [10]})
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle({'e_bufrid' : 7004,
                              'e_tblval' : [10],
                              'e_rval' : [10.]})
            self.assertTrue(False, 'should cause an init error')
        except:
            pass
        try:
            e = brp.BurpcEle({'e_bufrid' : 7004,
                              'store_type' : 'I',
                              'e_rval' : [10.]})
            self.assertTrue(False, 'should cause an init error')
        except:
            pass

    def test_BurpcEle_get(self):
        e = brp.BurpcEle(7004, [10])
        self.assertEqual(e.e_bufrid, 7004)
        self.assertEqual(e.e_tblval[0], 10)
        self.assertEqual(e.e_ival[0], 100)
        self.assertEqual(e.store_type, 'I')

    def test_BurpcEle_get_derived(self):
        e = brp.BurpcEle(7004, [10])
        self.assertEqual(e.e_cmcid, 1796)
        ## self.assertEqual(e.store_type, 'I')

    ## def test_BurpcEle_get_derived_inegrity(self):
    ##     pass
    ## def test_BurpcEle_get_error(self):
    ##     pass

    def test_BurpcEle_set(self):
        e  = brp.BurpcEle({'e_bufrid' : 7004, 'e_tblval' : [10]})
        e.e_bufrid = 1234
        self.assertEqual(e.e_bufrid, 1234)

    def test_BurpcEle_set_derived(self):
        e  = brp.BurpcEle({'e_bufrid' : 7006, 'e_tblval' : [10]})
        self.assertEqual(e.e_bufrid, 7006)
        self.assertEqual(e.e_cmcid, 1798)
        e.e_cmcid = 1796 #TODO: no going through put()!!!
        self.assertEqual(e.e_cmcid, 1796)
        self.assertEqual(e.e_bufrid, 7004)

    def test_BurpcEle_set_derived2(self):
        e  = brp.BurpcEle({'e_bufrid' : 7006, 'e_tblval' : [10]})
        self.assertEqual(e.e_bufrid, 7006)
        self.assertEqual(e.e_cmcid, 1798)
        e['e_cmcid'] = 1796
        self.assertEqual(e.e_cmcid, 1796)
        self.assertEqual(e.e_bufrid, 7004)

    def test_BurpcEle_set_error_key(self):
        e  = brp.BurpcEle({'e_bufrid' : 7006, 'e_tblval' : [10]})
        try:
            e.ptrkey = 'toto'
            self.assertTrue(False, 'should cause a set error')
        except KeyError:
            pass

    ## def test_BurpcEle_set_error_type(self):
    ##     pass

    def test_BurpcBlk_add_BurpcEle_r(self):
        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_FLOAT,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'e_bufrid' : 7004,
            'e_rval'   : [10.]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'e_bufrid' : 11001,
            'e_rval'   : [20.]
            })
        blk.append(ele2)
        ele1b = blk[0]
        ele2b = blk[1]

        self.assertEqual(blk.btyp, 64)
        self.assertEqual(blk.datyp, brp.BRP_DATYP_INTEGER)
        self.assertEqual(blk.store_type, brp.BRP_STORE_FLOAT)
        self.assertEqual(blk.nele, 2)
        self.assertEqual(blk.nval, 1)
        self.assertEqual(blk.nt, 1)

        self.assertEqual(ele1b.e_bufrid, 7004)
        self.assertEqual(ele1b.e_rval[0,0], 10.)
        self.assertEqual(ele1b.e_tblval[0,0], 1)
        self.assertEqual(ele2b.e_bufrid, 11001)
        self.assertEqual(ele2b.e_rval[0,0], 20.)
        self.assertEqual(ele2b.e_tblval[0,0], 20)

        self.assertEqual(blk.dlstele[0], 7004)
        self.assertEqual(blk.tblval[0,0,0], 1)
        self.assertEqual(blk.rval[0,0,0], 10.)
        self.assertEqual(blk.dlstele[1], 11001)
        self.assertEqual(blk.tblval[1,0,0], 20)
        self.assertEqual(blk.rval[1,0,0], 20.)

    #TODO: segFault with double
    def NeedFixing_test_BurpcBlk_add_BurpcEle_d(self):
        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_DOUBLE,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'e_bufrid' : 7004,
            'e_drval'   : [10.]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'e_bufrid' : 11001,
            'e_drval'   : [20.]
            })
        blk.append(ele2)
        ele1b = blk[0]
        ele2b = blk[1]

        self.assertEqual(blk.btyp, 64)
        self.assertEqual(blk.datyp, brp.BRP_DATYP_INTEGER)
        self.assertEqual(blk.store_type, brp.BRP_STORE_DOUBLE)
        self.assertEqual(blk.nele, 2)
        self.assertEqual(blk.nval, 1)
        self.assertEqual(blk.nt, 1)

        self.assertEqual(ele1b.e_bufrid, 7004)
        self.assertEqual(ele1b.e_drval[0,0], 10.)
        self.assertEqual(ele1b.e_tblval[0,0], 1)
        self.assertEqual(ele2b.e_bufrid, 11001)
        self.assertEqual(ele2b.e_drval[0,0], 20.)
        self.assertEqual(ele2b.e_tblval[0,0], 20)

        self.assertEqual(blk.dlstele[0], 7004)
        self.assertEqual(blk.tblval[0,0,0], 1)
        self.assertEqual(blk.drval[0,0,0], 10.)
        self.assertEqual(blk.dlstele[1], 11001)
        self.assertEqual(blk.tblval[1,0,0], 20)
        self.assertEqual(blk.drval[1,0,0], 20.)


    def test_BurpcBlk_add_BurpcEle_i(self):
        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_INTEGER,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'e_bufrid' : 7004,
            'e_ival'   : [10]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'e_bufrid' : 11001,
            'e_tblval'   : [20]
            })
        blk.append(ele2)
        ele1b = blk[0]
        ele2b = blk[1]

        self.assertEqual(blk.btyp, 64)
        self.assertEqual(blk.datyp, brp.BRP_DATYP_INTEGER)
        self.assertEqual(blk.store_type, brp.BRP_STORE_INTEGER)
        self.assertEqual(blk.nele, 2)
        self.assertEqual(blk.nval, 1)
        self.assertEqual(blk.nt, 1)

        self.assertEqual(ele1b.e_bufrid, 7004)
        self.assertEqual(ele1b.e_ival[0,0], 10)
        self.assertEqual(ele1b.e_tblval[0,0], 1)
        self.assertEqual(ele2b.e_bufrid, 11001)
        self.assertEqual(ele2b.e_ival[0,0], 20)
        self.assertEqual(ele2b.e_tblval[0,0], 20)

        self.assertEqual(blk.dlstele[0], 7004)
        self.assertEqual(blk.tblval[0,0,0], 1)
        self.assertEqual(blk.ival[0,0,0], 10)
        self.assertEqual(blk.dlstele[1], 11001)
        self.assertEqual(blk.tblval[1,0,0], 20)
        self.assertEqual(blk.ival[1,0,0], 20)

    def test_BurpcBlk_add_BurpcEle_missing(self):
        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'store_type' : brp.BRP_STORE_FLOAT,
            'e_bufrid' : 7004,
            'e_tblval'   : [1, rmn.BURP_TBLVAL_MISSING]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'store_type' : brp.BRP_STORE_FLOAT,
            'e_bufrid' : 11001,
            'e_tblval'   : [20, 0.]
            })
        blk.append(ele2)
        ele1b = blk[0]
        ele2b = blk[1]

        missVal = brp.brp_opt(rmn.BURPOP_MISSING)

        self.assertEqual(blk.btyp, 64)
        self.assertEqual(blk.datyp, brp.BRP_DATYP_INTEGER)
        self.assertEqual(blk.store_type, brp.BRP_STORE_FLOAT)
        self.assertEqual(blk.nele, 2)
        self.assertEqual(blk.nval, 2)
        self.assertEqual(blk.nt, 1)

        self.assertEqual(ele1b.e_bufrid, 7004)
        self.assertEqual(ele1b.e_rval[0,0], 10.)
        self.assertEqual(ele1b.e_tblval[0,0], 1)
        self.assertEqual(ele1b.e_rval[1,0], missVal)
        self.assertEqual(ele1b.e_tblval[1,0], rmn.BURP_TBLVAL_MISSING)

        self.assertEqual(ele2b.e_bufrid, 11001)
        self.assertEqual(ele2b.e_rval[0,0], 20.)
        self.assertEqual(ele2b.e_tblval[0,0], 20)
        self.assertEqual(ele2b.e_rval[1,0], 0.)
        self.assertEqual(ele2b.e_tblval[1,0], 0)

        self.assertEqual(blk.dlstele[0], 7004)
        self.assertEqual(blk.tblval[0,0,0], 1)
        self.assertEqual(blk.rval[0,0,0], 10.)
        self.assertEqual(blk.tblval[0,1,0], rmn.BURP_TBLVAL_MISSING)
        self.assertEqual(blk.rval[0,1,0], missVal)

        self.assertEqual(blk.dlstele[1], 11001)
        self.assertEqual(blk.tblval[1,0,0], 20)
        self.assertEqual(blk.rval[1,0,0], 20.)
        self.assertEqual(blk.tblval[1,1,0], 0)
        self.assertEqual(blk.rval[1,1,0], 0.)

    def test_BurpcBlk_add_BurpcEle_r_neg(self):
        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_FLOAT,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'e_bufrid' : 7004,
            'e_rval'   : [-10.]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'e_bufrid' : 11001,
            'e_rval'   : [-20.]
            })
        blk.append(ele2)
        ele1b = blk[0]
        ele2b = blk[1]

        self.assertEqual(blk.btyp, 64)
        self.assertEqual(blk.datyp, brp.BRP_DATYP_INTEGER)
        self.assertEqual(blk.store_type, brp.BRP_STORE_FLOAT)
        self.assertEqual(blk.nele, 2)
        self.assertEqual(blk.nval, 1)
        self.assertEqual(blk.nt, 1)

        self.assertEqual(ele1b.e_bufrid, 7004)
        self.assertEqual(ele1b.e_rval[0,0], -10.)
        self.assertEqual(ele1b.e_tblval[0,0], -2)
        self.assertEqual(ele2b.e_bufrid, 11001)
        self.assertEqual(ele2b.e_rval[0,0], -20.)
        self.assertEqual(ele2b.e_tblval[0,0], -21)

        self.assertEqual(blk.dlstele[0], 7004)
        self.assertEqual(blk.tblval[0,0,0], -2)
        self.assertEqual(blk.rval[0,0,0], -10.)
        self.assertEqual(blk.dlstele[1], 11001)
        self.assertEqual(blk.tblval[1,0,0], -21)
        self.assertEqual(blk.rval[1,0,0], -20.)

   #TODO: test mrbcvt_encode/decode with +/- values with id cvt=yes/no cycle for ival, tblval, rval..


   #TODO: test BurpcRPT add BurpcBlk

    def test_BurpcBlk_add_BurpcEle_err_shape(self):
        pass

    def test_BurpcBlk_add_BurpcEle_err_type(self):
        pass


    #TODO: tests for writing burp


if __name__ == "__main__":
    unittest.main() ## verbosity=9)

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
