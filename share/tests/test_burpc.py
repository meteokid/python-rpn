#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for burpc

See: http://iweb.cmc.ec.gc.ca/~afsdcvs/burplib_c/
"""

import os
import sys
import rpnpy.librmn.all as rmn
import rpnpy.burpc.all as brpc
import unittest
import ctypes as _ct
import numpy as _np

if sys.version_info > (3, ):
    long = int

#--- primitives -----------------------------------------------------

#TODO: mrfopn return None
#TODO: mrfopn takes BURP_MODE_READ, BURP_MODE_CREATE, BURP_MODE_APPEND as mode
#TODO: replace fnom + mrfopn with burp_open, mrdcls bu burp_close

class RpnPyBurpc(unittest.TestCase):

    burptestfile = 'bcmk_burp/2007021900.brp'
    #(path, itype, iunit)
    knownValues = (
        (burptestfile, rmn.WKOFFIT_TYPE_LIST['BURP'], 999), 
        )

    def getFN(self, name):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        return os.path.join(ATM_MODEL_DFILES.strip(), name)
        

    def test_ex1_read1(self):
        """burplib_c iweb doc example 1"""
        bs, br = brpc.c_brp_newblk(), brpc.c_brp_newblk()
        rs, rr = brpc.c_brp_newrpt(), brpc.c_brp_newrpt()
        istat = brpc.c_brp_SetOptChar("MSGLVL", "FATAL" )
        for mypath, itype, iunit in self.knownValues:
            istat = brpc.c_brp_open(iunit, self.getFN(mypath), "r")
            print("enreg {}".format(istat))
            brpc.RPT_SetHANDLE(rs,0)
            while brpc.c_brp_findrpt(iunit, rs) >= 0:
                if brpc.c_brp_getrpt(iunit, brpc.RPT_HANDLE(rs), rr) >= 0:
                    print("stnid = {}".format(brpc.RPT_STNID(rr)))
                    brpc.BLK_SetBKNO(bs, 0)
                    while brpc.c_brp_findblk(bs, rr) >= 0:
                        if brpc.c_brp_getblk(brpc.BLK_BKNO(bs), br, rr) >= 0:
                            print("block btyp = {}".format(brpc.BLK_BTYP(br)))
            istat = brpc.c_brp_close(iunit)
            ## self.assertEqual(funit, itype,
            ##                  mypath+':'+repr(funit)+' != '+repr(itype))
        brpc.brpc_free(bs, br, rs, rr)


    def test_ex2_readburp(self):
        """burplib_c iweb doc example 2"""
        bs, br = brpc.c_brp_newblk(), brpc.c_brp_newblk()
        rs, rr = brpc.c_brp_newrpt(), brpc.c_brp_newrpt()
        istat = brpc.c_brp_SetOptChar("MSGLVL", "FATAL" )
        for mypath, itype, iunit in self.knownValues:
            istat = brpc.c_brp_open(iunit, self.getFN(mypath), "r")
            print("Nombre Enreg = {}".format(istat))
            brpc.RPT_SetHANDLE(rs,0)
            while brpc.c_brp_findrpt(iunit, rs) >= 0:
                if brpc.c_brp_getrpt(iunit, brpc.RPT_HANDLE(rs), rr) < 0:
                    continue
                print("""
hhmm   ={:8d} flgs   ={:6d}  codtyp ={:6d}  stnids ={:9s}
blat   ={:8d} blon   ={:6d}  dx     ={:6d}  dy     ={:6d}  stnhgt ={:6d}
yymmdd ={:8d} oars   ={:6d}  runn   ={:6d}  nblk   ={:6d}  dlay   ={:6d}
""".format(brpc.RPT_TEMPS(rr), brpc.RPT_FLGS(rr), brpc.RPT_IDTYP(rr),
           brpc.RPT_STNID(rr), brpc.RPT_LATI(rr), brpc.RPT_LONG(rr),
           brpc.RPT_DX(rr), brpc.RPT_DY(rr), brpc.RPT_ELEV(rr),
           brpc.RPT_DATE(rr), brpc.RPT_OARS(rr), brpc.RPT_RUNN(rr),
           brpc.RPT_NBLK(rr), brpc.RPT_DRND(rr)))

                brpc.BLK_SetBKNO(bs, 0)
                while brpc.c_brp_findblk(bs, rr) >= 0:
                    if brpc.c_brp_getblk(brpc.BLK_BKNO(bs), br, rr) < 0:
                        continue
                    print("""
blkno  ={:6d}  nele   ={:6d}  nval   ={:6d}  nt     ={:6d}  bit0   ={:6d}
bdesc  ={:6d}  btyp   ={:6d}  nbit   ={:6d}  datyp  ={:6d}  bfam   ={:6d}
""".format(brpc.BLK_BKNO(br), brpc.BLK_NELE(br), brpc.BLK_NVAL(br), 
           brpc.BLK_NT(br), brpc.BLK_BIT0(br), brpc.BLK_BDESC(br), 
           brpc.BLK_BTYP(br), brpc.BLK_NBIT(br), brpc.BLK_DATYP(br),
           brpc.BLK_BFAM(br))) 
                    for k in range(brpc.BLK_NT(br)):
                        if brpc.BLK_BKNO(br) != 1:
                            print("\nobservation {}/{}".
                                  format(k+1, brpc.BLK_NT(br)))
                        mystr = "lstele ="
                        for i in range(brpc.BLK_NELE(br)):
                            mystr += "    {:0>6d}".format(brpc.BLK_DLSTELE(br,i))
                        print(mystr)
                        for j in range(brpc.BLK_NVAL(br)):
                            mystr = "tblval ="
                            for i in range(brpc.BLK_NELE(br)):
                                mystr += "{:10d}".format(brpc.BLK_TBLVAL(br,i,j,k))
                            print(mystr)

            istat = brpc.c_brp_close(iunit)
            ## self.assertEqual(funit, itype,
            ##                  mypath+':'+repr(funit)+' != '+repr(itype))
        brpc.brpc_free(bs, br, rs, rr)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
