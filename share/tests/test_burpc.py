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
        bs = brpc.c_brp_newblk()
        br = brpc.c_brp_newblk()
        rs = brpc.c_brp_newrpt()
        rr = brpc.c_brp_newrpt()
        istat = brpc.c_brp_SetOptChar("MSGLVL", "FATAL" )
        for mypath, itype, iunit in self.knownValues:
            istat = brpc.c_brp_open(iunit, self.getFN(mypath), "r")
            print("enreg {}".format(istat))
            brpc.RPT_SetHANDLE(rs,0)
            while(brpc.c_brp_findrpt(iunit, rs) >= 0):
                if (brpc.c_brp_getrpt(iunit, brpc.RPT_HANDLE(rs), rr) >= 0):
                    print("stnid = {}".format(brpc.RPT_STNID(rr)))
                    brpc.BLK_SetBKNO(bs, 0)
                    while(brpc.c_brp_findblk(bs, rr) >= 0):
                        if (brpc.c_brp_getblk(brpc.BLK_BKNO(bs), br, rr)>=0):
                            print("block btyp = {}".format(brpc.BLK_BTYP(br)))
            istat = brpc.c_brp_close(iunit)
            ## self.assertEqual(funit, itype,
            ##                  mypath+':'+repr(funit)+' != '+repr(itype))
        brpc.c_brp_freeblk(bs)
        brpc.c_brp_freeblk(br)
        brpc.c_brp_freerpt(rs)
        brpc.c_brp_freerpt(rr)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
