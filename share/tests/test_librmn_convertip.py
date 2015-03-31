#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.fstd98"""

import librmn.all as rmn
import unittest
## import ctypes as ct
## import numpy as np

#--- fstd98/convip --------------------------------------------------

class Librmn_ConvertIp_Test(unittest.TestCase):

    #lvlnew,lvlold,ipnew,ipold,kind
    #we need to specify 2 levels since the old style ip1 give is an approx of level in some cases
    ip1knownValues = (
        (0.,    0.,    15728640, 12001,rmn.KIND_ABOVE_SEA),
        (13.5,  15.,   8523608,  12004,rmn.KIND_ABOVE_SEA),
        (1500., 1500., 6441456,  12301,rmn.KIND_ABOVE_SEA),
        (5525., 5525., 6843956,  13106,rmn.KIND_ABOVE_SEA),
        (12750.,12750.,5370380,  14551,rmn.KIND_ABOVE_SEA),
        (0.,    0.,    32505856, 2000, rmn.KIND_SIGMA),
        (0.1,   0.1,   27362976, 3000, rmn.KIND_SIGMA),
        (0.02,  0.02,  28511552, 2200, rmn.KIND_SIGMA),
        (0.000003,0.,  32805856, 2000, rmn.KIND_SIGMA),
        (1024.,1024.,  39948288, 1024, rmn.KIND_PRESSURE),
        (850., 850.,   41744464, 850,  rmn.KIND_PRESSURE),
        (650., 650.,   41544464, 650,  rmn.KIND_PRESSURE),
        (500., 500.,   41394464, 500,  rmn.KIND_PRESSURE),
        (10.,  10.,    42043040, 10,   rmn.KIND_PRESSURE),
        (2.,   2.,     43191616, 1840, rmn.KIND_PRESSURE),
        (0.3,  0.3,    44340192, 1660, rmn.KIND_PRESSURE)
        )

    #(rp1.v1,rpn1.v2,rp1.kind), (rp2.v1,rpn2.v2,rp2.kind), (rp3.v1,rpn3.v2,rp3.kind), (ip1,ip2,ip3)
    ip123knownValues = (
        (rmn.FLOAT_IP(5525.,5525.,rmn.KIND_ABOVE_SEA),
         rmn.FLOAT_IP(2., 2.,rmn.KIND_HOURS),
         rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
         6843956, 177409344, 66060288),
         (rmn.FLOAT_IP(1500.,1500.,rmn.KIND_ABOVE_SEA),
          rmn.FLOAT_IP(12.,12.,rmn.KIND_HOURS),
          rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
          6441456, 176280768, 66060288)
        )
    
    ## #(rp1.v1,rpn1.v2,rp1.kind), (rp2.v1,rpn2.v2,rp2.kind), (rp3.v1,rpn3.v2,rp3.kind), (ip1,ip2,ip3)
    ## ip123knownErrors = (
    ## (rmn.FLOAT_IP(13.5, 1500.,rmn.KIND_ABOVE_SEA),
    ##  rmn.FLOAT_IP(3., 6.,rmn.KIND_HOURS),
    ##  rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
    ##  0,0,0),
    ## (rmn.FLOAT_IP(13.5, 1500.,rmn.KIND_ABOVE_SEA),
    ##  rmn.FLOAT_IP(3., 3.,rmn.KIND_ABOVE_SEA),
    ##  rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
    ##  0,0,0)
    ## )

    #(rp1.v1,rpn1.v2,rp1.kind), (rp2.v1,rpn2.v2,rp2.kind), (rp3.v1,rpn3.v2,rp3.kind), (ip1,ip2,ip3)
    ip123knownValues2 = (
        (rmn.FLOAT_IP(5525.,5525.,rmn.KIND_ABOVE_SEA),
         rmn.FLOAT_IP(2., 2.,rmn.KIND_HOURS),
         rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
         6843956, 177409344, 66060288),
        (rmn.FLOAT_IP(13.5, 1500.,rmn.KIND_ABOVE_SEA),
         rmn.FLOAT_IP(3., 3.,rmn.KIND_HOURS),
         rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
         8523608, 177509344, 6441456),
        (rmn.FLOAT_IP(1500.,1500.,rmn.KIND_ABOVE_SEA),
         rmn.FLOAT_IP(6.,12.,rmn.KIND_HOURS),
         rmn.FLOAT_IP(0., 0.,rmn.KIND_ARBITRARY),
         6441456, 176280768, 177809344)
        )
    
    ## ip123knownErrors2 = (
    ## (13.5, 1500.,rmn.KIND_ABOVE_SEA, 3., 6.,rmn.KIND_HOURS, 0,0,0), #rp1.v12,rp2.v12 == error
    ## (13.5, 1500.,rmn.KIND_ABOVE_SEA, 3., 3.,rmn.KIND_ABOVE_SEA, 0,0,0), #rp2.kind == level error
    ## )
    

    def test_ConvertIp_toIP(self):
        """convertIp to ip1 should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.ip1knownValues:
            ipnew2 = rmn.convertIp(rmn.CONVIP_P2IP_NEW,lvlnew,kind)
            self.assertEqual(ipnew2,ipnew)
            ipold2 = rmn.convertIp(rmn.CONVIP_P2IP_OLD,lvlold,kind)
            self.assertEqual(ipold2,ipold)

    def test_ConvertIp_fromIP(self):
        """ConvertIp from ip1 should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.ip1knownValues:
            (lvl2,kind2) = rmn.convertIp(rmn.CONVIP_IP2P,ipnew)
            self.assertEqual(kind2,kind)
            self.assertAlmostEqual(lvlnew,lvl2,6)
            (lvl2,kind2) = rmn.convertIp(rmn.CONVIP_IP2P,ipold)
            self.assertEqual(kind2,kind)
            self.assertAlmostEqual(lvlold,lvl2,6)

    def test_ConvertIp_Sanity(self):
        """ConvertIp fromIP then toIP should give back the provided IP"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.ip1knownValues:
            (lvl2,kind2) = rmn.convertIp(rmn.CONVIP_IP2P,ipnew)
            ipnew2 = rmn.convertIp(rmn.CONVIP_P2IP_NEW,lvl2,kind2)
            self.assertEqual(ipnew2,ipnew)
            (lvl2,kind2) = rmn.convertIp(rmn.CONVIP_IP2P,ipold)
            ipold2= rmn.convertIp(rmn.CONVIP_P2IP_OLD,lvl2,kind2)
            self.assertEqual(ipold2,ipold)

    def test_ConvertPKtoIP(self):
        """convertPKtoIP should give known result with known input"""
        for pk1,pk2,pk3,ip1,ip2,ip3 in self.ip123knownValues:
            (ip1b,ip2b,ip3b) = rmn.convertPKtoIP(pk1,pk2,pk3)
            self.assertEqual((ip1b,ip2b,ip3b),(ip1,ip2,ip3))

    def test_ConvertIPtoPK(self):
        """ConvertIPtoPK should give known result with known input"""
        for pk1,pk2,pk3,ip1,ip2,ip3 in self.ip123knownValues:
            pkvalues = rmn.convertIPtoPK(ip1,ip2,ip3)
            self.assertEqual(pkvalues[0].kind,pk1.kind)
            self.assertEqual(pkvalues[1].kind,pk2.kind)
            self.assertAlmostEqual(pkvalues[0].v1,pk1.v1,6)
            self.assertAlmostEqual(pkvalues[1].v1,pk2.v1,6)

    ## def testConvertPKtoIPKnownErrors(self):
    ##     """EncodeIp should raise an exeption for Known Errors"""
    ##     for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownErrors:
    ##         ip123 = rmn.convertPKtoIP((rp1v1,rp1v2,rp1k),(rp2v1,rp2v2,rp2k),(0.,0.,rmn.KIND_ARBITRARY))
    ##         self.assertEqual(ip123,None)
    ##         except:
    ##             self.assertEqual("Except rmn.error","Except rmn.error")

    def test_EncodeIp(self):
        """EncodeIp should give known result with known input"""
        for pk1,pk2,pk3,ip1,ip2,ip3 in self.ip123knownValues2:
            (ip1b,ip2b,ip3b) = rmn.EncodeIp(pk1,pk2,pk3)
            self.assertEqual((ip1b,ip2b,ip3b),(ip1,ip2,ip3))

    def test_DecodeIp(self):
        """DecodeIp should give known result with known input"""
        for pk1,pk2,pk3,ip1,ip2,ip3 in self.ip123knownValues2:
            pkvalues = rmn.DecodeIp(ip1,ip2,ip3)
            self.assertEqual(pkvalues[0].kind,pk1.kind,"DecodeIp(%d,%d,%d) Got: ip1k=%d expecting=%d : (%f,%f,%d)" % (ip1,ip2,ip3,pkvalues[0].kind,pk1.kind,pkvalues[0].v1,pkvalues[0].v2,pkvalues[0].kind))
            self.assertEqual(pkvalues[1].kind,pk2.kind,"DecodeIp(%d,%d,%d) Got: ip2k=%d expecting=%d : (%f,%f,%d)" % (ip1,ip2,ip3,pkvalues[1].kind,pk2.kind,pkvalues[1].v1,pkvalues[1].v2,pkvalues[1].kind))
            self.assertAlmostEqual(pkvalues[0].v1,pk1.v1,6)
            self.assertAlmostEqual(pkvalues[1].v1,pk2.v1,6)
            self.assertAlmostEqual(pkvalues[0].v2,pk1.v2,6)
            self.assertAlmostEqual(pkvalues[1].v2,pk2.v2,6)

    ## def testEncodeIpKnownErrors(self):
    ##     """EncodeIp should raise an exeption for Known Errors"""
    ##     for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownErrors:
    ##         try:
    ##             (ip1b,ip2b,ip3b) = rmn.EncodeIp([(rp1v1,rp1v2,rp1k),(rp2v1,rp2v2,rp2k),(0.,0.,rmn.KIND_ARBITRARY)])
    ##             self.assertEqual((ip1b,ip2b,ip3b),("Except rmn.error",))
    ##         except rmn.error:
    ##             self.assertEqual("Except rmn.error","Except rmn.error")


    ## def testSanity(self):
    ##     """EncodeIp(DecodeIp(n))==n for all n"""
    ##     for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownValues:
    ##         ((rp1v1b,rp1v2b,rp1kb),(rp2v1b,rp2v2b,rp2kb),(rp3v1b,rp3v2b,rp3kb)) = rmn.DecodeIp((ip1,ip2,ip3))
    ##         (ip1b,ip2b,ip3b) = rmn.EncodeIp([(rp1v1b,rp1v2b,rp1kb),(rp2v1b,rp2v2b,rp2kb),(rp3v1b,rp3v2b,rp3kb)])
    ##         self.assertEqual((ip1b,ip2b,ip3b),(ip1,ip2,ip3))


    def test_ip123_all_val(self):
        """ip123_all, ip123_vall should give known result with known input"""
        for pk1,pk2,pk3,ip1,ip2,ip3 in self.ip123knownValues:
            ip1a1 = rmn.ip1_all(pk1.v1,pk1.kind)
            ip1v1 = rmn.ip1_val(pk1.v1,pk1.kind)
            ip1a2 = rmn.ip1_all(pk1.v2,pk1.kind)
            ip1v2 = rmn.ip1_val(pk1.v2,pk1.kind)
            self.assertEqual(ip1a1,ip1)
            self.assertEqual(ip1v1,ip1)
            self.assertEqual(ip1a2,ip1)
            self.assertEqual(ip1v2,ip1)
            
            ip2a1 = rmn.ip2_all(pk2.v1,pk2.kind)
            ip2v1 = rmn.ip2_val(pk2.v1,pk2.kind)
            ip2a2 = rmn.ip2_all(pk2.v2,pk2.kind)
            ip2v2 = rmn.ip2_val(pk2.v2,pk2.kind)
            self.assertEqual(ip2a1,ip2)
            self.assertEqual(ip2v1,ip2)
            self.assertEqual(ip2a2,ip2)
            self.assertEqual(ip2v2,ip2)

            ip3a1 = rmn.ip3_all(pk3.v1,pk3.kind)
            ip3v1 = rmn.ip3_val(pk3.v1,pk3.kind)
            ip3a2 = rmn.ip3_all(pk3.v2,pk3.kind)
            ip3v2 = rmn.ip3_val(pk3.v2,pk3.kind)
            self.assertEqual(ip3a1,ip3)
            self.assertEqual(ip3v1,ip3)
            self.assertEqual(ip3a2,ip3)
            self.assertEqual(ip3v2,ip3)
            
#--- fstd98/????? ---------------------------------------------------


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
