#!/usr/bin/env python
"""Unit test for Fstdc.c"""

import Fstdc
import unittest
import numpy


class FstdcCigaxgKnownValues(unittest.TestCase):

    knownValues = (
        ('Grille_Amer_Nord PS 40km','N',(401,401),(200.5,200.5,40000.0,21.0),(2005,  2005,  2100,   400)),
        ('Grille_Europe PS 40km',   'N',(401,401),(200.5,220.5,40000.0,260.0),( 400,  1000, 29830, 57333)),
        ('Grille_Inde PS 40km',     'N',(401,401),(200.5,300.5,40000.0,190.0),( 400,  1700,  4217, 58335)),
        ('Grille_Hem_Sud PS 40km',  'S',(401,401),(200.5,200.5,40000.0,21.0),(2005,  2005,  2100,   400)),
        ('Grille_Canada PS 20km',   'N',(351,261),(121.5,281.5,20000.0,21.0),( 200,   210, 20548, 37716)),
        ('Grille_Maritimes PS 20km','N',(175,121),(51.5,296.5,20000.0,340.0),( 200,   200, 25513, 54024)),
        ('Grille_Quebec PS 20km',   'N',(199,155),(51.5,279.5,20000.0,0.0),( 200,     0, 23640, 37403)),
        ('Grille_Prairies PS 20km', 'N',(175,121),(86.5,245.5,20000.0,20.0),( 200,   200, 21001, 37054)),
        ('Grille_Colombie PS 20km', 'N',(175,121),(103.5,245.5,20000.0,30.0),( 200,   300, 19774, 37144)),
        ('Grille_USA PS 20km',      'N',(351,261),(121.0,387.5,20000.0,21.0),( 200,   210, 21094, 39002)),
        ('Grille_Global LatLon 0.5','L',(721,359),(-89.5,180.0,0.5,0.5),(  50,    50,    50, 18000))
#,#The Grille_GemLam10 PS is causing a problem, very tiny expected roundoff error
#('Grille_GemLam10 PS 10km', 'N',(1201,776),(536.0,746.0,10000.0,21.0),( 100,   210, 19416, 39622))
        )

    def testCigaxgKnownValues(self):
        """Cigaxg should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            xgout = Fstdc.cigaxg(proj,ig[0],ig[1],ig[2],ig[3])
            self.assertAlmostEqual(xgout[0],xg[0],1,name+'[0]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[1],xg[1],1,name+'[1]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[2],xg[2],1,name+'[2]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[3],xg[3],1,name+'[3]'+xgout.__repr__())

    def testCxgaigKnownValues(self):
        """Cxgaig should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            igout = Fstdc.cxgaig(proj,xg[0],xg[1],xg[2],xg[3])
            self.assertEqual(igout,ig,name+igout.__repr__())

    def testSanity(self):
        """cigaxg(cxgaig(n))==n for all n"""
        for name,proj,dims,xg,ig in self.knownValues:
            xgout = Fstdc.cigaxg(proj,ig[0],ig[1],ig[2],ig[3])
            igout = Fstdc.cxgaig(proj,xgout[0],xgout[1],xgout[2],xgout[3])
            self.assertEqual(igout,ig,name+igout.__repr__()+xgout.__repr__())


class FstdcLevel_to_ip1KnownValues(unittest.TestCase):

#lvlnew,lvlold,ipnew,ipold,kind
#we need to specify 2 levels since the old style ip1 give is an approx of level in some cases
    knownValues = (
    (0.,    0.,    15728640, 12001,0),
    (13.5,  15.,   8523608,  12004,0),
    (1500., 1500., 6441456,  12301,0),
    (5525., 5525., 6843956,  13106,0),
    (12750.,12750.,5370380,  14551,0),
    (0.,    0.,    32505856, 2000, 1),
    (0.1,   0.1,   27362976, 3000,1),
    (0.02,  0.02,  28511552, 2200,1),
    (0.000003,0.,  32805856, 2000,1),
    (1024.,1024.,  39948288, 1024,2),
    (850., 850.,   41744464, 850, 2),
    (650., 650.,   41544464, 650, 2),
    (500., 500.,   41394464, 500, 2),
    (10.,  10.,    42043040, 10,  2),
    (2.,   2.,     43191616, 1840,2),
    (0.3,  0.3,    44340192, 1660,2)
    )

    def testLevels_to_ip1KnownValues(self):
        """level_to_ip1 should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            (ipnew2,ipold2) = Fstdc.level_to_ip1([lvlnew],kind)[0]
            self.assertEqual(ipnew2,ipnew)
            (ipnew2,ipold2) = Fstdc.level_to_ip1([lvlold],kind)[0]
            self.assertEqual(ipold2,ipold)

    def testip1_to_levelsKnownValues(self):
        """ip1_to_level should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            (lvl2,kind2) = Fstdc.ip1_to_level([ipnew])[0]
            self.assertEqual(kind2,kind)
            self.assertAlmostEqual(lvlnew,lvl2,6)
            (lvl2,kind2) = Fstdc.ip1_to_level([ipold])[0]
            self.assertEqual(kind2,kind)
            self.assertAlmostEqual(lvlold,lvl2,6)

    def testSanity(self):
        """levels_to_ip1(ip1_to_levels(n))==n for all n"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            (lvl2,kind2) = Fstdc.ip1_to_level([ipnew])[0]
            (ipnew2,ipold2) = Fstdc.level_to_ip1([lvl2],kind2)[0]
            self.assertEqual(ipnew2,ipnew)
            (lvl2,kind2) = Fstdc.ip1_to_level([ipold])[0]
            (ipnew2,ipold2) = Fstdc.level_to_ip1([lvl2],kind2)[0]
            self.assertEqual(ipold2,ipold)


class Fstdc_ezgetlaloKnownValues(unittest.TestCase):

    la = numpy.array(
        [[-69.5, -69. , -68.5],
        [-69.5, -69. , -68.5],
        [-69.5, -69. , -68.5]]
        ,dtype=numpy.dtype('float32'),order='F')
    lo = numpy.array(
        [[ 180. ,  180. ,  180. ],
        [ 180.5,  180.5,  180.5],
        [ 181. ,  181. ,  181. ]]
        ,dtype=numpy.dtype('float32'),order='F')
    cla = numpy.array(
        [[[-69.75, -69.25, -68.75],
        [-69.75, -69.25, -68.75],
        [-69.75, -69.25, -68.75]],
        [[-69.25, -68.75, -68.25],
        [-69.25, -68.75, -68.25],
        [-69.25, -68.75, -68.25]],
        [[-69.25, -68.75, -68.25],
        [-69.25, -68.75, -68.25],
        [-69.25, -68.75, -68.25]],
        [[-69.75, -69.25, -68.75],
        [-69.75, -69.25, -68.75],
        [-69.75, -69.25, -68.75]]]
        ,dtype=numpy.dtype('float32'),order='F')
    clo = numpy.array(
        [[[ 179.75,  179.75,  179.75],
        [ 180.25,  180.25,  180.25],
        [ 180.75,  180.75,  180.75]],
        [[ 179.75,  179.75,  179.75],
        [ 180.25,  180.25,  180.25],
        [ 180.75,  180.75,  180.75]],
        [[ 180.25,  180.25,  180.25],
        [ 180.75,  180.75,  180.75],
        [ 181.25,  181.25,  181.25]],
        [[ 180.25,  180.25,  180.25],
        [ 180.75,  180.75,  180.75],
        [ 181.25,  181.25,  181.25]]]
        ,dtype=numpy.dtype('float32'),order='F')


    def test_Fstdc_ezgetlalo_KnownValues(self):
        """Fstdc_ezgetlalo should give known result with known input"""
        (ni,nj) = self.la.shape
        grtyp='L'
        grref='L'
        (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grtyp,-69.5,180.0,0.5,0.5)
        hasAxes = 0
        doCorners = 0
        (i0,j0) = (0,0)
        (la2,lo2) = Fstdc.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(None,None),hasAxes,(i0,j0),doCorners)
        if numpy.any(self.la!=la2):
            print("\nLAT Expected:\n",self.la)
            print("LAT Got:\n",la2)
        if numpy.any(self.lo!=lo2):
            print("\nLON Expected:\n",self.lo)
            print("LON Got:\n",lo2)
        self.assertFalse(numpy.any(self.la!=la2))
        self.assertFalse(numpy.any(self.lo!=lo2))

    def test_Fstdc_ezgetlalo_KnownValues2(self):
        """Fstdc_ezgetlalo corners should give known result with known input"""
        (ni,nj) = self.la.shape
        grtyp='L'
        grref='L'
        (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grtyp,-69.5,180.0,0.5,0.5)
        hasAxes = 0
        doCorners = 1
        (i0,j0) = (0,0)
        (la2,lo2,cla2,clo2) = Fstdc.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(None,None),hasAxes,(i0,j0),doCorners)
        if numpy.any(self.la!=la2):
            print("\nLAT Expected:\n",self.la)
            print("LAT Got:\n",la2)
        if numpy.any(self.lo!=lo2):
            print("\nLON Expected:\n",self.lo)
            print("LON Got:\n",lo2)
        self.assertFalse(numpy.any(self.la!=la2))
        self.assertFalse(numpy.any(self.lo!=lo2))
        for ic in range(0,4):
            if numpy.any(self.cla[ic,...]!=cla2[ic,...]):
                print("\n",ic,'cla')
                print("LAT Expected:\n",self.cla[ic,...])
                print("LAT Got:\n",cla2[ic,...])
            self.assertFalse(numpy.any(self.cla[ic,...]!=cla2[ic,...]))
            if numpy.any(self.clo[ic,...]!=clo2[ic,...]):
                print("\n",ic,'clo')
                print("\nLON Expected:\n",self.clo[ic,...])
                print("LON Got:\n",clo2[ic,...])
            self.assertFalse(numpy.any(self.clo[ic,...]!=clo2[ic,...]))

    def test_Fstdc_ezgetlalo_Z_KnownValues(self):
        """Fstdc_ezgetlalo with Z grid should give known result with known input"""
        (ni,nj) = self.la.shape
        grtyp='Z'
        grref='L'
        (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grref,0.,0.,1.,1.)
        xaxis = self.lo[:,0].reshape((self.lo.shape[0],1)).copy('FORTRAN')
        yaxis = self.la[0,:].reshape((1,self.la.shape[1])).copy('FORTRAN')
        hasAxes = 1
        doCorners = 0
        (i0,j0) = (0,0)
        (la2,lo2) = Fstdc.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xaxis,yaxis),hasAxes,(i0,j0),doCorners)
        if numpy.any(self.la!=la2):
            print("\nLAT Expected:\n",self.la)
            print("LAT Got:\n",la2)
        if numpy.any(self.lo!=lo2):
            print("\nLON Expected:\n",self.lo)
            print("LON Got:\n",lo2)
        self.assertFalse(numpy.any(self.la!=la2))
        self.assertFalse(numpy.any(self.lo!=lo2))

    def test_Fstdc_ezgetlalo_Dieze_KnownValues(self):
        """Fstdc_ezgetlalo with #-grid should give known result with known input"""
        (ni,nj) = self.la.shape
        grtyp='#'
        grref='L'
        (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grref,0.,0.,1.,1.)
        xaxis = self.lo[:,0].reshape((self.lo.shape[0],1)).copy('FORTRAN')
        yaxis = self.la[0,:].reshape((1,self.la.shape[1])).copy('FORTRAN')
        hasAxes = 1
        doCorners = 0
        (i0,j0) = (2,2)
        (la2,lo2) = Fstdc.ezgetlalo((ni-1,nj-1),grtyp,(grref,ig1,ig2,ig3,ig4),(xaxis,yaxis),hasAxes,(i0,j0),doCorners)
        if numpy.any(self.la[1:,1:]!=la2):
            print("\nLAT Expected:\n",self.la[1:,1:])
            print("LAT Got:\n",la2)
        if numpy.any(self.lo[1:,1:]!=lo2):
            print("\nLON Expected:\n",self.lo[1:,1:])
            print("LON Got:\n",lo2)
        self.assertFalse(numpy.any(self.la[1:,1:]!=la2))
        self.assertFalse(numpy.any(self.lo[1:,1:]!=lo2))

class FstdcInterpTests(unittest.TestCase):

    epsilon = 1.e-5

    def gridL(self,dlalo=0.5,nij=10):
        """provide grid and rec values for other tests"""
        grtyp='L'
        grref=grtyp
        la0 = 0.-dlalo*(nij/2.)
        lo0 = 180.-dlalo*(nij/2.)
        ig14 = (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grtyp,la0,lo0,dlalo,dlalo)
        axes = (None,None)
        hasAxes = 0
        ij0 = (1,1)
        doCorners = 0
        (la,lo) = Fstdc.ezgetlalo((nij,nij),grtyp,(grref,ig1,ig2,ig3,ig4),axes,hasAxes,ij0,doCorners)
        return (grtyp,ig14,(nij,nij),la,lo)

    def test_Fstdc_exinterp_KnownValues(self):
        """Fstdc_exinterp should give known result with known input"""
        (g1_grtyp,g1_ig14,g1_shape,la1,lo1) = self.gridL(0.5,6)
        (g2_grtyp,g2_ig14,g2_shape,la2,lo2) = self.gridL(0.25,8)
        axes = (None,None)
        ij0  = (1,1)
        g1ig14 = list(g1_ig14)
        g1ig14.insert(0,g1_grtyp)
        g2ig14 = list(g2_ig14)
        g2ig14.insert(0,g2_grtyp)
        la2b = Fstdc.ezinterp(la1,None,
            g1_shape,g1_grtyp,g1ig14,axes,0,ij0,
            g2_shape,g2_grtyp,g2ig14,axes,0,ij0,
            0)
        if numpy.any(numpy.abs(la2-la2b)>self.epsilon):
                print('g1:'+repr((g1_grtyp,g1_ig14,g1_shape)))
                print('g2:'+repr((g2_grtyp,g2_ig14,g2_shape)))
                print('la2:',la2)
                print('la2b:',la2b)
        self.assertFalse(numpy.any(numpy.abs(la2-la2b)>self.epsilon))


class FstdcConvertIp2PKnownValues(unittest.TestCase):

#lvlnew,lvlold,ipnew,ipold,kind
#we need to specify 2 levels since the old style ip1 give is an approx of level in some cases
    knownValues = (
    (0.,    0.,    15728640, 12001,Fstdc.KIND_ABOVE_SEA),
    (13.5,  15.,   8523608,  12004,Fstdc.KIND_ABOVE_SEA),
    (1500., 1500., 6441456,  12301,Fstdc.KIND_ABOVE_SEA),
    (5525., 5525., 6843956,  13106,Fstdc.KIND_ABOVE_SEA),
    (12750.,12750.,5370380,  14551,Fstdc.KIND_ABOVE_SEA),
    (0.,    0.,    32505856, 2000, Fstdc.KIND_SIGMA),
    (0.1,   0.1,   27362976, 3000, Fstdc.KIND_SIGMA),
    (0.02,  0.02,  28511552, 2200, Fstdc.KIND_SIGMA),
    (0.000003,0.,  32805856, 2000, Fstdc.KIND_SIGMA),
    (1024.,1024.,  39948288, 1024, Fstdc.KIND_PRESSURE),
    (850., 850.,   41744464, 850,  Fstdc.KIND_PRESSURE),
    (650., 650.,   41544464, 650,  Fstdc.KIND_PRESSURE),
    (500., 500.,   41394464, 500,  Fstdc.KIND_PRESSURE),
    (10.,  10.,    42043040, 10,   Fstdc.KIND_PRESSURE),
    (2.,   2.,     43191616, 1840, Fstdc.KIND_PRESSURE),
    (0.3,  0.3,    44340192, 1660, Fstdc.KIND_PRESSURE)
    )
    #TODO: add values for other kinds:
    #define KIND_ARBITRARY 3
    #define KIND_ABOVE_GND 4
    #define KIND_HYBRID 5
    #define KIND_THETA 6
    #define KIND_HOURS 10

    def testConvertIp2PKnownValues(self):
        """ConvertIp2P should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            (lvl2,kind2) = Fstdc.ConvertIp2P(ipnew,Fstdc.CONVIP_IP2P_DEFAULT)
            self.assertEqual(kind2,kind)
            self.assertAlmostEqual(lvlnew,lvl2,6)

    def testConvertP2IpKnownValues(self):
        """ConvertP2Ip should give known result with known input"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            ipnew2 = Fstdc.ConvertP2Ip(lvlnew,kind,Fstdc.CONVIP_STYLE_NEW)
            self.assertEqual(ipnew2,ipnew)

    def testSanity(self):
        """ConvertIp2P(ConvertP2Ip(n))==n for all n"""
        for lvlnew,lvlold,ipnew,ipold,kind in self.knownValues:
            (lvl2,kind2) = Fstdc.ConvertIp2P(ipnew,Fstdc.CONVIP_IP2P_DEFAULT)
            ipnew2 = Fstdc.ConvertP2Ip(lvl2,kind2,Fstdc.CONVIP_STYLE_NEW)
            self.assertEqual(ipnew2,ipnew)


class FstdcEncodeIpKnownValues(unittest.TestCase):

    #rp1.v1,rpn1.v2,rp1.kind, rp2.v1,rpn2.v2,rp2.kind, ip1,ip2,ip3
    knownValues = (
    (5525.,5525.,Fstdc.KIND_ABOVE_SEA, 2., 2.,Fstdc.KIND_HOURS, 6843956, 177409344, 66060288), #rp1,    rp2
    (13.5, 1500.,Fstdc.KIND_ABOVE_SEA, 3., 3.,Fstdc.KIND_HOURS, 8523608, 177509344, 6441456), #rp1.v12,rp2
    (1500.,1500.,Fstdc.KIND_ABOVE_SEA, 6.,12.,Fstdc.KIND_HOURS, 6441456, 176280768, 177809344)   #rp1,    rp2.v12
    )
    
    knownErrors = (
   (13.5, 1500.,Fstdc.KIND_ABOVE_SEA, 3., 6.,Fstdc.KIND_HOURS, 0,0,0), #rp1.v12,rp2.v12 == error
   (13.5, 1500.,Fstdc.KIND_ABOVE_SEA, 3., 3.,Fstdc.KIND_ABOVE_SEA, 0,0,0), #rp2.kind == level error
    )

    def testEncodeIpKnownValues(self):
        """EncodeIp should give known result with known input"""
        for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownValues:
            (ip1b,ip2b,ip3b) = Fstdc.EncodeIp([(rp1v1,rp1v2,rp1k),(rp2v1,rp2v2,rp2k),(0.,0.,Fstdc.KIND_ARBITRARY)])
            self.assertEqual((ip1b,ip2b,ip3b),(ip1,ip2,ip3))
        
    def testEncodeIpKnownErrors(self):
        """EncodeIp should raise an exeption for Known Errors"""
        for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownErrors:
            try:
                (ip1b,ip2b,ip3b) = Fstdc.EncodeIp([(rp1v1,rp1v2,rp1k),(rp2v1,rp2v2,rp2k),(0.,0.,Fstdc.KIND_ARBITRARY)])
                self.assertEqual((ip1b,ip2b,ip3b),("Except Fstdc.error",))
            except Fstdc.error:
                self.assertEqual("Except Fstdc.error","Except Fstdc.error")

    def testDecodeIpKnownValues(self):
        """DecodeIp should give known result with known input"""
        for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownValues:
            ((rp1v1b,rp1v2b,rp1kb),(rp2v1b,rp2v2b,rp2kb),(rp3v1b,rp3v2b,rp3kb)) = Fstdc.DecodeIp((ip1,ip2,ip3))
            self.assertEqual(rp1kb,rp1k)
            self.assertEqual(rp2kb,rp2k)
            self.assertAlmostEqual(rp1v1b,rp1v1,6)
            self.assertAlmostEqual(rp1v2b,rp1v2,6)
            self.assertAlmostEqual(rp2v1b,rp2v1,6)
            self.assertAlmostEqual(rp2v2b,rp2v2,6)


    def testSanity(self):
        """EncodeIp(DecodeIp(n))==n for all n"""
        for rp1v1,rp1v2,rp1k,rp2v1,rp2v2,rp2k,ip1,ip2,ip3 in self.knownValues:
            ((rp1v1b,rp1v2b,rp1kb),(rp2v1b,rp2v2b,rp2kb),(rp3v1b,rp3v2b,rp3kb)) = Fstdc.DecodeIp((ip1,ip2,ip3))
            (ip1b,ip2b,ip3b) = Fstdc.EncodeIp([(rp1v1b,rp1v2b,rp1kb),(rp2v1b,rp2v2b,rp2kb),(rp3v1b,rp3v2b,rp3kb)])
            self.assertEqual((ip1b,ip2b,ip3b),(ip1,ip2,ip3))

if __name__ == "__main__":
    unittest.main()

# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
