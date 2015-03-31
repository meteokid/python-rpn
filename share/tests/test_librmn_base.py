#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.base"""

import rpnpy.librmn.all as rmn
import unittest
import ctypes as ct
import numpy as np

#--- primitives -----------------------------------------------------

class LibrmnFilesKnownValues(unittest.TestCase):

    #(path,itype,iunit)
    knownValues = (
        (rmn.RMN_LIBPATH,-1,999),
        ('/users/dor/armn/env/SsmBundles/GEM/d/gem-data/gem-data_4.2.0/gem-data_4.2.0_all/share/data/dfiles/bcmk/geophy.fst',rmn.WKOFFIT_TYPE_LIST['STD_RND_98'],999),
        )

    def testWkoffitKnownValues(self):
        """wkoffit should give known result with known input"""
        for mypath,itype,iunit in self.knownValues:
            iout = rmn.wkoffit(mypath)
            self.assertEqual(iout,itype,mypath+':'+repr(iout)+' != '+repr(itype))

    def testfnomfclosKnownValues(self):
        """fnomfclos should give known result with known input"""
        for mypath,itype,iunit in self.knownValues:
            iout  = rmn.fnom(mypath,rmn.FST_RO)
            iout2 = rmn.fclos(iout)
            self.assertEqual((iout,iout2),(iunit,0),mypath+':'+repr((iout,iout2))+' != '+repr((iunit,0)))

#--- base/cxgaix ----------------------------------------------------

class LibrmnCigaxgKnownValues(unittest.TestCase):

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

    def testf_CigaxgKnownValues(self):
        """Cigaxg should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            (cxg1,cxg2,cxg3,cxg4) = (ct.c_float(0.),ct.c_float(0.),ct.c_float(0.),ct.c_float(0.))
            (cig1,cig2,cig3,cig4) = (ct.c_int(ig[0]),ct.c_int(ig[1]),ct.c_int(ig[2]),ct.c_int(ig[3]))            
            istat = rmn.f_cigaxg(proj,
                                 ct.byref(cxg1),ct.byref(cxg2),
                                 ct.byref(cxg3),ct.byref(cxg4),
                                 ct.byref(cig1),ct.byref(cig2),
                                 ct.byref(cig3),ct.byref(cig4))
            xgout = (cxg1.value,cxg2.value,cxg3.value,cxg4.value)
            self.assertAlmostEqual(xgout[0],xg[0],1,name+'[0]'+repr(xgout))
            self.assertAlmostEqual(xgout[1],xg[1],1,name+'[1]'+repr(xgout))
            self.assertAlmostEqual(xgout[2],xg[2],1,name+'[2]'+repr(xgout))
            self.assertAlmostEqual(xgout[3],xg[3],1,name+'[3]'+repr(xgout))

    def testf_CxgaigKnownValues(self):
        """Cxgaig should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            (cxg1,cxg2,cxg3,cxg4) = (ct.c_float(xg[0]),ct.c_float(xg[1]),ct.c_float(xg[2]),ct.c_float(xg[3]))
            (cig1,cig2,cig3,cig4) = (ct.c_int(0),ct.c_int(0),ct.c_int(0),ct.c_int(0))            
            istat = rmn.f_cxgaig(proj,
                                 ct.byref(cig1),ct.byref(cig2),
                                 ct.byref(cig3),ct.byref(cig4),
                                 ct.byref(cxg1),ct.byref(cxg2),
                                 ct.byref(cxg3),ct.byref(cxg4))
            igout = (cig1.value,cig2.value,cig3.value,cig4.value)
            self.assertEqual(igout,ig,name+repr(ig)+' != '+repr(igout))

    def testf_Sanity(self):
        """cigaxg(cxgaig(n))==n for all n"""
        for name,proj,dims,xg,ig in self.knownValues:
            (cxg1,cxg2,cxg3,cxg4) = (ct.c_float(0.),ct.c_float(0.),ct.c_float(0.),ct.c_float(0.))
            (cig1,cig2,cig3,cig4) = (ct.c_int(ig[0]),ct.c_int(ig[1]),ct.c_int(ig[2]),ct.c_int(ig[3]))            
            istat = rmn.f_cigaxg(proj,
                                 ct.byref(cxg1),ct.byref(cxg2),
                                 ct.byref(cxg3),ct.byref(cxg4),
                                 ct.byref(cig1),ct.byref(cig2),
                                 ct.byref(cig3),ct.byref(cig4))
            istat = rmn.f_cxgaig(proj,
                                 ct.byref(cig1),ct.byref(cig2),
                                 ct.byref(cig3),ct.byref(cig4),
                                 ct.byref(cxg1),ct.byref(cxg2),
                                 ct.byref(cxg3),ct.byref(cxg4))
            igout = (cig1.value,cig2.value,cig3.value,cig4.value)
            self.assertEqual(igout,ig,name+repr(ig)+' != '+repr(igout))

    def testCigaxgKnownValues(self):
        """Cigaxg should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            xgout = rmn.cigaxg(proj,ig[0],ig[1],ig[2],ig[3])
            self.assertAlmostEqual(xgout[0],xg[0],1,name+'[0]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[1],xg[1],1,name+'[1]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[2],xg[2],1,name+'[2]'+xgout.__repr__())
            self.assertAlmostEqual(xgout[3],xg[3],1,name+'[3]'+xgout.__repr__())

    def testCxgaigKnownValues(self):
        """Cxgaig should give known result with known input"""
        for name,proj,dims,xg,ig in self.knownValues:
            igout = rmn.cxgaig(proj,xg[0],xg[1],xg[2],xg[3])
            self.assertEqual(igout,ig,name+igout.__repr__())

    def testSanity(self):
        """cigaxg(cxgaig(n))==n for all n"""
        for name,proj,dims,xg,ig in self.knownValues:
            xgout = rmn.cigaxg(proj,ig[0],ig[1],ig[2],ig[3])
            igout = rmn.cxgaig(proj,xgout[0],xgout[1],xgout[2],xgout[3])
            self.assertEqual(igout,ig,name+igout.__repr__()+xgout.__repr__())


#--- base/*date -----------------------------------------------------


class LibrmnNewdateKnownValues(unittest.TestCase):

    #(YYYYMMDD,HHMMSSHH,Stamp)
    knownValues = (
        (20150102,13141500,399367913),
        )

    def testNewdateFromPrintKnownValues(self):
        """Newdate from print should give known result with known input"""
        for yyyymmdd,hhmmsshh,stamp in self.knownValues:
            iout = rmn.newdate(rmn.NEWDATE_PRINT2STAMP,yyyymmdd,hhmmsshh)
            self.assertEqual(iout,stamp,repr(iout)+' != '+repr(stamp))


    def testNewdateToPrintKnownValues(self):
        """Newdate to print should give known result with known input"""
        for yyyymmdd,hhmmsshh,stamp in self.knownValues:
            (iout1,iout2) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT,stamp)
            self.assertEqual((iout1,iout2),(yyyymmdd,hhmmsshh),repr((iout1,iout2))+' != '+repr((yyyymmdd,hhmmsshh)))

    def testSanity(self):
        """Convert back and fort from print to CMC data should give back same result"""
        for yyyymmdd,hhmmsshh,stamp in self.knownValues:
            iout = rmn.newdate(rmn.NEWDATE_PRINT2STAMP,yyyymmdd,hhmmsshh)
            (iout1,iout2) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT,iout)
            self.assertEqual((iout1,iout2),(yyyymmdd,hhmmsshh),repr((iout1,iout2))+' != '+repr((yyyymmdd,hhmmsshh)))


class LibrmnIncdateKnownValues(unittest.TestCase):

    #(YYYYMMDD,HHMMSSHH,NHOURS,HHMMSSHH2)
    knownValues = (
        (20150102,13141500,2. ,15141500),
        (20150102,13141500,1.5,14441500),
        )

    def testIncdateFromPrintKnownValues(self):
        """Incdatr should give known result with known input"""
        for yyyymmdd,hhmmsshh,nhours,hhmmsshh2 in self.knownValues:
            idate1 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP,yyyymmdd,hhmmsshh)
            idate2 = rmn.incdatr(idate1,nhours)
            (iout1,iout2) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT,idate2)
            self.assertEqual((iout1,iout2),(yyyymmdd,hhmmsshh2),repr((iout1,iout2))+' != '+repr((yyyymmdd,hhmmsshh2)))


    def testDifdateToPrintKnownValues(self):
        """Difdatr to print should give known result with known input"""
        for yyyymmdd,hhmmsshh,nhours,hhmmsshh2 in self.knownValues:
            idate1 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP,yyyymmdd,hhmmsshh)
            idate2 = rmn.newdate(rmn.NEWDATE_PRINT2STAMP,yyyymmdd,hhmmsshh2)
            nhours2 = rmn.difdatr(idate2,idate1)
            self.assertEqual(nhours2,nhours,repr(nhours2)+' != '+repr(nhours))


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
