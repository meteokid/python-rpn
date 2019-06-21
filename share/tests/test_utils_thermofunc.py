#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import rpnpy.utils.tdpack as tdpack

class RpnPyUtilsTDPack(unittest.TestCase):

    t0 = 293.15  # K
    p0 = 10100.  # Pa
    tt = [220.0, 240.0, 260.0, 280.0, 300.0, 320.0]  # K
    pp = [100000., 85000., 50000., 25000., 10000., 5000.]  # Pa
    hr = [0.1, 0.25, 0.5, 0.75, 0.9]  #

    # Fonction de tension de vapeur saturante (tetens) - ew ou ei selon tt
    ## FOEWF  = lambda TTT:
    def test_FOEWF(self):
        "function should give known values for known input"
        s = [-5.4764764057643429, -3.1220409744340221, -1.1408219069509415, 0.4838205947407207, 1.7547511168319814, 2.8467655381150121]
        a = [tdpack.FOEWF(t) for t in self.tt]
        ## print('FOEWF',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction de tension de vapeur saturante (tetens) - ew ou ei selon tt
    ## FOEW   = lambda TTT
    def test_FOEW(self):
        "function should give known values for known input"
        s = [2.555532004964181, 26.915325738580361, 195.17857757272182, 990.84431550342617, 3531.535162787357, 10524.933781438713]
        a = [tdpack.FOEW(t) for t in self.tt]
        ## print('FOEW',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant la derivee selon t de  ln ew (ou ln ei)
    ## FODLE  = lambda TTT
    def test_FODLE(self):
        "function should give known values for known input"
        s = [0.12880976091265736, 0.10758814760411922, 0.091209526827625315, 0.06875220877373836, 0.058734893902645725, 0.050757448582160719]
        a = [tdpack.FODLE(t) for t in self.tt]
        ## print('FODLE',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant l'humidite specifique saturante (qsat)
    ## FOQST  = lambda TTT,PRS
    def test_FOQST(self):
        "function should give known values for known input"
        s = [1.5895052082187768e-05, 1.8700093153019875e-05, 3.1790411276115593e-05, 6.358205103479476e-05, 0.00015896434191834202, 0.00031795940302288252, 0.00016742498368408612, 0.00019697410608077363, 0.00033488404371558961, 0.00066990443444308971, 0.0016757845219049303, 0.0033549860613801331, 0.0012148681066404448, 0.0014294428504633822, 0.0024315315559717537, 0.0048702604253056617, 0.012229951898473059, 0.024643075325838218, 0.0061860239988634404, 0.0072825070284055007, 0.012418738376316497, 0.025026368783025617, 0.064026714808628024, 0.13323818466044421, 0.022262647491711043, 0.026254037867988262, 0.04513600956221922, 0.092818232359002231, 0.25349590278600853, 0.59932846691999664, 0.068175437435684216, 0.080797189297146632, 0.14224477217235201, 0.31141168136869024, 1.0, 1.0]
        a = [tdpack.FOQST(t, p) for t in self.tt for p in self.pp]
        ## print('FOQST',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant la derivee de qsat selon t, qst est la sortie de foqst
    ## FODQS  = lambda QST,TTT
    def test_FODQS(self):
        "function should give known values for known input"
        s = [2.0474576377046686e-06, 2.408781904377319e-06, 4.0949943945354455e-06, 8.1903052793338076e-06, 2.0478137150507808e-05, 4.0964189310028255e-05, 1.8014776776239716e-05, 2.1194616202457258e-05, 3.6036887093920062e-05, 7.2103121790972446e-05, 0.00018047818057684368, 0.00036169274638212255, 0.00011088936088931564, 0.00013049207529095699, 0.00022210658939594038, 0.00044552901913850871, 0.0011237795286534798, 0.0022813474424057123, 0.0004269018123791746, 0.00050290453056647116, 0.00086026005555308606, 0.0017467891515763952, 0.0045732741378155693, 0.0099022118993143272, 0.0013252866944477323, 0.0015666333185327664, 0.0027237832467447358, 0.0057592086632444585, 0.017182965057688456, 0.048023746218384518, 0.0036037930399062594, 0.0043024457984174157, 0.0078441629736064958, 0.018798092538224384, 0.081606236178899755, 0.081606236178899755]
        a = [tdpack.FODQS(tdpack.FOQST(t, p), t)
             for t in self.tt for p in self.pp]
        ## print('FODQS',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant tension vap (eee) fn de hum sp (qqq) et prs
    ## FOEFQ  = lambda QQQ,PRS
    def test_FOEFQ(self):
        "function should give known values for known input"
        s = [30257.39478680777, 25718.785568786607, 15128.697393403885, 7564.3486967019426, 3025.7394786807772, 1512.8697393403886, 64565.996140437732, 54881.096719372072, 32282.998070218866, 16141.499035109433, 6456.5996140437728, 3228.2998070218864, 100000.0, 85000.0, 50000.0, 25000.0, 10000.0, 5000.0, 100000.0, 85000.0, 50000.0, 25000.0, 10000.0, 5000.0, 100000.0, 85000.0, 50000.0, 25000.0, 10000.0, 5000.0]
        qs0 = tdpack.FODQS(self.t0, self.p0)
        a = [tdpack.FOEFQ(qs0*hr, p) for hr in self.hr for p in self.pp]
        ## print('FOEFQ',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## # Fonction calculant hum sp (qqq) de tens. vap (eee) et pres (prs)
    ## ## FOQFE  = lambda EEE,PRS:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)


    # Fonction calculant temp virt. (tvi) de temp (ttt) et hum sp (qqq)
    ## FOTVT  = lambda TTT,QQQ:
    def test_FOTVT(self):
        "function should give known values for known input"
        s = [248.41325948241322, 270.9962830717235, 293.5793066610338, 316.1623302503441, 338.7453538396544, 361.3283774289647, 291.0331487060331, 317.49070767930885, 343.94826665258461, 370.40582562586036, 396.86338459913605, 423.32094357241181, 362.06629741206626, 394.98141535861771, 427.89653330516921, 460.81165125172066, 493.72676919827217, 526.64188714482361, 433.09944611809942, 472.47212303792662, 511.84479995775388, 551.21747687758113, 590.59015379740833, 629.96283071723553, 475.71933534171927, 518.96654764551192, 562.21375994930463, 605.46097225309722, 648.70818455688993, 691.95539686068264]
        qs0 = tdpack.FODQS(self.t0, self.p0)
        a = [tdpack.FOTVT(t, qs0*hr) for hr in self.hr for t in self.tt]
        ## print('FOTVT',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## # Fonction calculant temp virt. (tvi) de temp (ttt), hum sp (qqq) et masse sp des hydrometeores.
    ## ## FOTVHT = lambda TTT,QQQ,QQH:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    ## # Fonction calculant ttt de temp virt. (tvi) et hum sp (qqq)
    ## FOTTV  = lambda TVI,QQQ:
    def test_FOTTV(self):
        "function should give known values for known input"
        s = [220.0, 240., 260.0, 280.0, 300.0, 320.0, 220., 240.0, 260.0, 280.0, 300.0, 320.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 220.0, 240., 260.0, 280.0, 300.0, 320.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0]
        qs0 = tdpack.FODQS(self.t0, self.p0)
        a = [tdpack.FOTTV(tdpack.FOTVT(t, qs0*hr), qs0*hr)
             for hr in self.hr for t in self.tt]
        ## print('FOTTV',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## # Fonction calculant ttt de temp virt. (tvi), hum sp (qqq) et masse sp des hydrometeores (qqh)
    ## ## FOTTVH = lambda TVI,QQQ,QQH:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    ## # Fonction calculant hum rel de hum sp (qqq), temp (ttt) et pres (prs), hr = e/esat
    ## ## FOHR   = lambda QQQ,TTT,PRS:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    # Fonction calculant la chaleur latente de condensation
    ## FOLV   = lambda TTT:
    def test_FOLV(self):
        "function should give known values for known input"
        s = [2624171.7200000002, 2577831.7200000002, 2531491.7200000002, 2485151.7200000002, 2438811.7200000002, 2392471.7200000002]
        a = [tdpack.FOLV(t) for t in self.tt]
        ## print('FOLV',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant la chaleur latente de sublimation
    ## FOLS   = lambda TTT:
    def test_FOLS(self):
        "function should give known values for known input"
        s = [2827118.4983999999, 2834885.2664000001, 2836860.0344000002, 2833042.8023999999, 2823433.5704000001, 2808032.3383999998]
        a = [tdpack.FOLS(t) for t in self.tt]
        ## print('FOLS',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## # Fonction resolvant l'eqn. de poisson pour la temperature; note: si pf=1000*100, "fopoit" donne le theta standard
    ## ## FOPOIT = lambda T00,PR0,PF:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    ## # Fonction resolvant l'eqn. de poisson pour la pression
    ## ## FOPOIP = lambda T00,TF,PR0:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    # Fonction de vapeur saturante (tetens) (No ice)
    ## FOEWAF = lambda TTT:
    def test_FOEWAF(self):
        "function should give known values for known input"
        s = [-4.9854460736396238, -2.8051339276966805, -1.0139200499687713, 0.4838205947407207, 1.7547511168319814, 2.8467655381150121]
        a = [tdpack.FOEWAF(t) for t in self.tt]
        ## print('FOEWAF',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction de vapeur saturante (tetens) (No ice)
    ## FOEWA  = lambda TTT:
    def test_FOEWA(self):
        "function should give known values for known input"
        s = [4.1757365221141107, 36.951376552631444, 221.58733016595176, 990.84431550342617, 3531.535162787357, 10524.933781438713]
        a = [tdpack.FOEWA(t) for t in self.tt]
        ## print('FOEWA',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant la derivee selon t de ln ew (No ice)
    ## FODLA  = lambda TTT:
    def test_FODLA(self):
        "function should give known values for known input"
        s = [0.12085612074312818, 0.098335132397847941, 0.081569198045724864, 0.068752270849755381, 0.058734946934080469, 0.05075749441080097]
        a = [tdpack.FODLA(t) for t in self.tt]
        ## print('FODLA',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant l'humidite specifique saturante (No ice)
    ## FOQSA  = lambda TTT,PRS:
    def test_FOQSA(self):
        "function should give known values for known input"
        s = [2.5972656924603129e-05, 3.0556152088960495e-05, 5.1946133838016597e-05, 0.00010389554778660779, 0.00025976347340293544, 0.00051960898064718809, 0.00022986228803738773, 0.00027043288833447312, 0.00045978880999674278, 0.00091983466341064111, 0.0023015166414534294, 0.004609480977579343, 0.0013793843586912054, 0.0016230452468988122, 0.0027610834623508556, 0.0055314492499933948, 0.013898711022691149, 0.028034232852785213, 0.0061860239988634404, 0.0072825070284055007, 0.012418738376316497, 0.025026368783025617, 0.064026714808628024, 0.13323818466044421, 0.022262647491711043, 0.026254037867988262, 0.04513600956221922, 0.092818232359002231, 0.25349590278600853, 0.59932846691999664, 0.068175437435684216, 0.080797189297146632, 0.14224477217235201, 0.31141168136869024, 1.0, 1.0]
        a = [tdpack.FOQSA(t, p) for t in self.tt for p in self.pp]
        ## print('FOQSA',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    # Fonction calculant la derivee de qsat selon t (No ice)
    ## FODQA  = lambda QST,TTT:
    def test_FODQA(self):
        "function should give known values for known input"
        s = [1.9210328916506838e-06, 2.2600464018925237e-06, 3.8421400170441395e-06, 7.6845769819682996e-06, 1.9213669822229414e-05, 3.8434765923943679e-05, 1.6465433217862969e-05, 1.9371793611124602e-05, 3.2937569262984963e-05, 6.5901962116707787e-05, 0.00016495632815660767, 0.00033058570943799294, 9.9168985457392714e-05, 0.00011669980431892605, 0.00019863118478771351, 0.00039843913307331788, 0.0010050024171894237, 0.0020402219791402065, 0.0004269021978266336, 0.00050290498463639497, 0.00086026083227748234, 0.0017467907287433699, 0.0045732782670013551, 0.0099022208399707125, 0.001325287891042298, 0.0015666347330379621, 0.0027237857060346724, 0.0057592138632044498, 0.017182980572099559, 0.048023789578778897, 0.0036037962937525295, 0.0043024496830736825, 0.0078441700560610526, 0.018798109510925819, 0.081606309860752263, 0.081606309860752263]
        a = [tdpack.FODQA(tdpack.FOQST(t, p), t)
             for t in self.tt for p in self.pp]
        ## print('FODQA',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## # Fonction calculant l'humidite relative (No ice)
    ## ## FOHRA  = lambda QQQ,TTT,PRS:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = self.tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

    # saturation calculations in presence of liquid phase only, function for saturation vapor pressure (tetens) (mixed-phase mode)
    ## FESIF  = lambda TTT:
    def test_FESIF(self):
        "function should give known values for known input"
        s = [-5.476476405764342, -3.1220409744340216, -1.1408219069509415, 0.54940515532055323, 2.0083635492919187, 3.2804796055580447]
        a = [tdpack.FESIF(t) for t in self.tt]
        ## print('FESIF',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)
    ## FESI   = lambda TTT:
    def test_FESI(self):
        "function should give known values for known input"
        s = [2.5555320049641836, 26.915325738580375, 195.17857757272182, 1058.0067413523605, 4550.9913992611328, 16239.737250429776]
        a = [tdpack.FESI(t) for t in self.tt]
        ## print('FESI',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)
    ## FDLESI = lambda TTT:
    def test_FDLESI(self):
        "function should give known values for known input"
        s = [0.128809816359444, 0.10758819391595946, 0.091209566089208757, 0.0783050409219338, 0.067957297840555805, 0.059532946130633153]
        a = [tdpack.FDLESI(t) for t in self.tt]
        ## print('FDLESI',a)
        for a1, s1 in zip(a, s):
            self.assertAlmostEqual(a1, s1, places=6, msg=None,
                                   delta=None)

    ## ## FESMX  = lambda TTT,FFF:
    ## ## FESMXX = lambda FFF,FESI8,FOEWA8:\
    ## ## FDLESMX  = lambda TTT,FFF,DDFF:
    ## ## FDLESMXX = lambda TTT,FFF,DDFF,FOEWA8,FESI8,FESMX8:
    ## ## FQSMX  = lambda TTT,PRS,FFF:
    ## ## FQSMXX = lambda FESMX8,PRS:
    ## ## FDQSMX = lambda QSM,DLEMX:
    ## def test_(self):
    ##     "function should give known values for known input"
    ##     s = tt  # []
    ##     a = [tdpack.(t) for t in self.tt]
    ##     for a1, s1 in zip(a, s):
    ##         self.assertAlmostEqual(a1, s1, places=6, msg=None,
    ##                                delta=None)

class RpnPyUtilsThermofunc(unittest.TestCase):

    @unittest.skip('Need some thermofunc tests')
    def test_thermofunc(self):
        """
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.thermofunc as thermofunc
        self.assertTrue(False,'Need some thermofunc tests')


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
