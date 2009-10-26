"""Unit test for scripc.c"""

import scripc
import Fstdc
import unittest
import numpy

#class ScripcKnownValues(unittest.TestCase):
    #"""Check good behaviour of all class.fn on good input values"""
    #pass

#class ScripcBadInput(unittest.TestCase):
    #"""Check good behaviour of all class.fn on bad input values"""
    #pass

#class ScripcSanityCheck(unittest.TestCase):
    #"""Check that revesefn(fn(value)) == value"""

    #def testSanity(self):
        #"""cigaxg(cxgaig(n))==n for all n"""
        #self.assertTrue(expr[, msg])
        #self.assertFalse(expr[, msg])
        #self.assertEqual(first, second[, msg])
        #self.assertNotEqual(first, second[, msg])
        #self.assertAlmostEqual(first, second[, places[, msg]])
        #self.assertNotAlmostEqual(first, second[, places[, msg]])
        #self.assertRaises(exception, callable, ...)

#class ScripcTests(unittest.TestCase):
class ScripcTests:

    epsilon = 0.1

    def gridL(self,dlalo=0.5,ni=10,nj=10):
        """provide grid and rec values for other tests"""
        grtyp='L'
        grref=grtyp
        la0 = 0.-dlalo*(nj/2.)
        lo0 = 180.-dlalo*(ni/2.)
        ig14 = (ig1,ig2,ig3,ig4) =  Fstdc.cxgaig(grtyp,la0,lo0,dlalo,dlalo)
        axes = (None,None)
        hasAxes = 0
        ij0 = (1,1)
        doCorners = 1
        (la,lo,lac,loc) = Fstdc.ezgetlalo((ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),axes,hasAxes,ij0,doCorners)
        la  *= (numpy.pi/180.)
        lo  *= (numpy.pi/180.)
        lac *= (numpy.pi/180.)
        loc *= (numpy.pi/180.)
        return (la,lo,lac,loc)

    def test_Fstdc_exinterp_KnownValues(self):
        """Fstdc_exinterp should give known result with known input"""
        (la1,lo1,lac1,loc1) = self.gridL(0.5,19,17)
        (la2,lo2,lac2,loc2) = self.gridL(0.25,25,21)
        nbins = -1 #use default
        scripc.initOptions(nbins,scripc.TYPE_DISTWGT,scripc.NORM_FRACAREA,scripc.RESTRICT_LALO,scripc.REMAP_ONEWAY)
        scripc.setGridLatLonRad(scripc.INPUT_GRID,la1,lo1,lac1,loc1)
        scripc.setGridLatLonRad(scripc.OUTPUT_GRID,la2,lo2,lac2,loc2)
        (fromAddr,toAddr,weights) = scripc.getAddrWeights(scripc.MAPPING_FORWARD)
        for item in (la1,fromAddr,toAddr,weights):
            print '--'
            #print item.shape,,item.dtype
            print item.flags
        la1  *= (180./numpy.pi)
        la2  *= (180./numpy.pi)
        la2b = scripc.interp_o1(la1,fromAddr,toAddr,weights,la2.size)
        print '--',la2b.shape
        print la2b.flags
        la2b = la2b.reshape(la2.shape, order='Fortran')
        print '--',la2b.shape
        print la2b.flags
        if numpy.any(numpy.abs(100.*(la2-la2b)/la2)>self.epsilon):
                print 'la:',100.*(la2-la2b)/la2
        self.assertFalse(numpy.any(numpy.abs(100.*(la2-la2b)/la2)>self.epsilon))

if __name__ == "__main__":
    unittest.main()
#t = ScripcTests()
#t.test_Fstdc_exinterp_KnownValues()

# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
