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

class ScripcTests(unittest.TestCase):

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
        doCorners = 1
        (la,lo,lac,loc) = Fstdc.ezgetlalo((nij,nij),grtyp,(grref,ig1,ig2,ig3,ig4),axes,hasAxes,ij0,doCorners)
        return (la,lo,lac,loc)

    def test_Fstdc_exinterp_KnownValues(self):
        """Fstdc_exinterp should give known result with known input"""
        (la1,lo1,lac1,loc1) = self.gridL(0.5,6)
        (la2,lo2,lac2,loc2) = self.gridL(0.25,8)
        la1  *= (numpy.pi/180.)
        lo1  *= (numpy.pi/180.)
        lac1 *= (numpy.pi/180.)
        loc1 *= (numpy.pi/180.)
        la2  *= (numpy.pi/180.)
        lo2  *= (numpy.pi/180.)
        lac2 *= (numpy.pi/180.)
        loc2 *= (numpy.pi/180.)
        nbins = -1 #use default
        method = " " #use default
        type_of_norm = " " #use default
        type_of_restric = " " #use default
        print loc1.shape,loc2.shape
        print "to script_addr_"
        (fromAddr,toAddr,weights) = scripc.scripc_addr_wts(la1,lo1,lac1,loc1,la2,lo2,lac2,loc2,nbins,method,type_of_norm,type_of_restric)
        print "to scrip_interp"
        la1  *= (180./numpy.pi)
        la2  *= (180./numpy.pi)
        la2b = scripc.scripc_interp_o1(la1,fromAddr,toAddr,weights,la.size)
        if numpy.any(numpy.abs(la2-la2b)>self.epsilon):
                print 'g1:'+repr((g1_grtyp,g1_ig14,g1_shape))
                print 'g2:'+repr((g2_grtyp,g2_ig14,g2_shape))
                print 'la2:',la2
                print 'la2b:',la2b
        self.assertFalse(numpy.any(numpy.abs(la2-la2b)>self.epsilon))

if __name__ == "__main__":
    unittest.main()

# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
