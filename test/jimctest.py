"""Unit test for rpnstd.py and Fstd.py"""

import rpnstd
from jimc import *
from jim import *
import numpy
import unittest

class jimcXchHaloTest(unittest.TestCase):

    def testjimc_new_array(self):
        """jimc_new_array should give known result with known input"""
        igrid= 0
        ndiv = 2
        nk   = 1
        f    = jimc_new_array(ndiv,nk,igrid)
        halo = 2
        nij  = 4
        nijh = nij+2*halo
        ngrids = 10
        flags = {
            'C_CONTIGUOUS' : False,
            'F_CONTIGUOUS' : True,
            'OWNDATA' : True,
            'WRITEABLE' : True,
            'ALIGNED' : True,
            'UPDATEIFCOPY' : False
        }
        stride0=4
        self.assertEqual(f.flags['F_CONTIGUOUS'],flags['F_CONTIGUOUS'])
        self.assertEqual(f.strides[0],stride0)
        self.assertEqual(f.dtype,numpy.dtype('float32'))
        self.assertEqual(f.shape,(nijh,nijh,nk,ngrids))
        nk   = 5
        f    = jimc_new_array(ndiv,nk,igrid)
        self.assertEqual(f.shape,(nijh,nijh,nk,ngrids))
        igrid= 2
        f    = jimc_new_array(ndiv,nk,igrid)
        self.assertEqual(f.shape,(nijh,nijh,nk,1))


    def testjimc_grid(self):
        """jimc_grid should give known result with known input"""
        la0 = numpy.array(
            [[[ 26.56505203, -26.56505203,  26.56505203, -26.56505203,
            26.56505203, -26.56505203,  26.56505203, -26.56505203,
            26.56505203, -26.56505203],
            [ 42.42378998, -13.4416151 ,  42.42378998, -13.4416151 ,
            42.42378998, -13.4416151 ,  42.42378998, -13.4416151 ,
            42.42378998, -13.4416151 ],
            [ 58.28252411,   0.        ,  58.28252411,   0.        ,
            58.28252411,   0.        ,  58.28252411,   0.        ,
            58.28252411,   0.        ],
            [ 74.14126587,  13.4416151 ,  74.14126587,  13.4416151 ,
            74.14126587,  13.4416151 ,  74.14126587,  13.4416151 ,
            74.14126587,  13.4416151 ]],
        [[ 13.4416151 , -42.42378998,  13.4416151 , -42.42378998,
            13.4416151 , -42.42378998,  13.4416151 , -42.42378998,
            13.4416151 , -42.42378998],
            [ 30.37922096, -30.37922096,  30.37922096, -30.37922096,
            30.37922096, -30.37922096,  30.37922096, -30.37922096,
            30.37922096, -30.37922096],
            [ 46.35307312, -16.0450573 ,  46.35307312, -16.0450573 ,
            46.35307312, -16.0450573 ,  46.35307312, -16.0450573 ,
            46.35307312, -16.0450573 ],
            [ 63.43494797,   0.        ,  63.43494797,   0.        ,
            63.43494797,   0.        ,  63.43494797,   0.        ,
            63.43494797,   0.        ]],
        [[  0.        , -58.28252411,   0.        , -58.28252411,
            0.        , -58.28252411,   0.        , -58.28252411,
            0.        , -58.28252411],
            [ 16.0450573 , -46.35307312,  16.0450573 , -46.35307312,
            16.0450573 , -46.35307312,  16.0450573 , -46.35307312,
            16.0450573 , -46.35307312],
            [ 31.71747398, -31.71747398,  31.71747398, -31.71747398,
            31.71747398, -31.71747398,  31.71747398, -31.71747398,
            31.71747398, -31.71747398],
            [ 46.35307312, -16.0450573 ,  46.35307312, -16.0450573 ,
            46.35307312, -16.0450573 ,  46.35307312, -16.0450573 ,
            46.35307312, -16.0450573 ]],
        [[-13.4416151 , -74.14126587, -13.4416151 , -74.14126587,
            -13.4416151 , -74.14126587, -13.4416151 , -74.14126587,
            -13.4416151 , -74.14126587],
            [  0.        , -63.43494797,   0.        , -63.43494797,
            0.        , -63.43494797,   0.        , -63.43494797,
            0.        , -63.43494797],
            [ 16.0450573 , -46.35307312,  16.0450573 , -46.35307312,
            16.0450573 , -46.35307312,  16.0450573 , -46.35307312,
            16.0450573 , -46.35307312],
            [ 30.37922096, -30.37922096,  30.37922096, -30.37922096,
            30.37922096, -30.37922096,  30.37922096, -30.37922096,
            30.37922096, -30.37922096]]]
            ,dtype=numpy.dtype('float32'),order='FORTRAN')
        lo0 = numpy.array(
            [[[ -36.        ,    0.        ,   36.        ,   72.        ,
            108.        ,  144.        ,  180.        , -144.        ,
            -108.        ,  -72.        ],
            [ -36.        ,    9.50570583,   36.        ,   81.50570679,
            108.        ,  153.50570679,  180.        , -134.49429321,
            -108.        ,  -62.49429321],
            [ -36.        ,   18.        ,   36.        ,   90.        ,
            108.        ,  162.        ,  180.        , -126.        ,
            -108.        ,  -54.        ],
            [ -36.        ,   26.49429321,   36.        ,   98.49429321,
            108.        ,  170.49429321,  180.        , -117.50570679,
            -108.        ,  -45.50570679]],
        [[ -26.49429321,    0.        ,   45.50570679,   72.        ,
            117.50570679,  144.        , -170.49429321, -144.        ,
            -98.49429321,  -72.        ],
            [ -18.46699715,   17.53300285,   53.53300476,   89.53300476,
            125.53300476,  161.53300476, -162.46699524, -126.46699524,
            -90.46699524,  -54.46699524],
            [ -13.61382198,   26.26769829,   58.38617706,   98.2677002 ,
            130.38618469,  170.2677002 , -157.61381531, -117.7322998 ,
            -85.61382294,  -45.7322998 ],
            [   0.        ,   36.        ,   72.        ,  108.        ,
            144.        ,  180.        , -144.        , -108.        ,
            -72.        ,  -36.        ]],
        [[ -18.        ,    0.        ,   54.        ,   72.        ,
            126.        ,  144.        , -162.        , -144.        ,
            -90.        ,  -72.        ],
            [  -9.73230171,   22.38617706,   62.2677002 ,   94.38617706,
            134.2677002 ,  166.38618469, -153.7322998 , -121.61382294,
            -81.7322998 ,  -49.61382294],
            [   0.        ,   36.        ,   72.        ,  108.        ,
            144.        ,  180.        , -144.        , -108.        ,
            -72.        ,  -36.        ],
            [  13.61382198,   45.7322998 ,   85.61382294,  117.7322998 ,
            157.61381531, -170.2677002 , -130.38618469,  -98.2677002 ,
            -58.38617706,  -26.26769829]],
        [[  -9.50570583,    0.        ,   62.49429321,   72.        ,
            134.49429321,  144.        , -153.50570679, -144.        ,
            -81.50570679,  -72.        ],
            [   0.        ,   36.        ,   72.        ,  108.        ,
            144.        ,  180.        , -144.        , -108.        ,
            -72.        ,  -36.        ],
            [   9.73230171,   49.61382294,   81.7322998 ,  121.61382294,
            153.7322998 , -166.38618469, -134.2677002 ,  -94.38617706,
            -62.2677002 ,  -22.38617706],
            [  18.46699715,   54.46699524,   90.46699524,  126.46699524,
            162.46699524, -161.53300476, -125.53300476,  -89.53300476,
            -53.53300476,  -17.53300285]]]
            ,dtype=numpy.dtype('float32'),order='FORTRAN')
        ndiv    = 2
        (la,lo) = jimc_grid_la_lo(ndiv)
        la = numpy.reshape(la,(8,8,10),order='F')
        lo = numpy.reshape(lo,(8,8,10),order='F')
        self.assertEqual(la[2,6,0], 90.)
        self.assertEqual(la[6,2,1],-90.)
        for igrid in range(0,10):
            #print igrid,numpy.any(la[2:6,2:6,igrid]!=la0[...,igrid]),numpy.any(lo[2:6,2:6,igrid]!=lo0[...,igrid])
            if numpy.any(la[2:6,2:6,igrid]!=la0[...,igrid]):
                print 'la',la[2:6,2:6,igrid]!=la0[...,igrid]
                print numpy.array_repr(la[2:6,2:6,igrid],precision=1)
                print numpy.array_repr(la0[...,igrid],precision=1)
            self.assertFalse(numpy.any(la[2:6,2:6,igrid]!=la0[...,igrid]))
            if numpy.any(lo[2:6,2:6,igrid]!=lo0[...,igrid]):
                print 'lo',lo[2:6,2:6,igrid]!=lo0[...,igrid]
                print numpy.array_repr(lo[2:6,2:6,igrid],precision=1)
                print numpy.array_repr(lo0[...,igrid],precision=1)
            self.assertFalse(numpy.any(lo[2:6,2:6,igrid]!=lo0[...,igrid]))


    def testjimc_xch_halo(self):
        """jimc_xch_halo should give known result with known input"""
        a = numpy.array(
        [[[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]],

       [[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [ 822.,   12.,   22.,  212.,  222.,  412.,  422.,  612.,  622.,
          812.],
        [ 821.,   22.,   21.,  222.,  221.,  422.,  421.,  622.,  621.,
          822.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]],

       [[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [ 921.,  922.,  121.,  122.,  321.,  322.,  521.,  522.,  721.,
          722.],
        [  11.,  111.,  211.,  311.,  411.,  511.,  611.,  711.,  811.,
          911.],
        [  21.,  121.,  221.,  321.,  421.,  521.,  621.,  721.,  821.,
          921.],
        [ 999.,  211.,  999.,  411.,  999.,  611.,  999.,  811.,  999.,
           11.],
        [ 621.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]],

       [[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [ 922.,  912.,  122.,  112.,  322.,  312.,  522.,  512.,  722.,
          712.],
        [  12.,  112.,  212.,  312.,  412.,  512.,  612.,  712.,  812.,
          912.],
        [  22.,  122.,  222.,  322.,  422.,  522.,  622.,  722.,  822.,
          922.],
        [ 221.,  212.,  421.,  412.,  621.,  612.,  821.,  812.,   21.,
           12.],
        [ 421.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]],

       [[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [ 111., -999.,  311., -999.,  511., -999.,  711., -999.,  911.,
         -999.],
        [ 121.,  312.,  321.,  512.,  521.,  712.,  721.,  912.,  921.,
          112.],
        [ 211.,  311.,  411.,  511.,  611.,  711.,  811.,  911.,   11.,
          111.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]],

       [[   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,  712.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,  512.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.],
        [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.]]]
            ,dtype=numpy.dtype('float32'),order='FORTRAN')
        ndiv = 1
        nk   = 1
        f    = jimc_new_array(ndiv,nk,0)
        halo = 2
        nij    = f.shape[0] - 2*halo
        ijlist = range(halo,halo+nij)
        f[...] = 0.
        f = numpy.reshape(f,(f.shape[0],f.shape[1],10),order='F')
        ngrids = f.shape[2]
        for g in range(ngrids):
            for j in ijlist:
                for i in ijlist:
                    f[i,j,g] = g*100.+(j-1)*10.+i-1
        f[halo,nij+halo,0] =  999.
        f[nij+halo,halo,1] = -999.
        #print numpy.array_repr(f[1:7,1:7,0],precision=1)
        #print f
        #f0 = f.copy()
        f = numpy.reshape(f,(f.shape[0],f.shape[1],1,10),order='F')
        istat = jimc_xch_halo(f)
        f = numpy.reshape(f,(f.shape[0],f.shape[1],10),order='F')
        #print numpy.array_repr(f[1:7,1:7,0],precision=1)
        #print numpy.array_repr(f)
        #jlist = range(halo-1,halo+nij+2)
        #jlist.reverse()
        for igrid in range(0,10):
            #print g,numpy.any(f[2:6,2:6,g]!=f0[2:6,2:6,g])
            #print f[1:7,1:7,g]==f0[1:7,1:7,g]
            #print f[...,g]==f0[...,g]
            #print "\n"
            #for j in jlist:
            #    print j,':',numpy.array2string(f[:,j,igrid])
            #print igrid,"\n",numpy.array_repr(f[...,igrid].transpose())
            self.assertFalse(numpy.any(f[...,igrid]!=a[...,igrid]))


    def testjim_flatten(self):
        """jim_unflatten(jim_flatten(field)) should retunr field (except halo values)"""
        ndiv = 2
        halo = 2
        (la,lo) = jimc_grid_la_lo(ndiv)
        ij0 = halo
        ijn = la.shape[0] - halo
        nij = ijn-ij0
        NPI = ij0
        NPJ = ijn
        SPI = ijn
        SPJ = ij0
        la_flat = jim_flatten(la)
        la2     = jim_unflatten(la_flat)
        self.assertEqual(la_flat.shape,(nij*nij*10+2,1))
        self.assertEqual(la.shape,la2.shape)
        self.assertFalse(numpy.any(la2[NPI,NPJ,:,0]!=la[NPI,NPJ,:,0]))
        self.assertFalse(numpy.any(la2[SPI,SPJ,:,1]!=la[SPI,SPJ,:,1]))
        for igrid in range(0,10):
            self.assertFalse(numpy.any(la2[ij0:ijn,ij0:ijn,:,igrid]!=la[ij0:ijn,ij0:ijn,:,igrid]))
        la = numpy.rollaxis(la,2,0)
        la_flat = jim_flatten(la,nkfirst=True)
        la2     = jim_unflatten(la_flat,nkfirst=True)
        self.assertEqual(la_flat.shape,(1,nij*nij*10+2))
        self.assertEqual(la.shape,la2.shape)
        la  = numpy.rollaxis(la,0,3)
        la2 = numpy.rollaxis(la2,0,3)
        self.assertFalse(numpy.any(la2[NPI,NPJ,:,0]!=la[NPI,NPJ,:,0]))
        self.assertFalse(numpy.any(la2[SPI,SPJ,:,1]!=la[SPI,SPJ,:,1]))
        for igrid in range(0,10):
            self.assertFalse(numpy.any(la2[ij0:ijn,ij0:ijn,:,igrid]!=la[ij0:ijn,ij0:ijn,:,igrid]))


if __name__ == "__main__":
    unittest.main()

