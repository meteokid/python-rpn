#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""Unit tests for librmn.interp"""

import os
import rpnpy.librmn.all as rmn
from rpnpy import range as _range
import unittest
## import ctypes as ct
import numpy as np

class Librmn_interp_Test(unittest.TestCase):

    epsilon = 0.0005

    def setIG_L(self,gp):
        ig1234 = rmn.cxgaig(gp['grtyp'],gp['lat0'],gp['lon0'],
                            gp['dlat'],gp['dlon'])
        gp['ig1'] = ig1234[0]
        gp['ig2'] = ig1234[1]
        gp['ig3'] = ig1234[2]
        gp['ig4'] = ig1234[3]
        return gp

    def setIG_ZE(self,gp,offx=0):
        ig1234 = rmn.cxgaig(gp['grref'],gp['xlat1'],gp['xlon1'],
                            gp['xlat2'],gp['xlon2'])
        gp['ig1ref'] = ig1234[0]
        gp['ig2ref'] = ig1234[1]
        gp['ig3ref'] = ig1234[2]
        gp['ig4ref'] = ig1234[3]
        gp['ig1'] = ig1234[0]
        gp['ig2'] = ig1234[1]
        gp['ig3'] = ig1234[2]
        gp['ig4'] = ig1234[3]
        ## if offx: offx=1
        ## gp['ig1'] = 123+offx
        ## gp['ig2'] = 231+offx
        ## gp['ig3'] = 312+offx
        ## gp['ig4'] = 0
        return gp

    def getGridParams_L(self,offx=0):
        (ni,nj) = (90,180)
        if offx:
            offx=0.25
            (ni,nj) = (45,90)
        gp = {
            'shape' : (ni,nj),
            'ni' : ni,
            'nj' : nj,
            'grtyp' : 'L',
            'dlat' : 0.5,
            'dlon' : 0.5,
            'lat0' : 45.,
            'lon0' : 273.+offx
            }
        return self.setIG_L(gp)

    def getGridParams_ZE(self):
        (ni,nj) = (50,30)
        gp = {
            'shape' : (ni,nj),
            'ni' : ni,
            'nj' : nj,
            'grtyp' : 'Z',
            'grref' : 'E',
            'xlat1' : 0.,
            'xlon1' : 180.,
            'xlat2' : 0.,
            'xlon2' : 270.,
            'dlat' : 0.5,
            'dlon' : 0.5,
            'lat0' : 45.,
            'lon0' : 273.
            }
        gp['ax'] = np.empty((ni,1),dtype=np.float32,order='F')
        gp['ay'] = np.empty((1,nj),dtype=np.float32,order='F')
        for i in _range(ni):
            gp['ax'][i,0] = gp['lon0']+float(i)*gp['dlon']
        for j in _range(nj):
            gp['ay'][0,j] = gp['lat0']+float(j)*gp['dlat']
        return self.setIG_ZE(gp)

    def getGridParams_ZEYY(self,YY=0):
        nj = 31
        ni = (nj-1)*3 + 1
        (ni,nj) = (50,30)
        (xlat1,xlon1,xlat2,xlon2) = (0., 180., 0., 270.)
        if YY > 0:
            (xlat1,xlon1,xlat2,xlon2) =  (0., 180., 45., 180.)
        gp = {
            'shape' : (ni,nj),
            'ni' : ni,
            'nj' : nj,
            'grtyp' : 'Z',
            'grref' : 'E',
            'xlat1' : xlat1,
            'xlon1' : xlon1,
            'xlat2' : xlat2,
            'xlon2' : xlon2,
            'dlat' : (90./float(nj)),
            'dlon' : (270./float(ni)),
            'lat0' : -45.,
            'lon0' : 45.
            }
        gp['ax'] = np.empty((ni,1),dtype=np.float32,order='F')
        gp['ay'] = np.empty((1,nj),dtype=np.float32,order='F')
        for i in _range(ni):
            gp['ax'][i,0] = gp['lon0']+float(i)*gp['dlon']
        for j in _range(nj):
            gp['ay'][0,j] = gp['lat0']+float(j)*gp['dlat']
        return self.setIG_ZE(gp)


    def test_ezsetopt_ezgetopt(self):
        otplist = [
            (rmn.EZ_OPT_WEIGHT_NUMBER,2), #int
            (rmn.EZ_OPT_EXTRAP_VALUE,99.), #float
            (rmn.EZ_OPT_EXTRAP_DEGREE.lower(),rmn.EZ_EXTRAP_VALUE.lower()) #str
            ]
        for (o,v) in otplist:
            rmn.ezsetopt(o,v)
            v1 = rmn.ezgetopt(o,vtype=type(v))
            self.assertEqual(v1,v)

    def test_ezqkdef_ezgprm(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gprm = rmn.ezgprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k])
        rmn.gdrls(gid1)


    def test_ezqkdef_file_ezgprm_ezgfstp(self):
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/geophy.fst')
        funit = rmn.fstopenall(myfile,rmn.FST_RO)
        (ni,nj) = (201,100)
        gp = {
            'shape' : (ni,nj),
            'ni' : ni,
            'nj' : nj,
            'grtyp' : 'Z',
            'ig1'   : 2002,
            'ig2'   : 1000,
            'ig3'   : 0,
            'ig4'   : 0,
            'grref' : 'E',
            'ig1ref' : 900,
            'ig2ref' : 0,
            'ig3ref' : 43200,
            'ig4ref' : 43200,
            'iunit'  : funit
            }
        gid1 = rmn.ezqkdef(gp)
        a = rmn.ezgfstp(gid1)
        rmn.fstcloseall(funit)
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gprm = rmn.ezgxprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k])
        self.assertEqual(a['nomvarx'].strip(),'>>')
        self.assertEqual(a['nomvary'].strip(),'^^')
        rmn.gdrls(gid1)

    def test_ezqkdef_file_error(self):
        funit = -1
        (ni,nj) = (201,100)
        gp = {
            'shape' : (ni,nj),
            'ni' : ni,
            'nj' : nj,
            'grtyp' : 'Z',
            'ig1'   : 2002,
            'ig2'   : 1000,
            'ig3'   : 0,
            'ig4'   : 0,
            'grref' : 'E',
            'ig1ref' : 900,
            'ig2ref' : 0,
            'ig3ref' : 43200,
            'ig4ref' : 43200,
            'iunit'  : funit
            }
        try:
            gid1 = rmn.ezqkdef(gp)
            self.assertTrue(False, 'ezqkdef should raise a error with ref grid and invalid file unit')
        except rmn.EzscintError:
            pass

    def test_ezqkdef_ezgxprm(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp['ni'],gp['nj'],gp['grtyp'],
                           gp['ig1'],gp['ig2'],gp['ig3'],gp['ig4'])
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gp['grref'] = ''
        gp['ig1ref'] = 0
        gp['ig2ref'] = 0
        gp['ig3ref'] = 0
        gp['ig4ref'] = 0
        gprm = rmn.ezgxprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k])
        rmn.gdrls(gid1)

    def test_ezgkdef_fmem_ezgxprm(self):
        gp = self.getGridParams_ZE()
        gid1 = rmn.ezgdef_fmem(gp['ni'],gp['nj'],gp['grtyp'],gp['grref'],
                               gp['ig1ref'],gp['ig2ref'],gp['ig3ref'],gp['ig4ref'],
                               gp['ax'],gp['ay'])
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        gprm = rmn.ezgxprm(gid1)
        for k in gprm.keys():
            self.assertEqual(gp[k],gprm[k],'(%s) Expected: %s, Got: %s :: %s' % (k,repr(gp[k]),repr(gprm[k]),repr(gprm)))
        rmn.gdrls(gid1)

    def test_ezgkdef_fmem_gdgaxes(self):
        gp = self.getGridParams_ZE()
        gid1 = rmn.ezgdef_fmem(gp['ni'],gp['nj'],gp['grtyp'],gp['grref'],
                               gp['ig1ref'],gp['ig2ref'],gp['ig3ref'],gp['ig4ref'],
                               gp['ax'],gp['ay'])
        self.assertTrue(gid1>=0)
        gp['id'] = gid1
        axes = rmn.gdgaxes(gid1)
        self.assertEqual(axes['ax'].shape,gp['ax'].shape)
        self.assertEqual(axes['ay'].shape,gp['ay'].shape)
        for i in _range(gp['ni']):
            self.assertTrue(abs(axes['ax'][i,0]-gp['ax'][i,0])<self.epsilon)
        for j in _range(gp['nj']):
            self.assertTrue(abs(axes['ay'][0,j]-gp['ay'][0,j])<self.epsilon)
        rmn.gdrls(gid1)

    def test_ezqkdef_1subgrid(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        ng = rmn.ezget_nsubgrids(gid1)
        self.assertEqual(ng,1)
        subgid = rmn.ezget_subgridids(gid1)
        self.assertEqual(subgid,[gid1])
        rmn.gdrls(gid1)

    def test_ezqkdef_gdll(self):
        gp = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp)
        self.assertTrue(gid1>=0)
        gll = rmn.gdll(gid1)
        self.assertEqual(gp['shape'],gll['lat'].shape)
        self.assertEqual(gp['shape'],gll['lon'].shape)
        self.assertEqual(int(round(gp['lat0']*1000.)),
                         int(round(gll['lat'][0,0]*1000.)))
        self.assertEqual(int(round(gp['lon0']*1000.)),
                         int(round(gll['lon'][0,0]*1000.)))
        rmn.gdrls(gid1)

    def test_ezsint(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        gp2 = self.getGridParams_L(0.25)
        gid2 = rmn.ezqkdef(gp2)
        self.assertTrue(gid2>=0)
        setid = rmn.ezdefset(gid2, gid1)
        self.assertTrue(setid>=0)
        zin = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            zin[x,:] = x
        zout = rmn.ezsint(gid2,gid1,zin)
        self.assertEqual(gp2['shape'],zout.shape)
        for j in _range(gp2['nj']):
            for i in _range(gp2['ni']):
                self.assertTrue(abs((zin[i,j]+zin[i+1,j])/2.-zout[i,j]) < self.epsilon)
        #rmn.gdrls([gid1,gid2]) #TODO: Makes the test crash

    def test_ezsint_extrap(self):
        extrap_val = 99.
        gp = self.getGridParams_ZE()
        gid1 = rmn.ezgdef_fmem(gp['ni'],gp['nj'],gp['grtyp'],gp['grref'],
                               gp['ig1ref'],gp['ig2ref'],gp['ig3ref'],gp['ig4ref'],
                               gp['ax'],gp['ay'])
        self.assertTrue(gid1>=0)
        gp2 = self.getGridParams_L(0.25)
        gid2 = rmn.ezqkdef(gp2)
        self.assertTrue(gid2>=0)
        setid = rmn.ezdefset(gid2, gid1)
        self.assertTrue(setid>=0)
        zin = np.empty(gp['shape'], dtype=np.float32, order='F')
        for x in _range(gp['ni']):
            zin[x,:] = x
        rmn.ezsetopt(rmn.EZ_OPT_EXTRAP_VALUE, extrap_val)
        rmn.ezsetopt(rmn.EZ_OPT_EXTRAP_DEGREE.lower(),
                     rmn.EZ_EXTRAP_VALUE.lower())
        zout = rmn.ezsint(gid2, gid1, zin)
        self.assertEqual(gp2['shape'], zout.shape)
        self.assertEqual(extrap_val, np.max(zout))
        ## print('test_ezsint_extrap', np.min(zin), np.max(zin))
        ## print('test_ezsint_extrap', np.min(zout), np.max(zout))
        ## for j in _range(gp2['nj']):
        ##     print('test_ezsint_extrap', j, zout[:,j])
        #rmn.gdrls([gid1,gid2]) #TODO: Makes the test crash

    def test_ezuvint(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        gp2 = self.getGridParams_L(0.25)
        gid2 = rmn.ezqkdef(gp2)
        self.assertTrue(gid2>=0)
        setid = rmn.ezdefset(gid2, gid1)
        self.assertTrue(setid>=0)
        uuin = np.empty(gp1['shape'],dtype=np.float32,order='F')
        vvin = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            uuin[x,:] = x
        vvin = uuin*3.
        (uuout,vvout) = rmn.ezuvint(gid2,gid1,uuin,vvin)
        self.assertEqual(gp2['shape'],uuout.shape)
        self.assertEqual(gp2['shape'],vvout.shape)
        for j in _range(gp2['nj']):
            for i in _range(gp2['ni']):
                self.assertTrue(abs((uuin[i,j]+uuin[i+1,j])/2.-uuout[i,j]) < self.epsilon,'uvint, u: abs(%f-%f)=%f' % (((uuin[i,j]+uuin[i+1,j])/2),uuout[i,j],(uuin[i,j]+uuin[i+1,j])/2.-uuout[i,j]))

                self.assertTrue(abs((vvin[i,j]+vvin[i+1,j])/2.-vvout[i,j]) < self.epsilon,'uvint, v: abs(%f-%f)=%f' % (((vvin[i,j]+vvin[i+1,j])/2),vvout[i,j],(vvin[i,j]+vvin[i+1,j])/2.-vvout[i,j]))
                ## self.assertEqual((uuin[i,j]+uuin[i+1,j])/2.,uuout[i,j])
                ## self.assertEqual((vvin[i,j]+vvin[i+1,j])/2.,vvout[i,j])
        #rmn.gdrls([gid1,gid2]) #TODO: Makes the test crash


    def test_ezgkdef_fmem_gdxyfll(self):
        gp = self.getGridParams_ZE()
        gid1 = rmn.ezgdef_fmem(gp['ni'],gp['nj'],gp['grtyp'],gp['grref'],
                               gp['ig1ref'],gp['ig2ref'],gp['ig3ref'],gp['ig4ref'],
                               gp['ax'],gp['ay'])
        self.assertTrue(gid1>=0)
        lat = np.array([gp['ay'][0,0],gp['ay'][0,1]],dtype=np.float32,order='F')
        lon = np.array([gp['ax'][0,0],gp['ax'][1,0]],dtype=np.float32,order='F')
        xypts = rmn.gdxyfll(gid1, lat, lon)
        self.assertEqual(xypts['x'].shape,lat.shape)
        self.assertEqual(xypts['y'].shape,lat.shape)
        self.assertTrue(abs(xypts['x'][0]-1.)<self.epsilon)
        self.assertTrue(abs(xypts['y'][0]-1.)<self.epsilon)
        self.assertTrue(abs(xypts['x'][1]-2.)<self.epsilon)
        self.assertTrue(abs(xypts['y'][1]-2.)<self.epsilon)
        rmn.gdrls(gid1)


    def test_ezgkdef_fmem_gdllfxy(self):
        gp = self.getGridParams_ZE()
        gid1 = rmn.ezgdef_fmem(gp['ni'],gp['nj'],gp['grtyp'],gp['grref'],
                               gp['ig1ref'],gp['ig2ref'],gp['ig3ref'],gp['ig4ref'],
                               gp['ax'],gp['ay'])
        self.assertTrue(gid1>=0)
        xx = np.array([1.,2.],dtype=np.float32,order='F')
        yy = np.array([1.,3.],dtype=np.float32,order='F')
        llpts = rmn.gdllfxy(gid1, xx, yy)
        self.assertEqual(llpts['x'].shape,xx.shape)
        self.assertEqual(llpts['y'].shape,xx.shape)
        self.assertTrue(abs(llpts['lon'][0]-gp['ax'][0,0])<self.epsilon)
        self.assertTrue(abs(llpts['lat'][0]-gp['ay'][0,0])<self.epsilon)
        self.assertTrue(abs(llpts['lon'][1]-gp['ax'][1,0])<self.epsilon)
        self.assertTrue(abs(llpts['lat'][1]-gp['ay'][0,2])<self.epsilon)
        rmn.gdrls(gid1)


    def test_gdllsval(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        zin = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            zin[:,x] = x
        lat = np.array([gp1['lat0']+gp1['dlat']/2.],dtype=np.float32,order='F')
        lon = np.array([(gp1['lon0']+gp1['dlon'])/2.],dtype=np.float32,order='F')
        zout = rmn.gdllsval(gid1,lat,lon,zin)
        self.assertEqual(lat.shape,zout.shape)
        self.assertTrue(abs((zin[0,0]+zin[1,1])/2. - zout[0]) < self.epsilon)
        rmn.gdrls(gid1)

    def test_gdxysval(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        zin = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            zin[:,x] = x
        xx = np.array([1.5],dtype=np.float32,order='F')
        yy = np.array([1.5],dtype=np.float32,order='F')
        zout = rmn.gdxysval(gid1,xx,yy,zin)
        self.assertEqual(xx.shape,zout.shape)
        self.assertTrue(abs((zin[0,0]+zin[1,1])/2. - zout[0]) < self.epsilon)
        rmn.gdrls(gid1)


    def test_gdllvval(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        zin  = np.empty(gp1['shape'],dtype=np.float32,order='F')
        zin2 = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            zin[:,x] = x
            zin2[:,x] = x+1
        lat = np.array([gp1['lat0']+gp1['dlat']/2.],dtype=np.float32,order='F')
        lon = np.array([(gp1['lon0']+gp1['dlon'])/2.],dtype=np.float32,order='F')
        (zout,zout2) = rmn.gdllvval(gid1,lat,lon,zin,zin2)
        self.assertEqual(lat.shape,zout.shape)
        self.assertEqual(lat.shape,zout2.shape)
        self.assertTrue(abs((zin[0,0]+zin[1,1])/2. - zout[0]) < self.epsilon)
        self.assertTrue(abs((zin2[0,0]+zin2[1,1])/2. - zout2[0]) < self.epsilon)
        rmn.gdrls(gid1)


    def test_gdxyvval(self):
        gp1 = self.getGridParams_L()
        gid1 = rmn.ezqkdef(gp1)
        self.assertTrue(gid1>=0)
        zin  = np.empty(gp1['shape'],dtype=np.float32,order='F')
        zin2 = np.empty(gp1['shape'],dtype=np.float32,order='F')
        for x in _range(gp1['ni']):
            zin[:,x] = x
            zin2[:,x] = x+1
        xx = np.array([1.5],dtype=np.float32,order='F')
        yy = np.array([1.5],dtype=np.float32,order='F')
        (zout,zout2) = rmn.gdxyvval(gid1,xx,yy,zin,zin2)
        self.assertEqual(xx.shape,zout.shape)
        self.assertTrue(abs((zin[0,0]+zin[1,1])/2. - zout[0]) < self.epsilon)
        self.assertTrue(abs((zin2[0,0]+zin2[1,1])/2. - zout2[0]) < self.epsilon)
        rmn.gdrls(gid1)

    ## def test_gdsetmask_gdgetmask(self):
    ##     gp1 = self.getGridParams_L()
    ##     gid1 = rmn.ezqkdef(gp1)
    ##     self.assertTrue(gid1>=0)
    ##     mask = np.empty(gp1['shape'],dtype=np.intc,order='F')
    ##     mask[:,:] = 0
    ##     for i in _range(min(gp1['ni'],gp1['nj'])):
    ##         mask[i,i] = 1
    ##     rmn.gdsetmask(gid1,mask)
    ##     mask2 = rmn.gdgetmask(gid1)
    ##     print [mask[i,i] for i in _range(min(gp1['ni'],gp1['nj']))]
    ##     print [mask2[i,i] for i in _range(min(gp1['ni'],gp1['nj']))]
    ##     self.assertEqual(mask.shape,mask2.shape)
    ##     for j in _range(gp1['nj']):
    ##         for i in _range(gp1['ni']):
    ##             self.assertEqual(mask[i,j],mask2[i,j])
    ##     rmn.gdrls(gid1)


    def test_ezgkdef_fmem_YY_ezgxprm_supergrid(self):
        gp1 = self.getGridParams_ZEYY(0)
        gp2 = self.getGridParams_ZEYY(1)
        gid1 = rmn.ezgdef_fmem(gp1['ni'],gp1['nj'],gp1['grtyp'],gp1['grref'],
                               gp1['ig1ref'],gp1['ig2ref'],gp1['ig3ref'],gp1['ig4ref'],
                               gp1['ax'],gp1['ay'])
        gid2 = rmn.ezgdef_fmem(gp2['ni'],gp2['nj'],gp2['grtyp'],gp2['grref'],
                               gp2['ig1ref'],gp2['ig2ref'],gp2['ig3ref'],gp2['ig4ref'],
                               gp2['ax'],gp2['ay'])
        self.assertTrue(gid1>=0)
        self.assertTrue(gid2>=0)
        subgridid = [gid1,gid2]
        gp12 = {
            'ni' : gp1['ni'],
            'nj' : 2*gp1['nj'],
            'grtyp' : 'U',
            'grref' : 'F',
            'vercode' : 1
            }
        gid12 = rmn.ezgdef_supergrid(gp12['ni'],gp12['nj'],
                                     gp12['grtyp'],gp12['grref'],
                                     gp12['vercode'],subgridid)
        self.assertTrue(gid12>=0)
        ng = rmn.ezget_nsubgrids(gid12)
        self.assertEqual(ng,2)
        subgid = rmn.ezget_subgridids(gid12)
        self.assertEqual(len(subgid),2)
        self.assertEqual(subgid[0],gid1)
        self.assertEqual(subgid[1],gid2)


#TODO: test_ezgdef_supergrid

#TODO:    c_gdllwdval(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
#TODO:    c_gdxywdval(gdin, uuout, vvout, uuin, vvin, x, y, n)

#TODO:    c_ezsint_mdm(zout, mask_out, zin, mask_in)
#TODO:    c_ezuvint_mdm(uuout, vvout, mask_out, uuin, vvin, mask_in)
#TODO:    c_ezsint_mask(mask_out, mask_in)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
