#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

class RpnPyUtilsFstd3D(unittest.TestCase):


    def test_sort_ip1(self):
        import rpnpy.utils.fstd3d as fstd3d
        ip1s = [1195, 1196, 1197, 1198, 1199, 1199]
        eip1s = [1199, 1198, 1197, 1196, 1195]
        gip1s = fstd3d.sort_ip1(ip1s)
        self._cmp_obj(gip1s, eip1s, 'sort_ip1()')


    def test_get_levels_press(self):
        import os, os.path
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        filename = os.path.join(ATM_MODEL_DFILES,'bcmk')

        # Open existing file in Rear Only mode
        fileId = rmn.fstopenall(filename, rmn.FST_RO)

        # Get the pressure cube
        ipkeys  = fstd3d.get_levels_keys(fileId, 'TT', thermoMom='VIPT')
        ip1list = [ip1 for ip1,key in ipkeys['ip1keys']]
        shape   = rmn.fstinf(fileId, nomvar='TT')['shape'][0:2]
        press   = fstd3d.get_levels_press(fileId, ipkeys['vptr'], shape, ip1list)
        rmn.fstcloseall(fileId)

        exp = "# (200, 100) (200, 100) (200, 100, 80)"
        got = '# {} {} {}'.format(shape, press['rfld'].shape,
                                  press['phPa'].shape)
        self._cmp_obj(got, exp, 'get_levels_press()')


    def test_get_levels_keys(self):
        import os, os.path
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        filename = os.path.join(ATM_MODEL_DFILES,'bcmk')

        # Open existing file in Rear Only mode
        fileId = rmn.fstopenall(filename, rmn.FST_RO)

        # Get the pressure cube
        ipkeys  = fstd3d.get_levels_keys(fileId, 'TT', thermoMom='VIPT')

        rmn.fstcloseall(fileId)

        exp = "# Found 80 levels for TT"
        got = '# Found {} levels for TT'.format(len(ipkeys['ip1keys']))
        self._cmp_obj(got, exp, 'get_levels_keys()')


    def test_vgrid_new(self):
        import numpy  as np
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d
        import rpnpy.vgd.all as vgd
        from rpnpy.rpndate import RPNDate

        lvls  = (0.013,   0.027,    0.051,    0.075,
                 0.101,   0.127,    0.155,    0.185,    0.219,
                 0.258,   0.302,    0.351,    0.405,    0.460,
                 0.516,   0.574,    0.631,    0.688,    0.744,
                 0.796,   0.842,    0.884,    0.922,    0.955,
                 0.980,   0.995)
        rcoef1 = 0.
        rcoef2 = 1.
        pref   = 100000.
        dhm    = 10.
        dht    = 2.
        try:
            v = vgd.vgd_new_hybmd(lvls, rcoef1, rcoef2, pref, dhm, dht)
        except vgd.VGDError:
            raise

        vgrid = fstd3d.vgrid_new('VIPT', v, rfldError=False)

        self.assertEqual(len(vgrid['ip1s']), 28)


    @unittest.skip('test_vgrid_write needed')
    def test_vgrid_write(self):
        pass


    def test_vgrid_read(self):
        import os, os.path
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        myfile = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk')
        fileId  = rmn.fstopenall(myfile)

        vgrid = fstd3d.vgrid_read(fileId, nomvar='TT', ip2=0)
        self.assertEqual(len(vgrid['ip1s']), 80)
        self.assertEqual(vgrid['rfld']['nomvar'].strip(), 'P0')

        vgrid = fstd3d.vgrid_read(fileId, nomvar='J1')
        self.assertEqual(len(vgrid['ip1s']), 5)
 
        rmn.fstcloseall(fileId)


    def test_fst_read_3d(self):
        """
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d
        import rpnpy.vgd.all as vgd
        ## fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        ## CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        ## fileNameIn  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn = os.path.join(ATM_MODEL_DFILES, 'bcmk')

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)

        try:
            fileId = rmn.fstopenall(fileNameIn)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
            rec3d = fstd3d.fst_read_3d(fileId, nomvar='TT', getPress=True, verbose=False)
            vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_DISALLOW_SIGMA)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)

        self.assertAlmostEqual(rec3d['d'].mean(), -36.287384, places=2,
                               msg=None, delta=None)


    def test_fst_read_3d_abitrary(self):
        """
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d
        import rpnpy.vgd.all as vgd
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn = os.path.join(ATM_MODEL_DFILES, 'bcmk')

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)

        try:
            fileId = rmn.fstopenall(fileNameIn)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        ip1s = (1199, 1198, 1197, 1196, 1195)
        try:
            j1 = fstd3d.fst_read_3d(fileId, nomvar='J1', verbose=False)
            j2 = fstd3d.fst_read_3d(fileId, nomvar='J2',
                                    ip1=ip1s, verbose=False)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)

        self.assertAlmostEqual(j1['d'].mean(), 18.77356, places=3,
                               msg=None, delta=None)
        self.assertAlmostEqual(j2['d'].mean(), 8.8374939, places=3,
                               msg=None, delta=None)


    @unittest.skip('test_fst_write_3d needed')
    def test_fst_write_3d(self):
        pass


    def test_fst_new_3d(self):
        """
        """
        import os, sys, datetime
        import numpy  as np
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d
        import rpnpy.vgd.all as vgd
        ## fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        ## CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        ## fileNameIn  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn = os.path.join(ATM_MODEL_DFILES, 'bcmk')

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)

        try:
            fileId = rmn.fstopenall(fileNameIn)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
            rec3d = fstd3d.fst_read_3d(fileId, nomvar='TT', getPress=True, verbose=False)
            vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_DISALLOW_SIGMA)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)

        r2 = {}
        for k in ('nomvar', 'dateo', 'deet', 'npas', 'ip2', 'ip3', 'etiket', 'datyp', 'typvar', 'rfld', 'phPa'):
            r2[k] = rec3d[k]
        for k in ('g', 'v', 'ip1s'):  # remove legacy keys
            del(rec3d[k])
        r2 = fstd3d.fst_new_3d(r2, hgrid=rec3d['hgrid'], vgrid=rec3d['vgrid'],
                               dataArray=rec3d['d'])

        kskip = ('xtra1', 'nbits', 'key', 'lng','datev', 'swa')
        self._cmp_obj(r2, rec3d, 'fst_new_3d()', kskip)


    def _cmp_obj(self, got, exp, msg='', skip=[], errorAdditional=False):
        import numpy  as np
        import rpnpy.vgd.all as vgd
        val="\n\tExp: {}\n\tGot: {}".format(repr(exp), repr(got))
        self.assertEqual(type(got), type(exp), msg+val)
        #TODO: implement almostEqual for real values and arrays of...
        if isinstance(got, str):
            self.assertEqual(got.strip(), exp.strip(), msg+val)
        elif isinstance(got, np.ndarray):
            self.assertTrue(np.all(got == exp), msg+val)
        elif isinstance(got, (list, tuple)):
            self.assertTrue(np.all(got == exp), msg+val)
        elif isinstance(got, type(vgd.c_vgd_construct())):
            self.assertTrue(vgd.vgd_cmp(got, exp), msg+val)
        elif isinstance(got, dict):
            for k in exp.keys():
                if k in skip:
                    continue
                self.assertTrue(k in got.keys(), msg+':missing: ' + k + val)
                self._cmp_obj(got[k], exp[k], msg+'['+k+']', skip)
            for k in got.keys():
                if k in skip:
                    continue
                if errorAdditional:
                    self.assertTrue(k not in exp.keys(), msg+':additional: ' + k + val)
                else:
                    if k not in exp.keys():
                        print('WARNING: '+msg+':additional: ' + k)
        else:
            self.assertEqual(got, exp, msg+val)

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
