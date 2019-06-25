#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

class RpnPyUtilsFstd3D(unittest.TestCase):


    def test_fstdread3d(self):
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
            rec3d = fstd3d.fst_read_3d(fileId, nomvar='TT', getPress=True, verbose=True)
            vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_DISALLOW_SIGMA)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)

        self.assertAlmostEqual(rec3d['d'].mean(), -36.287384, places=6,
                               msg=None, delta=None)


    def test_fstdread3d_abitrary(self):
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
            j1 = fstd3d.fst_read_3d(fileId, nomvar='J1', verbose=True)
            j2 = fstd3d.fst_read_3d(fileId, nomvar='J2',
                                    ip1=ip1s, verbose=True)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)

        self.assertAlmostEqual(j1['d'].mean(), 18.77356, places=6,
                               msg=None, delta=None)
        self.assertAlmostEqual(j2['d'].mean(), 8.8374939, places=6,
                               msg=None, delta=None)


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
