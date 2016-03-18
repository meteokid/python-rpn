#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

class RpnPyUtilsFstd3D(unittest.TestCase):
    

    #---- Horizontal Interpolation


    def test_fstdread3d(self):
        """
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        import rpnpy.utils.fstd3d as fstd3d
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        fileNameIn  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)

        # Restric to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        try:
            fileId = rmn.fstopenall(fileNameIn)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            rec3d = fstd3d.fst_read_3d(fileId, nomvar='TT', getPress=True, verbose=True)
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)



if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
