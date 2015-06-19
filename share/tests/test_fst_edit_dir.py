#!/usr/bin/env python
"""Unit test for fst_edit_dir"""

import os
import unittest
from shutil import copy
from os import chmod, stat
from stat import S_IWUSR

import rpnpy.librmn.all as rmn

vname = 'UU'

class UnitTest_fst_edit_dir(unittest.TestCase):

    def read_dateo_npas(self,fname,vname):
        'Read date of original analysis and time-step number'

        iunit = rmn.fstopenall(fname, rmn.FST_RO)
        key   = rmn.fstinf(iunit, nomvar=vname)['key']
        recmeta = rmn.fstprm(key)
        rmn.fstcloseall(iunit)

        return recmeta['datev'],recmeta['dateo'],recmeta['deet'],recmeta['npas']

    def copy_file(self):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        ref_file = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_012')
        
        TMPDIR = os.getenv('TMPDIR')
        new_file = os.path.join(TMPDIR.strip(),'new.fst')
        
        copy(ref_file, new_file)
        chmod(new_file, stat(new_file).st_mode | S_IWUSR)

        return ref_file, new_file


    def test_fst_edit_dir_datev(self):
        """Changing datev with fst_edit_dir should update dateo accordingly"""
        
        [ref_file, new_file] = self.copy_file()

        #Compare before
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)

        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertEqual(datev,datev1)
        self.assertEqual(dateo,dateo1)
        
        #Edit datev in ref_file
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        datev2 = rmn.incdatr(datev,1.)
        dateo2 = rmn.incdatr(dateo,1.)

        fnew = rmn.fstopenall(new_file, rmn.FST_RW)
        key  = rmn.fstinf(fnew, nomvar=vname)['key']
        rmn.fst_edit_dir(key, datev=datev2)
        rmn.fstcloseall(fnew)

        #Compare after
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)
        
        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertNotEqual(datev,datev1)
        self.assertEqual(datev2,datev1)
        self.assertNotEqual(dateo,dateo1)
        self.assertEqual(dateo2,dateo1)



    def test_fst_edit_dir_dateo(self):
        """Changing dateo with fst_edit_dir should update datev accordingly"""
        
        [ref_file, new_file] = self.copy_file()

        #Compare before
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)

        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertEqual(datev,datev1)
        self.assertEqual(dateo,dateo1)
        
        #Edit dateo in ref_file
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        dateo2 = rmn.incdatr(dateo,1.)
        datev2 = rmn.incdatr(datev,1.)

        fnew = rmn.fstopenall(new_file, rmn.FST_RW)
        key  = rmn.fstinf(fnew, nomvar=vname)['key']
        rmn.fst_edit_dir(key, dateo=dateo2)
        rmn.fstcloseall(fnew)

        #Compare after
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)
        
        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertNotEqual(datev,datev1)
        self.assertEqual(datev2,datev1)
        self.assertNotEqual(dateo,dateo1)
        self.assertEqual(dateo2,dateo1)


    def test_fst_edit_dir_npas(self):
        """Changing npas with fst_edit_dir should update dateo accordingly"""
        
        [ref_file, new_file] = self.copy_file()

        #Compare before
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)

        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertEqual(datev,datev1)
        self.assertEqual(dateo,dateo1)
        
        #Edit npas in ref_file
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        npas2 = npas+1
        dateo2 = rmn.incdatr(dateo,-1.*deet/3600.)
        datev2 = datev

        fnew = rmn.fstopenall(new_file, rmn.FST_RW)
        key  = rmn.fstinf(fnew, nomvar=vname)['key']
        rmn.fst_edit_dir(key, npas=npas2)
        rmn.fstcloseall(fnew)

        #Compare after
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)
        
        self.assertEqual(deet,deet1)
        self.assertEqual(npas2,npas1)
        self.assertEqual(datev,datev1)
        self.assertEqual(datev2,datev1)
        self.assertNotEqual(dateo,dateo1)
        self.assertEqual(dateo2,dateo1)


    def test_fst_edit_dir_npas_keepdateo(self):
        """Changing npas with keepdate in fst_edit_dir should update datev accordingly"""
        
        [ref_file, new_file] = self.copy_file()

        #Compare before
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)

        self.assertEqual(deet,deet1)
        self.assertEqual(npas,npas1)
        self.assertEqual(datev,datev1)
        self.assertEqual(dateo,dateo1)
        
        #Edit npas in ref_file
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        npas2 = npas+1
        dateo2 = dateo
        datev2 = rmn.incdatr(datev,deet/3600.)

        fnew = rmn.fstopenall(new_file, rmn.FST_RW)
        key  = rmn.fstinf(fnew, nomvar=vname)['key']
        rmn.fst_edit_dir(key, npas=npas2,keep_dateo=True)
        rmn.fstcloseall(fnew)

        #Compare after
        [datev,  dateo,  deet,  npas]  = self.read_dateo_npas(ref_file,vname)
        [datev1, dateo1, deet1, npas1] = self.read_dateo_npas(new_file,vname)
        
        self.assertEqual(deet,deet1)
        self.assertEqual(npas2,npas1)
        self.assertNotEqual(datev,datev1)
        self.assertEqual(datev2,datev1)
        self.assertEqual(dateo,dateo1)
        self.assertEqual(dateo2,dateo1)


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
