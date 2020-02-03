#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

class rpnpyCookbook3(unittest.TestCase):
    

    #---- Horizontal Interpolation


    def test_ze(self):
        """
        Horizontal Interpolation
        
        See also:
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn  = os.path.join(ATM_MODEL_DFILES,'bcmk','geophy.fst')
        fileNameOut = '__myfstfileze.fst'

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)


        try:
            # Create Destination grid
            # Note: Destination grid can also be read from a file
            gp = {
                'grtyp' : 'Z',
                'grref' : 'E',
                'ni'    : 180,
                'nj'    : 90.,
                'lat0'  : 0.,
                'lon0'  : 0.,
                'dlat'  : 1.,
                'dlon'  : 1.,
                'xlat1' : 35.,
                'xlon1' : 250.,
                'xlat2' : 0.,
                'xlon2' : 340.
                }
            gOut = rmn.encodeGrid(gp)
            print("CB41: Defined a %s/%s grid of shape=%d, %d" %
                  (gOut['grtyp'], gOut['grref'], gOut['ni'], gOut['nj']))
        except:
            sys.stderr.write("Problem creating grid\n")
            sys.exit(1)

        # Open Files
        try:
            fileIdIn  = rmn.fstopenall(fileNameIn)
            fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            # Find and read record to interpolate with its grid 
            r = rmn.fstlir(fileIdIn, nomvar='MG')
            gIn = rmn.readGrid(fileIdIn, r)
            print("CB41: Read P0")

            # Set interpolation options and interpolate
            rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
            d = rmn.ezsint(gOut, gIn, r)
            print("CB41: Interpolate P0")

            # Create new record to write with interpolated data and 
            r2 = r.copy()    # Preserve meta from original record
            r2.update(gOut)  # update grid information
            r2.update({      # attach data and update specific meta
                'etiket': 'my_etk',
                'd'     : d
                })
            
            # Write record data + meta + grid to file
            rmn.fstecr(fileIdOut, r2)
            rmn.writeGrid(fileIdOut, gOut)
            print("CB41: Wrote interpolated P0 and its grid")
        except:
            pass
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileIdIn)
            rmn.fstcloseall(fileIdOut)
            #os.unlink(fileNameOut)  # Remove test file


    def test_zl(self):
        """
        Horizontal Interpolation
        
        See also:
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn  = os.path.join(ATM_MODEL_DFILES,'bcmk','geophy.fst')
        fileNameOut = '__myfstfilezl.fst'

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)


        try:
            # Create Destination grid
            # Note: Destination grid can also be read from a file
            gp = {
                'grtyp' : 'Z',
                'grref' : 'L',
                'ni'    : 90,
                'nj'    : 45,
                ## 'lat0'  : 35.,
                ## 'lon0'  : 250.,
                'lat0'  : 35.,
                'lon0'  : 350.,
                'dlat'  : 0.5,
                'dlon'  : 0.5
                }
            gOut = rmn.encodeGrid(gp)
            print("CB41: Defined a %s/%s grid of shape=%d, %d" %
                  (gOut['grtyp'], gOut['grref'], gOut['ni'], gOut['nj']))
        except:
            sys.stderr.write("Problem creating grid\n")
            sys.exit(1)

        # Open Files
        try:
            fileIdIn  = rmn.fstopenall(fileNameIn)
            fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            # Find and read record to interpolate with its grid 
            r = rmn.fstlir(fileIdIn, nomvar='MG')
            gIn = rmn.readGrid(fileIdIn, r)
            print("CB41: Read P0")

            # Set interpolation options and interpolate
            rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
            d = rmn.ezsint(gOut, gIn, r)
            print("CB41: Interpolate P0")

            # Create new record to write with interpolated data and 
            r2 = r.copy()    # Preserve meta from original record
            r2.update(gOut)  # update grid information
            r2.update({      # attach data and update specific meta
                'etiket': 'my_etk',
                'd'     : d
                })
            
            # Write record data + meta + grid to file
            rmn.fstecr(fileIdOut, r2)
            rmn.writeGrid(fileIdOut, gOut)
            print("CB41: Wrote interpolated P0 and its grid")
        except:
            pass
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileIdIn)
            rmn.fstcloseall(fileIdOut)
            #os.unlink(fileNameOut)  # Remove test file

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
