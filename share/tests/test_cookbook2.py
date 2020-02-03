#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

class rpnpyCookbook2(unittest.TestCase):
    

    #---- Horizontal Interpolation


    def test_41(self):
        """
        Horizontal Interpolation
        
        See also:
        """
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        fileNameIn  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        fileNameOut = 'p0fstfile.fst'

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        try:
            # Create Destination grid
            # Note: Destination grid can also be read from a file
            gp = {
                'grtyp' : 'Z',
                'grref' : 'E',
                'ni'    : 90,
                'nj'    : 45,
                'lat0'  : 35.,
                'lon0'  : 250.,
                'dlat'  : 0.5,
                'dlon'  : 0.5,
                'xlat1' : 0.,
                'xlon1' : 180.,
                'xlat2' : 1.,
                'xlon2' : 270.
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
            r = rmn.fstlir(fileIdIn, nomvar='P0')
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
            os.unlink(fileNameOut)  # Remove test file


    def test_41qd(self):
        import os, sys, datetime
        import rpnpy.librmn.all as rmn
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        fileNameOut = 'p0fstfileqd.fst'
        fileIdIn    = rmn.fstopenall(os.getenv('CMCGRIDF')+'/prog/regeta/'+fdate)
        fileIdOut   = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        gOut = rmn.defGrid_ZE(90, 45, 35., 250., 0.5, 0.5, 0., 180., 1., 270.)
        r    = rmn.fstlir(fileIdIn, nomvar='P0')
        gIn  = rmn.readGrid(fileIdIn, r)
        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
        d  = rmn.ezsint(gOut, gIn, r)
        r2 = r.copy()
        r2.update(gOut)
        r2.update({'etiket':'my_etk', 'd':d})
        rmn.fstecr(fileIdOut, r2)
        rmn.writeGrid(fileIdOut, gOut)
        rmn.fstcloseall(fileIdIn)
        rmn.fstcloseall(fileIdOut)
        os.unlink(fileNameOut)  # Remove test file


    #---- Vertical Interpolation


    def test_51(self):
        """
        Vertical Interpolation
        
        See also:
        scipy.interpolate.interp1d
        """
        import os, sys, datetime
        import numpy as np
        from scipy.interpolate import interp1d as scipy_interp1d
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd

        MB2PA = 100.

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open Input file
        hour        = 48
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_0' + str(hour)
        CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        fileNameOut = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        try:
            fileIdIn = rmn.fstopenall(fileNameOut, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the input file: %s\n" % fileNameOut)
            sys.exit(1)

        try:
            # Get the vgrid def present in the file
            # and the full list of ip1
            # and the surface reference field name for the coor
            vIn        = vgd.vgd_read(fileIdIn)
            ip1listIn0 = vgd.vgd_get(vIn, 'VIPT')
            rfldNameIn = vgd.vgd_get(vIn, 'RFLD')
            vkind    = vgd.vgd_get(vIn, 'KIND')
            vver     = vgd.vgd_get(vIn, 'VERS')
            VGD_KIND_VER_INV = dict((v, k) for k, v in vgd.VGD_KIND_VER.items())
            vtype = VGD_KIND_VER_INV[(vkind,vver)]
            print("CB51: Found vgrid type=%s (kind=%d, vers=%d) with %d levels, RFLD=%s" %
                  (vtype, vkind, vver, len(ip1listIn0), rfldNameIn))

            # Trim the list of thermo ip1 to actual levels in files for TT
            # since the vgrid in the file is a super set of all levels
            # and get their "key"
            ip1Keys = []
            rshape  = None
            for ip1 in ip1listIn0:
                (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
                key = rmn.fstinf(fileIdIn, nomvar='TT', ip2=hour, ip1=rmn.ip1_all(lval, lkind))
                if key is not None:
                    print("CB51: Found TT at ip1=%d, ip2=%d" % (ip1, hour))
                    ip1Keys.append((ip1, key['key']))
                    if rshape is None:
                        rshape = key['shape']
            rshape = (rshape[0], rshape[1], len(ip1Keys))
            
            # Read every level for TT at ip2=hour, re-use 2d array while reading
            # and store the data in a 3d array
            # with lower level at nk, top at 0 as in the model
            r2d = {'d' : None}
            r3d = None
            k = 0
            gIn = None
            for ip1, key in ip1Keys:
                try:
                    r2d = rmn.fstluk(key, dataArray=r2d['d'])
                    print("CB51: Read TT at ip1=%d, ip2=%d" % (ip1, hour))
                    if r3d is None:
                        r3d = r2d.copy()
                        r3d['d'] = np.empty(rshape, dtype=r2d['d'].dtype, order='F')
                    r3d['d'][:,:,k] = r2d['d'][:,:]
                    k += 1
                    if gIn is None:
                        gIn = rmn.readGrid(fileIdIn, r2d)
                        print("CB51: Read the horizontal grid descriptors")
                except:
                    pass

            # Add the vgrid and the actual ip1 list in the r3d dict, update shape and nk
            r3d['vgd']     = vIn
            r3d['ip1list'] = [x[0] for x in ip1Keys]
            r3d['shape']   = rshape
            r3d['nk']      = rshape[2]

            # Read the Input reference fields
            rfldIn = None
            if rfldNameIn:
                rfldIn = rmn.fstlir(fileIdIn, nomvar=rfldNameIn, ip2=hour)
                if rfldNameIn.strip() == 'P0':
                    rfldIn['d'][:] *= MB2PA
                print("CB51: Read input RFLD=%s at ip2=%d [min=%7.0f, max=%7.0f]" % (rfldNameIn, hour, rfldIn['d'].min(), rfldIn['d'].max()))
                
        except:
            raise # pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileIdIn)

        # Define the destination vertical grid/levels
        try:
            lvlsOut     = (500.,850.,1000.)
            vOut        = vgd.vgd_new_pres(lvlsOut)
            ip1listOut  = vgd.vgd_get(vOut, 'VIPT')
            rfldNameOut = vgd.vgd_get(vIn, 'RFLD')
            rfldOut     = None  # in this case, Pressure levels, there are no RFLD
            print("CB51: Defined a Pres vgrid with lvls=%s" % str(lvlsOut))
        except:
            sys.stderr.write("Problem creating a new vgrid\n")
            sys.exit(1)

        # Get input and output 3d pressure cubes
        try:
            ## if rfldIn is None:
            ##     rfldIn = 
            pIn  = vgd.vgd_levels(vIn,  ip1list=r3d['ip1list'], rfld=rfldIn['d'])
            print("CB51: Computed input  pressure cube, k0:[min=%7.0f, max=%7.0f],  nk:[min=%7.0f, max=%7.0f]" % (pIn[:,:,0].min(), pIn[:,:,0].max(), pIn[:,:,-1].min(), pIn[:,:,-1].max()))
            if rfldOut is None:
                rfldOut = rfldIn  # provide a dummy rfld for array shape
            pOut = vgd.vgd_levels(vOut, ip1list=ip1listOut,     rfld=rfldOut['d'])
            print("CB51: Computed output pressure cube, k0:[min=%7.0f, max=%7.0f],  nk:[min=%7.0f, max=%7.0f]" % (pOut[:,:,0].min(), pOut[:,:,0].max(), pOut[:,:,-1].min(), pOut[:,:,-1].max()))
        except:
            raise
            sys.stderr.write("Problem computing pressure cubes\n")
            sys.exit(1)

        # Use scipy.interpolate.interp1d to vertically interpolate
        try:
            ## f = scipy_interp1d(fromLvls, toLvls, kind='cubic',
            ##                    assume_sorted=True, bounds_error=False,
            ##                    fill_value='extrapolate', copy=False)
            
            ## # Unfortunately, looks like interp1d take colomn data
            ## f = scipy_interp1d(pIn, r3d['d'], kind='cubic',
            ##                    bounds_error=False,
            ##                    fill_value='extrapolate', copy=False)
            ## r3dout = f(pOut)
            
            ## # Unfortunately, assume_sorted, 'extrapolate' not support in my version
            ## extrap_value = 'extrapolate' # -99999.
            ## # Way too slow, needs a C implementation
            extrap_value = -999.
            ## for j in range(rshape[1]):
            ##     for i in range(rshape[0]):
            ##         f = scipy_interp1d(pIn[i,j,:], r3d['d'][i,j,:],
            ##                            kind='cubic',
            ##                            bounds_error=False,
            ##                            fill_value=extrap_value, copy=False)
            ##         r1d = f(pOut[i,j,:])
            ##         #print i,j,r1d
        except:
            raise
            sys.stderr.write("Problem Interpolating data\n")
            sys.exit(1)

        ## # Open output file
        ## fileNameOut = 'ttfstfile.fst'
        ## try:
        ##     fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        ## except:
        ##     sys.stderr.write("Problem opening output file: %s\n" % fileNameOut)
        ##     sys.exit(1)

        ## # Write interpolated values to file with grid and vgrid
        ## try:
        ##     for k in range(ip1listOut):
        ##         ip1 = ip1listOut[k]
        ##         (ldiagval, ldiagkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
        ##         print("CB51: Wrote TT(%7.2f %s): mean=%6.1f, std=%6.1f, min=%6.1f, max=%6.1f]" %
        ##               (ldiagval, rmn.kindToString(ldiagkind),
        ##                r3d['d'][:,:,k].mean(), r3d['d'][:,:,k].std(),
        ##                r3d['d'][:,:,k].min(), r3d['d'][:,:,k].max()))

        ##     rmn.writeGrid(fileIdOut, gIn)
        ##     print("CB51: Wrote the horizontal grid descriptors")
            
        ##     vgd.vgd_write(vOut, fileIdOut)
        ##     print("CB51: Wrote the vertical grid descriptor")
        ## except:
        ##     pass
        ## finally:
        ##     # Close file even if an error occured above
        ##     rmn.fstcloseall(fileIdOut)

    
    #---- Read + 3D Interpolation + Compute + write
        

    def test_61(self):
        """
        Semi 3D Interpolation
        
        See also:
        """
        import rpnpy.librmn.all as rmn


    #TODO: burp examples

if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
