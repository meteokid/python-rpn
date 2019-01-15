#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Barbara Casati <Barbara.Casati@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
"""
Interpolate RPNSTD rec to latlon points
"""
import sys
import optparse
import numpy as np
from scipy import interpolate
import rpnpy.librmn.all as rmn

if __name__ == "__main__":
    
    inttypelist = {
        'n' : rmn.EZ_INTERP_NEAREST,
        'l' : rmn.EZ_INTERP_LINEAR,
        'c' : rmn.EZ_INTERP_CUBIC
        }

    # Command line arguments
    desc="Interpolate RPNSTD rec to latlon points"
    usage = """
    %prog -f FSTFILE -n VARNAME -o OUTFILE [-l LOLAFILE] [-t INTTYPE]

    LOLAFILE format, one destination point per line:
       lon1  lat1
       lon2  lat2
       ...

    OUTPUT format
       lon1,  lat1, value1, extrap
       lon2,  lat2, value2, extrap
    """
    parser = optparse.OptionParser(usage=usage,description=desc)
    parser.add_option("-f","--fstfile",dest="fstfile",default="",
                      help="Name of RPN STD file containing records")
    parser.add_option("-n","--varname",dest="varname",default="",
                      help="Varname of the record to interpolate")
    parser.add_option("-l","--lolafile",dest="lolafile",default="/cnfs/dev/mrb/armn/armnbca/MesoVIC/VERA/VERA_8km_coordinates_lam_phi.txt",
                      help="Name of text file with destination coordinates, one 'lon lat' per line")
    parser.add_option("-t","--inttype",dest="inttype",default="linear",
                      help="Interpolation type: nearest, linear or cubic")
    parser.add_option("-o","--outfile",dest="outfile",default="",
                      help="Output file name")

    (options,args) = parser.parse_args()
    if not (options.varname and options.fstfile and options.outfile and options.lolafile and options.inttype):
        sys.stderr.write('Error: You need to specify a varname, an fst filename, an outfile name and a lolafile name.\n')
        parser.print_help()
        sys.exit(1)

    inttype = options.inttype[0].lower()
    if not (inttype in inttypelist.keys()):
        sys.stderr.write('Error: INTTYPE should be one of: nearest, linear or cubic.\n')
        parser.print_help()
        sys.exit(1)

    # Open and Read RPN STD file        
    try:
        rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)
        funit = rmn.fstopenall(options.fstfile,rmn.FST_RO)
        k = rmn.fstinf(funit,nomvar=options.varname)['key']
        data = rmn.fstluk(k)['d']
        meta = rmn.fstprm(k)
    except:
        raise rmn.RMNError('Problem opening/reading var=%s in File=%s' % (options.varname,options.fstfile))

    # Define input record grid
    try:
        meta['iunit'] = funit
        grid = rmn.ezqkdef(meta)
    except:
        raise rmn.RMNError('Problem defining input grid for var=%s in File=%s' % (options.varname,options.fstfile))

    # Read lat lon file
    try:    
        (lon,lat) = np.loadtxt(options.lolafile, dtype=np.float32, unpack=True)
        ## lat = np.asfortranarray(lat, dtype=np.float32)
        ## lon = np.asfortranarray(lon, dtype=np.float32)
    except:
        raise IOError('Problem reading the lola file: %s' % (options.lolafile))
    
    # Interpolate input data to lat lon and print
    rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE,inttypelist[inttype])
    #rmn.ezsetopt(rmn.EZ_OPT_EXTRAP_DEGREE,rmn.EZ_EXTRAP_MAX)

    (ni,nj) = data.shape
    outfile = open(options.outfile, 'w')
    for n in range(lat.size):
        (lat2,lon2) = (np.asarray([lat[n]]),np.asarray([lon[n]]))
        lldata2 = rmn.gdllsval(grid, lat2, lon2, data)
        xypos2  = rmn.gdxyfll(grid, lat2, lon2)
        extrap  = ''
        if (xypos2['x'][0] < 1. or xypos2['x'][0] > ni or
            xypos2['y'][0] < 1. or xypos2['y'][0] > nj):
            extrap='extrap'
        outfile.write("%9.5f, %9.5f, %9.5f, %s\n" %
                      (lon[n], lat[n], lldata2[0], extrap))
        del lldata2, lat2, lon2, xypos2
    outfile.close()

    # Close the RPN STD file
    try:
        rmn.fstcloseall(funit)
    except:
        pass
