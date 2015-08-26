#!/usr/bin/env python
"""
Interpolate RPNSTD rec to latlon points

fst_to_lalo.py --lolafile /fs/cetus/fs2/mrb/armn/armnbca/MesoVIC/GEM2.5/Case1/2007061900_024 \
               --fstfile /cnfs/dev/mrb/armn/armnbca/MesoVIC/CaseStudies/VERA/VERA_8km_coordinates_lam_phi.txt \
               --varname PN \
               --inttype nearest \
               --outfile toto.txt
"""
import optparse,sys
import rpnpy.librmn.all as rmn
import numpy as np
from scipy import interpolate

if __name__ == "__main__":
    
    inttypelist = {
        'n' : rmn.EZ_INTERP_NEAREST,
        'l' : rmn.EZ_INTERP_LINEAR,
        'c' : rmn.EZ_INTERP_CUBIC
        }

    #==== Command line arguments
    desc="Interpolate RPNSTD rec to latlon points"
    usage = """
    %prog -f FSTFILE -n VARNAME -o OUTFILE [-l LOLAFILE] [-t INTTYPE]

    LOLAFILE format:
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
    parser.add_option("-l","--lolafile",dest="lolafile",default="/cnfs/dev/mrb/armn/armnbca/MesoVIC/CaseStudies/VERA/VERA_8km_coordinates_lam_phi.txt",
                      help="Name of text file with destination coordinates, one 'lon lat' per line")
    parser.add_option("-t","--inttype",dest="inttype",default="linear",
                      help="Interpolation type: nearest, linear or cubic")
    parser.add_option("-o","--outfile",dest="outfile",default="",
                      help="Output file name")

    (options,args) = parser.parse_args()
    if not (options.varname and options.fstfile and options.outfile and options.lolafile and options.inttype):
        print "ERROR: You need to specify a varname, an fst filename, an outfile name and a lolafile name"
        parser.print_help()
        sys.exit(1)

    inttype = options.inttype[0].lower()
    if not (inttype in inttypelist.keys()):
        print "ERROR: INTTYPE should be one of: nearest, linear or cubic"
        parser.print_help()
        sys.exit(1)


    #==== Open and Read RPN STD file        
    try:
        rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)
        funit = rmn.fstopenall(options.fstfile,rmn.FST_RO)
        k = rmn.fstinf(funit,nomvar=options.varname)['key']
        data = rmn.fstluk(k)['d']
        meta = rmn.fstprm(k)
    except:
        raise rmn.RMNError('Problem opening/reading var=%s in File=%s' % (options.varname,options.fstfile))

    
    #==== Define input record grid
    try:
        meta['iunit'] = funit
        grid = rmn.ezqkdef(meta)
    except:
        raise rmn.RMNError('Problem defining input grid for var=%s in File=%s' % (options.varname,options.fstfile))

    
    #==== Read lat lon file
    try:    
        (lon,lat) = np.loadtxt(options.lolafile, dtype=np.float32, unpack=True)
        lat = np.asfortranarray(lat, dtype=np.float32)
        lon = np.asfortranarray(lon, dtype=np.float32)
    except:
        raise IOError('Problem reading the lola file: %s' % (options.lolafile))

    
    #==== Interpolate input data to lat lon and print
    rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE,inttypelist[inttype])
    #rmn.ezsetopt(rmn.EZ_OPT_EXTRAP_DEGREE,rmn.EZ_EXTRAP_MAX)

    (ni,nj) = data.shape
    outfile = open(options.outfile, 'w')
    for n in xrange(lat.size):
        (lat2,lon2) = (np.asarray([lat[n]]),np.asarray([lon[n]]))
        lldata2 = rmn.gdllsval(grid, lat2, lon2, data)
        xypos2  = rmn.gdxyfll(grid, lat2, lon2)
        extrap  = ''
        if (xypos2['x'][0] < 1. or xypos2['x'][0] > ni or
            xypos2['y'][0] < 1. or xypos2['y'][0] > nj):
            extrap='extrap'
        ## print "%9.5f, %9.5f, %9.5f, %s" % \
        ##     (lon[i], lat[i], lldata2[0], extrap)
        outfile.write("%9.5f, %9.5f, %9.5f, %s\n" %
                      (lon[n], lat[n], lldata2[0], extrap))
        del lldata2, lat2, lon2, xypos2
    outfile.close()

    
   #==== tests

    lon = np.where(lon<0.,lon+360.,lon)

    ## ==== Test gdll, gdxyfll(on points): ok
    ## lalo   = rmn.gdll(grid)
    ## lag = lalo['lat']
    ## log = lalo['lon']

    ## kla = rmn.fstinf(funit,nomvar='LA')['key']
    ## lad = rmn.fstluk(kla)['d']
    ## klo = rmn.fstinf(funit,nomvar='LO')['key']
    ## lod = rmn.fstluk(klo)['d']

    ## xypos  = rmn.gdxyfll(grid, lag, log)
    ## xx = xypos['x']
    ## yy = xypos['y']
    ## for i in xrange(ni):
    ##     i2 = i+1
    ##     for j in xrange(nj):
    ##         j2 = j+1
    ##         if abs(xx[i,j]-i2) > 0.01 or abs(yy[i,j]-j2) > 0.01:
    ##             print "(%d,%d) (%f,%f) xy" % (i2, j2, xx[i,j]-i2,yy[i,j]-j2)
    ##         if abs(lag[i,j]-lad[i,j]) > 0.01 or abs(lag[i,j]-lad[i,j]) > 0.01:
    ##             print "(%d,%d) (%f,%f) ll" % (i2, j2, lag[i,j]-lad[i,j], lag[i,j]-lad[i,j])


    ## ==== Test gdxyfll,gdllfxy(off points) scalar: ok
    ## for n in xrange(lat.size):
    ##     (la0s,lo0s) = (np.asarray([lat[n]]),np.asarray([lon[n]]))
    ##     xypos2 = rmn.gdxyfll(grid, la0s, lo0s)
    ##     xxs = xypos2['x']
    ##     yys = xypos2['y']
    ##     llpts2 = rmn.gdllfxy(grid, xxs, yys)
    ##     las = llpts2['lat']
    ##     los = llpts2['lon']
    ##     if abs(las[0] - lat[n]) > 0.01 or abs(los[0] - lon[n]) > 0.01:
    ##         print "%d llo(%f,%f) != lls(%f,%f) ll(s-o)" % (n,lat[n],lon[n], las[0],los[0])
    ##     del xypos2,xxs,yys,llpts2,las,los,la0s,lo0s


    ## ==== Test gdxyfll,gdllfxy(off points) array: ok
    ## xypos  = rmn.gdxyfll(grid, lat, lon)
    ## xxa = xypos['x']
    ## yya = xypos['y']
    ## llpts = rmn.gdllfxy(grid, xxa, yya)
    ## laa = llpts['lat']
    ## loa = llpts['lon']

    ## for n in xrange(lat.size):
    ##     (la0s,lo0s) = (np.asarray([lat[n]]),np.asarray([lon[n]]))
    ##     xypos2 = rmn.gdxyfll(grid, la0s, lo0s)
    ##     xxs = np.ascontiguousarray(xypos2['x'], dtype=np.float32)
    ##     yys = np.ascontiguousarray(xypos2['y'], dtype=np.float32)
    ##     if abs(laa[n] - lat[n]) > 0.01 or abs(loa[n] - lon[n]) > 0.01:
    ##         print "%d llo(%f,%f) != lla(%f,%f) ll(a-o) : xya(%f,%f) xys(%f,%f) nij(%d,%d)" % (n, lat[n],lon[n],laa[n],loa[n],xxa[n],yya[n],xxs[0],yys[0],ni,nj)
    ##     del xypos2,xxs,yys,la0s,lo0s


        
    ## for n in xrange(lat.size):
    ##     if abs(laa[n] - lat[n]) > 0.01 or abs(loa[n] - lon[n]) > 0.01:
    ##         print "%d lla(%f,%f) != llo(%f,%f) ll(a-o)" % (n, laa[n],loa[n],lat[n],lon[n])

    #==== print interpolation result
    ## data1d = data.flatten('F')
    ## lon1d = lod[:,0].flatten('F')
    ## lat1d = lad[0,:].flatten('F')
    ## print lon1d.shape, lat1d.shape, data.shape
    ## f = interpolate.interp2d(lon1d, lat1d, data,
    ##                          kind='linear', copy=True)
    ## lldata = rmn.gdllsval(grid, lat2, lon2, data1d)
    
        
    ## print 1,lalo['lon'][0,nj-1], lalo['lon'][ni-1,nj-1]
    ## print 2,lalo['lon'][0,0],    lalo['lon'][ni-1,0]
    ## print 3,lalo['lat'][0,nj-1], lalo['lat'][ni-1,nj-1]
    ## print 4,lalo['lat'][0,0],    lalo['lat'][ni-1,0]
    ## print 5,np.min(lalo['lat']-lad), np.max(lalo['lat']-lad), np.average(lad)
    ## print 6,np.min(lalo['lon']-lod), np.max(lalo['lon']-lod), np.average(lod)

    ## print 7, np.min(lad[0,:]), np.max(lad[0,:])
    ## print 8, np.min(lad[:,0]), np.max(lad[:,0])
    ## print 9, np.min(lod[:,0]), np.max(lod[:,0])
    ## print 10,np.min(lod[0,:]), np.max(lod[0,:])


    #==== Close the RPN STD file
    try:
        rmn.fstcloseall(funit)
    except:
        pass
    
# ./share/fst_to_lalo.py -f /fs/cetus/fs2/mrb/armn/armnbca/MesoVIC/GEM2.5/Case1/2007061900_024 -n TJ
