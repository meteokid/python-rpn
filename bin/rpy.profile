#!/usr/bin/env python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
import os, sys, glob, datetime, optparse
import matplotlib.pyplot as plt
import numpy as np

import rpnpy.librmn.all as rmn
import rpnpy.vgd.all as vgd

## def get_xmldict(varname):
##     import xml.etree.ElementTree as ET
##     AFSISIO = os.getenv('AFSISIO').strip()
##     xmldict = os.path.join(AFSISIO,'datafiles','constants','ops.variable_dictionary.xml')
##     tree = ET.parse(xmldict)
##     root = tree.getroot()
##     #for child in root: 
##     for child in root.iter('metvar'):
##         print child.find('nomvar').text.encode('ascii', 'replace')
##         for item in child.find('description').findall('short'): #/short lang=en
##             if item.get('lang') == 'en':
##                 print item.text.encode('ascii', 'replace')
##         print child.find('measure').find('real').find('units').text.encode('ascii', 'xmlcharrefreplace') #/real/units
        
##         ## for child1 in child:
##         ##         print child1.tag, child1.attrib


def get_data(fileId, xy, ll, varname, ip2=-1, datev=-1, verbose=False):

    xpts = [x[0] for x in xy]
    ypts = [x[1] for x in xy]
    lats = [l[0] for l in ll]
    lons = [l[1] for l in ll]

    # Get the vgrid definition present in the file
    if verbose:
        print("Getting vertical grid description")
    v = vgd.vgd_read(fileId)

    # Get the list of ip1 on thermo levels in this file
    tlvl = vgd.vgd_get(v, 'VIPT')
    mlvl = vgd.vgd_get(v, 'VIPM')
    if verbose:
        vkind    = vgd.vgd_get(v, 'KIND')
        vver     = vgd.vgd_get(v, 'VERS')
        VGD_KIND_VER_INV = dict((v, k) for k, v in vgd.VGD_KIND_VER.iteritems())
        vtype = VGD_KIND_VER_INV[(vkind,vver)]
        print("Found %d thermo and %d momentum levels of type %s" % (len(tlvl), len(mlvl), vtype))

    # Trim the list of thermo ip1 to actual levels in files for TT
    # since the vgrid in the file is a super set of all levels
    # and get their "key"
    tlvlkeys = []
    for ip1 in tlvl:
        (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
        key = rmn.fstinf(fileId, nomvar='TT', datev=datev, ip2=ip2, ip1=rmn.ip1_all(lval, lkind))
        if key is not None:
            tlvlkeys.append((ip1, key['key']))
            if datev == -1 or ip2 == -1:
                m = rmn.fstprm(key)
                datev = m['datev']
                ip2   = m['ip2']

    mlvlkeys = []
    for ip1 in mlvl:
        (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
        key = rmn.fstinf(fileId, nomvar='TT', datev=datev, ip2=ip2, ip1=rmn.ip1_all(lval, lkind))
        if key is not None:
            mlvlkeys.append((ip1, key['key']))
            if datev == -1 or ip2 == -1:
                m = rmn.fstprm(key)
                datev = m['datev']
                ip2   = m['ip2']
    if len(mlvlkeys) > len(tlvlkeys):
        if verbose: print("Using Momentum level list")
        tlvlkeys = mlvlkeys
    else:
        if verbose: print("Using Thermo level list")

    if verbose or len(tlvlkeys) == 0:
        print("Found %d records for %s ip2=%d datev=%d" % (len(tlvlkeys), varname, ip2, datev))
        
    if len(tlvlkeys) == 0:
        return None
    ip1list = [x[0] for x in tlvlkeys]

    # Read rfld and extract relevent points
    MB2PA = 100.
    rfldName = vgd.vgd_get(v, 'RFLD')
    shape = (len(xy)+len(ll),)
    rfld  = np.empty(shape, dtype=np.float32, order='F')
    rfld[:] = 1000. * MB2PA
    if rfldName:
        r = rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip1=ip2)
        if r is None:
            r = rmn.fstlir(fileId, nomvar=rfldName)
        if not r is None:
            if verbose:
                print("Read %s ip1=%d ip2=%d" % (r['nomvar'],r['ip1'],r['ip2']))
            g = rmn.readGrid(fileId, r)
            if len(xpts) > 0:
                v1 = rmn.gdxysval(g['id'], xpts, ypts, r['d'])
                rfld[0:len(xy)] = v1[:]
            if len(lats) > 0:
                v1 = rmn.gdllsval(g['id'], lats, lons, r['d'])
                rfld[len(xy):len(xy)+len(ll)] = v1[:]
            rfld[:] *= MB2PA
            
    # Compute pressure
    if verbose:
        print("Computing pressure levels")
    p = vgd.vgd_levels(v, rfld, ip1list)
    ## print p.shape, repr(rfld), len(ip1list)
    ## sys.exit(1)
    p[:,:] /= MB2PA

    #Allocate and read
    shape = (len(xy)+len(ll), len(tlvlkeys))
    data  = np.empty(shape, dtype=np.float32, order='F')
    ilvl = 0
    r['d'] = None
    g = None
    for ip1,key in tlvlkeys:
        r = rmn.fstluk(key, dataArray=r['d'])
        if g is None:
            g = rmn.readGrid(fileId, r)
        v = rmn.gdxysval(g['id'], xpts, ypts, r['d'])
        data[0:len(xy),ilvl] = v[:]
        v = rmn.gdllsval(g['id'], lats, lons, r['d'])
        data[len(xy):len(xy)+len(ll),ilvl] = v[:]
        if verbose:
            (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, r['ip1'])
            print("%3d: Read %s (ip2=%3d) (ip1=%9d ; %8f %s) (p=%8.2f) v=%s" %
                  (ilvl, r['nomvar'], r['ip2'], r['ip1'], lval,
                   rmn.kindToString(lkind), p[0,ilvl], str(data[:,ilvl])))
        ilvl +=1

    ts = r
    ts.update({
        'd' : data,
        'p' : p,
        'xy': xy,
        'll': ll
        })
    return ts


def plot_profile(varlist, title=None, axename=None, xzoom=None, yzoom=None, inLog=True):
    #TODO: get full name and units from dict
    #TODO: log or not log option
    if title is None:
        title = 'Profile of %s' % varlist[0]['nomvar']
    plt.title(title)
    if axename is None:
        axename = varlist[0]['nomvar']
    plt.xlabel(axename)        
    plt.ylabel('P [hPa]')

    font = {'family': 'monospace',#'serif',
            'weight': 'normal',
            'size': 'x-small',
            }

    markers = ('-','-.','--',':')
    colors = ('-','-.','--',':')
    imark = 0
    pmin = 9999.
    pmax = 0.
    for var in varlist:
        (varname, ip2, datev, xy, ll, d, p) = (var['nomvar'], var['ip2'], var['datev'], var['xy'], var['ll'], var['d'], var['p'])
        pmin = min(pmin, p.min())
        pmax = max(pmax, p.max())
        d1,d2 = rmn.newdate(rmn.NEWDATE_STAMP2PRINT, datev)
        vdatev = "%8.8d.%4.4d" % (d1,d2/10000)
        #TODO: filter data by yzoom
        for istat in xrange(len(xy)):
            (i,j) = xy[istat]
            y = p[istat,:][::-1]
            x = d[istat,:][::-1]
            plt.plot(x, y, markers[imark], label="%s xy:(%6.1f,%6.1f) datev=%s (%3dh)" % (varname,i,j, vdatev[0:12], ip2))
        for istat in xrange(len(ll)):
            (i,j) = ll[istat]
            y = p[len(xy)+istat,:][::-1]
            x = d[len(xy)+istat,:][::-1]
            plt.plot(x, y, markers[imark], label="%s ll:(%6.1f,%6.1f) datev=%s (%3dh)" % (varname,i,j, vdatev[0:13], ip2))
        imark += 1
        if imark >= len(markers): imark=0

    if isinstance(yzoom, (list, tuple)):
        if yzoom[0] is not None and yzoom[0] > pmin: pmin = yzoom[0]
        if yzoom[1] is not None and yzoom[1] < pmax: pmax = yzoom[1]
    # Tweak spacing to prevent clipping of ylabel
    #plt.subplots_adjust(left=0.15)
    #plt.xticks(np.arange(min(varlist[0]['t']), max(varlist[0]['t'])+1, 6.0))
    ax = plt.gca()
    if inLog: ax.set_yscale('log')
    ax.set_ylim((pmax,pmin))
    if isinstance(xzoom, (list, tuple)):
        ax.set_xlim(xzoom)
    ## ticks = (1000., 950., 900., 850., 800., 750., 500., 250., 100., 10.)
    ## ticks = np.arange(pmin,float(int(pmax/100.))*100.,50)
    ## ticks = list([pmin]) + list(np.arange(100.,pmax,100))
    ##ticks = list(np.arange(100.,pmax,100))
    ticks = [ x for x in (1000., 925., 850., 700., 500., 400., 300., 250., 200., 150., 100., 70.,50., 30., 20., 10.) if x >= pmin and x <= pmax]
    plt.yticks(ticks)
    ax.set_yticklabels([str(x) for x in ticks])
    plt.legend(prop=font)
    plt.grid()
    plt.show()


def xy2list(xy):
    """
    Split string with format
    x1,y1
    x1,y1,x2,y2
    (x1,y1),(x2,y2)
    """
    if xy is None or xy == '':
        return []
    xy2 = xy.replace('(','').replace(')','').split(',')
    return [(float(xy2[i*2]),float(xy2[i*2+1])) for i in xrange(len(xy2)//2)]
    

if __name__ == "__main__":

    ## ## fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
    ## fdate       = datetime.date.today().strftime('%Y%m%d') + '00_*'
    ## CMCGRIDF    = os.getenv('CMCGRIDF').strip()
    ## fileNameIn  = glob.glob(os.path.join(CMCGRIDF, 'prog', 'regeta', fdate))
    ## ## ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    ## ## fileNameIn = os.path.join(ATM_MODEL_DFILES, 'bcmk')

    ## rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
    ## fileId = rmn.fstopenall(fileNameIn, verbose=True)

    ## varname = 'TT'
    ## stations_xy = []#[(2,3),(20,30), (50,70)]
    ## stations_ll = [(45.,273.5),(46.,274.)]
    ## x1 = get_data(fileId, stations_xy, stations_ll, varname, -1, 12, verbose=True)
    ## x2 = get_data(fileId, stations_xy, stations_ll, varname, -1, 24, verbose=True)
    
    ## rmn.fstcloseall(fileId)
    
    ## plot_profile((x1,x2), yzoom=(500.,None), xzoom=(-10.,10.), inLog=False)
    ## sys.exit(0)

    # Command line arguments
    desc="Draw vertical profile"
    usage = """
    %prog -n varname [options] FILES
    """
    parser = optparse.OptionParser(usage=usage,description=desc)

    parser.add_option("-i","--input",dest="files",default='',
                      help="List of files or filename pattern")

    parser.add_option("-n","--nomvar",dest="nomvar",default='',
                      help="Variable Name")
    
    parser.add_option("","--ip2",dest="ip2",default=-1,
                      help="Filter records by ip2: 'v1, v2, v3'")
    parser.add_option("","--datev",dest="datev",default=None,
                      help="Filter records by Valid date (CMC date Stamp")
    parser.add_option("","--vdatev",dest="vdatev",default=None,
                      help="Filter records by Valid date (YYYYMMDD.HH)")

    parser.add_option("","--xy",dest="xy",default=None,
                      help="x,y position of stations to plot: '(x1,y1), (x2,y2),...'")
    parser.add_option("","--ll",dest="ll",default=None,
                      help="lat,lon of stations to plot: '(la1,lo1), (la2,lo2),...'")

    parser.add_option("","--title",dest="title",default=None,
                      help="Figure Title")
    parser.add_option("","--axe",dest="axename",default=None,
                      help="Figure Y axe name and utils")
    
    parser.add_option("","--xzoom",dest="xzoom",default=None,
                      help="min max value of the x axis: 'vmin, vmax'")
    parser.add_option("","--yzoom",dest="yzoom",default=None,
                      help="min max value of the y axis: 'vmin, vmax'")
    parser.add_option("","--inlog",dest="inlog",action="store_true",
                      help="Plot with log scale on the y axis")

    parser.add_option("-v","--verbose",dest="verbose",action="store_true",
                      help="Verbose mode")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        sys.stderr.write("\nError: Need to provided at least an input file.\n\n")
        parser.print_help()
        sys.exit(1)

    if not options.nomvar:
        sys.stderr.write("\nError: Need to specify the varname.\n\n")
        parser.print_help()
        sys.exit(1)
        
    if not (options.xy or options.ll):
        sys.stderr.write("\nError: Need to provided at least one station location.\n\n")
        parser.print_help()
        sys.exit(1)

    files = args if len(args) > 1 else glob.glob(args[0])
    xy    = xy2list(options.xy)
    ll    = xy2list(options.ll)
    datev = options.datev
    if datev is None and options.vdatev is not None:
        (yyyymmdd, hhmmsshh0) = options.vdatev.split('.')
        hhmmsshh = int(hhmmsshh0) * 10**(8-len(hhmmsshh0))
        datev = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, int(yyyymmdd), hhmmsshh)
    if datev is None:
        datev = -1

    ip2list =  -1 if options.ip2 is None else [int(x) for x in str(options.ip2).split(',')]
    xzoom = None if options.xzoom is None else [float(x) for x in str(options.xzoom).split(',')]
    yzoom = None if options.yzoom is None else [float(x) for x in str(options.yzoom).split(',')]

    if xzoom is not None:
        if not isinstance(xzoom,(list, tuple)) or len(xzoom) != 2:
            sys.stderr.write("\nError: Wrong xzoom specification.\n\n")
            parser.print_help()
            sys.exit(1)
    if yzoom is not None:
        if not isinstance(yzoom,(list, tuple)) or len(yzoom) != 2:
            sys.stderr.write("\nError: Wrong yzoom specification.\n\n")
            parser.print_help()
            sys.exit(1)

    try:
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        fileId = rmn.fstopenall(files, verbose=options.verbose)
    except:
        sys.stderr.write('ERROR: Problem opening: %s\n' % str(files))
        raise #sys.exit(1)

    try:
        #TODO: loop on datev
        varlist = [get_data(fileId, xy, ll, options.nomvar, ip2, datev, verbose=options.verbose)
                   for ip2 in ip2list]
    except:
        raise #pass
    finally:
        rmn.fstcloseall(fileId)

    if None in varlist:
        sys.stderr.write('ERROR: Problem getting requested fields\n')
        sys.exit(1)

    plot_profile(varlist, options.title, options.axename, xzoom, yzoom, options.inlog)
    
        
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
