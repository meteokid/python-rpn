#!/usr/bin/env python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Example:
   rpy.profile  -v --inlog \
        --var 'nomvar=tt, ip2=0' 'nomvar=tt, ip2=48' \
        --ll '45.,273.5' '46.,274.' \
        -i $CMCGRIDF/prog/regeta/$(date '+%Y%m%d')00_*
"""

import sys
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


def get_levels_keys(fileId, nomvar, datev=-1, ip2=-1, ip3=-1,
                    typvar=' ', etiket=' ',
                    vGrid=None, thermoMom='VIPT', verbose=False):
    """
    """
    #TODO: try to get the sorted ip1 list w/o vgrid, because vgrid doesn;t support 2 different vertical coor in the same file (or list of linked files)
    
    # Get the vgrid definition present in the file
    if vGrid is None:
        if verbose:
            print("Getting vertical grid description")
        vgd.vgd_put_opt('ALLOW_SIGMA', vgd.VGD_ALLOW_SIGMA)
        vGrid = vgd.vgd_read(fileId)
        
    vip  = vgd.vgd_get(vGrid, thermoMom)
    if verbose:
        vkind    = vgd.vgd_get(vGrid, 'KIND')
        vver     = vgd.vgd_get(vGrid, 'VERS')
        VGD_KIND_VER_INV = dict((val, key) for key, val in vgd.VGD_KIND_VER.iteritems())
        vtype = VGD_KIND_VER_INV[(vkind,vver)]
        print("Found %d %s levels of type %s" % (len(vip), thermoMom, vtype))

    # Trim the list of ip1 to actual levels in files for nomvar
    # since the vgrid in the file is a super set of all levels
    # and get their "key"
    vipkeys = []
    for ip1 in vip:
        (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
        key = rmn.fstinf(fileId, nomvar=nomvar, datev=datev, ip2=ip2, ip3=ip3,
                         ip1=rmn.ip1_all(lval, lkind),
                         typvar=typvar, etiket=etiket)
        if key is not None:
            vipkeys.append((ip1, key['key']))
            if datev == -1 or ip2 == -1 or ip3 == -1 or typvar.strip() == '' or etiket.strip() == '':
                meta   = rmn.fstprm(key)
                datev  = meta['datev']
                ip2    = meta['ip2']
                ip3    = meta['ip3']
                typvar = meta['typvar']
                etiket = meta['etiket']
    return (nomvar, datev, ip2, ip3, typvar, etiket, vGrid, vipkeys)


def get_data_profile(fileId, xy, ll, 
             nomvar, datev=-1, ip2=-1, ip3=-1, typvar=' ', etiket=' ',
             verbose=False):
    """
    """
    # Get the list of ip1 on thermo, momentum levels in this file
    vGrid = None
    (nomvar, datev, ip2, ip3, typvar, etiket, vGrid, viptkeys) = \
        get_levels_keys(fileId, nomvar, datev, ip2, ip3, typvar, etiket,
                        vGrid=vGrid, thermoMom='VIPT', verbose=verbose)

    (nomvar, datev, ip2, ip3, typvar, etiket, vGrid, vipmkeys) = \
        get_levels_keys(fileId, nomvar, datev, ip2, ip3, typvar, etiket,
                        vGrid=vGrid, thermoMom='VIPM', verbose=verbose)

    if len(vipmkeys) > len(viptkeys):
        if verbose: print("Using Momentum level list")
        viptkeys = vipmkeys
    elif verbose: print("Using Thermo level list")

    if verbose or len(viptkeys) == 0:
        print("Found %d records for %s ip2=%d datev=%d etiket=%s" % (len(viptkeys), nomvar, ip2, datev, etiket))
        
    if len(viptkeys) == 0:
        return None

    ip1list = [x[0] for x in viptkeys]
    xpts = [x[0] for x in xy]
    ypts = [x[1] for x in xy]
    lats = [l[0] for l in ll]
    lons = [l[1] for l in ll]

    # Read rfld and extract relevent points
    MB2PA = 100.
    rfldName = vgd.vgd_get(vGrid, 'RFLD')
    shape = (len(xy)+len(ll),)
    rfld  = np.empty(shape, dtype=np.float32, order='F')
    rfld[:] = 1000. * MB2PA
    if rfldName:
        rec = rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2, ip3=ip3,
                         typvar=typvar, etiket=etiket)
        if rec is None:
            rec = rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2,
                           typvar=typvar, etiket=etiket)
        if rec is None:
            rec = rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2,
                           etiket=etiket)
        if rec is None:
            rec = rmn.fstlir(fileId, nomvar=rfldName, datev=datev, ip2=ip2)
        if rec is None:
            rec = rmn.fstlir(fileId, nomvar=rfldName, datev=datev)
        if rec is None:
            rec = rmn.fstlir(fileId, nomvar=rfldName)
        if not rec is None:
            if verbose:
                print("Read %s ip1=%d ip2=%d ip3=%d typv=%s etk=%s" %
                      (rec['nomvar'], rec['ip1'], rec['ip2'], rec['ip3'],
                       rec['typvar'], rec['etiket']))
            g = rmn.readGrid(fileId, rec)
            if len(xpts) > 0:
                v1 = rmn.gdxysval(g['id'], xpts, ypts, rec['d'])
                rfld[0:len(xy)] = v1[:]
            if len(lats) > 0:
                v1 = rmn.gdllsval(g['id'], lats, lons, rec['d'])
                rfld[len(xy):len(xy)+len(ll)] = v1[:]
            rfld[:] *= MB2PA
            
    # Compute pressure
    if verbose:
        print("Computing pressure levels")
    pmb = vgd.vgd_levels(vGrid, rfld, ip1list)
    pmb[:,:] /= MB2PA

    #Allocate and read
    shape = (len(xy)+len(ll), len(viptkeys))
    data  = np.empty(shape, dtype=np.float32, order='F')
    ilvl  = 0
    rec   = {}
    rec['d'] = None
    hGrid = None
    for ip1,key in viptkeys:
        rec = rmn.fstluk(key, dataArray=rec['d'])
        if hGrid is None:
           hGrid = rmn.readGrid(fileId, rec)
        vals = rmn.gdxysval(hGrid['id'], xpts, ypts, rec['d'])
        data[0:len(xy),ilvl] = vals[:]
        vals = rmn.gdllsval(hGrid['id'], lats, lons, rec['d'])
        data[len(xy):len(xy)+len(ll),ilvl] = vals[:]
        if verbose:
            (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, rec['ip1'])
            print("%3d: Read %s ip1=(%9d, %8f%s) ip2=%3d ip3=%5d typv=%s etk=%s (p=%8.2f) v=%s" %
                  (ilvl, rec['nomvar'], rec['ip1'], lval, rmn.kindToString(lkind),
                   rec['ip2'], rec['ip3'], rec['typvar'], rec['etiket'], 
                   pmb[0,ilvl], str(data[:,ilvl])))
        ilvl +=1

    rec.update({
        'd'   : data,
        'pmb' : pmb,
        'xy'  : xy,
        'll'  : ll
        })
    return rec


def plot_profile(varlist, title=None, axename=None, xzoom=None, yzoom=None, inLog=True):
    """
    """
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
        if var is None:
            continue
        (varname, ip2, datev, xy, ll, d, pmb, etk) = \
            (var['nomvar'], var['ip2'], var['datev'], var['xy'], var['ll'],
             var['d'], var['pmb'], var['etiket'])
        pmin = min(pmin, pmb.min())
        pmax = max(pmax, pmb.max())
        d1,d2 = rmn.newdate(rmn.NEWDATE_STAMP2PRINT, datev)
        vdatev = "%8.8d.%4.4d" % (d1,d2/10000)
        #TODO: filter data by yzoom
        for istat in xrange(len(xy)):
            (i,j) = xy[istat]
            y = pmb[istat,:][::-1]
            x = d[istat,:][::-1]
            plt.plot(x, y, markers[imark],
                     label="%s xy:(%6.1f,%6.1f) datev=%s (%3dh) %12s" %
                     (varname,i,j, vdatev[0:12], ip2, etk))
        for istat in xrange(len(ll)):
            (i,j) = ll[istat]
            y = pmb[len(xy)+istat,:][::-1]
            x = d[len(xy)+istat,:][::-1]
            plt.plot(x, y, markers[imark],
                     label="%s ll:(%6.1f,%6.1f) datev=%s (%3dh) %12s" %
                     (varname,i,j, vdatev[0:13], ip2, etk))
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
    if isinstance(xy, str):
        xy = (xy, )
    xy2 = []
    for xy1 in xy:
        xy2 += xy1.replace('(','').replace(')','').split(',')
    return [(float(xy2[i*2]),float(xy2[i*2+1])) for i in xrange(len(xy2)//2)]


def varstr2dict(var):
    """
    Split string with format
    ...
    """
    mydict = {
        'nomvar' : ' ',
        'datev' : -1,
        'ip2' : -1,
        'ip3' : -1,
        'typvar' : ' ',
        'etiket' : ' '
        }
    mydict.update(dict([[k.strip() for k in keyval.split('=',1)] for keyval in var.split(',')]))
    if 'vdatev' in mydict.keys():
        (yyyymmdd, hhmmsshh0) = mydict['vdatev'].split('.')
        hhmmsshh = int(hhmmsshh0) * 10**(8-len(hhmmsshh0))
        datev = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, int(yyyymmdd), hhmmsshh)
        if datev is not None:
            mydict['datev'] = datev
    for k in ('ip2', 'ip3', 'datev'):
        mydict[k] = int(mydict[k])
    return mydict


if __name__ == "__main__":
    import argparse
    import glob

    # Command line arguments
    desc="Draw vertical profile"
    usage = """
    %(prog)s -i filenames [options]
    """
    epilog = "For var, accepted keywords are: \n   nomvar, ip2, ip3, typvar, etiket, datev, vdatev\nnomvar is mandatory\nany ommited keyword is equivalent to wildcard (select any)\ndatev is the CMC datetime stamp\nvdatev is the date human readable format: YYYYMMDD.hhmm\nif both datev and vdatev are provided, datev is ignored (using vdatev)"
    parser = argparse.ArgumentParser(
        description=desc, usage=usage, epilog=epilog,
        prefix_chars='-+', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input", dest="inputFile",
                        nargs='+', required=True, type=str, default=[],
                        help="Input RPN Std File name")

    parser.add_argument("--var", dest="varlist",
                        nargs='+', required=True, type=str, default=[],
                        metavar="'nomvar=AA, ip2=99, ...'",
                        help="Var selection criteria, e.g.: 'nomvar=AA, ip2=99, ip3=-1, typvar=A, etiket=myetk, datev=-1, vdatev=20160314.0000'")


    ## parserg1 = parser.add_mutually_exclusive_group()
    parser.add_argument("--xy", dest="xy",
                        nargs='+', type=str, default=[],
                        metavar="'x,y'",
                        help="x,y position of stations to plot: 'x1,y1' 'x2,y2' ...")

    parser.add_argument("--ll", dest="ll",
                        nargs='+', type=str, default=[],
                        metavar="'lat,lon'",
                        help="lat,lon of stations to plot: 'la1,lo1' 'la2,lo2' ...")

    
    parser.add_argument("--title", dest="title", default=None,
                        help="Figure Title")
    parser.add_argument("--axe", dest="axename", default=None,
                        help="Figure Y axe name and utils")

    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="Verbose mode")
        
    parser.add_argument("--xzoom", dest="xzoom", default=None, type=str,
                        help="min max value of the x axis: 'vmin, vmax'")
    parser.add_argument("--yzoom", dest="yzoom", default=None, type=str,
                        help="min max value of the y axis: 'vmin, vmax'")
    parser.add_argument("--inlog", dest="inlog", action="store_true",
                        help="Plot with log scale on the y axis")

    args = parser.parse_args()

    if len(args.xy) + len(args.ll) == 0:
        sys.stderr.write("\nError: Need to provide at least one xy or ll.\n\n")
        parser.print_help()
        sys.exit(1)

    files = args.inputFile if len(args.inputFile) > 1 else glob.glob(args.inputFile[0])
    xy    = xy2list(args.xy)
    ll    = xy2list(args.ll)
    xzoom = None if args.xzoom is None else [float(x) for x in str(args.xzoom).split(',')]
    yzoom = None if args.yzoom is None else [float(x) for x in str(args.yzoom).split(',')]

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
        fileId = rmn.fstopenall(files, verbose=args.verbose)
    except:
        sys.stderr.write('ERROR: Problem opening: %s\n' % str(files))
        raise #sys.exit(1)

    try:
        varlist = []
        for var in args.varlist:
            vardict = varstr2dict(var)
            if not vardict['nomvar'].strip():
                sys.stderr.write("\nWARNING: Skippping, no nomvar: {0}.\n\n".\
                                 format(var))
                continue
            varlist.append(
                get_data_profile(
                    fileId, xy, ll, vardict['nomvar'], vardict['datev'],
                    vardict['ip2'], vardict['ip3'], vardict['typvar'],
                    vardict['etiket'], verbose=args.verbose))
    except:
        raise #pass
    finally:
        rmn.fstcloseall(fileId)

    if None in varlist:
        sys.stderr.write('ERROR: Problem getting requested fields\n')
        sys.exit(1)

    plot_profile(varlist, args.title, args.axename, xzoom, yzoom, args.inlog)
    
        
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
