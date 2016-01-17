#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Kristjan Onu <Kristjan.Onu@canada.ca>
"""
Use Basemap to plot data from a RPN Standard File not on regular lat-lon grid

Usage: 
   export CMCGRIDF=???
   plot-irregular.py

See Also:
   https://basemaptutorial.readthedocs.org/en/latest/
"""
import os
import sys
import datetime
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as basemap
import rpnpy.librmn.all as rmn

if __name__ == "__main__":

    #forecast_name = datetime.date.today().strftime('%Y%m%d') + '12_040'
    forecast_name = '2015122312_040'
    CMCGRIDF = os.environ['CMCGRIDF'].strip()
    if not CMCGRIDF:
        sys.stderr.write('Error: Need to define CMCGRIDF env.var. to get the data files.\n')
        sys.exit(1)        
    my_file = os.path.join(CMCGRIDF, 'prog', 'gsloce', forecast_name)

    try:
        funit = rmn.fstopenall(my_file, rmn.FST_RO)
    except:
        sys.stderr.write('Error: Unable to open file: '+my_file+'\n')
        sys.exit(1)

    varname = 'tm2'
    try:
        sst_rec      = rmn.fstlir(funit, nomvar=varname, typvar='P@')
        sst_mask_rec = rmn.fstlir(funit, nomvar=varname, typvar='@@')
    except:
        sys.stderr.write('Error: Problem reading fields '+varname+' in file: '+my_file+'\n')
        sys.exit(1)

    # Prefered method to get grid lat, lon
    #    would work on any RPNSTD grid type (except 'X')
    ## try:
    ##     sst_rec['iunit'] = funit
    ##     sst_gridid = rmn.ezqkdef(sst_rec)  # use ezscint to retreive full grid
    ##     gridLatLon = rmn.gdll(sst_gridid)
    ##     lat = gridLatLon['lat']
    ##     lon = gridLatLon['lon']
    ## except:
    ##     sys.stderr.write('Error: Problem getting grid info for '+varname+' in file: '+my_file+'\n')
    ##     sys.exit(1)

    # Less prefered method to get grid lat, lon
    #    since it relies on reverse engeneering of grid def and ^^ >> values
    #    would only work with L grids encoded as a Z grid
    #    and when only one set of ^^ >> are in the file
    #    In this case it's the only way since the fields are not
    #    properly geo-referenced (grtyp='X')
    try:
        lat = rmn.fstlir(funit, nomvar='^^', dtype=np.float32)['d']
        lon = rmn.fstlir(funit, nomvar='>>', dtype=np.float32)['d']
    except:
        sys.stderr.write('Error: Problem getting grid info for '+varname+' in file: '+my_file+'\n')
        sys.exit(1)
            
    rmn.fstcloseall(funit)

    sst = np.ma.array(sst_rec['d'], mask=np.logical_not(sst_mask_rec['d']))

    crnr = np.array([[-72.5, 45], [-55, 52.5]])
    lat_ts = np.mean(crnr[:, 1])
    bmap = basemap.Basemap(projection='merc', llcrnrlat=crnr[0, 1],
                           urcrnrlat=crnr[1, 1], llcrnrlon=crnr[0, 0],
                           urcrnrlon=crnr[1, 0], lat_ts=lat_ts,
                           resolution='h')
    bmap.drawcoastlines()
    bmap.fillcontinents(color='.75', lake_color='1')
    parallels = np.linspace(crnr[0, 1], crnr[1, 1], 5).astype('int')
    bmap.drawparallels(parallels, labels=[True, False, False, False])
    meridians = np.linspace(crnr[0, 0], crnr[1, 0], 5).astype('int')
    bmap.drawmeridians(meridians, labels=[False, False, True, False])

    bmap.drawmapboundary(fill_color='1')
    bmap.drawmapscale(crnr[0, 0] + .1*np.diff(crnr[:, 0]),
                      crnr[0, 1] + .1*np.diff(crnr[:, 1]),
                      np.mean(crnr[:, 0]), np.mean(crnr[:, 1]),
                      100, barstyle='fancy')

    x, y = bmap(lon, lat)

    sst = scipy.constants.K2C(sst)

    sst_a = bmap.pcolor(x.ravel(), y.ravel(), sst.ravel(), tri=True)

    subsample = 5
    lines = bmap.plot(x.ravel()[::subsample], y.ravel()[::subsample])
    plt.setp(lines[0], 'linestyle', 'None')
    plt.setp(lines[0], 'marker', ',')
    plt.setp(lines[0], 'markerfacecolor', 'black')

    cbar = bmap.colorbar(sst_a, location='bottom', pad='5%')
    cbar.set_label(u'°C')

    plt.show()
    plt.savefig('pcolor_demo.png', dpi=300, transparent=True,
                              bbox_inches='tight')
