#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Barbara Casati <Barbara.Casati@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
"""
Export a single RPN STD field to a text file
"""
import sys
import optparse
import numpy as np
from scipy import interpolate
import rpnpy.librmn.all as rmn

if __name__ == "__main__":
    
    #==== Command line arguments
    desc="Export a single RPN STD field to a text file"
    usage = """
    %prog  -i FSTFILE.fst
           -o OUTFILE.txt
           --varname VARNAME
           [--ip1 IP1]
           [--ip2 IP2]
           [--datev DATEV]
           """
    parser = optparse.OptionParser(usage=usage,description=desc)
    parser.add_option("-i","--fstfile",dest="fstfile",default="",
                      help="Name of RPN STD file containing records")
    parser.add_option("-o","--outfile",dest="outfile",default="",
                      help="Output file name")
    parser.add_option("-n","--varname",dest="varname",default="",
                      help="Varname of the record to extract")
    parser.add_option("-l","--ip1",dest="ip1",default="-1",
                      help="ip1 of the record to extract (default=-1)")
    parser.add_option("-t","--ip2",dest="ip2",default="-1",
                      help="ip2 of the record to extract (default=-1)")
    parser.add_option("-d","--datev",dest="datev",default="-1",
                      help="valid date of the record to extract (encoded, default=-1)")

    (options,args) = parser.parse_args()
    if not (options.varname and options.fstfile and options.outfile):
        sys.stderr.write('Error: You need to specify a varname, an fst filename, an outfile name.\n')
        parser.print_help()
        sys.exit(1)

    #==== Open and Read RPN STD file
    print("+ Reading %s from file: %s" %  (options.varname, options.fstfile))
    try:
        #rmn.fstopt(rmn.FSTOP_MSGLVL, rmn.FSTOPI_MSG_CATAST)
        funit = rmn.fstopenall(options.fstfile,rmn.FST_RO)
    except:
        raise rmn.RMNError('Problem opening File=%s' % (options.varname,options.fstfile))

    try:
        k = rmn.fstinf(funit,
                       nomvar= options.varname,
                       ip1   = int(options.ip1),
                       ip2   = int(options.ip2),
                       datev = int(options.datev))['key']
    except:
        raise rmn.RMNError('Record not found var=%s, ip1=%s, ip2=%s, datev=%s in File=%s' %
                           (options.varname,options.ip1,options.ip2,options.datev,options.fstfile))

    try:
        data = rmn.fstluk(k)['d']
    except:
        raise rmn.RMNError('Problem reading var=%s, ip1=%s, ip2=%s, datev=%s in File=%s' %
                           (options.varname,options.ip1,options.ip2,options.datev,options.fstfile))

    try:
        rmn.fstcloseall(funit)
    except:
        pass

    #==== print in text file
    print("+ Writing %s to file: %s" %  (options.varname, options.outfile))
    (ni,nj) = data.shape
    outfile = open(options.outfile, 'w')
    for jj in range(nj):
        for ii in range(ni):
            outfile.write("%e " % (data[ii,jj]))
        outfile.write("\n")
    outfile.close()

