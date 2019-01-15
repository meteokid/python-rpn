#!/usr/bin/env python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Print a list of selected records in RPNStd file(s) along with requested meta and stats.

Examples:
    rpy.fstlist2 -k --stats \
        -i $ATM_MODEL_DFILES/bcmk/2009042700_000 \
        --nomvar '^^' '>>' '!!' HY  ## Excludes these nomvar
"""
import sys
import re
import argparse
import xml.etree.ElementTree

## # Open original file
## et = xml.etree.ElementTree.parse('file.xml')

## # Append new tag: <a x='1' y='abc'>body text</a>
## new_tag = xml.etree.ElementTree.SubElement(et.getroot(), 'a')
## new_tag.text = 'body text'
## new_tag.attrib['x'] = '1' # must be str; cannot be an int
## new_tag.attrib['y'] = 'abc'

## # Write back to file
## #et.write('file.xml')
## et.write('file_new.xml')

    
if __name__ == "__main__":

    et = xml.etree.ElementTree.parse('base-flow.xml')
    
    #et.write('flow.xml')
    
    # Command line arguments
    desc=""
    usage = """
    %(prog)s -i filename [options]

    """
    epilog=""

    parser = argparse.ArgumentParser(
        description=desc, usage=usage, epilog=epilog,
        prefix_chars='-+', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", dest="inputFile",
                        nargs='+', required=True, type=str, default=[],
                        help="Input RPN Std File name")

    parser.add_argument("-f", "--format", dest="format",
                        type=str, default=default_format,
                        help="Output Format")
    parser.add_argument("-k", "--addkeys", dest="addkeys",
                        action="store_true",
                        help="Change format to have key=value")
    parser.add_argument("--stats", dest="dostats",
                        action="store_true",
                        help="Add Statistics to the base format")
    
    ## parser.add_option("-s","--sort",dest="sort_key",default=None,
    ##                   help="Sort key")
    ## parser.add_option("-u","--unique",dest="sort_unique",action="store_true",
    ##                   help="Remove duplicate")
    ## parser.add_option("-r","--reverse",dest="sort_reverse",action="store_true",
    ##                   help="Reverse sorting order")

    parser.add_argument("--champs", dest="champs",
                        action="store_true",
                        help="List accepted format keys")

    parser.add_argument("++nomvar",  dest="f_nomvar",
                        nargs='*', type=str, default=[],
                        metavar='NOMVAR',
                        help="Filter records by nomvar values")
    parser.add_argument("++typvar",  dest="f_typvar",
                        nargs='*', type=str, default=[],
                        metavar='TYPVAR',
                        help="Filter records by typvar values")
    parser.add_argument("++etiket",  dest="f_etiket",
                        nargs='*', type=str, default=[],
                        metavar='ETIKET',
                        help="Filter records by etiket values")

    parser.add_argument("++ip1",  dest="f_ip1",
                        nargs='*', type=int, default=[],
                        metavar='IP1',
                        help="Filter records by ip1 values")
    parser.add_argument("++ip2",  dest="f_ip2",
                        nargs='*', type=int, default=[],
                        metavar='IP2',
                        help="Filter records by ip2 values")
    parser.add_argument("++ip3",  dest="f_ip3",
                        nargs='*', type=int, default=[],
                        metavar='IP3',
                        help="Filter records by ip3 values")

    parser.add_argument("++datev",  dest="f_datev",
                        nargs='*', type=int, default=[],
                        metavar='DATEV',
                        help="Filter records by Valid date (CMC date Stamp)")
    ## parser.add_argument("++vdatev", dest="f_vdatev",
    ##                     nargs='*', type=str, default=[],
    ##                     metavar='YYYYMMDD.hhmmss',
    ##                     help="Filter records by Valid date (YYYYMMDD.hhmmss)")

    parser.add_argument("--nomvar",  dest="e_nomvar",
                        nargs='*', type=str, default=[],
                        metavar='NOMVAR',
                        help="Filter out records by nomvar values")
    parser.add_argument("--typvar",  dest="e_typvar",
                        nargs='*', type=str, default=[],
                        metavar='TYPVAR',
                        help="Filter out records by typvar values")
    parser.add_argument("--etiket",  dest="e_etiket",
                        nargs='*', type=str, default=[],
                        metavar='ETIKET',
                        help="Filter out records by etiket values")
    
    parser.add_argument("--ip1",  dest="e_ip1",
                        nargs='*', type=int, default=[],
                        metavar='IP1',
                        help="Filter out records by ip1 values")
    parser.add_argument("--ip2",  dest="e_ip2",
                        nargs='*', type=int, default=[],
                        metavar='IP2',
                        help="Filter out records by ip2 values")
    parser.add_argument("--ip3",  dest="e_ip3",
                        nargs='*', type=int, default=[],
                        metavar='IP3',
                        help="Filter out records by ip3 values")
    
    parser.add_argument("--datev",  dest="e_datev",
                        nargs='*', type=int, default=[],
                        metavar='DATEV',
                        help="Filter out records by Valid date (CMC date Stamp)")
    ## parser.add_argument("--vdatev", dest="e_vdatev",
    ##                     nargs='*', type=str, default=[],
    ##                     metavar='YYYYMMDD.hhmmss',
    ##                     help="Filter out records by Valid date (YYYYMMDD.hhmmss)")

    args = parser.parse_args()

    if args.dostats:
        args.format += ':%mean%:%std%:%min%:%max%'
        
    if args.champs:
        print('%'+'%:%'.join(valid_keys)+'%')
        sys.exit(0)   
        
    keylist = ('nomvar', 'typvar', 'etiket', 'ip1', 'ip2', 'ip3', 'datev')
    (matchIn, matchOut) = ({}, {})
    for key in keylist:
        try:
            if key in ('nomvar', 'typvar', 'etiket'):
                matchIn[key] = [x.strip().lower() for x in getattr(args,'f_'+key)]
            else:
                matchIn[key] = getattr(args,'f_'+key)
        except:
            matchIn[key] = []
        try:
            if key in ('nomvar', 'typvar', 'etiket'):
                matchOut[key] = [x.strip().lower() for x in getattr(args,'e_'+key)]
            else:
                matchOut[key] = getattr(args,'e_'+key)
        except:
            matchOut[key] = []
        ## if args.verbose > 1:
        ##     print('Selecting {0:6s} in:{1:10s}, out:{2}'.format(key,str(matchIn[key]),str(matchOut[key])))

    try:
        print_list(args.inputFile, args.format, args.addkeys,  matchIn, matchOut)
    except:
        raise
        sys.exit(1)
        
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
