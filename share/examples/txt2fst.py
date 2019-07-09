#!/usr/bin/env python
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Convert a specialy formated text data file to an RPNStd File

Based on example found at:
https://wiki.cmc.ec.gc.ca/wiki/Python-RPN/2.1/examples#Edit:_New_file_and_records_from_scratch
"""

import os
import sys
import argparse
import gzip
import numpy as np
import rpnpy.librmn.all as rmn


TXT2FSTDICT = {
    'PA' : {'name':'PR', 'mul':1./1000., 'add':0.}
    }


def readTextFileData(filename):
    """
    Read specialy formated text data file
    """
    p = {}
    isdata =  False
    with gzip.open(filename, 'r') as f:
        for l in f:
            if l.strip().lower() == 'data':
                if (p['FieldType'], p['DataFormat']) != ('Grid', 'Comma_Delim_Ascii'):
                    raise ValueError('Expecting Comma delimited Gridded data')
                if not ('Width' in p.keys() and 'Height' in p.keys()):
                    raise KeyError('Missing Width and Height descriptors')
                isdata = True
                continue
            if isdata:
                p['Data'] = np.array(l.strip().replace('\t',' ').split(','),
                                     order='F', dtype=np.float32)
                # Data order: we expect Fortran order from lower-left corner to upper-right
                p['Data'] = np.reshape(p['Data'],
                                       (int(p['Width']), int(p['Height'])),
                                       order='F')
                break
            k, v = l.strip().replace('\t',' ').split(' ', 2)
            p[k] = v
    #TODO: make sure all needed keys are present
    return p


def txtFile2Rec(filename, verbose=False):
    """
    Read specialy formated text data file
    """
    if not os.path.isfile(filename):
        sys.stderr.write('WARNING: No such file (skipping): {}\n'.format(filename))
        raise IOError
    if not rmn.wkoffit(filename) in (rmn.WKOFFIT_TYPE_LIST['INCONNU'],
                                     rmn.WKOFFIT_TYPE_LIST['ASCII']):
        sys.stderr.write('WARNING: Not a text file (skipping): {}\n'.
                         format(filename))
        raise IOError

    if verbose:
        sys.stdout.write('Reading text file (skipping): {}\n'.format(filename))
    try:
        txtRec = readTextFileData(filename)
        if verbose:
            for k,v in txtRec.items():
                if k != 'Data':
                    sys.stdout.write('\t{} = {}\n'.format(k,v))
            sys.stdout.write('\tData shape : {}\n'.
                             format(txtRec['Data'].shape))
            sys.stdout.write('\tData: min: {}, max: {}, mean:{}\n'.
                             format(txtRec['Data'].min(),
                                    txtRec['Data'].max(),
                                    txtRec['Data'].mean()))
    except:
        sys.stderr.write('WARNING: Problem reading file (skipping): {}\n'.
                         format(filename))
        raise Exception
    return txtRec


def defineZPSGrid(txtRec, verbose=False):
    """
    Define a North polar sterographic grid, ezscint encoded

    See: https://wiki.cmc.ec.gc.ca/wiki/Python-RPN/2.1/rpnpy/librmn/grids#defGrid_PS
    """
    if txtRec['Projection'] != 'PolarStereographic':
        raise ValueError('Unknown projection: {}'.format(txtRec['Projection']))
    if int(float(txtRec['TrueLatitude'])) != 60:
        raise ValueError('Cannot encode TrueLatitude of: {}'.format(txtRec['TrueLatitude']))

    if verbose:
        sys.stdout.write('Defining a North PS grid: {}\n'.format(filename))
    ni, nj = txtRec['Data'].shape
 
    ## First define a base projection as a grid centered on the N-pole
    pi, pj = float(ni)/2., float(nj)/2.
    gp = {
        'grtyp' : 'N',
        'north' : True,
        'ni'    : ni,
        'nj'    : nj,
        'pi'    : pi,
        'pj'    : pj,
        'd60'   : float(txtRec['Scale']) * 1000.,
        'dgrw'  : float(txtRec['ReferenceLongitude']),
     }
    g = rmn.encodeGrid(gp)

    ## Second correct the base grid position
    lon0 = float(txtRec['LonCentre'])
    lat0 = float(txtRec['LatCentre'])
    xy = rmn.gdxyfll(g['id'], lat0, lon0)
    gp['pi'] = float(ni) - xy['x'][0]
    gp['pj'] = float(nj) - xy['y'][0]
    g = rmn.encodeGrid(gp)
    return g

 
def txt2fstRec(txtRec, verbose=False):
    """
    Create a new RPNStdRec from data in numpy array and some metadata

    See: https://wiki.cmc.ec.gc.ca/wiki/Python-RPN/2.1/examples#Edit:_New_file_and_records_from_scratch
    """
    if verbose:
        sys.stdout.write('Converting to RPNStd Rec: {}\n'.format(filename))
    try:
        nomvar = TXT2FSTDICT[txtRec['Title']]['name']
        mul    = TXT2FSTDICT[txtRec['Title']]['mul']
        add    = TXT2FSTDICT[txtRec['Title']]['add']
    except:
        nomvar, mul, add = txtRec['Title'], 1., 0.
    yyyymmdd = int(txtRec['ValidTime'][0:8])
    hhmmss00  = int(txtRec['ValidTime'][8:14]+'00')
    #TODO: get ip2, deet from txtRec['MinorProductParameters']
    ip2      = 0     # Forecast of 0h
    deet     = 3600  # Timestep in sec
    npas     = int(ip2*3600/deet) # Step number
    etiket   = txtRec['MajorProductType']
    datyp    = rmn.dtype_numpy2fst(txtRec['Data'].dtype)
    (ni, nj) = txtRec['Data'].shape

    fstRec = rmn.FST_RDE_META_DEFAULT.copy()       # Copy default record meta
    fstRec.update(defineZPSGrid(txtRec, verbose))  # Update with grid info
    fstRec.update({                                # Update with specific meta and data array
        'typvar': 'A',
        'nomvar': nomvar,
        'ip1'   : 0,      # Level
        'ip2'   : ip2,
        'dateo' : rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmss00),
        'deet'  : deet,
        'npas'  : npas,

        'ni'    : ni,
        'nj'    : nj,
        'nk'    : 1,
        'etiket': etiket,
        'nbits' : 16,
        'datyp' : rmn.dtype_numpy2fst(txtRec['Data'].dtype),
        'd'     : txtRec['Data'] * mul + add
        })
    return fstRec


if __name__ == "__main__":

    desc = "Convert a specialy formated data text file to an RPNStd File"
    usage = """
    %(prog)s -i FILENAME.txt [FILENAME2.txt] -o FILENAME.fst [options]
    """
    epilog = """
    """

    parser = argparse.ArgumentParser(
        description=desc, usage=usage, epilog=epilog,
        prefix_chars='-+', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="count", default=0,
                        help="Increase output verbosity")
    parser.add_argument("-i", "--input", dest="inputFile",
                        nargs='+', required=True, type=str, default=[],
                        metavar='FILENAME.txt',
                        help="Input Text File name")
    parser.add_argument("-o", "--output", dest="outputFile",
                        required=True, type=str, default=None,
                        metavar='FILENAME.fst',
                        help="Output RPN Std File name")
    args = parser.parse_args()


    # Restric to the minimum the number of messages printed by librmn
    rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

    if args.verbose:
        sys.stdout.write("Opening the output file: {}\n".
                         format(args.outputFile))
    try:
        fileIdOut = rmn.fstopenall(args.outputFile, rmn.FST_RW)
    except:
        sys.stderr.write("ERROR: Problem opening the file: {}\n".
                         format(args.outputFile))
        sys.exit(1)

    status = 0
    for filename in args.inputFile:
        try:
            txtRec = txtFile2Rec(filename, args.verbose)
            fstRec = txt2fstRec(txtRec, args.verbose)
            if args.verbose:
                (yyyymmdd, hhmmss00) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT,
                                                    fstRec['dateo'])
                sys.stdout.write("Writing to File: {} {}.{}\n".
                                 format(fstRec['nomvar'], yyyymmdd, hhmmss00))
            rmn.fstecr(fileIdOut, fstRec['d'], fstRec)
        except:
            sys.stderr.write("ERROR: Problem encountered ---- ABORT\n")
            rmn.fstcloseall(fileIdOut)
            status = 1
            break

    rmn.fstcloseall(fileIdOut)

    sys.exit(status)


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
