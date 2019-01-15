#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Camille Garnaud <Camille.Garnaud@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
"""
Perform aggregation on a field.
"""
import os
import sys
import datetime
import optparse
import pylab
import numpy as np
import rpnpy.librmn.all as rmn


def readRec(fname, nomvar):
    """
    Read specified record data from file and get its grid definition

    Args:
       fname  (str): Filename to read from
       nomvar (str): Record varname to read
    Returns:
       rec : Record meta, data and grid definition
    """
    print("+ Read %s from: %s" % (nomvar, fname))
    
    # Open File
    try:
        funit = rmn.fstopenall(fname, rmn.FST_RO)
    except:
        raise rmn.FSTDError("Problem Opening file: %s" % fname)
    
    # Read Record data and meta
    try:
        rec = rmn.fstlir(funit, nomvar=nomvar)
    except:
        raise rmn.FSTDError("Problem Reading %s record" % nomvar)
    
    # Get associated grid
    try:
        rec['iunit'] = funit
        rec_gridid   = rmn.ezqkdef(rec)  # use ezscint to get grid id
        rec['grid']  = rmn.decodeGrid(rec_gridid)
        del(rec['iunit']) # iunit will be irrelevant after file is closed below
    except:
        sys.stderr.write('Error: Problem getting grid info for '+nomvar+' in file: '+fname+'\n')
        sys.exit(1)
        
    # Close File
    rmn.fstcloseall(funit)
    
    return rec


def aggregateData(datain, aggFac):
    """
    Perform aggregation by a factor of aggFac
   
    Args:
       datain  (numpy.nd_array): array with input data
    Returns:
       (numpy.nd_array) Array with results from the aggregation

    """
    print("+ Aggregate data by a factor of %d" %  aggFac)

    # Create new data array for aggregated data
    # WARNING: rpnpy get its fields from librmn fst functions
    #          these fields are most of the times Fortran real*4 array
    #          (dtype=np.float32)
    #          while the default numpy array is C double
    #          (dtype=np.float64)
    # It is best to get the numpy dtype from the read/input record dtype
    # and to force order='F'
    ni_hi = datain.shape[0]
    nj_hi = datain.shape[1]
    ni_lo = max(1,ni_hi/aggFac)
    nj_lo = max(1,nj_hi/aggFac)
    shape_lo = (ni_lo, nj_lo)
    dataout = np.zeros(shape_lo, dtype=datain.dtype, order='F')

    # Perform aggregation
    for j_lo in range(nj_lo):
        for i_lo in range(ni_lo):
            i_hi_beg = i_lo*int(aggFac)
            j_hi_beg = j_lo*int(aggFac)
            i_hi_end = min((i_lo + 1)*int(aggFac) - 1, ni_hi-1)
            j_hi_end = min((j_lo + 1)*int(aggFac) - 1, nj_hi-1)
            dataout[i_lo,j_lo] =  pylab.mean(
                datain[i_hi_beg:i_hi_end+1, j_hi_beg:j_hi_end+1])

    return dataout
    

def aggregateGrid(rec, aggFac):
    """
    Define an aggregated grid and update record meta to match

    Warning: rec data will not be updated and can thus
             be inconsistent with the updated metadata defining the grid
   
    Args:
       rec    (dict): Record meta to aggregate
       aggFac (int) : aggregation factor
    Returns:
       rec : dict with aggregated record meta + new grid
    """
    print("+ Aggregate grid by a factor of %d" %  aggFac)

    if rec['grid']['grtyp'] != 'Z' or not rec['grid']['grref'] in ('L','E'):
        sys.stderr.write('Error: Grid not supported grtyp/grref: %s/%s\n' % (rec['grid']['grtyp'], rec['grid']['grref']))
        sys.exit(1)
        
    # Aggregate Axes
    ax2 = aggregateData(rec['grid']['ax'], aggFac)
    ay2 = aggregateData(rec['grid']['ay'], aggFac)

    # Update Grid info for new size and axes
    rec['grid']['ni'] = ax2.shape[0]
    rec['grid']['nj'] = ay2.shape[1]
    rec['grid']['shape'] = (ax2.shape[0], ay2.shape[1])
    rec['grid']['ax'] = ax2
    rec['grid']['ay'] = ay2
    rec['grid']['lat0'] = ay2[0,0]
    rec['grid']['lon0'] = ax2[0,0]
    rec['grid']['id'] = -1

    # Update grid tag/id for new axes
    (tag1, tag2) = rmn.getIgTags(rec['grid'])
    rec['grid']['tag1'] = tag1
    rec['grid']['tag2'] = tag2
    rec['grid']['tag3'] = 0
    rec['grid']['ig1'] = rec['grid']['tag1']
    rec['grid']['ig2'] = rec['grid']['tag2']
    rec['grid']['ig3'] = rec['grid']['tag3']

    # Update Rec info for new grid size and axes
    for k in ('ni','nj','shape'):
        rec[k] = rec['grid'][k]
    rec['ig1'] = rec['grid']['tag1']
    rec['ig2'] = rec['grid']['tag2']
    rec['ig3'] = rec['grid']['tag3']
    
    return rec


def aggregateRec(rec, aggFac):
    """
    Perform aggregation on a record and update its grid

    Args:
       rec    (dict): Record meta + data to aggregate
       aggFac (int) : aggregation factor
    Returns:
       rec2 : dict with aggregated record meta + data 
    """
    print("+ Aggregate %s by a factor of %d" % (rec['nomvar'], aggFac))
    
    # Create new record and update its meta data from input rec
    rec2 = rmn.FST_RDE_META_DEFAULT.copy()
    rec2.update(rec)
    
    # Create aggregated grid and update rec2 grid size meta
    rec2 = aggregateGrid(rec2 , aggFac)

    # Create new data array for aggregated data
    rec2['d'] = aggregateData(rec['d'], aggFac)
            
    return rec2


def writeRecGrid(rec, funit, fname=''):
    """
    Write the record grid info to previously opened file
    
    Args:
       rec   (dict): Record meta + grid info to write
       funit  (int): Unit number the the opened file to write to
       fname  (str): (optional) Filename to to write to
    Returns:
       None
    """
    print("+ Write grid for %s to: %s" % (rec['nomvar'], fname))
    rec2 = rmn.FST_RDE_META_DEFAULT.copy()
    rec2.update(rec)
    rec2['ip1'] = rec['grid']['tag1']
    rec2['ip2'] = rec['grid']['tag2']
    rec2['ip3'] = rec['grid']['tag3']
    rec2['grtyp'] = rec['grid']['grref']
    rec2['grref'] = rec['grid']['grref']
    rec2['ig1'] = rec['grid']['ig1ref']
    rec2['ig2'] = rec['grid']['ig2ref']
    rec2['ig3'] = rec['grid']['ig3ref']
    rec2['ig4'] = rec['grid']['ig4ref']

    for k in ('ig1ref','ig2ref','ig3ref','ig4ref'):
        rec2[k] = rec['grid'][k]

    rec2['d']     = rec['grid']['ax']
    rec2['ni']    = rec['grid']['ax'].shape[0]
    rec2['nj']    = rec['grid']['ax'].shape[1]
    rec2['shape'] = rec['grid']['ax'].shape
    rec2['nomvar'] = '>>'
    try:
        rmn.fstecr(funit,rec2['d'],rec2)
    except:
        raise rmn.FSTDError("Problem writing %s record" % rec2['nomvar'])
    
    rec2['d']     = rec['grid']['ay']
    rec2['ni']    = rec['grid']['ay'].shape[0]
    rec2['nj']    = rec['grid']['ay'].shape[1]
    rec2['shape'] = rec['grid']['ay'].shape
    rec2['nomvar'] = '^^'
    try:
        rmn.fstecr(funit,rec2['d'],rec2)
    except:
        raise rmn.FSTDError("Problem writing %s record" % rec2['nomvar'])

    return


def writeRec(fname, rec):
    """
    Write the record data along with grid info to file
    
    Args:
       fname  (str): Filename to to write to
       rec   (dict): Record meta + data + grid info to write
    Returns:
       None
    """
    print("+ Write %s to: %s" % (rec['nomvar'], fname))
            
    # Open File
    try:
        funit = rmn.fstopenall(fname, rmn.FST_RW)
    except:
        raise rmn.FSTDError("Problem Opening file: %s" % fname)
        
    # Write rec meta + data
    try:
        rmn.fstecr(funit,rec['d'],rec)
    except:
        raise rmn.FSTDError("Problem writing %s record" % rec['nomvar'])

    # Write grid (if need be)
    writeRecGrid(rec, funit, fname)
    
    # Close File
    rmn.fstcloseall(funit)
    
    return


if __name__ == "__main__":
    desc="Perform agreggation on a field"
    usage = """
    %prog -i INFILE -n VARNAME -o OUTFILE --fac AGG_FAC [--overwrite]
    """    
    # Default Options values
    forecast_name  = datetime.date.today().strftime('%Y%m%d') + '00_048'
    CMCGRIDF = os.getenv('CMCGRIDF').strip()
    inputFileName  = os.path.join(CMCGRIDF, 'prog', 'regpres', forecast_name)
    inputVarName   = 'PR'
    outputFileName = './output.fst'
    aggFac  = 3
    
    # Get Options from user
    parser = optparse.OptionParser(usage=usage,description=desc)
    parser.add_option("-i","--infile",dest="infile",default=inputFileName,
                      help="Name of the input RPN STD file containing record VARNAME")
    parser.add_option("-n","--varname",dest="varname",default=inputVarName,
                      help="Varname of the record to aggregate")
    parser.add_option("-o","--outfile",dest="outfile",default=outputFileName,
                      help="Name of the output RPN STD file")
    parser.add_option("-f","--fac",dest="aggFac",default="3",
                      help="Aggregation factor (int)")
    parser.add_option("-w","--overwrite",dest="overwrite",action="store_true",
                      help="Overwrite output file")
    (options,args) = parser.parse_args()
    
    # Read, Aggregate, Write
    rec  = readRec(options.infile, options.varname)
    rec2 = aggregateRec(rec, int(options.aggFac))
    rec2['etiket'] = 'AGGREGATED'
    if options.overwrite and os.path.exists(options.outfile):
        os.remove(options.outfile)
    writeRec(options.outfile, rec2)
