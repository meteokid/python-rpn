#!/usr/bin/python
## Smoothing of FN and GZ, for cloud layer algorithm using HRDPS/RDPS field in ETA format

import rpnpy.librmn.all as rmn
import numpy as np
import sys
import os
import scipy.ndimage.filters as spyfilt

edge_value = -9.
grid_space = 10
ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
fname = os.path.join(ATM_MODEL_DFILES.strip(),'bcmk/2009042700_000')
fname_out_temp = 'tmpout-sch.fst'

# Define name and etiket for temporary file - hardcoaded because vaariable name have to be what is prescribed below
FN_nom = 'TT'   # to be consistent with known variables in order to use SPOOKI
GZ_nom = 'GZ'   # to be consistent with known variables in order to use SPOOKI
TEMP_etiket = 'CLOUD_LRSsch'

def smooth_my_data(mydata, grid_space, edge_value, varname):
    """ """
    size_grid = grid_space*2 + 1    # to account for structure of size in function uniform_filter
    mydata_smooth = np.zeros(mydata.shape,dtype=mydata.dtype,order='FORTRAN')
    spyfilt.uniform_filter(mydata, output=mydata_smooth, size=size_grid, mode='constant')
    mydata[:,:] = edge_value
    (ni, nj) = mydata.shape
    mydata[grid_space:ni-grid_space, grid_space:nj-grid_space] = mydata_smooth[grid_space:ni-grid_space, grid_space:nj-grid_space]
    del mydata_smooth
    return mydata


def smooth_my_field(fid_in, fid_out, varname, outname, etiket, grid_space, edge_value):
    """ """
    # Get list of records
    try:
        klist = rmn.fstinl(fid_in, nomvar=varname)
    except:
        raise rmn.RMNError('Problem getting list of records for '+varname)
    # read, smooth and write field
    for k in klist:
        try:
            myfield = rmn.fstluk(k)
        except:
            raise rmn.RMNError('Problem in reading var: '+varname)
        try:
            myfield['d'] = smooth_my_data(myfield['d'], grid_space, edge_value, varname)
        except:
            raise rmn.RMNError('Problem in smoothing var: '+varname)
        try:
            myfield['nomvar'] = outname
            myfield['etiket'] = etiket
            rmn.fstecr(fid_out, myfield['d'], myfield)
        except:
            raise rmn.RMNError('Problem in writing var: %s at ip1=%d\n' % (outname,myfield['ip1']))
        del myfield['d'], myfield


# Open Files
try:
    fid = rmn.fstopenall(fname,rmn.FST_RO)#RO: read only; RW for read/write 
except:
    raise rmn.RMNError('Problem opening file '+fname)   
try:
    fid_out_temp = rmn.fstopenall(fname_out_temp,rmn.FST_RW) #RW for read/write
except:
    raise rmn.RMNError('Problem opening file '+fname_out_temp)   

# Read, Smooth and Write fields
smooth_my_field(fid, fid_out_temp, 'TT', FN_nom, TEMP_etiket, grid_space, edge_value)
smooth_my_field(fid, fid_out_temp, 'GZ', GZ_nom, TEMP_etiket, grid_space, edge_value)

# Close Files
try:
    rmn.fstcloseall(fid)
    rmn.fstcloseall(fid_out_temp)
except:
    pass

