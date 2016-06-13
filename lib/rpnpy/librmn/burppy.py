#!/usr/bin/env python

"""
Python BURP interface
"""

import rpnpy.librmn.all as rmn
import os
import numpy as _np
import ctypes as _ct
#from misc import dt2unix
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from datetime import datetime
#import re
#import subprocess
#from copy import deepcopy
import warnings
#try:
#    from mpl_toolkits.basemap import Basemap
#except ImportError:
#    warnings.warn("Basemap not loaded. Plotting functions require basemap.")

### STNIDS not being read properly!!!!!!!!!!!!!!!!!!!!!!!!!!!

class BurpFile:
    """
    Class for reading a burp file.

    File attributes
        -fname:     burp file name
        -mode:      mode of file ('r'=read, 'w'=write)
        -nrep:      number of reports in file
        -nblk_max:  max number of blocks per report in the file

    Report attributes, indexed by (rep)
        -nblk       number of blocks
        -stnids     station IDs
        -year       year of report
        -month      month of report
        -day        day of report
        -hour       hour of report
        -minute     minute of report
        -codtyp     BURP code type
        -lat        latitude (degrees)
        -lon        longitude (degrees between -180 and 180)

    Block attributes, indexed by (rep,blk)
        -btyp       btyp values
        -nlev       number of levels
        -nelements  number of code elements
        -data_typ   data type
        -bfam       bfam values

    Code attributes, indexed by (rep,blk,code)
        -elements   BURP code value

    Data values, indexed by (rep,blk,code,lev)
        -rval       real number data values (only defined after read_all is called) 

    """

    file_attr = ('nrep','nblk_max','nbuf')
    rep_attr = ('nblk','year','month','day','hour','minute','codtyp','flgs','dx','dy','alt','delay','rs','runn','sup','xaux','lat','lon','stnids')
    blk_attr = ('nelements','nlev','nt','bfam','bdesc','btyp','nbit','bit0','data_type')
    ele_attr = ('elements','rval')
    burp_attr = file_attr + rep_attr + blk_attr + ele_attr


    def __init__(self,fname, mode='r'):
        """
        Initializes BurpFile, checks if file exists and reads headers.

        Parameters
        ----------
          fname      BURP file name
          mode       IO mode, 'r' (read) or 'w' (write)
        """

        self.fname = fname
        self.mode = mode

        for attr in BurpFile.burp_attr:
            setattr(self, attr, None)

        # read mode
        if mode=='r':
            if not os.path.isfile(fname):
                raise IOError("Burp file not found: %s" % fname)
            
            self.nrep,self.nbuf = self._get_fileinfo()

            # nbuf will be used to set the buffer length in the Fortran subroutines
            # more space than the longest report itself is needed here, hence the multiplication
            self.nbuf *= 10  

            if self.nrep>0:                
                self._read_data()

        return


    def _get_fileinfo(self):
        """
        Reads some basic general information from the burp file
        without having to fully open the file.

        Returns
        -------
          nrep       number of reports in the file
          rep_max    length of longest report in the file
        """

        ier  = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
        unit = rmn.fnom(self.fname, rmn.FST_RO)

        nrep    = rmn.c_mrfnbr(unit)
        rep_max = rmn.c_mrfmxl(unit)
        
        ier = rmn.fclos(unit)

        return nrep,rep_max


    def _read_data(self):
        """
        Reads all the the BURP file data and puts the file data in the
        rep_attr, blk_attr, and ele_attr arrays.
        """

        MRBCVT_DECODE = 0
        MRBCVT_ENCODE = 1

        # open BURP file
        ier  = rmn.c_mrfopc(rmn.FSTOP_MSGLVL, rmn.FSTOPS_MSG_FATAL)
        unit = rmn.fnom(self.fname, rmn.FST_RO)
        nrep = rmn.c_mrfopn(unit, rmn.FST_RO)

        # report serach information
        (stnid,idtyp,lat,lon,date,temps,nsup,nxaux) = ('*********',-1,-1,-1,-1,-1,0,0)
        sup = _np.empty((1,), dtype=_np.int32)
        handle = 0

        # buffer for report data
        buf = _np.empty((self.nbuf,), dtype=_np.int32)
        buf[0] = self.nbuf

        # report header data
        itime = _ct.c_int(0)
        iflgs = _ct.c_int(0)
        stnids = ''
        idburp = _ct.c_int(0)
        ilat  = _ct.c_int(0)
        ilon  = _ct.c_int(0)
        idx   = _ct.c_int(0)
        idy   = _ct.c_int(0)
        ialt  = _ct.c_int(0)
        idelay = _ct.c_int(0)
        idate = _ct.c_int(0)
        irs   = _ct.c_int(0)
        irunn = _ct.c_int(0)
        nblk  = _ct.c_int(0)

        self.nblk   = _np.empty((self.nrep,), dtype=_np.int)
        self.year   = _np.empty((self.nrep,), dtype=_np.int)
        self.month  = _np.empty((self.nrep,), dtype=_np.int)
        self.day    = _np.empty((self.nrep,), dtype=_np.int)
        self.hour   = _np.empty((self.nrep,), dtype=_np.int)
        self.minute = _np.empty((self.nrep,), dtype=_np.int)
        self.codtyp = _np.empty((self.nrep,), dtype=_np.int)
        self.flgs   = _np.empty((self.nrep,), dtype=_np.int)
        self.dx     = _np.empty((self.nrep,), dtype=_np.int)
        self.dy     = _np.empty((self.nrep,), dtype=_np.int)
        self.alt    = _np.empty((self.nrep,), dtype=_np.int)
        self.delay  = _np.empty((self.nrep,), dtype=_np.int)
        self.rs     = _np.empty((self.nrep,), dtype=_np.int)
        self.runn   = _np.empty((self.nrep,), dtype=_np.int)
        self.sup    = _np.empty((self.nrep,), dtype=_np.int)
        self.xaux   = _np.empty((self.nrep,), dtype=_np.int)
        self.lon    = _np.empty((self.nrep,), dtype=_np.float)
        self.lat    = _np.empty((self.nrep,), dtype=_np.float)
        self.stnids = _np.empty((self.nrep,), dtype='|S9')

        # block header data
        nele  = _ct.c_int(0)
        nval  = _ct.c_int(0)
        nt    = _ct.c_int(0)
        bfam  = _ct.c_int(0)
        bdesc = _ct.c_int(0)
        btyp  = _ct.c_int(0)
        nbit  = _ct.c_int(0)
        bit0  = _ct.c_int(0)
        datyp = _ct.c_int(0)

        for attr in BurpFile.blk_attr:
            setattr(self, attr, _np.empty((self.nrep,), dtype=object))

        # block data
        self.elements = _np.empty((self.nrep,), dtype=object)
        self.rval     = _np.empty((self.nrep,), dtype=object)

        xaux = _np.empty((1,), dtype=_np.int32)

        # loop over reports
        for irep in xrange(nrep):
            
            # get next report and load data into buffer
            handle = rmn.c_mrfloc(unit,handle,stnid,idtyp,lat,lon,date,temps,sup,nsup)
            ier = rmn.c_mrfget(handle,buf)

            # get report header
            ier = rmn.c_mrbhdr(buf,itime,iflgs,stnids,idburp,ilat,ilon,idx,idy,ialt,
                               idelay,idate,irs,irunn,nblk,sup,nsup,xaux,nxaux)
            
            self.flgs[irep]   = iflgs.value
            self.codtyp[irep] = idburp.value
            self.dx[irep]     = idx.value
            self.dy[irep]     = idy.value
            self.alt[irep]    = ialt.value
            self.delay[irep]  = idelay.value
            self.rs[irep]     = irs.value
            self.runn[irep]   = irunn.value         
            self.nblk[irep]   = nblk.value            
            self.sup[irep]    = sup[0]
            self.xaux[irep]   = xaux[0]

            self.year[irep]   = idate.value/10000
            self.month[irep]  = (idate.value%10000)/100
            self.day[irep]    = idate.value%100
            self.hour[irep]   = itime.value/100
            self.minute[irep] = itime.value%100
  
            self.lon[irep] = ilon.value/100.
            self.lat[irep] = (ilat.value-9000.)/100.
            
            self.stnids[irep] = stnids
            
            for attr in BurpFile.blk_attr:
                getattr(self, attr)[irep] = _np.empty((nblk.value,), dtype=int)
                
            self.elements[irep] = _np.empty((nblk.value,), dtype=object)
            self.rval[irep]     = _np.empty((nblk.value,), dtype=object)

            # loop over blocks
            for iblk in xrange(nblk.value):

                # get block header
                ier = rmn.c_mrbprm(buf,iblk+1,nele,nval,nt,bfam,bdesc,btyp,nbit,bit0,datyp)
                
                self.nelements[irep][iblk] = nele.value
                self.nlev[irep][iblk]      = nval.value
                self.nt[irep][iblk]        = nt.value
                self.bfam[irep][iblk]      = bfam.value
                self.bdesc[irep][iblk]     = bdesc.value
                self.btyp[irep][iblk]      = btyp.value
                self.nbit[irep][iblk]      = nbit.value
                self.bit0[irep][iblk]      = bit0.value
                self.data_type[irep][iblk] = datyp.value
                
                lstele = _np.empty((nele.value,), dtype=_np.int32)
                nmax = nele.value*nval.value*nt.value
                tblval = _np.empty((nmax,), dtype=_np.int32)
                rval   = _np.empty((nmax,), dtype=_np.float32)

                # get block elements and values
                ier = rmn.c_mrbxtr(buf,iblk+1,lstele,tblval)

                # convert CMC codes to BUFR codes
                codes = _np.empty((nele.value,), dtype=_np.int32)
                ier = rmn.c_mrbdcl(lstele,codes,nele)
                
                self.elements[irep][iblk] = codes
                
                # convert integer table values if needed
                if datyp.value in (2,4):
                    ier = rmn.c_mrbcvt(lstele,tblval,rval,nele,nval,nt,MRBCVT_DECODE)
                elif datyp.value == 6:
                    rval = tblval #??transfer???
                    ## rval(j)=transfer(tblval(j),z4val)
                else:
                    pass #raise
        
                self.rval[irep][iblk] = _np.resize(rval, (nval.value,nele.value)).T


        # close BURP file
        ier = rmn.c_mrfcls(unit)
        ier = rmn.fclos(unit)

        # change lon to be between -180 and 180 degrees
        self.lon = self.lon % 360.
        self.lon[self.lon>180] = self.lon[self.lon>180] - 360.

        return 


    def get_rval(self,code=None,block=None,element=None,fmt=float,flatten=True,**kwargs):
        """
        Returns an array of values for a single element from the burp file.

        The returned array will have dimensions of (nrep,nlev_max) where
        nlev_max is the largest nlev value found in the query. 

        Values that are either not found or that are in reports smaller than nlev_max
        will be set to np.nan, unless fmt=int, in which case values not found set to -999.

        Parameters
        ----------
          code      BURP code
          btyp      select code from block with specified btyp
          block     block number to select codes from (starting from 1)
          element   element number to select (starting from 1)
          bfam      selects codes specified BFAM value 
          fmt       data type to be outputted
          flatten   if true will remove degenerate axis from the output array

        Returns
        -------
          outdata   BURP data from query
        """

        assert self.mode=='r', "BurpFile must be in read mode to use this function."
        assert code is not None or (block is not None and element is not None), "Either code or block and element have to be supplied to find BURP values."
        assert code is None or isinstance(code,int), "If specified, code must be an integer."
        assert block is None or isinstance(block,int), "If specified, block must be an integer."
        assert element is None or isinstance(element,int), "If specified, element must be an integer."
        assert fmt is float or int, "If specified, fmt must be float or int."

        for k in kwargs:
            if k in BurpFile.blk_attr:
                assert isinstance(kwargs[k],int), "If specified, %s must be an integer." % k
                if (block is not None and element is not None):
                    warnings.warn("Option \'%s\' is not used when both the block and element are specified." % k)
            else:
                raise Exception("Unknown parameter \'%s\'" % k)

        fill_val = _np.nan if fmt==float else -999
        mult_codes = False
        nlev_max = 0


        if block is not None and element is not None:
            # block and element positions are already known

            iblk = block-1
            iele = element-1
            outdata = []

            for irep in xrange(self.nrep):

                if self.nblk[irep]>iblk:
                    if self.nelements[irep][block-1]>iele:
                        outdata.append( self.rval[irep][iblk][iele] )
                        if self.nlev[irep][iblk]>nlev_max:
                            nlev_max = self.nlev[irep][iblk]
                    else:
                        outdata.append(_np.array([]))
                else:
                    outdata.append(_np.array([]))
            
        else:
            # get block and element positions from search criteria

            outdata = []

            for irep in xrange(self.nrep):

                # find all block,element pairs for the BURP code
                iele_code = [ _np.where(elem==code)[0] for elem in f.elements[irep] ]

                iblk = []
                iele = []

                # add all block,element pairs to iblk,iele that match search criteria
                for blk,i in enumerate(iele_code):
                    if len(i)==0:
                        continue
                    elif block is not None:
                        if blk!=block-1:
                            continue
                    elif element is not None:
                        if i!=element-1:
                            continue
                    else:
                        skip = False
                        for k in kwargs:
                            if getattr(self,k)[irep][blk]!=kwargs[k]:
                                skip = True
                                break
                        if skip:
                            continue

                    iblk.append(blk)
                    iele.append(i[0])


                if len(iblk)>1 and not mult_codes:
                    mult_codes = True
                    warnings.warn("More than one code in report. Returning first found value.")

                if len(iblk)>0:
                    outdata.append( self.rval[irep][iblk[0]][iele[0]] )
                    if self.nlev[irep][iblk[0]]>nlev_max:
                            nlev_max = self.nlev[irep][iblk[0]]
                else:
                    outdata.append(_np.array([]))
 

        # create array of dimension (nrep,nlev_max) with _np.nan padded values
        outdata = _np.array([ _np.hstack([i,_np.repeat(fill_val,nlev_max-len(i))]) for i in outdata ])

        # change data format if specified
        # automatically convert flags to integers
        flag_code = False if code is None else code/100000==2
        if fmt==int or flag_code:
            outdata = _np.array([[ int(round(i)) for i in line ] for line in outdata ])
        
        # if only one level, return 1D output array
        if flatten and nlev_max==1:
            outdata = outdata[:,0]

        return outdata

        


if __name__ == "__main__":
    
    f = BurpFile('/users/tor/arpx/msi/data/observations/test/derialt/2014070200_ompsnp')
