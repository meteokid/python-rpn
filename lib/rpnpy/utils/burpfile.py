#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Michael Sitwell <michael.sitwell@canada.ca>
# Copyright: LGPL 2.1

"""
Python High Level Interface for [[BURP]] files.

See Also:
    rpnpy.librmn.burp
    rpnpy.librmn.burp_const
    rpnpy.burpc.base
    rpnpy.burpc.brpobj
    rpnpy.librmn.proto_burp
    rpnpy.librmn.base
    rpnpy.librmn.fstd98
    rpnpy.librmn.const
"""

from rpnpy.librmn import burp as _brp
from rpnpy.librmn import base as _rb
from rpnpy.librmn import const as _rc
from rpnpy.librmn import burp_const as _rbc
import numpy as _np
import ctypes as _ct
from calendar import timegm as _timegm
from copy import deepcopy as _deepcopy
from datetime import datetime as _datetime
import warnings as _warnings
import os as _os
import re as _re

## _warnings.simplefilter('always', ImportWarning)
## _warnings.simplefilter('always', UserWarning)
_warnings.simplefilter('ignore', UserWarning)

try:
    import matplotlib.pyplot as _plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
except ImportError:
    _warnings.warn("matplotlib not loaded. Plotting functions require matplotlib.", ImportWarning)

try:
    from mpl_toolkits.basemap import Basemap as _Basemap
except ImportError:
    _warnings.warn("Basemap not loaded. Plotting functions require basemap.", ImportWarning)


class BurpFile:
    """
    Class for reading entirely a burp file into one object.

    myburpfile = BurpFile(fname, mode)

    Args:
        fname : burp file name
        mode  : I/O mode of file ('r'=read, 'w'=write, 'rw'=read or write)
    Raises:
        IOError if file not found, readable or writebale
        BurpError
    See Also:
        rpnpy.librmn.burp

    Notes:
      <pre>
      Report attributes, indexed by (report), length (nrep)
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

      Block attributes, indexed by (report, block), length (nrep, nblk)
        -btyp       btyp values
        -nlev       number of levels
        -nelements  number of code elements
        -datyp      data type
        -bfam       bfam values

      Code attributes, indexed by (report, block, element), length (nrep, nblk, nelements)
        -elements   BURP code value

      Data values, indexed by (report, block, element, level, group), length (nrep, nblk, nelements, nlev, nt)
        -rval       real number data values
      </pre>

    """
    file_attr = ('nrep', )
    rep_attr =  ('nblk', 'year', 'month', 'day', 'hour', 'minute', 'codtyp',
                 'flgs', 'dx', 'dy', 'alt', 'delay', 'rs', 'runn', 'sup',
                 'xaux', 'lat', 'lon', 'stnids')
    blk_attr =  ('nelements', 'nlev', 'nt', 'bfam', 'bdesc', 'btyp', 'nbit',
                 'bit0', 'datyp')
    ele_attr =  ('elements', 'rval')
    burp_attr = file_attr + rep_attr + blk_attr + ele_attr


    def __init__(self, fname, mode='r'):
        """
        Initializes BurpFile, checks if file exists and reads BURP data.

        Args:
            fname : burp file name
            mode  : I/O mode of file ('r'=read, 'w'=write, 'rw'=read or write)
        Raises:
            IOError if file not found, readable or writebale
            BurpError
        """
        self.fname = fname
        self.mode = mode

        for attr in BurpFile.burp_attr:
            setattr(self, attr, None)

        # read mode
        if 'r' in mode:
            if not _os.path.isfile(fname):
                raise IOError("Burp file not found: %s" % fname)

            self.nrep, nbuf = self._get_fileinfo()

            nbuf *= 2  # increase the buffer length as a precaution

            if self.nrep > 0:
                self._read_data(nbuf)

        return


    def __str__(self):
        """
        Return BURP file attributes representation.

        Returns:
            str, class instance attributes representation
        """
        return "<BurpFile instance>\n" + \
            "file name: \n  %s \n" % self.fname + \
            "IO mode: \n  \'%s\' \n" % self.mode + \
            "number of reports: \n  %i" % self.nrep


    def _get_fileinfo(self):
        """
        Reads some basic general information from the burp file
        without having to fully open the file.

        Returns:
            (nrep, rep_max), tuple where:
               nrep    : number of reports in the file
               rep_max : length of longest report in the file
        """
        assert 'r' in self.mode, "BurpFile must be in read mode to use this function."

        ier  = _brp.mrfopt(_rbc.BURPOP_MSGLVL, _rbc.BURPOP_MSG_FATAL)
        unit = _rb.fnom(self.fname, _rc.FST_RO)

        nrep    = _brp.mrfnbr(unit)
        rep_max = _brp.mrfmxl(unit)

        ier = _rb.fclos(unit)

        return nrep, rep_max


    def _calc_nbuf(self):
        """
        Calculates the minimum buffer length required for the longest report.

        Returns:
            int, minimum buffer length required for the longest report
        """
        nlev_max = _np.max([ nlev.max() if len(nlev)>0 else 0 for nlev in self.nlev ])
        nele_max = _np.max([ nele.max() if len(nele)>0 else 0 for nele in self.nelements ])
        nt_max   = _np.max([ nt.max() if len(nt)>0 else 0 for nt in self.nt ])
        nbit_max = _np.max([ nbit.max() if len(nbit)>0 else 0 for nbit in self.nbit ])
        nblk_max = self.nblk.max()

        M64 = lambda x: int(_np.floor((x+63)/64)*64)
        nbuf_blk = 128 + M64((nele_max-3)*16) + M64(nele_max*nlev_max*nt_max*nbit_max)

        return 2 * (330 + nblk_max * nbuf_blk)


    def _read_data(self, nbuf):
        """
        Reads all the the BURP file data and puts the file data in the
        rep_attr, blk_attr, and ele_attr arrays.

        Args:
            nbuf : buffer length for reading of BURP file
        Returns:
            None
        Raises:
            BurpError
        """
        assert 'r' in self.mode, "BurpFile must be in read mode to use this function."

        warn = True

        # open BURP file
        _brp.mrfopt(_rbc.BURPOP_MSGLVL, _rbc.BURPOP_MSG_FATAL)
        unit = _brp.burp_open(self.fname)
        nrep = _brp.mrfnbr(unit)

        self.nblk   = _np.empty((self.nrep, ), dtype=_np.int)
        self.year   = _np.empty((self.nrep, ), dtype=_np.int)
        self.month  = _np.empty((self.nrep, ), dtype=_np.int)
        self.day    = _np.empty((self.nrep, ), dtype=_np.int)
        self.hour   = _np.empty((self.nrep, ), dtype=_np.int)
        self.minute = _np.empty((self.nrep, ), dtype=_np.int)
        self.codtyp = _np.empty((self.nrep, ), dtype=_np.int)
        self.flgs   = _np.empty((self.nrep, ), dtype=_np.int)
        self.dx     = _np.empty((self.nrep, ), dtype=_np.int)
        self.dy     = _np.empty((self.nrep, ), dtype=_np.int)
        self.alt    = _np.empty((self.nrep, ), dtype=_np.int)
        self.delay  = _np.empty((self.nrep, ), dtype=_np.int)
        self.rs     = _np.empty((self.nrep, ), dtype=_np.int)
        self.runn   = _np.empty((self.nrep, ), dtype=_np.int)
        self.sup    = _np.empty((self.nrep, ), dtype=_np.int)
        self.xaux   = _np.empty((self.nrep, ), dtype=_np.int)
        self.lon    = _np.empty((self.nrep, ), dtype=_np.float)
        self.lat    = _np.empty((self.nrep, ), dtype=_np.float)
        self.stnids = _np.empty((self.nrep, ), dtype='|S9')

        for attr in BurpFile.blk_attr + BurpFile.ele_attr:
            setattr(self, attr, _np.empty((self.nrep, ), dtype=object))

        # loop over reports
        handle = 0
        buf    = nbuf
        for irep in range(nrep):

            # get next report and load data into buffer
            handle = _brp.mrfloc(unit, handle)
            buf    = _brp.mrfget(handle, buf)

            # get report header
            rhp = _brp.mrbhdr(buf)

            self.flgs[irep]   = rhp['flgs']
            self.codtyp[irep] = rhp['idtyp']
            self.dx[irep]     = rhp['idx']
            self.dy[irep]     = rhp['idy']
            self.alt[irep]    = rhp['ielev']
            self.delay[irep]  = rhp['drnd']
            self.rs[irep]     = rhp['oars']
            self.runn[irep]   = rhp['runn']
            self.nblk[irep]   = rhp['nblk']
            self.sup[irep]    = 0
            self.xaux[irep]   = 0

            self.year[irep]   = rhp['date']/10000
            self.month[irep]  = (rhp['date']%10000)/100
            self.day[irep]    = rhp['date']%100
            self.hour[irep]   = rhp['time']/100
            self.minute[irep] = rhp['time']%100

            self.lon[irep] = rhp['ilon']/100.
            self.lat[irep] = (rhp['ilat']-9000.)/100.

            self.stnids[irep] = rhp['stnid']

            for attr in BurpFile.blk_attr:
                getattr(self, attr)[irep] = _np.empty((rhp['nblk'], ), dtype=int)

            for attr in BurpFile.ele_attr:
                getattr(self, attr)[irep] = _np.empty((rhp['nblk'], ), dtype=object)

            # loop over blocks
            for iblk in range(rhp['nblk']):

                # get block header
                bhp = _brp.mrbprm(buf, iblk+1)

                self.nelements[irep][iblk] = bhp['nele']
                self.nlev[irep][iblk]      = bhp['nval']
                self.nt[irep][iblk]        = bhp['nt']
                self.bfam[irep][iblk]      = bhp['bfam']
                self.bdesc[irep][iblk]     = bhp['bdesc']
                self.btyp[irep][iblk]      = bhp['btyp']
                self.nbit[irep][iblk]      = bhp['nbit']
                self.bit0[irep][iblk]      = bhp['bit0']
                self.datyp[irep][iblk]     = bhp['datyp']

                # get block elements and values and
                # convert integer table values to real value
                if bhp['datyp'] < 5:
                    bdata = _brp.mrbxtr(buf, iblk+1)
                    rval  = _brp.mrbcvt_decode(bdata)
                elif bhp['datyp'] < 7:
                    bdata = _brp.mrbxtr(buf, iblk+1, dtype=_np.float32)
                    rval = bdata['tblval']
                else:
                    bdata = _brp.mrbxtr(buf, iblk+1)
                    rval  = bdata['tblval'].astype(_np.float32)
                    if warn:
                        _warnings.warn("Unrecognized data type value of %i. Unconverted table values will be returned." % bhp['datyp'])
                        warn = False

                # convert CMC codes to BUFR codes
                self.elements[irep][iblk] = _brp.mrbdcl(bdata['cmcids'])

                #TODO: since arrays are now allocated Fortran style,
                #      should we do a transpose?
                self.rval[irep][iblk] = rval

                # check that the element arrays have the correct dimensions
                if _np.any(self.elements[irep][iblk].shape != (self.nelements[irep][iblk])):
                    raise _brp.BurpError("elements array does not have the correct dimensions.")
                if _np.any(self.rval[irep][iblk].shape != (self.nelements[irep][iblk], self.nlev[irep][iblk], self.nt[irep][iblk])):
                    raise _brp.BurpError("rval array does not have the correct dimensions.")


        # close BURP file
        _brp.burp_close(unit)

        # change longitude to be between -180 and 180 degrees
        self.lon = self.lon % 360.
        self.lon[self.lon>180] = self.lon[self.lon>180] - 360.

        return


    def write_burpfile(self):
        """
        Writes BurpFile instance to a BURP file.

        Returns:
            None
        Raises:
            BurpError
        """
        assert 'w' in self.mode, "BurpFile must be in write mode to use this function."

        print("Writing BURP file to \'{}\'".format(self.fname))

        handle = 0
        nsup = 0
        nxaux = 0
        warn = True

        # convert lat, lon back into integers
        ilon = _np.round(100*self.lon).astype(int)
        ilat = _np.round(100*self.lat+9000).astype(int)
        ilon[ilon<0] = ilon[ilon<0]+36000

        idate = self.year*10000 + self.month*100 + self.day
        itime = self.hour*100 + self.minute

        # buffer for report data
        nbuf = self._calc_nbuf()
        buf = _np.empty((nbuf, ), dtype=_np.int32)
        buf[0] = nbuf

        # open BURP file
        _brp.mrfopt(_rbc.BURPOP_MSGLVL, _rbc.BURPOP_MSG_FATAL)
        unit = _brp.burp_open(self.fname, _rbc.BURP_MODE_CREATE)

        # loop over reports
        for irep in range(self.nrep):

            # write report header
            _brp.mrbini(unit, buf, itime[irep], self.flgs[irep], self.stnids[irep], self.codtyp[irep],
                        ilat[irep], ilon[irep], self.dx[irep], self.dy[irep], self.alt[irep],
                        self.delay[irep], idate[irep], self.rs[irep], self.runn[irep], self.sup[irep],
                        nsup, self.xaux[irep], nxaux)

            for iblk in range(self.nblk[irep]):

                nele = self.nelements[irep][iblk]
                nlev = self.nlev[irep][iblk]
                nt = self.nt[irep][iblk]

                # convert BUFR codes to CMC codes
                cmcids = _brp.mrbcol(self.elements[irep][iblk])

                # convert real values to integer table values
                if self.datyp[irep][iblk] < 5:
                    tblval = _brp.mrbcvt_encode(cmcids, self.rval[irep][iblk])
                    tblval = _np.ravel(tblval, order='F')
                else:
                    rval = _np.ravel(self.rval[irep][iblk], order='F')
                    tblval = _np.round(rval).astype(_np.int32)
                    if self.datyp[irep][iblk] > 6 and warn:
                        _warnings.warn("Unrecognized data type value of %i. Unconverted table values will be written." %  self.datyp[irep][iblk])
                        warn = False

                # add block to report
                _brp.mrbadd(buf, nele, nlev, nt, self.bfam[irep][iblk],
                            self.bdesc[irep][iblk],
                            self.btyp[irep][iblk], self.nbit[irep][iblk],
                            self.datyp[irep][iblk], cmcids, tblval)


            # write report
            _brp.mrfput(unit, handle, buf)

        # close BURP file
        _brp.mrfcls(unit)
        _rb.fclos(unit)

        return


    def get_rval(self, code=None, block=None, element=None, group=1, fmt=float, flatten=True, **kwargs):
        """
        Returns an array of values for a single element from the burp file.

        The returned array will have dimensions of (nrep, nlev_max) where
        nlev_max is the largest nlev value found in the query.

        Values that are either not found or that are in reports smaller than nlev_max
        will be set to np.nan, unless fmt=int, in which case values not found set to -999.

        Any block attribute can be provided as an argument to filter the query.

        Args:
            code    : BURP code
            block   : block number to select codes from (starting from 1)
            element : element number to select (starting from 1)
            group   : group number to select (starting from 1)
            btyp    : select code from block with specified btyp
            bfam    : selects codes specified BFAM value
            fmt     : data type to be outputted
            flatten : if true will remove degenerate axis from the output array
        Returns:
            list, outdata - BURP data from query
        Raises:
            BurpError
        """
        assert code is not None or (block is not None and element is not None), "Either code or block and element have to be supplied to find BURP values."
        assert code is None or isinstance(code, int), "If specified, code must be an integer."
        assert block is None or isinstance(block, int), "If specified, block must be an integer."
        assert element is None or isinstance(element, int), "If specified, element must be an integer."
        assert fmt is float or int, "If specified, fmt must be float or int."

        for k in kwargs:
            if k in BurpFile.blk_attr:
                assert isinstance(kwargs[k], int), "If specified, %s must be an integer." % k
                if (block is not None and element is not None):
                    _warnings.warn("Option \'%s\' is not used when both the block and element are specified." % k)
            else:
                raise _brp.BurpError("Unknown parameter \'%s\'" % k)

        fill_val = _np.nan if fmt==float else -999
        nlev_max = 0
        it = group-1
        warn = True

        if block is not None and element is not None:
            # block and element positions are already known

            iblk = block-1
            iele = element-1
            outdata = []

            for irep in range(self.nrep):

                if self.nblk[irep]>iblk:
                    if self.nelements[irep][block-1]>iele and self.nt[irep][iblk]>it:
                        outdata.append( self.rval[irep][iblk][iele,:,it] )
                        if self.nlev[irep][iblk]>nlev_max:
                            nlev_max = self.nlev[irep][iblk]
                    else:
                        outdata.append(_np.array([]))
                else:
                    outdata.append(_np.array([]))

        else:
            # get block and element positions from search criteria

            outdata = []

            for irep in range(self.nrep):

                # find all block, element pairs for the BURP code
                iele_code = [ _np.where(elem==code)[0] for elem in self.elements[irep] ]

                iblk = []
                iele = []

                # add all block, element pairs to iblk, iele that match search criteria
                for blk, i in enumerate(iele_code):
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
                            if getattr(self, k)[irep][blk]!=kwargs[k]:
                                skip = True
                                break
                        if skip:
                            continue

                    iblk.append(blk)
                    iele.append(i[0])


                if warn and len(iblk)>1:
                    _warnings.warn("More than one code in report. Returning first found value.")
                    warn = False

                if len(iblk)>0:
                    if self.nt[irep][iblk[0]]>it:
                        outdata.append( self.rval[irep][iblk[0]][iele[0],:,it] )
                        if self.nlev[irep][iblk[0]]>nlev_max:
                            nlev_max = self.nlev[irep][iblk[0]]
                    else:
                        outdata.append(_np.array([]))
                else:
                    outdata.append(_np.array([]))


        # create array of dimension (nrep, nlev_max) with values padded with fill_val
        outdata = _np.array([ _np.hstack([i, _np.repeat(fill_val, nlev_max-len(i))]) for i in outdata ])

        # change data format if specified
        if fmt==int:
            outdata = _np.array([[ int(round(i)) for i in line ] for line in outdata ])

        # if only one level, return 1D output array
        if flatten and nlev_max==1:
            outdata = outdata[:, 0]

        return outdata


    def get_datetimes(self, fmt='datetime'):
        """
        Returns the datetimes of the reports.

        Args:
            fmt : datetime return type:
                  'datetime' - datetime object
                  'string'   - string in YYYY-MM-DD HH:MM format
                  'int'      - integer list in [YYYYMMDD, HHMM] format
                  'unix'     - Unix timestamp
        Returns:
            list, datetimes of the reports
        """
        assert fmt in ('datetime', 'string', 'int', 'unix'), "Invalid format \'%s\'" % fmt

        dts = []
        for i in range(self.nrep):

            d = _datetime(self.year[i], self.month[i], self.day[i], self.hour[i], self.minute[i])

            if fmt in ('string', 'int'):
                d = d.isoformat().replace('T', ' ')
                d = d[:[x.start() for x in _re.finditer(':', d)][-1]]
                if fmt=='int':
                    d = [ int(i) for i in d.replace('-', '').replace(':', '').split() ]
            elif fmt=='unix':
                d = _timegm(d.timetuple())

            dts.append(d)

        return dts


    def get_stnids_unique(self):
        """
        Returns unique list of station IDs.

        Returns:
            list, station IDs
        """
        return list(set(self.stnids)) if not self.stnids is None else []



##### functions related to BurpFile #####

def print_btyp(btyp):
    """
    Prints BTYP decomposed into BKNAT, BKTYP, and BKSTP.

    Args:
        btyp : BTYP (int)
    Returns:
        None
    """
    b = bin(btyp)[2:].zfill(15)
    bknat = b[:4]
    bktyp = b[4:11]
    bkstp = b[11:]
    print("BKNAT  BKTYP    BKSTP")
    print("-----  -----    -----")
    print("{}   {}  {}".format(bknat, bktyp, bkstp))
    return


def copy_burp(brp_in, brp_out):
    """
    Copies all file, report, block, and code information from the
    input BurpFile to the output BurpFile, except does not copy over
    filename or mode.

    Args:
        brp_in  : BurpFile instance to copy data from
        brp_out : BurpFile instance to copy data to
    Returns:
        None
    """
    assert isinstance(brp_in, BurpFile) and isinstance(brp_out, BurpFile), "Input object must be an instance of BurpFile."
    for attr in BurpFile.burp_attr:
        setattr(brp_out, attr, _deepcopy(getattr(brp_in, attr)))
    return


def plot_burp(bf, code=None, cval=None, ax=None, level=0, mask=None, projection='cyl', cbar_opt={}, vals_opt={}, dparallel=30., dmeridian=60., fontsize=20, **kwargs):
    """
    Plots a BURP file. Will plot BUFR code if specified. Only plots a single level,
    which can be specified by the optional argument. Additional arguments not listed
    below will be passes to Basemap.scatter.

    Args:
        bf         : BurpFile instance
        code       : code to be plotted, if not set the observation locations will be plotted
        level      : level to be plotted
        mask       : mask to apply to data
        cval       : directly supply plotted values instead of using code argument
        ax         : subplot object
        projection : projection used by Basemap for plotting
        cbar_opt   : dictionary for colorbar options
        vals_opt   : dictionary for get_rval options if code is supplied
        dparallel  : spacing of parallels, if None will not plot parallels
        dmeridian  : spacing of meridians, if None will not plot meridians
        fontsize   : font size for labels of parallels and meridians

    Returns:
        {
            'm'    : Basemap.scatter object used for plotting
            'cbar' : colorbar object if used
        }
    """
    assert isinstance(bf, BurpFile), "First argument must be an instance of BurpFile"
    assert code is None or cval is None, "Only one of code, cval should be supplied as an argument"

    if ax is None:
        fig = _plt.figure(figsize=(18, 11))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if bf.nrep==0:
        return

    opt = kwargs.copy()
    if not 's' in opt.keys():
        opt['s'] = 10
    if not 'edgecolors' in opt.keys():
        opt['edgecolors'] = 'None'
    if not 'cmap' in opt.keys() and (code is not None or cval is not None):
        opt['cmap'] = _plt.cm.jet

    if code is not None:
        vals = bf.get_rval(code, **vals_opt)
        if len(vals.shape)>1:
            vals = vals[:, level]
        opt['c'] = vals
    elif cval is not None:
        opt['c'] = cval

    msk = _np.array([ stn[:2] for stn in bf.stnids ]) != '>>'  # don't plot resumes

    if not mask is None:
        msk = _np.logical_and(msk, mask)

    lon = bf.lon[msk]
    lat = bf.lat[msk]
    if code is not None or cval is not None:
        opt['c'] = opt['c'][msk]

    basemap_opt = {'projection':projection, 'resolution':'c'}

    if projection=='cyl':
        basemap_opt.update({'llcrnrlat':-90, 'urcrnrlat':90, 'llcrnrlon':-180, 'urcrnrlon':180})
    elif projection=='npstere':
        basemap_opt.update({'boundinglat':10., 'lon_0':0.})
    elif projection=='spstere':
        basemap_opt.update({'boundinglat':-10., 'lon_0':270.})

    m = _Basemap(ax=ax, **basemap_opt)

    m.drawcoastlines()

    xpt, ypt = m(lon, lat)

    sctr = m.scatter(xpt, ypt, **opt)

    if dparallel is not None:
        m.drawparallels(_np.arange(-90, 91, dparallel), labels=[1, 0, 0, 0], color='grey', fontsize=fontsize)
    if dmeridian is not None:
        m.drawmeridians(_np.arange(-180, 180, dmeridian), labels=[0, 0, 0, 1], color='grey', fontsize=fontsize)

    output = {'m':m}

    if code is not None or cval is not None:
        divider = _make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.5)
        cbar = fig.colorbar(sctr, ax=ax, cax=cax, **cbar_opt)
        output['cbar'] = cbar

    return output


# =========================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
