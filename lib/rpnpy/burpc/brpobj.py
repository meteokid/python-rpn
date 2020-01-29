#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc.burpc contains the wrapper classes to main burp_c C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]]
    documentation for more details.

See Also:
    rpnpy.burpc.base
    rpnpy.burpc.proto
    rpnpy.burpc.const
    rpnpy.librmn.burp
    rpnpy.utils.burpfile
"""
import ctypes as _ct
import numpy  as _np
# import numpy.ctypeslib as _npc
from rpnpy.burpc import proto as _bp
from rpnpy.burpc import const as _bc
from rpnpy.burpc import BurpcError
import rpnpy.librmn.all as _rmn
from rpnpy import C_WCHAR2CHAR_COND as _C_WCHAR2CHAR_COND
from rpnpy import C_CHAR2WCHAR_COND as _C_CHAR2WCHAR_COND
from rpnpy import C_MKSTR as _C_MKSTR

from rpnpy import integer_types as _integer_types
from rpnpy import range as _range

# Block shape (nele, nval, nt), Fortran order
_BLKIDX = lambda blk, e, v, t: e + blk[0].nele * (v + blk[0].nval * t)

class _BurpcObjBase(object):
    """
    Base class for BurpFiles, BurpRpt, BurpBlk, BurpEle

    See Also:
        BurpFiles
        BurpRpt
        BurpBlk
        BurpEle
    """
    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.todict())+')'

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        return self.next()

    def _getattr0(self, name):
        name = _C_CHAR2WCHAR_COND(name)
        return getattr(self, '_'+self.__class__.__name__+name)

    def __getattr__(self, name):
        try:
            name = _C_CHAR2WCHAR_COND(name)
            try:
                return _C_CHAR2WCHAR_COND(self.get(name))
            except:
                print(name,repr(self.get(name)))
                raise
        except KeyError as e:
            raise AttributeError(e)
            ## return super(self.__class__, self).__getattr__(name)
            ## return super(_BurpcObjBase, self).__getattr__(name)

    def __getitem__(self, name):
        name = _C_CHAR2WCHAR_COND(name)
        return self.get(name)

    def __delitem__(self, name):
        name = _C_CHAR2WCHAR_COND(name)
        return self.delete(name)
        ## try:
        ##     return self.delete(name)
        ## except KeyError:
        ##     return super(_BurpcObjBase, self).__delitem__(name)

    ## def __setattr__(self, name, value):
    ##     try:
    ##         return self.put(name, value)
    ##     except AttributeError:
    ##         return super(_BurpcObjBase, self).__setattr__(name, value)

    def __setitem__(self, name, value):
        name = _C_CHAR2WCHAR_COND(name)
        value = _C_CHAR2WCHAR_COND(value)
        return self.put(name, value)

    #TODO: def __delattr__(self, name):
    #TODO: def __coerce__(self, other):
    #TODO: def __cmp__(self, other):
    #TODO: def __sub__(self, other):
    #TODO: def __add__(self, nhours):
    #TODO: def __isub__(self, other):
    #TODO: def __iadd__(self, nhours):

    def update(self, values):
        """
        Update attributes with provided values in a dict
        """
        if not isinstance(values, (dict, self.__class__)):
            raise TypeError("Type not supported for values: "+str(type(values)))
        for k in self._getattr0('__attrlist'):
            try:
                self.__setitem__(k, values[k])
            except (KeyError, AttributeError):
                pass

    def getptr(self):
        """
        Return the pointer to the BURP object structure
        """
        return  self._getattr0('__ptr')

    def todict(self):
        """
        Return the list of {attributes : values} as a dict
        """
        return dict([(k, getattr(self, k)) for k in
                     self._getattr0('__attrlist') +
                     self._getattr0('__attrlist2')])

    ## def get(self, name):  #to be defined by child class
    ## def delete(self, name):  #to be defined by child class
    ## def put(self, name, value):  #to be defined by child class
    ## def next(self):  #to be defined by child class

    #TODO: add list/dict type operators: count?, extend?, index?, insert?, pop?, remove?, reverse?, sort?... see help([]) help({}) for other __?__ operators


class BurpcFile(_BurpcObjBase):
    """
    Python Class to refer to, interact with a BURP file using the burp_c lib

    bfile = BurpcFile(filename)
    bfile = BurpcFile(filename, filemode)
    bfile = BurpcFile(filename, filemode, funit)

    Attributes:
        filename : Name of the opened file
        filemode : Access specifier mode used when opening the file
                   Should be one of:
                   BRP_FILE_READ, BRP_FILE_WRITE, BRP_FILE_APPEND
        funit    : File unit number

    Examples:
    >>> import os, os.path
    >>> import rpnpy.burpc.all as brp
    >>> import rpnpy.librmn.all as rmn
    >>> m = brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
    >>> print('# nrep = '+str(len(bfile)))
    # nrep = 47544
    >>>
    >>> #get the first report in file
    >>> rpt = bfile[0]
    >>>
    >>> # Get 1st report matching stnid 'A********'
    >>> rpt = bfile.get({'stnid' : 'A********'})
    >>> print('# stnid={stnid}, handle={handle}'.format(**rpt.todict()))
    # stnid=ASEU05   , handle=33793
    >>>
    >>> # Get next report matching stnid 'A********'
    >>> rpt = bfile.get({'stnid' : 'A********', 'handle': rpt.handle})
    >>> print('# stnid={stnid}, handle={handle}'.format(**rpt.todict()))
    # stnid=AF309    , handle=1199105
    >>>
    >>> # Loop over all report and print info
    >>> for rpt in bfile:
    ...     if rpt.stnid.strip() == '71915':
    ...        print('# stnid=' + repr(rpt.stnid))
    # stnid='71915    '
    >>>
    >>> # Close the file
    >>> del bfile
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
    >>>
    >>> # Open file in write mode with auto file closing and error handling
    >>> with brp.BurpcFile('tmpburpfile.brp', brp.BRP_FILE_WRITE) as bfileout:
    ...     # Copy report with stnid GOES11 to the new file
    ...     rpt = bfile.get({'stnid' : 'GOES11   '})
    ...     bfileout.append(rpt)
    >>> del bfile  # bfileout was auto closed at the end of the 'with' code block
    >>>
    >>> #Verify that the report was written to tmpburpfile.brp
    >>> bfile = brp.BurpcFile('tmpburpfile.brp')
    >>> rpt = bfile.get({'stnid' : 'GOES11   '})
    >>> print('# stnid=' + repr(rpt.stnid))
    # stnid='GOES11   '
    >>> # The file will auto close at the end of the program

    See Also:
        BurpcRpt
        rpnpy.burpc.base.brp_open
        rpnpy.burpc.base.brp_close
        rpnpy.burpc.base
        rpnpy.burpc.const
    """
    __attrlist  = ("filename", "filemode", "funit")
    __attrlist2 = ()

    def __init__(self, filename, filemode='r', funit=0):
        self.filename = _C_CHAR2WCHAR_COND(filename)
        self.filemode = _C_CHAR2WCHAR_COND(filemode)
        self.funit    = funit
        if isinstance(filename, dict):
            if 'filename' in filename.keys():
                self.filename = _C_CHAR2WCHAR_COND(filename['filename'])
            if 'filemode' in filename.keys():
                self.filemode = _C_CHAR2WCHAR_COND(filename['filemode'])
            if 'funit' in filename.keys():
                self.funit = filename['funit']
        self.__iteridx   = BurpcRpt() #0
        self.__handles   = []
        self.__rpt       = None
        fstmode, brpmode, brpcmode = _bp.brp_filemode(self.filemode)
        self.funit       = _rmn.get_funit(self.filename, fstmode, self.funit)
        self.nrep        = _bp.c_brp_open(self.funit,
                                          _C_WCHAR2CHAR_COND(self.filename),
                                          _C_WCHAR2CHAR_COND(brpcmode))
        self.__brpmode = brpmode
        if self.nrep < 0:
            raise BurpcError('Problem opening with mode {} the file: {}'
                             .format(repr(brpcmode), repr(self.filename)))
        self.__ptr = self.funit

    def __del__(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, mytype, myvalue, mytraceback):
        self._close()

    def __len__(self):
        return max(0, self.nrep)

    def __iter__(self):
        self.__iteridx = BurpcRpt() #0
        return self

    def next(self):  # Python 2
        """
        Get the next item in the iterator, Internal function for python 2 iter

        Do not call explictly, this will be used in 'for loops' and other iterators.
        """
        if _bp.c_brp_findrpt(self.funit, self.__iteridx.getptr()) >= 0:
            self.__rpt = BurpcRpt()
            if _bp.c_brp_getrpt(self.funit, self.__iteridx.handle,
                                self.__rpt.getptr()) >= 0:
                return self.__rpt
        self.__iteridx = BurpcRpt()
        raise StopIteration

    ## def __setitem__(self, name, value):
    ##     #TODO: Should replace the rpt found with getitem(name) or add a new one

    def _close(self):
        if self.funit:
            istat = _bp.c_brp_close(self.funit)
        self.funit = None

    ## def del(self, search): #TODO: __delitem__
    ##     raise Error

    def get(self, key=None, rpt=None):
        """
        Find a report and get its meta + data

        rpt = burpfile.get(report_number)
        rpt = burpfile.get(rpt)
        rpt = burpfile.get(rptdict)

        Args:
            key : Search criterions
                  if int, return the ith ([0, nrep[) report in file
                  if dict or BurpcRpt, search report matching given params
            rpt : (optional) BurpcRpt used to put the result to recycle memory
        Return:
            BurpcRpt if a report match the search key
            None otherwise
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        #TODO: review rpt recycling
        ## rpt = BurpcRpt()
        rpt = rpt if isinstance(rpt, BurpcRpt) else BurpcRpt(rpt)
        if key is None or isinstance(key, (BurpcRpt, dict)):
            key = key if isinstance(key, BurpcRpt) else BurpcRpt(key)
            if _bp.c_brp_findrpt(self.funit, key.getptr()) >= 0:
                if _bp.c_brp_getrpt(self.funit, key.handle,
                                    rpt.getptr()) >= 0:
                    return rpt
            return None
        elif isinstance(key, _integer_types):
            if key < 0 or key >= self.nrep:
                raise IndexError('Index out of range: [0:{}['.format(self.nrep))
            if key >= len(self.__handles):
                i0 = len(self.__handles)
                key1 = BurpcRpt()
                if i0 > 0:
                    key1.handle = self.__handles[-1]
                for i in _range(i0, key+1):
                    if _bp.c_brp_findrpt(self.funit, key1.getptr()) >= 0:
                        self.__handles.append(key1.handle)
                    else:
                        break
            if _bp.c_brp_getrpt(self.funit, self.__handles[key],
                                rpt.getptr()) >= 0:
                return rpt
        else:
            raise TypeError("For Name: {}, Not Supported Type: {}".
                            format(repr(key), str(type(key))))

    def put(self, where, rpt):
        """
        Write a report to the burp file

        burpfile.put(BRP_END_BURP_FILE, rpt)
        burpfile.put(rpt.handle, rpt)

        Args:
            where : location to write report to
                    if None or BRP_END_BURP_FILE, append to the file
                    if int, handle of report to replace in file
            rpt   : BurpcRpt to write
        Return:
            None
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        if not isinstance(rpt, BurpcRpt):
            raise TypeError("rpt should be of type BurpcRpt, got: {}, ".
                            format(str(type(rpt))))
        if self.__brpmode not in (_rmn.BURP_MODE_CREATE,
                                   _rmn.BURP_MODE_APPEND):
            raise BurpcError('BurpcFile.put(): file must be opened with '+
                             'write flag, got: {}'.format(self.__brpcmode))
        append = where is None
        if append:
            where = _bc.BRP_END_BURP_FILE
        ## elif isinstance(where, (BurpcRpt, dict)): #TODO:
        ## elif isinstance(where, _integer_types): #TODO: same indexing as get, how to specify a handle?
        else:
            raise TypeError("For where: {}, Not Supported Type: {}".
                            format(repr(where), str(type(where))))

        self.__handles = []  #TODO: ?best place to invalidate the cache?
        rpt.append_flush(self.funit)
        prpt = rpt.getptr() if isinstance(rpt, BurpcRpt) else rpt
        if _bp.c_brp_writerpt(self.funit, prpt, where) < 0:
            raise BurpcError('BurpcFile.put(): Problem in brp_writerpt')
        if append:
            self.nrep += 1

    def append(self, rpt):
        """
        Append a report to the burp file

        burpfile.append(rpt)

        Args:
            rpt : BurpcRpt to write
        Return:
            None
        Raises:
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        self.put(None, rpt)


class BurpcRpt(_BurpcObjBase):
    """
    Python Class equivalent of the burp_c's BURP_RPT C structure to hold
    the BURP report data

    rpt1 = BurpcRpt()
    rpt2 = BurpcRpt(rpt1)
    rpt3 = BurpcRpt(report_meta_dict)

    Attributes:
        handle : Report handle
        nsize  : report data size
        temps  : Observation time/hour (HHMM)
        flgs   : Global flags
                 (24 bits, Bit 0 is the right most bit of the word)
                 See BURP_FLAGS_IDX_NAME for Bits/flags desc.
        stnid  : Station ID
                 If it is a surface station, STNID = WMO number.
                 The name is aligned at left and filled with
                 spaces. In the case of regrouped data,
                 STNID contains blanks.
        idtype : Report Type
        lati   : Station latitude (1/100 of degrees)
                 with respect to the south pole. (0 to 1800)
                 (100*(latitude+90)) of a station or the
                 lower left corner of a box.
        longi  : Station longitude (1/100 of degrees)
                 (0 to 36000) of a station or lower left corner of a box.
        dx     : Width of a box for regrouped data (degrees)
        dy     : Height of a box for regrouped data (degrees)
        elev   : Station altitude (metres)
        drnd   : Reception delay: difference between the
                 reception time at CMC and the time of observation
                 (TIME). For the regrouped data, DRND indicates
                 the amount of data. DRND = 0 in other cases.
        date   : Report valid date (YYYYMMDD)
        oars   : Reserved for the Objective Analysis. (0-->65535)
        runn   : Operational pass identification.
        dblk   : "deffered append" blocks
                 Due to BURP API, blocks cannot be added to a report before
                 its header is written to a file. Hence they are kept separetely
                 as a list in "dblk" until the report is added to a file.
        nblk   : number of blocks w/ "deffered append" blocks
        nblk0  : number of blocks w/o "deffered append" blocks
        lngr   : 
        time   : Observation time/hour (HHMM)
        timehh : Observation time hour part (HH)
        timemm : Observation time minutes part (MM)
        flgsl  : Global flags as a list of int
                 See BURP_FLAGS_IDX for Bits/flags desc.
        flgsd  : Description of set flgs, comma separated
        idtyp  : Report Type
        idtypd : Report Type description
        ilat   : lati
        lat    : Station latitude (degrees)
        ilon   : longi
        lon    : Station longitude (degrees)
        idx    : Width of a box for regrouped data
                 (delta lon, 1/10 of degrees)
        rdx    : Width of a box for regrouped data (degrees)
        idy    : Height of a box for regrouped data
                 (delta lat, 1/10 of degrees)
        rdy    : Height of a box for regrouped data (degrees)
        ielev  : Station altitude (metres + 400.) (0 to 8191)
        relev  : Station altitude (metres)
        dateyy : Report valid date (YYYY)
        datemm : Report valid date (MM)
        datedd : Report valid date (DD)
        sup    : supplementary primary keys array
                 (reserved for future expansion).
        nsup   : number of sup
        xaux   : supplementary auxiliary keys array
                 (reserved for future expansion).
        nxaux  : number of xaux

    Examples:
    >>> import os, os.path
    >>> import rpnpy.burpc.all as brp
    >>> import rpnpy.librmn.all as rmn
    >>> m = brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
    >>>
    >>> # get the first report in file and print some info
    >>> rpt = bfile[0]
    >>> print("# report date={}, time={}".format(rpt.date, rpt.time))
    # report date=20070219, time=0
    >>>
    >>> # Copy a report
    >>> rpt1 = brp.BurpcRpt(rpt)
    >>> rpt1.date = 20171010
    >>> print("# report date={}, time={}".format(rpt.date, rpt.time))
    # report date=20070219, time=0
    >>> print("# report date={}, time={}".format(rpt1.date, rpt1.time))
    # report date=20171010, time=0
    >>>
    >>> # get the first block in report
    >>> blk = rpt[0]
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 1, data, data seen by OA at altitude, global model
    >>>
    >>> # get first block matching btyp == 15456
    >>> blk = rpt.get({'btyp':15456})
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 6, flags, data seen by OA at altitude, global model
    >>>
    >>> # Loop over all blocks in report and print info for last one
    >>> for blk in rpt:
    ...     pass  # Do something with the block
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 12, data, data seen by OA at altitude, global model
    >>>
    >>> # New empty report
    >>> rpt = brp.BurpcRpt()
    >>>
    >>> # New report from dict
    >>> rpt = brp.BurpcRpt({'date' : 20171111, 'temps' : 1213})
    >>> print("# report date={}, time={}".format(rpt.date, rpt.time))
    # report date=20171111, time=1213

    See Also:
        BurpcFile
        BurpcBlk
        rpnpy.burpc.base.brp_newrpt
        rpnpy.burpc.base.brp_freerpt
        rpnpy.burpc.base.brp_findrpt
        rpnpy.burpc.base.brp_getrpt
        rpnpy.burpc.base
        rpnpy.burpc.const
    """
    __attrlist = ("handle", "nsize", "temps", "flgs", "stnid",
                  "idtype", "lati", "longi", "dx", "dy", "elev",
                  "drnd", "date", "oars", "runn", "lngr")
    __attrlist2 = ('time', 'timehh', 'timemm', 'flgsl', 'flgsd',
                   'idtyp', 'idtypd', 'ilat', 'lat', 'ilon', 'lon',
                   'idx', 'rdx', 'idy', 'rdy', 'ielev', 'relev',
                   'dateyy', 'datemm', 'datedd', 'dblk', 'nblk', 'nblk0',
                   'sup', 'nsup', 'xaux', 'nxaux')
    __attrlist2names = {
        'rdx'   : 'dx',
        'rdy'   : 'dy',
        'relev' : 'elev'
        }

    def __init__(self, rpt=None):
        self.__bkno = 0
        self.__blk  = None
        self.__dblk = []
        self.__derived = None
        self.__attrlist2names_keys = self.__attrlist2names.keys()
        self.__ptr = None
        if rpt is None:
            ## print 'NEW:',self.__class__.__name__
            self.__ptr = _bp.c_brp_newrpt()
        elif isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
            ## print 'NEW:',self.__class__.__name__,'ptr'
            self.__ptr = rpt  #TODO: copy?
        else:
            ## print 'NEW:',self.__class__.__name__,'update'
            self.__ptr = _bp.c_brp_newrpt()
            self.update(rpt)

    def __del__(self):
        ## print 'DEL:',self.__class__.__name__
        _bp.c_brp_freerpt(self.__ptr) #TODO

    ## def __len__(self): #TODO: not working with this def... find out why and fix it?
    ##     if self.nblk:
    ##         return self.nblk
    ##     return 0

    def __iter__(self):
        self.__bkno = 0
        return self

    def next(self): # Python 2:
        """
        Get the next item in the iterator, Internal function for python 2 iter

        Do not call explictly, this will be used in 'for loops' and other iterators.
        """
        if self.__bkno >= self.nblk:
            self.__bkno = 0
            raise StopIteration
        self.__blk = self.get(self.__bkno, self.__blk)
        self.__bkno += 1
        return self.__blk

    def get(self, key=None, blk=None):
        """
        Find a block and get its meta + data

        value = rpt.get(attr_name)
        blk   = rpt.get(item_number)
        blk   = rpt.get(blk)
        blk   = rpt.get(blkdict)

        Args:
            key : Attribute name or Search criterions
                  if str, return the attribute value
                  if int, return the ith ([0, nblk[) block in file
                  if dict or BurpcBlk, search block matching given params
            blk : (optional) BurpcBlk use to put the result to recycle memory
        Return:
            Attribute value or
            BurpcBlk if a report match the search
            None otherwise
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "rpt.attr_name"
            instead of "rpt.get('attr_name')"
        """
        key = _C_CHAR2WCHAR_COND(key)
        if key in self.__class__.__attrlist:
            v = getattr(self.__ptr[0], key)  #TODO: use proto fn?
            return _C_CHAR2WCHAR_COND(v)
        elif key in self.__class__.__attrlist2:
            try:
                key2 = self.__attrlist2names[key]
            except KeyError:
                key2 = key
            return _C_CHAR2WCHAR_COND(self._derived_attr()[key2])
        elif isinstance(key, _integer_types):
            key += 1
            if key < 1 or key > self.nblk:
                raise IndexError('Index out of range: [0:{}['.format(self.nblk))
            #TODO: review blk recycling
            ## blk = blk if isinstance(blk, BurpcBlk) else BurpcBlk(blk)
            if key <= self.nblk0:
                blk = BurpcBlk()
                if _bp.c_brp_getblk(key, blk.getptr(), self.getptr()) < 0:
                    raise BurpcError('Problem in c_brp_getblk: {}/{}'
                                     .format(key, self.nblk0))
                return blk
            else:
                return self.__dblk[key-self.nblk0-1]
        elif key is None or isinstance(key, (BurpcBlk, dict)):
            #TODO: implement search in "deffered append blk"
            search = key if isinstance(key, BurpcBlk) else BurpcBlk(key)
            if _bp.c_brp_findblk(search.getptr(), self.getptr()) >= 0:
                #TODO: review blk recycling
                ## blk = blk if isinstance(blk, BurpcBlk) else BurpcBlk(blk)
                blk = BurpcBlk()
                if _bp.c_brp_getblk(search.bkno, blk.getptr(),
                                    self.getptr()) >= 0:
                    return blk
            return None
        raise KeyError("{} object has no such key: {}"
                       .format(self.__class__.__name__, repr(key)))

    def __setattr__(self, key, value): #TODO: move to super class
        return self.put(key, value)

    def put(self, key, value):
        """
        Add a block to the report or set attribute value

        rpt.put(attr_name, value)

        Args:
            key   : Attribute name
            value : Value to set or blk object to set
        Return:
            None
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "rpt.attr_name = value"
            instead of "rpt.put('attr_name', value)"
        """
        ## rpt.put(bkno, blk)
        ## rpt.put(blk0, blk)
        ## rpt.put(blkdict, blk)
        ##
        ## Args:
        ##     key   : Attribute name or Search criterions
        ##             if str, set the attribute value
        ##             if int, set the ith ([0, nblk[) block in report
        ##             if dict or BurpcBlk, replace block matching given params
        key = _C_CHAR2WCHAR_COND(key)
        bvalue = _C_WCHAR2CHAR_COND(value)
        value = _C_CHAR2WCHAR_COND(value)
        if key == 'stnid':
            self.__derived = None
            _bp.c_brp_setstnid(self.__ptr, bvalue)
        elif key in self.__class__.__attrlist:
            self.__derived = None
            if self.__ptr[0].getType(key) == _ct.c_int:
                bvalue = int(float(bvalue))            
            setattr(self.__ptr[0], key, bvalue)  #TODO: use proto fn?
            return
        elif key in self.__class__.__attrlist2:
            #TODO: encode other items on the fly
            raise AttributeError(self.__class__.__name__+
                                 " object cannot set derived attribute '"+
                                 key+"'")
        elif isinstance(key, _integer_types): #TODO:
            raise BurpcError('BurpcRpt.put(index, blk): not yet implemented with specific index, try the BurpcRpt.append(blk) method')
        ## elif isinstance(key, (BurpcBlk, dict)): #TODO:
        elif key is None:
            if not isinstance(value, BurpcBlk):
                try:
                    value = BurpcBlk(value)
                except:
                    raise TypeError('Provided value should be of type BurpcBlk')
            self.__dblk.append(value)
            self.__derived = None
        else:
            return super(self.__class__, self).__setattr__(key, value)
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+key+"'")

    def append_flush(self, iunit):
        """
        Add report to file and flush the deffered blocks into it.

        rpt.append_flush(iunit)

        Args:
            iunit : Burp file unit number, opened with BURP_MODE_CREATE
        Return:
            None
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            BurpcError on any other error

        Notes:
            This method is to be called from the BurpcFile class, NOT directly
        """
        if len(self.__dblk) == 0:
            if self.nsize > 0 and self.nblk0 > 0:
                _bp.c_brp_updrpthdr(iunit, self.__ptr)
            else:
                _bp.c_brp_putrpthdr(iunit, self.__ptr)
            return
        blksize0 = 0
        for blk in self.__dblk:
            blksize0 += _rmn.LBLK(blk.nele, blk.nval, blk.nt, blk.nbit)
        blksize  = int(_rmn.LRPT(blksize0) * 1.5)  # minimum size * 1.5 (ad hoc)
        if self.__ptr[0].nsize <= 0:
            _bp.c_brp_allocrpt(self.__ptr, blksize)
            _bp.c_brp_clrrpt(self.__ptr)
        else:
            blksize2 = self.__ptr[0].nsize + blksize
            _bp.c_brp_resizerpt(self.__ptr, blksize2)
        ## print self.nsize,self.nblk0
        if self.nsize > 0 and self.nblk0 > 0:
            _bp.c_brp_updrpthdr(iunit, self.__ptr)
        else:
            _bp.c_brp_putrpthdr(iunit, self.__ptr)
        for blk in self.__dblk:
            if _bp.c_brp_putblk(self.__ptr, blk.getptr()) < 0:
                raise BurpcError('BurpcRpt.append_flush(): problem in c_brp_putblk()')
        self.__derived = None
        self.__dblk = []

    def append(self, blk):
        """
        Append a block to report

        rpt.append(blk)

        Args:
            blk : BurpcBlk to append
        Return:
            None
        Raises:
            TypeError  on not supported types or args
            BurpcError on any other error
        """
        self.put(None, blk)

    def _derived_attr(self):
        """Return dict with derived attributs (Cached version)"""
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()


    def __derived_attr(self):
        """Return dict with derived attributs"""
        itime = getattr(self.__ptr[0], 'temps')
        iflgs = getattr(self.__ptr[0], 'flgs')
        flgs_dict = _rmn.flags_decode(iflgs, raise_error=False)
        idtyp = getattr(self.__ptr[0], 'idtype')
        ilat  = getattr(self.__ptr[0], 'lati')
        ilon  = getattr(self.__ptr[0], 'longi')
        idx   = getattr(self.__ptr[0], 'dx')
        idy   = getattr(self.__ptr[0], 'dy')
        ialt  = getattr(self.__ptr[0], 'elev')
        idate = getattr(self.__ptr[0], 'date')
        nblk  = getattr(self.__ptr[0], 'nblk')
        try:
            idtyp_desc = _rmn.BURP_IDTYP_DESC[str(idtyp)]
        except KeyError:
            idtyp_desc = ''
        return {
            'time'  : itime,
            'timehh': itime // 100,
            'timemm': itime % 100,
            'flgs'  : flgs_dict['flgs'],
            'flgsl' : flgs_dict['flgsl'],
            'flgsd' : flgs_dict['flgsd'],
            'stnid' : _C_CHAR2WCHAR_COND(getattr(self.__ptr[0], 'stnid')),
            'idtyp' : idtyp,
            'idtypd': idtyp_desc,
            'ilat'  : ilat,
            'lat'   : (float(ilat)/100.) - 90.,
            'ilon'  : ilon,
            'lon'   : float(ilon)/100.,
            'idx'   : idx,
            'dx'    : float(idx)/10.,
            'idy'   : idy,
            'dy'    : float(idy)/10.,
            'ielev' : ialt,
            'elev'  : float(ialt) - 400.,
            'drnd'  : getattr(self.__ptr[0], 'drnd'),
            'date'  : idate,
            'dateyy': idate // 10000,
            'datemm': (idate % 10000) // 100,
            'datedd': (idate % 10000) % 100,
            'oars'  : getattr(self.__ptr[0], 'oars'),
            'runn'  : getattr(self.__ptr[0], 'runn'),
            'dblk'  : self.__dblk,
            'nblk'  : nblk + len(self.__dblk),
            'nblk0' : nblk, #Actual nb blocks w/o defered append blk
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
            }


#TODO: class BurpcBlkPlus(BurpcBlk): BurpcBlk + BurpcRpt attributes
## class BurpcRptBlk(BurpcBlk):
##     """
##     """


class BurpcBlk(_BurpcObjBase):
    """
    Python Class equivalent of the burp_c's BURP_BLK C structure to hold
    the BURP block data

    blk1 = BurpcBlk()
    blk2 = BurpcBlk(blk1)
    blk3 = BurpcBlk(block_meta_dict)

    Attributes:
        bkno  : block number
        nele  : Number of meteorological elements in a block.
                1st dimension of the array TBLVAL(block). (0-127)
        nval  : Number of values per element.
                2nd dimension of TBLVAL(block). (0-255)
        nt    : Number of groups of NELE by NVAL values in a block.
                3rd dimension of TBLVAL(block).
        bfam  : Family block descriptor. (0-31)
        bdesc : Block descriptor. (0-2047) (not used)
        btyp  : Block type (0-2047), made from 3 components:
                BKNAT: kind component of Block type
                BKTYP: Data-type component of Block type
                BKSTP: Sub data-type component of Block type
        nbit  : Number of bits per value.
                When we add a block, we should insure that the number of bits
                specified is large enough to represent the biggest value
                contained in the array of values in TBLVAL.
                The maximum number of bits is 32.
        bit0  : Number of the first right bit from block,
                calculated automatically by the software.
                (0-->2**26-1) (always a multiple of 64 minus 1)
        datyp : Data type (for packing/unpacking).
                See rpnpy.librmn.burp_const BURP_DATYP_LIST and BURP_DATYP2NUMPY_LIST
                0 = string of bits (bit string)
                2 = unsigned integers
                3 = characters (NBIT must be equal to 8)
                4 = signed integers
                5 = uppercase characters (the lowercase characters
                    will be converted to uppercase during the read.
                    (NBIT must be equal to 8)
                6 = real*4 (ie: 32bits)
                7 = real*8 (ie: 64bits)
                8 = complex*4 (ie: 2 times 32bits)
                9 = complex*8 (ie: 2 times 64bits)
                Note: Type 3 and 5 are processed like strings of bits thus,
                      the user should do the data compression himself.
        store_type : Type of data in table val, one of:
                     BRP_STORE_INTEGER, BRP_STORE_FLOAT,
                     BRP_STORE_DOUBLE, BRP_STORE_CHAR
        max_nval : 
        max_nele : 
        max_nt : 
        max_len : 
        lstele  : list of coded elements (CMCID)
                  shape: (nele, )
        dlstele : list of decoded elements (BUFRID)
                  shape: (nele, )
        tblval  : table of coded values
                  or table of decoded int values (BRP_STORE_INTEGER)
                  shape: (nele, nval, nt), Fortran order
        ival    : table of decoded values of type int (BRP_STORE_INTEGER)
                  shape: (nele, nval, nt), Fortran order
        rval    : table of decoded values of type real/float (BRP_STORE_FLOAT)
                  shape: (nele, nval, nt), Fortran order
        drval   : table of decoded values of type real/float double (BRP_STORE_DOUBLE)
                  shape: (nele, nval, nt), Fortran order
        charval : table of decoded values of type char (BRP_STORE_CHAR)
                  shape: (nele, nval, nt), Fortran order
        bknat : block type, kind component
        bknat_multi : block type, kind component, uni/multi bit
                                  0=uni, 1=multi
        bknat_kind : block type, kind component, kind value
                                  See BURP_BKNAT_KIND_DESC
        bknat_kindd : desc of bknat_kind
        bktyp : block type, Data-type component
        bktyp_alt : block type, Data-type component, surf/alt bit
                                  0=surf, 1=alt
        bktyp_kind : block type, Data-type component, flags
                                  See BURP_BKTYP_KIND_DESC
        bktyp_kindd : desc of bktyp_kind
        bkstp : block type, Sub data-type component
        bkstpd : desc of bktyp_kindd
        datypd : Data type name/desc

    Examples:
    >>> import os, os.path
    >>> import rpnpy.burpc.all as brp
    >>> import rpnpy.librmn.all as rmn
    >>> m = brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
    >>>
    >>> # get the first report in file and print some info
    >>> rpt = bfile[0]
    >>>
    >>> # get the first block in report
    >>> blk = rpt[0]
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 1, data, data seen by OA at altitude, global model
    >>>
    >>> # Copy a block
    >>> blk1 = brp.BurpcBlk(blk)
    >>> blk1.btyp = 6
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 1, data, data seen by OA at altitude, global model
    >>> print("# block bkno = {}, {}, {}".format(blk1.bkno, blk1.bknat_kindd, blk1.bktyp_kindd))
    # block bkno = 1, data, observations (ADE)
    >>>
    >>> # get the first element in blk
    >>> ele = blk[0]
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval[0,0]))
    # 10004: PRESSURE, (units=PA), shape=[1, 1] : value=100.0
    >>>
    >>> # Loop over all elements in block and print info for last one
    >>> for ele in blk:
    ...     pass  # Do something with the element
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={:7.2e}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval[0,0]))
    # 13220: NATURAL LOG SFC SPEC HUMIDITY (2M), (units=LN(KG/KG)), shape=[1, 1] : value=1.00e+30
    >>>
    >>> # New empty block
    >>> blk = brp.BurpcBlk()
    >>>
    >>> # New block from dict
    >>> blk = brp.BurpcBlk({'bkno' : 1, 'btyp' : 6})
    >>> print("# block bkno = {}, {}, {}".format(blk.bkno, blk.bknat_kindd, blk.bktyp_kindd))
    # block bkno = 1, data, observations (ADE)

    See Also:
        BurpcFile
        BurpcRpt
        BurpcEle
        rpnpy.burpc.base.brp_newblk
        rpnpy.burpc.base.brp_freeblk
        rpnpy.burpc.base.brp_findblk
        rpnpy.burpc.base.brp_getblk
        rpnpy.burpc.base
        rpnpy.burpc.const
   """
    __attrlist = ("bkno", "nele", "nval", "nt", "bfam", "bdesc", "btyp",
                  "bknat", "bktyp", "bkstp", "nbit", "bit0", "datyp",
                  "store_type",
                  ## "lstele", "dlstele", "tblval", "rval", "drval", "charval",
                  "max_nval", "max_nele", "max_nt", "max_len")
    __attrlist_np_1d = ("lstele", "dlstele")
    __attrlist_np_3d = ("tblval", "ival", "rval", "drval", "charval")
    __attrlist2 = ('bkno', 'nele', 'nval', 'nt', 'bfam', 'bdesc', 'btyp',
                   'bknat', 'bknat_multi', 'bknat_kind', 'bknat_kindd',
                   'bktyp', 'bktyp_alt', 'bktyp_kind', 'bktyp_kindd',
                   'bkstp', 'bkstpd', 'nbit', 'bit0', 'datyp', 'datypd')
    __PTRKEY2NUMPY = {
        'tblval'  : _np.int32,
        'ival'    : _np.int32,
        'rval'    : _np.float32,
        'drval'   : _np.float64,
        'charval' : _np.uint8
        }

    def __init__(self, blk=None):
        self.__eleno = 0
        self.__derived = None
        self.__ptr = None
        if blk is None:
            self.__ptr = _bp.c_brp_newblk()
        elif isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
            self.__ptr = blk  #TODO: copy?
        elif isinstance(blk, dict):
            self.__ptr = _bp.c_brp_newblk()
            self.update(blk)
        elif isinstance(blk, self.__class__):
            self.__ptr = _bp.c_brp_newblk()
            for ele in blk:
                self.append(ele)
            self.update(blk)
        else:
            raise TypeError('BurpcBlk: cannot init with blk of type:{}'
                            .format(type(blk)))
        self.reset_arrays()


    def __del__(self):
        ## print 'DEL:',self.__class__.__name__
        _bp.c_brp_freeblk(self.__ptr)

    ## def __len__(self): #TODO: not working with this def... find out why and fix it?
    ##     l = self.nele  # getattr(self.__ptr[0], 'nele')
    ##     print '\nblklen=',self.nele, self.nval, self.nt
    ##     if l >= 0:
    ##         return l
    ##     return 0

    def __iter__(self):
        self.__eleno = 0
        return self

    def next(self): # Python 2
        """
        Get the next item in the iterator, Internal function for python 2 iter

        Do not call explictly, this will be used in 'for loops' and other iterators.
        """
        if self.__eleno >= self.nele:
            self.__eleno = 0
            raise StopIteration
        ele = self._getelem(self.__eleno)
        self.__eleno += 1
        return ele

    def get(self, key):
        """
        Get a block attribute or Element

        value = blk.get(attr_name)
        ele   = blk.get(element_number)

        Args:
            key : Attribute name or Search criterions
                  if str, return the attribute value
                  if int, return the ith ([0, nblk[) block in file
                  if dict or BurpcBlk, search block matching given params
        Return:
            Attribute value or
            BurpcEle if a report match the search
            None otherwise
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "blk.attr_name"
            instead of "blk.get('attr_name')"
       """
        ## print 'getattr:', key
        key = _C_CHAR2WCHAR_COND(key)
        if key in self.__class__.__attrlist_np_1d:
            if self.__arr[key] is None:
                v = _C_CHAR2WCHAR_COND(getattr(self.__ptr[0], key))
                self.__arr[key] = _np.ctypeslib.as_array(v, (self.nele,))
            return self.__arr[key]
        elif key in self.__class__.__attrlist_np_3d:
            if self.__arr[key] is None:
                key2 = 'tblval'
                if self.__arr[key2] is None:
                    v = _C_CHAR2WCHAR_COND(getattr(self.__ptr[0], key2))
                    self.__arr[key2] = _np.ctypeslib.as_array(v,
                                        (self.nt, self.nval, self.nele)).T
                if key != key2:
                    dtype = self.__PTRKEY2NUMPY[key]
                    cmcids = _np.asfortranarray(
                        _np.ctypeslib.as_array(self.__ptr[0].lstele,
                                               (self.nele, )),
                        dtype=_np.int32)
                    shape = (self.nele, self.nval, self.nt)
                    self.__arr[key] = _np.reshape(_np.asfortranarray(
                        _rmn.mrbcvt_decode(cmcids,
                                           self.__arr[key2].copy(order='F')),
                        dtype=dtype), shape, order='F')
            return self.__arr[key]
        elif key in self.__class__.__attrlist:
            return _C_CHAR2WCHAR_COND(getattr(self.__ptr[0], key))  #TODO: use proto fn?
        elif key in self.__class__.__attrlist2:
            if not self.__derived:
                self.__derived = self._derived_attr()
            return self.__derived[key]
        elif isinstance(key, _integer_types):
            return self._getelem(key)
        #TODO: isinstance(key, BurpcEle)
        #TODO: isinstance(key, dict)
        else:
            raise KeyError("{} object has no such key: {}"
                           .format(self.__class__.__name__, repr(key)))

    def __setattr__(self, key, value): #TODO: move to super class
        return self.put(key, value)

    def put(self, key, value):
        """
        Add an element to the block or set attribute value

        blk.put(attr_name, value)
        blk.put(eleno, ele)
        blk.put(ele0, ele)
        blk.put(eledict, ele)

        Args:
            key   : Attribute name or Search criterions
                    if str, set the attribute value
                    if int, set the ith ([0, nblk[) element in block
                    if dict or BurpcBlk, replace element matching given params
            value : Value to set or blk object to set
        Return:
            None
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "blk.attr_name = value"
            instead of "blk.put('attr_name', value)"
         """
        ## print 'setattr:', key
        key = _C_CHAR2WCHAR_COND(key)
        bvalue = _C_WCHAR2CHAR_COND(value)
        value = _C_CHAR2WCHAR_COND(value)
        if key in self.__class__.__attrlist:
            self.__derived = None
            if self.__ptr[0].getType(key) == _ct.c_int:
                bvalue = int(float(bvalue))
            return setattr(self.__ptr[0], key, bvalue) #TODO: use proto fn?
        elif key in self.__class__.__attrlist2:
            #TODO: encode other items on the fly
            raise AttributeError(self.__class__.__name__+
                                 " object cannot set derived attribute '"+
                                 key+"'")
        elif key is None or isinstance(key, _integer_types):
            self._putelem(key, value)
        ## elif isinstance(key, (BurpcEle, dict)):
        ##     raise BurpcError('BurpcBlk.put(index, BurpcEle) - Not yet implemented')  #TODO
        ##     #Find element index/idx matching BurpcEle or dict
        ##     #self._putelem(idx, value)
        ## elif key is None and isinstance(value, BurpcEle): #TODO
        ##     #check if bloc big enough
        ##     #check if type match
        ##     #check if other meta match
        ##     #add lstele or dlstele+encode
        ##     #add tblval or ?rval?+encode
        ##     #TODO: option to replace an element (name != none)
        else:
            return super(self.__class__, self).__setattr__(key, value)

    def append(self, ele):
        """
        Append an element to the block

        blk.append(ele)

        Args:
            ele : BurpcEle to append
        Return:
            None
        Raises:
            TypeError  on not supported types or args
            BurpcError on any other error
        """
        self.put(None, ele)

    #TODO: add list type operators: count?, extend?, index?, insert?, pop?, remove?, reverse?, sort?... see help([]) for other __?__ operators

    def reset_arrays(self):
        """
        Clear data tables

        blk.reset_arrays()

        Args:
            None
        Return:
            None
        Raises:
            None
        """
        self.__arr = {
            "lstele"  : None,
            "dlstele" : None,
            "tblval"  : None,
            "ival"    : None,
            "rval"    : None,
            "drval"   : None,
            "charval" : None
            }

    def _derived_attr(self):
        """Return dict with derived attributs (Cached version)"""
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()

    def __derived_attr(self):
        """Return dict with derived attributs"""
        btyp  = getattr(self.__ptr[0], 'btyp')
        datyp = getattr(self.__ptr[0], 'datyp')
        try:
            datypd = _rmn.BURP_DATYP_NAMES[datyp]
        except KeyError:
            datypd = ''
        params = {
            'bkno'  : getattr(self.__ptr[0], 'bkno'),
            'nele'  : getattr(self.__ptr[0], 'nele'),
            'nval'  : getattr(self.__ptr[0], 'nval'),
            'nt'    : getattr(self.__ptr[0], 'nt'),
            'bfam'  : getattr(self.__ptr[0], 'bfam'),  #TODO: provide decoded bfam?
            'bdesc' : _C_CHAR2WCHAR_COND(getattr(self.__ptr[0], 'bdesc')),
            'btyp'  : btyp,
            'nbit'  : getattr(self.__ptr[0], 'nbit'),
            'bit0'  : getattr(self.__ptr[0], 'bit0'),
            'datyp' : datyp,
            'datypd': datypd
            }
        if btyp >= 0:
            params.update(_rmn.mrbtyp_decode(btyp))
        else:
            params.update({
                'bknat'       : -1,
                'bknat_multi' : -1,
                'bknat_kind'  : -1,
                'bknat_kindd' : -1,
                'bktyp'       : -1,
                'bktyp_alt'   : -1,
                'bktyp_kind'  : -1,
                'bktyp_kindd' : -1,
                'bkstpd'      : -1
                })
        return params

    def _getelem(self, index):
        """indexing from 0 to nele-1"""
        if index < 0 or index >= self.nele:
            raise IndexError('Index out of range [0, {}[, got: {}'
                             .format(self.nele, index))
        params = {'e_cmcid' : self.lstele[index]}
        params['e_tblval'] = self.tblval[index, :, :]
        params['store_type'] = _C_CHAR2WCHAR_COND(self.store_type)
        return BurpcEle(params)

    def _putelem(self, index, values):
        """indexing from 0 to nele-1"""
        if index is None:
            index = max(0, self.nele)
        if not isinstance(index, _integer_types):
            raise TypeError('Provided index should be of type int')
        if index < 0 or index > max(0, self.nele):
            raise IndexError('Index out of range [0, {}[, got: {}'
                             .format(self.nele, index))
        if not isinstance(values, BurpcEle):
            try:
                values = BurpcEle(values)
            except:
                raise TypeError('Provided value should be of type BurpcEle')
        store_type = _C_WCHAR2CHAR_COND(values.store_type)
        if self.nele > 0 and self.__ptr[0].store_type != store_type:
            raise TypeError('Provided value should be of type: {}, got: {}'
                            .format(self.__ptr[0].store_type,
                                    store_type))

        shape = (max(index+1, self.nele), max(values.nval, self.nval), max(values.nt, self.nt))
        if shape != (self.nele, self.nval, self.nt):
            if self.nele <= 0:
                _bp.c_brp_allocblk(self.__ptr, shape[0], shape[1], shape[2])
                self.__ptr[0].store_type = store_type
            else:
                #TODO: should restrict resizing to avoid loosing values
                _bp.c_brp_resizeblk(self.__ptr, shape[0], shape[1], shape[2])
        self.__derived = None
        self.reset_arrays()

        self.__ptr[0].lstele[index]  = values.e_cmcid
        self.__ptr[0].dlstele[index] = values.e_bufrid

        ## self.__ptr[0].tblval[index, 0:values.nval, 0:values.nt] = \
        ##     values.e_tblval[0:values.nval, 0:values.nt]

        ## i0 = _BLKIDX(values.nval, values.nt, index, 0, 0)
        ## i1 = _BLKIDX(values.nval, values.nt, index, values.nval, values.nt)
        ## self.__ptr[0].tblval[i0:i1] = \
        ##     values.e_tblval[0:values.nval, 0:values.nt]

        #TODO: recode to avoid for loops
        _BLKIDX1 = lambda shape, e, v, t: e + shape[0] * (v + shape[1] * t)
        ## for it in range(values.nt):
        ##     for iv in range(values.nval):
        for iv,it in _np.ndindex((values.nval,values.nt)):
            self.__ptr[0].tblval[_BLKIDX1(shape, index, iv, it)] = \
                values.e_tblval[iv, it]
 
        #TODO: check with charval... dims may be different


#TODO: class BurpcElePlus(BurpcEle): BurpcEle + BurpcBlk + BurpcRpt attributes
## class BurpcRptBlkEle(BurpcBlk):
##     """
##     """


class BurpcEle(_BurpcObjBase):
    """
    Python Class to hold a BURP block element's data and meta

    ele1 = BurpcEle(e_bufrid, e_rval)
    ele2 = BurpcEle(ele1)
    ele3 = BurpcEle(element_meta_dict)

    Attributes:
        e_cmcid    : Element CMC code name (lstele)
        e_bufrid   : Element BUFR code as found in BUFR table B (dlstele)
        e_bufrid_F : Type part of Element code (e.g. F=0 for obs)
        e_bufrid_X : Class part of Element code
        e_bufrid_Y : Class specific Element code part of Element code
        e_cvt      : Flag for conversion (1=need units conversion)
        e_desc     : Element description
        e_units    : Units desciption
        e_scale    : Scaling factor for element value conversion
        e_bias     : Bias for element value conversion
        e_nbits    : nb of bits for encoding value
        e_multi    : 1 means descriptor is of the "multi" or
                     repeatable type (layer, level, etc.) and
                     it can only appear in a "multi" block of data
        e_error    : 0 if bufrid found in BURP table B, -1 otherwise
        nval       : Number of values per element.
                     1st dimension of e_tblval, e_rval, e_drval
        nt         : Number of groups of NVAL values in an element.
                     2nd dimension of e_tblval, e_rval, e_drval
        shape      : (nval, nt)
        store_type : Type of data in table val, one of:
                     BRP_STORE_INTEGER, BRP_STORE_FLOAT,
                     BRP_STORE_DOUBLE, BRP_STORE_CHAR
        ptrkey     : name of table used to store values depending on store_type,
                     one of: 'e_tblval', 'e_rval', 'e_drval', 'e_charval'
        e_tblval   : table of decoded int values (BRP_STORE_INTEGER)
                     shape: (nval, nt)
        e_rval     : table of decoded values of type real/float (BRP_STORE_FLOAT)
                     shape: (nval, nt)
        e_drval    : table of decoded values of type real/float double (BRP_STORE_DOUBLE)
                     shape: (nval, nt)
        e_charval  : table of decoded values of type char (BRP_STORE_CHAR)
                     shape: (nval, nt)

    Examples:
    >>> import os, os.path
    >>> import rpnpy.burpc.all as brp
    >>> import rpnpy.librmn.all as rmn
    >>> m = brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
    >>>
    >>> # get the first report in file and print some info
    >>> rpt = bfile[0]
    >>>
    >>> # get the first block in report
    >>> blk = rpt[0]
    >>>
    >>> # get the first element in blk
    >>> ele = blk[0]
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval[0,0]))
    # 10004: PRESSURE, (units=PA), shape=[1, 1] : value=100.0
    >>>
    >>> # Copy an Element
    >>> ele1 = brp.BurpcEle(ele)
    >>> ele1.e_bufrid = 13220
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval[0,0]))
    # 10004: PRESSURE, (units=PA), shape=[1, 1] : value=100.0
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={}"
    ...       .format(ele1.e_bufrid, ele1.e_desc, ele1.e_units, ele1.nval, ele1.nt, ele1.e_rval[0,0]))
    # 13220: NATURAL LOG SFC SPEC HUMIDITY (2M), (units=LN(KG/KG)), shape=[1, 1] : value=100.0
    >>>
    >>> # Loop over all elements in block and print info for last one
    >>> for ele in blk:
    ...     pass  # Do something with the element
    >>> print("# {}: {}, (units={}), shape=[{}, {}] : value={:7.2e}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval[0,0]))
    # 13220: NATURAL LOG SFC SPEC HUMIDITY (2M), (units=LN(KG/KG)), shape=[1, 1] : value=1.00e+30
    >>>
    >>> # New Element
    >>> ele = brp.BurpcEle(10004, [10000.])
    >>> print("# {}: {}, (units={})".format(ele.e_bufrid, ele.e_desc, ele.e_units))
    # 10004: PRESSURE, (units=PA)
    >>>
    >>> # New Element from dicy
    >>> ele = brp.BurpcEle({'e_bufrid' : 10004, 'e_rval' : [10000., 10010.]})
    >>> print("# {}: {}, (units={}), shape=[{}, {}], value={}"
    ...       .format(ele.e_bufrid, ele.e_desc, ele.e_units, ele.nval, ele.nt, ele.e_rval.ravel()))
    # 10004: PRESSURE, (units=PA), shape=[2, 1], value=[ 10000.  10010.]

    See Also:
        BurpcFile
        BurpcRpt
        BurpcBlk
        rpnpy.burpc.base
        rpnpy.burpc.const
    """
    __attrlist = ('e_bufrid', 'e_cmcid', 'store_type', 'shape', 'ptrkey',
                  'e_ival', 'e_rval', 'e_drval', 'e_charval', 'e_tblval', )
    __attrlist2 = ('e_error', 'e_cmcid', 'e_bufrid', 'e_bufrid_F',
                   'e_bufrid_X', 'e_bufrid_Y', 'e_cvt', 'e_desc',
                   'e_units', 'e_scale', 'e_bias', 'e_nbits', 'e_multi',
                   'nval', 'nt', 'shape')
    __PTRKEY2NUMPY = {
        'e_tblval'  : _np.int32,
        'e_ival'    : _np.int32,
        'e_rval'    : _np.float32,
        'e_drval'   : _np.float64,
        'e_charval' : _np.uint8
        }
    __PTRKEY2STORE_TYPE = {
        'e_tblval'  : _bc.BRP_STORE_INTEGER,
        'e_ival'    : _bc.BRP_STORE_INTEGER,
        'e_rval'    : _bc.BRP_STORE_FLOAT,
        'e_drval'   : _bc.BRP_STORE_DOUBLE,
        'e_charval' : _bc.BRP_STORE_CHAR
        }
    __PTRKEY2STORE_TYPE_INV = {
        _bc.BRP_STORE_INTEGER : 'e_ival',
        _bc.BRP_STORE_FLOAT   : 'e_rval',
        _bc.BRP_STORE_DOUBLE  : 'e_drval',
        _bc.BRP_STORE_CHAR    : 'e_charval'
        }
    def __init__(self, bufrid, tblval=None): #TODO:, shape=None):
        if isinstance(bufrid, _integer_types):
            bufrid = {
                'e_bufrid' : bufrid,
                'e_tblval' : tblval
                }
        elif not isinstance(bufrid, (dict, self.__class__)):
            raise TypeError('bufrid should be of type int, BurpEle or dict')
        self.__derived = None
        self.__ptr     = dict([(k, None) for k in self.__attrlist])
        self.update(bufrid) #TODO: update should check type
        ptrkey = _C_CHAR2WCHAR_COND(self.__ptr['ptrkey'])
        if (self.__ptr['e_bufrid'] is None or
            ptrkey is None or
            self.__ptr[ptrkey] is None):
            raise BurpcError('{} {}: incomplete initialization'
                             .format(self.__class__.__name__, repr([self.__ptr['e_bufrid'], ptrkey, self.e_tblval, self.e_rval, self.e_drval, self.e_charval])))

    def __setattr__(self, name, value): #TODO: move to super class
        return self.put(name, value)

    ## def next(self):
    ##     raise Error #TODO: loop through nval?

    def get(self, key): #TODO: if int (or slice any indexing, refer to tblval)
        """
        Get Burpc Element meta or data

        value = ele.get(attr_name)

        Args:
             key : Attribute name or Search criterions
                   if str, get the attribute value
                   if int, get the ith ([0, nval[) val in the element
        Return:
             Attribute value
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "ele.attr_name"
            instead of "ele.get('attr_name')"
        """
        key = _C_CHAR2WCHAR_COND(key)
        if key in self.__class__.__attrlist:
            return _C_CHAR2WCHAR_COND(self.__ptr[key])
        elif key in self.__class__.__attrlist2:
            return _C_CHAR2WCHAR_COND(self._derived_attr()[key])
        ## elif isinstance(key, _integer_types): #TODO:
        raise KeyError("{} object has no such key: {}"
                       .format(self.__class__.__name__, repr(key)))

    def reshape(self, shape=None):
        """
        Gives a new shape to the data array without changing its data.

        ele.reshape((nval, nt))

        Args:
             shape : (nval, nt)
                     where:
                     nval : Number of values per element.
                            1st dimension of e_tblval, e_rval, e_drval
                     nt   : Number of groups of NVAL values in an element.
                            2nd dimension of e_tblval, e_rval, e_drval
        Return:
             None
        Raises:
            TypeError  on not supported types or args
            BurpcError on any other error
        """
        self.__derived = None  # Reset nval, nt, shape
        if shape is None:
            #TODO: shouldn't we nullify the actual table then
            self.__ptr['shape'] = None
            return
        if isinstance(shape, _integer_types):
            shape = (shape, )
        if not isinstance(shape, (list, tuple)):
            raise TypeError('Provided shape must be a list')
        if len(shape) == 1:
            shape = (shape[0], 1)
        elif len(shape) > 2:
            raise BurpcError('{}: Array shape must be 2d: {}'
                             .format(self.__class__.__name__,
                                     repr(shape)))
        ptrkey = _C_CHAR2WCHAR_COND(self.__ptr['ptrkey'])
        if ptrkey is not None:
            if self.__ptr[ptrkey].size != shape[0] * shape[1]:
                raise BurpcError('{}: array size and provided shape does not match: {}'
                                 .format(self.__class__.__name__,
                                         repr(self.__ptr[ptrkey].shape)))
            self.__ptr[ptrkey] = \
                _np.reshape(self.__ptr[ptrkey], shape, order='F')
        if ptrkey != 'e_tblval' and \
            self.__ptr['e_tblval'] is not None:
            self.__ptr['e_tblval'] = \
                _np.reshape(self.__ptr['e_tblval'],
                            shape, order='F')
        self.__ptr['shape'] = shape

    def put(self, key, value):
        """
        Set Burpc Element meta or data

        ele.put(key, value)

        Args:
            key   : Attribute name
                    if str, set the attribute value
                    if int, set the ith ([0, nval[) val in the element
            value : Value to set
        Return:
            None
        Raises:
            KeyError   on not not found key
            TypeError  on not supported types or args
            BurpcError on any other error

        Notes:
            For attributes value, the prefered way is to
            use "ele.attr_name = value"
            instead of "ele.put('attr_name', value)"
        """
        key = _C_CHAR2WCHAR_COND(key)
        if key == 'ptrkey':
            raise KeyError('{}: Cannot set: {}'
                             .format(self.__class__.__name__,
                                     repr(key)))
        elif key == 'e_bufrid':
            self.__derived = None
            self.__ptr[key] = value
            self.__ptr['e_cmcid'] = _rmn.mrbcol(value)
        elif key == 'e_cmcid':
            self.__derived = None
            self.__ptr[key] = value
            self.__ptr['e_bufrid'] = _rmn.mrbdcl(value)
        elif key == 'store_type':
            bvalue = _C_WCHAR2CHAR_COND(value)
            value = _C_CHAR2WCHAR_COND(value)
            if value in _bc.BRP_STORE_TYPE2NUMPY.keys():
                if self.__ptr[key] is None:
                    self.__ptr[key] = bvalue
                elif _C_CHAR2WCHAR_COND(self.__ptr[key]) != value:
                    raise BurpcError('{}: Cannot change: {}'
                                     .format(self.__class__.__name__,
                                             repr(key)))
            elif value is not None:
                raise ValueError('Store type ({}) can only be one of: {}'
                                 .format(repr(value),
                                         repr(_bc.BRP_STORE_TYPE2NUMPY.keys())))
        elif key == 'shape':
            self.reshape(value)
        elif key in ('e_tblval', 'e_ival', 'e_rval', 'e_drval', 'e_charval'):
            if value is None:
                return
            self.__derived = None
            #TODO: when updating from another BuprcEle, both e_tablval and e_?val are passed... avoid double definition
            if key == 'e_tblval':
                self._put_tblval(value)
            else:
                #TODO: allow e_val: automatic type selection
                self._put_irdcval(key, value)
        elif key in self.__class__.__attrlist:
            self.__derived = None
            #TODO: check type
            self.__ptr[key] = _C_WCHAR2CHAR_COND(value)
            ## return setattr(self.__ptr, key, value) #TODO: use proto fn?
        else:
            return super(self.__class__, self).__setattr__(key, value)

    ## def delete(self, key):
    ##     raise BurpcError('{}: Cannot delete: {}'
    ##                      .format(self.__class__.__name__, repr(key)))

    def _tblval2eval(self):
        #TODO: decode to tblval... may want to strictly use burpc fn (create fake BurpcBlk, put id+rval, brp.c_brp_convertblk(br, brp.BRP_MKSA_to_BUFR), extract tblval
        key = 'e_tblval'
        dtype = self.__PTRKEY2NUMPY[key]
        try:
            ptrkeytype = _C_CHAR2WCHAR_COND(self.__ptr['store_type'])
            ptrkey = self.__PTRKEY2STORE_TYPE_INV[ptrkeytype]
            self.__ptr['ptrkey'] = _C_WCHAR2CHAR_COND(ptrkey)
        except KeyError:
            ptrkey = None
        if ptrkey:
            e_cmcid = _np.asfortranarray(self.__ptr['e_cmcid'], dtype=_np.int32)
            shape = [1] + list(self.__ptr[key].shape)
            e_tblval = _np.reshape(_np.asfortranarray(self.__ptr[key],
                                                      dtype=dtype),
                                   shape, order='F').copy(order='F')
            val3d = _rmn.mrbcvt_decode(e_cmcid, e_tblval)
            dtype = self.__PTRKEY2NUMPY[ptrkey]
            self.__ptr[ptrkey] = _np.reshape(_np.asfortranarray(val3d,
                                                                dtype=dtype),
                                             shape[1:3], order='F')

    def _eval2tblval(self, key):
        #TODO: encode to tblval... may want to strictly use burpc fn (create fake BurpcBlk, put id+rval, brp.c_brp_convertblk(br, brp.BRP_MKSA_to_BUFR), extract tblval
        key = _C_CHAR2WCHAR_COND(key)
        dtype = _np.float32  # Always float32, expected by mrbcvt_encode
        ptrkey = 'e_tblval'
        e_cmcid = _np.asfortranarray(self.__ptr['e_cmcid'], dtype=_np.int32)
        shape = [1] + list(self.__ptr[key].shape)
        val3d = _np.reshape(_np.asfortranarray(self.__ptr[key], dtype=dtype),
                            shape, order='F').copy(order='F')
        self.__ptr[ptrkey] = _np.reshape(_rmn.mrbcvt_encode(e_cmcid, val3d),
                                         shape[1:3], order='F')

    #TODO: when setting e_tblval values, recompute e_?val and viceversa

    def _put_tblval(self, value):
        key = 'e_tblval'
        if self.__ptr['ptrkey'] is None:
            self.__ptr['ptrkey'] = key
        if self.__ptr['store_type'] is None:
            self.__ptr['store_type'] = \
                _C_WCHAR2CHAR_COND(self.__PTRKEY2STORE_TYPE[key])
        dtype = self.__PTRKEY2NUMPY[key]
        if isinstance(value, _np.ndarray):
            value = value.copy()
        self.__ptr[key] = _np.asfortranarray(value, dtype=dtype)
        self.reshape(self.__ptr[key].shape)
        if (self.__ptr['e_ival'] == self.__ptr['e_rval'] ==
            self.__ptr['e_drval'] == self.__ptr['e_charval'] == None):
            self._tblval2eval()

    def _put_irdcval(self, key, value):
        key = _C_CHAR2WCHAR_COND(key)
        ptrkey = _C_CHAR2WCHAR_COND(self.__ptr['ptrkey'])
        if not (ptrkey is None or ptrkey == key):
            raise BurpcError('{}: Cannot change store type'
                            .format(self.__class__.__name__))
        self.__ptr['ptrkey'] = _C_WCHAR2CHAR_COND(key)
        self.__ptr['store_type'] = \
            _C_WCHAR2CHAR_COND(self.__PTRKEY2STORE_TYPE[key])
        dtype = self.__PTRKEY2NUMPY[key]
        if isinstance(value, _np.ndarray):
            value = value.copy()
        self.__ptr[key] = _np.asfortranarray(value, dtype=dtype)
        self.reshape(self.__ptr[key].shape)
        self._eval2tblval(key)

    def _derived_attr(self):
        """Return dict with derived attributs (Cached version)"""
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()

    def __derived_attr(self):
        """Return dict with derived attributs"""
        params = _rmn.mrbcvt_dict_bufr(self.__ptr['e_bufrid'], False)
        nval, nt = 0, 0
        ptrkey = _C_CHAR2WCHAR_COND(self.__ptr['ptrkey'])
        if ptrkey is not None:
            nval = self.__ptr[ptrkey].shape[0]
            try:
                nt = self.__ptr[ptrkey].shape[1]
            except IndexError:
                nt = 1
        params.update({
            'nval'  : nval,
            'nt'    : nt,
            'shape' : (nval, nt)
            })
        return params


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
