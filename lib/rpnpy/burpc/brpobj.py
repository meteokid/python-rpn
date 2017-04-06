#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc.burpc contains the wrapper classes to main burp_c C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] documentation for more details.

See Also:
    rpnpy.burpc.base
    rpnpy.burpc.proto
    rpnpy.burpc.const
"""
import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc
from rpnpy.burpc import proto as _bp
from rpnpy.burpc import const as _bc
from rpnpy.burpc import BurpcError
import rpnpy.librmn.all as _rmn

from rpnpy import integer_types as _integer_types

class _BurpcObjBase(object):
    """
    """
    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.todict())+')'

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        return self.next()

    def _getattr0(self, name):
        return getattr(self,'_'+self.__class__.__name__+name)

    def __getattr__(self, name):
        try:
            return self.get(name)
        except KeyError as e:
            raise AttributeError(e)
            ## return super(self.__class__, self).__getattr__(name)
            ## return super(_BurpcObjBase, self).__getattr__(name)

    def __getitem__(self, name):
        return self.get(name)

    def __delitem__(self, name):
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
        return dict([(k, getattr(self,k)) for k in
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
    >>> rpt = bfile[0]  #get the first report in file
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
    ...     print('# stnid=' + repr(rpt.stnid))
    ...     break
    # stnid='71915    '
    >>>
    >>> # Close the file
    >>> del bfile
    >>>
    >>> # Open file in read only mode
    >>> bfile = brp.BurpcFile(filename)
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
        self.filename = filename
        self.filemode = filemode
        self.funit    = funit
        if isinstance(filename, dict):
            if 'filename' in filename.keys():
                self.filename = filename['filename']
            if 'filemode' in filename.keys():
                self.filemode = filename['filemode']
            if 'funit' in filename.keys():
                self.funit = filename['funit']
        self.__iteridx   = BurpcRpt() #0
        self.__handles   = []
        self.__rpt       = BurpcRpt()
        fstmode, brpmode, brpcmode = _bp.brp_filemode(self.filemode)
        self.funit       = _rmn.get_funit(self.filename, fstmode, self.funit)
        self.nrep        = _bp.c_brp_open(self.funit, self.filename, brpcmode)
        if self.nrep < 0:
            raise BurpcError('Problem opening with mode {} the file: {}'
                             .format(repr(brpcmode), repr(self.filename)))
        self.__ptr = self.funit

    def __del__(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._close()

    def __len__(self):
        return max(0, self.nrep)

    def __iter__(self):
        self.__iteridx = BurpcRpt() #0
        return self

    def next(self):  # Python 2
        """
        Get the next item in the iterator, Internal function for python 2 iter
        """
        if _bp.c_brp_findrpt(self.funit, self.__iteridx.getptr()) >= 0:
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

    def get(self, search=None, rpt=None):
        """
        Find a report and get its meta + data

        Args:
            search : Search criterions
                     if int, return the ith ([0, nrep[) report in file
                     if dict or BurpcRpt, search report matching given params
            rpt    : (optional) BurpcRpt use to put the result to recycle memory
        Return:
            BurpcRpt if a report match the search
            None otherwise
        Raises:
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        rpt = rpt if isinstance(rpt, BurpcRpt) else BurpcRpt(rpt)
        if search is None or isinstance(search, (BurpcRpt, dict)):
            search = search if isinstance(search, BurpcRpt) else BurpcRpt(search)
            if _bp.c_brp_findrpt(self.funit, search.getptr()) >= 0:
                if _bp.c_brp_getrpt(self.funit, search.handle,
                                    rpt.getptr()) >= 0:
                    return rpt
            return None
        elif isinstance(search, _integer_types):
            if search < 0 or search >= self.nrep:
                raise IndexError('Index out of range: [0:{}['.format(self.nrep))
            if search >= len(self.__handles):
                i0 = len(self.__handles)
                search1 = BurpcRpt()
                if i0 > 0:
                    search1.handle = self.__handles[-1]
                for i in range(i0, search+1):
                    if _bp.c_brp_findrpt(self.funit, search1.getptr()) >= 0:
                        self.__handles.append(search1.handle)
                    else:
                        break
            if _bp.c_brp_getrpt(self.funit, self.__handles[search],
                                rpt.getptr()) >= 0:
                return rpt
        else:
            raise TypeError("For Name: {}, Not Supported Type: {}".format(repr(search), str(type(search))))

    def put(self, where, rpt):
        """
        Write a report to the burp file

        Args:
            where : location to write report to
                    if None or BRP_END_BURP_FILE, append to the file
                    if int, handle of report to replace in file
            rpt   : BurpcRpt to write
        Return:
            None
        Raises:
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        if not isinstance(rpt, BurpcRpt):
            raise TypeError("rpt should be of type BurpcRpt, got: {}, ".format(str(type(rpt))))
        append = where is None
        if append:
            where = _bc.BRP_END_BURP_FILE
        ## elif isinstance(where, (BurpcRpt, dict)): #TODO:
        ## elif isinstance(where, _integer_types): #TODO: same indexing as get, how to specify a handle?
        else:
            raise TypeError("For where: {}, Not Supported Type: {}".format(repr(where), str(type(where))))

        self.__handles = [] #TODO: is it the best place to invalidate the cache?
        #TODO: conditional _bb.brp_updrpthdr
        ## _bb.brp_updrpthdr(self.funit, rpt)
        prpt = rpt.getptr() if isinstance(rpt, BurpcRpt) else rpt
        if _bp.c_brp_writerpt(self.funit, prpt, where) < 0:
            raise BurpcError('Problem in brp_writerpt')
        if append: #TODO: only if writeok
            self.nrep += 1

    def append(self, rpt):
        """
        Append a report to the burp file

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

    TODO: constructor examples

    Attributes:
        TODO

    Examples:
    >>> import os, os.path
    >>> import rpnpy.burpc.all as brp
    >>> import rpnpy.librmn.all as rmn
    >>> m = brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    >>> ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
    >>> filename = os.path.join(ATM_MODEL_DFILES,'bcmk_burp','2007021900.brp')
    >>>

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
                  "drnd", "date", "oars", "runn", "nblk", "lngr")
    __attrlist2 = ('time', 'timehh', 'timemm', 'flgsl', 'flgsd',
                   'idtyp', 'idtypd', 'ilat', 'lat', 'ilon', 'lon',
                   'idx', 'rdx', 'idy', 'rdy', 'ielev', 'relev',
                   'dateyy', 'datemm', 'datedd',
                   'sup', 'nsup', 'xaux', 'nxaux')
    __attrlist2names = {
        'rdx'   : 'dx',
        'rdy'   : 'dy',
        'relev' : 'elev'
        }

    def __init__(self, rpt=None):
        self.__bkno = 0
        self.__blk  = BurpcBlk()
        self.__derived = None
        self.__attrlist2names_keys = self.__attrlist2names.keys()
        if rpt is None:
            ## print 'NEW:',self.__class__.__name__
            self.__ptr = _bp.c_brp_newrpt()
        elif isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
            ## print 'NEW:',self.__class__.__name__,'ptr'
            self.__ptr = rpt
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

    def next(self): # Python 2:
        """
        Get the next item in the iterator, Internal function for python 2 iter
        """
        if self.__bkno >= self.nblk:
            self.__bkno = 0
            raise StopIteration
        self.__blk = self.get(self.__bkno, self.__blk)
        if _bp.c_brp_getblk(self.__bkno+1, self.__blk.getptr(),
                            self.getptr()) < 0:
            self.__bkno = 0
            raise BurpcError('Problem in c_brp_getblk')
        self.__bkno += 1
        return self.__blk

    def get(self, name=None, blk=None):
        """
        Find a block and get its meta + data

        blk.get(attr_name)
        blk.get(item_number)
        blk.get(blk)
        blk.get(blkdict)

        Args:
            name   : Attribute name or Search criterions
                     if str, return the attribute value
                     if int, return the ith ([0, nrep[) report in file
                     if dict or BurpcBlk, search block matching given params
            blk    : (optional) BurpcBlk use to put the result to recycle memory
        Return:
            Attribute value or
            BurpcBlk if a report match the search
            None otherwise
        Raises:
            TypeError  on not supported types or args
            IndexError on out of range index
            BurpcError on any other error
        """
        if name in self.__class__.__attrlist:
            return getattr(self.__ptr[0], name)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            try:
                name2 = self.__attrlist2names[name]
            except KeyError:
                name2 = name
            return self.derived_attr()[name2]
        elif isinstance(name, _integer_types):
            name += 1
            if name < 1 or name > self.nblk:
                raise IndexError('Index out of range: [0:{}['.format(self.nblk))
            blk = blk if isinstance(blk, BurpcBlk) else BurpcBlk(blk)
            if _bp.c_brp_getblk(name, blk.getptr(), self.getptr()) < 0:
                raise BurpcError('Problem in c_brp_getblk')
            return blk
        elif name is None or isinstance(name, (BurpcBlk, dict)):
            search = name if isinstance(name, BurpcBlk) else BurpcBlk(name)
            if _bp.c_brp_findblk(search.getptr(), self.getptr()) >= 0:
                blk = blk if isinstance(blk, BurpcBlk) else BurpcBlk(blk)
                if _bp.c_brp_getblk(search.bkno, blk.getptr(),
                                    self.getptr()) >= 0:
                    return blk
            return None
        raise KeyError("{} object has no such key: {}"
                       .format(self.__class__.__name__, repr(name)))

    def __setattr__(self, name, value): #TODO: move to super class
        return self.put(name, value)

    def put(self, name, value):
        """
        """
        if name == 'stnid':
            self.__derived = None
            _bp.c_brp_setstnid(self.__ptr, value)
        elif name in self.__class__.__attrlist:
            self.__derived = None
            return setattr(self.__ptr[0], name, value)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            #TODO: encode other items on the fly
            raise AttributeError(self.__class__.__name__+" object cannot set derived attribute '"+name+"'")
        ## elif isinstance(name, _integer_types): #TODO:
        ## elif name is None or isinstance(name, (BurpcBlk, dict)): #TODO:
        else:
            return super(self.__class__, self).__setattr__(name, value)
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")

    def append(self, value):
        """
        """
        self.put(None, value)

    def derived_attr(self): #TODO: remove/hide?
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()

    def __derived_attr(self):
        """ """
        itime = getattr(self.__ptr[0], 'temps')
        iflgs = getattr(self.__ptr[0], 'flgs')
        flgs_dict = _rmn.flags_decode(iflgs)
        idtyp = getattr(self.__ptr[0], 'idtype')
        ilat  = getattr(self.__ptr[0], 'lati')
        ilon  = getattr(self.__ptr[0], 'longi')
        idx   = getattr(self.__ptr[0], 'dx')
        idy   = getattr(self.__ptr[0], 'dy')
        ialt  = getattr(self.__ptr[0], 'elev')
        idate = getattr(self.__ptr[0], 'date')
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
            'stnid' : getattr(self.__ptr[0], 'stnid'),
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
            'nblk'  : getattr(self.__ptr[0], 'nblk'),
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
            }


#TODO: class BurpcBlkPlus(BurpcBlk): BurpcBlk + BurpcRpt attributes
        
class BurpcBlk(_BurpcObjBase):
    """
    Python Class equivalent of the burp_c's BURP_BLK C structure to hold
    the BURP block data

    TODO: constructor examples

    Attributes:
        TODO:
    """
    __attrlist = ("bkno", "nele", "nval", "nt", "bfam", "bdesc", "btyp",
                  "bknat", "bktyp", "bkstp", "nbit", "bit0", "datyp",
                  "store_type",
                  ## "lstele", "dlstele", "tblval", "rval","drval", "charval",
                  "max_nval", "max_nele", "max_nt", "max_len")
    __attrlist_np_1d = ("lstele", "dlstele")
    __attrlist_np_3d = ("tblval", "rval", "drval", "charval")
    __attrlist2 = ('bkno', 'nele', 'nval', 'nt', 'bfam', 'bdesc', 'btyp',
                   'bknat', 'bknat_multi', 'bknat_kind', 'bknat_kindd',
                   'bktyp', 'bktyp_alt', 'bktyp_kind', 'bktyp_kindd',
                   'bkstp', 'bkstpd', 'nbit', 'bit0', 'datyp', 'datypd')

    def __init__(self, blk=None):
        self.__eleno = 0
        self.__derived = None
        to_update = False
        if blk is None:
            ## print 'NEW:',self.__class__.__name__
            self.__ptr = _bp.c_brp_newblk()
        elif isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
            ## print 'NEW:',self.__class__.__name__,'ptr'
            self.__ptr = blk
        else:
            ## print 'NEW:',self.__class__.__name__,'update'
            self.__ptr = _bp.c_brp_newblk()
            to_update = True
        if to_update:
            self.update(blk)
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

    def next(self): # Python 2
        """
        Get the next item in the iterator, Internal function for python 2 iter
        """
        if self.__eleno >= self.nele:
            self.__eleno = 0
            raise StopIteration
        ele = self._getelem(self.__eleno)
        self.__eleno += 1
        return ele

    def get(self, name):
        ## print 'getattr:', name
        if name in self.__class__.__attrlist_np_1d:
            if self.__arr[name] is None:
                v = getattr(self.__ptr[0], name)
                self.__arr[name] = _np.ctypeslib.as_array(v, (self.nele,))
            return self.__arr[name]
        elif name in self.__class__.__attrlist_np_3d:
            if self.__arr[name] is None:
                v = getattr(self.__ptr[0], name)
                self.__arr[name] = _np.ctypeslib.as_array(v,
                                        (self.nt, self.nval, self.nele)).T
            return self.__arr[name]
        elif name in self.__class__.__attrlist:
            return getattr(self.__ptr[0], name)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            if not self.__derived:
                self.__derived = self.derived_attr()
            return self.__derived[name]
        elif isinstance(name, _integer_types):
            return self._getelem(name)
        else:
            raise KeyError("{} object has no such key: {}"
                           .format(self.__class__.__name__, repr(name)))

    def __setattr__(self, name, value): #TODO: move to super class
        return self.put(name, value)

    def put(self, name, value):
        ## print 'setattr:', name
        if name in self.__class__.__attrlist:
            self.__derived = None
            return setattr(self.__ptr[0], name, value) #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            #TODO: encode other items on the fly
            raise AttributeError(self.__class__.__name__+" object cannot set derived attribute '"+name+"'")
        ## elif isinstance(name, _integer_types): #TODO:
        ## elif name is None or isinstance(name, (BurpcEle, dict)): #TODO:
        else:
            return super(self.__class__, self).__setattr__(name, value)

    def append(self, value):
        self.put(None, value)
    #TODO: add list type operators: count?, extend?, index?, insert?, pop?, remove?, reverse?, sort?... see help([]) for other __?__ operators

    def reset_arrays(self):
        ## print "reset array"
        self.__arr = {
            "lstele"  : None,
            "dlstele" : None,
            "tblval"  : None,
            "rval"    : None,
            "drval"   : None,
            "charval" : None
            }

    def derived_attr(self): #TODO: remove/hide?
        """ """
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()

    def __derived_attr(self):
        """ """
        btyp  = getattr(self.__ptr[0], 'btyp')
        datyp = getattr(self.__ptr[0], 'datyp')
        try:
            datypd = _rmn.BURP_DATYP_NAMES[datyp]
        except:
            datypd = ''
        params = {
            'bkno'  : getattr(self.__ptr[0], 'bkno'),
            'nele'  : getattr(self.__ptr[0], 'nele'),
            'nval'  : getattr(self.__ptr[0], 'nval'),
            'nt'    : getattr(self.__ptr[0], 'nt'),
            'bfam'  : getattr(self.__ptr[0], 'bfam'),  #TODO: provide decoded bfam?
            'bdesc' : getattr(self.__ptr[0], 'bdesc'),
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
            raise IndexError
        params = {'e_cmcid' : self.lstele[index]}
        hasValues = False
        try:
            params['e_rval'] = self.rval[index,:,:]
            hasValues = True
        except:
            pass
        try:
            params['e_drval'] = self.drval[index,:,:]
            hasValues = True
        except:
            pass
        try:
            params['e_charval'] = self.charval[index,:]
            hasValues = True
        except:
            pass
        if not hasValues: #TODO: also provide tblval?
            params['e_tblval'] = self.tblval[index,:,:]
        return BurpcEle(params)

    def putelem(self, index, values): #TODO: merge into put()
        """indexing from 0 to nele-1"""
        if not(index is None or instance(index, _integer_types)):
            raise TypeError('Provided index should be of type int')
        if (instance(index, _integer_types) and 
            index < 0 or index >= self.nele):
            raise IndexError
        else:
            pass #TODO increase size, adjust nele?
        if not instance(values, BurpcEle):
            try:
                values = BurpcEle(values)
            except:
                raise TypeError('Provided value should be of type BurpcEle')
        raise BurpcError('Not yet implemented')
        #TODO: check if type match
        #TODO: check if dims match
        #TODO: brp_resizeblk to add elem if need be
        #TODO: copy e_cmcid into dlstele + brp_encodeblk
        #TODO: copy e_rval into rval + brp_convertblk(br, brp.BRP_MKSA_to_BUFR) (or copy tblval and convert inverse)?


#TODO: class BurpcElePlus(BurpcEle): BurpcEle + BurpcBlk + BurpcRpt attributes

class BurpcEle(_BurpcObjBase):
    """
    Python Class to hold a BURP block element's data and meta

    TODO: constructor examples

    Attributes:
        TODO:
    """
    __attrlist = ('e_bufrid', 'e_cmcid', 'store_type', 'shape', 'ptrkey',
                  'e_tblval','e_rval', 'e_drval', 'e_charval')
    __attrlist2 = ('e_error', 'e_cmcid', 'e_bufrid', 'e_bufrid_F',
                   'e_bufrid_X', 'e_bufrid_Y', 'e_cvt', 'e_desc',
                   'e_units', 'e_scale', 'e_bias', 'e_nbits', 'e_multi',
                   'nval', 'nt', 'shape')
    __PTRKEY2NUMPY = {
        'e_tblval'  : _np.int32,
        'e_rval'    : _np.float32,
        'e_drval'   : _np.float64,
        'e_charval' : _np.uint8
        }
    __PTRKEY2STORE_TYPE = {
        'e_tblval'  : _bc.BRP_STORE_INTEGER,
        'e_rval'    : _bc.BRP_STORE_FLOAT,
        'e_drval'   : _bc.BRP_STORE_DOUBLE,
        'e_charval' : _bc.BRP_STORE_CHAR
        }

    def __init__(self, bufrid, tblval=None, shape=None): #TODO: use shape
        if isinstance(bufrid, _integer_types):
            bufrid = {
                'e_bufrid' : bufrid,
                'e_tblval' : tblval
                }
        elif not isinstance(bufrid, (dict, self.__class__)):
            raise TypeError('bufrid should be of type int')
        self.__derived = None
        self.__ptr     = dict([(k, None) for k in self.__attrlist])
        self.update(bufrid) #TODO: update should check type
        if (self.__ptr['e_bufrid'] is None or
            self.__ptr['ptrkey'] is None or
            self.__ptr[self.__ptr['ptrkey']] is None):
            raise BurpcError('{}: incomplete initialization'
                             .format(self.__class__.__name__))

    def __setattr__(self, name, value): #TODO: move to super class
        return self.put(name, value)

    ## def next(self):
    ##     raise Error #TODO

    def get(self, name): #TODO: if int (or slice any indexing, refer to tblval)
        """
        Get Burpc Element meta or data
        """
        ## if name == 'e_tblval' : #TODO: special case if ptrkey!=e_tblval
        ## elif name == 'e_rval' :
        ## elif name == 'e_drval' :
        ## elif name == 'e_charval':
        #TODO: allow e_val: automatic type selection
        if name in self.__class__.__attrlist:
            return self.__ptr[name]
        elif name in self.__class__.__attrlist2:
            return self.derived_attr()[name]
        ## elif isinstance(name, _integer_types):
        raise KeyError("{} object has no such key: {}"
                       .format(self.__class__.__name__, repr(name)))

    def reshape(self, shape=None):
        if shape is None:
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
                                     repr(self.__ptr[name].shape)))
        if self.__ptr['ptrkey'] is not None:
            if self.__ptr[self.__ptr['ptrkey']].size != shape[0] * shape[1]:
                raise BurpcError('{}: array size and provided shape does not match: {}'
                                 .format(self.__class__.__name__,
                                         repr(self.__ptr[self.__ptr['ptrkey']].shape)))
            self.__ptr[self.__ptr['ptrkey']] = _np.reshape(self.__ptr[self.__ptr['ptrkey']],
                                                   shape, order='F')
        self.__ptr['shape'] = shape

    def put(self, name, value):
        if name == 'ptrkey':
            raise KeyError('{}: Cannot set: {}'
                             .format(self.__class__.__name__,
                                     repr(name)))
        elif name == 'e_bufrid':
            self.__derived = None
            self.__ptr[name] = value
            self.__ptr['e_cmcid'] = _rmn.mrbcol(value)
        elif name == 'e_cmcid':
            self.__derived = None
            self.__ptr[name] = value
            self.__ptr['e_bufrid'] = _rmn.mrbdcl(value)
        elif name == 'store_type':
            if value is None:
                return
            if value in _bc.BRP_STORE_TYPE2NUMPY.keys():
                if (self.__ptr[name] is None or
                    ## self.__ptr[name] == _bc.BRP_STORE_INTEGER or
                    self.__ptr[name] == value):
                    self.__ptr[name] = value
                else:
                    raise BurpcError('{}: Cannot change: {}'
                                     .format(self.__class__.__name__,
                                             repr(name)))
            else:
                raise ValueError('Store type ({}) can only be one of: {}'
                                 .format(repr(value),
                                         repr(_bc.BRP_STORE_TYPE2NUMPY.keys())))
        elif name == 'shape':
            if value is None:
                self.__ptr['shape'] = None
            else:
                self.reshape(value)
        elif name in ('e_tblval', 'e_rval', 'e_drval', 'e_charval'):
            #TODO: use e_ival for int instead of e_tblval (alias)
            #TODO: allow e_val: automatic type selection
            self.__derived = None
            if value is None:
                return
            if not (self.__ptr['ptrkey'] is None or self.__ptr['ptrkey'] == name):
                raise BurpcError('{}: Cannot change store type'
                                 .format(self.__class__.__name__))
            self.__ptr['ptrkey'] = name
            if name != 'e_tblval':
                self.__ptr['store_type'] = self.__PTRKEY2STORE_TYPE[name]
            dtype = self.__PTRKEY2NUMPY[name]
            self.__ptr[name] = _np.array(value, order='F', dtype=dtype)
            if len(self.__ptr[name].shape) == 1:
                self.__ptr[name] = _np.reshape(self.__ptr[name],
                            (self.__ptr[name].shape[0], 1), order='F')
            elif len(self.__ptr[name].shape) > 2:
                raise BurpcError('{}: Array shape must be 2d: {}'
                                 .format(self.__class__.__name__,
                                         repr(self.__ptr[name].shape)))
            if self.__ptr['shape'] != self.__ptr[name].shape:
                self.reshape(self.__ptr['shape'])
            self.__ptr['shape'] = self.__ptr[name].shape
        elif name in self.__class__.__attrlist:
            self.__derived = None
            #TODO: check type
            self.__ptr[name] = value
            ## return setattr(self.__ptr, name, value) #TODO: use proto fn?
        else:
            return super(self.__class__, self).__setattr__(name, value)
        ## else:
        ##     raise KeyError #TODO

    ## def delete(self, name):
    ##     raise BurpcError('{}: Cannot delete: {}'
    ##                      .format(self.__class__.__name__, repr(name)))

    def derived_attr(self): #TODO: remove/hide?
        """ """
        if not self.__derived:
            self.__derived = self.__derived_attr()
        return self.__derived.copy()

    def __derived_attr(self):
        """ """
        #TODO: rval, drval, charval...
        params = _rmn.mrbcvt_dict_bufr(self.__ptr['e_bufrid'], False)
        nval, nt = 0, 0
        if self.__ptr['ptrkey'] is not None:
            nval = self.__ptr[self.__ptr['ptrkey']].shape[0]
            try:
                nt = self.__ptr[self.__ptr['ptrkey']].shape[1]
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
