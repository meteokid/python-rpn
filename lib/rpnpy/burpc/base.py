#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module burpc.base contains python wrapper to main burp_c C functions

Notes:
    The functions described below are a very close ''port'' from the original
    [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
    You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] documentation for more details.

See Also:
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

## _C_MKSTR = _ct.create_string_buffer

_filemodelist = {
    'r' : (_rmn.FST_RO,     _rmn.BURP_MODE_READ),
    'w' : (_rmn.FST_RW,     _rmn.BURP_MODE_CREATE),
    'a' : (_rmn.FST_RW_OLD, _rmn.BURP_MODE_APPEND)
    }
_filemodelist_inv = dict([
    (v[1], k) for k, v in _filemodelist.items()
    ])


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

    def update(self, rpt):
        """ """
        if not isinstance(rpt, (dict, self.__class__)):
            raise TypeError("Type not supported for rpt: "+str(type(rpt)))
        for k in self._getattr0('__attrlist'):
            try:
                self.__setitem__(k, rpt[k])
            except:
                pass

    def getptr(self):
        """ """
        return  self._getattr0('__ptr')

    def todict(self):
        """ """
        return dict([(k, getattr(self,k)) for k in
                     self._getattr0('__attrlist')])

    ## def get(self, name):  #to be defined by child class
    ## def delete(self, name):  #to be defined by child class
    ## def put(self, name, value):  #to be defined by child class
    ## def next(self):  #to be defined by child class

    #TODO: add list/dict type operators: count?, extend?, index?, insert?, pop?, remove?, reverse?, sort?... see help([]) help({}) for other __?__ operators

class BurpcFile(_BurpcObjBase):
    """
    Python Class to refer to, interact with a BURP file using the burp_c lib

    TODO: constructor examples

    Attributes:
        filename :
        filemode :
        funit    :
        TODO
    """
    __attrlist = ("filename", "filemode", "funit")

    def __init__(self, filename, filemode='r', funit=None):
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
        self.__search  = BurpcRpt()
        self.__handles = []
        self.__rpt     = BurpcRpt()
        self.funit, self.nrep = brp_open(self.filename, self.filemode, funit=self.funit, getnbr=True)
        self.__ptr = self.funit

    def __del__(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._close()

    def __len__(self):
        return self.nrep

    def next(self):  # Python 2:
        #TODO: should we use indexing instead of search?
        self.__search = brp_findrpt(self.funit, self.__search)
        if not self.__search:
            self.__search = BurpcRpt()
            raise StopIteration
        return brp_getrpt(self.funit, self.__search.handle)

    ## def __setitem__(self, name, value):
    ##     #TODO: Should replace the rpt found with getitem(name) or add a new one

    def _close(self):
        """ """
        if self.funit:
            istat = _bp.c_brp_close(self.funit)
        self.funit = None

    ## def del(self, search): #TODO: __delitem__
    ##     raise Error

    def get(self, search=None, rpt=None):
        """Find a report and get its meta + data"""
        if search is None or isinstance(search, (BurpcRpt, dict)):
            search = brp_findrpt(self.funit, search)
            if search:
                return brp_getrpt(self.funit, search.handle, rpt)
            return None
        elif isinstance(search, (long, int)):
            if search < 0 or search >= self.nrep:
                raise IndexError('Index out of range: [0:{}['.format(self.nrep))
            if search >= len(self.__handles):
                i0 = len(self.__handles)
                search1 = None
                if i0 > 0:
                    search1 = self.__handles[-1]
                for i in range(i0, search+1):
                    search1 = brp_findrpt(self.funit, search1)
                    if search1:
                        self.__handles.append(search1.handle)
                    else:
                        break
            return brp_getrpt(self.funit, self.__handles[search], rpt)
        else:
            raise TypeError("For Name: {}, Not Supported Type: {}".format(repr(search), str(type(search))))

    def put(self, where, rpt):
        """ """
        if not isinstance(rpt, BurpcRpt):
            raise TypeError("rpt should be of type BurpcRpt, got: {}, ".format(str(type(rpt))))
        append = where is None
        if append:
            where = _bc.BRP_END_BURP_FILE
        ## elif isinstance(where, (BurpcRpt, dict)): #TODO:
        ## elif isinstance(where, (long, int)): #TODO:
        else:
            raise TypeError("For where: {}, Not Supported Type: {}".format(repr(where), str(type(where))))

        self.__handles = [] #TODO: is it the best place to invalidate the cache?
        #TODO: conditional brp_updrpthdr
        brp_updrpthdr(self.funit, rpt)
        brp_writerpt(self.funit, rpt, where)
        if append: #TODO: only if writeok
            self.nrep += 1

    def append(self, value):
        self.put(None, value)


class BurpcRpt(_BurpcObjBase):
    """
    Python Class equivalent of the burp_c's BURP_RPT C structure to hold
    the BURP report data

    TODO: constructor examples

    Attributes:
        TODO
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
        if self.__bkno >= self.nblk:
            self.__bkno = 0
            raise StopIteration
        ## return brp_getblk(self.__bkno, self.__blk, self)
        self.__bkno += 1
        ## self.__blk = brp_getblk(self.__bkno, None, self)
        self.__blk = brp_getblk(self.__bkno, self.__blk, self)
        return self.__blk

    def get(self, name=None, blk=None):
        """Find a block and get its meta + data"""
        if name in self.__class__.__attrlist:
            return getattr(self.__ptr[0], name)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            try:
                name2 = self.__attrlist2names[name]
            except KeyError:
                name2 = name
            return self.derived_attr()[name2]
        elif isinstance(name, (int, long)):
            name += 1
            if name < 1 or name > self.nblk:
                raise IndexError('Index out of range: [0:{}['.format(self.nblk))
            return brp_getblk(name, blk=blk, rpt=self)
        elif name is None or isinstance(name, (BurpcBlk, dict)):
            name2 = brp_findblk(name, self)
            if name2:
                bkno = name2.bkno
                return brp_getblk(bkno, blk=blk, rpt=self)
            return None
        raise KeyError("{} object has no such key: {}"
                       .format(self.__class__.__name__, repr(name)))

    def __setattr__(self, name, value): #TODO: move to super class
        return self.put(name, value)

    def put(self, name, value):
        if name == 'stnid':
            self.__derived = None
            _bp.c_brp_setstnid(self.__ptr, value)
        elif name in self.__class__.__attrlist:
            self.__derived = None
            return setattr(self.__ptr[0], name, value)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            #TODO: encode other items on the fly
            raise AttributeError(self.__class__.__name__+" object cannot set derived attribute '"+name+"'")
        ## elif isinstance(name, (int, long)): #TODO:
        ## elif name is None or isinstance(name, (BurpcBlk, dict)): #TODO:
        else:
            return super(self.__class__, self).__setattr__(name, value)
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")

    def append(self, value):
        self.put(None, value)

    def derived_attr(self):
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
            'runn'  : getattr(self.__ptr[0], 'runn'),  #TODO: provide decoded runn?
            'nblk'  : getattr(self.__ptr[0], 'nblk'),
            'sup'   : None,
            'nsup'  : 0,
            'xaux'  : None,
            'nxaux' : 0
            }


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
        elif isinstance(name, (long, int)):
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
        ## elif isinstance(name, (long, int)): #TODO:
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

    def derived_attr(self):
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
        params.update(_rmn.mrbtyp_decode(btyp))
        return params

    def _getelem(self, index):
        """indexing from 0 to nele-1"""
        #TODO: return BurpcEle object
        if index < 0 or index >= self.nele:
            raise IndexError
        params = self.todict()
        params.update(_rmn.mrbcvt_dict(self.lstele[index], False))
        params.update({
            'e_ele_no' : index,
            'e_tblval' : self.tblval[index,:,:],
            'e_val'    : None,
            'e_rval'   : None,
            'e_drval'  : None,
            'e_charval': None,
            })
        try:
            params['e_val'] = self.rval[index,:,:]
            params['e_rval'] = params['e_val']
        except:
            pass
        try:
            params['e_val'] = self.drval[index,:,:]
            params['e_drval'] = params['e_val']
        except:
            pass
        try:
            params['e_val'] = self.charval[index,:,:]
            params['e_charval'] = params['e_val']
        except:
            pass
        return params

    def putelem(self, index, values):
        """indexing from 0 to nele-1"""
        if not instance(values, dict):
            raise TypeError
        #TODO: check if all needed params are provided
        #TODO: check if dims match

## class BurpcEle(object):
##     """
##     Python Class to hold a BURP block element's data and meta

##     TODO: constructor examples

##     Attributes:
##         TODO:
##     """
##     __attrlist = ("id", "nval", "nt", "datyp", "store_type",
##                   "tblval", "rval","drval", "charval")
##             ## 'e_ele_no' : index,
##             ## 'e_tblval' : self.tblval[index,:,:],
##             ## 'e_val'    : None,
##             ## 'e_rval'   : None,
##             ## 'e_drval'  : None,
##             ## 'e_charval': None,
##     __attrlist2 = ('e_error', 'e_cmcid', 'e_bufrid', 'e_bufrid_F',
##                    'e_bufrid_X', 'e_bufrid_Y', 'e_cvt', 'e_desc',
##                    'e_units', 'e_scale', 'e_bias', 'e_nbits', 'e_multi')

##     def __init__(self, ele=None):
##         self.__derived = None
##         if ele is None:
##             pass #TODO: allow?
##         elif isinstance(ele, BurpcEle):
##             #TODO: copy?
##         elif isinstance(ele, dict):
##             #TODO: defaults
##             #TODO: Check that minimal stuff provided: id, values and/or shape
##         else:
##             raise TypeError

##     def update(self, ele):
##         self.__derived = None
##         if isinstance(ele, BurpcEle):
##         elif isinstance(ele, dict):
##         else:
##             raise TypeError

##     ## def __del__(self):
##     ##     ## print 'DEL:',self.__class__.__name__

##     def __repr__(self):
##         return self.__class__.__name__+'('+ repr(self.todict())+')'

##     ## def __getattr__(self, name):
##     ## def __setattr__(self, name, value):
##     ## def __getitem__(self, name):
##     ## def __setitem__(self, name, value):

##     def size():
##         """ """
##         return self.nval * self.nt

##     def shape():
##         """ """
##         return self.nval, self.nt

##     def todict(self):
##         """ """
##         return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])

##     def update(self, blk):
##         """ """


def brp_opt(optName, optValue=None):
    """
    Set/Get BURP file options

    brp_opt(optName, optValue)

    Args:
        optName  : name of option to be set or printed
                   or one of these constants:
                   BURPOP_MISSING, BURPOP_MSGLVL
        optValue : value to be set (float or string) (optional)
                   If not set or is None mrfopt will get the value
                   otherwise mrfopt will set to the provided value
                   for optName=BURPOP_MISSING:
                      a real value for missing data
                   for optName=BURPOP_MSGLVL, one of these constants:
                      BURPOP_MSG_TRIVIAL,   BURPOP_MSG_INFO,  BURPOP_MSG_WARNING,
                      BURPOP_MSG_ERROR,     BURPOP_MSG_FATAL, BURPOP_MSG_SYSTEM
    Returns:
        str or float, optValue
    Raises:
        KeyError   on unknown optName
        TypeError  on wrong input arg types
        BurpError  on any other error

    Examples:
    >>> import rpnpy.burpc.all as brp
    >>> # Restrict to the minimum the number of messages printed by librmn
    >>> brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
    'SYSTEM   '

    See Also:
        rpnpy.librmn.burp.mrfopt
        rpnpy.librmn.burp_const
    """
    if not optName in (_rmn.BURPOP_MISSING, _rmn.BURPOP_MSGLVL):
        raise KeyError("Uknown optName: {}".format(optName))

    if optValue is None:
        if optName == _rmn.BURPOP_MISSING:
            return _bp.c_brp_msngval() #TODO: should it be option.value?
        else:
            raise KeyError("Cannot get value for optName: {}".format(optName))

    if isinstance(optValue, str):
        istat = _bp.c_brp_SetOptChar(optName, optValue)
        if istat != 0:
            raise BurpcError('c_brp_SetOptChar: {}={}'.format(optName,optValue))
    elif isinstance(optValue, float):
        istat = _bp.c_brp_SetOptFloat(optName, optValue)
        if istat != 0:
            raise BurpcError('c_mrfopr:{}={}'.format(optName,optValue), istat)
    else:
        raise TypeError("Cannot set optValue of type: {0} {1}"\
                        .format(type(optValue), repr(optValue)))
    return optValue


def brp_open(filename, filemode='r', funit=None, getnbr=False):
    """
    #TODO: desc
    """
    if filemode in _filemodelist_inv.keys():
        filemode = _filemodelist_inv[filemode]
    try:
        fstmode, brpmode = _filemodelist[filemode]
    except:
        raise ValueError('Unknown filemode: "{}", should be one of: {}'
                        .format(filemode, repr(_filemodelist.keys())))
    #TODO: Check format/existence of file depending on mode as in burp_open
    if not funit:
        try:
            funit = _rmn.fnom(filename, fstmode)
            _rmn.fclos(funit)  #TODO: too hacky... any way to reserve a unit w/o double open?
        except _rmn.RMNBaseError:
            funit = None
    if not funit:
        raise BurpcError('Problem associating a unit with file: {} (mode={})'
                        .format(filename, filemode))
    nrep = _bp.c_brp_open(funit, filename, filemode)
    if getnbr:
        return (funit, nrep)
    return funit


def brp_close(funit):
    """
    #TODO: desc
    """
    if isinstance(funit, BurpcFile):
        funit.close()
    elif isinstance(funit, (long, int)):
        istat = _bp.c_brp_close(funit)
        if istat < 0:
            raise BurpcError('Problem closing burp file unit: "{}"'
                             .format(funit))
    else:
        raise TypeError('funit is type="{}"'.format(str(type(funit))) +
                        ', should be an "int" or a "BurpcFile"')


def brp_free(*args):
    """
    Free pointer intances to BURP_RPT and BURP_BLK

    brpc_free(myBURP_RPTptr)
    brpc_free(myBURP_BLKptr, myBURP_RPTptr)

    Args:

    Return:
        None
    Raises:
        TypeError on not supported types or args
    """
    for x in args:
        if isinstance(x, _ct.POINTER(_bp.BURP_RPT)):
            _bp.c_brp_freerpt(x)
        elif isinstance(x, _ct.POINTER(_bp.BURP_BLK)):
            _bp.c_brp_freeblk(x)
        ## elif isinstance(x, BurpcRpt, BurpcBlk):
        ##     x.__del__()
        else:
            raise TypeError("Not Supported Type: "+str(type(x)))


def brp_findrpt(funit, rpt=None): #TODO: rpt are search keys, change name
    """
    """
    if isinstance(funit, BurpcFile):
        funit = funit.funit
    if not rpt:
        rpt = BurpcRpt()
        rpt.handle = 0
    elif isinstance(rpt, (int, long)):
        handle = rpt
        rpt = BurpcRpt()
        rpt.handle = handle
    elif not isinstance(rpt, BurpcRpt):
        rpt = BurpcRpt(rpt)
    if _bp.c_brp_findrpt(funit, rpt.getptr()) >= 0:
        return rpt
    return None


def brp_getrpt(funit, handle=0, rpt=None):
    """
    """
    if isinstance(funit, BurpcFile):
        funit = funit.funit
    if isinstance(handle, BurpcRpt):
        if not rpt:
            rpt = handle
        handle = handle.handle
    if not isinstance(rpt, BurpcRpt):
        rpt = BurpcRpt(rpt)
    if _bp.c_brp_getrpt(funit, handle, rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_getrpt')
    return rpt


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BurpcRpt(rpt)
    if not isinstance(rpt, BurpcRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not blk:
        blk = BurpcBlk()
        blk.bkno = 0
    elif isinstance(blk, (int, long)):
        bkno = blk
        blk = BurpcBlk()
        blk.bkno = bkno
    elif not isinstance(blk, BurpcBlk):
        blk = BurpcBlk(blk)
    if _bp.c_brp_findblk(blk.getptr(), rpt.getptr()) >= 0:
        return blk
    return None


def brp_getblk(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BurpcRpt(rpt)
    if not isinstance(rpt, BurpcRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not isinstance(blk, BurpcBlk):
        blk = BurpcBlk(blk)
    else:
        blk.reset_arrays()
    if _bp.c_brp_getblk(bkno, blk.getptr(), rpt.getptr()) < 0:
        raise BurpcError('Problem in c_brp_getblk')
    return blk


## def brp_allocrpt(rpt, size):
##     """
##     """
##     if isinstance(rpt, BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_allocrpt(rpt, size) < 0:
##         raise BurpcError('Problem in brp_allocrpt')

## def brp_resizerpt(rpt, size):
##     """
##     """
##     if isinstance(rpt, BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_resizerpt(rpt, size) < 0:
##         raise BurpcError('Problem in brp_resizerpt')

## def brp_clrrpt(rpt):
##     """
##     """
##     if isinstance(rpt, BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_clrrpt(rpt) < 0:
##         raise BurpcError('Problem in c_brp_clrrpt')

## def brp_putrpthdr(funit, rpt):
##     """
##     """
##     if isinstance(funit, BurpcFile):
##         funit = funit.funit
##     if isinstance(rpt, BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if _bp.c_brp_putrpthdr(funit, rpt) < 0:
##         raise BurpcError('Problem in c_brp_putrpthdr')

def brp_updrpthdr(funit, rpt):
    """
    """
    if isinstance(funit, BurpcFile):
        funit = funit.funit
    if isinstance(rpt, BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_updrpthdr(funit, rpt) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')

def brp_writerpt(funit, rpt, where=_bc.BRP_END_BURP_FILE):
    """
    """
    if isinstance(funit, BurpcFile):
        funit = funit.funit
    if isinstance(rpt, BurpcRpt):
        rpt = rpt.getptr()
    if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if _bp.c_brp_writerpt(funit, rpt, where) < 0:
        raise BurpcError('Problem in c_brp_updrpthdr')

## def brp_allocblk(blk, nele=1, nval=1, nt=1):
##     """
##     """
##     #TODO: should we take nele, nval, nt from blk if not provided
##     if isinstance(blk, BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_allocblk(blk, nele, nval, nt) < 0:
##         raise BurpcError('Problem in c_brp_allocblk')

## def brp_resizeblk(blk, nele=1, nval=1, nt=1):
##     """
##     """
##     #TODO: should we take nele, nval, nt from blk if not provided
##     if isinstance(blk, BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_resizeblk(blk, nele, nval, nt) < 0:
##         raise BurpcError('Problem in c_brp_resizeblk')

## def brp_encodeblk(blk):
##     """
##     """
##     if isinstance(blk, BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_encodeblk(blk) < 0:
##         raise BurpcError('Problem in c_brp_encodeblk')

## def brp_convertblk(blk, mode=_bc.BRP_MKSA_to_BUFR):
##     """
##     """
##     if isinstance(blk, BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_convertblk(blk, mode) < 0:
##         raise BurpcError('Problem in c_brp_convertblk')

## def brp_putblk(rpt, blk):
##     """
##     """
##     if isinstance(rpt, BurpcRpt):
##         rpt = rpt.getptr()
##     if not isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
##         raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
##     if isinstance(blk, BurpcBlk):
##         blk = blk.getptr()
##     if not isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use blk or type={}'+str(type(blk)))
##     if _bp.c_brp_putblk(rpt, blk) < 0:
##         raise BurpcError('Problem in c_brp_putblk')

## def c_brp_copyblk(dst_blk, src_blk):
##     """
##     """
##     if isinstance(dst_blk, BurpcBlk):
##         dst_blk = dst_blk.getptr()
##     if not isinstance(dst_blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use dst_blk or type={}'+str(type(dst_blk)))
##     if isinstance(src_blk, BurpcBlk):
##         src_blk = src_blk.getptr()
##     if not isinstance(src_blk, _ct.POINTER(_bp.BURP_BLK)):
##         raise TypeError('Cannot use src_blk or type={}'+str(type(src_blk)))
##     if _bp.c_brp_copyblk(dst_blk, src_blk) < 0:
##         raise BurpcError('Problem in c_brp_copyblk')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
