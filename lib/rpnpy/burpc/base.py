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

class BurpFile(object):
    """
    """
    def __init__(self,filename, filemode='r', funit=None):
        self.filename = filename
        self.filemode = filemode
        self.__search = BurpRpt()
        self.__rpt    = BurpRpt()
        self.funit, self.nrep = brp_open(self.filename, self.filemode, funit=funit, getnbr=True)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.filename, self.filemode)

    def __len__(self):
        return self.nrep

    def __iter__(self):
        return self

    def __next__(self):  # Python 3
        self.__search = brp_findrpt(self.funit, self.__search)
        if not self.__search:
            self.__search = BurpRpt()
            raise StopIteration
        ## return self.__search
        ## return brp_getrpt(self.funit, self.__search.handle, self.__rpt)
        return brp_getrpt(self.funit, self.__search.handle)

    def next(self):  # Python 2:
        return self.__next__()

    #TODO: __delitem__?

    def __getitem__(self, name): #TODO: should call getrpt
        if isinstance(name, (BurpRpt, dict, long, int)):
            #TODO: if int or long, should we return the ith report?
            return self.getrpt(name)
        else:
            raise TypeError("Not Supported Type: "+str(type(name)))          
 
    ## def __setitem__(self, name, value):
    ##     #TODO: Should replace the rpt found with getitem(name)
        
    def close(self):
        if self.funit:
            istat = _bp.c_brp_close(self.funit)
        self.funit = None

    def getrpt(self, search=None, rpt=None):
        """Find a report and get its meta + data"""
        search = brp_findrpt(self.funit, search)
        if search:
            return brp_getrpt(self.funit, search.handle, rpt)
        return None

        
class BurpRpt(object):
    """
    Python Class equivalenet of the burp_c's BURP_RPT C structure to hold
    the BURP report data

    TODO: constructor examples

    Attibutes:
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
        self.__blk  = BurpBlk()
        self.__derived = None
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

    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.todict())+')'
        
    ## def __len__(self): #TODO: not working with this def
    ##     if self.nblk:
    ##         return self.nblk
    ##     return 0

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        if self.__bkno >= self.nblk:
            self.__bkno = 0
            raise StopIteration
        ## return brp_getblk(self.__bkno, self.__blk, self)
        self.__bkno += 1
        ## self.__blk = brp_getblk(self.__bkno, None, self)
        self.__blk = brp_getblk(self.__bkno, self.__blk, self)
        return self.__blk

    def next(self): # Python 2:
        return self.__next__()

    def __getattr__(self, name):
        if name in self.__class__.__attrlist:
            return getattr(self.__ptr[0], name)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            try:
                name = self.__attrlist2names[name]
            except KeyError:
                pass
            if not self.__derived:
                self.__derived = self.derived_attr()
            return self.__derived[name]
        else:
            raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
            ## return super(self.__class__, self).__getattr__(name)
        
    def __setattr__(self, name, value):
        if name == 'stnid':
            self.__derived = None
            _bp.c_brp_setstnid(self.__ptr, value)
        elif name in self.__class__.__attrlist:
            self.__derived = None
            return setattr(self.__ptr[0], name, value)  #TODO: use proto fn?
        #TODO: encode other items on the fly
        elif name in self.__class__.__attrlist2:
            raise AttributeError(self.__class__.__name__+" object cannot set derived attribute '"+name+"'")
        else:
            return super(self.__class__, self).__setattr__(name, value)
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
    
    def __getitem__(self, name):
        if name in (self.__class__.__attrlist + self.__class__.__attrlist2):
            return self.__getattr__(name)
        elif isinstance(name, (int, long)):
            if name < 1 or name > self.nblk:
                raise IndexError('Index out of range: 1:'+str(self.nblk))
            return brp_getblk(name, self.__blk, self) #TODO: self.getblk()?
        elif isinstance(name, (BurpBlk, dict)):
            return self.getblk(name)
        else:
            raise KeyError("No Such Key: "+repr(name))
    
    def __setitem__(self, name, value):
        #TODO: should setitem set the ith block?
        return self.__setattr__(name, value)

    #TODO: def __delitem__(self, name):
    #TODO: def __delattr__(self, name):
    #TODO: def __coerce__(self, other):
    #TODO: def __cmp__(self, other):
    #TODO: def __sub__(self, other):
    #TODO: def __add__(self, nhours):
    #TODO: def __isub__(self, other):
    #TODO: def __iadd__(self, nhours):

    def update(self, rpt):
        """ """
        if not isinstance(rpt, (dict, BurpRpt)):
            raise TypeError("Type not supported for rpt: "+str(type(rpt)))
        for k in self.__class__.__attrlist:
            try:
                self.__setitem__(k, rpt[k])
            except:
                pass
        
    def getptr(self):
        """ """
        return self.__ptr
    
    def todict(self):
        """ """
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])

    def getblk(self, search=None, blk=None):
        """Find a block and get its meta + data"""
        search = brp_findblk(search, self)
        if search:
            bkno = search.bkno
            return brp_getblk(bkno, blk=blk, rpt=self)
        return None

    def derived_attr(self):
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
        

class BurpBlk(object):
    """
    Python Class equivalenet of the burp_c's BURP_BLK C structure to hold
    the BURP block data

    TODO: constructor examples

    Attibutes:
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

    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.todict())+')'
        
    def __getattr__(self, name):
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
                ## self.__arr[name].flags['F_CONTIGUOUS'] = True
            return self.__arr[name]
        elif name in self.__class__.__attrlist:
            return getattr(self.__ptr[0], name)  #TODO: use proto fn?
        elif name in self.__class__.__attrlist2:
            if not self.__derived:
                self.__derived = self.derived_attr()
            return self.__derived[name]            
        else:
            raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")

    def __setattr__(self, name, value):
        ## print 'setattr:', name
        if name in self.__class__.__attrlist:
            self.__derived = None
            return setattr(self.__ptr[0], name, value) #TODO: use proto fn?
        #TODO: encode other items on the fly
        elif name in self.__class__.__attrlist2:
            raise AttributeError(self.__class__.__name__+" object cannot set derived attribute '"+name+"'")
        else:
            return super(self.__class__, self).__setattr__(name, value)

    def __getitem__(self, name):
        if isinstance(name, (long, int)):
            return self.getelem(name)
        try:
            return self.__getattr__(name)
        except:
            raise KeyError("No Such Key: "+repr(name))

    def __setitem__(self, name, value):
        #TODO: should setitem set the ith element?
        if name in self.__class__.__attrlist:
            return self.__setattr__(name, value)
        else:
            raise KeyError("No Such Key: "+repr(name))

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        if self.__eleno >= self.nele:
            self.__eleno = 0
            raise StopIteration
        ele = self.getelem(self.__eleno)
        self.__eleno += 1
        return ele

    def next(self): # Python 2:
        return self.__next__()

    #TODO: def __delitem__(self, name):
    #TODO: def __delattr__(self, name):
    #TODO: def __coerce__(self, other):
    #TODO: def __cmp__(self, other):
    #TODO: def __sub__(self, other):
    #TODO: def __add__(self, nhours):
    #TODO: def __isub__(self, other):
    #TODO: def __iadd__(self, nhours):

    def update(self, blk):
        """ """
        if not isinstance(blk, (dict, BurpBlk)):
            raise TypeError("Type not supported for blk: "+str(type(blk)))
        for k in self.__class__.__attrlist:
            try:
                self.__setitem__(k, blk[k])
            except:
                pass

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

    def getptr(self):
        return self.__ptr
    
    def todict(self):
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])

    def derived_attr(self):
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

    def getelem(self, index):
        """indexing from 0 to nele-1"""
        if index < 0 or index >= self.nele:
            raise IndexError
        params = self.todict()
        params.update(_rmn.mrbcvt_dict(self.lstele[index]))
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
    if isinstance(funit, BurpFile):
         funit.close()
    elif isinstance(funit, (long, int)):
        istat = _bp.c_brp_close(funit)
        if istat < 0:
            raise BurpcError('Problem closing burp file unit: "{}"'
                             .format(funit))
    else:
        raise TypeError('funit is type="{}"'.format(str(type(funit))) +
                        ', should be an "int" or a "BurpFile"')
        

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
        ## elif isinstance(x, BurpRpt, BurpBlk):
        ##     x.__del__()
        else:
            raise TypeError("Not Supported Type: "+str(type(x)))


def brp_findrpt(funit, rpt=None): #TODO: rpt are search keys, change name
    """
    """
    if isinstance(funit, BurpFile):
         funit = funit.funit
    if not rpt:
        rpt = BurpRpt()
        rpt.handle = 0
    elif isinstance(rpt, (int, long)):
        handle = rpt
        rpt = BurpRpt()
        rpt.handle = handle
    elif not isinstance(rpt, BurpRpt):
        rpt = BurpRpt(rpt)
    if _bp.c_brp_findrpt(funit, rpt.getptr()) >= 0:
        return rpt
    return None

    
def brp_getrpt(funit, handle=0, rpt=None):
    """
    """
    if isinstance(funit, BurpFile):
         funit = funit.funit
    if isinstance(handle, BurpRpt):
        if not rpt:
            rpt = handle
        handle = handle.handle
    if not isinstance(rpt, BurpRpt):
        rpt = BurpRpt(rpt)
    if _bp.c_brp_getrpt(funit, handle, rpt.getptr()) < 0:
        raise BRUPCError('Problem in c_brp_getrpt')
    return rpt


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BurpRpt(rpt)
    if not isinstance(rpt, BurpRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not blk:
        blk = BurpBlk()
        blk.bkno = 0
    elif isinstance(blk, (int, long)):
        bkno = blk
        blk = BurpBlk()
        blk.bkno = bkno
    elif not isinstance(blk, BurpBlk):
        blk = BurpBlk(blk)
    if _bp.c_brp_findblk(blk.getptr(), rpt.getptr()) >= 0:
        return blk
    return None

    
def brp_getblk(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BurpRpt(rpt)
    if not isinstance(rpt, BurpRpt):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))

    if not isinstance(blk, BurpBlk):
        blk = BurpBlk(blk)
    else:
        blk.reset_arrays()
    if _bp.c_brp_getblk(bkno, blk.getptr(), rpt.getptr()) < 0:
        raise BRUPCError('Problem in c_brp_getblk')
    return blk

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
