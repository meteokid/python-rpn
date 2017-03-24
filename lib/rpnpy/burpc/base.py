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
from rpnpy.burpc import BURPCError
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

#TODO: name convention BurpFile? BURPfile?
class BURP_FILE(object):
    """
    """
    def __init__(self,filename, filemode='r', funit=None):
        self.filename = filename
        self.filemode = filemode
        self.__search = BURP_RPT_PTR()
        self.__rpt    = BURP_RPT_PTR()
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
        ## return _rmn.mrfnbr(self.funit)  #TODO: only use libburp_c API
        return self.nrep

    def __iter__(self):
        return self

    def __next__(self): # Python 3
        self.__search = brp_findrpt(self.funit, self.__search)
        if not self.__search:
            self.__search = BURP_RPT_PTR()
            raise StopIteration
        ## return self.__search
        ## return brp_getrpt(self.funit, self.__search.handle, self.__rpt)
        return brp_getrpt(self.funit, self.__search.handle)

    def next(self): # Python 2:
        return self.__next__()

        
    #TODO: __delitem__?

    ## def __getitem__(self, name):
    ##     if isinstance(name, (int, long)):
    ##         #TODO return ith report
    ##     elif isinstance(name, BURP_RPT_PTR):
    ##         #TODO return report matching search
    ##     elif isinstance(name, dict):
    ##         #TODO: create BURP_RPT_PTR with dict
    ##         #TODO return report matching search
    ##     else:
    ##         raise KeyError("No Such Key: "+repr(name))
 
    ## def __setitem__(self, name, value):
    ##     #TODO: should setitem set the ith block?
        
    def close(self):
        if self.funit:
            istat = _bp.c_brp_close(self.funit)
        self.funit = None

    #TODO: name convention get_rpt
    def getrpt(self, search=None, rpt=None):
        """Find a report and get its meta + data"""
        search = brp_findrpt(self.funit, search)
        if search:
            return brp_getrpt(self.funit, search.handle, rpt)
        return None

        
#TODO: name convention BurpRpt? BURPrpt
class BURP_RPT_PTR(object):
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
    
    def __init__(self, rpt=None):
        self.__bkno = 0
        self.__blk  = BURP_BLK_PTR()
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
        return self.__class__.__name__+'('+ repr(self.to_dict())+')'
        
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
        #TODO: decode other items on the fly
        else:
            raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
            ## return super(self.__class__, self).__getattr__(name)
        
    def __setattr__(self, name, value):
        if name == 'stnid':
            _bp.c_brp_setstnid(self.__ptr, value)
        elif name in self.__class__.__attrlist:
            return setattr(self.__ptr[0], name, value)  #TODO: use proto fn?
        #TODO: encode other items on the fly
        else:
            return super(self.__class__, self).__setattr__(name, value)
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
    
    def __getitem__(self, name):
        if name in self.__class__.__attrlist:
            return self.__getattr__(name)
        elif isinstance(name, (int, long)):
            if name < 1 or name > self.nblk:
                raise IndexError('Index out of range: 1:'+str(self.nblk))
            return brp_getblk(name, self.__blk, self) #TODO: self.getblk()?
        ## elif isinstance(name, BURP_RPT_PTR):
        ##     #TODO return report matching search
        ## elif isinstance(name, dict):
        ##     #TODO: create BURP_RPT_PTR with dict
        ##     #TODO return report matching search
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
    #TODO: def update(self, mydict_or_BURPRPT):

    def update(self, rpt):
        """ """
        if not isinstance(rpt, (dict, BURP_RPT_PTR)):
            raise TypeError("Type not supported for rpt: "+str(type(rpt)))
        for k in self.__class__.__attrlist:
            try:
                self.__setitem__(k, rpt[k])
            except:
                pass
        
    def get_ptr(self):
        """ """
        return self.__ptr
    
    def to_dict(self):
        """ """
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])

    #TODO: name convention get_blk
    def getblk(self, search=None, blk=None):
        """Find a block and get its meta + data"""
        search = brp_findblk(search, self)
        if search:
            bkno = search.bkno
            return brp_getblk(bkno, blk=blk, rpt=self)
        return None


class BURP_BLK_PTR(object):
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
    
    def __init__(self, blk=None):
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
        ## if len(self.__class__.__attrlist) == 0:
        ##     self.__class__.__attrlist = [v[0] for v in self.__ptr[0]._fields_]
        ##     self.__class__.__attrlist_np_1d = ["lstele", "dlstele"]
        ##     self.__class__.__attrlist_np_3d = ["tblval", "rval", "drval", "charval"]
        if to_update:
            self.update(blk)
        self.reset_arrays()

                    
    def __del__(self):
        ## print 'DEL:',self.__class__.__name__
        _bp.c_brp_freeblk(self.__ptr)

    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.to_dict())+')'
        
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
        #TODO: decode other items on the fly
        else:
            raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")

    def __setattr__(self, name, value):
        ## print 'setattr:', name
        if name in self.__class__.__attrlist:
            return setattr(self.__ptr[0], name, value) #TODO: use proto fn?
        #TODO: encode other items on the fly
        else:
            return super(self.__class__, self).__setattr__(name, value)

    def __getitem__(self, name):
        #TODO: should getitem access the ith element?
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
        if not isinstance(blk, (dict, BURP_BLK_PTR)):
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

    def get_ptr(self):
        return self.__ptr
    
    def to_dict(self):
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])


        
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
            raise BURPCError('c_brp_SetOptChar: {}={}'.format(optName,optValue))
    elif isinstance(optValue, float):
        istat = _bp.c_brp_SetOptFloat(optName, optValue)
        if istat != 0:
            raise BURPCError('c_mrfopr:{}={}'.format(optName,optValue), istat)
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
        raise BURPCError('Problem associating a unit with file: {} (mode={})'
                        .format(filename, filemode))
    nrep = _bp.c_brp_open(funit, filename, filemode)
    if getnbr:
        return (funit, nrep)
    return funit


def brp_close(funit):
    """
    #TODO: desc
    """
    if isinstance(funit, BURP_FILE):
         funit.close()
    elif isinstance(funit, (long, int)):
        istat = _bp.c_brp_close(funit)
        if istat < 0:
            raise BURPCError('Problem closing burp file unit: "{}"'
                             .format(funit))
    else:
        raise TypeError('funit is type="{}"'.format(str(type(funit))) +
                        ', should be an "int" or a "BURP_FILE"')
        

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
        ## elif isinstance(x, BURP_RPT_PTR, BURP_BLK_PTR):
        ##     x.__del__()
        else:
            raise TypeError("Not Supported Type: "+str(type(x)))


def brp_findrpt(funit, rpt=None): #TODO: rpt are search keys, change name
    """
    """
    if isinstance(funit, BURP_FILE):
         funit = funit.funit
    if not rpt:
        rpt = BURP_RPT_PTR()
        rpt.handle = 0
    elif isinstance(rpt, (int, long)):
        handle = rpt
        rpt = BURP_RPT_PTR()
        rpt.handle = handle
    elif not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
    if _bp.c_brp_findrpt(funit, rpt.get_ptr()) >= 0:
        return rpt
    return None

    
def brp_getrpt(funit, handle=0, rpt=None):
    """
    """
    if isinstance(funit, BURP_FILE):
         funit = funit.funit
    if isinstance(handle, BURP_RPT_PTR):
        if not rpt:
            rpt = handle
        handle = handle.handle
    if not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
    if _bp.c_brp_getrpt(funit, handle, rpt.get_ptr()) < 0:
        raise BRUPCError('Problem in c_brp_getrpt')
    return rpt


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BURP_RPT_PTR(rpt)
    if not isinstance(rpt, BURP_RPT_PTR):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))
    if not blk:
        blk = BURP_BLK_PTR()
        blk.bkno = 0
    elif isinstance(blk, (int, long)):
        bkno = blk
        blk = BURP_BLK_PTR()
        blk.bkno = bkno
    elif not isinstance(blk, BURP_BLK_PTR):
        blk = BURP_BLK_PTR(blk)
    if _bp.c_brp_findblk(blk.get_ptr(), rpt.get_ptr()) >= 0:
        return blk
    return None

    
def brp_getblk(bkno, blk=None, rpt=None): #TODO: how can we get a block in an empty report?
    """
    """
    if isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
        rpt = BURP_RPT_PTR(rpt)
    if not isinstance(rpt, BURP_RPT_PTR):
        raise TypeError('Cannot use rpt or type={}'+str(type(rpt)))

    if not isinstance(blk, BURP_BLK_PTR):
        blk = BURP_BLK_PTR(blk)
    else:
        blk.reset_arrays()
    if _bp.c_brp_getblk(bkno, blk.get_ptr(), rpt.get_ptr()) < 0:
        raise BRUPCError('Problem in c_brp_getblk')
    return blk

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
