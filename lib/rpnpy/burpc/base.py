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
from rpnpy.burpc  import const as _bc
from rpnpy.burpc  import BURPCError
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


class BURP_FILE(object):
    """
    """
    def __init__(self,filename, filemode='r'):
        self.filename = filename
        self.filemode = filemode
        self.funit, self.nrep = brp_open(self.filename, self.filemode, getnbr=True)

    def __del__(self):
        self._close()

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.filename, self.filemode)

    def __len__(self):
        ## return _rmn.mrfnbr(self.funit)  #TODO: only use libburp_c API
        return self.nrep

    def _close(self):
        if self.funit:
            istat = _bp.c_brp_close(self.funit)
        self.funit = None

    def getrpt(self, search=None, rpt=None):
        """Find a report and get its meta + data"""
        search = brp_findrpt(self.funit, search)
        if search:
            return brp_getrpt(self.funit, search.handle, rpt)
        return None

        
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
        if rpt is None:
            ## print 'NEW:',self.__class__.__name__
            self.__ptr = _bp.c_brp_newrpt()            
        elif isinstance(rpt, _ct.POINTER(_bp.BURP_RPT)):
            ## print 'NEW:',self.__class__.__name__,'ptr'
            self.__ptr = rpt
        ## if isinstance(rpt, BURP_RPT_PTR):  #TODO: copy fn instead?
        ##     print 'NEW:',self.__class__.__name__,'alias'
        ##     self.__ptr = rpt.get_ptr()
        ## elif isinstance(rpt, dict):  #TODO:
        else:
            raise TypeError("Type not supported for rpt: "+str(type(rpt)))
    
    def __del__(self):
        ## print 'DEL:',self.__class__.__name__
        _bp.c_brp_freerpt(self.__ptr) #TODO

    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.to_dict())+')'
        
    ## def __len__(self): #TODO: not working with this def
    ##     if self.nblk:
    ##         return self.nblk
    ##     return 0
    
    def __getattr__(self, name):
        if name in self.__class__.__attrlist:
            #TODO: special case for arrays
            return getattr(self.__ptr[0], name)
        else:
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
            return super(self.__class__, self).__getattr__(name)

    def __setattr__(self, name, value):
        if name == 'stnid':
            _bp.c_brp_setstnid(self.__ptr, value)
        elif name in self.__class__.__attrlist:
            #TODO: special case for arrays
            return setattr(self.__ptr[0], name, value)
        else:
            return super(self.__class__, self).__setattr__(name, value)

    def __getitem__(self, name):
        return self.__getattr__(name)
    
    def __setitem__(self, name, value):
        return self.__setattr__(name, value)

    #TODO: def __delattr__(self, name):
    #TODO: def __coerce__(self, other):
    #TODO: def __cmp__(self, other):
    #TODO: def __sub__(self, other):
    #TODO: def __add__(self, nhours):
    #TODO: def __isub__(self, other):
    #TODO: def __iadd__(self, nhours):
    #TODO: def update(mydict):
    
    def get_ptr(self):
        return self.__ptr  #TODO: should it be a copy?
    
    def to_dict(self):
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])

    def getblk(self, search=None, rpt=None, blk=None):
        """Find a block and get its meta + data"""
        search = brp_findblk(search, rpt)
        bkno = search.bkno
        ## if isinstance(search, (int, long)):
        ##     bkno = search
        ## else:
        ##     search = brp_findblk(search, rpt)
        ##     bkno = search.bkno
        if search:
            return brp_getblk(bkno, blk=blk, rpt=rpt)
        return None


class BURP_BLK_PTR(object):
    """
    Python Class equivalenet of the burp_c's BURP_BLK C structure to hold
    the BURP block data

    TODO: constructor examples

    Attibutes:
        TODO:
    """    
    __attrlist = ()
    
    def __init__(self, blk=None):
        ## self.__ptr = _bp.c_brp_newblk()
        if blk is None:
            ## print 'NEW:',self.__class__.__name__
            self.__ptr = _bp.c_brp_newblk()
        elif isinstance(blk, _ct.POINTER(_bp.BURP_BLK)):
            ## print 'NEW:',self.__class__.__name__,'ptr'
            self.__ptr = blk
        ## if isinstance(blk, BURP_BLK_PTR):  #TODO: copy fn instead?
        ##     print 'NEW:',self.__class__.__name__,'alias'
        ##     self.__ptr = blk.get_ptr()
        ## elif isinstance(blk, dict): #TODO
        else:
            raise TypeError("Type not supported for blk: "+str(type(blk)))
        if len(self.__class__.__attrlist) == 0:
            self.__class__.__attrlist = [v[0] for v in self.__ptr[0]._fields_]

    #TODO: __enter__, __exit__ to use with statement

    def __del__(self):
        ## print 'DEL:',self.__class__.__name__
        _bp.c_brp_freeblk(self.__ptr) #TODO

    def __repr__(self):
        return self.__class__.__name__+'('+ repr(self.to_dict())+')'
        
    def __getattr__(self, name):
        if name in self.__class__.__attrlist:
            #TODO: special case for arrays
            return getattr(self.__ptr[0], name)
        else:
            ## raise AttributeError(self.__class__.__name__+" object has not attribute '"+name+"'")
            return super(self.__class__, self).__getattr__(name)

    def __setattr__(self, name, value):
        if name in self.__class__.__attrlist:
            #TODO: special case for arrays
            return setattr(self.__ptr[0], name, value)
        else:
            return super(self.__class__, self).__setattr__(name, value)

    def __getitem__(self, name):
        return self.__getattr__(name)
    
    def __setitem__(self, name, value):
        return self.__setattr__(name, value)

    #TODO: def __delattr__(self, name):
    #TODO: def __coerce__(self, other):
    #TODO: def __cmp__(self, other):
    #TODO: def __sub__(self, other):
    #TODO: def __add__(self, nhours):
    #TODO: def __isub__(self, other):
    #TODO: def __iadd__(self, nhours):
    #TODO: def update(mydict):

    def get_ptr(self):
        return self.__ptr  #TODO: should it be a copy?
    
    def to_dict(self):
        return dict([(k, getattr(self,k)) for k in self.__class__.__attrlist])


    #TODO: to numpy array
        
#TODO: desc
brp_opt  = _rmn.mrfopt  #TODO: only use libburp_c API to be ready for burp-sql api

def brp_open(filename, filemode, getnbr=False):
    """
    #TODO: desc
    """
    if filemode in _filemodelist_inv.keys():
        filemode = _filemodelist_inv[filemode]
    try:
        fstmode, brpmode = _filemodelist[filemode]
    except:
        raise BURPCError('Unknown filemode: "{}", should be one of: {}'
                        .format(filemode, repr(_filemodelist.keys())))
    ## return _rmn.burp_open(filename, filemode) #TODO: only use libburp_c API to be ready for burp-sql api
    #TODO: Check format/existence of file depending on mode as in burp_open
    funit = _rmn.fnom(filename, fstmode)
    _rmn.fclos(funit)  #TODO: too hacky... any way to reserve a unit w/o double open?
    if not funit:
        raise BurpError('Problem associating a unit with the file: {0}'
                        .format(filename))
    nrep = _bp.c_brp_open(funit, filename, filemode)
    if getnbr:
        return (funit, nrep)
    return funit


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


def brp_findrpt(iunit, rpt=None): #TODO: rpt are search keys, change name
    """
    """
    if not rpt:
        rpt = BURP_RPT_PTR()
        rpt.handle = 0
    elif isinstance(rpt, (int, long)):
        handle = rpt
        rpt = BURP_RPT_PTR()
        rpt.handle = handle
    elif not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
    if _bp.c_brp_findrpt(iunit, rpt.get_ptr()) >= 0:
        return rpt
    return None

    
def brp_getrpt(iunit, handle, rpt):
    """
    """
    if not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
    if _bp.c_brp_getrpt(iunit, handle, rpt.get_ptr()) < 0:
        raise BRUPCError('Problem in c_brp_getrpt')
    return rpt


def brp_findblk(blk, rpt): #TODO: blk are search keys, change name
    """
    """
    if not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
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

    
def brp_getblk(bkno, blk, rpt):
    """
    """
    if not isinstance(rpt, BURP_RPT_PTR):
        rpt = BURP_RPT_PTR(rpt)
    if not isinstance(blk, BURP_BLK_PTR):
        blk = BURP_BLK_PTR(blk)    
    if _bp.c_brp_getblk(bkno, blk.get_ptr(), rpt.get_ptr()) < 0:
        raise BRUPCError('Problem in c_brp_getblk')
    return blk

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
