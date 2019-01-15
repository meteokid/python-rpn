#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Mario Lepine <mario.lepine@ec.gc.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Module rpn_helpers contains the helpers functions/classes for the rpn package
"""
import rpnpy.version as rpn_version
import Fstdc

LEVEL_KIND_MSL=0 #metres above sea level
LEVEL_KIND_SIG=1 #Sigma
LEVEL_KIND_PMB=2 #Pressure [mb]
LEVEL_KIND_ANY=3 #arbitrary code
LEVEL_KIND_MGL=4 #metres above ground level
LEVEL_KIND_HYB=5 #hybrid coordinates [hy]
LEVEL_KIND_TH=6 #theta [th]

def levels_to_ip1(levels, kind):
    """Encode level value into ip1 for the specified kind

    ip1_list = levels_to_ip1(level_list, kind)
    @param level_list list of level values [units depending on kind]
    @param kind   type of levels [units] to be encoded
        kind = 0: levels are in height [m] (metres) with respect to sea level
        kind = 1: levels are in sigma [sg] (0.0 -> 1.0)
        kind = 2: levels are in pressure [mb] (millibars)
        kind = 3: levels are in arbitrary code
        Looks like the following are not suppored yet in the fortran func convip,
        because they depend upon the local topography
            kind = 4: levels are in height [M] (metres) with respect to ground level
            kind = 5: levels are in hybrid coordinates [hy]
            kind = 6: levels are in theta [th]
    @return list of encoded level values-tuple ((ip1new, ip1old), ...)
    @exception TypeError if level_list is not a tuple or list
    @exception ValueError if kind is not an int in range of allowed kind

    Example of use (and doctest tests):

    >>> levels_to_ip1([0., 13.5, 1500., 5525., 12750.], 0)
    [(15728640, 12001), (8523608, 12004), (6441456, 12301), (6843956, 13106), (5370380, 14551)]
    >>> levels_to_ip1([0., 0.1, .02, 0.00678, 0.000003], 1)
    [(32505856, 2000), (27362976, 3000), (28511552, 2200), (30038128, 2068), (32805856, 2000)]
    >>> levels_to_ip1([1024., 850., 650., 500., 10., 2., 0.3], 2)
    [(39948288, 1024), (41744464, 850), (41544464, 650), (41394464, 500), (42043040, 10), (43191616, 1840), (44340192, 1660)]
    """
    if not type(levels) in (type(()), type([])):
        raise ValueError, 'levels_to_ip1: levels should be a list or a tuple; '+repr(levels)
    if type(kind) <> type(0):
        raise TypeError, 'levels_to_ip1: kind should be an int in range [0, 3]; '+repr(kind)
    elif not kind in (0, 1, 2, 3): #(0, 1, 2, 3, 4, 5, 6): 
        raise ValueError, 'levels_to_ip1: kind should be an int in range [0, 3]; '+repr(kind)
    if type(levels) == type(()):
        ip1_list = Fstdc.level_to_ip1(list(levels), kind)
    else:
        ip1_list = Fstdc.level_to_ip1(levels, kind)
    if not ip1_list:
        raise TypeError, 'levels_to_ip1: wrong args type; levels_to_ip1(levels, kind)'
    return(ip1_list)


def ip1_to_levels(ip1list):
    """Decode ip1 value into (level, kind)

    levels_list = ip1_to_levels(ip1list)
    @param ip1list list of ip1 values to decode
    @return list of decoded level values-tuple ((level, kind), ...)
        kind = 0: levels are in height [m] (metres) with respect to sea level
        kind = 1: levels are in sigma [sg] (0.0 -> 1.0)
        kind = 2: levels are in pressure [mb] (millibars)
        kind = 3: levels are in arbitrary code
        kind = 4: levels are in height [M] (metres) with respect to ground level
        kind = 5: levels are in hybrid coordinates [hy]
        kind = 6: levels are in theta [th]
    @exception TypeError if ip1list is not a tuple or list

    Example of use (and doctest tests):

    >>> [(int(x*10.e6+0.5), y) for x, y in ip1_to_levels([0, 1, 1000, 1199, 1200, 1201, 9999, 12000, 12001, 12002, 13000])]
    [(0, 2), (10000000, 2), (10000000000, 2), (10000000, 3), (0, 3), (500, 2), (7999000, 1), (10000000, 1), (0, 0), (50000000, 0), (49950000000, 0)]
    >>> [(int(x*10.e6+0.5), y) for x, y in ip1_to_levels([15728640, 12001, 8523608, 12004, 6441456, 12301, 6843956, 13106, 5370380, 14551])]
    [(0, 0), (0, 0), (135000000, 0), (150000000, 0), (15000000000, 0), (15000000000, 0), (55250000000, 0), (55250000000, 0), (127500000000, 0), (127500000000, 0)]
    >>> [(int(x*10.e6+0.5), y) for x, y in ip1_to_levels([32505856, 2000, 27362976, 3000, 28511552, 2200, 30038128, 2068, 32805856, 2000])]
    [(0, 1), (0, 1), (1000000, 1), (1000000, 1), (200000, 1), (200000, 1), (67800, 1), (68000, 1), (30, 1), (0, 1)]
    >>> [(int(x*10.e6+0.5), y) for x, y in ip1_to_levels([39948288, 1024, 41744464, 850, 41544464, 650, 41394464, 500, 42043040, 10, 43191616, 1840, 44340192, 1660])]
    [(10240000000, 2), (10240000000, 2), (8500000000, 2), (8500000000, 2), (6500000000, 2), (6500000000, 2), (5000000000, 2), (5000000000, 2), (100000000, 2), (100000000, 2), (20000000, 2), (20000000, 2), (3000000, 2), (3000000, 2)]
    """
    if not type(ip1list) in (type(()), type([])):
        raise TypeError, 'ip1_to_levels: levels should be a list or a tuple'

    if type(ip1list) == type(()):
        levels = Fstdc.ip1_to_level(list(ip1list))
    else:
        levels = Fstdc.ip1_to_level(ip1list)
    if not levels:
        raise TypeError, 'ip1_to_levels: wrong args type; ip1_to_levels(ip1list)'
    return(levels)


def cxgaig(grtyp, xg1, xg2=None, xg3=None, xg4=None):
    """Encode grid definition values into ig1-4 for the specified grid type

    (ig1, ig2, ig3, ig4) = cxgaig(grtyp, xg1, xg2, xg3, xg4):
    (ig1, ig2, ig3, ig4) = cxgaig(grtyp, (xg1, xg2, xg3, xg4)):

    @param grtyp
    @param xg1 xg1 value (float) or tuple of the form (xg1, xg2, xg3, xg4)
    @param xg2 xg2 value (float)
    @param xg3 xg3 value (float)
    @param xg4 xg4 value (float)
    @return Tuple of encoded grid desc values (ig1, ig2, ig3, ig4)
    @exception TypeError if args are of wrong type
    @exception ValueError if grtyp is not in ('A', 'B', 'E', 'G', 'L', 'N', 'S')

    Example of use (and doctest tests):

    >>> cxgaig('N', 200.5, 200.5, 40000.0, 21.0)
    (2005, 2005, 2100, 400)
    >>> cxgaig('N', 200.5, 220.5, 40000.0, 260.0)
    (400, 1000, 29830, 57333)
    >>> cxgaig('S', 200.5, 200.5, 40000.0, 21.0)
    (2005, 2005, 2100, 400)
    >>> cxgaig('L', -89.5, 180.0, 0.5, 0.5)
    (50, 50, 50, 18000)
    >>> ig1234 = (-89.5, 180.0, 0.5, 0.5)
    >>> cxgaig('L', ig1234)
    (50, 50, 50, 18000)

    Example of bad use (and doctest tests):

    >>> cxgaig('L', -89.5, 180  , 0.5, 0.5)
    Traceback (most recent call last):
    ...
    TypeError: cxgaig error: ig1, ig2, ig3, ig4 should be of type real:(-89.5, 180, 0.5, 0.5)
    >>> cxgaig('I', -89.5, 180.0, 0.5, 0.5)
    Traceback (most recent call last):
    ...
    ValueError: cxgaig error: grtyp ['I'] must be one of ('A', 'B', 'E', 'G', 'L', 'N', 'S')
    """
    validgrtyp = ('A', 'B', 'E', 'G', 'L', 'N', 'S') #I
    if xg2 == xg3 == xg4 == None and type(xg1) in (type([]), type(())) and len(xg1) == 4:
        (xg1, xg2, xg3, xg4) = xg1
    if None in (grtyp, xg1, xg2, xg3, xg4):
        raise TypeError, 'cxgaig error: missing argument, calling is cxgaig(grtyp, xg1, xg2, xg3, xg4)'
    elif not grtyp in validgrtyp:
        raise ValueError, 'cxgaig error: grtyp ['+repr(grtyp)+'] must be one of '+repr(validgrtyp)
    elif not (type(xg1) == type(xg2) == type(xg3) == type(xg4) == type(0.)):
        raise TypeError, 'cxgaig error: ig1, ig2, ig3, ig4 should be of type real:'+repr((xg1, xg2, xg3, xg4))
    else:
       return(Fstdc.cxgaig(grtyp, xg1, xg2, xg3, xg4))


def cigaxg(grtyp, ig1, ig2=None, ig3=None, ig4=None):
    """Decode grid definition values into xg1-4 for the specified grid type

    (xg1, xg2, xg3, xg4) = cigaxg(grtyp, ig1, ig2, ig3, ig4):
    (xg1, xg2, xg3, xg4) = cigaxg(grtyp, (ig1, ig2, ig3, ig4)):

    @param grtyp
    @param ig1 ig1 value (int) or tuple of the form (ig1, ig2, ig3, ig4)
    @param ig2 ig2 value (int)
    @param ig3 ig3 value (int)
    @param ig4 ig4 value (int)
    @return Tuple of decoded grid desc values (xg1, xg2, xg3, xg4)
    @exception TypeError if args are of wrong type
    @exception ValueError if grtyp is not in ('A', 'B', 'E', 'G', 'L', 'N', 'S')

    Example of use (and doctest tests):

    >>> cigaxg('N', 2005,  2005,  2100,   400)
    (200.5, 200.5, 40000.0, 21.0)
    >>> cigaxg('N', 400,  1000, 29830, 57333)
    (200.5013427734375, 220.4964141845703, 40000.0, 260.0)
    >>> cigaxg('S', 2005,  2005,  2100,   400)
    (200.5, 200.5, 40000.0, 21.0)
    >>> cigaxg('L', 50,    50,    50, 18000)
    (-89.5, 180.0, 0.5, 0.5)
    >>> ig1234 = (50,    50,    50, 18000)
    >>> cigaxg('L', ig1234)
    (-89.5, 180.0, 0.5, 0.5)

    Example of bad use (and doctest tests):

    >>> cigaxg('L', 50,    50,    50, 18000.)
    Traceback (most recent call last):
    ...
    TypeError: cigaxg error: ig1, ig2, ig3, ig4 should be of type int:(50, 50, 50, 18000.0)
    >>> cigaxg('I', 50,    50,    50, 18000)
    Traceback (most recent call last):
    ...
    ValueError: cigaxg error: grtyp ['I'] must be one of ('A', 'B', 'E', 'G', 'L', 'N', 'S')
    """
    validgrtyp = ('A', 'B', 'E', 'G', 'L', 'N', 'S') #I
    if ig2 == ig3 == ig4 == None and type(ig1) in (type([]), type(())) and len(ig1) == 4:
        (ig1, ig2, ig3, ig4) = ig1
    if None in (grtyp, ig1, ig2, ig3, ig4):
        raise TypeError, 'cigaxg error: missing argument, calling is cigaxg(grtyp, ig1, ig2, ig3, ig4)'
    elif not grtyp in validgrtyp:
        raise ValueError, 'cigaxg error: grtyp ['+repr(grtyp)+'] must be one of '+repr(validgrtyp)
    elif not (type(ig1) == type(ig2) == type(ig3) == type(ig4) == type(0)):
        raise TypeError, 'cigaxg error: ig1, ig2, ig3, ig4 should be of type int:'+repr((ig1, ig2, ig3, ig4))
    else:
        return(Fstdc.cigaxg(grtyp, ig1, ig2, ig3, ig4))


def dump_keys_and_values(self):
    """Return a string with comma separated key=value of all parameters
    """
    result=''
    keynames = self.__dict__.keys()
    keynames.sort()
    for name in keynames:
        result=result+name+'='+repr(self.__dict__[name])+' , '
    return result[:-3]  # eliminate last blank comma blank sequence


class RPNParm:
    """Base methods for all RPN standard file descriptor classes
    """
    def __init__(self, model, reference, extra):
        for name in reference.keys():            # copy initial values from reference
            self.__dict__[name]=reference[name]  # bypass setatttr method for new attributes
        if model != None:
            if isinstance(model, RPNParm):        # update with model attributes
               self.update(model)
            elif type(model) == type({}):     # update with dict
               self.update(model)
            else:
                raise TypeError, 'RPNParm.__init__: model must be an RPNParm class instances'
        for name in extra.keys():                # add extras using own setattr method
            setattr(self, name, extra[name])

    def allowedKeysVals(self):
        """function must be defined in subclass, return dict of allowed keys/vals, vals are default/wildcard values"""
        return {}

    def update(self, with1, updateToWild=True):
        """Replace RPN attributes of an instance with RPN attributes from another
        values not in list of allowed parm keys are ignored
        also update to wildcard (-1 or '') values
        unless updateToWild=False

        myRPNparm.update(otherRPNparm) #update myRPNparm values with otherRPNparm
        myRPNparm.update(otherRPNparm, updateToWild=False) #update myRPNparm values with otherRPNparm if not a wildcard
        @param otherRPNparm list of params=value to be updated, instance of RPNParm or derived class
        @param updateToWild if False prevent from updating to a wildcard value (default = True)
        @exception TypeError if otherRPNparm is of wrong type
        """
        allowedKeysVals = self.allowedKeysVals()
        if isinstance(with1, RPNParm):  # check if class=RPNParm
            for name in with1.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in allowedKeysVals.keys()):
                    if (updateToWild
                        or with1.__dict__[name] != allowedKeysVals[name]):
                        self.__dict__[name]=with1.__dict__[name]
                #else:
                #    print "cannot set:"+name+repr(allowedKeysVals.keys())
        elif type(with1) == type({}):
            for name in with1.keys():
                if name in self.__dict__.keys():
                   if (updateToWild
                        or with1[name] != allowedKeysVals[name]):
                        setattr(self, name, with1[name])
        else:
            raise TypeError, 'RPNParm.update: can only operate on RPNParm class instances or dict'

    def update_cond(self, with1):
        """Short form for RPNParm.update(with1, False)
        """
        self.update(with1, False)

    def update_by_dict(self, with1):
        """[Deprecated] Equivalent to RPNParm.update(with1)
        """
        self.update(with1)

    def update_by_dict_from(self, frm, with1):
        """TODO: documentation
        """
        for name in with1.keys():
            if name in self.__dict__.keys():
                setattr(self, name, frm.__dict__[name])

    def __setattr__(self, name, value):
        """Set RPNParm attribute value, will only accept a set of allowed attribute (rpnstd params list)
        myRPNparm.ip1 = 0
        @exception TypeError if value is of the wrong type for said attribute
        @exception ValueError if not a valid/allowed attribute
        """
        if name in self.__dict__.keys():                   # is attribute name valid ?
            if type(value) == type(self.__dict__[name]):   # right type (string or int))
                if type(value) == type(''):
                    reflen=len(self.__dict__[name])        # string, remember length
                    self.__dict__[name]=(value.upper()+reflen*' ')[:reflen]
                else:
                    self.__dict__[name]=value              # integer
            else:
                if self.__dict__[name] == None:
                    if type(value) == type(''):
                        self.__dict__[name]=value.upper()
                    else:
                        self.__dict__[name]=value
                else:
                    raise TypeError, 'RPNParm: Wrong type for attribute '+name+'='+repr(value)
        else:
            raise ValueError, 'RPNParm: attribute '+name+' does not exist for class '+repr(self.__class__)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return self.__dict__[name]

    def findnext(self, flag=True):
        """set/reset next match flag
        myRPNparm.findnext(True)  #set findnext flag to true
        myRPNparm.findnext(False) #set findnext flag to false
        """
        self.__dict__['nxt'] = 0
        if flag:
            self.__dict__['nxt'] = 1
        return self

    def wildcard(self):
        """Reset keys to undefined/wildcard"""
        self.update(self.allowedKeysVals())

    def __str__(self):
        return dump_keys_and_values(self)

    def __repr__(self):
        return repr(self.__dict__)


class RPNGridHelper:
    """Base RPNGrid Helper class - must be subclassed
    provides stubs of attr and methodes
    """
    addAllowedKeysVals = {}
    baseEzInterpArgs = {
        'shape' : (0, 0),
        'grtyp' : ' ',
        'g_ig14': (' ', 0, 0, 0, 0),
        'xy_ref': (None, None),
        'hasRef': 0,
        'ij0'   : (1, 1)
    }

    def parseArgs(self, keys, args):
        """Return a dict with parsed args for the specified grid type"""
        return {}

    def argsCheck(d):
        """Check Grid params, raise an error if not ok"""
        pass

    def getEzInterpArgs(self, keyVals, isSrc):
        """Return the list of needed args for Fstdc.ezinterp from the provided params"""
        return None

    def toScripGridName(self, keyVals):
        """Return a hopefully unique grid name for the provided params"""
        a = list(sg_a['g_ig14'])
        a.extend(sg_a['shape'])
        name = "grd%s-%i-%i-%i-%i-%i-%i" % a
        return name

    def toScripGridPreComp(self, keyVals, name=None):
        """Return a Scrip grid instance for the specified grid type (Precomputed addr&weights)"""
        shape = (4, keyVals['shape'][0], keyVals['shape'][1])
        if name is None:
            name = self.toScripGridName(keyVals)
        return scrip.ScripGrid(name, shape=shape)

    def toScripGrid(self, keyVals, name=None):
        """Return a Scrip grid instance for the specified grid type"""
        return None

    def reshapeDataForScrip(self, keyVals, data):
        """Return reformated data suitable for SCRIP"""
        return data

    def reshapeDataFromScrip(self, keyVals, data):
        """Inverse operation of reshapeDataForScrip (use helper)"""
        return data



if __name__ == "__main__":
    import doctest
    doctest.testmod()

# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
