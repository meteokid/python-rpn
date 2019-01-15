"""Base Classes to build meteo file/record/metadata classes.

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
    @date: 2010-02
"""

class BaseMeta:
    """Base Class to hold meteo file/record metadata
    
    It is basicly a dictionary with fixed key list, default values and fixed values type

    This class is meant to be subclassed.
    All subclass must at least define the __getDefaultKeyVal methode
    which return the dict of default key and values.
    """
    def getDefaultKeyVal(self):
        """Return dict with list of allowed keys with their default values/type
        May also set ignoreBadKeys in this s/r to override default
        This is the main function subclass has to implement
        """
        return {}
        
    def __init__(self,other={}):
        self.ignoreBadKeys = False
        self.__defaults = self.getDefaultKeyVal()
        self.__allowedKeys = self.__defaults.keys()
        self.__dict = self.__defaults
        
        self.update(other)

    def __setitem__(self,key,val):
        if key in self.__allowedKeys:
            if self.__defaults[key] is None \
                   or type(val)==type(self.__defaults[key]) \
                   or isinstance(val,self.__defaults[key].__class__):
                self.__dict[key] = val
            else:
                raise TypeError, 'Provided value is of the wrong type:'+str(type(val))
        elif not self.ignoreBadKeys:
            raise ValueError, 'Trying to set a key not in allowedKeys list:'+str(key)

    def __getitem__(self,key):
        return self.__dict[key]

    def __cmp__(self,other):
        if isinstance(other,self.__class__):
            d2 = other.__dict
        elif (type(other)==type({})):
            d2 = other
        else:
            raise TypeError, 'Cannot compare with this type of object:'+str(type(other))
        return d2.__cmp__(self.__dict)

    def __str__(self):
        return str(self.__class__)+'('+str(self.__dict)+')'

    def __repr__(self):
        return str(self.__class__)+'('+repr(self.__dict)+')'

    def update(self,other):
        if isinstance(other,self.__class__) or type(other)==type(self):
            d2 = other.__dict
        elif type(other)==type({}):
            d2 = other
        else:
            raise TypeError, 'Cannot update with this type of object:'+str(type(other))
        self.__dict.update(d2)
        #return self
    
    #TODO: conditional update (if other[key]!=self.__defaults[key])

    def keys(self):
        return self.__dict.keys()

    def values(self):
        return self.__dict.values()

    def items(self):
        return self.__dict.items()

    def has_key(self,key):
        return self.__dict.has_key(key)
    
    def copy(self):
        """Return a copy"""
        return BaseMeta(self)

    def deepcopy(self):
        pass #TODO: implement
    
    def reset(self):
        """Reverts all meta values to their defaults"""
        self.__dict = self.__defaults


class BaseRec(BaseMeta):
    """
    """

    def __init__(self,other={}):
        BaseMeta.__init__(self,other)
        self.data = None

    #TODO: copy, deepcopy, __cmp__?


#TODO: class geoRefRec(BaseRec):

class BaseFile:
    """
    """
    def __getDefaultRecClass(self):
        """Return default allowed rec Class
        May also set ignoreBadClass, allowSubClass to override default

        This function needs to be subclassed
        """
        return BaseRec.__class__

    def __init__(self,filename,options=''):
        self.ignoreBadClass = False
        self.allowSubClass  = True
        self.__recClass = self.__getDefaultRecClass()

        self.filename = filename
        self.options = options
        self.recs = []
        #open file? or raise Error
        #get rec list? if existing file (optional since it may be a loooong process)

    def __del__(self):
        pass #close file

    def __isRecClassOK(self,rec):
        ok = False
        if self.allowSubClass:
            if isinstance(rec,self.__recClass): ok = True
        else:
            if rec.__class__ == self.__recClass: ok = True
        return ok

    def __delitem__(self):
        pass #delete record
    
    def __setitem__(self,key,rec):
        if self.__isRecClassOK(rec):
            self.write(rec)
        elif not self.ignoreBadClass:
            raise TypeError, 'Provided rec is of the wrong class'
    
    def __getitem__(self,key):
        return self.read(key)

    def find(self,searchKeys):
        pass

    def read(self):
        pass

    def write(self,rec):
        pass

    def erase(self):
        pass

    def append(self,rec):
        pass

    #TODO: define fn for iteration on rec
    #TODO: define fn to behave like a list

    def __str__(self):
        return str(self.__class__)+'('+str(self.filename)+')'

    def __repr__(self):
        return str(self.__class__)+'('+repr(self.filename)+')'


if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
