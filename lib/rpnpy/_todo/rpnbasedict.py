 
"""Module RPNBaseDict contains the Base RPN data/meta container class.

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""


#TODO: replace a == None by "a is None"


import copy

#class RPNBaseDict(object):
class RPNBaseDict(dict):
    """Base RPN data/meta container class.
    
    Examples of use (also doctests):
    #TODO:
    """
    (KDEF, KTYPE, KWILD) = (0, 1, 2)
    
    def _getDefaultKeyVal(self): #To Be overridden
        return {}
    
    def _getAltName(self, name): #To Be overridden
        return name
    
    def _getAltValue(self, name, name0, value): #To Be overridden
        return value
    
    def _checkSetItem(self, name, value): #To Be overridden
        return True
    
    def _postSetItem(self, name, value): #To Be overridden
        pass
    
    #---- 
    
    def __init__(self, other=None):
        super(RPNBaseDict, self).__setattr__('_defaultKeyVal', self._getDefaultKeyVal())
        for name in self._defaultKeyVal.keys():
            for key in (RPNBaseDict.KDEF, RPNBaseDict.KTYPE, RPNBaseDict.KWILD):
                if not key in self._defaultKeyVal[name].keys():
                    self._defaultKeyVal[name][key] = None 
            super(RPNBaseDict, self).__setitem__(name, self._defaultKeyVal[name][RPNBaseDict.KDEF])
        if other:
            self.update(other)
    
    def __getattr__(self, name):
        return self.__getitem__(name)

    def __getitem__(self, name0):
        name = self._getAltName(name0)        
        return super(RPNBaseDict, self).__getitem__(name)
    
    def __setattr__(self, name, value):
        if name[0] != '_':
            return self.__setitem__(name, value)
        else:
            return super(RPNBaseDict, self).__setattr__(name, value)
    
    def __setitem__(self, name0, value0):
        name = self._getAltName(name0)
        value = self._getAltValue(name, name0, value0)
        if name[0] != '_':
            if not name in self._defaultKeyVal.keys():
                raise ValueError, 'Cannot set new RPNData item: '+repr(name)
            if value != None and self._defaultKeyVal[name][RPNBaseDict.KTYPE] != None and not isinstance(value, self._defaultKeyVal[name][RPNBaseDict.KTYPE]):
                raise TypeError, 'RPNData['+repr(name)+'] must be of type '+repr(self._defaultKeyVal[name][RPNBaseDict.KTYPE])+' (provided: '+repr(type(value))+')'
            if not self._checkSetItem(name, value):
                raise  TypeError, 'Cannot set RPNData['+repr(name)+'] = '+repr(value)
        #print 'setitem["'+name+'"] = '+repr(value)
        tmp = super(RPNBaseDict, self).__setitem__(name, value)
        self._postSetItem(name, value)
        return tmp


    def __delattr__(self, name):
        #if name in self._defaultKeyVal.keys():
        #    raise ValueError, 'RPNDate: Cannot delete '+name
        #return super(RPNDate, self).__delattr__(name)
        raise ValueError, 'RPNDate: Cannot delete '+str(name)
    
    def __delitem__(self, name):
        #if name in self._defaultKeyVal.keys():
        #    raise ValueError, 'RPNDate: Cannot delete '+name
        #return super(RPNDate, self).__delattr__(name)
        raise ValueError, 'RPNDate: Cannot delete '+str(name)

    
    def __coerce__(self, other):
        return None
    
    def __repr__(self):
        items = self.items()
        items.sort()
        return self.__class__.__name__+'('+repr(items)+')'
    
    def deepcopy(self):
        return self.__class__(copy.deepcopy(self.items()))
    
    def update(self, other=None, cond=False):
        """Update Class items values from object of same class, {}, [] or ()
        
        object1.update(object2, conditional)
        @param object1 object to be updated
        @param object2 object with new values, can be of
                       same type as object1,
                       dict {name1:valie1, ...},
                       list [[name1, valu1e1], ...], or
                       tuple ((name1, valu1e1), ...)
                       type of object2 values must match type of object1 values
        @param conditional If true, will not update if value == wildcard,
                       the subclass can defile a wildcard for every
                       accepted item name
        @return None
        """
        if isinstance(other, type({})):
            for key in other.keys():
                if not (cond and other[key] == self._defaultKeyVal[key][RPNBaseDict.KWILD] and other[key] != None):
                    self[key] = other[key]    #TODO:try deepcopy
        elif isinstance(other, type([])) or isinstance(other, type((1, ))):
            for item in other:
                if len(item) != 2  or not (isinstance(item, type([])) or isinstance(item, type((1, )))):
                    raise TypeError, \
                          'RPNData.update(object), object must be of type '+ \
                          self.__class__.__name__+ \
                          ', list or tuple of tuples ((name1, value1), ...)'
                if not (cond and item[1] == self._defaultKeyVal[item[0]][RPNBaseDict.KWILD] and item[1] != None):
                    self[item[0]] = item[1]   #TODO:try deepcopy
        else:
            raise TypeError, 'RPNData.update(object), object must be of type '+self.__class__.__name__+', list or tuple'
        #return self

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
