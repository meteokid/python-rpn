 
"""Module RPNData contains the classes used to manipulate RPN style data/meta and interact with RPNRec in RPNSTD files

    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpnpy.version
from rpnpy.rpnbasedict import *
from rpnpy.rpndate import *
from rpnpy.rpnhgrid import *
from rpnpy.rpnvgrid import *

import numpy as np

class RPNData(RPNBaseDict):
    """RPN data/meta container class.
    Hold record data and metadata, needed to interact with RPNRec in RPNSTD files
    TODO: replace name/type/etiket by id(str) + extra(dict)
    TODO: replace date by timeinfo?

    The RPNData instance has the folowing attributes:
        d     : Actual Data            (type=numpy.ndarray)
        name  : Name/tag/id            (type=string)
        type  : Type of Data           (type=string)
        etiket: Secondary Tag          (type=string)
        vgrid : Vertical Description   (type=RPNVGgrid)
        hgrid : Horizontal Description (type=RPNHGgrid)
        date  : Date/DateRange         (type=RPNDate)

    
    Examples of use (also doctests):
    
    >>> myRPNData = RPNData()
    >>> #myRPNData.a               #except KeyError:
    >>> #myRPNData.d = ' '         #except TypeError:
    >>> myRPNData.name = 't2'
    >>> print myRPNData
    RPNData([('d', None), ('date', None), ('etiket', None), ('hgrid', None), ('name', 't2'), ('type', None), ('vgrid', None)])
    
    >>> d1 = myRPNData             #copy as a reference (Shallow copy)
    >>> d1
    RPNData([('d', None), ('date', None), ('etiket', None), ('hgrid', None), ('name', 't2'), ('type', None), ('vgrid', None)])
    >>> d1['name'] = 't0'          #Shallow copy modifies both original and copy
    >>> print d1['name']
    t0
    >>> d1.name = 't3'             #d1.name and d1['name'] are equivelent
    >>> print myRPNData.name, d1.name, d1['name']
    t3 t3 t3
    
    >>> d2 = myRPNData.deepcopy()  #deepcopy... TODO:check for numpyarry if copied
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', None), ('hgrid', None), ('name', 't3'), ('type', None), ('vgrid', None)])
    >>> print myRPNData.name, d2.name
    t3 t3
    >>> d2.name = 't4'             #deepcopy modifies only the copy
    >>> print myRPNData.name, d2.name
    t3 t4
    
    >>> d2.update(d1)
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', None), ('hgrid', None), ('name', 't3'), ('type', None), ('vgrid', None)])
    
    >>> d2.update({'etiket': 'test'})
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test'), ('hgrid', None), ('name', 't3'), ('type', None), ('vgrid', None)])
    
    >>> d2.update([['name','list'],['etiket','test2']])
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test2'), ('hgrid', None), ('name', 'list'), ('type', None), ('vgrid', None)])
    
    >>> d2.update([('name','list2')])
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test2'), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    
    >>> d2.update((('etiket','test3'),))
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test3'), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    
    >>> d2.update({'etiket': ' '},cond=True)
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test3'), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    
    >>> d2.update((('etiket', ' '),),cond=True)
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', 'test3'), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    
    >>> d2.update({'etiket': ' '})
    >>> print d2
    RPNData([('d', None), ('date', None), ('etiket', ' '), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    
    >>> #d2.update((('etiket','test3',' '),))  #except TypeError:
    >>> #d2.update(['etiket','test3'])        #except TypeError:
    >>> #d2.update(' ')                        #except TypeError:
    
    >>> d3 = RPNData(d2)           #Equivalent to d3 = d2.deepcopy()
    >>> print d3                   #        or to d3 = RPNData().update(d2)
    RPNData([('d', None), ('date', None), ('etiket', ' '), ('hgrid', None), ('name', 'list2'), ('type', None), ('vgrid', None)])
    """
    
    def _getDefaultKeyVal(self):
        (KDEF,KTYPE,KWILD) = (RPNBaseDict.KDEF,RPNBaseDict.KTYPE,RPNBaseDict.KWILD)
        return {
            'd'     : {KDEF:None, KTYPE:np.ndarray,  KWILD:None},
            'name'  : {KDEF:None, KTYPE:type(''),       KWILD:' '},
            'type'  : {KDEF:None, KTYPE:type(''),       KWILD:' '},
            'etiket': {KDEF:None, KTYPE:type(''),       KWILD:' '},
            'vgrid' : {KDEF:None, KTYPE:type(RPNVGgrid),KWILD:None},
            'hgrid' : {KDEF:None, KTYPE:type(RPNHGgrid),KWILD:None},
            'date'  : {KDEF:None, KTYPE:type(RPNDate),  KWILD:None}
            }

    def __init__(self,other=None):
        RPNBaseDict.__init__(self,other)
        #TODO: check consitency 'd' with vgrid and hgrid

    def _checkSetItem(self,name,value):
        #TODO: check consitency 'd' with vgrid and hgrid
        return True


if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
