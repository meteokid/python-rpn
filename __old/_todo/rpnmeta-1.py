"""Module RPNMeta contains the classes used to manipulate RPN Standard Files metadata

    @author: Mario Lepine <mario.lepine@ec.gc.ca>
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpn_version
import Fstdc
from rpn_helpers import *


class RPNKeys(RPNParm):
    """RPN standard file Primary descriptors class, used to search for a record.
    Descriptors are:
    {'nom':'    ','type':'  ','etiket':'            ','date':-1,'ip1':-1,'ip2':-1,'ip3':-1,'handle':-2,'nxt':0,'fileref':None}
    TODO: give examples of instanciation
    """
    def __init__(self,model=None,**args):
        RPNParm.__init__(self,model,self.allowedKeysVals(),args)

    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        return self.searchKeysVals()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {'nom':'    ','type':'  ','etiket':'            ','datev':-1,'ip1':-1,'ip2':-1,'ip3':-1,'handle':-2,'nxt':0,'fileref':None}

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {'nom':'    ','type':'  ','etiket':'            ','datev':0,'ip1':0,'ip2':0,'ip3':0,'handle':-2,'nxt':0,'fileref':None}


class RPNDesc(RPNParm):
    """RPN standard file Auxiliary descriptors class, used when writing a record or getting descriptors from a record.
    Descriptors are:
    {'grtyp':'X','dateo':0,'deet':0,'npas':0,'ig1':0,'ig2':0,'ig3':0,'ig4':0,'datyp':0,'nbits':0,'xaxis':None,'yaxis':None,'xyref':(None,None,None,None,None),'griddim':(None,None)}
    TODO: give examples of instanciation
    """
    def __init__(self,model=None,**args):
        RPNParm.__init__(self,model,self.allowedKeysVals(),args)

    def allowedKeysVals(self):
       """Return a dict of allowed Keys/Vals"""
       return  self.searchKeysVals()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {'grtyp':' ','dateo':-1,'deet':-1,'npas':-1,'ig1':-1,'ig2':-1,'ig3':-1,'ig4':-1,'datyp':-1,'nbits':-1,'ni':-1,'nj':-1,'nk':-1}

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {'grtyp':'X','dateo':0,'deet':0,'npas':0,'ig1':0,'ig2':0,'ig3':0,'ig4':0,'datyp':4,'nbits':16,'ni':1,'nj':1,'nk':1}


class RPNMeta(RPNKeys,RPNDesc):
    """RPN standard file Full set (Primary + Auxiliary) of descriptors class, needed to write a record, can be used for search.

    myRPNMeta = RPNMeta()
    myRPNMeta = RPNMeta(anRPNMeta)
    myRPNMeta = RPNMeta(anRPNMeta,nom='TT')
    myRPNMeta = RPNMeta(nom='TT')

    @param anRPNMeta another instance of RPNMeta to copy data from
    @param nom [other descriptors can be used, see below] comma separated metadata key=value pairs
    @exception TypeError if anRPNMeta is not an instance of RPNMeta

    Descriptors are:
        'nom':'    ',
        'type':'  ',
        'etiket':'            ',
        'ip1':-1,'ip2':-1,'ip3':-1,
        'ni':-1,'nj':-1,'nk':-1,
        'dateo':0,
        'deet':0,
        'npas':0,
        'grtyp':'X',
        'ig1':0,'ig2':0,'ig3':0,'ig4':0,
        'datyp':0,
        'nbits':0,
        'handle':-2,
        'nxt':0,
        'fileref':None,
        'datev':-1

    Examples of use (also doctests):

    >>> myRPNMeta = RPNMeta() #New RPNMeta with default/wildcard descriptors
    >>> d = myRPNMeta.__dict__.items()
    >>> d.sort()
    >>> d
    [('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', -1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', '    '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> myRPNMeta = RPNMeta(nom='GZ',ip2=1)  #New RPNMeta with all descriptors to wildcard but nom,ip2
    >>> d = myRPNMeta.__dict__.items()
    >>> d.sort()
    >>> d
    [('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', 1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', 'GZ  '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> myRPNMeta.ip1
    -1
    >>> myRPNMeta2 = myRPNMeta #shallow copy (reference)
    >>> myRPNMeta2.ip1 = 9 #this will also update myRPNMeta.ip1
    >>> myRPNMeta.ip1
    9
    >>> myRPNMeta2 = RPNMeta(myRPNMeta)   #make a deep-copy
    >>> myRPNMeta.ip3
    -1
    >>> myRPNMeta2.ip3 = 9 #this will not update myRPNMeta.ip3
    >>> myRPNMeta.ip3
    -1
    >>> myRPNMeta.nom = 'TT'
    >>> myRPNMeta2 = RPNMeta(myRPNMeta,nom='GZ',ip2=8)   #make a deep-copy and update nom,ip2 values
    >>> (myRPNMeta.nom,myRPNMeta.ip1,myRPNMeta.ip2)
    ('TT  ', 9, 1)
    >>> (myRPNMeta2.nom,myRPNMeta2.ip1,myRPNMeta2.ip2)
    ('GZ  ', 9, 8)

    TODO: test update() and update_cond()
    """
    def __init__(self,model=None,**args):
        RPNParm.__init__(self,model,self.allowedKeysVals(),args)
        if model != None:
            if isinstance(model,RPNParm):
                self.update(model)
            else:
                raise TypeError,'RPNMeta: cannot initialize from arg #1'
        for name in args.keys(): # and update with specified attributes
            setattr(self,name,args[name])

    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        a = RPNKeys.allowedKeysVals(self)
        a.update(RPNDesc.allowedKeysVals(self))
        return a

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        a = RPNKeys.searchKeysVals(self)
        a.update(RPNDesc.searchKeysVals(self))
        return a

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        a = RPNKeys.defaultKeysVals(self)
        a.update(RPNDesc.defaultKeysVals(self))
        return a

    #TODO: move to either RPNFile; axis are no longuer part of meta data... part of grids but read action should be part of file class
    #def getaxis(self,axis=None):
        #"""Return the grid axis rec of grtyp ('Z','Y','#')

        #(myRPNRecX,myRPNRecY) = myRPNMeta.getaxis()
        #myRPNRecX = myRPNMeta.getaxis('X')
        #myRPNRecY = myRPNMeta.getaxis(axis='Y')

        #@param axis which axis to return (X, Y or None), default=None returns both axis
        #@exception TypeError if RPNMeta.grtyp is not in ('Z','Y','#')
        #@exception TypeError if RPNMeta.fileref is not an RPNFile
        #@exception ValueError if grid descriptors records (>>,^^) are not found in RPNMeta.fileref
        #"""
        #if not (self.grtyp in ('Z','Y','#')):
            #raise ValueError,'getaxis error: can not get axis from grtyp='+self.grtyp
        #if not isinstance(self.fileref,RPNFile):
            #raise TypeError,'RPNMeta.getaxis: ERROR - cannot get axis, no fileRef'
        #searchkeys = RPNKeys(ip1=self.ig1,ip2=self.ig2)
        #if self.grtyp != '#':
            #searchkeys.update_by_dict({'ip3':self.ig3})
        #searchkeys.nom = '>>'
        #xaxisrec = self.fileref[searchkeys]
        #searchkeys.nom = '^^'
        #yaxisrec = self.fileref[searchkeys]
        #if (xaxiskeys == None or yaxiskeys == None):
            #raise ValueError,'RPNMeta.getaxis: ERROR - axis grid descriptors (>>,^^) not found'
        #if type(axis) == type(' '):
            #if axis.upper() == 'X':
                #return xaxisrec
            #elif axis.upper() == 'Y':
                #return yaxisrec
                ##axisdata=self.yaxis.ravel()
        #return (xaxisrec,yaxisrec)

    def __repr__(self):
        return 'RPNMeta'+repr(self.__dict__)



if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
