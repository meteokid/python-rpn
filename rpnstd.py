#TODO: should be able to search all rec tags and get the list of recs (handle or RPNMeta)
#TODO: consistant naming in doc for; rec, data , meta, grid...
#TODO: class RPNFields (a collection of related rec: levels,#...)
#TODO: expand RPNGrid to accept multi-grids and work better with #-grids
#TODO: convert to/from NetCDF

"""Module RPNd contains the classes used to access RPN Standard Files (rev 2000)

    class RPNFile    : a RPN standard file
    class RPNRec     : a RPN standard file rec data (numpy.ndarray)) & meta (RPNMeta)
    class RPNGrid    : a RPN standard file grid Description, parameters (RPNParm) and axis data/meta (RPNRec)
    class RPNMeta    : RPN standard file rec metadata
    class RPNDate    : RPN STD Date representation; RPNDate(DATESTAMP) or RPNDate(YYYYMMDD,HHMMSShh)
    class RPNDateRange: Range of RPNData - DateStart,DateEnd,Delta

    class RPNKeys    : search tags (nom, type, etiket, date, ip1, ip2, ip3)
    class RPNDesc    : auxiliary tags (grtyp, ig1, ig2, ig3, ig4,  dateo, deet, npas, datyp, nbits)

    @author: Mario Lepine <mario.lepine@ec.gc.ca>
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import rpn_version
from rpn_helpers import *
#from jim import *
#import scrip

#import sys
import types
import datetime
import pytz
import numpy
#from cStringIO import StringIO
import Fstdc


FILE_MODE_RO=Fstdc.FSTDC_FILE_RO
FILE_MODE_RW=Fstdc.FSTDC_FILE_RW
FILE_MODE_RW_OLD=Fstdc.FSTDC_FILE_RW_OLD


class RPNFile:
    """Python Class implementation of the RPN standard file interface
    instanciating this class actually opens the file
    deleting the instance close the file

    myRPNFile = RPNFile(name,mode)
    @param name file name (string)
    @param mode Type of file (string,optional), FILE_MODE_RO, FILE_MODE_RW, FILE_MODE_RW_OLD

    @exception TypeError if name is not
    @exception IOError if unable to open file

    Examples of use:

    myRPNFile = RPNFile(name,mode)       #opens the file
    params = myRPNFile.info(seachParams) #get matching record params
    params = myRPNFile.info(FirstRecord) #get params of first rec on file
    params = myRPNFile.info(NextMatch)   #get next matching record params
    myRPNRec = myRPNFile[seachParams]    #get matching record data and params
    myRPNRec = myRPNFile[FirstRecord]    #get data and params of first rec on file
    myRPNRec = myRPNFile[NextMatch]      #get next matching record data and params
    myRPNFile[params]   = mydataarray    #append data and tags to file
    myRPNFile[myRPNRec] = myRPNRec.d     #append data and tags to file
    myRPNFile.write(myRPNRec)            #append data and tags to file
    myRPNFile.write(myRPNRec,rewrite=True) #rewrite data and tags to file
    myRPNFile.rewrite(myRPNRec)            #rewrite data and tags to file
    myRPNFile.append(myRPNRec)             #append data and tags to file

    myRPNFile[myRPNRec] = None           #erase record
    myRPNFile[params.handle] = None      #erase record
    del myRPNFile                        #close the file

    """
    def __init__(self,name=None,mode=FILE_MODE_RW) :
        if (not name) or type(name) <> type(''):
            raise TypeError,'RPNFile, need to provide a name for the file'
        self.filename=name
        self.lastread=None
        self.lastwrite=None
        self.options=mode
        #TODO: catch stdout/stderr - not catching c stderr/stdout!
        #stderrout = (sys.stderr,sys.stdout)
        #sys.stdout = sys.stderr = StringIO()
        #TODO: if FILE_MODE_RO or FILE_MODE_RW_OLD: check is file exist and is readable
        #TODO: if FILE_MODE_RW_OLD or FILE_MODE_RW_OLD: if file exist, check if can write, if not check if dir can write

        # Set self.iun to None before the call to fstouv, so that the member exists for
        # the __del__ if fstouv raises an exception
        self.iun = None

        self.iun = Fstdc.fstouv(0,self.filename,self.options)
        #(sys.stderr,sys.stdout) = stderrout
        if (self.iun == None):
          raise IOError,(-1,'failed to open standard file',self.filename)
        #else:
          #print 'R.P.N. Standard File (2000) ',name,' is open with options:',mode,' UNIT=',self.iun

    def voir(self,options='NEWSTYLE'):
        """Print the file content listing"""
        Fstdc.fstvoi(self.iun,options)

    def close(self):
        if (self.iun != None):
          Fstdc.fstfrm(self.iun)
          #print 'file ',self.iun,' is closed, filename=',self.filename
          self.iun = None

    def rewind(self):
        if (self.iun != None):
          Fstdc.fstrwd(self.iun)

    def __del__(self):
        """Close File"""
        self.close()
        del self.filename
        del self.lastread
        del self.lastwrite
        del self.options
        del self.iun

    def __getitem__(self,key):
        """Get the record, meta and data (RPNRec), corresponding to the seach keys from file

        myrec = myRPNfile[mykey]
        @param mykey search keys for RPNFile.info()
        @return instance of RPNRec with data and meta of the record; None if rec not found
        """
        params = self.info(key)         # 1 - get handle
        if params == None:              # oops !! not found
            return None
        target = params.handle
        array=Fstdc.fstluk(target)   # 2 - get data
        #TODO: make ni,nj,nk consistent?
        #TODO: update self.grid?
        return RPNRec(array,params)

    def edit_dir_entry(self,key):
      """Edit (zap) directory entry referenced by handle

      myRPNdfile.edit_dir_entry(myNewRPNParams)

      myNewRPNParams.handle must be a valid rec/file handle as retrieved by myRPNdfile.info()
      """
      return(Fstdc.fst_edit_dir(key.handle,key.date,key.deet,key.npas,-1,-1,-1,key.ip1,key.ip2,key.ip3,
                                key.type,key.nom,key.etiket,key.grtyp,key.ig1,key.ig2,key.ig3,key.ig4,key.datyp))

    def info(self,key,list=False):
        """Seach file for next record corresponding to search keys
        Successive calls will go further in the file.
        Search index can be reset to begining of file with myRPNfile.info(FirstRecord)
        If key.handle >=0, return key w/o search and w/o checking the file

        myRPNparms = myRPNfile.info(FirstRecord)
        myRPNparms = myRPNfile.info(mykeys)
        myRPNparms = myRPNfile.info(NextMatch)
        myRPNparms = myRPNfile.info(mykeys,list=True)
        @param mykeys search keys, can be an instance RPNParm or derived classes (RPNKeys, RPNDesc, RPNMeta, RPNRec)
        @param list if true, return a list of all rec RPNMeta matching the search keys (handle is then ignored)
        @return a RPNMeta instance of the record with proper handle, return None if not found
        @exception TypeError if

        Accepted seach keys: nom,type,
                              etiket,ip1,ip2,ip3,datev,handle
        TODO: extend accepted seach keys to all RPNMeta keys

        The myRPNfile.lastread parameter is set with values of all latest found rec params
        """
        if isinstance(key,RPNKeys):# RPNMeta is derived from RPNKeys
            if list:
                mylist = Fstdc.fstinl(self.iun,key.nom,key.type,
                              key.etiket,key.ip1,key.ip2,key.ip3,
                              key.datev)
                mylist2 = []
                for item in mylist:
                    result=RPNMeta()
                    result.update_by_dict(item)
                    result.fileref=self
                    mylist2.append(result)
                self.lastread=mylist[-1]
                return mylist2
            elif key.nxt == 1:               # get NEXT one thatmatches
                self.lastread=Fstdc.fstinf(self.iun,key.nom,key.type,
                              key.etiket,key.ip1,key.ip2,key.ip3,
                              key.datev,key.handle)
            else:                          # get FIRST one that matches
                if key.handle >= 0 :       # handle exists, return it
                    return key #TODO: may want to check if key.handle is valid
                self.lastread=Fstdc.fstinf(self.iun,key.nom,key.type,
                              key.etiket,key.ip1,key.ip2,key.ip3,
                              key.datev,-2)
        elif key==NextMatch:               # fstsui, return FstHandle instance
            self.lastread=Fstdc.fstinf(self.iun,' ',' ',' ',0,0,0,0,-1)
        else:
            raise TypeError,'RPNFile.info(), search keys arg is not of a valid type'
        result=RPNMeta()
        if self.lastread != None:
#            self.lastread.__dict__['fileref']=self
            result.update_by_dict(self.lastread)
            result.fileref=self
#            print 'DEBUG result=',result
        else:
            return None
        return result # return handle

    def erase(self,index):
        """Erase data and tags of rec in RPN STD file

        myRPNfile.erase(myRPNparms)
        myRPNfile.erase(myRPNrec)
        myRPNfile.erase(myRPNrec.handle)

        @param myRPNparms       values of rec parameters, must be a RPNMeta instance (or derived class)
        @param myRPNrec.handle  file handle of the rec to erase (int)
        @exception TypeError if myRPNrec.handle is not of accepted type
        @exception ValueError if invalid record handle is provided (or not found from myRPNparms)
        """
        if (isinstance(index,RPNDesc)): # set of keys
            meta = index
            if index.handle < 0:
                meta = self.info(index)
            target = meta.handle
        elif type(index) == type(0):  # handle
            target = index
        else:
            raise TypeError, 'RPNFile: index must provide a valid handle to erase a record'
        if meta.handle >= 0:
            #print 'erasing record with handle=',target,' from file'
            self.lastwrite=Fstdc.fsteff(target)
        else:
            raise ValueError, 'RPNFile: invalid record handle'


    def __setitem__(self,index,value):
        """[re]write data and tags of rec in RPN STD file

        myRPNfile.info[myRPNparms] = mydataarray
        myRPNfile.info[myRPNrec]   = myRPNrec.d
        myRPNfile.info[myRPNrec]   = None #erase the record corresponding to myRPNrec.handle
        myRPNfile.info[myRPNrec.handle] = None #erase the record corresponding to handle

        @param myRPNparms  values of rec parameters, must be a RPNMeta instance (or derived class)
        @param mydataarray data to be written, must be numpy.ndarray instance
        @exception TypeError if args are of wrong type
        @exception TypeError if params.handle is not valid when erasing (value=None)
        @exception ValueError if not able to find a rec corresponding to metaparams when erasing (value=None)
        """
        if value == None:
            self.erase(index)
        elif isinstance(index,RPNMeta) and type(value) == numpy.ndarray:
            self.lastwrite=0
            #print 'writing data',value.shape,' to file, keys=',index
            #print 'dict = ',index.__dict__
            if (value.flags.farray):
              #print 'fstecr Fortran style array'
              Fstdc.fstecr(value,
                         self.iun,index.nom,index.type,index.etiket,index.ip1,index.ip2,
                         index.ip3,index.dateo,index.grtyp,index.ig1,index.ig2,index.ig3,
                         index.ig4,index.deet,index.npas,index.nbits,index.datyp)
            else:
              #print 'fstecr C style array'
              Fstdc.fstecr(numpy.reshape(numpy.transpose(value),value.shape),
                         self.iun,index.nom,index.type,index.etiket,index.ip1,index.ip2,
                         index.ip3,index.dateo,index.grtyp,index.ig1,index.ig2,index.ig3,
                         index.ig4,index.deet,index.npas,index.nbits,index.datyp)
        else:
           raise TypeError,'RPNFile write: value must be an array and index must be RPNMeta or RPNRec'

    def write(self,data,meta=None,rewrite=False):
        """Write a RPNRec to the file

        myRPNRec.write(myRPNRec)
        myRPNRec.write(myArray,myRPNMeta)
        myRPNRec.write(myRPNRec,rewrite=false)
        myRPNRec.write(myArray,myRPNMeta,rewrite=true)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        meta2=meta
        data2=data
        if meta == None and isinstance(data,RPNRec):
            meta2 = data
            data2 = data.d
        elif not(isinstance(meta,RPNMeta) and type(data) == numpy.ndarray):
            raise TypeError,'RPNFile write: value must be an array and index must be RPNMeta or RPNRec'
        if rewrite:
            try:
                self.erase(meta2)
            except:
                pass
        self.__setitem__(meta2,data2)

    def append(self,data,meta=None):
        """Append a RPNRec to the file, shortcut for write(...,rewrite=False)

        myRPNRec.append(myRPNRec)
        myRPNRec.append(myArray,myRPNMeta)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        self.write(data,meta,rewrite=False)

    def rewrite(self,data,meta=None):
        """Write a RPNRec to the file, rewrite if record handle is found and exists
        shortcut for write(...,rewrite=True)

        myRPNRec.rewrite(myRPNRec)
        myRPNRec.rewrite(myArray,myRPNMeta)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        self.write(data,meta,rewrite=True)


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
        return self.searchKeysVals().copy()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {'nom':'    ','type':'  ','etiket':'            ','datev':-1,'ip1':-1,'ip2':-1,'ip3':-1,'handle':-2,'nxt':0,'fileref':None}

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {'nom':'    ','type':'  ','etiket':'            ','datev':0,'ip1':0,'ip2':0,'ip3':0,'handle':-2,'nxt':0,'fileref':None}

class RPNDesc(RPNParm):
    """RPN standard file Auxiliary descriptors class, used when writing a record or getting descriptors from a record.
    Descriptors are:
    {'grtyp':'X','dateo':0,'deet':0,'npas':0,'ig1':0,'ig2':0,'ig3':0,'ig4':0,'datyp':-1,'nbits':0,'xaxis':None,'yaxis':None,'xyref':(None,None,None,None,None),'griddim':(None,None)}
    TODO: give examples of instanciation
    """
    def __init__(self,model=None,**args):
        RPNParm.__init__(self,model,self.allowedKeysVals(),args)

    def allowedKeysVals(self):
       """Return a dict of allowed Keys/Vals"""
       return  self.searchKeysVals().copy()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {'grtyp':' ','dateo':-1,'deet':-1,'npas':-1,'ig1':-1,'ig2':-1,'ig3':-1,'ig4':-1,'datyp':-1,'nbits':-1,'ni':-1,'nj':-1,'nk':-1}

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {'grtyp':'X','dateo':0,'deet':0,'npas':0,'ig1':0,'ig2':0,'ig3':0,'ig4':0,'datyp':-1,'nbits':16,'ni':1,'nj':1,'nk':1}


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
        'datyp':-1,
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
            elif type(model) == type({}):
                self.update(model)
            else:
                raise TypeError,'RPNMeta: cannot initialize from arg #1'
        for name in args.keys(): # and update with specified attributes
            setattr(self,name,args[name])

    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        a = RPNKeys.allowedKeysVals(self).copy()
        a.update(RPNDesc.allowedKeysVals(self).copy())
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

    def getaxis(self,axis=None):
        """Return the grid axis rec of grtyp ('Z','Y','#')

        (myRPNRecX,myRPNRecY) = myRPNMeta.getaxis()
        myRPNRecX = myRPNMeta.getaxis('X')
        myRPNRecY = myRPNMeta.getaxis(axis='Y')

        @param axis which axis to return (X, Y or None), default=None returns both axis
        @exception TypeError if RPNMeta.grtyp is not in ('Z','Y','#')
        @exception TypeError if RPNMeta.fileref is not an RPNFile
        @exception ValueError if grid descriptors records (>>,^^) are not found in RPNMeta.fileref
        """
        if not (self.grtyp in ('Z','Y','#')):
            raise ValueError,'getaxis error: can not get axis from grtyp='+self.grtyp
        if not isinstance(self.fileref,RPNFile):
            raise TypeError,'RPNMeta.getaxis: ERROR - cannot get axis, no fileRef'
        searchkeys = RPNKeys(ip1=self.ig1,ip2=self.ig2)
        if self.grtyp != '#':
            searchkeys.update_by_dict({'ip3':self.ig3})
        searchkeys.nom = '>>'
        xaxisrec = self.fileref[searchkeys]
        searchkeys.nom = '^^'
        yaxisrec = self.fileref[searchkeys]
        if (xaxisrec == None or yaxisrec == None): # csubich -- variable name typo
            raise ValueError,'RPNMeta.getaxis: ERROR - axis grid descriptors (>>,^^) not found'
        if type(axis) == type(' '):
            if axis.upper() == 'X':
                return xaxisrec
            elif axis.upper() == 'Y':
                return yaxisrec
                #axisdata=self.yaxis.ravel()
        return (xaxisrec,yaxisrec)

    def __repr__(self):
        return 'RPNMeta'+repr(self.__dict__)


class RPNGrid(RPNParm):
    """RPNSTD-type grid description

    >>> g = RPNGrid(grtyp='N',ig14=(1,2,3,4),shape=(4,1))
    >>> (g.grtyp,g.shape,g.ig14)
    ('N', (4, 1), (1, 2, 3, 4))

    myRPNGrid = RPNGrid(grtyp='Z',xyaxis=(myRPNRecX,myRPNRecY))
    myRPNGrid = RPNGrid(grtyp='#',shape=(200,150),xyaxis=(myRPNRecX,myRPNRecY))
    myRPNGrid = RPNGrid(myRPNRec)
    myRPNGrid = RPNGrid(keys=myRPNRec)
    myRPNGrid = RPNGrid(myRPNRec,xyaxis=(myRPNRecX,myRPNRecY))

    @param keys
    @param grtyp
    @param shape
    @param ig14
    @param xyaxis
    @exception ValueError
    @exception TypeError

    >>> g = RPNGrid(grtyp='N',ig14=(1,2,3,4),shape=(200,150))
    >>> (g.grtyp,g.shape,g.ig14)
    ('N', (200, 150), (1, 2, 3, 4))
    >>> g2 = RPNGrid(g)
    >>> (g2.grtyp,g2.shape,g2.ig14)
    ('N', (200, 150), (1, 2, 3, 4))
    >>> d = RPNMeta(grtyp='N',ig1=1,ig2=2,ig3=3,ig4=4,ni=200,nj=150)
    >>> g3 = RPNGrid(d)
    >>> (g3.grtyp,g3.shape,g3.ig14)
    ('N', (200, 150), (1, 2, 3, 4))

    Icosahedral Grid prototype:
    grtyp = I
    ig1 = griddiv
    ig2 = grid tile (1-10) 2d, 0=NP,SP,allpoints (1D vect)
    ig3,ig4

    #(I) would work much the same way as #(L) ?

    """
    base_grtyp = ('A','B','E','G','L','N','S')
    ref_grtyp  = ('Z','Y','#')
    helper     = None

    def allowedKeysVals(self):
        """
        """
        a = {
            'grtyp': ' ',
            'ig14':(0,0,0,0),
            'shape':(0,0)
        }
        try:
            if self.helper:
                a.update(self.helper.addAllowedKeysVals.copy())
        except:
            pass
        return a

    def getGrtyp(self,keys,args=None):
        """Return grid type from class init args"""
        grtyp = ''
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError,'RPNGrid: args should be of type dict'
        if keys:
            if 'grtyp' in args.keys():
                raise ValueError, 'RPNGrid: cannot provide both keys and grtyp: '+repr(args)
            elif isinstance(keys,RPNMeta) or isinstance(keys,RPNGrid):
                grtyp = keys.grtyp
            elif type(keys) == type({}):
                if 'grtyp' in keys.keys():
                    grtyp = keys['grtyp']
                else:
                    raise ValueError,'RPNGrid: not able to find grtyp'
            else:
                raise TypeError, 'RPNGrid: wrong type for keys'
        else:
            if 'grtyp' in args.keys():
                grtyp = args['grtyp']
            else:
                raise ValueError,'RPNGrid: not able to find grtyp'
        return grtyp

    def parseArgs(self,keys,args=None):
        """Return a dict with parsed args for the specified grid type"""
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError,'RPNGrid: args should be of type dict'
        kv = self.allowedKeysVals().copy()
        if keys:
            if 'grtyp' in args.keys():
                raise ValueError, 'RPNGrid: cannot provide both keys and grtyp: '+repr(args)
            elif isinstance(keys,RPNMeta):
                kv['grtyp'] = keys.grtyp
                kv['ig14']  = (keys.ig1,keys.ig2,keys.ig3,keys.ig4)
                kv['shape'] = (keys.ni,keys.nj)
                if 'grid' in keys.__dict__.keys() and isinstance(keys.grid,RPNRec):
                    kv.update(keys.grid.__dict__)
            elif isinstance(keys,RPNGrid):
                kv.update(keys.__dict__)
        kv.update(args)
        if self.helper:
            kv.update(self.helper.parseArgs(keys,args))
        self.argsCheck(kv)
        return kv

    def argsCheck(self,d):
        """Check Grid params, raise an error if not ok"""
        grtyp = d['grtyp']
        shape = d['shape']
        ig14  = d['ig14']
        if not (type(grtyp) == type('')
            and type(shape) in (type([]),type(())) and len(shape)==2
            and (type(shape[0]) == type(shape[1]) == type(0))
            and type(ig14) in (type([]),type(())) and len(ig14)==4 \
            and (type(ig14[0]) == type(ig14[1]) == type(ig14[2]) == type(ig14[3]) == type(0)) ):
            raise ValueError, 'RPNGrid: invalid arg value'
        if self.helper:
            self.helper.argsCheck(d)

    def __init__(self,keys=None,**args):
        grtyp = self.getGrtyp(keys,args)
        if type(grtyp) == type(' '):
            grtyp = grtyp.upper()
        className = "RPNGrid%s" % grtyp
        if grtyp in self.base_grtyp:
            className = "RPNGridBase"
        elif grtyp in self.ref_grtyp:
            className = "RPNGridRef"
        try:
            myClass = globals()[className]
            self.__dict__['helper'] = myClass()
        except:
            pass
        if not self.helper:
            raise ValueError,'RPNGrid: unrecognize grtyp '+repr(grtyp)
        RPNParm.__init__(self,None,self.allowedKeysVals().copy(),{})
        self.update(self.parseArgs(keys,args))

    def interpol(self,fromData,fromGrid=None):
        """Interpolate (scalar) some gridded data to grid

        destData = myDestGrid.interpol(fromData,fromGrid)
        destRec  = myDestGrid.interpol(fromRec)

        Short for of calling:
        destData = myDestGrid.interpolVect(fromData,None,fromGrid)
        destRec  = myDestGrid.interpolVect(fromRec)

        See RPNGrid.interpolVect() methode for documentation
        """
        return self.interpolVect(fromData,None,fromGrid)

    #@static
    def interpolVectValidateArgs(self,fromDataX,fromDataY=None,fromGrid=None):
        """Check args for InterpolVect
        """
        recx  = None
        recy  = None
        isRec = False
        if (type(fromDataX) == numpy.ndarray
            and isinstance(fromGrid,RPNGrid)):
            recx   = RPNRec()
            recx.d = fromDataX
            try:
                recx.setGrid(fromGrid)
            except:
                raise ValueError, 'RPNGrid.interpolVect: fromGrid incompatible with fromDataX'
            if (type(fromDataY) == numpy.ndarray):
                recy = RPNRec()
                recy.d = fromDataY
                try:
                    recy.setGrid(fromGrid)
                except:
                    raise ValueError, 'RPNGrid.interpolVect: fromGrid incompatible with fromDataY'
            elif fromDataY:
                raise TypeError, 'RPNGrid.interpolVect: fromDataY should be of same type as fromDataX'
        elif isinstance(fromDataX,RPNRec):
            isRec = True
            if fromGrid:
                raise TypeError, 'RPNGrid.interpolVect: cannot provide both an RPNRec for fromDataX and fromGrid'
            recx = fromDataX
            recx.setGrid()
            if isinstance(fromDataY,RPNRec):
                recy = fromDataY
                recy.setGrid()
                if recx.grid != recy.grid:
                    raise ValueError, 'RPNGrid.interpolVect: data X and Y should be on the same grid'
            elif fromDataY:
                raise TypeError, 'RPNGrid.interpolVect: fromDataY should be of same type as fromDataX'
        else:
            raise TypeError, 'RPNGrid.interpolVect: data should be of numpy.ndarray or RPNRec type'
        isVect = (recy!=None)
        return (recx,recy,isVect,isRec)

    def getEzInterpArgs(self,isSrc):
        """Return the list of needed args for Fstdc.ezinterp (use helper)"""
        #TODO: may want to check helper, helper.getEzInterpArgs
        return self.helper.getEzInterpArgs(self.__dict__,isSrc)

##     def toScripGridPreComp(self,name=None):
##         """Return a Scrip grid instance for Precomputed addr&weights (use helper)"""
##         return self.helper.toScripGridPreComp(self.__dict__,name)

##     def toScripGrid(self,name=None):
##         """Return a Scrip grid instance (use helper)"""
##         if not('scripGrid' in self.__dict__.keys() and self.scripGrid):
##             self.__dict__['scripGrid'] = self.helper.toScripGrid(self.__dict__,name)
##         return self.scripGrid

##     def reshapeDataForScrip(self,data):
##         """Return reformated data suitable for SCRIP (use helper)"""
##         return self.helper.reshapeDataForScrip(self.__dict__,data)

##     def reshapeDataFromScrip(self,data):
##         """Inverse operation of reshapeDataForScrip (use helper)"""
##         return self.helper.reshapeDataFromScrip(self.__dict__,data)

    def interpolVect(self,fromDataX,fromDataY=None,fromGrid=None):
        """Interpolate some gridded scalar/vectorial data to grid
        """
        (recx,recy,isVect,isRec) = self.interpolVectValidateArgs(fromDataX,fromDataY,fromGrid)
        recyd = None
        if isVect:
            recyd = recy.d
        (sg,dg) = (recx.grid,self)
        dataxy = (None,None)
        sg_a = sg.getEzInterpArgs(isSrc=True)
        dg_a = dg.getEzInterpArgs(isSrc=False)
        if sg_a and dg_a:
            a = RPNGridHelper.baseEzInterpArgs.copy()
            a.update(sg_a)
            sg_a = a
            a = RPNGridHelper.baseEzInterpArgs.copy()
            a.update(dg_a)
            dg_a = a
            dataxy = Fstdc.ezinterp(recx.d,recyd,
                    sg_a['shape'],sg_a['grtyp'],sg_a['g_ig14'],sg_a['xy_ref'],sg_a['hasRef'],sg_a['ij0'],
                    dg_a['shape'],dg_a['grtyp'],dg_a['g_ig14'],dg_a['xy_ref'],dg_a['hasRef'],dg_a['ij0'],
                    isVect)
            if isVect==0:
                dataxy = (dataxy,None)
        else:
            raise TypeError, 'RPNGrid.interpolVect: Cannot perform interpolation between specified grids type'
##             #if verbose: print "using SCRIP"
##             if isVect!=1:
##                 #TODO: remove this exception when SCRIP.interp() does vectorial interp
##                 raise TypeError, 'RPNGrid.interpolVect: SCRIP.interp() Cannot perform vectorial interpolation yet!'
##             #Try to interpolate with previously computed addr&weights (if any)
##             sg_a = sg.toScripGridPreComp()
##             dg_a = dg.toScripGridPreComp()
##             try:
##                 scripObj = scrip.Scrip(sg_a,dg_a)
##             except:
##                 #Try while computing lat/lon and addr&weights
##                 sg_a = sg.toScripGrid()
##                 dg_a = dg.toScripGrid()
##             if sg_a and dg_a:
##                 scripObj = scrip.Scrip(sg_a,dg_a)
##                 datax  = sg.reshapeDataForScrip(recx.d)
##                 datax2 = scrip.scripObj.interp(datax)
##                 dataxy = (dg.reshapeDataFromScrip(datax2),None)
##             else:
##                 raise TypeError, 'RPNGrid.interpolVect: Cannot perform interpolation between specified grids type'
        if isRec:
            recx.d = dataxy[0]
            recx.setGrid(self)
            if isVect:
                recy.d = dataxy[1]
                recy.setGrid(self)
                return (recx,recy)
            else:
                return recx
        else:
            if isVect:
                return (dataxy[0],dataxy[1])
            else:
                return dataxy[0]

    def __repr__(self):
        return 'RPNGrid'+repr(self.__dict__)


class RPNGridBase(RPNGridHelper):
    """RPNGrid Helper class for RPNSTD-type grid description for basic projections
    """
    def parseArgs(self,keys,args):
        """Return a dict with parsed args for the specified grid type"""
        return {} #TODO: accept real values (xg14)

    #@static
    def argsCheck(self,d):
        """Check Grid params, raise an error if not ok"""
        if not (d['grtyp'] in RPNGrid.base_grtyp):
            raise ValueError, 'RPNGridBase: invalid grtyp value:'+repr(d['grtyp'])

    def getEzInterpArgs(self,keyVals,isSrc):
        """Return the list of needed args for Fstdc.ezinterp from the provided params"""
        a = RPNGridHelper.baseEzInterpArgs.copy()
        a['shape']  = keyVals['shape']
        a['grtyp']  = keyVals['grtyp']
        a['g_ig14'] = list(keyVals['ig14'])
        a['g_ig14'].insert(0,keyVals['grtyp'])
        return a

##     def toScripGrid(self,keyVals,name=None):
##         """Return a Scrip grid instance for the specified grid type"""
##         sg_a = self.getEzInterpArgs(keyVals,False)
##         doCorners = 1
##         (la,lo,cla,clo) = Fstdc.ezgetlalo(sg_a['shape'],sg_a['grtyp'],sg_a['g_ig14'],sg_a['xy_ref'],sg_a['hasRef'],sg_a['ij0'],doCorners)
##         if name is None:
##             name = self.toScripGridName(keyVals)
##         la  *= (numpy.pi/180.)
##         lo  *= (numpy.pi/180.)
##         cla *= (numpy.pi/180.)
##         clo *= (numpy.pi/180.)
##         return scrip.ScripGrid(name,(la,lo,cla,clo))


class RPNGridRef(RPNGridHelper):
    """RPNGrid Helper class for RPNSTD-type grid description for grid reference
    Preferably use the generic RPNGrid class to indirectly get an instance
    """
    addAllowedKeysVals = {
        'xyaxis':(None,None),
        'g_ref':None
    }

    def parseArgs(self,keys,args):
        """Return a dict with parsed args for the specified grid type"""
        kv = {}
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError,'RPNGridRef: args should be of type dict'
        #if keys is another grid or FstRec w/ grid
        #   or xyaxis was present in keys,args
        #   then it would already have been included
        #   only need to get xyaxis and g_ref from keys
        #   TODO: may not want to do this if they were provided in args or keys.grid
        if keys and isinstance(keys,RPNMeta):
            kv['xyaxis'] = keys.getaxis() # csubich -- proper spelling is without capital
            kv['g_ref']  = RPNGrid(kv['xyaxis'][0])
        for k in self.addAllowedKeysVals:
            if k in args.keys():
                kv[k] = args[k]
        return kv

    def argsCheck(self,d):
        """Check Grid params, raise an error if not ok"""
        xyaxis = d['xyaxis']
        if not (d['grtyp'] in RPNGrid.ref_grtyp
            and type(xyaxis) in (type([]),type(()))
            and len(xyaxis)==2
            and isinstance(xyaxis[0],RPNRec)
            and isinstance(xyaxis[1],RPNRec) ):
            raise ValueError, 'RPNGridRef: invalid value'

    def getEzInterpArgs(self,keyVals,isSrc):
        """Return the list of needed args for Fstdc.ezinterp from the provided params"""
        if keyVals['grtyp'] == 'Y' and isSrc:
            return None
        a = keyVals['g_ref'].getEzInterpArgs(isSrc)
        if a is None:
            return None
        a['grtyp']  = keyVals['grtyp']
        a['xy_ref'] = (keyVals['xyaxis'][0].d,keyVals['xyaxis'][1].d)
        a['hasRef'] = 1
        a['ij0'] = (1,1)
        a['shape'] = keyVals['shape']
        if keyVals['grtyp'] == '#':
            a['ij0'] = keyVals['ig14'][2:]
        return a

##     def toScripGridName(self,keyVals):
##         """Return a hopefully unique grid name for the provided params"""
##         ij0 = (1,1)
##         if keyVals['grtyp'] == '#':
##             ij0 = keyVals['ig14'][2:]
##         name = "grd%s%s-%i-%i-%i-%i-%i-%i-%i-%i" % (
##         keyVals['grtyp'],keyVals['g_ref'].grtyp,
##         keyVals['g_ref'].ig14[0],keyVals['g_ref'].ig14[1],
##         keyVals['g_ref'].ig14[2],keyVals['g_ref'].ig14[3],
##         keyVals['shape'][0],keyVals['shape'][1],
##         ij0[0],ij0[1])
##         return name

##     def toScripGrid(self,keyVals,name=None):
##         """Return a Scrip grid instance for the specified grid type"""
##         if name is None:
##             name = self.toScripGridName(keyVals)
##         sg_a = self.getEzInterpArgs(keyVals,False)
##         if sg_a is None:
##             #TODO: make this work for # grids and other not global JIM grids
##             scripGrid = keyVals['g_ref'].toScripGrid(keyVals['g_ref'].__dict__,name)
##         else:
##             doCorners = 1
##             (la,lo,cla,clo) = Fstdc.ezgetlalo(sg_a['shape'],sg_a['grtyp'],sg_a['g_ig14'],sg_a['xy_ref'],sg_a['hasRef'],sg_a['ij0'],doCorners)
##             la  *= (numpy.pi/180.)
##             lo  *= (numpy.pi/180.)
##             cla *= (numpy.pi/180.)
##             clo *= (numpy.pi/180.)
##             scripGrid = scrip.ScripGrid(name,(la,lo,cla,clo))
##         return scripGrid


class RPNRec(RPNMeta):
    """Standard file record, with data (ndarray class) and full set of descriptors (RPNMeta class)

    Example of use (and doctest tests):

    >>> r = RPNRec()
    >>> r.d
    array([], dtype=float64)
    >>> r = RPNRec([1,2,3,4])
    >>> r.d
    array([1, 2, 3, 4])
    >>> a = numpy.array([1,2,3,4],order='FORTRAN',dtype='float32')
    >>> r = RPNRec(a)
    >>> r.d
    array([ 1.,  2.,  3.,  4.], dtype=float32)
    >>> a[1] = 5
    >>> r.d #r.d is a reference to a, thus changing a changes a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r = RPNRec(a.copy(),RPNMeta(grtyp='X'))
    >>> r.d
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> a[1] = 9
    >>> r.d #r.d is a copy of a, thus changing a does not change a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r.grtyp
    'X'
    >>> r = RPNRec([1,2,3,4])
    >>> r2 = RPNRec(r)
    >>> d = r2.__dict__.items()
    >>> d.sort()
    >>> d
    [('d', array([1, 2, 3, 4])), ('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grid', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', -1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', '    '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> r.d[1] = 9 #r2 is a copy of r, thus this does not change r2.d
    >>> r2.d
    array([1, 2, 3, 4])

    @param data data part of the rec, can be a python list, numpy.ndarray or another RPNRec
    @param meta meta part of the record (RPNMeta), if data is an RPNRec it should not be provided
    @exceptions TypeError if arguments are not of valid type
    """
    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        a = RPNMeta.allowedKeysVals(self).copy()
        a['d'] = None
        a['grid'] = None
        return a

    def __init__(self,data=None,meta=None):
        RPNMeta.__init__(self)
        if data == None:
            self.d = numpy.array([])
        elif type(data) == numpy.ndarray:
            self.d = data
        elif type(data) == type([]):
            self.d = numpy.array(data)
        elif isinstance(data,RPNRec):
            if meta:
                raise TypeError,'RPNRec: cannot initialize with both an RPNRec and meta'
            self.d = data.d.copy()
            meta = RPNMeta(data)
        else:
            raise TypeError,'RPNRec: cannot initialize data from arg #1'
        if meta:
            if isinstance(meta,RPNMeta):
                self.update(meta)
            elif type(meta) == type({}):
                self.update_by_dict(meta)
            else:
                raise TypeError,'RPNRec: cannot initialize parameters from arg #2'

    def __setattr__(self,name,value):   # this method cannot create new attributes
        if name == 'd':
            if type(value) == numpy.ndarray:
                self.__dict__[name]=value
            else:
                raise TypeError,'RPNRec: data should be an instance of numpy.ndarray'
        elif name == 'grid':
            if isinstance(value,RPNGrid):
                self.__dict__[name]=value
            else:
                raise TypeError,'RPNRec: grid should be an instance of RPNGrid'
        else:
            RPNMeta.__setattr__(self,name,value)

    def interpol(self,togrid):
        """Interpolate RPNRec to another grid (horizontally)

        myRPNRec.interpol(togrid)
        @param togrid grid where to interpolate
        @exception ValueError if myRPNRec does not contain a valid grid desc
        @exception TypeError if togrid is not an instance of RPNGrid
        """
        if isinstance(togrid,RPNGrid):
            if not isinstance(self.grid,RPNGrid):
                self.setGrid()
            if self.grid:
                self.d = togrid.interpol(self.d,self.grid)
                self.setGrid(togrid)
            else:
                raise ValueError,'RPNRec.interpol(togrid): unable to determine actual grid of RPNRec'
        else:
            raise TypeError,'RPNRec.interpol(togrid): togrid should be an instance of RPNGrid'

    def setGrid(self,newGrid=None):
        """Associate a grid to the RPNRec (or try to get grid from rec metadata)

        myRPNRec.setGrid()
        myRPNRec.setGrid(newGrid)

        @param newGrid grid to associate to the record (RPNGrid)
        @exception ValueError if newGrid does not have same shape as rec data or if it's impossible to determine grid params
        @exception TypeError if newGrid is not an RPNGrid

        >>> r = RPNRec([1,2,3,4],RPNMeta())
        >>> g = RPNGrid(grtyp='N',ig14=(1,2,3,4),shape=(4,1))
        >>> (g.grtyp,g.shape,g.ig14)
        ('N', (4, 1), (1, 2, 3, 4))
        >>> r.setGrid(g)
        >>> (r.grtyp,(r.ni,r.nj),(r.ig1,r.ig2,r.ig3,r.ig4))
        ('N', (4, 1), (1, 2, 3, 4))
        >>> (r.grid.grtyp,r.grid.shape,r.grid.ig14)
        ('N', (4, 1), (1, 2, 3, 4))
        """
        if newGrid:
            if isinstance(newGrid,RPNGrid):
                ni = max(self.d.shape[0],1)
                nj = 1
                if len(self.d.shape)>1:
                    nj = max(self.d.shape[1],1)
                if (ni,nj) != newGrid.shape:
                    raise ValueError,'RPNRec.setGrid(newGrid): rec data and newGrid do not have the same shape'
                else :
                    self.grid = newGrid
                    self.grtyp = newGrid.grtyp
                    (self.ig1,self.ig2,self.ig3,self.ig4) = newGrid.ig14
                    (self.ni,self.nj) = newGrid.shape
            else:
                raise TypeError,'RPNRec.setGrid(newGrid): newGrid should be an instance of RPNGrid'
        else:
            self.grid = RPNGrid(self)
            if self.grid:
                self.grtyp = self.grid.grtyp
                (self.ig1,self.ig2,self.ig3,self.ig4) = self.grid.ig14
                (self.ni,self.nj) = self.grid.shape
            else:
                raise ValueError,'RPNRec.setGrid(): unable to determine actual grid of RPNRec'

    def __repr__(self):
        kv = self.__dict__.copy()
        rd = repr(kv['d'])
        rg = repr(kv['grid'])
        del kv['d']
        del kv['grid']
        return 'RPNRec{meta='+repr(RPNMeta(kv))+', grid='+rg+', d='+rd+'}'


class RPNDate:
    """RPN STD Date representation

    myRPNDate = RPNDate(DATESTAMP)
    myRPNDate = RPNDate(YYYYMMDD,HHMMSShh)
    myRPNDate = RPNDate(myDateTime)
    myRPNDate = RPNDate(myRPNMeta)
    @param DATESTAMP CMC date stamp or RPNDate object
    @param YYYYMMDD  Int with Visual representation of YYYYMMDD
    @param HHMMSShh  Int with Visual representation of HHMMSShh
    @param myDateTime Instance of Python DateTime class
    @param myRPNMeta Instance RPNMeta with dateo,deet,npas properly set
    @exception TypeError if parameters are wrong type
    @exception ValueError if myRPNMeta

    >>> d1 = RPNDate(20030423,11453500)
    >>> d1
    RPNDate(20030423,11453500)
    >>> d2 = RPNDate(d1)
    >>> d2
    RPNDate(20030423,11453500)
    >>> d2.incr(48)
    RPNDate(20030425,11453500)
    >>> d1-d2
    -48.0
    >>> a = RPNMeta(dateo=d1.stamp,deet=1800,npas=3)
    >>> d3 = RPNDate(a)
    >>> d3
    RPNDate(20030423,13153500)
    >>> utc = pytz.timezone("UTC")
    >>> d4 = datetime.datetime(2003,04,23,11,45,35,0,tzinfo=utc)
    >>> d5 = RPNDate(d4)
    >>> d5
    RPNDate(20030423,11453500)
    >>> d6 = d5.toDateTime()
    >>> d6 == d4
    True
    """
    stamp = 0

    def __init__(self,word1,word2=-1):
        if isinstance(word1,datetime.datetime):
            (yyyy,mo,dd,hh,mn,ss,dummy,dummy2,dummy3) = word1.utctimetuple()
            cs = int(word1.microsecond/10000)
            word1 = yyyy*10000+mo*100+dd
            word2 = hh*1000000+mn*10000+ss*100+cs
        if isinstance(word1,RPNDate):
            self.stamp = word1.stamp
        elif isinstance(word1,RPNMeta):
            if word1.deet<0 or word1.npas<0 or word1.dateo<=0 :
                raise ValueError, 'RPNDate: Cannot compute date from RPNMeta'
            nhours = (1.*word1.deet*word1.npas)/3600.
            self.stamp=Fstdc.incdatr(word1.dateo,nhours)

        elif type(word1) == type(0):    # integer type
            if (word2 == -1):
                self.stamp = word1
            else:
                dummy=0
                (self.stamp,dummy1,dummy2) = Fstdc.newdate(dummy,word1,word2,3)
        else:
            raise TypeError, 'RPNDate: arguments should be of type int'

    def __sub__(self,other):
        "Time difference between 2 dates"
        return(Fstdc.difdatr(self.stamp,other.stamp))

    def incr(self,temps):
        """Increase Date by the specified number of hours

        @param temps Number of hours for the RPNDate to be increased
        @return self

        @exception TypeError if temps is not of int or real type
        """
        if ((type(temps) == type(1)) or (type(temps) == type(1.0))):
            nhours = 0.0
            nhours = temps
            self.stamp=Fstdc.incdatr(self.stamp,nhours)
            return(self)
        else:
            raise TypeError,'RPNDate.incr: argument should be int or real'

    def toDateTime(self):
        """Return the DateTime obj representing the RPNDate

        >>> myRPNDate = RPNDate(20030423,11453600)
        >>> myDateTime = myRPNDate.toDateTime()
        >>> myDateTime
        datetime.datetime(2003, 4, 23, 11, 45, 35, tzinfo=<UTC>)

        #TODO: oups 1 sec diff!!!
        """
        word1 = word2 = 0
        (dummy,word1,word2) = Fstdc.newdate(self.stamp,word1,word2,-3)
        d = "%8.8d.%8.8d" % (word1, word2)
        yyyy = int(d[0:4])
        mo = int(d[4:6])
        dd = int(d[6:8])
        hh = int(d[9:11])
        mn = int(d[11:13])
        ss = int(d[13:15])
        cs = int(d[15:17])
        utc = pytz.timezone("UTC")
        return datetime.datetime(yyyy,mo,dd,hh,mn,ss,cs*10000,tzinfo=utc)

    def __repr__(self):
        word1 = word2 = 0
        (dummy,word1,word2) = Fstdc.newdate(self.stamp,word1,word2,-3)
        return "RPNDate(%8.8d,%8.8d)" % (word1, word2)


class RPNDateRange:
    """RPN STD Date Range representation

    RPNDateRange(DateStart,DateEnd,Delta)
    @param DateStart RPNDate start of the range
    @param DateEnd   RPNDate end of the range
    @param Delta     Increment of the range iterator, hours, real

    @exception TypeError if parameters are wrong type

    >>> d1 = RPNDate(20030423,11453500)
    >>> d2 = RPNDate(d1)
    >>> d2.incr(48)
    RPNDate(20030425,11453500)
    >>> dr = RPNDateRange(d1,d2,6)
    >>> dr
    RPNDateRage(from:(20030423,11453500), to:(20030425,11453500), delta:6) at (20030423,11453500)
    >>> dr.lenght()
    48.0
    >>> dr.next()
    RPNDate(20030423,17453500)
    >>> dr = RPNDateRange(d1,d2,36)
    >>> dr
    RPNDateRage(from:(20030423,17453500), to:(20030425,11453500), delta:36) at (20030423,17453500)
    >>> dr.next()
    RPNDate(20030425,05453500)
    >>> dr.next() #returns None because it is past the end of DateRange
    """
    #TODO: make this an iterator
    dateDebut=-1
    dateFin=-1
    delta=0.0
    now=-1

    def __init__(self,debut=-1,fin=-1,delta=0.0):
        if isinstance(debut,RPNDate) and isinstance(fin,RPNDate) and ((type(delta) == type(1)) or (type(delta) == type(1.0))):
            self.dateDebut=debut
            self.now=debut
            self.dateFin=fin
            self.delta=delta
        else:
            raise TypeError,'RPNDateRange: arguments type error RPNDateRange(RPNDate,RPNDate,Real)'

    def lenght(self):
        """Provide the duration of the date range
        @return Number of hours
        """
        return abs(self.dateFin-self.dateDebut)

    def remains():
        """Provide the number of hours left in the date range
        @return Number of hours left in the range
        """
        return abs(self.dateFin-self.now)

    def next(self):
        """Return the next date/time in the range (step of delta hours)
        @return next RPNDate, None if next date is beyond range
        """
        self.now.incr(self.delta)
        if (self.dateFin-self.now)*self.delta < 0.:
            return None
        return RPNDate(self.now)

    def reset(self):
        """Reset the RPNDateRange iterator to the range start date"""
        self.now=self.dateDebut

    def __repr__(self):
        d1 = repr(self.dateDebut)
        d2 = repr(self.dateFin)
        d0 = repr(self.now)
        return "RPNDateRage(from:%s, to:%s, delta:%d) at %s" % (d1[7:27],d2[7:27],self.delta,d0[7:27])


FirstRecord=RPNMeta()
NextMatch=None


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
