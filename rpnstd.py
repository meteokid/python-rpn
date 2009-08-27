## Automatically adapted for numpy.oldnumeric Jan 10, 2008 by

""" module Fstd contains the classes used to access RPN Standard Files (rev 2000)

    class FstFile    : a RPN standard file
    class FstRec     : a RPN standard file rec data (numpy.ndarray)) & meta (FstParms)
    class FstParms   : combined set of tags (search and auxiliary), RPN standard file rec meta (FstKeys, FstDesc)
    class FstKeys    : search tags (nom, type, etiket, date, ip1, ip2, ip3)
    class FstDesc    : auxiliary tags (grtyp, ig1, ig2, ig3, ig4,  dateo, deet, npas, datyp, nbits)
    class FstDate    : RPN STD Date representation; FstDate(DATESTAMP) or FstDate(YYYYMMDD,HHMMSShh)
    class FstDate    : RPN STD Date Range: FstDateRange(DateStart,DateEnd,Delta)
    class FstMapDesc :
    class Grid       :

    class FstExclude :
    class FstSelect  :

    @author: Mario Lepine <mario.lepine@ec.gc.ca>
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
    @date: 2009-08
"""
import types
import numpy
import Fstdc

# primary set of descriptors has two extra items, used to read/scan file
# handle carries the last handle associated with the keys ,
# or -1 if next match, or -2 if match is to start at beginning of file
X__PrimaryDesc={'nom':'    ','type':'  ','etiket':'            ',
                'date':-1,'ip1':-1,'ip2':-1,'ip3':-1,'handle':-2,'nxt':0,'fileref':None}

# descriptive part of the keys, returned by read/scan, needed for write
X__AuxiliaryDesc={'grtyp':'X','dateo':0,'deet':0,'npas':0,
               'ig1':0,'ig2':0,'ig3':0,'ig4':0,'datyp':0,'nbits':0,
               'xaxis':None,'yaxis':None,'xyref':(None,None,None,None,None),'griddim':(None,None)}

# wild carded descriptive part of the keys (non initialized)
W__AuxiliaryDesc={'grtyp':' ','dateo':-1,'deet':-1,'npas':-1,
               'ig1':-1,'ig2':-1,'ig3':-1,'ig4':-1,'datyp':-1,'nbits':-1}

X__Criteres={'nom':['    '],'type':['  '],'etiket':['            '],
        'date':[-1],'ip1':[-1],'ip2':[-1],'ip3':[-1],
        'grtyp':[' '],'dateo':[-1],'deet':[-1],'npas':[-1],
        'ig1':[-1],'ig2':[-1],'ig3':[-1],'ig4':[-1],
        'ni':[-1],'nj':[-1],'nk':[-1],'datyp':[-1],'nbits':[-1]}

Sgrid__Desc={'nom':'    ','type':'  ','etiket':'            ','date':-1,'ip1':-1,'ip2':-1,'ip3':-1,
             'deet':0,'npas':0,'datyp':0,'nbits':0,'dateo':-1}

Tgrid__Desc={'grtyp':'X','ig1':-1,'ig2':-1,'ig3':-1,'ig4':-1,'xyref':(None,None,None,None,None),'griddim':(None,None),
             'xaxis':None,'yaxis':None}


X__FullDesc={}
X__FullDesc.update(X__PrimaryDesc)
X__FullDesc.update(X__AuxiliaryDesc)

W__FullDesc={}
W__FullDesc.update(X__PrimaryDesc)
W__FullDesc.update(W__AuxiliaryDesc)

X__DateDebut=-1
X__DateFin=-1
X__Delta=0.0

def Predef_Grids():
  """Intentiate Predefined Grid configurations as global Objects

  global Grille_Amer_Nord, Grille_Europe, Grille_Inde, Grille_Hem_Sud, Grille_Canada, Grille_Maritimes
  global Grille_Quebec, Grille_Prairies, Grille_Colombie, Grille_USA, Grille_Global, Grille_GemLam10
  """
  global Grille_Amer_Nord, Grille_Europe, Grille_Inde, Grille_Hem_Sud, Grille_Canada, Grille_Maritimes
  global Grille_Quebec, Grille_Prairies, Grille_Colombie, Grille_USA, Grille_Global, Grille_GemLam10
  Grille_Amer_Nord=Grid(grtyp='N',ninj=(401,401),ig14=cxgaig('N',200.5,200.5,40000.0,21.0))  # PS 40km
  Grille_Europe=Grid(grtyp='N',ninj=(401,401),ig14=cxgaig('N',200.5,220.5,40000.0,-100.0))   # PS 40km
  Grille_Inde=Grid(grtyp='N',ninj=(401,401),ig14=cxgaig('N',200.5,300.5,40000.0,-170.0))     # PS 40km
  Grille_Hem_Sud=Grid(grtyp='S',ninj=(401,401),ig14=cxgaig('S',200.5,200.5,40000.0,21.0))    # PS 40km
  Grille_Canada=Grid(grtyp='N',ninj=(351,261),ig14=cxgaig('N',121.5,281.5,20000.0,21.0))     # PS 20km
  Grille_Maritimes=Grid(grtyp='N',ninj=(175,121),ig14=cxgaig('N',51.5,296.5,20000.0,-20.0))  # PS 20km
  Grille_Quebec=Grid(grtyp='N',ninj=(199,155),ig14=cxgaig('N',51.5,279.5,20000.0,0.0))       # PS 20km
  Grille_Prairies=Grid(grtyp='N',ninj=(175,121),ig14=cxgaig('N',86.5,245.5,20000.0,20.0))    # PS 20km
  Grille_Colombie=Grid(grtyp='N',ninj=(175,121),ig14=cxgaig('N',103.5,245.5,20000.0,30.0))   # PS 20km
  Grille_USA=Grid(grtyp='N',ninj=(351,261),ig14=cxgaig('N',121.0,387.5,20000.0,21.0))        # PS 20km
  Grille_Global=Grid(grtyp='L',ninj=(721,359),ig14=cxgaig('L',-89.5,180.0,0.5,0.5))          # LatLon 0.5 Deg
  Grille_GemLam10=Grid(grtyp='N',ninj=(1201,776),ig14=cxgaig('N',536.0,746.0,10000.0,21.0))  # PS 10km


#def printdaterange():
    #"""Print date range defined in the module
    #"""
    #global X__DateDebut,X__DateFin,X__Delta
    #print 'Debug printdaterange debut fin delta=',X__DateDebut,X__DateFin,X__Delta

#def resetdaterange():
    #"""Reset date range defined in the module to anydate
    #"""
    #global X__DateDebut,X__DateFin,X__Delta
    #X__DateDebut=-1
    #X__DateFin=-1
    #X__Delta=0.0

def dump_keys_and_values(self):
    """Return a string with comma separated key=value of all parameters
    """
    result=''
    keynames = self.__dict__.keys()
    keynames.sort()
    for name in keynames:
        result=result+name+'='+repr(self.__dict__[name])+' , '
    return result[:-3]  # eliminate last blank comma blank sequence


LEVEL_KIND_MSL=0 #metres above sea level
LEVEL_KIND_SIG=1 #Sigma
LEVEL_KIND_PMB=2 #Pressure [mb]
LEVEL_KIND_ANY=3 #arbitrary code
LEVEL_KIND_MGL=4 #metres above ground level
LEVEL_KIND_HYB=5 #hybrid coordinates [hy]
LEVEL_KIND_TH=6 #theta [th]

def levels_to_ip1(levels,kind):
    """Encode level value into ip1 for the specified kind

    ip1_list = levels_to_ip1(level_list,kind)
    @param level_list list of level values [units depending on kind]
    @param kind   type of levels [units] to be encoded
        kind = 0: levels are in height [m] (metres) with respect to sea level
        kind = 1: levels are in sigma [sg] (0.0 -> 1.0)
        kind = 2: levels are in pressure [mb] (millibars)
        Looks like the following are not suppored yet in the fortran func convip
            kind = 3: levels are in arbitrary code
            kind = 4: levels are in height [M] (metres) with respect to ground level
            kind = 5: levels are in hybrid coordinates [hy]
            kind = 6: levels are in theta [th]
    @return list of encoded level values-tuple ((ip1new,ip1old),...)

    Example of use (and doctest tests):

    >>> levels_to_ip1([0.,13.5,1500.,5525.,12750.],0)
    [(15728640, 12001), (8523608, 12004), (6441456, 12301), (6843956, 13106), (5370380, 14551)]
    >>> levels_to_ip1([0.,0.1,.02,0.00678,0.000003],1)
    [(32505856, 2000), (27362976, 3000), (28511552, 2200), (30038128, 2068), (32805856, 2000)]
    >>> levels_to_ip1([1024.,850.,650.,500.,10.,2.,0.3],2)
    [(39948288, 1024), (41744464, 850), (41544464, 650), (41394464, 500), (42043040, 10), (43191616, 1840), (44340192, 1660)]
    """
    if not type(levels) in (type(()),type([])):
        raise ValueError,'levels_to_ip1: levels should be a list or a tuple; '+levels.__repr__()
    if type(kind) <> type(0):
        raise TypeError,'levels_to_ip1: kind should be an int in range [0,6]; '+kind.__repr__()
    elif not kind in (0,1,2): #(0,1,2,3,4,5,6):
        raise ValueError,'levels_to_ip1: kind should be an int in range [0,6]; '+kind.__repr__()
    if type(levels) == type(()):
        ip1_list = Fstdc.level_to_ip1(list(levels),kind)
    else:
        ip1_list = Fstdc.level_to_ip1(levels,kind)
    if not ip1_list:
        raise TypeError,'levels_to_ip1: wrong args; levels_to_ip1(levels,kind)'
    return(ip1_list)


def ip1_to_levels(ip1list):
    """Decode ip1 value into (level,kind)

    levels_list = ip1_to_levels(ip1list)
    @param ip1list list of ip1 values to decode
    @return list of decoded level values-tuple ((level_list,kind),...)
        kind = 0: levels are in height [m] (metres) with respect to sea level
        kind = 1: levels are in sigma [sg] (0.0 -> 1.0)
        kind = 2: levels are in pressure [mb] (millibars)
        kind = 3: levels are in arbitrary code
        kind = 4: levels are in height [M] (metres) with respect to ground level
        kind = 5: levels are in hybrid coordinates [hy]
        kind = 6: levels are in theta [th]

    Example of use (and doctest tests):

    >>> ip1_to_levels([0,1,1000,1199,1200,1201,9999,12000,12001,12002,13000])
    [(0.0, 2), (1.0, 2), (1000.0, 2), (1.0, 3), (0.0, 3), (4.9999998736893758e-05, 2), (0.79989999532699585, 1), (1.0, 1), (0.0, 0), (5.0, 0), (4995.0, 0)]
    >>> ip1_to_levels([15728640, 12001,8523608, 12004,6441456, 12301,6843956, 13106,5370380, 14551])
    [(0.0, 0), (0.0, 0), (13.5, 0), (15.0, 0), (1500.0, 0), (1500.0, 0), (5525.0, 0), (5525.0, 0), (12750.0, 0), (12750.0, 0)]
    >>> ip1_to_levels([32505856, 2000,27362976, 3000,28511552, 2200,30038128, 2068,32805856, 2000])
    [(0.0, 1), (0.0, 1), (0.10000000149011612, 1), (0.099999994039535522, 1), (0.019999999552965164, 1), (0.019999999552965164, 1), (0.0067799999378621578, 1), (0.0067999996244907379, 1), (3.0000001061125658e-06, 1), (0.0, 1)]
    >>> ip1_to_levels([39948288, 1024,41744464, 850,41544464, 650,41394464, 500,42043040, 10,43191616, 1840,44340192, 1660])
    [(1024.0, 2), (1024.0, 2), (850.0, 2), (850.0, 2), (650.0, 2), (650.0, 2), (500.0, 2), (500.0, 2), (10.0, 2), (10.0, 2), (2.0, 2), (2.0, 2), (0.30000001192092896, 2), (0.30000001192092896, 2)]
    """
    if not type(ip1list) in (type(()),type([])):
        raise ValueError,'ip1_to_levels: levels should be a list or a tuple'

    if type(ip1list) == type(()):
        levels = Fstdc.ip1_to_level(list(ip1list))
    else:
        levels = Fstdc.ip1_to_level(ip1list)
    if not levels:
        raise TypeError,'ip1_to_levels: wrong args; ip1_to_levels(ip1list)'
    return(levels)


def cxgaig(grtyp,xg1,xg2=None,xg3=None,xg4=None):
    """Encode grid definition values into ig1-4 for the specified grid type

    (ip1,ip2,ip3,ip4) = cxgaig(grtyp,xg1,xg2,xg3,xg4):
    (ip1,ip2,ip3,ip4) = cxgaig(grtyp,(xg1,xg2,xg3,xg4)):

    Example of use (and doctest tests):

    >>> cxgaig('N',200.5, 200.5, 40000.0, 21.0)
    (2005, 2005, 2100, 400)
    >>> cxgaig('N',200.5, 220.5, 40000.0, 260.0)
    (400, 1000, 29830, 57333)
    >>> cxgaig('S',200.5, 200.5, 40000.0, 21.0)
    (2005, 2005, 2100, 400)
    >>> cxgaig('L',-89.5, 180.0, 0.5, 0.5)
    (50, 50, 50, 18000)
    >>> ig1234 = (-89.5, 180.0, 0.5, 0.5)
    >>> cxgaig('L',ig1234)
    (50, 50, 50, 18000)

    Example of bad use (and doctest tests):

    >>> cxgaig('L',-89.5, 180  , 0.5, 0.5)
    Traceback (most recent call last):
    ...
    TypeError: cxgaig error: ig1,ig2,ig3,ig4 should be of type real:(-89.5, 180, 0.5, 0.5)
    >>> cxgaig('I',-89.5, 180.0, 0.5, 0.5)
    Traceback (most recent call last):
    ...
    ValueError: cxgaig error: grtyp ['I'] must be one of ('A', 'B', 'E', 'G', 'L', 'N', 'S')
    """
    validgrtyp = ('A','B','E','G','L','N','S')
    if xg2 == xg3 == xg4 == None and type(xg1) in (type([]),type(())) and len(xg1) == 4:
        (xg1,xg2,xg3,xg4) = xg1
    if None in (grtyp,xg1,xg2,xg3,xg4):
        raise TypeError,'cxgaig error: missing argument, calling is cxgaig(grtyp,xg1,xg2,xg3,xg4)'
    elif not grtyp in validgrtyp:
        raise ValueError,'cxgaig error: grtyp ['+grtyp.__repr__()+'] must be one of '+validgrtyp.__repr__()
    elif not (type(xg1) == type(xg2) == type(xg3) == type(xg4) == type(0.)):
        raise TypeError,'cxgaig error: ig1,ig2,ig3,ig4 should be of type real:'+(xg1,xg2,xg3,xg4).__repr__()
    else:
       return(Fstdc.cxgaig(grtyp,xg1,xg2,xg3,xg4))


def cigaxg(grtyp,ig1,ig2=None,ig3=None,ig4=None):
    """Decode grid definition values into xg1-4 for the specified grid type

    (xp1,xp2,xp3,xp4) = cigaxg(grtyp,ig1,ig2,ig3,ig4):
    (xp1,xp2,xp3,xp4) = cigaxg(grtyp,(ig1,ig2,ig3,ig4)):

    Example of use (and doctest tests):

    >>> cigaxg('N',2005,  2005,  2100,   400)
    (200.5, 200.5, 40000.0, 21.0)
    >>> cigaxg('N',400,  1000, 29830, 57333)
    (200.50123596191406, 220.49647521972656, 40000.0, 260.0)
    >>> cigaxg('S',2005,  2005,  2100,   400)
    (200.5, 200.5, 40000.0, 21.0)
    >>> cigaxg('L',50,    50,    50, 18000)
    (-89.5, 180.0, 0.5, 0.5)
    >>> ig1234 = (50,    50,    50, 18000)
    >>> cigaxg('L',ig1234)
    (-89.5, 180.0, 0.5, 0.5)

    Example of bad use (and doctest tests):

    >>> cigaxg('L',50,    50,    50, 18000.)
    Traceback (most recent call last):
    ...
    TypeError: cxgaig error: ig1,ig2,ig3,ig4 should be of type int:(50, 50, 50, 18000.0)
    >>> cigaxg('I',50,    50,    50, 18000)
    Traceback (most recent call last):
    ...
    ValueError: cxgaig error: grtyp ['I'] must be one of ('A', 'B', 'E', 'G', 'L', 'N', 'S')
    """
    validgrtyp = ('A','B','E','G','L','N','S')
    if ig2 == ig3 == ig4 == None and type(ig1) in (type([]),type(())) and len(ig1) == 4:
        (ig1,ig2,ig3,ig4) = ig1
    if None in (grtyp,ig1,ig2,ig3,ig4):
        raise TypeError,'cigaxg error: missing argument, calling is cigaxg(grtyp,ig1,ig2,ig3,ig4)'
    elif not grtyp in validgrtyp:
        raise ValueError,'cigaxg error: grtyp ['+grtyp.__repr__()+'] must be one of '+validgrtyp.__repr__()
    elif not (type(ig1) == type(ig2) == type(ig3) == type(ig4) == type(0)):
        raise TypeError,'cigaxg error: ig1,ig2,ig3,ig4 should be of type int:'+(ig1,ig2,ig3,ig4).__repr__()
    else:
        return(Fstdc.cigaxg(grtyp,ig1,ig2,ig3,ig4))


class FstFile:
    """Python Class implementation of the RPN standard file interface

       newfile=FstFile(name='...',mode='...')  open file (fstouv)
           name is a character string containing the file name
           mode is a string containing RND SEQ R/O
       ex: newfile=FstFile('myfile','RND+R/O')
           FstHandle=FstFile[FstParm]       get matching record
           FstHandle=FstFile[0]             get next matching record (fstsui)
           FstRecord=FstFile[FstHandle]     get data associated with handle
           FstFile[FstParm]=array           append/rewrite data and tags to file
           del newfile                      close the file

        @param name name of the file
        @param mode Type of file (optional)

        @exception IOError if unable to open file
    """
    def __init__(self,name='total_nonsense',mode='RND+STD') :
        self.filename=name
        self.lastread=None
        self.lastwrite=None
        self.options=mode
        self.iun = Fstdc.fstouv(0,self.filename,self.options)
        if (self.iun == None):
          raise IOError,(-1,'failed to open standard file',self.filename)
        else:
          print 'R.P.N. Standard File (2000) ',name,' is open with options:',mode,' UNIT=',self.iun

    def voir(self,options='NEWSTYLE'):
        """Print the file content listing"""
        Fstdc.fstvoi(self.iun,options)

    def __del__(self):
        """Close File"""
        if (self.iun != None):
          Fstdc.fstfrm(self.iun)
          print 'file ',self.iun,' is closed, filename=',self.filename
        del self.filename
        del self.lastread
        del self.lastwrite
        del self.options
        del self.iun

    def __getitem__(self,key):
        """Get record meta (FstParms) and data (numpy array) from file
        (meta,data) = myfstfile[mykey]
        """
        params = self.info(key)         # 1 - get handle
        if params == None:              # oops !! not found
            return (None,None)
        target = params.handle
        array=Fstdc.fstluk(target)   # 2 - get data
        return (params,array)               # return keys and data arrray

    def edit_dir_entry(self,key):
      """Edit (zap) directory entry referenced by handle"""
      return(Fstdc.fst_edit_dir(key.handle,key.date,key.deet,key.npas,-1,-1,-1,key.ip1,key.ip2,key.ip3,
                                key.type,key.nom,key.etiket,key.grtyp,key.ig1,key.ig2,key.ig3,key.ig4,key.datyp))

    def info(self,key):
        """Get handle associated with key"""
        if isinstance(key,FstParm):         # fstinf, return FstHandle instance
            if key.nxt == 1:               # get NEXT one thatmatches
                self.lastread=Fstdc.fstinf(self.iun,key.nom,key.type,
                              key.etiket,key.ip1,key.ip2,key.ip3,key.date,key.handle)
            else:                           # get FIRST one that matches
                if key.handle >= 0 :       # handle exists, return it
                    return key
                self.lastread=Fstdc.fstinf(self.iun,key.nom,key.type,
                              key.etiket,key.ip1,key.ip2,key.ip3,key.date,-2)
        elif key==NextMatch:                # fstsui, return FstHandle instance
            self.lastread=Fstdc.fstinf(self.iun,' ',' ',' ',0,0,0,0,-1)
        else:
            raise TypeError   # invalid "index"
        result=FstParms()
        if self.lastread != None:
#            self.lastread.__dict__['fileref']=self
            result.update_by_dict(self.lastread)
            result.fileref=self
#            print 'DEBUG result=',result
        else:
            return None
        return result # return handle

    def __setitem__(self,index,value):
        """[re]write data and tags"""
        if (value == None):
            if (isinstance(index,FstParm)): # set of keys
                target = index.handle
            elif type(index) == type(0):  # handle
                target = index
            else:
                raise TypeError, 'FstFile: index must provide a valid handle to erase a record'
            print 'erasing record with handle=',target,' from file'
            self.lastwrite=Fstdc.fsteff(target)
            # call to fsteff goes here
        elif (isinstance(index,FstParms)) and (type(value) == type(numpy.array([]))):
            # call to fstecr goes here with rewrite flag (index=true/false)
            self.lastwrite=0
#            print 'writing data',value.shape,' to file, keys=',index
#            print 'dict = ',index.__dict__
            if (value.flags.farray):
              print 'fstecr Fortran style array'
              Fstdc.fstecr(value,
                         self.iun,index.nom,index.type,index.etiket,index.ip1,index.ip2,
                         index.ip3,index.dateo,index.grtyp,index.ig1,index.ig2,index.ig3,
                         index.ig4,index.deet,index.npas,index.nbits)
            else:
              print 'fstecr C style array'
              Fstdc.fstecr(numpy.reshape(numpy.transpose(value),value.shape),
                         self.iun,index.nom,index.type,index.etiket,index.ip1,index.ip2,
                         index.ip3,index.dateo,index.grtyp,index.ig1,index.ig2,index.ig3,
                         index.ig4,index.deet,index.npas,index.nbits)
        else:
           raise TypeError,'FstFile write: value must be an array and index must be FstParms'


class Grid:
    "Base method to attach a grid description to a fstd field"
    def __init__(self,keysndata=(None,None),(xkeys,xaxis)=(None,None),(ykeys,yaxis)=(None,None),ninj=(None,None),grtyp=None,ig14=(None,None,None,None),vector=None):
      if grtyp != None:           # grid is defined by grtyp,ig1,ig2,ig3,ig4
        ig1,ig2,ig3,ig4 = ig14
        if (ig1==None or ig2==None or ig3==None or ig4==None):
           raise TypeError,'Grid: ig14 tuple (ig1,ig2,ig3,ig4) must be specified'
        lescles=FstParms()
        lescles.xyref=(grtyp,ig1,ig2,ig3,ig4)
        lescles.grtyp=grtyp
        lescles.ig1=ig1
        lescles.ig2=ig2
        lescles.ig3=ig3
        lescles.ig4=ig4
        lescles.griddim=ninj
        ledata=None
      else:
        if keysndata != (None,None) and isinstance(keysndata,tuple):
          (lescles,ledata)=keysndata
          if ((not isinstance(lescles,FstParms)) or (not isinstance(ledata,numpy.ndarray))):
            raise TypeError,'Grid: argument keysndata is not a tuple of type (Fstkeys,data)'
        else:
          raise TypeError,'Grid: argument keysndata is not a tuple of type (Fstkeys,data)'
        if ( (xkeys,xaxis)!=(None,None) and (ykeys,yaxis)!=(None,None) ):       # xaxis any yaxis provided
          lescles.xyref=(xkeys.grtyp,xkeys.ig1,xkeys.ig2,xkeys.ig3,xkeys.ig4)
          lescles.xaxis=xaxis
          lescles.yaxis=yaxis.ravel()
          lescles.griddim=(xaxis.shape[0],yaxis.shape[1])
        else:
          if lescles.grtyp == 'Z' or lescles.grtyp == 'Y':       # get xaxis and yaxis
            (xcles,xdata) = lescles.getaxis('X')
            (ycles,ydata) = lescles.getaxis('Y')
          else:                                    # only keysndata, grid defined by grtyp,ig1,ig2,ig3,ig4 from keys
            lescles.xyref=(lescles.grtyp,lescles.ig1,lescles.ig2,lescles.ig3,lescles.ig4)
            if ninj != (None,None):
              lescles.griddim=ninj
            else:
              if ledata == None:
                raise TypeError,'Grid: argument ninj must be specified when data field is missing'
              else:
                lescles.griddim=(ledata.shape[0],ledata.shape[1])
        if vector != None:
          if (lescles.nom=='UU  '):
            (clesvv,champvv) = lescles.fileref[FstKeys(nom='VV',type=lescles.type,date=lescles.date,etiket=lescles.etiket,ip1=lescles.ip1,ip2=lescles.ip2,ip3=lescles.ip3)]
            if clesvv == None:
              print 'Grid error: VV record not found'
              return
            self.keys2 = clesvv
            self.field2 = champvv
        else:
          self.keys2= None
          self.field2=None
#      print 'Grid DEBUG lescles.nom date dateo=',lescles.nom,lescles.date,lescles.dateo
#      print 'Grid DEBUG lescles.xyref=',lescles.xyref
#      print 'Grid DEBUG lescles.griddim=',lescles.griddim
      self.keys = lescles
      self.field = ledata
#      print 'Grid termine'
#      print ' '

    def __getitem__(self,tgrid):              # interpolate to target grid
      if isinstance(tgrid,Grid):
        tgrtyp=tgrid.keys.grtyp
        txyref=tgrid.keys.xyref
        tgriddim=tgrid.keys.griddim
        txaxis=tgrid.keys.xaxis
        tyaxis=tgrid.keys.yaxis
      else:
        raise TypeError,'Grid: argument is not a Grid instance'
      xyref=self.keys.xyref
      ks=self.keys
      print 'Debug Grid interpolation for ',self.keys.nom,' from grid',xyref,' griddim=',ks.griddim,' to grid',txyref,' griddim=',tgriddim
      srcflag=ks.xaxis != None
      dstflag=txaxis != None
      vecteur=self.keys2 != None
#      print 'Degug Grid __getitem__ srcflag=',srcflag,' dstflag=',dstflag
      newkeys=FstParms()
      newkeys.update_by_dict_from(self.keys,Sgrid__Desc)
      newkeys.update_by_dict_from(tgrid.keys,Tgrid__Desc)
#      print 'Debug Grid __getitem__ newkeys=',newkeys.nom,newkeys.xyref
      if (tgrid.field == None):
#        print 'Debug Grid __getitem__ creating dummy array'
        dummyarray=numpy.zeros( (2,2) )
        newgrid=Grid((newkeys,dummyarray))
      else:
        newgrid=Grid((newkeys,tgrid.field))
      if vecteur:
        print 'Degug Grid __getitem__ interpolation vectorielle'
        (newarray,newarray2)=Fstdc.ezinterp(self.field,self.field2,ks.griddim,ks.grtyp,xyref,ks.xaxis,ks.yaxis,srcflag,tgriddim,tgrtyp,txyref,txaxis,tyaxis,dstflag,vecteur)
        newgrid.field=newarray
        newgrid.field2=newarray2
        newgrid.keys2=self.keys2
        newgrid.keys2.update_by_dict_from(tgrid.keys,Tgrid__Desc)
      else:
        newarray=Fstdc.ezinterp(self.field,None,ks.griddim,ks.grtyp,xyref,ks.xaxis,ks.yaxis,srcflag,tgriddim,tgrtyp,txyref,txaxis,tyaxis,dstflag,vecteur)
        newgrid.field=newarray
#      print 'newarray info=',newarray.shape,newarray.flags
#      print 'Debug newgrid.keys=',newgrid.keys.nom,newgrid.keys.xyref
#      if newgrid.keys2 != None:
#        print 'Debug newgrid.keys2',newgrid.keys2.nom,newgrid.keys.xyref
      return(newgrid)


class FstParm:
    "Base methods for all RPN standard file descriptor classes"
    def __init__(self,model,reference,extra):
        for name in reference.keys():            # copy initial values from reference
            self.__dict__[name]=reference[name]  # bypass setatttr method for new attributes
        if model != None:
            if isinstance(model,FstParm):        # update with model attributes
               self.update(model)
            else:
                raise TypeError,'FstParm.__init__: model must be an FstParm class instances'
        for name in extra.keys():                # add extras using own setattr method
            setattr(self,name,extra[name])

    def update(self,with):
        "Replace Fst attributes of an instance with Fst attributes from another"
        if isinstance(with,FstParm) and isinstance(self,FstParm):  # check if class=FstParm
            for name in with.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in X__FullDesc.keys()):
                    self.__dict__[name]=with.__dict__[name]
        else:
            raise TypeError,'FstParm.update: can only operate on FstParm class instances'

    def update_cond(self,with):
        "Conditional Replace Fst attributes if not wildcard values"
        if isinstance(with,FstParm) and isinstance(self,FstParm):  # check if class=FstParm
            for name in with.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in X__FullDesc.keys()):
                    if (with.__dict__[name] != W__FullDesc[name]):
                        self.__dict__[name]=with.__dict__[name]
        else:
            raise TypeError,'FstParm.update_cond: can only operate on FstParm class instances'

    def update_by_dict(self,with):
        for name in with.keys():
            if name in self.__dict__.keys():
                setattr(self,name,with[name])

    def update_by_dict_from(self,frm,with):
        for name in with.keys():
            if name in self.__dict__.keys():
                setattr(self,name,frm.__dict__[name])

    def __setattr__(self,name,value):   # this method cannot create new attributes
        if name in self.__dict__.keys():                   # is attribute name valid ?
            if type(value) == type(self.__dict__[name]):   # right type (string or int))
                if type(value) == type(''):
                    reflen=len(self.__dict__[name])        # string, remember length
                    self.__dict__[name]=(value+reflen*' ')[:reflen]
                else:
                    self.__dict__[name]=value              # integer
            else:
                if self.__dict__[name] == None:
#                   print 'Debug***** None name=',name
                   self.__dict__[name]=value
                else:
                    raise TypeError,'FstParm: Wrong type for attribute '+name+'='+value.__repr__()
        else:
            raise ValueError,'FstParm: attribute'+name+'does not exist for class'+self.__class__.__repr__()

    def __setitem__(self,name,value):
        self.__setattr__(name,value)

    def __getitem__(self,name):
        return self.__dict__[name]

    def findnext(self,flag=1):                  # set/reset next match flag
        self.nxt = flag
        return self

    def wildcard(self):                  # reset keys to undefined
        self.update_by_dict(W__FullDesc)

    def __str__(self):
        return dump_keys_and_values(self)

    def __repr__(self):
        return self.__dict__.__repr__()


class FstKeys(FstParm):
    "Primary descriptors, used to search for a record"
    def __init__(self,model=None,**args):
        FstParm.__init__(self,model,X__PrimaryDesc,args)

class FstDesc(FstParm):
    "Auxiliary descriptors, used when writing a record or getting descriptors from a record"
    def __init__(self,model=None,**args):
        FstParm.__init__(self,model,X__AuxiliaryDesc,args)

class FstParms(FstKeys,FstDesc):
    "Full set of descriptors, Primary + Auxiliary, needed to write a record, can be used for search"
    def __init__(self,model=None,**args):
        FstKeys.__init__(self)   # initialize Key part
        FstDesc.__init__(self)   # initialize Auxiliary part
        if model != None:
            if isinstance(model,FstParm):
                self.update(model)
            else:
                raise TypeError,'FstParms: cannot initialize from arg #1'
        for name in args.keys(): # and update with specified attributes
            setattr(self,name,args[name])

    def getaxis(self,axis=None):
       if not (self.grtyp in ('Z','Y','#')):
         raise ValueError,'getaxis error: can not get axis from grtyp=',self.grtyp
         return(None,None)
       if (self.xaxis == None and self.yaxis == None):
         searchkeys = FstKeys(ip1=self.ig1,ip2=self.ig2)
         if self.grtyp != '#':
            searchkeys.update_by_dict({'ip3':self.ig3})
         searchkeys.update_by_dict({'nom':'>>'})
         (xaxiskeys,xaxisdata) = self.fileref[searchkeys]
         searchkeys.update_by_dict({'nom':'^^'})
         (yaxiskeys,yaxisdata) = self.fileref[searchkeys]
         if (xaxiskeys == None or yaxiskeys == None):
           print 'getaxis error: axis grid descriptors (>>,^^) not found'
           return (None,None)
         self.xaxis=xaxisdata
         self.yaxis=yaxisdata
         self.xyref = (xaxiskeys.grtyp,xaxiskeys.ig1,xaxiskeys.ig2,xaxiskeys.ig3,xaxiskeys.ig4)
         ni=xaxisdata.shape[0]
         nj=yaxisdata.shape[1]
         self.griddim=(ni,nj)
       axiskeys=FstParms()
       axiskeys.xyref=self.xyref
       axiskeys.griddim=self.griddim
       if axis == 'X':
         axisdata=self.xaxis
       elif axis == 'Y':
         axisdata=self.yaxis.ravel()
       else:
         axisdata=(self.xaxis,self.yaxis.ravel())
       return(axiskeys,axisdata)


class FstCriterias:
    "Base methods for RPN standard file selection criteria input filter classes"
    def __init__(self,reference,exclu,extra):
        self.__dict__['exclu'] = exclu
        for name in reference.keys():            # copy initial values from reference
            self.__dict__[name]=[]+reference[name]  # bypass setatttr method for new attributes
        for name in extra.keys():                # add extras using own setattr method
            setattr(self,name,extra[name])

    def update(self,with):
        "Replace Fst attributes of an instance with Fst attributes from another"
        if isinstance(with,FstCriterias) and isinstance(self,FstCriterias):  # check if class=FstParm
            for name in with.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in X__Criteres.keys()):
                    self.__dict__[name]=with.__dict__[name]
        else:
            raise TypeError,'FstParm.update: can only operate on FstCriterias class instances'

    def update_cond(self,with):
        "Conditional Replace Fst attributes if not wildcard values"
        if isinstance(with,FstCriterias) and isinstance(self,FstCriterias):  # check if class=FstCriterias
            for name in with.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in X__Criteres.keys()):
                    if (with.__dict__[name] != X__Criteres[name]):
                        self.__dict__[name]=with.__dict__[name]
        else:
            raise TypeError,'FstCriterias.update_cond: can only operate on FstCriterias class instances'

    def update_by_dict(self,with):
        for name in with.keys():
            if name in self.__dict__.keys():
                setattr(self,name,with[name])

    def isamatch(self,with):
        "Check attributes for a match, do not consider wildcard values"
        global X__DateDebut,X__DateFin,X__Delta
        if isinstance(with,FstParm) and isinstance(self,FstCriterias):  # check if class=FstParm
            match = 1
            for name in with.__dict__.keys():
                if (name in self.__dict__.keys()) and (name in X__FullDesc.keys()):
                    if (self.__dict__[name] != X__Criteres[name]):      # check if wildcard
                        if (name == 'date'):
                            print 'Debug isamatch name=',name
                            if (X__DateDebut != -1) or (X__DateFin != -1):      # range of dates
                                print 'Debug range de date debut fin delta',X__DateDebut,X__DateFin,X__Delta
                                print 'Debug range with self',name,with.__dict__[name],self.__dict__[name]
                                match = match & Fstdc.datematch(with.__dict__[name],X__DateDebut,X__DateFin,X__Delta)
                            else:
                                print 'Debug check ',name,with.__dict__[name],self.__dict__[name]
                                match = match & (with.__dict__[name] in self.__dict__[name])
                        else:
                            print 'Debug check ',name,with.__dict__[name],self.__dict__[name]
                            match = match & (with.__dict__[name] in self.__dict__[name])
            return match
        else:
            raise TypeError,'FstCriterias.isamatch: can only operate on FstParm, FstCriterias class instances'

    def __setattr__(self,name,values):   # this method cannot create new attributes
        if name in self.__dict__.keys():                   # is attribute name valid ?
            if type(values) == type([]):
                self.__dict__[name]=[]
                for value in values:
                    if type(value) == type(''):
                        reflen=len(X__Criteres[name][0])        # string, remember length
                        self.__dict__[name].append((value+reflen*' ')[:reflen])
                    else:
                        self.__dict__[name].append(value)              # integer
            else:
                self.__dict__[name]=[]
                if type(values) == type(''):
                    reflen=len(X__Criteres[name][0])        # string, remember length
                    self.__dict__[name].append((values+reflen*' ')[:reflen])
                else:
                    self.__dict__[name].append(values)              # integer
        else:
            raise ValueError,'attribute'+name+'does not exist for class '+self.__class__.__repr__()

    def __str__(self):
        return dump_keys_and_values(self)

class FstSelect(FstCriterias):
    "Selection criterias for RPN standard file input filter"
    def __init__(self,**args):
        FstCriterias.__init__(self,X__Criteres,0,args)

class FstExclude(FstCriterias):
    "Exclusion criterias for RPN standard file input filter"
    def __init__(self,**args):
        FstCriterias.__init__(self,X__Criteres,1,args)


class FstRec(FstParms):
    """Standard file record, with data (ndarray class) and full set of descriptors (FstParms class)

    Example of use (and doctest tests):

    >>> r = FstRec()
    >>> r.d
    array([], dtype=float64)
    >>> r = FstRec([1,2,3,4])
    >>> r.d
    array([1, 2, 3, 4])
    >>> a = numpy.array([1,2,3,4],order='FORTRAN',dtype='float32')
    >>> r = FstRec(a)
    >>> r.d
    array([ 1.,  2.,  3.,  4.], dtype=float32)
    >>> a[1] = 5
    >>> r.d #r.d is a reference to a, thus changing a changes a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r = FstRec(a.copy())
    >>> r.d
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> a[1] = 9
    >>> r.d #r.d is a copy of a, thus changing a does not change a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r.grtyp
    'X'
    """
    def __init__(self,data=None,params=None):
        if data == None:
            self.d = numpy.array([])
        elif type(data) == numpy.ndarray:
            self.d = data
        elif type(data) == type([]):
            self.d = numpy.array(data)
        else:
            raise TypeError,'FstRec: cannot initialize data from arg #1'
        FstParms.__init__(self)
        if params:
            if isinstance(params,FstParm):
                self.update(params)
            elif type(params) == type({}):
                self.update_by_dict(params)
            else:
                raise TypeError,'FstRec: cannot initialize parameters from arg #2'

    def __setattr__(self,name,value):   # this method cannot create new attributes
        if name == 'd':
            if type(value) == numpy.ndarray:
                self.__dict__[name]=value
            else:
                raise TypeError,'FstRec: data should be an instance of numpy.ndarray'
        else:
            FstParms.__setattr__(self,name,value)


class FstDate:
    """RPN STD Date representation

    FstDate(DATESTAMP) or FstDate(YYYYMMDD,HHMMSShh)
    @param DATESTAMP CMC date stamp or FstDate object
    @param YYYYMMDD  Int with Visual representation of YYYYMMDD
    @param HHMMSShh  Int with Visual representation of HHMMSShh

    @exception TypeError if parameters are wrong type
    """
    stamp = 0

    def __init__(self,word1,word2=-1):
        if isinstance(word1,FstDate):
            self.stamp = word1.stamp
        elif type(word1) == type(0):    # integer type
            if (word2 == -1):
                self.stamp = word1
            else:
                dummy=0
                (self.stamp,dummy1,dummy2) = Fstdc.newdate(dummy,word1,word2,3)
        else:
            raise TypeError, 'FstDate: arguments should be of type int'

    def __sub__(self,other):
        "Time difference between 2 dates"
        return(Fstdc.difdatr(self.stamp,other.stamp))

    def incr(self,temps):
        """Increase Date by the specified number of hours

        @param temps Number of hours for the FstDate to be increased
        @return self

        @exception TypeError if temps is not of int or real type
        """
        if ((type(temps) == type(1)) or (type(temps) == type(1.0))):
            nhours = 0.0
            nhours = temps
            self.stamp=Fstdc.incdatr(self.stamp,nhours)
            print 'Debug idate=',idate,type(idate)
            return(self)
        else:
            raise TypeError,'FstDate.incr: argument should be int or real'


class FstDateRange:
    """RPN STD Date Range representation

    FstDateRange(DateStart,DateEnd,Delta)
    @param DateStart FstDate start of the range
    @param DateEnd   FstDate end of the range
    @param Delta     Increment of the range iterator, hours, real

    @exception TypeError if parameters are wrong type
    """
    #TODO: make this an iterator
    dateDebut=-1
    dateFin=-1
    delta=0.0
    now=-1

    def __init__(self,debut=-1,fin=-1,delta=0.0):
        if isinstance(debut,FstDate) and isinstance(fin,FstDate) and ((type(delta) == type(1)) or (type(delta) == type(1.0))):
            self.dateDebut=debut
            self.now=debut
            self.dateFin=fin
            self.delta=delta
        else:
            raise TypeError,'FstDateRange: arguments type error FstDateRange(FstDate,FstDate,Real)'

    def lenght(self):
        """Provide the duration of the date range
        @return Number of hours
        """
        return abs(self.dateFin-self.dateDebut)

    def next(self):
        """Return the next date/time in the range (step of delta hours)
        @return next FstDate, None if next date is beyond range
        """
        self.now.incr(self.delta)
        if (self.dateFin-self.now)*self.delta < 0.:
            return None
        return FstDate(self.now)

    def reset(self):
        """Reset the FstDateRange iterator to the range start date"""
        self.now=self.dateDebut


class FstMapDesc:
    "Map Descriptors with lat1,lon1, lat2,lon2, rot"
    def __init__(self,key,xs1=0.,ys1=0.,xs2=0.,ys2=0.,ni=0,nj=0):
        print 'Debug FstMapDesc grtyp ig1-4=',key.grtyp,key.ig1,key.ig2,key.ig3,key.ig4
#        print 'Debug FstMapDesc xs1,ys1,xs2,ys2=',xs1,ys1,xs2,ys2
        if isinstance(key,FstParm):
          print 'Debug FstMapDesc appel a Fstdc.mapdscrpt'
          print 'xs1,ys1,xs2,ys2=',xs1,ys1,xs2,ys2
          self.geodesc=Fstdc.mapdscrpt(xs1,ys1,xs2,ys2,ni,nj,key.grtyp,key.ig1,key.ig2,key.ig3,key.ig4)
#       (self.lat1,self.lon1,self.lat2,self.lon2,self.rot)=Fstdc.mapdscrpt(xs1,ys1,xs2,ys2,key.grtyp,key.ig1,key.ig2,key.ig3,key.ig4)
        else:
            raise TypeError,'FstMapdesc: invalid key'

FirstRecord=FstKeys()
NextMatch=None
Predef_Grids()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
