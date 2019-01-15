#!/usr/bin/env python
import sys,re,os,string
from random import randint

def whoami():
    return sys._getframe(1).f_code.co_name

class DictWithProp(dict):
    """Dict with properties attached to a value
    >>> a = DictWithProp()
    >>> a['nq'] = ('q',3)
    >>> a['nd'] = ('d',8)
    >>> a['nn'] = 'n'
    >>> (a['nq'],a['nd'],a['nn'])
    ('q', 'd', 'n')
    >>> (a.idx('nq'),a.idx('nd'),a.idx('nn'))
    (3, 8, None)
    >>> a['nq'] = 'b'
    >>> (a['nq'],a['nd'],a['nn'])
    ('b', 'd', 'n')
    >>> (a.idx('nq'),a.idx('nd'),a.idx('nn'))
    (3, 8, None)
    >>> k = a.keys()
    >>> k.sort()
    >>> k
    ['nd', 'nn', 'nq']
    >>> del(a['nd'])
    >>> k = a.keys()
    >>> k.sort()
    >>> k
    ['nn', 'nq']
    >>> a.setProp('nq','s','ok')
    >>> a.getProp('nq','s')
    'ok'
    >>> k = a.propKeys('nq')
    >>> k.sort()
    >>> k
    ['idx', 's', 'v']
    """
    def __init__(self):
        super(DictWithProp,self).__init__()

    def __setitem__(self,name,value_idx):
        if isinstance(value_idx,type((1,))):
            value_dict = {'v':value_idx[0],'idx':value_idx[1]}
        else:
            value_dict = {'v':value_idx,'idx':None}
        if name in self.keys():
            value_dict1 = super(DictWithProp,self).__getitem__(name)
            if not value_dict['idx']: del(value_dict['idx'])
            value_dict1.update(value_dict)
            value_dict = value_dict1
        tmp = super(DictWithProp,self).__setitem__(name,value_dict)
        return value_dict
   
    def __getitem__(self,name):
        return super(DictWithProp,self).__getitem__(name)['v']
    
    def idx(self,name):
        return self.getProp(name,'idx')
    
    def getProp(self,name,prop):
        return super(DictWithProp,self).__getitem__(name)[prop]
    
    def setProp(self,name,prop,value):
        d = super(DictWithProp,self).__getitem__(name)
        d[prop] = value
        super(DictWithProp,self).__setitem__(name,d)
        
    def propKeys(self,name):
        d = super(DictWithProp,self).__getitem__(name)
        return d.keys()


class ListWithProp(list):
    """List with properties attached to a value
    """
    def __init__(self):
        super(ListWithProp,self).__init__()

    ## def __getitem__(self,name):
    ##     return super(DictWithProp,self).__getitem__(name)[0]
    
    ## def idx(self,name):
    ##     return super(DictWithProp,self).__getitem__(name)[1]


class ModelCfgBase(DictWithProp):
    """Base class to manipulate model's config data
    """
    def __init__(self,file=None,data=None,debug=0):
        self.verbose = False
        if debug: self.verbose = True
        (self.rawdata, self.lexdata, self.parsedata) = ('', None ,None)
        DictWithProp.__init__(self)
        if file: self._read(file)
        if data:
            if  type(data) == type(''):
                self.rawdata += data
            else:
                raise TypeError('Provided data must be a string')
        if self.rawdata: self._lex()

    def _read(self,filename):
        """Read input file"""
        if self.verbose: print "Attempting to read from "+filename
        try:
            fd = open(filename,"rb")
            try:
                self.rawdata = string.join(fd.readlines())
            finally:
                fd.close()
        except IOError:
            if self.verbose:
                sys.stderr.write("Warning: file "+filename+" does not exist or is not readable\n")
            raise
         
    def _lex(self,rawdata=None):
        """Split read file into parsable components"""
        if not rawdata: rawdata = self.rawdata
        #self.lexdata = shlex.split(rawdata)
        self.lexdata = re.findall(r'([\s]+|[\n]+|,|=|(?:[^\s,"]|"(?:\\.|[^"])*")+)',rawdata)
        for ii in range(len(self.lexdata)-1,0,-1):
            item = self.lexdata[ii]
            if item[0] not in ['"',"'"]:
                kv = string.split(item,'=',1)
                #print 'kv:',[item,len(kv),kv]
                if len(kv) == 2 and kv[0]:
                    if kv[1]:
                        self.lexdata = self.lexdata[0:ii]+[kv[0],'=',kv[1]]+self.lexdata[ii+1:]
                    else:
                        self.lexdata = self.lexdata[0:ii]+[kv[0],'=']+self.lexdata[ii+1:]        

    def _addSection(self,secname,lexidx):
        self[secname] = (DictWithProp(),lexidx)
    def _markSectionEnd(self,secname,lexidx):
        if secname in self.keys():
            self.setProp(secname,'idxend',lexidx)
    def _addKey(self,secname,keyname,lexidx):
        if not secname in self.keys():
            self._addSection(secname,-1)
        self[secname][keyname] = (ListWithProp(),lexidx)
    def _addVal(self,secname,keyname,value,lexidx):
        if not secname in self.keys():
            self._addSection(secname,-1)
        if not keyname in self[secname].keys():
            self._addKey(secname,keyname,-1)
        self[secname][keyname].append((value,lexidx))
        
    def _renameSection(self,oldname,newname):
        if oldname in self.keys():
            self.setProp(oldname,'status','renamed')
            self.setProp(oldname,'newname',newname)
    def _deleteSection(self,secname):
        if secname in self.keys():
            for keyname in self[secname].keys():
                self.rm(secname,keyname)
            self.setProp(secname,'status','deleted')
    def _deleteKeyValues(self,secname,keyname):
        if self.get(secname,keyname):
            for ii in xrange(len(self[secname][keyname])):
                self[secname][keyname][ii] = ''

    def get(self,secname,keyname):
        try:
            return self[secname][keyname]
        except:
            return None
    def set(self,secname,keyname,value):
        self._deleteKeyValues(secname,keyname)
        self[secname][keyname][0] = value
    def rm(self,secname,keyname):
        if self.get(secname,keyname):
            self._deleteKeyValues(secname,keyname)
            self[secname].setProp(keyname,'status','deleted')
    def mv(self,secname,oldkeyname,newkeyname):
        if self.get(secname,oldkeyname):
            self[secname].setProp(oldkeyname,'status','renamed')
            self[secname].setProp(oldkeyname,'newname',newkeyname)
    def ls(self,secname=None):
        seclist = self.keys()
        if secname: seclist = [secname]
        for mysec in seclist:
            for mykey in self[mysec].keys():
                for (myval,myline) in self[mysec][mykey]:
                    #TODO: if deleted, renamed, changed val,...
                    print mysec,'/',mykey,'=',myval


class FortranNamelist(ModelCfgBase):
    """Manipulate Fortran namelists
    """
    def __init__(self,file='gem_settings.nml',debug=0):
        """Settings class constructor"""
        if debug: self.verbose = True
        ModelCfgBase.__init__(self,debug=debug)
        self.lexdata2 = []
        self._read(file)
        self._lex()
        self._parse()

    def _lex(self,rawdata=None):
        """Split read file into parsable components"""
        if not rawdata: rawdata = self.rawdata
        ModelCfgBase._lex(self,rawdata)
        mynml = None
        for ii in xrange(len(self.lexdata)):
            item = self.lexdata[ii]
            if re.match(r'^[\s\n]+|,$',item): continue
            if len(item) > 0 and item[0] == '&':
                if len(item) > 1:
                    mynml = item[1:]
                else:
                    mynml = None
            if mynml: self.lexdata2.append((ii,item))
            if item == '/': mynml = None

    def _parse(self):
        """Parse lexed input data from the settings file"""
        mynml = None
        mykey = None
        self.parsedata = []
        for ii in xrange(len(self.lexdata2)):
            (iim,iip,itemm,itemp) = (None,None,None,None)
            if ii > 0 :(iim,itemm) = self.lexdata2[ii-1]
            (ii0,item0) = self.lexdata2[ii]
            if ii < len(self.lexdata2)-1: (iip,itemp) = self.lexdata2[ii+1]
            if len(item0) > 1 and item0[0] == '&':
                (mynml,mykey) = (item0[1:].lower(),None)
                if not mynml in self.keys():
                    self._addSection(mynml,ii0)
            elif mynml and item0 == '/':
                self._markSectionEnd(mynml,ii0)
                (mynml,mykey) = (None,None)
            elif mynml and item0 == '=':
                mykey = itemm.lower()
                self._addKey(mynml,mykey,iim)
            elif mynml and itemp == '=':
                mykey = item0.lower()
                self._addKey(mynml,mykey,ii0)
            elif mynml and mykey and itemp != '=':
                self._addVal(mynml,mykey,item0,ii0)

    def isNml(self,namelist):
        """Check if namelist exists"""
        return namelist in self.keys()

    def mkNml(self,namelist):
        """Create empty namelist if does not exists"""
        if (not self.isNml(namelist)):
            self._addSection(namelist,None)





class _DataMethods(dict):
    """Internal Get/Set methods for model data structures"""

    OK = 1
    ERROR = None

    def __init__(self):
        self.verbose = False

    def set(self,key,value,namelist=None):
        """Substitute a given value for a given key or add to namelist if necessary"""
        status = self._listop(whoami(),key,value,namelist)
        return(status)

    def get(self,key,namelist=None):
        """Retrieve a key's value"""
        value = None
        self._value = None        
        status = self._listop(whoami(),key,value,namelist)        
        return(self._value)

    def rm(self,key,namelist=None):
        """Remove an entry from the configurations"""
        value = None
        status = self._listop(whoami(),key,value,namelist)
        return(status)

    def write(self,file,reformat=False):
        """Write namelist to output file"""
        if self.verbose: print "Writing to "+file
        try:
            os.remove(file)
            fd = open(file,'wb')
            try:
                self.view(unit=fd,reformat=reformat)
            finally:
                fd.close()
        except OSError:
            if self.verbose:
                sys.stderr.write("Error: could not remove file "+file+" before writing\n")
            raise
        except IOError:
            if self.verbose:
                sys.stderr.write("Error: could not open file "+file+" for write\n")
            raise
        return(_DataMethods.OK)
    
    def _get_markup(self,string):
        """Retrieve prefix and suffix markups from string"""
        markup={'pre':'','post':''}        
        markup['pre'] = re.search('(^\s*)',string).group(1)
        markup['post'] = re.search('((\s|,|\n)*$)',string).group(1)
        return(markup)

    def _markup(self,string,markup):
        """Add prefix and suffix markups to string"""
        ostring = markup['pre']+string+markup['post']
        return(ostring)

    def _set(self,key,value,nml):
        """Implement value setting"""
        key_template = " key "
        value_template = " value,\n"
        try:
            lkey = key.lower()
            if lkey in self["namelist"][nml]["rec"]:
                self["namelist"][nml]["rec"][lkey]["value"] = value
                self["namelist"][nml]["rec"][lkey]["value_fmt"] = self._markup(value,self["namelist"][nml]["rec"][lkey]["value_markup"])
            else:
                self["namelist"][nml]["rec"][lkey] = {'name':key,'name_markup':self._get_markup(key_template),
                                                      'value':str(value),'value_markup':self._get_markup(value_template),
                                                      'value_fmt':self._markup(str(value),self._get_markup(value_template)),
                                                      'index':len(self["namelist"][nml]["rec"])}                
            self["namelist"][nml]["rec"][lkey]["updated"] = True
            return(_DataMethods.OK)
        except KeyError:
            if self.verbose:
                sys.stderr.write("Warning: invalid namelist "+nml+" selected.  Key "+key+" not modified.\n")
        return(_DataMethods.ERROR)

    def _get(self,key,value,nml):
        """Implement value getting"""
        try:
            self._value = self["namelist"][nml]["rec"][key.lower()]['value']            
            return(_DataMethods.OK)
        except KeyError:
            if self.verbose:
                sys.stderr.write("Warning: the key "+key+" was not found in "+nml+".  Returning None.\n")
            self._value = None
            raise KeyError('Key '+key+' does not exist')
        return(_DataMethods.ERROR)

    def _rm(self,key,value,nml):
        """Implement key removal"""
        try:
            del self["namelist"][nml]["rec"][key.lower()]
            return(_DataMethods.OK)
        except KeyError:
            if self.verbose:
                sys.stderr.write("Warning: the key "+key+" was not found in "+nml+".  No removal performed.\n")
        return(_DataMethods.ERROR)

    def _readAndParse(self):
        """Read input file and dispatch to appropriate parser"""
        if self.verbose: print "Attempting to read from "+self["file"]
        try:
            fd = open(self["file"],"rb")
            try:
                data = fd.readlines()
            finally:
                fd.close()
        except IOError:
            if self.verbose:
                sys.stderr.write("Warning: file "+self["file"]+" does not exist or is not readable\n")
            return(_DataMethods.ERROR)
        parseFunction = getattr(self,"_parse"+self.__class__.__name__)        
        return(parseFunction(data))

class Configs(_DataMethods):
    """Container for configuration file data and methods

    INTRODUCTION
    This module provides tools that allow a user to parse, query, modify and write
    a model run configuration file (i.e. configexp.cfg).
        
    CONFIGS CLASS
        
    METHODS
    cfgs = model.Configs(file='') - Class constructor.  This method reads and
             parses the input file into a nested dictionary contained in the
             returned Configs object.
    cfgs.set(key,value) - Set the named 'key' to the specified value 'value'.
             This method returns cfgs.OK on success and cfgs.ERROR on error.
    cfgs.get(key) - Retrieve the value (returned) of the named 'key'.
    cfgs.rm(key) - Remove the named 'key' from the configuration file.  This
             method returns cfgs.OK on success and cfgs.ERROR on error.
    cfgs.view(unit=None) - Produce pretty-printed output of the current
             configuration.
    cfgs.write(file) - Generate an output file containing the current
             configuration.

    CLASS VARIABLES
    Configs.OK      - Successful completion code for return
    Configs.ERROR   - Error code for return

    """

    def __init__(self,file='configexp.cfg',debug=0):
        """Configs class constructor"""
        _DataMethods.__init__(self)
        self["file"] = file
        if debug:
            self.verbose = True
        self.className = self.__class__.__name__
        if (self._readAndParse() == Configs.ERROR):
            raise IOError('Problem opening/Parsing configuration file: '+file)

    def _parseConfigs(self,data):
        """Parse raw input from the config file"""
        self["namelist"] = {}
        self["namelist"][self.className] = {"index":0,"rec":{}}
        cnt = 0
        for fullLine in data:
            fullLine = re.sub('^[\s\t]*export[\s\t]','',fullLine,1)
            line = re.sub('\s|;','',fullLine)
            entry = re.split('#',line)            
            key = re.split('=',entry[0])
            try:
                if len(key[1]) > 0:
                    self["namelist"][self.className]["rec"][key[0].lower()] = {'name':key[0],'name_markup':self._get_markup(key[0]),
                                                                               'value':key[1],'value_markup':self._get_markup(key[1]),
                                                                               'updated':False,'index':cnt}
                    cnt += 1
            except IndexError:
                pass
        return(Configs.OK)

    def view(self,unit=None,reformat=False):
        """Print the configurations"""
        fd = unit and unit or sys.stdout
        keyList = self["namelist"][self.className]["rec"].items()
        keyList.sort()
        for (key,entry) in keyList:
            fd.write(str(entry['name'])+'='+str(entry['value'])+';\n')
        fd.write('\n')
        return(Configs.OK)

    def _listop(self,method,key,value,section):
        """Dispatch to perform the requested list operation"""
        if not len(key):
            if self.verbose:
                sys.stderr.write("Error: no key specified\n")
            return(Configs.ERROR)
        opFunction = getattr(self,"_"+method)
        return(opFunction(key,value,self.className))
    
class Settings(_DataMethods):
    """Manipulate a model configuration file (namelist)
        
    INTRODUCTION
    This module provides tools that allow a user to parse, query, modify and write
    a model configuration file (i.e. gem_settings.nml).
        
    SETTINGS CLASS
        
    METHODS
    sets = model.Settings(file='') - Class constructor.  This method reads and
             parses the input file into a nested dictionary contained in the
             returned Settings object.
    sets.set(key,value,namelist='') - Set the named 'key' to the specified value
             'value' in the namelist 'namelist'.  If no namelist is provided, then
             all namelists are searched and the first matched key has its value set.
             This method returns sets.OK on success and sets.ERROR on error.
    sets.get(key,namelist='') - Retrieve the value (returned) of the named 'key'
             in the specified namelist.  If no namelist is provided, then all
             namelists are searched and the value of the first matching key is
             returned.
    sets.rm(key,namelist='') - Remove the named 'key' from the specified namelist.
             If no namelist is provided, then all namelists are searched and the first
             matching key is removed.  This method returns sets.OK on success and
             sets.ERROR on error.
    sets.isNml(namelist) - Check for the presence of the given 'namelist'
    sets.mkNml(namelist) - Create a new namelist with the given name
    sets.view(unit=None) - Produce pretty-printed output of the current
             configuration.
    sets.write(file) - Generate an output file containing the current
             configuration.

    CLASS VARIABLES
    Settings.OK      - Successful completion code for return
    Settings.ERROR   - Error code for return
         
    """
         
    def __init__(self,file='gem_settings.nml',debug=0):
        """Settings class constructor"""
        _DataMethods.__init__(self)
        self["file"] = file
        if debug:
            self.verbose = True
        if (self._readAndParse() == Settings.ERROR):
            raise IOError('Problem opening/Parsing settings file:'+file)
    
    def _parseSettings(self,data):
        """Parse raw input data from the settings file"""        
        delim=':'
        stream = ''.join([entry for entry in data if not re.match('^\s*#',entry)])        
        namelists = re.split('(&\s*\S+)',stream)        
        self["header"] = namelists[0]
        self["namelist"] = {}
        boundary='[\w\.\'\"-]'  #everything that can represent a value entry boundary
        endNml = re.compile('(^\s*/[^&]*)',re.M)
        lineFeed = re.compile('\n')
        nml_cnt = 0
        initialized = 0
        for item in namelists:            
            m = re.search('^&\s*(?P<nml>\S+)',item,re.M)
            if m:
                namelistName = m.group('nml')
                namelistName2 = namelistName
                if (namelistName2 in self["namelist"].keys()):
                    sys.stderr.write('WARNING: multiple occurences of namelist: &'+namelistName+'\n         You should rename the ones you are not using.\n')
                    namelistName2 = namelistName2+'_____'+str(randint(0,99999))
                self["namelist"][namelistName2] = {"index":nml_cnt,"rec":{},"markup":self._get_markup(item),"value_fmt":item}
                nml_cnt += 1
                initialized = 1
                rec_cnt = 0
            elif initialized:                
                contentNml = endNml.split(item)[0]
                try:
                    self["namelist"][namelistName2]["term"] = endNml.split(item)[1]
                except IndexError:
                    sys.stderr.write('Error: namelist entry &'+namelistName+' is not properly closed\n')
                    return(Settings.ERROR)
                var = 'unassigned'
                contentNml2 = re.sub(r'("[^"]+?"|\'[^\']+?\')', lambda m: m.group(0).replace("=", "\x00"), contentNml)
                for item in re.split(r'(\s*\w+\s*=)',contentNml2):
                    item_streaming = lineFeed.sub(' ',item)
                    if re.search('=',item_streaming):
                        item_streaming = item.replace("\x00", "=")
                        var_fmt = re.sub('=','',item)
                        var = re.sub(r'[\s,]','',var_fmt)
                    else:
                        item_streaming = item_streaming.replace("\x00", "=")
                        if var != 'unassigned':
                            strippedItem = re.sub(r'[,\s\n]',delim,item_streaming)
                            strippedItem = re.sub(r'(?<='+boundary+')'+delim+'+(?='+boundary+')',',',strippedItem)
                            strippedItem = re.sub(delim,'',strippedItem)
                            self["namelist"][namelistName2]["rec"][var.lower()] = {'name':var,'name_markup':self._get_markup(var_fmt),
                                                                                  'value':strippedItem,'value_markup':self._get_markup(item),
                                                                                  'value_fmt':item.replace("\x00", "="),'updated':False,'index':rec_cnt}
                            rec_cnt += 1
                initialized = 0
        return(Settings.OK)

    def isNml(self,namelist):
        """Check if namelist exists"""
        try:
            a = self["namelist"][namelist]
            return True
        except KeyError:
            return False

    def mkNml(self,namelist):
        """Create empty namelist if does not exists"""
        if (not self.isNml(namelist)):
            self["namelist"][namelist] = {"index":len(self["namelist"]),"rec":{},"value_fmt":'&'+namelist+'\n',
                                          "markup":self._get_markup(namelist),"term":"\n/\n\n"}
    
    def view(self,unit=None,reformat=False):
        """Print the configurations"""
        fd = unit and unit or sys.stdout
        fd.write(self["header"])
        namelistList = self["namelist"].keys()
        if reformat:
            namelistList.sort()
        else:
            namelistList = []
            for i in range(0,len(self["namelist"])):
                for nml in self["namelist"].keys():
                    try:
                        if self["namelist"][nml]['index'] == i:
                            namelistList.append(nml)
                    except KeyError:
                        continue
        for nml in namelistList:
            if reformat:
                fd.write('&'+str(nml)+'\n')
            else:                
                fd.write(self._markup(self["namelist"][nml]["value_fmt"],self["namelist"][nml]["markup"]))
            keyList = self["namelist"][nml]["rec"].items()
            if reformat:
                keyList.sort()
            else:
                keyList = []
                for i in range(0,len(self["namelist"][nml]["rec"])):
                    for (key,entry) in self["namelist"][nml]["rec"].items():
                        if self["namelist"][nml]["rec"][key]['index'] == i:
                            keyList.append((key,entry))                
            for (key,entry) in keyList:
                if entry['updated'] or reformat:
                    value = entry['value']
                    startIndex = 72-(len(key)+4)
                    while len(value) > startIndex:
                        index = value.find(',',startIndex)
                        if (index < 0):
                            startIndex = len(value)+1
                            continue
                        pad = ''
                        for i in range(0,len(key)+4): pad += ' '
                        value = value[0:index]+",\n"+pad+value[index+1:]
                        startIndex += 72
                    value = self._markup(str(value),entry['value_markup'])
                else:
                    value = entry['value_fmt']                    
                if reformat:
                    fd.write(' '+str(entry['name'])+' = '+str(value)+' ,\n')
                else:
                    fd.write(self._markup(entry['name'],entry['name_markup'])+'='+str(value))
            fd.write(self["namelist"][nml]["term"])
        return(Settings.OK)

    def _listop(self,method,key,value,namelist):
        """Dispatch to perform the requested list operation"""
        if not len(key):
            if self.verbose:
                sys.stderr.write("Error: no key specified\n")
            return(Settings.ERROR)
        if (method is 'set' and namelist and not self.isNml(namelist)):
            self.mkNml(namelist)
        opFunction = getattr(self,"_"+method)
        nml_list = (namelist) and [namelist] or self["namelist"].keys()
        for nml in nml_list:
            if namelist or self["namelist"][nml]["rec"].has_key(key.lower()):                
                status = opFunction(key,value,nml)
                if status is Settings.OK: return(status)
        if self.verbose:
            sys.stderr.write(" ".join(["Warning: unable to",method,key,"since key does not exist in specified namelist(s)\n"]))
        raise KeyError("Key "+key+" does not exist")
        return(Settings.ERROR)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    a = FortranNamelist()
    a.ls()
    ## sets=Settings()
    ## err=sets.set('Cstv_dt_8',10)
    ## timestep=sets.get('Cstv_dt_8')
    ## print timestep
    ## err=sets.rm('Cstv_dt_8')
    ## timestep=sets.get('Cstv_dt_8')

