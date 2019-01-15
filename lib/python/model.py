#!/usr/bin/env python
import sys,re,os,inspect

def whoami():
    return sys._getframe(1).f_code.co_name

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
                self["namelist"][namelistName] = {"index":nml_cnt,"rec":{},"markup":self._get_markup(item),"value_fmt":item}
                nml_cnt += 1
                initialized = 1
                rec_cnt = 0
            elif initialized:                
                contentNml = endNml.split(item)[0]
                try:
                    self["namelist"][namelistName]["term"] = endNml.split(item)[1]
                except IndexError:
                    sys.stderr.write('Error: namelist entry '+namelistName+' is not properly closed\n')
                    return(Settings.ERROR)
                var = 'unassigned'          
                for item in re.split('(\s*\w+\s*=)',contentNml):                    
                    item_streaming = lineFeed.sub(' ',item)
                    if re.search('=',item_streaming):
                        var_fmt = re.sub('=','',item)
                        var = re.sub('[\s,]','',var_fmt)                        
                    else:
                        if var != 'unassigned':
                            strippedItem = re.sub('[,\s\n]',delim,item_streaming)                            
                            strippedItem = re.sub('(?<='+boundary+')'+delim+'+(?='+boundary+')',',',strippedItem)
                            strippedItem = re.sub(delim,'',strippedItem)
                            self["namelist"][namelistName]["rec"][var.lower()] = {'name':var,'name_markup':self._get_markup(var_fmt),
                                                                                  'value':strippedItem,'value_markup':self._get_markup(item),
                                                                                  'value_fmt':item,'updated':False,'index':rec_cnt}
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
    sets=Settings()
    err=sets.set('Cstv_dt_8',10)
    timestep=sets.get('Cstv_dt_8')
    print timestep
    err=sets.rm('Cstv_dt_8')
    timestep=sets.get('Cstv_dt_8')

