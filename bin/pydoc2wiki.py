#!/usr/bin/env python
"""
Long

Short


"""

import sys
import re
import pydoc

tmpl = {}
tmpl['head'] = """
__NOTITLE__ 
= Python RPN: @MODULE@ =
{{:Python-RPN/2.0/navbox}} 
@DESC@ 
{{roundbox|
The functions described below are a very close ''port'' from the original [[librmn]]'s [[Librmn/FSTDfunctions|FSTD]] package.<br>
You may want to refer to the [[Librmn/FSTDfunctions|FSTD documentation]] for more details.
}}
__TOC__

@CLASS@
@FUNC@
@DATA@
"""

tmpl['funclist'] = """== Functions ==
@DATA@
"""

tmpl['methodlist'] = """==== @NAME@ Methods ====
@DATA@
"""

tmpl['classlist'] = """== Classes ==
<pre>
@TREE@
</pre>
@DATA@
"""

tmpl['datalist'] = """== Constants ==
<source lang="python">
@DATA@
</source>
"""

tmpl['attrlist'] = """==== @NAME@ Attributes ====
<source lang="python">
@DATA@
</source>
"""

tmpl['desc'] = """
@SHORT@

@LONG@
"""

tmpl['func'] = """=== @NAME@ ===
Function <tt>@NAME@</tt>: @SHORT@
<source lang="python">
@NAME@(@ARGS@):
   '''
@DESC@
   '''
</source>
@SEEALSO@
"""

tmpl['method'] = """===== @NAME@ =====
Method <tt>@NAME@</tt>: @SHORT@
<source lang="python">
@NAME@(@ARGS@):
   '''
@DESC@
   '''
</source>
@SEEALSO@
"""

tmpl['class'] = """=== @NAME@ ===
Class <tt>@NAME@</tt>: @SHORT@<br>
Child of: @SUPER@
<source lang="python">
@NAME@(@ARGS@):
   '''
@DESC@
   '''
</source>
@SEEALSO@
@ATTR@
@METH@
"""

def get_seealso(mystr, allSymbols=[]):
    if not mystr.strip():
        return ''
    #return 'See Also:\n' + '\n'.join(['* '+x for x in mystr.strip().split('\n')])
    l = []
    for x in mystr.strip().split('\n'):
        l += [y.strip() for y in x.split(" ") if y.strip()]
    s = 'See Also:\n'
    for x in l:
        if x:
            x2 = curFile+'.'+x
            ## if x in allSymbols or x2 in allSymbols:
            ##     s += '* [[' + x + ']]\n'
            ## else:
            ##     s += '* ' + x + '\n'
            if x in allSymbolsDictKeys or x2 in allSymbolsDictKeys:
                try:
                    s += '* [[' + linkPrefix + allSymbolsDict[x] + '|' + x + ']]\n'
                except:
                    s += '* [[' + linkPrefix + allSymbolsDict[x2] + '|' + x + ']]\n'
            else:
                s += '* ' + x + '\n'
    return s
    
    
def get_indent(myline, mychars=' '):
    return len(myline) - len(myline.lstrip(mychars))


def do_indent(mystr, mychars='   '):
    return mychars + mystr.replace('\n', '\n'+mychars)

def do_undent(mylinelist, mychars=' '):
    l = []
    n = get_indent(mylinelist[0], mychars)
    for myline in mylinelist:
        l.append(myline[n:])
    return l


def split_sections(mylinelist, mymatch=''):
    d = {}
    key  = ''
    d[key] = []
    n = len(mymatch)
    for myline in mylinelist:
        myflag = True
        if mymatch:
            myflag = myline[0:n] == mymatch
        if myline and get_indent(myline) == 0 and myflag:
            key = myline.strip()
            d[key] = []
        else:
            d[key].append(myline)
    return d


def split_sections_title(mylinelist, myheaders={}):
    d = {}
    key  = ''
    d[key] = []
    for k in myheaders.keys():
        d[k] = []
    for myline in mylinelist:
        flag = False
        for k in myheaders.keys():
            v = myheaders[k]
            if myline[0:len(v)] == v:
                key = k
                flag = True
        if not flag:
            d[key].append(myline)
    return d


class MyDocFileDataDesc(dict):

    def __init__(self, mystr):
        ## self.sections = {
        ##     'args'     : 'Args:',
        ##     'return'   : 'Returns:',
        ##     'raise'    : 'Raises:',
        ##     'example'  : 'Examples:',
        ##     'seealso'  : 'See Also:'
        ##     }
        self.sections = {
            'seealso'  : 'See Also'
            }
        self[''] = mystr
        self['short'] = mystr.strip()
        self['long']  = ''
        for k in self.sections.keys():
            self[k] = ''
        if not mystr.rstrip():
            return
        #TODO: 1st desc line may be improperly indented
        d = split_sections_title(do_undent(mystr.split('\n')), self.sections)
        for k in d.keys():
            d[k] = '\n'.join([x for x in d[k] if not re.match('[ ]*----[-]*[ ]*^', x)])
        l = re.split('\n[ ]*\n', d[''], 1)
        d['short'] = l[0].strip()
        try:
            d['long'] = l[1].strip()
        except:
            d['long'] = ''
        self.update(d)

    def toWiki(self, allSymbols=[]):
        s = tmpl['desc']
        for k in self.keys():
            s = s.replace('@'+k.upper()+'@', self[k].rstrip())
        return s


class MyDocFileDict(dict):

    def __init__(self, mystr, myclass, mymatch=''):
        if not mystr.rstrip():
            return
        mylist = split_sections(do_undent(mystr.split('\n')), mymatch)
        for k in mylist.keys():
            if k:
                a = myclass(k, '\n'.join(mylist[k]))
                if not a['name'][0] == '_':
                    self[a['name']] = a

    def __repr__(self):
        mystr = self.__class__.__name__ + '(' 
        mystr += repr(super(self.__class__, self))
        mystr += ')'
        return mystr
    
    def toWiki(self, allSymbols=[], tmpl='', inClass=False):
        if not len(self):
            return ''
        s = ''
        for x in self.keys():
            s += self[x].toWiki(allSymbols, inClass=inClass)
        if tmpl:
            return tmpl.replace('@DATA@', s.rstrip(' \n'))
        else:
            return s

    def symbols(self):
        return self.keys()


class MyDocFileDataDict(MyDocFileDict):
    
    def __init__(self, mystr):
        super(self.__class__, self).__init__(mystr, MyDocFileData)

    def toWiki(self, allSymbols=[], tmpl=tmpl['datalist'], inClass=False):
        return super(self.__class__, self).toWiki(allSymbols, tmpl=tmpl, inClass=inClass)
        

class MyDocFileFunctionDict(MyDocFileDict):
    
    def __init__(self, mystr):
        super(self.__class__, self).__init__(mystr, MyDocFileFunction)

    def toWiki(self, allSymbols=[], tmpl=tmpl['funclist'], inClass=False):
        return super(self.__class__, self).toWiki(allSymbols, tmpl=tmpl, inClass=inClass)


class MyDocFileClassDict(MyDocFileDict):
    
    def __init__(self, mystr):
        super(self.__class__, self).__init__(mystr, MyDocFileClass, 'class ')

    def toWiki(self, allSymbols=[], tmpl=tmpl['classlist'], inClass=False):
        return super(self.__class__, self).toWiki(allSymbols, tmpl=tmpl, inClass=True)

    def symbols(self):
        l = [] #self.keys()
        for x in self.keys():
            l += self[x].symbols()
        return l
            

class MyDocFileObj(dict):

    def __repr__(self):
        mystr = self.__class__.__name__ + '('
        mystr += dict.__repr__(self)
        mystr += ')'
        return mystr

    def toWiki(self, allSymbols=[], inClass=False):
        return repr(self)+'<br>\n'

    def symbols(self):
        return [self['name']]


class MyDocFileData(MyDocFileObj):
    
    def __init__(self, mystr, dummy=''):
        l = mystr.strip().split('=', 1)
        self['name'] = l[0].strip()
        self['v'] = l[1].strip()
        self['desc'] = ''

    def toWiki(self, allSymbols=[], inClass=False):
        return "%s = %s  # %s\n" % (self['name'], self['v'], self['desc'])


class MyDocFileFunction(MyDocFileObj):
    
    def __init__(self, myhead, mystr):
        self['name'] = myhead.strip().split('(', 1)[0]
        if self['name'] == myhead.strip():
            self['name'] = myhead.strip().split(' ', 1)[0]
            self['args'] = myhead.strip().split(' ', 1)[1]
        else:
            self['args'] = myhead.strip().split('(', 1)[1].rstrip(' )')
        self['desc'] = MyDocFileDataDesc(mystr)

    def toWiki(self, allSymbols=[], inClass=False):
        s = tmpl['func']
        if inClass:
            s = tmpl['method']
        return s\
            .replace('@NAME@', self['name'])\
            .replace('@ARGS@', self['args'])\
            .replace('@DESC@', do_indent(self['desc'].toWiki(allSymbols).strip()))\
            .replace('@SHORT@', self['desc']['short'].strip())\
            .replace('@SEEALSO@', get_seealso(self['desc']['seealso'], allSymbols))
        ## return "%s(%s)\n%s\n" % (self['name'], self['args'], self['desc'].toWiki(allSymbols))

class MyDocFileClass(MyDocFileObj):
    
    def __init__(self, myhead, mystr):
        class_sec_head = {
            'methods'  : 'Methods defined here:',
            'attr'     : 'Data and other attributes defined here:',
            'methods2' : 'Methods inherited from ',
            'attr2'    : 'Data and other attributes inherited from ',
            'junk'     : '----------------------------------------------------------------------',
            'junk2'     : 'Method resolution order:',
            'junk3'     : 'Data descriptors inherited from '
            }
        self['name'] = myhead.strip().split(' ',1)[1].split('(',1)[0]
        try:
            self['super'] = myhead.strip().split('(',1)[1].rstrip(' )')
        except:
            self['super'] = ''
        a = split_sections_title(do_undent(mystr.split('\n'), ' |'), class_sec_head)
        self['desc'] = MyDocFileDataDesc('\n'.join(a['']))
        self['attr'] = MyDocFileDataDict('\n'.join(a['attr'] + a['attr2']))
        for x in self['attr'].keys():
            if x[0] == '_':
                del self['attr'][x]
        self['methods'] = MyDocFileFunctionDict('\n'.join(a['methods'] + a['methods']))
        self['args'] = ''
        for x in self['methods'].keys():
            #self['methods'][x]['args'] = re.sub('^self[ ]*,[ ]*','',self['methods'][x]['args'].strip())
            self['methods'][x]['args'] = ', '.join([y.strip() for y in self['methods'][x]['args'].split(',')][1:])
            if x[0] == '_':
                if x == '__init__':
                    self['args'] = self['methods'][x]['args']
                    for y in self['methods'][x]['desc'].keys():
                        if y in self['desc'].keys():
                            self['desc'][y] += self['methods'][x]['desc'][y]
                        else:
                            self['desc'][y] = self['methods'][x]['desc'][y]
                del self['methods'][x]
            self['methods'][x]['name'] = self['name']+'.'+self['methods'][x]['name']

    def toWiki(self, allSymbols=[], inClass=False):
        return tmpl['class']\
            .replace('@SUPER@', self['super'])\
            .replace('@ARGS@', self['args'])\
            .replace('@ATTR@', self['attr'].toWiki(allSymbols, tmpl=tmpl['attrlist']).rstrip(' \n'))\
            .replace('@METH@', self['methods'].toWiki(allSymbols, tmpl=tmpl['methodlist'], inClass=True))\
            .replace('@DESC@', do_indent(self['desc'].toWiki(allSymbols).strip(' \n')))\
            .replace('@SHORT@', self['desc']['short'].strip())\
            .replace('@NAME@', self['name'])\
            .replace('@SEEALSO@', get_seealso(self['desc']['seealso'], allSymbols))

    def symbols(self):
        return [self['name']] \
            + [self['name']+'.'+x for x in self['attr']] \
            + [self['name']+'.'+x for x in self['methods']]


class MyDocFile(MyDocFileObj):

    def __init__(self, name):
        self['data'] = MyDocFileDataDict('')
        self['func'] = MyDocFileFunctionDict('')
        self['class'] = MyDocFileClassDict('')
        self['desc'] = MyDocFileDataDesc('')
        self['tree'] = ''
        self['name'] = name
        self['raw'] = re.sub('.\b', '', pydoc.render_doc(name))
        d = split_sections(self['raw'].split('\n'))
        if 'DATA' in d.keys():
            self['data'] = MyDocFileDataDict('\n'.join(d['DATA']))
        if 'FUNCTIONS' in d.keys():
            self['func'] = MyDocFileFunctionDict('\n'.join(d['FUNCTIONS']))
        if 'CLASSES' in d.keys():
            for x in do_undent(d['CLASSES']):
                if re.match(r'[ ]*class ', x):
                    break
                self['tree'] += x + '\n'
            self['tree'] = self['tree'].rstrip()
            self['class'] = MyDocFileClassDict('\n'.join(d['CLASSES']))
        if 'DESCRIPTION' in d.keys():
            self['desc'] = MyDocFileDataDesc('\n'.join(d['DESCRIPTION']))
            self['name1'] = d['NAME'][0].strip()
        else:
            x = d['NAME'][0].split('-', 1)
            self['name1'] = x[0].strip()
            self['desc'] = MyDocFileDataDesc(x[1].strip())
        self['file'] = d['FILE'][0].strip()

    def toWiki(self, allSymbols=[]):
        d = {
            '@MODULE@': self['name'],
            '@DESC@':'', '@CLASS@':'', '@FUNC@':'', '@DATA@':''
            }
        if self['desc']:
            d['@DESC@'] = self['desc'].toWiki(allSymbols)
        if self['class']:
            d['@CLASS@'] = self['class'].toWiki(allSymbols)
        if self['func']:
            d['@FUNC@'] = self['func'].toWiki(allSymbols)
        if self['data']:
            d['@DATA@'] = self['data'].toWiki(allSymbols)
        if self['tree']:
            d['@TREE@'] = self['tree']
        s = tmpl['head']
        for k in d.keys():
            s = s.replace(k, d[k])
        return s

    def symbols(self):
        return [self['name']] \
            + [self['name']+'.'+x for x in self['data'].symbols()] \
            + [self['name']+'.'+x for x in self['func'].symbols()] \
            + [self['name']+'.'+x for x in self['class'].symbols()]

    def symbolsDict(self):
        name2 = self['name'].replace('.','/')
        d = {self['name'] : name2}
        for x in self['data'].symbols() + self['func'].symbols() + self['class'].symbols():
            d[self['name']+'.'+x] = name2+'#'+x
        return d


if __name__ == "__main__":
    moduleNames = [
        'rpn_helpers',
        'Fstdc',
        'rpnstd',
        'rpnpy.ftnnml',
        'rpnpy.openanything',
        'rpnpy.rpndate',
        'rpnpy.librmn.all',
        'rpnpy.librmn.base',
        'rpnpy.librmn.const',
        'rpnpy.librmn.grids',
        'rpnpy.librmn.fstd98',
        'rpnpy.librmn.interp',
        'rpnpy.librmn.llacar',
        'rpnpy.librmn.proto'
        ]
    docdir = './doc/'
    linkPrefix = 'Python-RPN/2.0/'
    m = []
    allSymbolsDict = {}
    for x in moduleNames:
        curFile = x
        y = MyDocFile(x)
        m += [y]
        allSymbolsDict.update(y.symbolsDict())
    allSymbolsDictKeys = allSymbolsDict.keys()
    curFile = ''
    for x in m:
        curFile = x['name']
        filename = docdir+curFile+'.txt'
        print("Producing doc for: %s" % curFile)
        try:
            fd = open(filename, "wb")
            try:
                fd.write(x.toWiki(allSymbolsDictKeys))
            except IOError:
                raise IOError(" Oops! Cannot wrtie to file: %s" % (filename))
            finally:
                fd.close()
        except IOError:
            raise IOError(" Oops! Cannot open file: %s" % (filename))
