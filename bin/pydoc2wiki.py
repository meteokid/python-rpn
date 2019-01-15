#!/usr/bin/env python
"""
Long

Short


"""

import sys
import re
import pydoc
import inspect
import pprint
pp = pprint.PrettyPrinter(indent=4)

## topnotes="""
## {{roundbox|
## The functions described below are a very close ''port'' from the original [[librmn]]'s [[Librmn/FSTDfunctions|FSTD]] and [[Vgrid]] packages.<br>
## You may want to refer to the [[Librmn/FSTDfunctions|FSTD]] and [[Vgrid]] documentation for more details.
## }}
## """
topnotes=""
linkPrefix = 'Python-RPN/2.1/'
tmpl = {}
tmpl['head'] = """
[[Category:python]]
__NOTITLE__
= Python RPN: @MODULE@ =
{{:Python-RPN/2.1/navbox}}
{| style='background-color:white; border: 0px #fff solid; width=82%;'>
|-
|
@DESC@
@NOTES@
@EXAMPLE@
@SEEALSO@
__TOC__
|}
{| style='background-color:white; border: 0px #fff solid; width=82%;'>
|-
|
@DETAILS@
@CLASS@
@FUNC@
@DATA@
"""

tmpl['funclist'] = """== Functions ==
{| class='wikitable'
@SUMMARY@
|}
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

tmpl['funcalias'] = """=== @NAME1@ ===
Function <tt>@NAME1@</tt> is an alias for function [[#@NAME2@|@NAME2@]]
"""

tmpl['func'] = """=== @NAME@ ===
Function <tt>@NAME@</tt>: @SHORT@
<source lang="python">
@NAME@(@ARGS@):
   '''
@DESC@
   '''
</source>
@EXAMPLE@
@NOTES@
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
@EXAMPLE@
@NOTES@
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
@EXAMPLE@
@NOTES@
@SEEALSO@
@ATTR@
@METH@
"""

def get_warning(mystr):
    if not mystr.strip():
        return ''
    mystrlist = mystr.strip().split('\n')
    s = '{{warning|\n'
    s += mystrlist[0]+'\n'
    if len(mystrlist) > 1:
        s += '\n'.join(do_undent(mystrlist[1:]))
    s += '\n}}\n'
    return s

def get_notes(mystr):
    if not mystr.strip():
        return ''
    mystrlist = mystr.strip().split('\n')
    s = '{{roundboxtop}}\n'
    s += mystrlist[0]+'\n'
    if len(mystrlist) > 1:
        s += '\n'.join(do_undent(mystrlist[1:]))
    s += '\n{{roundboxbot}}\n'
    return s

def get_example(mystr):
    if not mystr.strip():
        return ''
    s = 'Examples:\n'
    s += '<source lang="python">\n'
    s += mystr.strip().replace('\n','\n#').replace('\n#... ','\n').replace('\n#...','\n').replace('\n#>>> ','\n').replace('\n#>>>','\n').replace('>>> ','').replace('>>>','').replace('\n##','\n#')
    s += '</source>\n'
    return s

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
            if myline[0:len(v)].lower() == v.lower():
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
            'seealso'  : 'See Also',
            'example'  : 'Examples:',
            'notes'    : 'Notes:',
            'warning'  : 'Warning:',
            'details'  : 'Details:',
            }
        self[''] = mystr
        self['short'] = mystr.strip('# ')
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
        d['short'] = l[0].strip('# ')
        try:
            d['long'] = l[1].strip('# ')
        except:
            d['long'] = ''
        self.update(d)

    def toWiki(self, allSymbols=[]):
        s = tmpl['desc']
        for k in self.keys():
            s = s.replace('@'+k.upper()+'@', self[k].rstrip('# '))
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
        ## mystr += repr(super(self.__class__, self))
        mystr += dict.__repr__(dict(self))
        mystr += ')'
        return mystr

    def toWiki(self, allSymbols=[], tmpl='', inClass=False):
        if not len(self):
            return ''
        s = ''
        for x in sorted(self.keys()):
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
        mystr = super(self.__class__, self).toWiki(allSymbols, tmpl=tmpl, inClass=inClass)
        s = ''
        for x in sorted(self.keys()):
            o = self[x]
            s += '|-\n'
            s += "| [[#%s|%s]] || %s \n" % (o['name'], o['name'], o['desc']['short'].strip())
        return mystr.replace('@SUMMARY@', s.rstrip(' \n#'))


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
        mystr += dict.__repr__(dict(self))
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
        names = self['name'].split('=',1)
        if len(names) > 1:
            s = tmpl['funcalias']
            return s\
                .replace('@NAME1@', names[0].strip())\
                .replace('@NAME2@', names[1].strip())

        s = tmpl['func']
        if inClass:
            s = tmpl['method']

        return s\
            .replace('@NAME@', self['name'])\
            .replace('@ARGS@', self['args'].replace('lambda ',''))\
            .replace('@DESC@', do_indent(self['desc'].toWiki(allSymbols).strip()))\
            .replace('@SHORT@', self['desc']['short'].strip('# '))\
            .replace('@EXAMPLE@', get_example(self['desc']['example']))\
            .replace('@NOTES@', get_warning(self['desc']['warning']) + get_notes(self['desc']['notes']))\
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
            .replace('@EXAMPLE@', get_example(self['desc']['example']))\
            .replace('@NOTES@', get_warning(self['desc']['warning']) + get_notes(self['desc']['notes']))\
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
        self['synmols'] = __import__(name,fromlist=[name.split('.')[-1]])
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
        self['desc'] = MyDocFileDataDesc(self['synmols'].__doc__)
        self['file'] = d['FILE'][0].strip()

    def toWiki(self, allSymbols=[]):
        d = {
            '@MODULE@': self['name'],
            '@DESC@':'', '@CLASS@':'', '@FUNC@':'', '@DATA@':'',
            '@EXAMPLE@':'', '@NOTES@':'', '@SEEALSO@':'',
            '@DETAILS@':'', '@TREE@':''
            }
        dodata = True
        if self['desc']['details']:
            if self['desc']['details'].strip().split('\n')[0] == 'See Source Code':
                dodata = False
                try:
                    fd = open(self['file'],"rb")
                    try:     rawdata = fd.readlines()
                    finally: fd.close()
                except IOError:
                    raise IOError(" Oops! File does not exist or is not readable: {0}".format(self['file']))
                ok = False
                for l in rawdata:
                    if l.strip() == '##DETAILS_START':
                        ok = True
                    elif l.strip() == '##DETAILS_END':
                        ok = False
                    elif ok:
                        if l[0] == '#':
                            d['@DETAILS@'] += l[1:]
                        else:
                            d['@DETAILS@'] += l
            else:
                d['@DETAILS@'] = self['desc']['details']
        if self['desc']:
            d['@DESC@'] = self['desc'].toWiki(allSymbols)
        if dodata and self['class']:
            d['@CLASS@'] = self['class'].toWiki(allSymbols, tmpl=tmpl['classlist'].replace('@TREE@',self['tree']))
        if dodata and self['func']:
            d['@FUNC@'] = self['func'].toWiki(allSymbols)
        if dodata and self['data']:
            d['@DATA@'] = self['data'].toWiki(allSymbols)
        if dodata and self['tree']:
            d['@TREE@'] = self['tree']
        if self['desc']['example']:
            d['@EXAMPLE@'] = get_example(self['desc']['example'])
        if self['desc']['notes']:
            d['@NOTES@'] = get_warning(self['desc']['warning']) + get_notes(self['desc']['notes'])
        if self['desc']['seealso']:
            d['@SEEALSO@'] = get_seealso(self['desc']['seealso'], allSymbols)

        s = tmpl['head']
        ## pp.pprint(d)
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
        'rpnpy.librmn',
        'rpnpy.librmn.all',
        'rpnpy.librmn.base',
        'rpnpy.librmn.const',
        'rpnpy.librmn.grids',
        'rpnpy.librmn.fstd98',
        'rpnpy.librmn.interp',
        'rpnpy.librmn.burp',
        'rpnpy.librmn.burp_const',
        'rpnpy.utils.fstd3d',
        'rpnpy.utils.burpfile',
        'rpnpy.utils.llacar',
        'rpnpy.utils.thermoconsts',
        'rpnpy.utils.thermofunc',
        'rpnpy.utils.tdpack',
        'rpnpy.utils.tdpack_consts',
        'rpnpy.librmn.proto',
        'rpnpy.librmn.proto_burp',
        'rpnpy.vgd',
        'rpnpy.vgd.all',
        'rpnpy.vgd.base',
        'rpnpy.vgd.const',
        'rpnpy.vgd.proto',
        'rpnpy.burpc',
        'rpnpy.burpc.all',
        'rpnpy.burpc.base',
        'rpnpy.burpc.brpobj',
        'rpnpy.burpc.const',
        'rpnpy.burpc.proto',
        ]
    docdir = './doc/'
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
        print("Producing doc for: {}".format(curFile))
        try:
            fd = open(filename, "wb")
            try:
                fd.write(x.toWiki(allSymbolsDictKeys))
            except IOError:
                raise IOError(" Oops! Cannot write to file: {}".format(filename))
            finally:
                fd.close()
        except IOError:
            raise IOError(" Oops! Cannot open file: ()".format(filename))
