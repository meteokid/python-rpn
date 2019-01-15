#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
"""
import os.path
import shutil
import sys
import xml.etree.ElementTree as ET
import verif_common as _vc

_MAX_MEMBERS = 999


def maestro_exp_clean(spath=None, keep=[], date='2009042700'):
    """
    """
    if spath is None:
        spath = os.getenv('SEQ_EXP_HOME').strip()
    if not spath:
        raise IOError('spath not provided and SEQ_EXP_HOME not set.')
    if not os.path.isdir(spath):
         raise IOError('No such dir, spath: '+str(spath))
    #TODO: should we also clean hub and listing dir?
    for x in ('modules', 'resources', 'logs', 'sequencing'):
        d = os.path.join(spath,x)
        sys.stdout.write('Cleaning: '+d+'\n') #TODO: use logging module
        for a in os.listdir(d):
            if a not in keep:
                b = os.path.join(d,a)
                sys.stdout.write('Removing: '+b+'\n') #TODO: use logging module
                _vc.rm_rf(b)


def maestro_set_date(spath=None, date='2009042700'):
    """
    """
    for x in ('nodelog','toplog'):
        x2 = date + '0000_' + x
        sys.stdout.write('Creates: logs/' + x2 + '\n') #TODO: use logging module
        d = os.path.join(spath, 'logs', x2)
        _vc.touch(d)


def maestro_exp_set_wj_links(spath=None):
    """
    """
    if spath is None:
        spath = os.getenv('SEQ_EXP_HOME').strip()
    if not spath:
        raise IOError('spath not provided and SEQ_EXP_HOME not set.')
    if not os.path.isdir(spath):
         raise IOError('No such dir, spath: '+str(spath))
    truehost = os.getenv('ORDENV_TRUEHOST').strip()
    for subdir in ('hub', 'listings'):
        hpath = os.path.join(spath,subdir)
        thpath = os.path.join(hpath,truehost)
        #TODO: check if thpath exists
        wjpath = os.path.join(hpath,'wj')
        if os.path.islink(thpath):
            _vc.rm_rf(wjpath)
            os.symlink(os.path.join('.',truehost), wjpath)


class Mobject(object):
    """
    Maestro Object
    """
    baseIndent = ".   "  # " "*3

    def __init__(self, name, allowedSubClass=None, maxMembers=0):
        self.name    = name
        self.dep     = []  # For DEPENDS_ON sumbit clause
        self.parent  = None
        self.members = []
        self.allowedSubClass = allowedSubClass \
            if allowedSubClass is not None else Mobject
        self.maxMembers = maxMembers
        self.resOkKeys  = None
        self.iscontainer = True
        self.res = {}
        self.res_elem = []

    def toStr(self, indent=0):
        mystr = '{2}{0}("{1}") <{3}> '.format(self.__class__.__name__, self.name,
                                       self.baseIndent*indent, self.getNodesPathStr())
        if self.maxMembers != 0:
            mystr += '[\n' + \
                ",\n".join([m.toStr(indent+1)
                            for m in self.members]).rstrip(',\n') + ']'
        return mystr

    def __repr__(self):
        return self.toStr()

    def add(self, obj):
        ## print("{} Add: {}".format(self.__class__.__name__, repr(obj)))
        if len(self.members) >= self.maxMembers:
             raise IndexError("[{}] Oops, trying to add too many members: {}/{}"
                              .format(self.__class__.__name__,
                                      len(self.members), self.maxMembers))
        if not (self.allowedSubClass and isinstance(obj, self.allowedSubClass)):
           raise TypeError("[{}] Oops, trying to add an unsupported object: {}"
                            .format(self.__class__.__name__, type(obj)))
        obj.parent = self
        self.members.append(obj)

    def addDep(self, obj):
        self.dep.append(obj)

    def getNodesList(self, nodes=[]):
        nodes += [self]
        if self.parent:
            return self.parent.getNodesList(nodes)
        return nodes

    def getNodesPathStr(self, basepath='/'):
        nodes = self.getNodesList(nodes=[])
        nodesname = [n.name for n in reversed(nodes)]
        return os.path.join(basepath, os.path.join(*nodesname))

    def getSubmitNodes(self, mtype):
        e1 = ET.Element("SUBMITS", {'sub_name': self.name})
        e2 = ET.Element(mtype, {'name': self.name})
        for dep in self.dep:
            ## deppath = dep.getNodesPathStr()
            deppath = './'+dep.name
            e2.append(ET.Element("DEPENDS_ON", {'dep_name' : deppath,
                                                'type' : "node",
                                                'status' : "end"}))
        return [e1, e2]

    def setRes(self, res, container=False):
        if self.resOkKeys is None:
            res1 = res.copy()
        else:
            res1 = dict((k,v) for k,v in res.items() if k in self.resOkKeys)
        if container == self.iscontainer:
            self.res.update(res1)
            if 'queue' in self.res and (
                ('immediate' in self.res and str(self.res['immediate']) == '1') or
                ('machine' in self.res and str(self.res['machine']).lower() == 'wj')
                ):
                del(self.res['queue'])
        for m in self.members:
            m.setRes(res1, container)

    def writeRes(self, basepath='/'):
        if len(self.res):
            resfile = self.getNodesPathStr(basepath)
            if self.iscontainer:
                resfile = os.path.join(resfile,'container')
            resfile += '.xml'
            if not os.path.isdir(os.path.dirname(resfile)):
                os.makedirs(os.path.dirname(resfile))
            root  = ET.Element("NODE_RESOURCES")
            batch = ET.SubElement(root, "BATCH")
            for k,v in self.res.items():
                batch.set(k, str(v))
            for e in self.res_elem:
                root.append(e)
            root = _vc.elementTreeIndent(root)
            tree = ET.ElementTree(root)
            tree.write(resfile)
        for m in self.members:
            m.writeRes(basepath)

    def writeFlow(self, basepath='/'):
        for m in self.members:
            m.writeFlow(basepath)

    def cfgPath(self, basepath=None):
        if basepath is None:
            spath = os.getenv('SEQ_EXP_HOME').strip()
            basepath = os.path.join(spath, 'modules')
        cfgfile = self.getNodesPathStr(basepath)
        if self.iscontainer:
            cfgfile = os.path.join(cfgfile,'container')
        cfgfile += '.cfg'
        if not os.path.isdir(os.path.dirname(cfgfile)):
            os.makedirs(os.path.dirname(cfgfile))
        return cfgfile

    def writeCfgMine(self, basepath='/'):
        pass  # Needs to be implemented by subclass

    def writeCfg(self, basepath='/'):
        self.writeCfgMine(basepath)
        for m in self.members:
            m.writeCfg(basepath)

    def getFlowNodes(self):
        raise Error('Needs to be implemented by subclass')


class Mtask(Mobject):
    """
    Maestro Task
    """
    def __init__(self, name):
        Mobject.__init__(self, name)
        self.iscontainer = False
        self.res = {
            "catchup"    : "1",
            "cpu"        : "1",
            "cpu_multiplier": "1",
            "immediate"  : "${TESTS_IMMEDIATE}",
            "machine"    : "${FRONTEND}",
            "memory"     : "1G",
            "soumet_args": "-waste 100",
            "wallclock"  : "10"
            }

    #TODO: def writeTsk()

    def getFlowNodes(self):
        return self.getSubmitNodes('TASK')


class MtaskCopy(Mtask):
    """
    Maestro Task with pre-existings task file and resources
    """
    def __init__(self, name, moduleSrc, moduleDst):
        Mtask.__init__(self, name)
        self._copyTask(moduleSrc, moduleDst)

    def _copyTask(self, moduleSrc, moduleDst):
        spath = os.getenv('SEQ_EXP_HOME').strip()
        # Copy Task File
        tpath = os.path.join(spath, 'modules', moduleSrc, self.name) + '.tsk'
        if not os.path.isfile(tpath):
            raise IOError("[{}] Oops, no such task file: {}".
                          format(self.__class__.__name__, tpath))
        tpath2 = os.path.join(spath, 'modules', moduleDst, self.name) + '.tsk'
        if not os.path.isdir(os.path.dirname(tpath2)):
            os.makedirs(os.path.dirname(tpath2))
        try:
            shutil.copyfile(tpath, tpath2)
        except IOError:
            raise # pass
        if not os.path.isfile(tpath2):
            raise IOError("[{}] Oops, can't copy task file to: {}".
                          format(self.__class__.__name__, tpath2))
        # Copy cfg File
        cpath = os.path.join(spath, 'modules', moduleSrc, self.name) + '.cfg'
        if os.path.isfile(cpath):
            cpath2 = os.path.join(spath, 'modules', moduleDst, self.name) + '.cfg'
            try:
                shutil.copyfile(cpath, cpath2)
            except IOError:
                raise # pass
            if not os.path.isfile(cpath2):
                raise IOError("[{}] Oops, can't copy task cfg file to: {}".
                              format(self.__class__.__name__, cpath2))
        # Get Resources
        rpath = os.path.join(spath, 'resources', moduleSrc, self.name) + '.xml'
        if os.path.isfile(rpath):
            batch = ET.parse(rpath).getroot()[0]
            for e in ET.parse(rpath).getroot():
                if e.tag.lower() == 'batch':
                    self.setRes(batch.attrib)
                else:
                    self.res_elem.append(e)


class Mmodule(Mobject):
    """
    Maestro Module container
    """
    def __init__(self, name, allowedSubClass=None, maxMembers=_MAX_MEMBERS):
        if allowedSubClass is None:
            allowedSubClass = (Mmodule, Mfamily, Mtask)
        Mobject.__init__(self, name, allowedSubClass, maxMembers)

    def addDep(self, obj):
        raise IndexError("[{}] Oops, trying to add a dep to a module".
                         format(self.__class__.__name__))

    def writeFlow(self, basepath='/'):
        root = ET.Element("MODULE", {'name': self.name})
        for m in self.members:
            m.writeFlow(basepath)
            for e in m.getFlowNodes():
                root.append(e)

        flowfile = os.path.join(basepath, self.name, 'flow.xml')
        if not os.path.isdir(os.path.dirname(flowfile)):
            os.makedirs(os.path.dirname(flowfile))
        root = _vc.elementTreeIndent(root)
        tree = ET.ElementTree(root)
        tree.write(flowfile)

        return self.getSubmitNodes('MODULE')

    def getFlowNodes(self):
        return self.getSubmitNodes('MODULE')


    def setEntryModule(self):
        spath = os.getenv('SEQ_EXP_HOME').strip()
        if not (os.path.isdir(spath) and os.access(spath, os.W_OK)):
            raise IOError('[{}] Oops, SEQ_EXP_HOME not found or not writable: {}'.
                          format(self.__class__.__name__, spath))
        mpath = os.path.join(spath, 'modules', self.name)
        if not os.path.isdir(mpath):
            raise IOError('[{}] Oops, module not found: {}'.
                          format(self.__class__.__name__, mpath))
        lpath = os.path.join(spath, 'EntryModule')
        if os.path.exists(lpath) or os.path.islink(lpath):
            os.unlink(lpath)
        try:
            mpath = os.path.join('modules', self.name)
            os.symlink(mpath, lpath)
        except OSError:
            raise # pass
        if not os.path.islink(lpath):
            raise IOError('[{}] Oops, Was not able to creato link:\n\t{}\n\t\t-> {}'.
                          format(self.__class__.__name__, lpath, mpath))


class MmoduleCopy(Mmodule):
    """
    Maestro Module container with pre-existings list of tasks, resources and flow
    """
    def __init__(self, name):
        Mmodule.__init__(self, name)
        self._copyTasks()

    def _copyTasks(self):
        spath = os.getenv('SEQ_EXP_HOME').strip()
        mpath = os.path.join(spath, 'modules', self.name)
        if not os.path.isdir(mpath):
            raise ValueError("[{}] Oops, no such module dir: {}".
                             format(self.__class__.__name__, mpath))
        rpath0 = os.path.join(spath, 'resources', self.name)
        tasklist = [os.path.splitext(os.path.basename(a))[0]
                    for a in os.listdir(mpath) if a.endswith(".tsk")]
        for a in tasklist:
            t = Mtask(a)
            self.add(t)
            rpath = os.path.join(rpath0, a) + '.xml'
            batch = ET.parse(rpath).getroot()[0]
            for e in ET.parse(rpath).getroot():
                if e.tag.lower() == 'batch':
                    t.setRes(batch.attrib)
                else:
                    t.res_elem.append(e)
        rpath = os.path.join(rpath0, 'container') + '.xml'
        if os.path.isfile(rpath):
            batch = ET.parse(rpath).getroot()[0]
            self.setRes(batch.attrib, container=True)

    def writeFlow(self, basepath='/'):
        for m in self.members:
            m.writeFlow(basepath)


class Mfamily(Mobject):
    """
    Maestro Family container
    """
    def __init__(self, name, allowedSubClass=None, maxMembers=_MAX_MEMBERS):
        if allowedSubClass is None:
            allowedSubClass = (Mmodule, Mfamily)
        Mobject.__init__(self, name, allowedSubClass, maxMembers)

    def getFlowNodes(self):
        e1, e2 = self.getSubmitNodes('FAMILY')
        for m in self.members:
            for e3 in m.getFlowNodes():
                e2.append(e3)
        return [e1, e2]


if __name__ == "__main__":
    from pprint import pprint
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
