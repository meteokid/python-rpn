 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
Set of common verif tools
"""
import os
import shutil
import sys


def get_var_from_dot_script(path, vardict):
    """
    """
    cmd = ". {} >/dev/null".format(path)
    kv = list(vardict.items())
    for k,v in vardict.items():
        cmd += "; echo ${{{0}:-{1}}}".format(k,v)
    linesout = run_shell_cmd(cmd)
    for i in range(len(kv)):
        vardict[kv[i][0]] = linesout[i].strip()
    return vardict


def run_shell_cmd(cmd):
    """Run/execute a command as a sub shell process

    Args:
       cmd : (str) command and args to send execute
    Returns:
       list, stdout of the shell command
    """
    import subprocess
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         close_fds=True)
    #TODO: get stderr as well
    f_i, f_o = (p.stdin, p.stdout)
    #f_i, f_o = os.popen4(cmd);
    lines = f_o.readlines()
    return lines


def read_okvalues_file(filename):
    """Check values min, max, mean for specific var...

    Args:
        filename : name of the "dictionary" file for expected min max values
                   Format, CSV, one var per line: VARNAME, MINVAL, MAXVAL
    Returns:
        dict, expected min max values
             { varname : (minval, maxval), ... }
    """
    okvalues = None
    try:
        fd = open(filename, "r")
        try:
            #TODO: decode --  line.decode(sys.stdin.encoding)
            okvalues = [line.strip().split(',') for line in fd
                        if not (len(line.rstrip()) and line.rstrip().startswith('#'))]
        finally:
            fd.close()
    except IOError:
        raise IOError(" Oops! File does not exist or is not readable: {0}".
                      format(filename))
    if okvalues:
        okvalues = dict((a[0].strip().upper(), (float(a[1]),float(a[2])))
                        for a in okvalues if len(a) >= 3)
    return okvalues


def elementTreeIndent(elem, level=0, spacer="   "):
    """
    """
    i = "\n" + level*spacer
    j = "\n" + (level-1)*spacer
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + spacer
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            elementTreeIndent(subelem, level+1, spacer)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        for k,v in elem.items():
            del(elem.attrib[k])
            elem.attrib[i + spacer + k] = v
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


def rm_rf(path):
    """
    """
    #TODO: add a follow links option
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.unlink(path)


def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)


if __name__ == "__main__":
    from pprint import pprint
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
