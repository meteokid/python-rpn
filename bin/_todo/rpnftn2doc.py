#!/usr/bin/env python
"""Extract RPN style doc from a Fortran file

   Usage:
       rpnftn2doc.py -i filename [-s sections] [-l sub_number_list] [-v] [-d]
           filename: File to parse for doc strings
           sections: comma separated list of sections
               if none specified: output the full doc section
           sub_number_list: comma separated list subs (position number)
                            to displya info about
           -v : verbose
           -d : list all avail sections and number of sub/func present in file

   If used as a module, rpnftn2doc.py, implement 2 functions:
       getrpnftndoc   : extract in a list of dictionaries
                        the parsed doc string, one dic per sub/func found
       printrpnftndoc : print the doc string requested sections
       (see the doc string (__doc__) of the above fn for more details)

   The extracted source code doc must be of the form:

   ***[spf] Some desc (optional)
   * Some more desc (optional)
   * (optional empty lines)
   Some code (optional)
   *section_tag
   * Some more desc (optional)
   * section comments (optional)
   Some section code (optional)
   ** (end of desc section)

   For fortran90 code, the 1st '*' is replaced by a '!'

   Change log:
       version 2:
           - support multiple sub/fn in a single file (multiple doc sections)
           - fix for functions
       version 3:
           - generalize tags (section names)
           - f90 comment style support

   To-Do:
       - do not catch r'^[*!]\*\*.*' inside the code
       - catch doc section end with forgotten r'^[*!]*.*' ending

"""

__author__ = 'Stephane Chamberland (stephane.chamberland@ec.gc.ca)'
__version__ = '$Revision: 1.3 $'[11:-2]
__date__ = '$Date: 2005/09/05 21:16:24 $'
__copyright__ = 'Copyright (c) 2005 Stephane Chamberland'
__license__ = 'LGPL'

import sys
import getopt
import re

from rpnpy.openanything import openAnything

def getrpnftndoc(filename):
    """Extract in a list of dictionaries the parsed RPN style doc string
    of a Fortran file, one dic per sub or fn of prog

    getrpnftndoc(filename)
    """

    #Define regex
    refunc = re.compile(r'^\s+([a-zA-Z][a-zA-Z0-9_*]+\s+function)\s+([a-zA-Z][a-zA-Z0-9_]+).*',re.I)
    resub = re.compile(r'^\s+(subroutine|program)\s+([a-zA-Z][a-zA-Z0-9_]+).*',re.I)
    retag = re.compile(r'^[*!]([a-zA-Z]+).*',re.I)
    rebgn = re.compile(r'^[*!]\*\*([a-zA-Z].*)',re.I)
    reend = re.compile(r'^[*!]\*.*',re.I)

    ignorelist = [
        re.compile(r'^#include\s+<model_macros_f\.h>',re.I),
        re.compile(r'^#include\s+[\'"]impnone.cdk[\'"]',re.I),
        re.compile(r'^\*[cv]dir\s+',re.I),
        re.compile(r'^(\*|\s*!)?\s*$',re.I)
    ]

    #read file
    fsock = openAnything(filename)
    a = fsock.read()
    mylist = a.splitlines()

    #initialize var for loop
    indoc = False
    insec = None
    counter = 0
    linecounter = 0
    mydocdic = [{
        'filename':filename,
        'subname':'',
        'startline':0,
        'all':''
        },
    ]

    #loop through lines
    for myline in mylist:
        linecounter += 1
        keepline = True
        ignore = False
        for reitem in ignorelist:
            if re.search(reitem,myline,re.I):
                ignore = True
        if indoc and not ignore:
            mydocdic[counter]['all'] += myline+"\n"
            if re.search(retag,myline,re.I):
                insec = (re.sub(retag,r'\1',myline)).lower()
                keepline = False
            elif re.search(resub,myline,re.I):
                insec = 'main'
                mydocdic[counter]['subname'] = \
                    re.sub(resub,r'\2',myline)
                mydocdic[counter]['startline'] = linecounter
            elif re.search(refunc,myline,re.I):
                insec = 'main'
                mydocdic[counter]['subname'] = \
                    re.sub(refunc,r'\2',myline)
                mydocdic[counter]['startline'] = linecounter
            #elif re.search(r'^\s+',myline): #code line
            #    pass
            elif re.search(r'^[*cC!]\s+\W+',myline): #in section comment line
                pass
            elif re.search(r'^[^*cC!]\s*\W+',myline): #code line
                pass
            elif re.search(reend,myline): #end of doc section
                indoc = False
                keepline = False
            if insec and keepline:
                if insec in mydocdic[counter].keys():
                    mydocdic[counter][insec] += str(myline)+"\n"
                else:
                    mydocdic[counter][insec] = myline+"\n"
        elif re.search(rebgn,myline):
            indoc = True
            insec = 'title'
            mydocdic[counter]['all'] += myline+"\n"
            mydocdic[counter][insec] = myline+"\n"
            #if len(subsecs)<1 or insec in subsecs:
            #    print insec,myline
    return mydocdic


def printrpnftndoc(filename, verbose=False, sublist=[], \
    subsecs=[], listdic=False):
    """call getrpnftndoc to extract the parsed RPN style doc string
    of a Fortran file and print desired subsections.

    printrpnftndoc(filename,verbose=False,sublist=[],subsecs=[])
        sublist: list of sub numbers to display info about
        subsecs list items accepted values:
            title,main,author,revision,object,arguments,implicits
    """
    mydocdic = getrpnftndoc(filename)
    sublist2 = sublist
    if len(sublist)<1:
         sublist2 = range(len(mydocdic))
    for subcnt in sublist2:
        subcnt = int(subcnt)
        if subcnt <= len(mydocdic):
            if verbose:
                print "!!!Found "+mydocdic[subcnt]['subname']+" in file "+mydocdic[subcnt]['filename']+" at line "+str(mydocdic[subcnt]['startline'])
            if listdic:
                print '!!!Avail. sec for sub/func/prog #',subcnt,'(',mydocdic[subcnt]['subname'],')',mydocdic[subcnt].keys()
            else:
                if len(subsecs)<1:
                    print mydocdic[subcnt]['all'].replace("\n\n","\n").strip()
                else:
                    for item in subsecs:
                        try:
                            print str(mydocdic[subcnt][item]).replace("\n\n","\n").strip()
                        except:
                            pass


def usage():
    """Print usage."""
    print __doc__


def main(argv):
    """Inline arguments parsing before call to printrpnftndoc"""
    filename = ""
    sections = []
    verbose=False
    listdic =False
    list = []
    try:
        opts, args = getopt.getopt(argv, \
            "Hi:s:l:vd", \
            ["help","filename=","sections=","list=","verbose","dic"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-H", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-i","--filename"):
            filename = arg
        elif opt in ("-s","--sections"):
            sections = arg.split(",")
        elif opt in ("-v","--verbose"):
            verbose = True
        elif opt in ("-d","--dic"):
            listdic = True
        elif opt in ("-l","--list"):
            list = arg.split(",")
    #Get namelist opt/val
    if (filename):
        printrpnftndoc(filename,verbose=verbose,sublist=list,subsecs=sections,listdic=listdic)
    else:
        usage()
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
