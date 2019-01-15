#!/usr/bin/env python
"""Extract Fortran namlist specially crafted info from source code
   Usage:
"""

import sys
import re
import argparse
import textwrap

def getnmldoc(inFile, nmlName):
    ## read file
    try:
        fd = open(inFile,"r")
        try:     alllines = fd.readlines()
        finally: fd.close()
    except IOError:
        raise IOError(" Oops! File does not exist or is not readable: %s" % (inFile))

    renml = re.compile(r'^\s*namelist\s*/\s*([^\s/]+)\s*/\s*([^\s]+)',re.I)
    redecl = re.compile(r'^\s*([^\s:]+)\s*::\s*([^\s=]+)\s*=\s*([^\s].*)',re.I)

    nml = []
    for lineno in range(len(alllines)):
        myline = alllines[lineno].strip()
        match = re.match(renml, myline)
        if match:
            nmlname1 = match.group(1).lower()
            nmlvar  = match.group(2).lower()
            if nmlname1 == nmlName:
                myline1 = alllines[lineno-1].strip()
                match2  = re.match(redecl, myline1)
                vartype, varname, varval = '', '', ''
                if match2:
                    vartype, varname, varval = match2.groups()
                desc = []
                i = -2
                while lineno+i > 0:
                    myline1 = alllines[lineno+i].lstrip()
                    if myline1[0:2] != '!#':
                        break
                    i = i - 1
                    desc = [myline1[2:].lstrip().rstrip()] + desc
                nml.append({
                    'nml'  : nmlname1,
                    'name' : nmlvar,
                    'type' : vartype,
                    'val'  : varval,
                    'desc' : "\n".join(desc)
                    })
    return nml


def nmldoc2wiki(nmlName, nml):
    print(textwrap.dedent("""\
        === {} Namelist  ===

        {{| class="wikitable"
        |-
        ! style="width: 10em;" | Name
        ! style="width: 40em;" | Description
        ! style="width: 10em;" | Default Value
        ! Type
        """.format(nmlName)))
    for myvar in nml:
        print("""
|-
| {name} ||
{0}
| {val} || {type}""".format(myvar['desc'].replace("|",":"),**myvar))
    print("|}")



def nmldoc2md(nmlName, nml):
    print(textwrap.dedent("""\
        ### {} Namelist

        | Name          | Description            |  Default Value | Type |
        | ------------- | ---------------------- | -------------- | ---- |"""
        .format(nmlName)))
    for myvar in nml:
        print("| {name} | {0} | {val} | {type} |".
              format(myvar['desc'].replace("\n*","\n-").replace("\n","<br>").replace("|",":"), **myvar))
    print("\n")


if __name__ == "__main__":
    desc="Produce doc (wiki or markdown) from Fortran Namelist source code"
    usage = "%(prog)s -i filename -n namelist [options]"
    epilog=""""
    Source code must have the following format:

       !# Some Multi-lines description
       !# of the option, with optional list
       !# * item1 of list
       !# * item2 of list
       TYPE :: VARNAME = DEFAULT_VALUE
       namelist / NAMELIST_NAME / VARNAME
   """

    parser = argparse.ArgumentParser(
        description=desc, usage=usage, epilog=epilog,
        prefix_chars='-+', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", dest="inputFile",
                        required=True, type=str, default=None,
                        help="Input fortran source code file")
    parser.add_argument("-n", "--nml", dest="nmlName",
                        required=True, type=str, default=None,
                        help="Namelist Name")
    parser.add_argument("--md", dest="md",
                        action="store_true",
                        help="Produce markdown instead of wiki markup")
    parser.add_argument("--sort", dest="sort",
                        action="store_true",
                        help="Sort Nml Var alphabetically")
    args = parser.parse_args()

    nml = getnmldoc(args.inputFile, args.nmlName.lower())
    if args.sort:
        d = dict([(v['name'], v) for v in nml])
        k=d.keys()
        nml = [d[k] for k in sorted(d.keys())]
    ## for var in nml:
    ##     print("/{nml}/{name} [{type}::{name}={val}]".format(**var))
    if args.md:
        nmldoc2md(args.nmlName.lower(), nml)
    else:
        nmldoc2wiki(args.nmlName.lower(), nml)

