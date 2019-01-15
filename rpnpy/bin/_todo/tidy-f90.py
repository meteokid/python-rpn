#!/usr/bin/env python
"""
Tidy up F90 source code...
may want to work with the tidy program
http://www.unb.ca/fredericton/science/chem/ajit/tidy/
then do a second pass to adjust to local stuff

USAGE:
    tidy.f90 -i filename.ftn90 > filename-new.ftn90
"""
__author__ = 'Stephane Chamberland (stephane.chamberland@ec.gc.ca)'
__version__ = '$Revision: 1.0 $'[11:-2]
__date__ = '$Date: 2010/09/10$'
__copyright__ = 'Copyright (c) 2005 Stephane Chamberland'
__license__ = 'LGPL'

import sys
import getopt
import re
import string

from rpnpy.openanything import openAnything


def indent_f90(myfilestr):
    
    indent_str0 = 3*' '
    indent_cont_str0 = 5*' '

    prefix = r'^[ \t]*([1-9]+|[a-z]+[a-z1-9]*:)*[ \t]*'
    
    openkw = re.compile(prefix+r'\b(subroutine|function|[a-z]+[ \t]+function|module|interface|do|else|case|forall|where)\b',re.I)
    ifkw     = re.compile(prefix+r'\bif\b',re.I)
    elseifkw = re.compile(prefix+r'\belseif\b',re.I)
    thenkw   = re.compile(r'\bthen[ \t]*$',re.I)
    closekw  = re.compile(r'^[ \t]*([1-9]+[ \t]+)?(end|enddo|else|endif|[1-9]+[ \t]+continue)\b',re.I)

    myfilelines = myfilestr.splitlines()
    i = 0
    indent = 0
    iscont = 0
    isif   = 0
    for myline in myfilelines:
        if not iscont:
            if re.search(closekw, myline):
                indent =  max(0,indent - 1)
            isif   = 0
            if re.search(ifkw,myline):
                isif = 1
            if re.search(elseifkw,myline):
                indent =  max(0,indent - 1)
                isif = 1
        
        indentstr = indent*indent_str0
        if iscont:
            indentstr = indentstr + indent_cont_str0
        myfilelines[i] = re.sub(r'^[ \t]*',indentstr,myline)

        if not iscont and re.search(openkw, myline):
            indent =  indent + 1
            
        if isif and re.search(thenkw,myline):
            indent =  indent + 1

        iscont = 0
        if re.search(r'&[ \t]*',myline):
            iscont = 1
        i = i+1

    #TODO: special case for number labels
    
    return string.join(myfilelines,'\n')


def pull_comm_str(myfilestr):
    """Extract comments and string blocks
    """
    
    commentlist = []
    stringslist = []
    
    myfilelines = myfilestr.splitlines()
    i = 0
    for myline in myfilelines:
        ipos = 0
        mynewline = myfilelines[i]
        isend = 0
        while not isend:
            
            #find markers
            try:
                spos = string.index(mynewline,"'",ipos)
            except:
                spos = len(mynewline)
            try: 
                spos2 = string.index(mynewline,'"',ipos)
            except:
                spos2 = len(mynewline)
            spos = min(spos,spos2)
            try:
                cpos = string.index(mynewline,"!",ipos)
            except:
                cpos = len(mynewline)
            ipos = min(spos,cpos)
            
            #update pull out comments and strings
            if ipos < len(mynewline):
                if ipos == cpos:
                    commstr = mynewline[ipos:]
                    if re.match(r'![ \t]*$',commstr): #rm empty comments
                        commstr = ''
                    if commstr and re.match(r'[ \t]*!',mynewline):
                        ii = len(stringslist)
                        stringslist.append((i,commstr[1:],'!',''))
                        mynewline = mynewline[:ipos] + '!' + str(ii) + '!'
                    else:
                        commentlist.append((i,commstr))
                        mynewline = mynewline[:ipos]
                    isend = 1
                else:
                    strchar = mynewline[ipos]
                    ipos = ipos + 1
                    spos = string.find(mynewline,strchar,ipos)
                    if (len(mynewline[spos:]) == 1):
                        isend = 1
                    #TODO: if spos == -1 error
                    #TOTO: check escape char
                    ii = len(stringslist)
                    stringslist.append((i,mynewline[ipos:spos],strchar,strchar))
                    mynewline0 = mynewline[:ipos] + str(ii)
                    ipos = len(mynewline0)+1
                    mynewline = mynewline0 + mynewline[spos:]
            else:
                isend = 1
                    
        myfilelines[i] = mynewline
        i = i+1
        
    return (string.join(myfilelines,'\n'),commentlist,stringslist)

    
def push_comm_str(myfilestr,commentlist,stringslist):
    """re-put comments and string blocks
    """
    i = 0
    for (myline,mystr,myquote,myquote2) in stringslist:
    #print(re.sub(pattern,replace,myfilestr))
        myfilestr = re.sub(myquote+str(i)+myquote,myquote+mystr+myquote2,myfilestr)
        i = i+1

    myfilelines = myfilestr.splitlines()
    for (myline,mycomm) in commentlist:
        if mycomm:
            myfilelines[myline] = myfilelines[myline] + '  ' + mycomm
            
    return string.join(myfilelines,'\n')


def tidy_f90(filename):
    """echo a tidy up version of a f90 program

    tidy_f90(filename)
    """
    
    emptypattern = [
        # - merge multiple empty lines (empty comment == empty line)
        (r'\n[ \t]*![ \t]*\n',"\n\n"),
        (r'\n([ \t]*\n)+',    "\n\n")
        ]
        
    #Define regex
    patternlist = [
        # - remove trailing blanks
        (r'[ \t]+\n',         "\n"),
        # - remove extra head blanks in comments (more than 3)
        (r'\n[ \t]*![ \t][ \t][ \t]+',"\n!   "),
        # - remove extra spaces around parentesis [except after if]
        (r'[ \t]*\([ \t]*',   "("),
        (r'[ \t]*\)',         ")"),
        (r'if\(',             "if ("),
        # - max one space after comma (none before)
        (r'[ \t]+,',          ","),    #TODO: should not catch '\n[ \t]+\('
        (r',[ \t]+',          ", "),
        # - at least one space before trailing &
        (r'&[ \t]*\n',        " &\n"),  #TODO: should not add one if there
        # - update doc format
        (r'!author','!@author'),
        (r'!revision','!@revisions'),
        (r'!arguments','!@arguments'),
        (r'!object','!@description'),
        (r'\n!\*\*[ \t]*s/r[ \t]*','\n!/**\n!@objective'),
        (r'\n!\*[ \t]*\n','\n!**/\n'),
        # - misc
        (r'end[ \t]+do','enddo'),
        (r'end[ \t]+if','endif'),
        (r'else[ \t]+if','elseif')
        ]

    # - replace .eq. (and other) notation, one space around, lowercase
    oprlist = [ (r'\.eq\.',r'=='),
                (r'\.ne\.',r'/='),
                (r'\.gt\.',r'>'),
                (r'\.ge\.',r'>='),
                (r'\.lt\.',r'<'),
                (r'\.le\.',r'<=')
                ]

    postfixlist = [ (r'\b[ \t]*\.or\.[ \t]*\b',' .or. '),
                    (r'\b[ \t]*\.and\.[ \t]*\b',' .and. '),
                    (r'[ \t]*=[ \t]*',' = '),
                    (r'\b\([ \t]*len[ \t]*=[ \t]*\b','(len='),
                    (r'\b\([ \t]*kind[ \t]*=[ \t]*\b','(kind='),
                    (r'= +=','=='),
                    (r'/ +=','/='),
                    (r'> +=','>='),
                    (r'< +=','<='),
                    (r'= +>','=>')
                    ]

    pattern = []
    replace = []
    
    for (opr1,opr2) in patternlist:
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
    
    for (opr1,opr2) in oprlist:
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
        
    # - one and only on space around = and other operators
    for (opr1,opr2) in oprlist: #TODO:not in comments/string
        pattern.append(re.compile(r'[ \t]*'+opr2+r'[ \t]*',re.I))
        replace.append(' '+opr2+' ')

    
    for (opr1,opr2) in postfixlist:
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
        
        
    #read file
    fsock = openAnything(filename)
    myfilestr = fsock.read()

    #s = re.sub("(?<= )(.+)(?= )", lambda m: "can use a callable for the %s text too" % m.group(1), s)
    #print(re.sub(pattern,replace,myfilestr))

    (myfilestr,commentlist,stringslist) = pull_comm_str(myfilestr)

    i = 0
    for mypatt in pattern:
        myfilestr = re.sub(mypatt,replace[i],myfilestr)
        i=i+1

    myfilestr = indent_f90(myfilestr)

    myfilestr = push_comm_str(myfilestr,commentlist,stringslist)

    pattern = []
    replace = []
    for (opr1,opr2) in emptypattern:
        pattern.append(re.compile(opr1,re.I))
        replace.append(opr2)
 
    i = 0
    for mypatt in pattern:
        myfilestr = re.sub(mypatt,replace[i],myfilestr)
        i=i+1
        
    #TODO:
    # - replace licence
    # - last end should with function/sub name
    # - lower case [but not parameters]
    # - external and parameters... now attributes
    # - split too long lines
    # - add () on subroutine name and call sub when missing

    return myfilestr

def usage():
    """Print usage."""
    print(__doc__)

    
def  main(argv):
    """Inline arguments parsing before call to printrpnftndoc"""
    filename = ""
    sections = []
    verbose=False
    listdic =False
    list = []
    try:
        opts, args = getopt.getopt(argv, \
            "Hi:v", \
            ["help","filename=","verbose"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-H", "--help"):
            usage()
            sys.exit(1)
        elif opt in ("-i","--filename"):
            filename = arg
        elif opt in ("-v","--verbose"):
            verbose = True
    #Get namelist opt/val
    if (filename):
        print(tidy_f90(filename))
    else:
        usage()
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])

def toto():
    a="""!
!notempty
a=b !
c='!et' !2e
    """
    (a,b,c) = pull_comm_str(a)
    print(a,b,c)
    a = push_comm_str(a,b,c)
    print(a)
