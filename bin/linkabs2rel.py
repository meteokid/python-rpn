#!/usr/bin/python
import os,string,sys,optparse

def getCommonPrefix(path1,path2):
    #path1 and path2 must be absolute path to dir or files
    path1l=string.split(path1,'/')
    path2l=string.split(path2,'/')
    cnt=0
    common='/'
    while cnt < len(path1l) and cnt < len(path2l) and path1l[cnt] == path2l[cnt]:
        common = os.path.join(common,path1l[cnt])
        cnt = cnt+1
    if common != '/':
        common=os.path.normpath(common)+'/'
    return common

def getRelPath(fromdir,todir,nodotslash=1):
    #make sure they are absolute path:
    #fromdir = os.path.realpath(os.path.abspath(fromdir))
    #todir   = os.path.realpath(os.path.abspath(todir))
    fromdir = os.path.abspath(fromdir) + '/'
    todir   = os.path.abspath(todir) + '/'
    common  = getCommonPrefix(fromdir,todir)
    #common  = os.path.commonprefix((fromdir,todir))
    fromdir = os.path.normpath(string.replace(fromdir,common,'',1))

    todir   = os.path.normpath(string.replace(todir,common,'',1))
    pathprefix=''
    while fromdir:
        if fromdir != '.':
            pathprefix=os.path.join(pathprefix,'..')
        fromdir=os.path.split(fromdir)[0]
    reltopath = os.path.join(pathprefix,todir)
    if nodotslash:
        if reltopath[0:2]=='./':
            reltopath = reltopath[2:]
        elif reltopath=='.':
            reltopath = ''
    #print 'getRelPath: ' +  reltopath 
    return reltopath


def linkabstorel(mypath,mytest,myverbose):
    mylkname = os.path.abspath(mypath)
    mytarget = os.path.abspath(os.readlink(mypath))
    
    ##     if os.path.isdir(mypath):
    ##         mylkname = mylkname+'/'
    ##         mytarget = mytarget+'/'

    (mylkname0,mylkname1) = os.path.split(mylkname)
    (mytarget0,mytarget1) = os.path.split(mytarget)
    
    ##     print "mypath   : ",mypath
    ##     print "mylkname : ",mylkname
    ##     print "mytarget : ",mytarget
    ##     print " : "
    ##     print "mylkname0 : ",mylkname0
    ##     print "mylkname1 : ",mylkname1
    ##     print "mytarget0 : ",mytarget0
    ##     print "mytarget1 : ",mytarget1
    ##     print " : "
    ##     print "mylkname0 : ",os.path.realpath(mylkname0)
    ##     print "mytarget0 : ",os.path.realpath(mytarget0)
    ##     print " : "
    ##     print "relpath   : ",getRelPath(mytarget0,mylkname0)
    ##     print "relapth2  : ",getRelPath(os.path.realpath(mytarget0),
    ##                            os.path.realpath(mylkname0))
    
    myrelpath = getRelPath(os.path.realpath(mylkname0),
                           os.path.realpath(mytarget0))
    #might check if relpath is shorter this following way
    #myrelpath = getRelPath(mylkname0,mytarget0)
    
    mytargetnew = os.path.normpath(os.path.join(myrelpath,mytarget1))
    
    if not mytest:
        os.unlink(mylkname)
        os.chdir(mylkname0)
        os.symlink(mytargetnew,mylkname1)
    if myverbose:
        print "From ",mypath," --> ",os.readlink(mypath)
        print "To   ",mypath," --> ",mytargetnew

#===============================================================
if __name__ == "__main__":

    # Command line arguments
    desc="""Convert filesystem links from abs-path to rel-path if possible"""
    usage = """
    %prog [-v] [-n] LINKNAME"""
    parser = optparse.OptionParser(usage=usage,description=desc)
    parser.add_option("-v","--verbose",dest="verbose",action="store_true",
                      help="Verbose mode")
    parser.add_option("-n","--dry-run",dest="dryrun",action="store_true",
                      help="Performe a dry ryn, no printing")
    (options,args) = parser.parse_args()
    if not len(args):
        parser.print_help()
        sys.exit(1)   

    myverbose  = options.verbose
    mytest     = options.dryrun
    if mytest:
        myverbose = 1

    for mylinkpath in args:
        if os.path.exists(mylinkpath):
            if os.path.islink(mylinkpath):
                if os.path.isabs(os.readlink(mylinkpath)):
                    linkabstorel(os.path.abspath(mylinkpath),mytest,myverbose)
                else:
                    if myverbose: print mylinkpath," Link already relative"
            else:
                if myverbose: print mylinkpath, " Not a link"
        else:
            if myverbose: print mylinkpath," Not such file or directory"


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
