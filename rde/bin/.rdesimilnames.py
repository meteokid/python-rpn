#!/usr/bin/env python
import os,string,optparse,difflib
from glob import glob

def listfiles(folder,recurse=True):
    for root, folders, files in os.walk(folder,followlinks=True):
        if recurse or root==folder:
            for filename in files:
                yield os.path.join(root, filename)

if __name__ == "__main__":
    usage = "usage: \n\t%prog [-R] NAME PATHLIST"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-R","--recurse",dest="recurse",action="store_true",
                      help="Recursively search down list of PATH sub dirs",metavar="KEY")
    (options,args) = parser.parse_args()
    myname = args[0]
    mydirlist = args[1:]
    mymatches = []
    for mydir in mydirlist:
        filelist = []
        mymatches0 = []
        for myfile in listfiles(mydir,options.recurse):
            mybasename = os.path.basename(myfile)
            (myname0, myext) = os.path.splitext(mybasename)
            if mybasename[0:4] != '.rm.' and myext not in ('.o','.mod'):
                if os.path.basename(myname) in [myname0,myname0+'.',mybasename] or myname == myfile:
                    mymatches0.append(mybasename)
                else:
                    if mybasename not in filelist:
                        filelist.append(mybasename)
        if (len(mymatches0)<=0):
            for myfile in glob(mydir+'/'+myname+'*'):
                mybasename = os.path.basename(myfile)
                (myname0, myext) = os.path.splitext(mybasename)
                if mybasename[0:4] != '.rm.' and myext not in ('.o','.mod'):
                    mymatches0.append(mybasename)
        if (len(mymatches0)<=0):
            mymatches0 += difflib.get_close_matches(myname,filelist)#,n=9,cutoff=0.5
        mymatches += mymatches0
    c = set(mymatches) #sorted(set(a))
    d = [os.path.basename(item) for item in c]
    e = []
    for item in d:
        if (myname in (os.path.splitext(item)[0],item)):
            e.append(item)
    if (len(e)<=0):
        print(string.join(d,' '))
    else:
        print(string.join(e,' '))
