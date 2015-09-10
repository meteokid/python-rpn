#!/usr/bin/env python
import os,optparse,sys,re,fnmatch
import datetime


def getmonth(month):
    month = str(month).lstrip().strip()
    if re.match('[12][09][0-9][0-9]-[01][0-9]-[0-3][0-9]',month):
        return month[0:7].replace('-','.')
    elif re.match('([01][0-9]|[0-9])-[0-3][0-9]',month):
        month = int(month.split('-')[0])
        if month > 0 and month < 13:
            return '????.%2.2d' % month
    return '????.??'


def getversionlist(versionstr,vfilter,lmergeminor):
    versionstr = str(versionstr).lstrip().strip()
    versions = versionstr.replace('rpnpy','').replace('GEM/x','').replace('_all','').replace('_','').split(' ')
    versions = [a.strip() for a in versions if a.strip() != '']
        
    if len(versions) > 0:
        versions = sorted(set(versions))
    else:
        versions = ['?.?.?']

    if len(vfilter) > 0:
        versions2 = []
        for v in versions:
            for f in vfilter:
                if fnmatch.fnmatch(v, f):
                    versions2.append(v)
                    break
        versions = versions2

    if lmergeminor:
        versions = ['.'.join(v.split('.')[:-1]) for v in versions]

    return [v for v in versions if v]


def getusername(userstr,ufilter):
    userstr = str(userstr).lstrip().strip()
    user = userstr.split('@')[0].strip()

    if user in ('','whoami'):
        user == '??????'
    elif userstr and user == userstr:
        return None
    
    if len(ufilter) > 0:
        if not (True in [fnmatch.fnmatch(user, u) for u in ufilter]):
            user = ''

    return user


def getstats2(myfile,myarray,ufilter,vfilter,lmergeminor,lverbose):
    """Parse a stat file to count users and version

    Stat file format:
       Text file, One line per use, Each line with the following format:
       YYYY-MM-DD : VERSIONS : USERNAME@HOSTNAME : BASE_ARCH
    """
    try:
        sys.stderr.write('Parsing File: '+myfile+'\n')
        fd = open(myfile,"rb")
        try:     rawdata = fd.readlines()
        finally: fd.close()
    except IOError:
        raise IOError(" Oops! File does not exist or is not readable: %s" % (myfile))
    cnt = 0
    nerrors = 0
    for myline in rawdata:
        cnt += 1
        if cnt == len(rawdata)/50:
            cnt = 0
            sys.stderr.write('.')
        if not myline.strip(): continue
        items = myline.strip().split(':')
        if len(items) < 3:
            nerrors += 1
            if lverbose:
                sys.stderr.write('Warning: Ignoring supsicious line (n): '+myline.strip()+'\n')
            continue
        
        month    = getmonth(items[0])
        versions = getversionlist(items[1],vfilter,lmergeminor)
        user     = getusername(items[2],ufilter)

        if user is None:
            nerrors += 1
            if lverbose:
                sys.stderr.write('Warning: Ignoring supsicious line (u): '+myline.strip()+'\n')
            continue
        if len(versions) == 0 and len(vfilter) == 0:
            nerrors += 1
            if lverbose:
                sys.stderr.write('Warning: Ignoring supsicious line (v): '+myline.strip()+'\n')
            continue
        
        for v in versions:
            if v.replace('@','').strip() != v.strip():
                nerrors += 1
                if lverbose:
                    sys.stderr.write('Warning: Ignoring supsicious line (@): '+myline.strip()+'\n')
                break
            if (not v.strip() in ['?.?','?.?.?','GEM/x']) and re.sub('[0-9]','',v.strip()) == v.strip():
                nerrors += 1
                if lverbose:
                    sys.stderr.write('Warning: Ignoring supsicious version (9): '+v.strip()+'\n')
                continue
            
            if not month in myarray.keys():
                myarray[month] = {}
            if not v in myarray[month].keys():
                myarray[month][v] = {}
            if not user in myarray[month][v].keys():
                myarray[month][v][user] = 0
            myarray[month][v][user] += 1

    sys.stderr.write(' (%d parsing errors)\n' % (nerrors))
    return myarray


STATMODS = {
    'm,v' : 0,
    'v,u' : 1,
    'u,v' : 2,
    'm,u' : 3,
    }
def arrayreduce(myarray,imode,luserNotUsage):
    """
    myarray[month][version][user] = usage
    imode = 0 : [month][version]
    imode = 1 : [version][user]
    imode = 2 : [user][version]
    imode = 3 : [month][user]
    """
    mytot    = 0
    myarray0 = {}
    myarray1 = {}
    myarray2 = {}
    for m in myarray.keys():
        for v in myarray[m].keys():
            for u in myarray[m][v].keys():
                if imode == STATMODS['m,v']:
                    (k1,k2) = (m,v)
                elif imode == STATMODS['v,u']:
                    (k1,k2) = (v,u)
                elif imode == STATMODS['u,v']:
                    (k1,k2) = (u,v)
                elif imode == STATMODS['m,u']:
                    (k1,k2) = (m,u)

                if luserNotUsage:
                    myarray0.update(myarray[m][v])
                else:
                    mytot += myarray[m][v][u]
                   
                if k1 not in myarray1.keys():
                    if luserNotUsage:
                        myarray1[k1] = {}
                    else:
                        myarray1[k1] = 0
                if luserNotUsage:
                    myarray1[k1].update(myarray[m][v])
                else:
                    myarray1[k1] += myarray[m][v][u]
                
                if k1 not in myarray2.keys():
                    myarray2[k1] = {}
                if k2 not in myarray2[k1].keys():
                    if luserNotUsage:
                        myarray2[k1][k2] = {}
                    else:
                        myarray2[k1][k2] = 0
                if luserNotUsage:
                    myarray2[k1][k2].update(myarray[m][v])
                else:
                    myarray2[k1][k2] += myarray[m][v][u]

    if luserNotUsage:
        mytot = len(myarray0.keys())
        for m in myarray1.keys():
            myarray1[m] = len(myarray1[m].keys())
            for k1 in myarray2[m].keys():
                myarray2[m][k1] = len(myarray2[m][k1].keys())

    return (mytot,myarray1,myarray2)


def printstats2(mymsg,myarray,ldetails,ufilter,vfilter,imode,luserNotUsage):
    """
    myarray[month][version][user] = usage
    imode = 0 : month version
    imode = 1 : version user
    imode = 2 : user  version
    imode = 3 : month user
    """
    print mymsg
    if ufilter or vfilter:
        print '# Filter: user=',ufilter,'; version=',vfilter

    (mytot,myarray1,myarray2) = arrayreduce(myarray,imode,luserNotUsage)

    onetime = 0
    k1list = sorted(myarray2.keys())
    for k1 in k1list:
        print '%8s = %6d' % (k1,myarray1[k1]),        
        if  myarray1[k1] == 1:
            onetime += 1
        if not ldetails:
            print
            continue
        k2list = sorted(myarray2[k1].keys())
        for k2 in k2list:
            #print ':%8s @ %-8s = %6d' % (k1,k2,myarray2[k1][k2]),
            print '[%-6s = %6d]' % (k2,myarray2[k1][k2]),
        print
    
    return (mytot,onetime,len(myarray1.keys()),len(myarray2.keys()))


if __name__ == "__main__":
    usage = "usage: \n\t%prog [-a] [-u] [-v] [-m] [-d] [--filter-user=PATTERN] [--filter-version=PATTERN] FILENAMES"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-V","--verbose",dest="lverbose",action="store_true",
                      help="Print parsing error messages")
    parser.add_option("-a","--all",dest="lall",action="store_true",
                      help="Print all stats (as -u -v -d -m)")
    parser.add_option("-u","--users",dest="lusers",action="store_true",
                      help="Print stats by users")
    parser.add_option("-v","--versions",dest="lversions",action="store_true",
                      help="Print stats by versions")
    parser.add_option("","--merge-minor",dest="lmergeminor",action="store_true",
                      help="Merge stats for minor versions numbers (last digit)")
    parser.add_option("-m","--monthly",dest="lmonths",action="store_true",
                      help="Print stats by months")
    parser.add_option("-d","--details",dest="ldetails",action="store_true",
                      help="Print details",metavar="DEDETAILS")
    parser.add_option("","--filter-user",dest="ufilter",default='',
                      help="Filter stats for matching user",metavar="'PATTERN1 PATTERN2 ...'")
    parser.add_option("","--filter-version",dest="vfilter",default='',
                      help="Filter stats for matching version",metavar="'PATTERN1 PATTERN2 ...'")
    (options,args) = parser.parse_args()
    if len(args) == 0:
        sys.stderr.write('\nError: You need to provide at least a stat file.\n')
        parser.print_help()
        sys.exit(1)
    if options.lall:
        options.lusers = options.ldetails = options.lversions = options.lmonths = True
    ufilterlist  = [i for i in options.ufilter.strip().split(' ') if i]
    vfileterlist = [i for i in options.vfilter.strip().split(' ') if i]

    myarray    = {}
    for myfile in args:
        myarray = getstats2(myfile,myarray,ufilterlist,vfileterlist,options.lmergeminor,options.lverbose)
    (total,onetime) = (0,0)

    totaluser = '?'
    onetimeuser = '?'
    totalusage = '?'
    totalversion = '?'
    totalmonth = '?'

    if options.lusers:
        (totalusage,onetimeuser,n1,totaluser) = printstats2('#==== Per User Usage Stats ====',myarray,options.ldetails,options.ufilter,options.vfilter,STATMODS['u,v'],False)

    if options.lversions: 
        (totalusage,onetime1,n1,totalversion) = printstats2('#==== Per Version Usage Stats ====',myarray,options.ldetails,options.ufilter,options.vfilter,STATMODS['v,u'],False)
         
    if options.lversions: 
        (total,onetime1,n1,n1) = printstats2('#==== Per Version User Stats ====',myarray,options.ldetails,options.ufilter,options.vfilter,STATMODS['v,u'],True)

    if options.lmonths: 
        (totalusage,onetime1,n1,totalmonth) = printstats2('#==== Per Month Usage Stats ====',myarray,options.ldetails,options.ufilter,options.vfilter,STATMODS['m,v'],False)

    if options.lusers and options.lmonths:
        (totaluser,onetime1,n1,totalmonth) = printstats2('#==== Per Month User Stats ====',myarray,options.ldetails,options.ufilter,options.vfilter,STATMODS['m,v'],True)

    print '#==== Stats Summary ===='
    if options.ufilter or options.vfilter:
        print '# Filter: user=',options.ufilter,'; version=',options.vfilter
    print 'Nb of Months  = ',totalmonth
    print 'Total Version = ',totalversion
    print 'Total User    = ',totaluser, "(onetime user=",onetimeuser,')'
    print 'Total Usage   = ',totalusage

