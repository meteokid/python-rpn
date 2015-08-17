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


def getstats(myfile,myusers,myversions,mymonths,ufilter,vfilter,lmergeminor):
    """Parse a stat file to count users and version

    Stat file format:
       Text file, One line per use, Each line with the following format:
       YYYY-MM-DD : VERSIONS : USERNAME@HOSTNAME : BASE_ARCH
    """
    try:
        fd = open(myfile,"rb")
        try:     rawdata = fd.readlines()
        finally: fd.close()
    except IOError:
        raise IOError(" Oops! File does not exist or is not readable: %s" % (myfile))
    for myline in rawdata: 
        if not myline.strip(): continue
        items = myline.strip().split(':')
        if len(items) < 3:
            sys.stderr.write('Warning: Ignoring supsicious line (n): '+myline.strip()+'\n')
            continue
        
        month    = getmonth(items[0])
        versions = getversionlist(items[1],vfilter,lmergeminor)
        user     = getusername(items[2],ufilter)

        if user is None:
            sys.stderr.write('Warning: Ignoring supsicious line (u): '+myline.strip()+'\n')
            continue
        if len(versions) == 0 and len(vfilter) == 0:
            sys.stderr.write('Warning: Ignoring supsicious line (v): '+myline.strip()+'\n')
            continue
        
        for v in versions:
            if v.replace('@','').strip() != v.strip():
                sys.stderr.write('Warning: Ignoring supsicious line (@): '+myline.strip()+'\n')
                break
            if (not v.strip() in ['?.?','?.?.?','GEM/x']) and re.sub('[0-9]','',v.strip()) == v.strip():
                sys.stderr.write('Warning: Ignoring supsicious version (9): '+v.strip()+'\n')
                continue
            
            if not user in myusers.keys():
                myusers[user] = {}
            if not v in myusers[user].keys():
                myusers[user][v] = 0
            myusers[user][v] += 1
            
            if not month in mymonths.keys():
                mymonths[month] = {}
            if not v in mymonths[month].keys():
                mymonths[month][v] = 0
            mymonths[month][v] += 1

            if not v in myversions.keys():
                myversions[v] = {}
            if not user in myversions[v].keys():
                myversions[v][user] = 0
            myversions[v][user] += 1
    return myusers,myversions,mymonths


def printstats(mymsg,myarray,ldetails,ufilter,vfilter):
    print mymsg
    if ufilter or vfilter:
        print '# Filter: user=',ufilter,'; version=',vfilter
    (total,onetime) = (0,0)
    keys  = sorted(myarray.keys())
    for k in keys:
        subtotal = sum([myarray[k][i] for i in myarray[k].keys()])
        if subtotal == 1: onetime += 1
        total += subtotal
        print '%8s = %6d' % (k,subtotal),
        if not ldetails:
            print
            continue
        vlist = sorted(myarray[k].keys())
        for v in vlist:
            print ':%8s @ %-8s = %6d' % (k,v,myarray[k][v]),
        print
    return total,onetime


if __name__ == "__main__":
    usage = "usage: \n\t%prog [-a] [-u] [-v] [-m] [-d] [--filter-user=PATTERN] [--filter-version=PATTERN] FILENAMES"
    parser = optparse.OptionParser(usage=usage)
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

    myusers    = {}
    myversions = {}
    mymonths   = {}
    for myfile in args:
        (myusers,myversions,mymonths) = getstats(myfile,myusers,myversions,mymonths,ufilterlist,vfileterlist,options.lmergeminor)
    (total,onetime) = (0,0)
    
    if options.lusers:
        (total,onetime) = printstats('#==== Per User Stats ====',myusers,options.ldetails,options.ufilter,options.vfilter)
    if options.lversions: 
        (total1,onetime1) = printstats('#==== Per Version Stats ====',myversions,options.ldetails,options.ufilter,options.vfilter)
        if total == 0: (total,onetime) = (total1,onetime1)
    if options.lmonths: 
        (total1,onetime1) = printstats('#==== Per Month Stats ====',mymonths,options.ldetails,options.ufilter,options.vfilter)
        if total == 0: (total,onetime) = (total1,onetime1)

    print '#==== Stats Summary ===='
    if options.ufilter or options.vfilter:
        print '# Filter: user=',options.ufilter,'; version=',options.vfilter
    print 'Total Version = ',len(myversions)
    print 'Total User    = ',len(myusers), "(onetime user=",onetime,')'
    print 'Total Usage   = ',total
        
