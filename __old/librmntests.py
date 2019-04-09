#!/usr/bin/env python
from librmn.proto import *
from librmn.const import *
from librmn.base import *
from librmn.fstd98 import *

print c_fst_version()

value = FSTOPS_MSG_DEBUG
istat = fstopt(FSTOP_MSGLVL,value,FSTOP_GET)
istat = fstopt(FSTOP_MSGLVL,value,FSTOP_SET)

value = FSTOPS_MSG_FATAL
istat = fstopt(FSTOP_TOLRNC,value,FSTOP_GET)
istat = fstopt(FSTOP_TOLRNC,value,FSTOP_SET)
istat = fstopt(FSTOP_TOLRNC,value,FSTOP_GET)

filename = '2009042700_000'

print wkoffit(filename)
print isFST(filename)

iunit = fnom(filename,FST_RO);
istat = fstouv(iunit,FST_RO)

#istat = c_fstvoi(iunit,'')
print c_fstnbrv(iunit)
ni=-1;nj=-1;nk=-1
istat = fstinf(iunit)
print 'fstinf=',istat
key = istat['key']
istat = fstprm(key)
print 'fstprm=',istat

keylist = fstinl(iunit,nomvar='tt')
print 'fstinl=',len(keylist),keylist
myrec = fstluk(keylist[0])
istat = fstecr(iunit,myrec['d'],myrec)

istat = fstfrm(iunit);
istat = fclos(iunit);


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
