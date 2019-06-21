#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
This page is a collection of recipes from our own material and from other users.
You may freely copy the code and build upon it.

Please feel free to leave different examples from your own code in the talk/discussion section of this page. It may be useful to others and we'll try to include them below.

These example assumes you already know the python language version 2.* basics and are familiar with the numpy python package.
If it is not already the case you may head to:
* Dive into python : for a good introductory tutorial on python
* python2 official doc : for more tutorial and reference material on the python2 language
* Numpy official doc : for more tutorial and reference material on the numpy python package

Before you can use the python commands, you need to load the python module into your shell session environment and python session.
The rest of the tutorial will assume you already did this.

  PYVERSION="$(python -V 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)"
  . s.ssmuse.dot ENV/py/${PYVERSION}/rpnpy/???

Every script should start by importing the needed classes and functions.
Here we load all the librmn functions and constants into a python object. The name rmn is given to imported objects as a shortcut.

  python
  >>> import rpnpy.librmn.all as rmn
  >>> import rpnpy.burpc.all as brp

See Also:
  * RPNpy Tutorial
  * RPNpy Reference
  * http://iweb.cmc.ec.gc.ca/~afsdcvs/burplib_c/
"""

import unittest

import os
import sys
from rpnpy import integer_types as _integer_types
from rpnpy import C_WCHAR2CHAR as _C_WCHAR2CHAR
from rpnpy import C_CHAR2WCHAR as _C_CHAR2WCHAR
from rpnpy import C_MKSTR as _C_MKSTR

if sys.version_info < (3, ):
    range = xrange
else:
    long = int

RPNPY_NOLONGTEST = os.getenv('RPNPY_NOLONGTEST', None)

class RpnPyBurpcTests(unittest.TestCase):

    #==== Example 1 =============================================

    def sub_test_ex1_read1(self, logfile="tmp/test_ex1_read1.log"):
        """burplib_c iweb doc example 1"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        iunit = 999
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        logfid.write("enreg {}\n".format(istat))
        bs = brp.c_brp_newblk()
        br = brp.c_brp_newblk()
        rs = brp.c_brp_newrpt()
        rr = brp.c_brp_newrpt()
        brp.RPT_SetHANDLE(rs,0)
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) >= 0:
                logfid.write("stnid = {}\n".format(brp.RPT_STNID(rr)))
                brp.BLK_SetBKNO(bs, 0)
                while brp.c_brp_findblk(bs, rr) >= 0:
                    if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) >= 0:
                        logfid.write("block btyp = {}\n".format(brp.BLK_BTYP(br)))
        istat = brp.c_brp_close(iunit)
        brp.c_brp_freeblk(bs)
        brp.c_brp_freeblk(br)
        brp.c_brp_freerpt(rs)
        brp.c_brp_freerpt(rr)
        if logfid != sys.stdout: logfid.close()


    def sub_test_ex1_read1_py(self, logfile="test_ex1_read1_py.log"):
        """burplib_c iweb doc example 1"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        with brp.BurpcFile(infile) as bfile:
            logfid.write("enreg {}\n".format(len(bfile)))
            for rr in bfile:
                logfid.write("stnid = {}\n".format(rr.stnid))
                for br in rr:
                    logfid.write("block btyp = {}\n".format(br.btyp))
        if logfid != sys.stdout: logfid.close()


    def test_ex1_read1_cmp(self):
        """burplib_c iweb doc example 1, compare results from 2 itf"""
        import os
        import filecmp
        if len(testlist) > 0 and not '1' in testlist:
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex1_read1.log")
        logfile2 = os.path.join(TMPDIR, "test_ex1_read1_py.log")
        self.sub_test_ex1_read1(logfile=logfile1)
        self.sub_test_ex1_read1_py(logfile=logfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2, shallow=False))

    #==== Example 2 =============================================

    def sub_test_ex2_readburp(self, infile=None,
                          logfile="test_ex2_readburp.log"):
        """burplib_c iweb doc example 2"""
        import os, sys
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        if not infile:
            infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                                  'bcmk_burp/2007021900.brp')
        iunit = 999
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        logfid.write("Nombre Enreg = {}\n".format(istat))
        brp.RPT_SetHANDLE(rs,0)
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                continue
            if  brp.RPT_STNID(rr) == '>>POSTALT':
                continue
            ## if  brp.RPT_STNID(rr) != '>>POSTALT': #TODO: check problem with postalt, tblval are different from one run to another
            ##     continue
            logfid.write("""
hhmm   ={:8d} flgs   ={:6d}  codtyp ={:6d}  stnids ={:9s}
blat   ={:8d} blon   ={:6d}  dx     ={:6d}  dy     ={:6d}  stnhgt ={:6d}
yymmdd ={:8d} oars   ={:6d}  runn   ={:6d}  nblk   ={:6d}  dlay   ={:6d}\n
""".format(brp.RPT_TEMPS(rr), brp.RPT_FLGS(rr), brp.RPT_IDTYP(rr),
           brp.RPT_STNID(rr), brp.RPT_LATI(rr), brp.RPT_LONG(rr),
           brp.RPT_DX(rr), brp.RPT_DY(rr), brp.RPT_ELEV(rr),
           brp.RPT_DATE(rr), brp.RPT_OARS(rr), brp.RPT_RUNN(rr),
           brp.RPT_NBLK(rr), brp.RPT_DRND(rr)))
            brp.BLK_SetBKNO(bs, 0)
            while brp.c_brp_findblk(bs, rr) >= 0:
                if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                    continue
                logfid.write("""
blkno  ={:6d}  nele   ={:6d}  nval   ={:6d}  nt     ={:6d}  bit0   ={:6d}
bdesc  ={:6d}  btyp   ={:6d}  nbit   ={:6d}  datyp  ={:6d}  bfam   ={:6d}\n
""".format(brp.BLK_BKNO(br), brp.BLK_NELE(br), brp.BLK_NVAL(br),
           brp.BLK_NT(br), brp.BLK_BIT0(br), brp.BLK_BDESC(br),
           brp.BLK_BTYP(br), brp.BLK_NBIT(br), brp.BLK_DATYP(br),
           brp.BLK_BFAM(br)))
                for k in range(brp.BLK_NT(br)):
                    if brp.BLK_NT(br) != 1:
                        logfid.write("\nobservation {}/{}\n".
                                      format(k+1, brp.BLK_NT(br)))
                    mystr = "lstele ="
                    for i in range(brp.BLK_NELE(br)):
                        mystr += "    {:0>6d}".format(brp.BLK_DLSTELE(br,i))
                    logfid.write(mystr+"\n")
                    for j in range(brp.BLK_NVAL(br)):
                        mystr = "tblval = {} {} {}".format(brp.RPT_STNID(rr),k,j)
                        for i in range(brp.BLK_NELE(br)):
                            mystr += "{:10d}".format(brp.BLK_TBLVAL(br,i,j,k))
                        logfid.write(mystr+"\n")
        istat = brp.c_brp_close(iunit)
        brp.brp_free(bs, br, rs, rr)
        if logfid != sys.stdout: logfid.close()


    def sub_test_ex2_readburp_py(self, infile=None,
                             logfile="test_ex2_readburp.log"):
        """burplib_c iweb doc example 2"""
        import os, sys
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        if not infile:
            infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                                  'bcmk_burp/2007021900.brp')
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        with brp.BurpcFile(infile) as bfile:
            logfid.write("Nombre Enreg = {}\n".format(len(bfile)))
            for rr in bfile:
                if  rr.stnid == '>>POSTALT':
                    continue
                ## if  rr.stnid != '>>POSTALT':
                ##     continue
                logfid.write("""
hhmm   ={temps:8d} flgs   ={flgs:6d}  codtyp ={idtype:6d}  stnids ={stnid:9s}
blat   ={lati:8d} blon   ={longi:6d}  dx     ={dx:6d}  dy     ={dy:6d}  stnhgt ={elev:6d}
yymmdd ={date:8d} oars   ={oars:6d}  runn   ={runn:6d}  nblk   ={nblk:6d}  dlay   ={drnd:6d}\n
""".format(**rr.todict()))
                for br in rr:
                    ## brptr = br.getptr()
                    logfid.write("""
blkno  ={bkno:6d}  nele   ={nele:6d}  nval   ={nval:6d}  nt     ={nt:6d}  bit0   ={bit0:6d}
bdesc  ={bdesc:6d}  btyp   ={btyp:6d}  nbit   ={nbit:6d}  datyp  ={datyp:6d}  bfam   ={bfam:6d}\n
""".format(**br.todict()))
                    n=0
                    for k in range(br.nt):
                        if br.nt != 1:
                            logfid.write("\nobservation {}/{}\n".
                                         format(k+1, br.nt))
                        mystr = "lstele ="
                        for i in range(br.nele):
                            mystr += "    {:0>6d}".format(np.asscalar(br.dlstele[i]))
                        logfid.write(mystr+"\n")
                        for j in range(br.nval):
                            mystr = "tblval = {} {} {}".format(rr.stnid,k,j)
                            for i in range(br.nele):
                                mystr += "{:10d}".format(np.asscalar(br.tblval[i,j,k]))
                                ## a = brp.BLK_TBLVAL(brptr,i,j,k)
                                ## b = np.asscalar(br.tblval[i,j,k])
                                ## mystr += "{:10d}".format(a)
                                ## mystr += "{:10d}".format(b)
                                ## mystr += "[{}]".format(a==b)
                                ## brp.BLK_SetTBLVAL(br.getptr(),i,j,k,n)
                                ## a = brp.BLK_TBLVAL(brptr,i,j,k)
                                ## b = np.asscalar(br.tblval[i,j,k])
                                ## mystr += "{:10d}".format(a)
                                ## mystr += "{:10d}".format(b)
                                ## mystr += "[{}]".format(a==b)
                                ## n += 1
                            logfid.write(mystr+"\n")
                ## break
        if logfid != sys.stdout: logfid.close()


    def test_ex2_readburp_cmp(self):
        """burplib_c iweb doc example 1, compare results from 2 itf"""
        import os
        import filecmp
        if len(testlist) > 0 and not '2' in testlist:
            return
        if RPNPY_NOLONGTEST:
            print("RPNPY_NOLONGTEST: Skipping test_ex2_readburp_cmp")
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex2_readburp.log")
        logfile2 = os.path.join(TMPDIR, "test_ex2_readburp_py.log")
        self.sub_test_ex2_readburp(logfile=logfile1)
        self.sub_test_ex2_readburp_py(logfile=logfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2, shallow=False))

    #==== Example 3 =============================================

    def sub_test_ex3_obs(self, logfile="test_ex3_obs.log"):
        """burplib_c iweb doc example 3"""
        import os, sys
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        iunit = 999
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        logfid.write("Nombre Enreg = {}\n".format(istat))
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        counters = {}
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                continue
            if brp.RPT_STNID(rr)[0] != '^':
                try:
                    counters[brp.RPT_STNID(rr)] += 1
                except:
                    counters[brp.RPT_STNID(rr)] = 1
                continue

            brp.BLK_SetBKNO(bs, 0)
            trouve = False
            while brp.c_brp_findblk(bs, rr) >= 0:
                if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                    continue
                if (brp.BLK_BKNAT(br) & 3) == 2:
                    try:
                        counters[brp.RPT_STNID(rr)] += brp.BLK_NT(br)
                    except:
                        counters[brp.RPT_STNID(rr)] = brp.BLK_NT(br)
                    trouve = True
                    break
            if trouve:
                counters[brp.RPT_STNID(rr)] += 1
        istat = brp.c_brp_close(iunit)
        brp.brp_free(bs, br, rs, rr)
        i, total =  0, 0
        ## for k in sorted(counters.keys()):
        for k in counters.keys():
            logfid.write("{})\t{}\t{}\n".format(i,k,counters[k]))
            i += 1
            total += counters[k]
        logfid.write("-----\t--------\t--------\n")
        logfid.write("     \tTotal   \t{}\n".format(total))
        if logfid != sys.stdout: logfid.close()


    def sub_test_ex3_obs_py(self, logfile="test_ex3_obs_py.log"):
        """burplib_c iweb doc example 3"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        counters = {}
        with brp.BurpcFile(infile) as bfile:
            logfid.write("Nombre Enreg = {}\n".format(len(bfile)))
            for rr in bfile:
                if  rr.stnid[0] != '^':
                    try:
                        counters[rr.stnid] += 1
                    except:
                        counters[rr.stnid] = 1
                    continue
                trouve = False
                for br in rr:
                    ## if (br.bknat & 3) == 2:
                    if br.bknat_kindd == 'desc3d':
                        try:
                            counters[rr.stnid] += br.nt
                        except:
                            counters[rr.stnid] = br.nt
                        trouve = True
                        break
                if trouve:
                    counters[rr.stnid] += 1
        i, total =  0, 0
        ## for k in sorted(counters.keys()):
        for k in counters.keys():
            logfid.write("{})\t{}\t{}\n".format(i,k,counters[k]))
            i += 1
            total += counters[k]
        logfid.write("-----\t--------\t--------\n")
        logfid.write("     \tTotal   \t{}\n".format(total))
        if logfid != sys.stdout: logfid.close()


    def test_ex3_obs_cmp(self):
        """burplib_c iweb doc example 3, compare results from 2 itf"""
        import os, filecmp
        if len(testlist) > 0 and not '3' in testlist:
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex3_obs.log")
        logfile2 = os.path.join(TMPDIR, "test_ex3_obs_py.log")
        self.sub_test_ex3_obs(logfile=logfile1)
        self.sub_test_ex3_obs_py(logfile=logfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2))

    #==== Example 4 =============================================

    def sub_test_ex4_elements(self, logfile="test_ex4_elements.log"):
        """burplib_c iweb doc example 4"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        iunit = 999
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        elems = set()
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                continue
            brp.BLK_SetBKNO(bs, 0)
            while brp.c_brp_findblk(bs, rr) >= 0:
                if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                    continue
                if (brp.BLK_BKNAT(br) & 3) == 3:
                    continue
                for i in range(brp.BLK_NELE(br)):
                    elems.add(brp.BLK_DLSTELE(br, i))
        istat = brp.c_brp_close(iunit)
        brp.brp_free(bs, br, rs, rr)
        for v in elems:
            logfid.write("{} {}\n".format(v, repr(rmn.mrbcvt_dict(v, False))))
        if logfid != sys.stdout: logfid.close()

    def sub_test_ex4_elements_py(self, logfile="test_ex4_elements_py.log"):
        """burplib_c iweb doc example 4"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        logfid = open(logfile, "w") if logfile else sys.stdout
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        elems = set()
        with brp.BurpcFile(infile) as bfile:
            for rr in bfile:
                for br in rr:
                    ## if br.bknat_kindd != 'flags':
                    ##     if (br.bknat & 3) == 3:
                    ##         print(br.bknat, (br.bknat & 3), br.bknat_kindd, br.bknat_multi, br.bknat_kind)
                    #TODO: review
                    ## if br.bknat_kindd != 'flags':
                    ##     elems.update(list(br.dlstele))
                    if (br.bknat & 3) == 3:
                        continue
                    elems.update(list(br.dlstele))
                    ## for i in range(br.dlstele.size):
                    ##      elems.add(br.dlstele[i])
        for v in elems:
            logfid.write("{} {}\n".format(v, repr(rmn.mrbcvt_dict(v, False))))
        if logfid != sys.stdout: logfid.close()


    def test_ex4_elements_cmp(self):
        """burplib_c iweb doc example 4, compare results from 2 itf"""
        import os, filecmp
        if len(testlist) > 0 and not '4' in testlist:
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex4_elements.log")
        logfile2 = os.path.join(TMPDIR, "test_ex4_elements_py.log")
        self.sub_test_ex4_elements(logfile=logfile1)
        self.sub_test_ex4_elements_py(logfile=logfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2))

    #==== Example 5 =============================================

    def sub_test_ex5_write1(self, logfile="test_ex5_write1.log",
                                  outfile="test_ex5_write1.brp"):
        """burplib_c iweb doc example 5"""
        import os, sys
        import rpnpy.burpc.all as brp
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        iunit, ounit = 999, 998
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        istat = brp.c_brp_open(ounit, _C_WCHAR2CHAR(outfile), _C_WCHAR2CHAR("w"))
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        brp.RPT_SetHANDLE(rs, 0)
        brp.RPT_SetTEMPS(rs, 2300)
        brp.RPT_SetIDTYP(rs, 32)
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                continue
            brp.RPT_SetTEMPS(rr, 2200)
            brp.c_brp_updrpthdr(ounit, rr)
            brp.c_brp_writerpt(ounit, rr, brp.BRP_END_BURP_FILE)
        istat = brp.c_brp_close(iunit)
        istat = brp.c_brp_close(ounit)
        brp.brp_free(bs, br, rs, rr)
        # Print out content of created file for camparison
        self.sub_test_ex2_readburp(infile=outfile, logfile=logfile)


    def sub_test_ex5_write1_py(self, logfile="test_ex5_write1_py.log",
                                     outfile="test_ex5_write1_py.brp"):
        """burplib_c iweb doc example 5"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        infile = os.path.join(os.getenv('ATM_MODEL_DFILES').strip(),
                              'bcmk_burp/2007021900.brp')
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        idtyp = rmn.BURP_IDTYP_IDX['PILOT']  ## 32
        bfilei = brp.BurpcFile(infile)
        with brp.BurpcFile(outfile, rmn.BURP_MODE_CREATE) as bfileo:
            rs = brp.BurpcRpt({'handle' : 0, 'temps' : 2300, 'idtype': idtyp})
            rr = None
            while True:
                rr = bfilei.get(rs, rr)
                if not rr:
                    break
                rs.handle = rr.handle
                rr.temps  = 2200
                bfileo.append(rr)
                ## print('len bfileo=', len(bfileo))
        # Print out content of created file for camparison
        self.sub_test_ex2_readburp_py(infile=outfile, logfile=logfile)


    def test_ex5_write1_cmp(self):
        """burplib_c iweb doc example 5, compare results from 2 itf"""
        import os, filecmp
        if len(testlist) > 0 and not '5' in testlist:
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex5_write1.log")
        outfile1 = os.path.join(TMPDIR, "test_ex5_write1.brp")
        logfile2 = os.path.join(TMPDIR, "test_ex5_write1_py.log")
        outfile2 = os.path.join(TMPDIR, "test_ex5_write1_py.brp")
        self.sub_test_ex5_write1(logfile=logfile1, outfile=outfile1)
        self.sub_test_ex5_write1_py(logfile=logfile2, outfile=outfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2, shallow=False))

    #==== Example 6 =============================================

    def sub_test_ex6_write2(self, logfile="test_ex6_write2_py.log",
                                  outfile="test_ex6_write2.brp"):
        """burplib_c iweb doc example 6"""
        import os, sys
        import rpnpy.burpc.all as brp
        ounit = 20
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(ounit, _C_WCHAR2CHAR(outfile),
                               _C_WCHAR2CHAR("w"))

        rr, tmp = brp.c_brp_newrpt(), brp.c_brp_newblk()
        br, br2 = brp.c_brp_newblk(), brp.c_brp_newblk()

        # remplir entete enregistrement
        brp.RPT_SetTEMPS(rr , 1200    )
        brp.RPT_SetFLGS( rr , 0       )
        brp.RPT_SetSTNID(rr , "74724")
        brp.RPT_SetIDTYP(rr , 32      )
        brp.RPT_SetLATI( rr , 14023   )
        brp.RPT_SetLONG( rr , 27023   )
        brp.RPT_SetDX(   rr , 0       )
        brp.RPT_SetDY(   rr , 0       )
        brp.RPT_SetELEV( rr , 0       )
        brp.RPT_SetDRND( rr , 0       )
        brp.RPT_SetDATE( rr , 20050317)
        brp.RPT_SetOARS( rr , 0       )

        # allouer espace pour l'enregistremen pour ajouter des blocs
        brp.c_brp_allocrpt(rr, 10000)
        brp.c_brp_resizerpt(rr, 20000) # on peut reallouer + espace
        #print("rr apres resize: "+ str(brp.RPT_NSIZE(rr)))

        # on peut mettre le contenu du rapport a 0, cela n'affecte pas le header
        brp.c_brp_clrrpt(rr)

        # ici c'est preparer l'ecriture de l'enrgistrement dans le fichier
        # NOTE: c_brp_putrpthdr() doit absolument etre appele AVANT
        #       le premier appel a c_brp_putblk()
        brp.c_brp_putrpthdr(ounit, rr)

        # Section ajout de blocs dans l'enregistrement

        # Ici on indique que l'on desire remplir, le bloc br de valueurs reelles
        brp.BLK_SetSTORE_TYPE(br, brp.BRP_STORE_FLOAT)

        # setter les params du bloc BFAM, BDESC et BTYP
        brp.BLK_SetBFAM(br,  0)
        brp.BLK_SetBDESC(br, 0)
        brp.BLK_SetBTYP(br,  64)

        # allouer espace pour remplir le bloc
        # ici pour 2 elements et 1 valeur par element et 1 groupe nelem X nval
        brp.c_brp_allocblk(br, 2, 1, 1)

        # Les indices en C commencent par 0,
        # on remplit les elements: 7004 et 11001
        brp.BLK_SetDLSTELE(br, 0, 7004)
        brp.BLK_SetDLSTELE(br, 1, 11001)

        # Compacter les elements
        brp.c_brp_encodeblk(br)

        # remplir les valeures pour chacun des elements
        brp.BLK_SetRVAL(br, 0, 0, 0, 10.0)  # pour 7004
        brp.BLK_SetRVAL(br, 1, 0, 0, 20.0)  # pour 11001

        # on a rempli les valeurs reelles alors les convertir selon la
        # table burp en entiers qui seront enregistres dans le fichier burp
        ## if brp.c_brp_convertblk(br) < 0:
        if brp.c_brp_convertblk(br, brp.BRP_MKSA_to_BUFR) < 0:
            sys.exit(1)

        # on met le bloc br dans l'enrgistrement rr
        if brp.c_brp_putblk(rr, br) < 0:
            sys.exit(1)

        # on peut faire une copie du bloc br, br2 est une copie de br
        # tous les attributs de br le seront pour br2
        brp.c_brp_copyblk(br2, br)

        # on met le bloc br2 dans l'enrgistrement rr
        if brp.c_brp_putblk(rr, br2) < 0:
            sys.exit(1)

        ## # on peut redimensionner le bloc br pour contenir 5 elements,
        ## # 10 valeurs par element et aussi 2 tuiles de 5 X 10
        ## # comme ici c'est une expansion du bloc donc on retrouvera les
        ## # elements et leurs valeures (precedentes aux memes indices)
        ## brp.c_brp_resizeblk(br, 5, 10, 2)

        ## # on ajoute ce bloc a l'enregistrement
        ## brp.c_brp_putblk(rr, br)

        # ici on fait une copie de br, tmp est une copie de br
        brp.c_brp_copyblk(tmp, br)

        # redimensionner le bloc tmp, ici il s'agit d'une reduction
        # 3 elements, 2 valeures par element et 1 tuile de 3 x 2
        brp.c_brp_resizeblk(tmp, 3, 2, 1)

        # Ici on indique que l'on desire remplir, le bloc br de val entieres
        brp.BLK_SetSTORE_TYPE(tmp, brp.BRP_STORE_INTEGER)

        # setter l'element 3 a 11003
        brp.BLK_SetDLSTELE(tmp, 2, 11003)
        # et ses valeures entieres
        brp.BLK_SetTBLVAL(tmp, 2, 0, 0, 15)
        brp.BLK_SetTBLVAL(tmp, 2, 1, 0, 30)

        # compacter les elements du bloc
        brp.c_brp_encodeblk(tmp)

        # ajouter le bloc tmp a l'enrgistrement rr
        brp.c_brp_putblk(rr, tmp)

        # ajouter l'enrgistrement dans le fichier
        if brp.c_brp_writerpt(ounit, rr, brp.BRP_END_BURP_FILE) < 0:
            sys.exit(1)

        # fermeture de fichier burp
        if brp.c_brp_close(ounit) < 0:
            sys.exit(1)

        # liberer ressources
        brp.brp_free(rr, br, br2, tmp)

        self.sub_test_ex2_readburp_py(infile=outfile, logfile=logfile)


    def sub_test_ex6_write2_py(self, logfile="test_ex6_write2_py.log",
                                     outfile="test_ex6_write2_py.brp"):
        """burplib_c iweb doc example 6"""
        import os, sys
        import rpnpy.librmn.all as rmn
        import rpnpy.burpc.all as brp
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)

        rpt = brp.BurpcRpt({
            'temps'  : 1200,
            'flgs'   : 0,        #todo
            'stnid'  : '74724',
            'idtype' : rmn.BURP_IDTYP_IDX['PILOT'],  ## 32
            'lati'   : rmn.BRP_RLAT2ILAT(50.23),     ## 14023
            'longi'  : rmn.BRP_RLON2ILON(270.23),    ## 27023
            'dx'     : rmn.BRP_RDX2IDX(0.),  ## 0
            'dy'     : rmn.BRP_RDY2IDY(0.),  ## 0
            'elev'   : 0,  #todo: rmn.BRP_RELEV2IELEV(0.)
            'drnd'   : 0,
            'date'   : rmn.BRP_YMD2IDATE(2005,3,17),
            'oars'   : 0,
            })

        bknat_multi = rmn.BURP_BKNAT_MULTI_IDX['uni']
        bknat_kind  = rmn.BURP_BKNAT_KIND_IDX['data']
        bknat       = rmn.mrbtyp_encode_bknat(bknat_multi, bknat_kind)
        bktyp_alt   = rmn.BURP_BKTYP_ALT_IDX['surf']
        bktyp_kind  = 4  ## See BURP_BKTYP_KIND_DESC, 'derived data, entry to the OA at surface, global model',
        bktyp       = rmn.mrbtyp_encode_bktyp(bktyp_alt, bktyp_kind)
        bkstp       = 0  ## See BURP_BKSTP_DESC
        btyp        = rmn. mrbtyp_encode(bknat, bktyp, bkstp)

        blk = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_FLOAT,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })

        ele1 = brp.BurpcEle({
            'e_bufrid' : 7004,
            'e_rval'   : [10.]
            })
        blk.append(ele1)
        ele2 = brp.BurpcEle({
            'e_bufrid' : 11001,
            'e_rval'   : [20.]
            })
        blk.append(ele2)
        rpt.append(blk)

        # on peut faire une copie du bloc blk (tous les attributs)
        # et l'ajouter au rapport
        rpt.append(brp.BurpcBlk(blk))

        # Nouveau bloc a partir des elements de blk (ele1, ele2) avec
        # changement de type, "re-dimention" et
        # ajout d'un element supplementaire
        tmp = brp.BurpcBlk({
            'store_type' : brp.BRP_STORE_INTEGER,
            'bfam'   : 0,
            'bdesc'  : 0,
            'btyp'   : btyp,  ## 64
            })
        missingVal = rmn.mrfopt(rmn.BURPOP_MISSING)
        tmp.append(brp.BurpcEle({
            'store_type' : brp.BRP_STORE_INTEGER,
            'e_bufrid' : ele1.e_bufrid,
            'e_tblval' : [ele1.e_tblval[0,0], rmn.BURP_TBLVAL_MISSING]
            }))
        tmp.append(brp.BurpcEle({
            'store_type' : brp.BRP_STORE_INTEGER,
            'e_bufrid' : ele2.e_bufrid,
            'e_tblval' : [ele2.e_tblval[0,0], rmn.BURP_TBLVAL_MISSING]
            }))
        tmp.append(brp.BurpcEle({
            'store_type' : brp.BRP_STORE_INTEGER,
            'e_bufrid' : 11003,
            'e_tblval' : [15, 30]
            }))

        rpt.append(tmp)

        with brp.BurpcFile(outfile, rmn.BURP_MODE_CREATE) as bfileo:
            bfileo.append(rpt)

        self.sub_test_ex2_readburp_py(infile=outfile, logfile=logfile)


    def test_ex6_write2_cmp(self):
        """burplib_c iweb doc example 6, compare results from 2 itf"""
        import os, filecmp
        if len(testlist) > 0 and not '6' in testlist:
            return
        TMPDIR = os.getenv('TMPDIR', '/tmp')
        logfile1 = os.path.join(TMPDIR, "test_ex6_write2.log")
        outfile1 = os.path.join(TMPDIR, "test_ex6_write2.brp")
        logfile2 = os.path.join(TMPDIR, "test_ex6_write2_py.log")
        outfile2 = os.path.join(TMPDIR, "test_ex6_write2_py.brp")
        self.sub_test_ex6_write2(logfile=logfile1, outfile=outfile1)
        self.sub_test_ex6_write2_py(logfile=logfile2, outfile=outfile2)
        self.assertTrue(filecmp.cmp(logfile1, logfile2, shallow=False))

    #==== Example 7 =============================================

    #==== Utilities =============================================

    #TODO: examples for most burp utils

## brptotxt     Afficher sous format texte particulier le contenu de nos fichier BURP dbase des METAR pour les besoins de AWWS. brptotxt -usage

## btyp Pour afficher la signification d'un btyp        btyp -h

## burpcmp      Comparaison de fichiers BURP    burpcmp -usage

## burpdiff     Ordonne les enregistrements BURP dans le même ordre puis les compare à l'aide de xxdiff       burpdiff -usage

## burpmatch    Match BURP reports
## (without replacement == once 2 reports are matched subsequent reports that share the same match criteria will not be matched with the original 2 reports)
## (see selectbrp for "with replacement")       burpmatch -usage

## bvoir        S.V.P. utilisez "liburp fileburp -enrgs"
## Pour afficher le contenu des cles d'en-tete d'un fichier BURP        man bvoir

## codbuf       Obtenir rapidement la signification d'un element BUFR   codbuf <noelembufr>

## codltr       Permet d'afficher a l'ecran la signification d'un code lettre (ex: MiMi, dd, fff) et si sa signification refere a un no.table, affiche la signification de la table.    codltr <LETTRES A RECHERCHER>

## codtbl       Permet d'afficher a l'ecran la signification d'un element bufr ou un element local et s'il s'agit d'un "code table", affiche la signification du "code table"   Documentation

## codtyp       Permet d'afficher a l'ecran la definition du codtyp     codtyp <code>

## completebrp  Effectuer une operation "complete" avec 2 fichiers BURP.        completebrp -usage

## degrup       Dégroupe les fichiers BURP     degrup -usage

## extrait/extract      Extrait les en-tetes des bulletins de rawdata que nous recevons ou un bulletin complet avec l'en-tete
## (extract extrait en plus l'heure de reception du bulletin)   max extrait
## extrait -h

## filtburp     Pour filtrer le contenu d'un fichier BURP (complement de EDITBRP)       man filtburp

## flgs24       Pour afficher la signification du flag de 24 bits de l'en-tete  flgs24 <code>

## flgsbm       Pour afficher la signification d'une valeur trouvée dans un bloc marqueur      flgsbm <code>

## flgtbl       Pour afficher la signification des bits allumés des flag-tables (CREX/BUFR Table B)    flgtbl -usage

## kprmk        Permet de garder uniquement la partie RMK d'un bulletin ASCII METAR et d'éliminer la partie principale.        Documentation

## liburp       Pour lire un fichier BURP et en voir une sortie ASCII (differente de Rdburp)    liburp -usage

## reflex       Pour fusionner ou recuperer l'espace perdu des fichiers BURP    man reflex

## rmv_rmk      Permet de garder uniquement la partie principale d'un bulletin ASCII METAR et d'éliminer la partie RMK.        Documentation

## runn Pour afficher la signification de runn  runn <code>

## sa_sm_mtr    Lire 2 ou 3 fichiers BURP de type SA, SM ou METAR et afficher les valeurs de éléments demandés où il y a plus qu'un rapport avec les même lat/lon/date/heure.      sa_sm_mtr -usage

## selectbrp    Select BURP reports
## (with replacement == once 2 reports are matched the program will continue to match them against subsequent reports that share the same match criteria)
## (see burpmatch for "without replacement")    selectbrp -usage

## stationlist  Affiche les stations, par codtyp, contenues dans un fichier burp.       stationlist -h

## tmoinstd     Calculer T-Td (écart du point de rosée) pour chacun des enregistrements du fichier BURP specifie.     tmoinstd -usage

## Xdatamon

## editbrp      Pour manipuler des fichiers BURP (selection d'enregistrements)  . ssmuse-sh -d rpn/utils/15.2

## nbgen        Genere le nom d'un fichier BURP en suivant la nomenclature operationnelle       CMDS
## . ssmuse-sh -d cmoi/base/20141216/

## reflex       Pour fusionner ou recuperer l'espace perdu des fichiers BURP    . ssmuse-sh -d rpn/utils/15.2

## SATPLOT      Satellite Data Plotting and Analysis Tool       ARMA
## stephen.macpherson@ec.gc.ca

## sigma        Permet de tracer des cartes de contours ou de coupes d'éléments météorologiques grave a partir d'un fichier standard à accèss direct. Les fichiers produits sont metacod et segfile et peuvent êtres visualisés à l'aide de metaview ou xmetaview.     . ssmuse-sh -d rpn/utils/15.2


## ade.regrup   on utilise ade.regrup pour traiter les donnees satwinds dans evalalt

## deri.bextrep Gardant pour chaque stn le rapport le plus pres de la date donnee et se trouvant dans la fenetre donnee

## deri.blcklst Flag observations according to blacklist rules

## deris.bmrgsasm       Combine les rapports SM (synop) avec les rapports SA

## deris.bsfderiv       Surface Derivate

## deris.cleanburp      Enlever les blocs avec dimension de 0 ou des elements BUFR invalides

## deris.lndmask        Screen out observations over land

## deris.ssmidbretr     Retrieve data from satellite database according to user directives

## deriv.aircraftcorr   Aircraft temperature bias correction

## deriv.blacklistgbgps Blacklister Ground-Based GPS selon les parametres de monitoring

## deriv.bmrguasm       Fusionne les rapports concernant la meme station (ASCII)

## deriv.bmrguasmb      Fusionne les rapports concernant la meme station (BUFR)

## deriv.buaderiv       Upper Air Derivate

## deriv.degrupsatwinds Sauver en donnees degroupees

## deriv.fuaderiv       Fake Upper Air Derivate (simulate derivate flags)

## deriv.llglob Ajouter la hauteur de la surface

## deriv.nodups Eliminer les observations considerees des duplications

## deriv.prepareua      Interpolate missing pressure values, sort levels, eliminate duplicates (ASCII)

## deriv.prepareuab     Interpolate missing pressure values, sort levels, eliminate duplicates (BUFR)

## deriv.resume Generate a summary record (>>DERIALT)

## deriv.satqcssmis     Lire les coefficients de correction de biais

## deriv.thinning0aircrafts     Select aircraft observations according to thinning rules

## deriv.thinning0amsua Enlever les donnees corrompues (AMSUA)

## deriv.thinning0amsub Enlever les donnees corrompues (AMSUB)

## deriv.thinning0blackua       Apply blacklisting to raobs data (ASCII)

## deriv.thinning0blackuab      Apply blacklisting to raobs data (BUFR)

## deriv.thinning0csr   Apply selection rules to CSR observations

## deriv.thinning0gbgps Selectionner GBGPS selon la proximite au surface (SYNOP)

## deriv.thinning0satwinds      Read atmospheric motion wind observations and select according to thinning0 rules

## deriv.thinning0scat  Programme principal de degroupage/selection des Quikscat/ASCAT (KNMI)

## deriv.tovsreducer    Reduire le nombre de tovs d'un fichier burp

## deriv.trackqc        Controle de la qualite des donnees regroupees

## deriv.trajraobs      Adds trajectory of the balloon (ASCII)

## deriv.trajraobsb     Adds trajectory of the balloon (BUFR)

## deriv.ua4dprecedence Give precedence to 4d reports (with trajectory elements)

## deriv.uabcor Ajouter la correction de biais dans un fichier d'observations de raobs

## deriv.uaconvertelem  Rearrange and calculate elements to the order expected by the derivate system

## cdict        Outil pour comparer les dictionnaires de stations du CMC et du WMO
## (Avant de s'en servir l'utilisateur doit s'assurer d'avoir les bonnes informations dans la base de données cdict, ie: le dictionnaire de stations du CMC et celui de l'OMM. Ainsi que de maintenir à jour ces informations. Sinon cdict retourne de l'information qui peut être obsolète)        Installation sur demande        cdict -h

## genprof      Produit un fichier burp en interpolant aux niveaux et aux temps demandés (Profiles)    . ssmuse-sh -d cmoi/base/20141216/
## genprof      Documentation

## hdecode      Page web pour décoder les headers du GTS (T1T2A1A2ii decoder)  hdecode

## monitoring   Meteorological observation quality and availability monitoring web page monitoring

## msc-adas     Reception de données AMDAR canadiennes en format ARINC-620, monitoring et contrôle de qualité de celles-ci, et génération de bulletins BUFR pour envoi sur le GTS  msc-adas        Documentation

## r.arcad      Outil diagnostic pour évaluer les résultats de cycles d'assimilation de données      . ssmuse-sh -d rpn/utils/15.2/
## r.arcad      Documentation
## Josée Morneau

## qc3fuzzy
## gc3fuzzy2    Contôle de qualité des observations de surface

    #==== Example ? =============================================

    def _test_misc(self):
        a = brp.BurpcRpt()
        print(a)
        a.handle = 1
        print(a)
        a['stnid'] = '12334567'
        print(a)
        print('---')
        a = a.getptr()
        print(a)
        print(a[0])
        print('---')
        a = brp.BurpcBlk()
        print(a)
        print('---')
        a = a.getptr()
        print(a)
        print(a[0])
        #TODO: test iter
        #TODO: test indexing FILE, RPT with int
                ## print '-----0'
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.getptr(),0), brp.BLK_DLSTELE(br.getptr(),1)
                ## print '-----1'
                ## br.dlstele[0] = 999
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.getptr(),0), brp.BLK_DLSTELE(br.getptr(),1)
                ## print '-----2'
                ## brp.BLK_SetDLSTELE(br.getptr(),0,998)
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.getptr(),0), brp.BLK_DLSTELE(br.getptr(),1)
                ## print '-----3'
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.getptr()), brp.RPT_STNID(rr.getptr())
                ## print '-----4'
                ## rr.idtype = 123
                ## rr.stnid  = '1234567890'
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.getptr()), brp.RPT_STNID(rr.getptr())
                ## print '-----5'
                ## brp.RPT_SetIDTYP(rr.getptr(), 456)
                ## brp.RPT_SetSTNID(rr.getptr(), 'abcdefghij')
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.getptr()), brp.RPT_STNID(rr.getptr())
                ## sys.exit(0)
        return



    def totosub_test_ex2_readburp_c(self):
        """burplib_c iweb doc example 2"""
        infile = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        bfile = brp.BurpcFile(infile)
        print("Nombre Enreg = {}".format(len(bfile)))
        rs = brp.BurpcRpt()
        rs.handle, rr, br = 0, None, None
        while True:
            rr = bfile.get(rs, rr)
            if not rr: break
            rs.handle = rr.handle
            print("""
hhmm   ={:8d} flgs   ={:6d}  codtyp ={:6d}  stnids ={:9s}
blat   ={:8d} blon   ={:6d}  dx     ={:6d}  dy     ={:6d}  stnhgt ={:6d}
yymmdd ={:8d} oars   ={:6d}  runn   ={:6d}  nblk   ={:6d}  dlay   ={:6d}
""".format(rr.temps, rr.flgs, rr.idtype, rr.stnid,
           rr.lati, rr.longi, rr.dx, rr.dy, rr.elev,
           rr.date, rr.oars, rr.runn, rr.nblk, rr.drnd))
            ## bkno = 0
            ## while True:
            for bkno in range(rr.nblk):
                br = rr.get(bkno, rr, br)
                if not br: break
                ## bkno = br.bkno
                print("""
blkno  ={:6d}  nele   ={:6d}  nval   ={:6d}  nt     ={:6d}  bit0   ={:6d}
bdesc  ={:6d}  btyp   ={:6d}  nbit   ={:6d}  datyp  ={:6d}  bfam   ={:6d}
""".format(br.bkno, br.nele, br.nval, br.nt, br.bit0,
           br.bdesc, br.btyp, br.nbit, br.datyp, br.bfam))
                for k in range(br.nt):
                    if br.bkno != 1:
                        print("\nobservation {}/{}".format(k+1, br.nt))
                    mystr = "lstele ="
                    for i in range(br.nele):
                        mystr += "    {:0>6d}".format(np.asscalar(br.dlstele[i]))
                    print(mystr)
                    for j in range(br.nval):
                        mystr = "tblval ="
                        for i in range(br.nele):
                            ## mystr += "{:10d}".format(np.asscalar(br.tblval[i,j,k]))
                            mystr += "{:10d}".format(np.asscalar(br.tblval[i,j,k]))
                        print(mystr)

        del bfile

##     def sub_test_ex2_readburp_e(self):
##         """burplib_c iweb doc example 2"""
##         infile = self.knownValues[0][0]
##         brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
##         with brp.BurpcFile(infile) as bfile:
##             print("Nombre Enreg = {}".format(len(bfile)))
##             for rr in bfile:
##                 print("""
## hhmm   ={:8d} flgs   ={:6d}  codtyp ={:6d}  stnids ={:9s}
## blat   ={:8d} blon   ={:6d}  dx     ={:6d}  dy     ={:6d}  stnhgt ={:6d}
## yymmdd ={:8d} oars   ={:6d}  runn   ={:6d}  nblk   ={:6d}  dlay   ={:6d}
## """.format(rr.temps, rr.flgs, rr.idtype, rr.stnid,
##            rr.lati, rr.longi, rr.dx, rr.dy, rr.elev,
##            rr.date, rr.oars, rr.runn, rr.nblk, rr.drnd))
##                 for br in rr:


    #TODO: def sub_test_ex2_readburp_e(self):

    def todosub_test_ex3_obs(self):
        """burplib_c iweb doc example 3"""
        infile, itype, iunit = self.knownValues[0]
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        istat = brp.c_brp_open(iunit, _C_WCHAR2CHAR(infile), _C_WCHAR2CHAR("r"))
        print("Nombre Enreg = {}".format(istat))
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        counters = {} #TODO: preserve order? then store info in list, index in dict
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                continue
            if brp.RPT_STNID(rr)[0] != '^':
                try:
                    counters[brp.RPT_STNID(rr)] += 1
                except:
                    counters[brp.RPT_STNID(rr)] = 1
                continue

            brp.BLK_SetBKNO(bs, 0)
            trouve = False
            while brp.c_brp_findblk(bs, rr) >= 0:
                if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                    continue
                if (brp.BLK_BKNAT(br) & 3) == 2:
                    try:
                        counters[brp.RPT_STNID(rr)] += brp.BLK_NT(br)
                    except:
                        counters[brp.RPT_STNID(rr)] = brp.BLK_NT(br)
                    trouve = True
                    break
            if trouve:
                counters[brp.RPT_STNID(rr)] += 1
        istat = brp.c_brp_close(iunit)
        brp.brp_free(bs, br, rs, rr)
        i, total =  0, 0
        for k in counters.keys():
            print("{})\t{}\t{}".format(i,k,counters[k]))
            i += 1
            total += counters[k]
        print("-----\t--------\t--------")
        print("     \tTotal   \t{}".format(total))


    def todosub_test_ex3_obs_c(self):
        """burplib_c iweb doc example 3"""
        infile, itype, iunit = self.knownValues[0]
        istat = brp.c_brp_SetOptChar(_C_WCHAR2CHAR("MSGLVL"),
                                     _C_WCHAR2CHAR("FATAL"))
        bfile = brp.BurpcFile(infile)
        print("Nombre Enreg = {}".format(len(bfile)))
        rs = brp.BurpcRpt()
        rr, br = None, None
        counters = {} #TODO: preserve order? then store info in list, index in dict
        while True:
            rr = bfile.get(rs, rr)
            if not rr: break
            rs.handle = rr.handle
            if rr.stnid[0] != '^':
                try:
                    counters[rr.stnid] += 1
                except:
                    counters[rr.stnid] = 1
                continue
            trouve = False
            for bkno in range(rr.nblk):
                br = rr.get(bkno, rr, br)
                if not br: break
                if (br.bknat & 3) == 2:  #TODO: use const with decoded bknat
                    print('bknat_multi', br.bknat & 3, rmn.mrbtyp_decode_bknat(br.bknat), rmn.BURP2BIN(br.bknat,8), rmn.BURP2BIN2LIST(br.bknat,8), rmn.BURP2BIN2LIST_BUFR(br.bknat,8))
                    try:
                        counters[rr.stnid] += br.nt
                    except:
                        counters[rr.stnid] = br.nt
                    trouve = True
                    break
            if trouve:
                counters[rr.stnid] += 1
        del bfile
        i, total =  0, 0
        for k in counters.keys():
            print("{})\t{}\t{}".format(i,k,counters[k]))
            i += 1
            total += counters[k]
        print("-----\t--------\t--------")
        print("     \tTotal   \t{}".format(total))


    #TODO: def sub_test_ex3_obs_e(self):


testlist = []
if __name__ == "__main__":
    while len(sys.argv) > 1:
        testlist.append(sys.argv.pop())
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
