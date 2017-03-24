#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
"""
Unit tests for burpc

See: http://iweb.cmc.ec.gc.ca/~afsdcvs/burplib_c/
"""

import os
import sys
import rpnpy.librmn.all as rmn
import rpnpy.burpc.all as brp
import unittest
## import ctypes as ct
import numpy as np

if sys.version_info > (3, ):
    long = int

class RpnPyBurpc(unittest.TestCase):

    burptestfile = 'bcmk_burp/2007021900.brp'
    #(path, itype, iunit)
    knownValues = (
        (burptestfile, rmn.WKOFFIT_TYPE_LIST['BURP'], 999), 
        )

    def getFN(self, name):
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        return os.path.join(ATM_MODEL_DFILES.strip(), name)

    def _test_ex1_read1(self):
        """burplib_c iweb doc example 1"""
        mypath, itype, iunit = self.knownValues[0]
        istat = brp.c_brp_SetOptChar("MSGLVL", "FATAL" )
        istat = brp.c_brp_open(iunit, self.getFN(mypath), "r")
        print("enreg {}".format(istat))
        bs = brp.c_brp_newblk()
        br = brp.c_brp_newblk()
        rs = brp.c_brp_newrpt()
        rr = brp.c_brp_newrpt()
        brp.RPT_SetHANDLE(rs,0)
        while brp.c_brp_findrpt(iunit, rs) >= 0:
            if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) >= 0:
                print("stnid = {}".format(brp.RPT_STNID(rr)))
                brp.BLK_SetBKNO(bs, 0)
                while brp.c_brp_findblk(bs, rr) >= 0:
                    if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) >= 0:
                        print("block btyp = {}".format(brp.BLK_BTYP(br)))
        istat = brp.c_brp_close(iunit)
        brp.c_brp_freeblk(bs)
        brp.c_brp_freeblk(br)
        brp.c_brp_freerpt(rs)
        brp.c_brp_freerpt(rr)


    def _test_ex1_read1_b(self):
        """burplib_c iweb doc example 1"""
        mypath = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        bfile = brp.BURP_FILE(self.getFN(mypath))
        print("enreg {}".format(len(bfile)))
        rs, rr, br = 0, None, None
        while rs is not None:
            rs = brp.brp_findrpt(bfile.funit, rs)
            if not rs: break
            rr = brp.brp_getrpt(bfile.funit, rs.handle, rr)
            print("stnid = {}".format(rr.stnid))
            bs = 0
            while bs is not None:
                bs = brp.brp_findblk(bs, rr)
                if not bs: break
                br = brp.brp_getblk(bs.bkno, br, rr)
                print("block btypb = {}".format(br.btyp))
        del bfile


    def _test_ex1_read1_c(self):
        """burplib_c iweb doc example 1"""
        mypath = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        bfile = brp.BURP_FILE(self.getFN(mypath))
        print("enreg {}".format(len(bfile)))
        rs = brp.BURP_RPT_PTR()
        rs.handle, rr, br = 0, None, None
        while True:
            rr = bfile.getrpt(rs, rr)
            if not rr: break
            rs.handle = rr.handle
            print("stnid = {}".format(rr.stnid))
            for bkno in range(rr.nblk):
                br = rr.getblk(bkno, rr, br)
                if not br: break
                ## bkno = br.bkno
                print("block btypc = {}".format(br.btyp))
        del bfile

        
    def _test_ex1_read1_d(self):
        """burplib_c iweb doc example 1"""
        mypath = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        bfile = brp.BURP_FILE(self.getFN(mypath))
        print("enreg {}".format(len(bfile)))
        for rr in bfile:
            print("stnid = {}".format(rr.stnid))
            for br in rr:
                print("block btypd = {}".format(br.btyp))
        del bfile


    def _test_ex1_read1_e(self):
        """burplib_c iweb doc example 1"""
        mypath = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        with brp.BURP_FILE(self.getFN(mypath)) as bfile:
            print("enreg {}".format(len(bfile)))
            for rr in bfile:
                print("stnid = {}".format(rr.stnid))
                for br in rr:
                    print("block btype = {}".format(br.btyp))

      
    def _test_misc(self):
        a = brp.BURP_RPT_PTR()
        print(a)
        a.handle = 1
        print(a)
        a['stnid'] = '12334567'
        print(a)
        print '---'
        a = a.get_ptr()
        print(a)
        print(a[0])
        print '---'
        a = brp.BURP_BLK_PTR()
        print(a)
        print '---'
        a = a.get_ptr()
        print(a)
        print(a[0])
        #TODO: test iter
        #TODO: test indexing FILE, RPT with int
                ## print '-----0'
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.get_ptr(),0), brp.BLK_DLSTELE(br.get_ptr(),1)
                ## print '-----1'
                ## br.dlstele[0] = 999
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.get_ptr(),0), brp.BLK_DLSTELE(br.get_ptr(),1)
                ## print '-----2'
                ## brp.BLK_SetDLSTELE(br.get_ptr(),0,998)
                ## print br.dlstele[0], br.dlstele[1]
                ## print brp.BLK_DLSTELE(br.get_ptr(),0), brp.BLK_DLSTELE(br.get_ptr(),1)
                ## print '-----3'
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.get_ptr()), brp.RPT_STNID(rr.get_ptr())
                ## print '-----4'
                ## rr.idtype = 123
                ## rr.stnid  = '1234567890'
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.get_ptr()), brp.RPT_STNID(rr.get_ptr())
                ## print '-----5'
                ## brp.RPT_SetIDTYP(rr.get_ptr(), 456)
                ## brp.RPT_SetSTNID(rr.get_ptr(), 'abcdefghij')
                ## print rr.idtype, rr.stnid, brp.RPT_IDTYP(rr.get_ptr()), brp.RPT_STNID(rr.get_ptr())
                ## sys.exit(0)
        return


    def _test_ex2_readburp(self):
        """burplib_c iweb doc example 2"""
        bs, br = brp.c_brp_newblk(), brp.c_brp_newblk()
        rs, rr = brp.c_brp_newrpt(), brp.c_brp_newrpt()
        istat = brp.c_brp_SetOptChar("MSGLVL", "FATAL" )
        for mypath, itype, iunit in self.knownValues:
            istat = brp.c_brp_open(iunit, self.getFN(mypath), "r")
            print("Nombre Enreg = {}".format(istat))
            brp.RPT_SetHANDLE(rs,0)
            while brp.c_brp_findrpt(iunit, rs) >= 0:
                if brp.c_brp_getrpt(iunit, brp.RPT_HANDLE(rs), rr) < 0:
                    continue
                print("""
hhmm   ={:8d} flgs   ={:6d}  codtyp ={:6d}  stnids ={:9s}
blat   ={:8d} blon   ={:6d}  dx     ={:6d}  dy     ={:6d}  stnhgt ={:6d}
yymmdd ={:8d} oars   ={:6d}  runn   ={:6d}  nblk   ={:6d}  dlay   ={:6d}
""".format(brp.RPT_TEMPS(rr), brp.RPT_FLGS(rr), brp.RPT_IDTYP(rr),
           brp.RPT_STNID(rr), brp.RPT_LATI(rr), brp.RPT_LONG(rr),
           brp.RPT_DX(rr), brp.RPT_DY(rr), brp.RPT_ELEV(rr),
           brp.RPT_DATE(rr), brp.RPT_OARS(rr), brp.RPT_RUNN(rr),
           brp.RPT_NBLK(rr), brp.RPT_DRND(rr)))

                brp.BLK_SetBKNO(bs, 0)
                while brp.c_brp_findblk(bs, rr) >= 0:
                    if brp.c_brp_getblk(brp.BLK_BKNO(bs), br, rr) < 0:
                        continue
                    print("""
blkno  ={:6d}  nele   ={:6d}  nval   ={:6d}  nt     ={:6d}  bit0   ={:6d}
bdesc  ={:6d}  btyp   ={:6d}  nbit   ={:6d}  datyp  ={:6d}  bfam   ={:6d}
""".format(brp.BLK_BKNO(br), brp.BLK_NELE(br), brp.BLK_NVAL(br), 
           brp.BLK_NT(br), brp.BLK_BIT0(br), brp.BLK_BDESC(br), 
           brp.BLK_BTYP(br), brp.BLK_NBIT(br), brp.BLK_DATYP(br),
           brp.BLK_BFAM(br))) 
                    for k in range(brp.BLK_NT(br)):
                        if brp.BLK_BKNO(br) != 1:
                            print("\nobservation {}/{}".
                                  format(k+1, brp.BLK_NT(br)))
                        mystr = "lstele ="
                        for i in range(brp.BLK_NELE(br)):
                            mystr += "    {:0>6d}".format(brp.BLK_DLSTELE(br,i))
                        print(mystr)
                        for j in range(brp.BLK_NVAL(br)):
                            mystr = "tblval ="
                            for i in range(brp.BLK_NELE(br)):
                                mystr += "{:10d}".format(brp.BLK_TBLVAL(br,i,j,k))
                            print(mystr)

            istat = brp.c_brp_close(iunit)
            ## self.assertEqual(funit, itype,
            ##                  mypath+':'+repr(funit)+' != '+repr(itype))
        brp.brp_free(bs, br, rs, rr)


    def _test_ex2_readburp_c(self):
        """burplib_c iweb doc example 2"""
        mypath = self.knownValues[0][0]
        brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
        bfile = brp.BURP_FILE(self.getFN(mypath))
        print("Nombre Enreg = {}".format(len(bfile)))
        rs = brp.BURP_RPT_PTR()
        rs.handle, rr, br = 0, None, None
        while True:
            rr = bfile.getrpt(rs, rr)
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
                br = rr.getblk(bkno, rr, br)
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

##     def _test_ex2_readburp_e(self):
##         """burplib_c iweb doc example 2"""
##         mypath = self.knownValues[0][0]
##         brp.brp_opt(rmn.BURPOP_MSGLVL, rmn.BURPOP_MSG_SYSTEM)
##         with brp.BURP_FILE(self.getFN(mypath)) as bfile:
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
        

    #TODO: def _test_ex2_readburp_e(self):

    def _test_ex3_obs(self):
        """burplib_c iweb doc example 3"""
        mypath, itype, iunit = self.knownValues[0]
        istat = brp.c_brp_SetOptChar("MSGLVL", "FATAL" )
        istat = brp.c_brp_open(iunit, self.getFN(mypath), "r")
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


    def _test_ex3_obs_c(self):
        """burplib_c iweb doc example 3"""
        mypath, itype, iunit = self.knownValues[0]
        istat = brp.c_brp_SetOptChar("MSGLVL", "FATAL" )
        bfile = brp.BURP_FILE(self.getFN(mypath))
        print("Nombre Enreg = {}".format(len(bfile)))
        rs = brp.BURP_RPT_PTR()
        rr, br = None, None
        counters = {} #TODO: preserve order? then store info in list, index in dict
        while True:
            rr = bfile.getrpt(rs, rr)
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
                br = rr.getblk(bkno, rr, br)
                if not br: break
                if (br.bknat & 3) == 2:  #TODO: use const with decoded bknat
                    print 'bknat_multi', br.bknat & 3, rmn.mrbtyp_decode_bknat(br.bknat), rmn.BURP2BIN(br.bknat,8), rmn.BURP2BIN2LIST(br.bknat,8), rmn.BURP2BIN2LIST_BUFR(br.bknat,8)
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


    #TODO: def _test_ex3_obs_e(self):


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
