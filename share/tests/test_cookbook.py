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
  >>> import rpnpy.vgd.all as vgd

See Also:
  * RPNpy Tutorial
  * RPNpy Reference
"""

import unittest
import sys
if sys.version_info < (3, ):
    range = xrange
else:
    long = int

class rpnpyCookbook(unittest.TestCase):
    

    #---- Read / Queries

    
    def test_11(self):
        """
        Queries: Find records, Read meta
        
        This example shows how to
        * get the list of all records matching selection creterions
        * get the metadata and decode it
        * then print some values
        
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.DecodeIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.librmn.base.newdate
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.const
        """
        import os, sys
        import rpnpy.librmn.all as rmn

        # Open all RPNStd files in the $ATM_MODEL_DFILES/bcmk/ directory
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk')
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Get the list of record keys matching nomvar='TT', ip2=12
            keylist = rmn.fstinl(fileId, nomvar='TT', ip2=12)
            
            for k in keylist:
                
                # Get the record meta data
                m = rmn.fstprm(k)

                # Decode ip1, and dateo
                (rp1, rp2, rp3) = rmn.DecodeIp(m['ip1'], m['ip2'], m['ip3'])
                level  = "%f %s" % (rp1.v1, rmn.kindToString(rp1.kind))
                (yyyymmdd,hhmmsshh) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT,m['dateo'])
                dateo  = "%8.8d.%8.8d" % (yyyymmdd,hhmmsshh)
                nhours = float(m['npas'] * m['deet']) / 3600.
                
                print("CB11: %s (%d, %d) level=%s, dateo=%s + %s" %
                      (m['nomvar'], m['ip1'], m['ip2'], level, dateo, nhours))
        except:
            pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)


    def test_11qd(self):
        import os, sys
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileId = rmn.fstopenall(ATM_MODEL_DFILES+'/bcmk', rmn.FST_RO)
        for k in rmn.fstinl(fileId, nomvar='TT', ip2=12):
            m = rmn.fstprm(k)
            rp1 = rmn.DecodeIp(m['ip1'], m['ip2'], m['ip3'])[0]
            level  = "%f %s" % (rp1.v1, rmn.kindToString(rp1.kind))
            dateo  = "%8.8d.%8.8d" % rmn.newdate(rmn.NEWDATE_STAMP2PRINT,m['dateo'])
            print("CB11qd: %s (%d, %d) level=%s, dateo=%s + %s" %
                  (m['nomvar'], m['ip1'], m['ip2'], level, dateo, float(m['npas'] * m['deet']) / 3600.))
        rmn.fstcloseall(fileId)


    def test_11b(self):
        """
        Queries: List/Find records

        This example shows how to
        * get the list of all records matching selection creterions
        * encode parameters (ip1, datev) values to use selection criteria
                
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.ip1_all
        rpnpy.librmn.fstd98.ip2_all
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.DecodeIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.librmn.base.newdate
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.const
        """
        import os, sys
        import rpnpy.librmn.all as rmn

        # Open all RPNStd files in $ATM_MODEL_DFILES/bcmk_p/anlp2015070706_000
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk_p','anlp2015070706_000')
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Encode selection criteria
            ip1   = rmn.ip1_all(500., rmn.LEVEL_KIND_PMB)
            datev = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, 20150707, 6000000)

            # Get the list of record keys matching
            # nomvar='TT' at 500mb and valide date=20150707.06000000
            keylist = rmn.fstinl(fileId, nomvar='TT', ip1=ip1, datev=datev)
            print("CB11b: Found %d records matching TT, ip1=%d, datev=%d" %
                  (len(keylist), ip1, datev))

            # Get every record meta data
            for k in keylist:
                m = rmn.fstprm(k)
                print("CB11b: %s (%d, %d, %s)" % (m['nomvar'], m['ip1'], m['ip2'], m['datev']))
        except:
            pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)

            
    def test_11bqd(self):
        import os, sys
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileId = rmn.fstopenall(ATM_MODEL_DFILES+'/bcmk_p/anlp2015070706_000', rmn.FST_RO)
        for k in rmn.fstinl(fileId, nomvar='TT',
                            ip1=rmn.ip1_all(500., rmn.LEVEL_KIND_PMB),
                            datev=rmn.newdate(rmn.NEWDATE_PRINT2STAMP, 20150707, 6000000)):
                m = rmn.fstprm(k)
                print("CB11bqd: %s (%d, %d, %s)" % (m['nomvar'], m['ip1'], m['ip2'], m['datev']))
        rmn.fstcloseall(fileId)



    def test_12(self):
        """
        Queries: Get Data and Meta

        This example shows how to
        * get the data of all records matching selection creterions
        * then print statistics

        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.DecodeIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.const
        """
        import os, sys
        import rpnpy.librmn.all as rmn

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open all RPNStd files in the $ATM_MODEL_DFILES/bcmk/ directory
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk')
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Get the list of record keys matching nomvar='TT'
            keylist = rmn.fstinl(fileId, nomvar='TT', ip2=12)
            
            for k in keylist:
                
                # Get the record meta data
                r = rmn.fstluk(k)
                d = r['d']

                # Decode level info and Print statistics
                rp1 = rmn.DecodeIp(r['ip1'], r['ip2'], r['ip3'])[0]
                level  = "%f %s" % (rp1.v1, rmn.kindToString(rp1.kind))
                
                print("CB12: %s (level=%s, ip2=%d) mean=%f, std=%f, min=%f, max=%f" %
                      (r['nomvar'], level, r['ip2'], d.mean(), d.std(), d.min(), d.max()))
        except:
            pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)


    def test_12qd(self):
        import os, sys
        import rpnpy.librmn.all as rmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileId = rmn.fstopenall(ATM_MODEL_DFILES+'/bcmk', rmn.FST_RO)
        for k in rmn.fstinl(fileId, nomvar='TT', ip2=2):
            r = rmn.fstluk(k)
            print("CB12qd: %s (ip1=%s, ip2=%d) mean=%f, std=%f, min=%f, max=%f" %
                  (r['nomvar'], r['ip1'], r['ip2'], r['d'].mean(), r['d'].std(), r['d'].min(), r['d'].max()))
        rmn.fstcloseall(fileId)


    def test_13(self):
        """
        Queries: Get Horizontal Grid info

        This example shows how to get the horizontal grid definition (including
        axes and ezscint id) from a previsouly read record

        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.const
        """
        import os, sys
        import rpnpy.librmn.all as rmn

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open file
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName = os.path.join(ATM_MODEL_DFILES, 'bcmk','geophy.fst')
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Get/read the MG record
            r = rmn.fstlir(fileId, nomvar='ME')

            # Get the grid definition of the record
            g = rmn.readGrid(fileId, r)

            print("CB13: %s grtyp/ref=%s/%s, ni/nj=%d,%d, gridID=%d" %
                  (r['nomvar'], g['grtyp'], g['grref'], g['ni'], g['nj'], g['id']))
            print("     lat0/lon0  =%f, %f" % (g['lat0'], g['lon0']))
            print("     ldlat/dlon =%f, %f" % (g['dlat'], g['dlon']))
            print("     xlat`/xlon1=%f, %f; xlat2/xlon2=%f, %f" %
                  (g['xlat1'], g['xlon1'], g['xlat2'], g['xlon2']))
            print("     ax: min=%f, max=%f; ay: min=%f, max=%f" %
                  (g['ax'].min(), g['ax'].max(), g['ay'].min(), g['ay'].max()))
        except:
            pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)


    def test_13qd(self):
        import os, sys
        import rpnpy.librmn.all as rmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileId = rmn.fstopenall(ATM_MODEL_DFILES+'/bcmk/geophy.fst', rmn.FST_RO)
        r = rmn.fstlir(fileId, nomvar='ME')
        g = rmn.readGrid(fileId, r)
        print("CB13qd: %s grtyp/ref=%s/%s, ni/nj=%d,%d, gridID=%d" %
              (r['nomvar'], g['grtyp'], g['grref'], g['ni'], g['nj'], g['id']))
        print("     lat0/lon0  =%f, %f" % (g['lat0'], g['lon0']))
        print("     ldlat/dlon =%f, %f" % (g['dlat'], g['dlon']))
        print("     xlat`/xlon1=%f, %f; xlat2/xlon2=%f, %f" %
              (g['xlat1'], g['xlon1'], g['xlat2'], g['xlon2']))
        print("     ax: min=%f, max=%f; ay: min=%f, max=%f" %
              (g['ax'].min(), g['ax'].max(), g['ay'].min(), g['ay'].max()))
        rmn.fstcloseall(fileId)


    def test_14(self):
        """
        Queries: Get Vertical Grid Info

        This example shows how to get the vertical grid definition and print some info.

        See also:
        rpnpy.librmn.fstd98.fstopt
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.convertIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_get
        rpnpy.librmn.const
        rpnpy.vgd.const
        """
        import os, sys, datetime
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open file
        fdate    = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF = os.getenv('CMCGRIDF').strip()
        fileName  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Get the vgrid definition present in the file
            v = vgd.vgd_read(fileId)

            # Get Some info about the vgrid
            vkind    = vgd.vgd_get(v, 'KIND')
            vver     = vgd.vgd_get(v, 'VERS')
            tlvl     = vgd.vgd_get(v, 'VIPT')
            try:
                ip1diagt = vgd.vgd_get(v, 'DIPT')
            except:
                ip1diagt = 0
            (ldiagval, ldiagkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1diagt)
            VGD_KIND_VER_INV = dict((v, k) for k, v in vgd.VGD_KIND_VER.items())
            vtype = VGD_KIND_VER_INV[(vkind,vver)]
            print("CB14: Found vgrid type=%s (kind=%d, vers=%d) with %d levels and diag level=%7.2f%s (ip1=%d)" %
                (vtype, vkind, vver, len(tlvl), ldiagval, rmn.kindToString(ldiagkind), ip1diagt))
        except:
            raise
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)


    def test_14b(self):
        """
        Queries: Get Vertical Grid info, Read 3D Field

        This example shows how to
        * get the vertical grid definition.
        * use it to read a 3D field (records on all levels)
        * then print a profile for this var

        See also:
        rpnpy.librmn.fstd98.fstopt
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.fstlinf
        rpnpy.librmn.fstd98.fstlluk
        rpnpy.librmn.fstd98.convertIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_get
        rpnpy.librmn.const
        rpnpy.vgd.const
        """
        import os, sys, datetime
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open file
        fdate     = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF  = os.getenv('CMCGRIDF').strip()
        fileName  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:
            # Get the vgrid definition present in the file
            v = vgd.vgd_read(fileId)

            # Get the list of ip1 on thermo levels in this file
            tlvl = vgd.vgd_get(v, 'VIPT')

            # Trim the list of thermo ip1 to actual levels in files for TT
            # since the vgrid in the file is a super set of all levels
            # and get their "key"
            tlvlkeys = []
            rshape = None
            for ip1 in tlvl:
                (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
                key = rmn.fstinf(fileId, nomvar='TT', ip2=48, ip1=rmn.ip1_all(lval, lkind))
                if key is not None:
                    tlvlkeys.append((ip1, key['key']))
                    if rshape is None:
                        rshape = key['shape']
            rshape = (rshape[0], rshape[1], len(tlvlkeys))
            
            # Read every level for TT at ip2=48, re-use 2d array while reading
            # and store the data in a 3d array
            # with lower level at nk, top at 0 as in the model
            # Note that for efficiency reasons, if only a profile was needed,
            # only that profile would be saved instead of the whole 3d field
            r2d = {'d' : None}
            r3d = None
            k = 0
            for ip1, key in tlvlkeys:
                try:
                    r2d = rmn.fstluk(key, dataArray=r2d['d'])
                    if r3d is None:
                        r3d = r2d.copy()
                        r3d['d'] = np.empty(rshape, dtype=r2d['d'].dtype, order='F')
                    r3d['d'][:,:,k] = r2d['d'][:,:]
                    k += 1
                except:
                    pass
        except:
            pass
        finally:
            # Close file even if an error occured above
            rmn.fstcloseall(fileId)

        # Add the vgrid and the actual ip1 list in the r3d dict, update shape and nk
        r3d['vgd']     = v
        r3d['ip1list'] = [x[0] for x in tlvlkeys]
        r3d['shape']   = rshape
        r3d['nk']      = rshape[2]

        # Print a profile of TT and stats by level
        (i1, j1) = (rshape[0]//2, rshape[1]//2)
        print("CB14b: The TT profile at point (%d, %d) is:" % (i1, j1))
        for k in range(rshape[2]):
            ip1 = r3d['ip1list'][k]
            (ldiagval, ldiagkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
            print("CB14b: TT(%d, %d, %7.2f %s) = %6.1f C [mean=%6.1f, std=%6.1f, min=%6.1f, max=%6.1f]" %
                  (i1, j1, ldiagval, rmn.kindToString(ldiagkind), r3d['d'][i1,j1,k],
                   r3d['d'][:,:,k].mean(), r3d['d'][:,:,k].std(), r3d['d'][:,:,k].min(), r3d['d'][:,:,k].max()))


    def test_14bqd(self):
        import os, sys, datetime
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)
        fdate     = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF  = os.getenv('CMCGRIDF').strip()
        fileId = rmn.fstopenall(CMCGRIDF+'/prog/regeta/'+fdate, rmn.FST_RO)
        v = vgd.vgd_read(fileId)
        (tlvlkeys, rshape) = ([], None)
        for ip1 in vgd.vgd_get(v, 'VIPT'):
            (lval, lkind) = rmn.convertIp(rmn.CONVIP_DECODE, ip1)
            key = rmn.fstinf(fileId, nomvar='TT', ip2=48, ip1=rmn.ip1_all(lval, lkind))
            if key is not None: tlvlkeys.append((ip1, key['key']))
            if rshape is None and key is not None: rshape = key['shape']
        (r2d, r3d, k, rshape) = ({'d' : None}, None, 0, (rshape[0], rshape[1], len(tlvlkeys)))
        for ip1, key in tlvlkeys:
            r2d = rmn.fstluk(key, dataArray=r2d['d'])
            if r3d is None:
                r3d = r2d.copy()
                r3d['d'] = np.empty(rshape, dtype=r2d['d'].dtype, order='F')
            r3d['d'][:,:,k] = r2d['d'][:,:]
        rmn.fstcloseall(fileId)
        r3d.update({'vgd':v, 'ip1list':[x[0] for x in tlvlkeys], 'shape':rshape, 'nk':rshape[2]})
        (i1, j1) = (rshape[0]//2, rshape[1]//2)
        print("CB14bqd: The TT profile at point (%d, %d) is:" % (i1, j1))
        for k in range(rshape[2]):
            (ldiagval, ldiagkind) = rmn.convertIp(rmn.CONVIP_DECODE, r3d['ip1list'][k])
            print("CB14bqd: TT(%d, %d, %7.2f %s) = %6.1f C [mean=%6.1f, std=%6.1f, min=%6.1f, max=%6.1f]" %
                  (i1, j1, ldiagval, rmn.kindToString(ldiagkind), r3d['d'][i1,j1,k],
                   r3d['d'][:,:,k].mean(), r3d['d'][:,:,k].std(), r3d['d'][:,:,k].min(), r3d['d'][:,:,k].max()))
 
    #---- Write / Edit


    def test_21(self):
        """
        Edit: In-Place Record Meta Edition (a la editfst zap w/o copy)
        
        This example shows how to
        * select records in a RPNStd file
        * edit in place their metadata

        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.fst_edit_dir
        rpnpy.librmn.fstd98.fsteff
        rpnpy.librmn.const
        """
        import sys, os, os.path, stat, shutil
        import rpnpy.librmn.all as rmn
        
        # Copy a file locally to be able to edit it and set write permission
        fileName  = 'geophy.fst'
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName0 = os.path.join(ATM_MODEL_DFILES,'bcmk',fileName)
        shutil.copyfile(fileName0, fileName)
        st = os.stat(fileName)
        os.chmod(fileName, st.st_mode | stat.S_IWRITE)

        # Open existing file in Rear/Write mode
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RW_OLD)
        except:
            sys.stderr.write("Problem opening the file: %s\n" % fileName)
            sys.exit(1)

        try:        
            # Get the list of all records key in file
            keylist = rmn.fstinl(fileId)

            # Iterate implicitely on list of records to change the etiket
            rmn.fst_edit_dir(keylist, etiket='MY_NEW_ETK')
        except:
            pass
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileId)
            os.unlink(fileName)  # Remove test file


    def test_21qd(self):
        import sys, os, os.path, stat, shutil
        import rpnpy.librmn.all as rmn
        fileName  = 'geophy.fst'
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileName0 = os.path.join(ATM_MODEL_DFILES,'bcmk',fileName)
        shutil.copyfile(fileName0, fileName)
        st = os.stat(fileName)
        os.chmod(fileName, st.st_mode | stat.S_IWRITE)
        fileId = rmn.fstopenall(fileName, rmn.FST_RW_OLD)
        rmn.fst_edit_dir(rmn.fstinl(fileId), etiket='MY_NEW_ETK')
        rmn.fstcloseall(fileId)
        os.unlink(fileName)  # Remove test file


    def test_22(self):
        """
        Edit: Copy records (a la editfst desire)

        This example shows how to
        * select records in a RPNStd file
        * read the record data + meta
        * write the record data + meta

        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.fsrinl
        rpnpy.librmn.fstd98.fsrluk
        rpnpy.librmn.fstd98.fsrecr
        rpnpy.librmn.const
        """
        import os, os.path, sys
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameIn  = os.path.join(ATM_MODEL_DFILES,'bcmk')
        fileNameOut = 'myfstfile.fst'

        # Open Files
        try:
            fileIdIn  = rmn.fstopenall(fileNameIn)
            fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            # Get the list of records to copy
            keylist1 = rmn.fstinl(fileIdIn, nomvar='UU')
            keylist2 = rmn.fstinl(fileIdIn, nomvar='VV')

            for k in keylist1+keylist2:
                # Read record data and meta from fileNameIn
                r = rmn.fstluk(k)
                
                # Write the record to fileNameOut
                rmn.fstecr(fileIdOut, r)

                print("CB22: Copied %s ip1=%d, ip2=%d, dateo=%s" %
                      (r['nomvar'], r['ip1'], r['ip2'], r['dateo']))
        except:
            pass
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileIdIn)
            rmn.fstcloseall(fileIdOut)
            os.unlink(fileNameOut)  # Remove test file


    def test_22qd(self):
        import os, os.path
        import rpnpy.librmn.all as rmn
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES').strip()
        fileNameOut = 'myfstfile.fst'
        fileIdIn  = rmn.fstopenall(ATM_MODEL_DFILES+'/bcmk')
        fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        for k in rmn.fstinl(fileIdIn, nomvar='UU') + rmn.fstinl(fileIdIn, nomvar='VV'):
            rmn.fstecr(fileIdOut, rmn.fstluk(k))
        rmn.fstcloseall(fileIdIn)
        rmn.fstcloseall(fileIdOut)
        os.unlink(fileNameOut)  # Remove test file


    def test_23(self):
        """
        Edit: Read, Edit, Write records with meta, grid and vgrid

        This example shows how to
        * select records in a RPNStd file
        * read the record data + meta
        * edit/use record data and meta (compute the wind velocity)
        * write the recod data + meta
        * copy (read/write) the record grid descriptors
        * copy (read/write) the file vgrid descriptor

        See also:
        rpnpy.librmn.fstd98.fstopt
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.fsrinl
        rpnpy.librmn.fstd98.fsrluk
        rpnpy.librmn.fstd98.fsrlir
        rpnpy.librmn.fstd98.fsrecr
        rpnpy.librmn.grids.readGrid
        rpnpy.librmn.grids.writeGrid
        rpnpy.vgd.base.vgd_read
        rpnpy.vgd.base.vgd_write
        rpnpy.librmn.const
        rpnpy.vgd.const
        """
        import os, sys, datetime
        from scipy.constants import knot as KNOT2MS
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        CMCGRIDF    = os.getenv('CMCGRIDF').strip()
        fileNameIn  = os.path.join(CMCGRIDF, 'prog', 'regeta', fdate)
        fileNameOut = 'uvfstfile.fst'

        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Open Files
        try:
            fileIdIn  = rmn.fstopenall(fileNameIn)
            fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        except:
            sys.stderr.write("Problem opening the files: %s, %s\n" % (fileNameIn, fileNameOut))
            sys.exit(1)

        try:
            # Copy the vgrid descriptor
            v = vgd.vgd_read(fileIdIn)
            vgd.vgd_write(v, fileIdOut)
            print("CB23: Copied the vgrid descriptor")
            
            # Loop over the list of UU records to copy 
            uu = {'d': None}
            vv = {'d': None}
            uvarray = None
            copyGrid = True
            for k in rmn.fstinl(fileIdIn, nomvar='UU'):
                # Read the UU record data and meta from fileNameIn
                # Provide data array to re-use memory
                uu = rmn.fstluk(k, dataArray=uu['d'])
                
                # Read the corresponding VV
                # Provide data array to re-use memory
                vv = rmn.fstlir(fileIdIn, nomvar='VV',
                                ip1=uu['ip1'], ip2=uu['ip2'], datev=uu['datev'],
                                dataArray=vv['d'])

                # Compute the wind modulus in m/s
                # Copy metadata from the UU record
                # Create / re-use memory space for computation results
                uv = uu.copy()
                if uvarray is None:
                    uvarray = np.empty(uu['d'].shape, dtype=uu['d'].dtype, order='F')
                uv['d'] = uvarray
                uv['d'][:,:] = np.sqrt(uu['d']**2. + vv['d']**2.)
                uv['d'] *= KNOT2MS  # Convert from knot to m/s

               # Set new record name and Write it to fileNameOut
                uv['nomvar'] = 'WSPD'
                rmn.fstecr(fileIdOut, uv)
                
                print("CB23: Wrote %s ip1=%d, ip2=%d, dateo=%s : mean=%f" %
                      (uv['nomvar'], uv['ip1'], uv['ip2'], uv['dateo'], uv['d'].mean()))

                # Read and Write grid (only once, all rec are on the same grid)
                if copyGrid:
                    copyGrid = False
                    g = rmn.readGrid(fileIdIn, uu)
                    rmn.writeGrid(fileIdOut, g)
                    print("CB23: Copied the grid descriptors")
        except:
            pass
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileIdIn)
            rmn.fstcloseall(fileIdOut)
            os.unlink(fileNameOut)  # Remove test file


    def test_23qd(self):
        import os, sys, datetime
        from scipy.constants import knot as KNOT2MS
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd
        fdate       = datetime.date.today().strftime('%Y%m%d') + '00_048'
        fileNameOut = 'uvfstfile.fst'
        fileIdIn    = rmn.fstopenall(os.getenv('CMCGRIDF')+'/prog/regeta/'+fdate)
        fileIdOut   = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        vgd.vgd_write(vgd.vgd_read(fileIdIn), fileIdOut)
        (uu, vv, uvarray, copyGrid) = ({'d': None}, {'d': None}, None, True)
        for k in rmn.fstinl(fileIdIn, nomvar='UU'):
            uu = rmn.fstluk(k, dataArray=uu['d'])
            vv = rmn.fstlir(fileIdIn, nomvar='VV', ip1=uu['ip1'], ip2=uu['ip2'],
                            datev=uu['datev'],dataArray=vv['d'])
            if uvarray is None:
                uvarray = np.empty(uu['d'].shape, dtype=uu['d'].dtype, order='F')
            uv = uu.copy()
            uv.update({'d':uvarray, 'nomvar': 'WSPD'})
            uv['d'][:,:] = np.sqrt(uu['d']**2. + vv['d']**2.) * KNOT2MS
            rmn.fstecr(fileIdOut, uv)
            if copyGrid:
                copyGrid = False
                rmn.writeGrid(fileIdOut, rmn.readGrid(fileIdIn, uu))
        rmn.fstcloseall(fileIdIn)
        rmn.fstcloseall(fileIdOut)
        os.unlink(fileNameOut)  # Remove test file


    def test_24(self):
        """
        Edit: New file from scratch

        This example shows how to
        * Create a record meta and data from scratch
        * Create grid descriptors for the data
        * Create vgrid descriptor for the data
        * write the recod data + meta
        * write the grid descriptors
        * write the vgrid descriptor

        See also:
        rpnpy.librmn.fstd98.fstopt
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.fsrecr
        rpnpy.librmn.fstd98.dtype_fst2numpy
        rpnpy.librmn.grids.encodeGrid
        rpnpy.librmn.grids.defGrid_ZE
        rpnpy.librmn.grids.writeGrid
        rpnpy.librmn.base.newdate
        rpnpy.vgd.base.vgd_new
        rpnpy.vgd.base.vgd_new_pres
        rpnpy.vgd.base.vgd_new_get
        rpnpy.vgd.base.vgd_write
        rpnpy.librmn.const
        rpnpy.vgd.const
        """
        import os, sys
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd
        
        # Restrict to the minimum the number of messages printed by librmn
        rmn.fstopt(rmn.FSTOP_MSGLVL,rmn.FSTOPI_MSG_CATAST)

        # Create Record grid
        gp = {
            'grtyp' : 'Z',
            'grref' : 'E',
            'ni'    : 90,
            'nj'    : 45,
            'lat0'  : 35.,
            'lon0'  : 250.,
            'dlat'  : 0.5,
            'dlon'  : 0.5,
            'xlat1' : 0.,
            'xlon1' : 180.,
            'xlat2' : 1.,
            'xlon2' : 270.
        }
        g = rmn.encodeGrid(gp)
        print("CB24: Defined a %s/%s grid of shape=%d, %d" %
              (gp['grtyp'], gp['grref'], gp['ni'], gp['nj']))
            
        # Create Record vgrid
        lvls = (500.,850.,1000.)
        v = vgd.vgd_new_pres(lvls)
        ip1list = vgd.vgd_get(v, 'VIPT')
        print("CB24: Defined a Pres vgrid with lvls=%s" % str(lvls))
        
        # Create Record data + meta
        datyp   = rmn.FST_DATYP_LIST['float_IEEE_compressed']
        npdtype = rmn.dtype_fst2numpy(datyp)
        rshape  = (g['ni'], g['nj'], len(ip1list))
        r = rmn.FST_RDE_META_DEFAULT.copy() # Copy default record meta
        r.update(g)                         # Update with grid info
        r.update({                          # Update with specific meta and data array
            'nomvar': 'MASK',
            'dateo' : rmn.newdate(rmn.NEWDATE_PRINT2STAMP, 20160302, 1800000),
            'nk'    : len(ip1list),
            'ip1'   : 0,     # level, will be set later, level by level
            'ip2'   : 6,     # Forecast of 6h
            'deet'  : 3600,  # Timestep in sec
            'npas'  : 6,     # Step number
            'etiket': 'my_etk',
            'nbits' : 32,    # Keep full 32 bits precision for that field
            'datyp' : datyp, # datyp (above) float_IEEE_compressed
            'd'     : np.empty(rshape, dtype=npdtype, order='F')
            })
        print("CB24: Defined a new record of shape=%d, %d" % (r['ni'], r['nj']))

        
        # Compute record values
        r['d'][:,:,:] = 0.
        r['d'][10:-11,5:-6,:] = 1.
        
        # Open Files
        fileNameOut = 'newfromscratch.fst'
        try:
            fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        except:
            sys.stderr.write("Problem opening the file: %s, %s\n" % fileNameOut)
            sys.exit(1)

        # Write record data + meta + grid + vgrid to file
        try:
            r2d = r.copy()
            r2d['nk']  = 1
            for k in range(len(ip1list)):
                r2d['ip1'] = ip1list[k]
                r2d['d'] = np.asfortranarray(r['d'][:,:,k])
                rmn.fstecr(fileIdOut, r2d['d'], r2d)
                print("CB24: wrote %s at ip1=%d" % (r2d['nomvar'], r2d['ip1']))
            rmn.writeGrid(fileIdOut, g)
            print("CB24: wrote the grid descriptors")
            vgd.vgd_write(v, fileIdOut)
            print("CB24: wrote the vgrid descriptor")
        except:
            raise
        finally:
            # Properly close files even if an error occured above
            # This is important when editing to avoid corrupted files
            rmn.fstcloseall(fileIdOut)
            os.unlink(fileNameOut)  # Remove test file


    def test_24qd(self):
        import os, sys
        import numpy as np
        import rpnpy.librmn.all as rmn
        import rpnpy.vgd.all as vgd
        g = rmn.defGrid_ZE(90, 45, 35., 250., 0.5, 0.5, 0., 180., 1., 270.)
        lvls = (500.,850.,1000.)
        v = vgd.vgd_new_pres(lvls)
        ip1list = vgd.vgd_get(v, 'VIPT')
        datyp   = rmn.FST_DATYP_LIST['float_IEEE_compressed']
        npdtype = rmn.dtype_fst2numpy(datyp)
        rshape  = (g['ni'], g['nj'], len(ip1list))
        r = rmn.FST_RDE_META_DEFAULT.copy()
        r.update(g)
        r.update({
            'nomvar': 'MASK',   'nk'    : len(ip1list),
            'dateo' : rmn.newdate(rmn.NEWDATE_PRINT2STAMP, 20160302, 1800000),
            'ip2'   : 6,        'deet'  : 3600, 'npas'  : 6,
            'etiket': 'my_etk', 'datyp' : datyp,
            'd'     : np.empty(rshape, dtype=npdtype, order='F')
            })
        r['d'][:,:,:] = 0.
        r['d'][10:-11,5:-6,:] = 1.
        fileNameOut = 'newfromscratch.fst'
        fileIdOut = rmn.fstopenall(fileNameOut, rmn.FST_RW)
        r2d = r.copy()
        for k in range(len(ip1list)):
            r2d.update({'nk':1, 'ip1':ip1list[k], 'd':np.asfortranarray(r['d'][:,:,k])})
            rmn.fstecr(fileIdOut, r2d['d'], r2d)
        rmn.writeGrid(fileIdOut, g)
        vgd.vgd_write(v, fileIdOut)
        rmn.fstcloseall(fileIdOut)
        os.unlink(fileNameOut)  # Remove test file


if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
