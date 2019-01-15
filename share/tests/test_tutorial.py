#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
This basic tutorial will walk you through a few basic use cases of the rpnpy python package. You may build on these to develop your own applications. Reference material is available for the full details on the classes and functions used here.

This tutorial assumes you already know the python language version 2.* basics and are familiar with the numpy python package.
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

See Also:
  * RPNpy Cookbook
  * RPNpy Reference
"""

import unittest

class rpnpyTutorial(unittest.TestCase):

    
    def test_1(self):
        """
        Open/Close File
        
        Note: The following constants may be used to set the file mode: rmn.FST_RW , rmn.FST_RW_OLD , rmn.FST_RO

        See also:
        rpnpy.librmn.fstd98.isFST
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os
        import rpnpy.librmn.all as rmn

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/geophy.fst')

        if not rmn.isFST(fileName):
            raise rmn.FSTDError("Not an FSTD file: %s " % fileName)

        # Open
        try:
             fileId = rmn.fstopenall(fileName,rmn.FST_RO)
        except:
            raise rmn.FSTDError("File not found/readable: %s" % fileName)

        # ...

        # Close
        rmn.fstcloseall(fileId)        

        
    def test_2(self):
        """
        Find Record / Get Metadata

        Most librmn FSTD functions are supported to look for records matching the provided selection criterion.
        Criteria not specified are not used for selection (wildcard).
        The fstprm function can then be used to get the record metadata.

        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinfx
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstsui
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os
        import rpnpy.librmn.all as rmn

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')

        # Open
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            raise rmn.FSTDError("File not found/readable: %s" % fileName)

        # Find
        try:
            k1      = rmn.fstinf(fileId)['key']                    #No criterion, this match the first rec in the file
            pr_key  = rmn.fstinf(fileId, nomvar='PR')['key']       #this match the first record named PR
            # pr_key2 = rmn.fstinfx(pr_key, fileId, nomvar='PR')['key']  #this would match the next  record named PR
        except:
            raise rmn.FSTDError("Problem searching in File: %s" % fileName)

        if not pr_key:
            raise rmn.FSTDError("Record not found in File: %s" % fileName)

        # Read Metadata
        try:
            pr_meta = rmn.fstprm(pr_key)
            for k in ('nomvar', 'dateo', 'npas'):
                print("%s = %s" % (k, str(pr_meta[k])))
        except:
            raise rmn.FSTDError("Error: Problem getting record metadata")

        # ...

        # Close
        rmn.fstcloseall(fileId)
        

    def test_3(self):
        """
        Read a record

        fstluk is the main function used to read record data and metadata after a search for the record handle. Other Librmn functions that find/seclect and read at the same time can also be used: fstlir, fstlirx, fstlis
        
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstlirx
        rpnpy.librmn.fstd98.fstlis
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os
        import numpy as np
        import rpnpy.librmn.all as rmn
        
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
        pr_rec = None

        #Open file and read record data and metadata
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
            pr_key = rmn.fstinf(fileId, nomvar='PR')['key']  #this match the first record named PR
            if pr_key:
                pr_rec = rmn.fstluk(pr_key)
        except:
            raise rmn.FSTDError("Problem reading record in file: %s" % fileName)

        # Close
        rmn.fstcloseall(fileId)

        # Computations
        if pr_rec:
            average = np.average(pr_rec['d'])
            print("Read nomvar=%s, dateo=%s, hour=%s, average=%s" % 
                  (pr_rec['nomvar'], str(pr_rec['dateo']), str(pr_rec['ip2']), str(average)))
    
    def test_4(self):
        """
        Change record metadata

        You can change the metadata of a record in an FSTD file in place (a la Editfst's zap function) with a simple call to
        fst_edit_dir. No need to write the record in another file.
        All parameters not specified in the call will be kept to their present value.
        
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fst_edit_dir
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os, os.path, stat, shutil
        import rpnpy.librmn.all as rmn

        # Take an editable copy of the file
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName0 = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
        fileName = 'some_rpnstd_file.fst'
        shutil.copyfile(fileName0, fileName)
        st = os.stat(fileName)
        os.chmod(fileName, st.st_mode | stat.S_IWRITE)

        # Change nomvar for the PR record
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RW)
            pr_key = rmn.fstinf(fileId, nomvar='PR')['key']    #Match the first record named PR
            if pr_key:
                rmn.fst_edit_dir(pr_key, nomvar='PR0', ip2=0)   #Rename the field to PR0 and set ip2 to 0 (zero)
        except:
            raise rmn.FSTDError("Problem editing record meta in File: %s" % fileName)
 
        # Close
        rmn.fstcloseall(fileId)

        # Erase test file
        os.unlink(fileName)

    
    def test_5(self):
        """
        Erase a record

        You can delete a record in an FSTD file (a la Editfst's exclude function) with a simple call to fsteff.
        No need to write all the other records in another file.
        
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fsteff
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os, os.path, stat, shutil
        import rpnpy.librmn.all as rmn

        # Take an editable copy of the file
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName0 = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
        fileName = 'some_rpnstd_file.fst'
        shutil.copyfile(fileName0, fileName)
        st = os.stat(fileName)
        os.chmod(fileName, st.st_mode | stat.S_IWRITE)

        # Erase record named PR in file
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RW)
            pr_key = rmn.fstinf(fileId, nomvar='PR')['key']   #Match the first record named PR
            if pr_key:
                rmn.fsteff(pr_key) #Erase previously found record
        except:
            raise rmn.FSTDError("Problem erasing record in File: %s" % fileName)

        # Close
        rmn.fstcloseall(fileId)

        # Erase test file
        os.unlink(fileName)


    def test_6(self):
        """
        Write a record

        Starting from the read a record example above we can change the data and meta before writing
        it as another record in the same file or in another file.
        
        See also:
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstluk
        rpnpy.librmn.fstd98.fstlir
        rpnpy.librmn.fstd98.fstecr
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os, os.path, sys
        import numpy as np
        import rpnpy.librmn.all as rmn

        # open input file and read PR record
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName  = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
        except:
            raise rmn.FSTDError("Problem opening File: %s" % fileName)
        try:
            pr_rec = rmn.fstlir(fileId, nomvar='PR')  # Read 1st record matching nomvar=PR
        except:
            sys.stdout.write("Problem reading record in File: %s" % fileName)
        finally:
            rmn.fstcloseall(fileId)

        # open output file and write record
        fileName  = 'some_rpnstd_file.fst'
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RW)
        except:
            raise rmn.FSTDError("Problem opening File: %s" % fileName)
        try:
            pr_data  = pr_rec['d']
            pr_data /= max(1.e-5, np.amax(pr_data))
            pr_rec['nomvar'] = 'PRN1'
            rmn.fstecr(fileId, pr_data, pr_rec)
        except:
            sys.stdout.write("Problem writing record in File: %s" % fileName)
        finally:
            rmn.fstcloseall(fileId)
            
        # Erase test file
        os.unlink(fileName)


    def test_7(self):
        """
        Manipulating Dates

        Dates in FSTD files are encoded integers. Conversion to more human friendly formats
        and manipulation can be done using the RPNDate and RPNDateRange classes or with the newdate function.
        
        See also:
        rpnpy.rpndate.RPNDate
        rpnpy.rpndate.RPNDateRange
        rpnpy.librmn.base.newdate
        rpnpy.librmn.base.incdatr
        rpnpy.librmn.base.difdatr
        rpnpy.librmn.const
        """
        from rpnpy.rpndate import RPNDate, RPNDateRange
 
        d1 = RPNDate(20030423, 11453500)
        d2 = RPNDate(d1)
        d2 = d2.incr(48)
        dr = RPNDateRange(d1, d2, 6)
        print(str(dr.lenght()))
        # 48.0
 
        for d3 in dr:
            print(str(d3))
        # RPNDate(20030423,11453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=     0.0)
        # RPNDate(20030423,17453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=     6.0)
        # RPNDate(20030423,23453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    12.0)
        # RPNDate(20030424,05453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    18.0)
        # RPNDate(20030424,11453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    24.0)
        # RPNDate(20030424,17453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    30.0)
        # RPNDate(20030424,23453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    36.0)
        # RPNDate(20030425,05453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    42.0)
        # RPNDate(20030425,11453500) ; RPNDate(20030423,11453500,dt=  3600.0,nstep=    48.0)

        ## Lower level functions can also be used wherever more convenient.
        import rpnpy.librmn.all as rmn
 
        yyyymmdd = 20150102 #Jan 2nd, 2015
        hhmmsshh = 13141500 #13h 14min 15sec
        cmcdate  = rmn.newdate(rmn.NEWDATE_PRINT2STAMP, yyyymmdd, hhmmsshh)
 
        nhours   = 6.
        cmcdate2 = rmn.incdatr(cmcdate, nhours)
 
        (yyyymmdd2, hhmmsshh2) = rmn.newdate(rmn.NEWDATE_STAMP2PRINT, cmcdate2)
        print("%06d.%06d + %4.1fh = %06d.%06d" % (yyyymmdd, hhmmsshh, nhours, yyyymmdd2, hhmmsshh2))
        # 20150102.13141500 +  6.0h = 20150102.19141500
 
        nhours2 = rmn.difdatr(cmcdate2, cmcdate)
        print("%06d.%06d - %06d.%06d = %4.1fh" % (yyyymmdd2, hhmmsshh2, yyyymmdd, hhmmsshh, nhours2))
        # 20150102.19141500 - 20150102.13141500 =  6.0h

        
    def test_8(self):
        """
        Manipulating Level information
        
        Level (time and height) information in FSTD files are encoded as integers in ip1,ip2,ip3 and
        sometime in the Vgrid !! record.
        To make things more complex, 2 encoding formats are supported, old and new.
        Conversion to more human friendly formats and manipulation can be done using many rpnpy functions.
        """
        pass


    def test_9(self):
        """
        Decoding values

        When reading an FSTD record metadata, the ip1, ip2, ip3 contains the encoded time and levels values.
        In the old format, ip1 is used for the level value and ip2 is a none encoded time value in hours.
        In the new format, all ip1, ip2, ip3 can be used to specify time and level as well as ranges.
        
        See also:
        rpnpy.librmn.fstd98.convertIp
        rpnpy.librmn.fstd98.convertIPtoPK
        rpnpy.librmn.fstd98.DecodeIp
        rpnpy.librmn.fstd98.kindToString
        rpnpy.librmn.proto.FLOAT_IP
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os
        import rpnpy.librmn.all as rmn

        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')

        # Get list of records
        try:
            fileId  = rmn.fstopenall(fileName, rmn.FST_RO)
            keyList = rmn.fstinl(fileId, nomvar='TT')
        except:
            raise rmn.FSTDError("Problem getting list of TT records from file: %s" % fileName)

        # Get metadata and Decode level value
        try:
            for k in keyList:
                recMeta = rmn.fstprm(k)
                (level, ikind) = rmn.convertIp(rmn.CONVIP_DECODE, recMeta['ip1'])
                kindstring     = rmn.kindToString(ikind)
                print("Found %s at level %f %s" % (recMeta['nomvar'], level, kindstring))
        except:
            raise rmn.FSTDError("Problem getting metadata for TT from file: %s " % fileName)

        rmn.fstcloseall(fileId)


    def test_10(self):
        """
        Encoding values
        
        Encoding values are usefull in 2 situations:
        * providing the metadata when writing a record, it is best to encode in the new format then
        * specify search criterions to read a record, it is best to search for the old and new formats,
          the ip1_all and ip2_all functions can be used for that sake as long as no value range are needed.

        See also:
        rpnpy.librmn.fstd98.convertIp
        rpnpy.librmn.fstd98.convertIPtoPK
        rpnpy.librmn.fstd98.EncodeIp
        rpnpy.librmn.fstd98.ip1_all
        rpnpy.librmn.fstd98.ip2_all
        rpnpy.librmn.fstd98.ip3_all
        rpnpy.librmn.proto.FLOAT_IP
        rpnpy.librmn.fstd98.fstopenall
        rpnpy.librmn.fstd98.fstinf
        rpnpy.librmn.fstd98.fstinl
        rpnpy.librmn.fstd98.fstprm
        rpnpy.librmn.fstd98.fstcloseall
        rpnpy.librmn.fstd98.FSTDError
        rpnpy.librmn.RMNError
        rpnpy.librmn.const
        """
        import os
        import rpnpy.librmn.all as rmn
 
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
 
        ip1new = rmn.convertIp(rmn.CONVIP_ENCODE,    850., rmn.KIND_PRESSURE)
        ip1old = rmn.convertIp(rmn.CONVIP_ENCODE_OLD, 10., rmn.KIND_ABOVE_SEA)
 
        ip1newall = rmn.ip1_all(1., rmn.KIND_HYBRID)
 
        # Use ip1newall as a search criterion to find a record
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
            ktt    = rmn.fstinf(fileId, nomvar='TT', ip1=ip1newall)
        except:
            raise rmn.FSTDError("Problem finding of TT with ip1=%d record from file: %s" % (ip1newall, fileName))
 
        if not ktt:
            print("Not Found: TT with ip1=%d record from file: %s" % (ip1newall, fileName))


    def test_11(self):
        """
        Grids and Interpolation

        This tutorial section will show you how to define a grid in a few different way and use
        the Ezscint package interface in python-RPN to interpolate data from one grid to another.

        This tutorial parts suppose you already know about how grids are defined in FSTD files.
        If it is not already the case you may head to:
        * A tentative FSTD grids tutorial
        * http://web-mrb.cmc.ec.gc.ca/science//si/eng/si/misc/grilles.html
        """
        pass


    def test_12(self):
        """
        Defining Grids

        Grids can be defined from
        * a record in a FSTD file or
        * directly by providing parameters.
        These grids parameters can be used to geolocate the data points and
        for interpolation operations (see below).

        See also:
        """
        import os
        import rpnpy.librmn.all as rmn
 
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')
 
        # Define a grid from parameters, a cylindrical equidistant (LatLon) grid with 0.5 deg spacing.
        paramsL = {
            'grtyp' : 'L', # Cylindrical equidistant projection
            'ni'   :  90,  # Grid dimension: 90 by 45 points
            'nj'   :  45,
            'lat0' :   0., # Grid lower-left (south-west) corner (point 1,1) is located at 0N, 180E
            'lon0' : 180.,
            'dlat' :   0.5,# The grid has a resolution (grid spacing) of 0.5 deg. on both axes
            'dlon' :   0.5
            }
        try:
            gridL = rmn.encodeGrid(paramsL)
        except:
            raise rmn.FSTDError("Problem defining a grid with provided parameters: %s " % str(paramsL))
 
        # Get a grid definition from a record in a FSTD file
        try:
            fileId = rmn.fstopenall(fileName, rmn.FST_RO)
            prKey  = rmn.fstinf(fileId, nomvar='PR')['key']
            prMeta = rmn.fstprm(prKey)              # Get the record metadata along with partial grid info
            prMeta['iunit'] = fileId
            prGrid0 = rmn.ezqkdef(prMeta)           # use ezscint to retreive full grid info
            prGrid  = rmn.decodeGrid(prGrid0)       # Decode all the grid parameters values
            rmn.fstcloseall(fileId)
        except:
            raise rmn.FSTDError("Problem getting PR record grid meta from file: %s" % fileName)


    def test_13(self):
        """
        Interpolating Data

        Interpolating data to/from known FSTD grids is made easy with the Ezscint package.
        There are a few exceptions though
        * you can only interpolate to a Y grid, not from it.
        * multi-parts grids (Yin-Yang, ...) have to be dealth with in a special way (see below)
        In this example we'll interpolate forecast data onto the analysis grid to make some computations
        
        See also:
        """
        import os
        import rpnpy.librmn.all as rmn
 
        ATM_MODEL_DFILES = os.getenv('ATM_MODEL_DFILES')
        fileName0 = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_000')  #Analysis
        fileName1 = os.path.join(ATM_MODEL_DFILES.strip(), 'bcmk/2009042700_012')  #Forecast
 
        # Get data and grid definition for P0 in the 1st FSTD file
        try:
            fileId   = rmn.fstopenall(fileName0, rmn.FST_RO)
            p0Data1  = rmn.fstlir(fileId, nomvar='P0')  # Get the record data and metadata along with partial grid info
            p0Data1['iunit'] = fileId
            p0GridId = rmn.ezqkdef(p0Data1)             # use ezscint to retreive a grid id
            p0Grid1  = rmn.decodeGrid(p0GridId)         # Decode all the grid parameters values
            rmn.fstcloseall(fileId)
        except:
            raise rmn.FSTDError("Problem getting P0 record grid meta from file: %s" % fileName0)
 
        # Get data and grid definition for P0 in the 2nd FSTD file
        try:
            fileId   = rmn.fstopenall(fileName1, rmn.FST_RO)
            p0Data2  = rmn.fstlir(fileId, nomvar='P0', ip2=12)  # Get the record data and metadata along with partial grid info
            p0Data2['iunit'] = fileId
            p0GridId = rmn.ezqkdef(p0Data2)                     # use ezscint to retreive a grid id
            p0Grid2  = rmn.decodeGrid(p0GridId)                 # Decode all the grid parameters values
            rmn.fstcloseall(fileId)
        except:
            raise rmn.FSTDError("Problem getting P0 record grid meta from file: %s " % fileName1)
 
        # Make a cubic interpolation of p0Data2 onto p0Grid1 with extrapolated values set to Minvalue of the field
        rmn.ezsetopt(rmn.EZ_OPT_EXTRAP_DEGREE, rmn.EZ_EXTRAP_MIN)
        rmn.ezsetopt(rmn.EZ_OPT_INTERP_DEGREE, rmn.EZ_INTERP_LINEAR)
        p0Data2_onGrid1 = rmn.ezsint(p0Grid1['id'], p0Grid2['id'], p0Data2['d'])
 
        # Make some computation
        p0Diff = p0Data2_onGrid1 - p0Data1['d']

        
    def test_14(self):
        """
        """
        pass
    
if __name__ == "__main__":
    unittest.main()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
