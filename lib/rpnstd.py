#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mario Lepine <mario.lepine@canada.ca>
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Author: Christopher Subich <Christopher.Subich@canada.ca>
# Copyright: LGPL 2.1

"""
Module RPN contains the classes used to access RPN Standard Files (rev 2000).
"""

#__docformat__ = 'restructuredtext'

import rpnpy.version as rpn_version
from rpn_helpers import *
import types
import datetime
import pytz
import numpy
import Fstdc


FILE_MODE_RO     = Fstdc.FSTDC_FILE_RO
FILE_MODE_RW     = Fstdc.FSTDC_FILE_RW
FILE_MODE_RW_OLD = Fstdc.FSTDC_FILE_RW_OLD


class RPNFile:
    """
    Python Class implementation of the RPN standard file interface.

    Instanciating this class actually opens the file.
    Deleting the instance close the file.

    Attributes:
       filename  :
       lastread  :
       lastwrite :
       options   :
       iun       :

    Raises:
       TypeError
       IOError    if unable to open file

    Examples:
    myRPNFile = RPNFile(name, mode)      #opens the file
    params = myRPNFile.info(seachParams) #get matching record params
    params = myRPNFile.info(FirstRecord) #get params of first rec on file
    params = myRPNFile.info(NextMatch)   #get next matching record params
    myRPNRec = myRPNFile[seachParams]    #get matching record data and params
    myRPNRec = myRPNFile[FirstRecord]    #get data and params of first rec on file
    myRPNRec = myRPNFile[NextMatch]      #get next matching record data and params
    myRPNFile[params]   = mydataarray    #append data and tags to file
    myRPNFile[myRPNRec] = myRPNRec.d     #append data and tags to file
    myRPNFile.write(myRPNRec)            #append data and tags to file
    myRPNFile.write(myRPNRec, rewrite=True)  #rewrite data and tags to file
    myRPNFile.rewrite(myRPNRec)          #rewrite data and tags to file
    myRPNFile.append(myRPNRec)           #append data and tags to file
    myRPNFile[myRPNRec] = None           #erase record
    myRPNFile[params.handle] = None      #erase record
    del myRPNFile                        #close the file

    See Also:
       RPNRec
    """

    def __init__(self, name=None, mode=FILE_MODE_RW):
        """
        Constructor.

        Args:
           name : string
              file name
           mode : string, optional
              Type of file, FILE_MODE_RO, FILE_MODE_RW, FILE_MODE_RW_OLD
        """
        if not (isinstance(name, str) and name):
            raise TypeError('RPNFile, need to provide a name for the file')
        self.filename = name
        self.lastread = None
        self.lastwrite = None
        self.options = mode
        self.iun = None
        self.iun = Fstdc.fstouv(0, self.filename, self.options)
        if self.iun is None:
            raise IOError(-1, 'failed to open standard file', self.filename)

    def voir(self, options='NEWSTYLE'):
        """
        Print the file content listing.
        """
        Fstdc.fstvoi(self.iun, options)

    def close(self):
        if self.iun != None:
            Fstdc.fstfrm(self.iun)
            #print 'file ', self.iun, ' is closed, filename=', self.filename
            self.iun = None

    def rewind(self):
        pass

    def __del__(self):
        """
        Close File
        """
        self.close()
        del self.filename
        del self.lastread
        del self.lastwrite
        del self.options
        del self.iun

    def __getitem__(self, key):
        """
        Get the record, meta and data (RPNRec), corresponding to the seach keys from file

        myrec = myRPNfile[mykey]

        Args:
           mykey:
               search keys for RPNFile.info()
        Returns:
           RPNRec instance, data + meta of the record
           None if rec not found
        """
        params = self.info(key)         # 1 - get handle
        if params is None:              # oops !! not found
            return None
        target = params.handle
        array = Fstdc.fstluk(target)   # 2 - get data
        #TODO: make ni, nj, nk consistent?
        #TODO: update self.grid?
        if params.datyp == 7:
            # String types need some special handling.  They are returned from
            # Fstdc.fstluk as N-dimensional arrays of one character, but we can
            # do better.  For datyp=7, ni corresponds to the maximum length of
            # the string.

            # Officially, this datatype requires nj and nk to be 1 -- see
            #    http://web-mrb.cmc.ec.gc.ca/mrb/si/eng/si/index.html
            # for the fuller documentation.  Unofficially, this requirement
            # is broken by GEM itself, for writing out an array of string-based
            # station names with nj != 1.  We'll support that convention.

            # Create a string of the appropriate maximum length
            dtype_str = numpy.dtype((numpy.str_, params.ni))

            # Now, as an additional complication, there is an incompatibility bug
            # within rmnlib.  Prior to version 15.1, strings were written to file
            # with their high-order bit flipped.  With version 15.1 and later, they
            # are written properly.  This is consistent within versions, but it means
            # that reading strings across the version barrier will give inconsistent
            # results.

            # To detect whether this has happened (as a precursor to fixing it),
            # let's look at the last row of output.  Fortran strings are space-padded
            # to the maximum length, so we would expect the last row of output to mostly
            # consist of ' ' characters.  If instead they mostly have the high-bit set,
            # then we can reasonably conclude that we're reading across a version
            # incompatibility.

            # Reshape a potentially 1D or 3D array into a 2D array
            array = array.reshape(params.ni, -1)

            if numpy.sum(array[params.ni-1, :].view(numpy.uint8) > 127) > \
                    0.5*params.nj*params.nk:
                # Correct the error
                array = (array.view(numpy.uint8) ^ 128).view((numpy.str_, 1))

            # Replace the array view with one that collapses individual characters
            array = array.view(dtype_str).reshape((1, -1))

            # Remove the space padding
            for kk in range(0, array.shape[1]):
                array[0, kk] = array[0, kk].rstrip(' ')

            # And restore the proper shape.  Having a singleton first dimension is
            # useless, so collapse that away.  This will be a no-op for variables
            # that are simply one single string, and for variables such as the gem
            # timeseries station list we'll get a 1D array for what was a 1D array
            # of strings in the original Fortran
            if params.nk > 1:
                array = array.reshape((params.nj, params.nk))
            elif params.nj > 1:
                array = array.reshape(params.nj)
            else:
                array = array.reshape(1)

        return RPNRec(array, params)

    def __contains__(self, key):
        """
        Returns True if 'key' is contained in this RPNFile, 'False' otherwise

        is_here = mykey in myRPNfile
        @param mykey Search key passed to RPNFile.info() (instance of RPNMeta)
        @return True if the key is present, False otherwise
        """
        try:
            self.info(key)
            # self.info raises an exception if the key isn't found, so
            # reaching this point means that the record is in the file.
            return True
        except Fstdc.error:
            # An exception means that the key was not found
            return False

    def edit_dir_entry(self, key):
        """
        Edit (zap) directory entry referenced by handle

        myRPNdfile.edit_dir_entry(myNewRPNParams)

        myNewRPNParams.handle must be a valid rec/file handle as retrieved by myRPNdfile.info()
        """
        return(Fstdc.fst_edit_dir(key.handle, key.date, key.deet, key.npas,
                                  -1, -1, -1, key.ip1, key.ip2, key.ip3,
                                  key.type, key.nom, key.etiket, key.grtyp,
                                  key.ig1, key.ig2, key.ig3, key.ig4,
                                  key.datyp))

    def info(self, key, list=False):
        """
        Seach file for next record corresponding to search keys

        Successive calls will go further in the file.
        Search index can be reset to begining of file with myRPNfile.info(FirstRecord)
        If key.handle >=0, return key w/o search and w/o checking the file

        myRPNparms = myRPNfile.info(FirstRecord)
        myRPNparms = myRPNfile.info(mykeys)
        myRPNparms = myRPNfile.info(NextMatch)
        myRPNparms = myRPNfile.info(mykeys, list=True)

        Args:
           mykeys: RPNParm or derived classes (RPNKeys, RPNDesc, RPNMeta, RPNRec)
              search keys
           list: bool
              if true, return a list of all rec RPNMeta matching the search keys (handle is then ignored)
        Returns:
           RPNMeta instance of the record with proper handle, return None if not found
        Raises:
           TypeError if

        Accepted seach keys: nom, type,
                              etiket, ip1, ip2, ip3, datev, handle
        TODO: extend accepted seach keys to all RPNMeta keys

        The myRPNfile.lastread parameter is set with values of all latest found rec params
        """
        if isinstance(key, RPNKeys):# RPNMeta is derived from RPNKeys
            if list:
                mylist = Fstdc.fstinl(self.iun, key.nom, key.type,
                              key.etiket, key.ip1, key.ip2, key.ip3,
                              key.datev)
                mylist2 = []
                for item in mylist:
                    # The parameters read via rmnlib don't include datev,
                    # but that's useful for searching later on (it's
                    # part of RPNKeys).  We can calculate datev, however,
                    # from dateo, npas, and deet
                    if item['dateo'] != -1:
                        dateo = item['dateo']
                        npas = item['npas']
                        deet = item['deet']
                        try:
                            datev = RPNDate(
                                      RPNDate(dateo).toDateTime() +
                                      npas* datetime.timedelta(seconds=deet)
                                    ).stamp
                            item['datev'] = datev
                        except ValueError:
                            # A ValueError here indicates that dateo wasn't
                            # a valid date to begin with.  This will happen
                            # if dateo is a made-up value like '0'.  In this
                            # case, it doesn't make sense to try to compute
                            # a datev
                            pass
                    result = RPNMeta()
                    result.update_by_dict(item)
                    result.fileref = self
                    mylist2.append(result)
                if len(mylist) > 0:
                    self.lastread = mylist[-1]
                return mylist2
            elif key.nxt == 1:               # get NEXT one thatmatches
                self.lastread = Fstdc.fstinf(self.iun, key.nom, key.type,
                              key.etiket, key.ip1, key.ip2, key.ip3,
                              key.datev, key.handle)
            else:                          # get FIRST one that matches
                # If key.handle > 0, then this might be a duplicate check;
                # if it is, then the key will be a full RPNMeta, which has
                # the necessary information for __getitem__ and we can return
                # the key without further processing.  If it is not such an
                # instance (aka, a bare RPNKeys), then the handle isn't
                # useful and we should ignore it.
                if isinstance(key, RPNMeta) and key.handle >= 0:
                    return key #TODO: may want to check if key.handle is valid
                self.lastread = Fstdc.fstinf(self.iun, key.nom, key.type,
                                             key.etiket, key.ip1, key.ip2,
                                             key.ip3, key.datev, -2)
        elif key == NextMatch:               # fstsui, return FstHandle instance
            self.lastread = Fstdc.fstinf(self.iun, ' ', ' ', ' ', 0, 0, 0, 0, -1)
        else:
            raise TypeError('RPNFile.info(), search keys arg is not of a valid type')
        result = RPNMeta()
        if self.lastread != None:
#            self.lastread.__dict__['fileref']=self
            if self.lastread['dateo'] != -1:
                dateo = self.lastread['dateo']
                npas = self.lastread['npas']
                deet = self.lastread['deet']
                try:
                    datev = RPNDate(
                              RPNDate(dateo).toDateTime() +
                              npas* datetime.timedelta(seconds=deet)
                            ).stamp
                    self.lastread['datev'] = datev
                except ValueError:
                    # A ValueError here indicates that dateo wasn't
                    # a valid date to begin with.  This will happen
                    # if dateo is a made-up value like '0'.  In this
                    # case, it doesn't make sense to try to compute
                    # a datev
                    pass

            result.update_by_dict(self.lastread)
            result.fileref = self
#            print 'DEBUG result=', result
        else:
            return None
        return result # return handle

    def erase(self, index):
        """Erase data and tags of rec in RPN STD file

        myRPNfile.erase(myRPNparms)
        myRPNfile.erase(myRPNrec)
        myRPNfile.erase(myRPNrec.handle)

        @param myRPNparms       values of rec parameters, must be a RPNMeta instance (or derived class)
        @param myRPNrec.handle  file handle of the rec to erase (int)
        @exception TypeError if myRPNrec.handle is not of accepted type
        @exception ValueError if invalid record handle is provided (or not found from myRPNparms)
        """
        if isinstance(index, RPNDesc): # set of keys
            meta = index
            if index.handle < 0:
                meta = self.info(index)
            target = meta.handle
        elif type(index) == type(0):  # handle
            target = index
        else:
            raise TypeError('RPNFile: index must provide a valid handle to erase a record')
        if meta.handle >= 0:
            #print 'erasing record with handle=', target, ' from file'
            self.lastwrite = Fstdc.fsteff(target)
        else:
            raise ValueError('RPNFile: invalid record handle')


    def __setitem__(self, index, value):
        """[re]write data and tags of rec in RPN STD file

        myRPNfile.info[myRPNparms] = mydataarray
        myRPNfile.info[myRPNrec]   = myRPNrec.d
        myRPNfile.info[myRPNrec]   = None #erase the record corresponding to myRPNrec.handle
        myRPNfile.info[myRPNrec.handle] = None #erase the record corresponding to handle

        @param myRPNparms  values of rec parameters, must be a RPNMeta instance (or derived class)
        @param mydataarray data to be written, must be numpy.ndarray instance
        @exception TypeError if args are of wrong type
        @exception TypeError if params.handle is not valid when erasing (value=None)
        @exception ValueError if not able to find a rec corresponding to metaparams when erasing (value=None)
        """
        if value is None:
            self.erase(index)
        elif isinstance(index, RPNMeta) and type(value) == numpy.ndarray:
            self.lastwrite = 0
            #print 'writing data', value.shape, ' to file, keys=', index
            #print 'dict = ', index.__dict__

            # Check to see if we're writing string data, with strings of greater
            # than one length.  If the strings have length 1, then we can presume
            # we're writing a character array and can proceed.
            if value.dtype.kind == 'S' and value.dtype.itemsize > 1:
                # Get the maximum number of characters per string in the existing array
                max_strlen = value.dtype.itemsize
                # If this is less than implied by the Meta's .ni value, expand these strings
                # into an enclosing array
                if max_strlen > index.ni:
                    value = numpy.array(value, dtype=(numpy.str_, index.ni))
                # Now, re-interperet these values as a multidimensional array of single
                # characters.

                # First, ensure that this array is in Fortran order.  Strings are indexed
                # along the first dimension, which varies fastest.  Later reshaping will
                # require Fortran order.

                # Additionally, make a copy of the array so that we can space-pad the
                # end of the strings.  This should enhance compatibility with Fortran
                # programs that happen to read these strings.

                value = value.copy(order='F')

                value1d = value.reshape([-1]) # This creates an alternate view of the memory
                for kk in range(0, len(value1d)):
                    value1d[kk] = value1d[kk] + ' '*(max_strlen-len(value1d[kk]))

                # Second, view this array as consisting of 1D characters.  This will
                # screw up the shape, to be fixed next.

                value1 = value.view(dtype=(numpy.str_, 1))

                # Finally, reshape the array
                value = value1.reshape((-1, ) + value.shape, order='F')

                print(value)

            if value.flags.farray:
                #print 'fstecr Fortran style array'
                # Check to see if the memory order is contiguous, else make
                # a contiguous, Fortran-ordered copy for writing.
                if value.flags.f_contiguous == False:
                    value = numpy.array(value, order='F')
                Fstdc.fstecr(value, self.iun, index.nom, index.type,
                                 index.etiket, index.ip1, index.ip2, index.ip3,
                                 index.dateo, index.grtyp, index.ig1,
                                 index.ig2, index.ig3, index.ig4, index.deet,
                                 index.npas, index.nbits, index.datyp)
            else:
                #print 'fstecr C style array'
                Fstdc.fstecr(numpy.reshape(numpy.transpose(value), value.shape),
                             self.iun, index.nom, index.type, index.etiket,
                             index.ip1, index.ip2, index.ip3, index.dateo,
                             index.grtyp, index.ig1, index.ig2, index.ig3,
                             index.ig4, index.deet, index.npas, index.nbits,
                             index.datyp)
        else:
           raise TypeError('RPNFile write: value must be an array and index must be RPNMeta or RPNRec')

    def write(self, data, meta=None, rewrite=False):
        """Write a RPNRec to the file

        myRPNRec.write(myRPNRec)
        myRPNRec.write(myArray, myRPNMeta)
        myRPNRec.write(myRPNRec, rewrite=false)
        myRPNRec.write(myArray, myRPNMeta, rewrite=true)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        meta2 = meta
        data2 = data
        if meta is None and isinstance(data, RPNRec):
            meta2 = data
            data2 = data.d
        elif not(isinstance(meta, RPNMeta) and type(data) == numpy.ndarray):
            raise TypeError('RPNFile write: value must be an array and index must be RPNMeta or RPNRec')
        if rewrite:
            try:
                self.erase(meta2)
            except:
                pass
        self.__setitem__(meta2, data2)

    def append(self, data, meta=None):
        """Append a RPNRec to the file, shortcut for write(..., rewrite=False)

        myRPNRec.append(myRPNRec)
        myRPNRec.append(myArray, myRPNMeta)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        self.write(data, meta, rewrite=False)

    def rewrite(self, data, meta=None):
        """Write a RPNRec to the file, rewrite if record handle is found and exists
        shortcut for write(..., rewrite=True)

        myRPNRec.rewrite(myRPNRec)
        myRPNRec.rewrite(myArray, myRPNMeta)

        @param myRPNRec an instance of RPNRec with data and meta/params to be written
        @param myArray an instance of numpy.ndarray
        @param myRPNMeta an instance of RPNMeta with meta/params to be written
        @exception TypeError if args are of wrong type
        """
        self.write(data, meta, rewrite=True)


class RPNKeys(RPNParm):
    """RPN standard file Primary descriptors class, used to search for a record.

    Descriptors are:
    {'nom':'    ', 'type':'  ', 'etiket':'            ', 'date':-1, 'ip1':-1, 'ip2':-1, 'ip3':-1, 'handle':-2, 'nxt':0, 'fileref':None}
    TODO: give examples of instanciation
    """
    def __init__(self, model=None, **args):
        RPNParm.__init__(self, model, self.allowedKeysVals(), args)

    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        return self.searchKeysVals().copy()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {
            'nom' : '    ',
            'type' : '  ',
            'etiket' : '            ',
            'datev' : -1,
            'ip1' : -1,
            'ip2' : -1,
            'ip3' : -1,
            'handle' : -2,
            'nxt' : 0,
            'fileref' : None
            }

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {
            'nom' : '    ',
            'type' : '  ',
            'etiket' : '            ',
            'datev' : 0,
            'ip1' : 0,
            'ip2' : 0,
            'ip3' : 0,
            'handle' : -2,
            'nxt' : 0,
            'fileref' : None
            }

class RPNDesc(RPNParm):
    """RPN standard file Auxiliary descriptors class, used when writing a record or getting descriptors from a record.

    Descriptors are:
    {'grtyp':'X', 'dateo':0, 'deet':0, 'npas':0, 'ig1':0, 'ig2':0, 'ig3':0, 'ig4':0, 'datyp':-1, 'nbits':0, 'xaxis':None, 'yaxis':None, 'xyref':(None, None, None, None, None), 'griddim':(None, None)}
    TODO: give examples of instanciation
    """
    def __init__(self, model=None, **args):
        RPNParm.__init__(self, model, self.allowedKeysVals(), args)

    def allowedKeysVals(self):
       """Return a dict of allowed Keys/Vals"""
       return  self.searchKeysVals().copy()

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        return {
            'grtyp':' ',
            'dateo':-1,
            'deet':-1,
            'npas':-1,
            'ig1':-1,
            'ig2':-1,
            'ig3':-1,
            'ig4':-1,
            'datyp':-1,
            'nbits':-1,
            'ni':-1,
            'nj':-1,
            'nk':-1
            }

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        return {
            'grtyp' : 'X',
            'dateo' : 0,
            'deet' : 0,
            'npas' : 0,
            'ig1' : 0,
            'ig2' : 0,
            'ig3' : 0,
            'ig4' : 0,
            'datyp' : -1,
            'nbits' : 16,
            'ni' : 1,
            'nj' : 1,
            'nk' : 1
            }


class RPNMeta(RPNKeys, RPNDesc):
    """RPN standard file Full set (Primary + Auxiliary) of descriptors class,
       needed to write a record, can be used for search.

    myRPNMeta = RPNMeta()
    myRPNMeta = RPNMeta(anRPNMeta)
    myRPNMeta = RPNMeta(anRPNMeta, nom='TT')
    myRPNMeta = RPNMeta(nom='TT')

    @param anRPNMeta another instance of RPNMeta to copy data from
    @param nom [other descriptors can be used, see below] comma separated metadata key=value pairs
    @exception TypeError if anRPNMeta is not an instance of RPNMeta

    Descriptors are:
        'nom':'    ',
        'type':'  ',
        'etiket':'            ',
        'ip1':-1, 'ip2':-1, 'ip3':-1,
        'ni':-1, 'nj':-1, 'nk':-1,
        'dateo':0,
        'deet':0,
        'npas':0,
        'grtyp':'X',
        'ig1':0, 'ig2':0, 'ig3':0, 'ig4':0,
        'datyp':-1,
        'nbits':0,
        'handle':-2,
        'nxt':0,
        'fileref':None,
        'datev':-1

    Examples of use (also doctests):

    >>> myRPNMeta = RPNMeta() #New RPNMeta with default/wildcard descriptors
    >>> d = [x for x in myRPNMeta.__dict__.items()]
    >>> d.sort()
    >>> d
    [('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', -1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', '    '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> myRPNMeta = RPNMeta(nom='GZ', ip2=1)  #New RPNMeta with all descriptors to wildcard but nom, ip2
    >>> d = [x for x in myRPNMeta.__dict__.items()]
    >>> d.sort()
    >>> d
    [('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', 1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', 'GZ  '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> myRPNMeta.ip1
    -1
    >>> myRPNMeta2 = myRPNMeta #shallow copy (reference)
    >>> myRPNMeta2.ip1 = 9 #this will also update myRPNMeta.ip1
    >>> myRPNMeta.ip1
    9
    >>> myRPNMeta2 = RPNMeta(myRPNMeta)   #make a deep-copy
    >>> myRPNMeta.ip3
    -1
    >>> myRPNMeta2.ip3 = 9 #this will not update myRPNMeta.ip3
    >>> myRPNMeta.ip3
    -1
    >>> myRPNMeta.nom = 'TT'
    >>> myRPNMeta2 = RPNMeta(myRPNMeta, nom='GZ', ip2=8)   #make a deep-copy and update nom, ip2 values
    >>> (myRPNMeta.nom, myRPNMeta.ip1, myRPNMeta.ip2)
    ('TT  ', 9, 1)
    >>> (myRPNMeta2.nom, myRPNMeta2.ip1, myRPNMeta2.ip2)
    ('GZ  ', 9, 8)

    TODO: test update() and update_cond()
    """
    def __init__(self, model=None, **args):
        RPNParm.__init__(self, model, self.allowedKeysVals(), args)
        if model != None:
            if isinstance(model, RPNParm):
                self.update(model)
            elif type(model) == type({}):
                self.update(model)
            else:
                raise TypeError('RPNMeta: cannot initialize from arg #1')
        for name in args.keys(): # and update with specified attributes
            setattr(self, name, args[name])

    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        a = RPNKeys.allowedKeysVals(self).copy()
        a.update(RPNDesc.allowedKeysVals(self).copy())
        return a

    def searchKeysVals(self):
        """Return a dict of search Keys/Vals"""
        a = RPNKeys.searchKeysVals(self)
        a.update(RPNDesc.searchKeysVals(self))
        return a

    def defaultKeysVals(self):
        """Return a dict of sensible default Keys/Vals"""
        a = RPNKeys.defaultKeysVals(self)
        a.update(RPNDesc.defaultKeysVals(self))
        return a

    def getaxis(self, axis=None):
        """Return the grid axis rec of grtyp ('Z', 'Y', '#')

        (myRPNRecX, myRPNRecY) = myRPNMeta.getaxis()
        myRPNRecX = myRPNMeta.getaxis('X')
        myRPNRecY = myRPNMeta.getaxis(axis='Y')

        @param axis which axis to return (X, Y or None), default=None returns both axis
        @exception TypeError if RPNMeta.grtyp is not in ('Z', 'Y', '#')
        @exception TypeError if RPNMeta.fileref is not an RPNFile
        @exception ValueError if grid descriptors records (>>, ^^) are not found in RPNMeta.fileref
        """
        if not self.grtyp in ('Z', 'Y', '#'):
            raise ValueError('getaxis error: can not get axis from grtyp=' +
                             self.grtyp)
        if not isinstance(self.fileref, RPNFile):
            raise TypeError('RPNMeta.getaxis: ERROR - cannot get axis, no fileRef')
        searchkeys = RPNKeys(ip1=self.ig1, ip2=self.ig2)
        if self.grtyp != '#':
            searchkeys.update_by_dict({'ip3':self.ig3})
        searchkeys.nom = '>>'
        xaxisrec = self.fileref[searchkeys]
        searchkeys.nom = '^^'
        yaxisrec = self.fileref[searchkeys]
        if xaxisrec is None or yaxisrec is None: # csubich -- variable name typo
            raise ValueError('RPNMeta.getaxis: ERROR - axis grid descriptors (>>, ^^) not found')
        if type(axis) == type(' '):
            if axis.upper() == 'X':
                return xaxisrec
            elif axis.upper() == 'Y':
                return yaxisrec
                #axisdata=self.yaxis.ravel()
        return (xaxisrec, yaxisrec)

    def __repr__(self):
        return 'RPNMeta'+repr(self.__dict__)


class RPNGrid(RPNParm):
    """RPNSTD-type grid description

    >>> g = RPNGrid(grtyp='N', ig14=(1, 2, 3, 4), shape=(4, 1))
    >>> (g.grtyp, g.shape, g.ig14)
    ('N', (4, 1), (1, 2, 3, 4))

    myRPNGrid = RPNGrid(grtyp='Z', xyaxis=(myRPNRecX, myRPNRecY))
    myRPNGrid = RPNGrid(grtyp='#', shape=(200, 150), xyaxis=(myRPNRecX, myRPNRecY))
    myRPNGrid = RPNGrid(myRPNRec)
    myRPNGrid = RPNGrid(keys=myRPNRec)
    myRPNGrid = RPNGrid(myRPNRec, xyaxis=(myRPNRecX, myRPNRecY))

    @param keys
    @param grtyp
    @param shape
    @param ig14
    @param xyaxis
    @exception ValueError
    @exception TypeError

    >>> g = RPNGrid(grtyp='N', ig14=(1, 2, 3, 4), shape=(200, 150))
    >>> (g.grtyp, g.shape, g.ig14)
    ('N', (200, 150), (1, 2, 3, 4))
    >>> g2 = RPNGrid(g)
    >>> (g2.grtyp, g2.shape, g2.ig14)
    ('N', (200, 150), (1, 2, 3, 4))
    >>> d = RPNMeta(grtyp='N', ig1=1, ig2=2, ig3=3, ig4=4, ni=200, nj=150)
    >>> g3 = RPNGrid(d)
    >>> (g3.grtyp, g3.shape, g3.ig14)
    ('N', (200, 150), (1, 2, 3, 4))

    Icosahedral Grid prototype:
    grtyp = I
    ig1 = griddiv
    ig2 = grid tile (1-10) 2d, 0=NP, SP, allpoints (1D vect)
    ig3, ig4

    #(I) would work much the same way as #(L) ?

    """
    base_grtyp = ('A', 'B', 'E', 'G', 'L', 'N', 'S')
    ref_grtyp  = ('Z', 'Y', '#')
    helper     = None

    def allowedKeysVals(self):
        """
        """
        a = {
            'grtyp': ' ',
            'ig14':(0, 0, 0, 0),
            'shape':(0, 0)
        }
        try:
            if self.helper:
                a.update(self.helper.addAllowedKeysVals.copy())
        except:
            pass
        return a

    def getGrtyp(self, keys, args=None):
        """Return grid type from class init args"""
        grtyp = ''
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError('RPNGrid: args should be of type dict')
        if keys:
            if 'grtyp' in args.keys():
                raise ValueError('RPNGrid: cannot provide both keys and grtyp: '+repr(args))
            elif isinstance(keys, RPNMeta) or isinstance(keys, RPNGrid):
                grtyp = keys.grtyp
            elif type(keys) == type({}):
                if 'grtyp' in keys.keys():
                    grtyp = keys['grtyp']
                else:
                    raise ValueError('RPNGrid: not able to find grtyp')
            else:
                raise TypeError('RPNGrid: wrong type for keys')
        else:
            if 'grtyp' in args.keys():
                grtyp = args['grtyp']
            else:
                raise ValueError('RPNGrid: not able to find grtyp')
        return grtyp

    def parseArgs(self, keys, args=None):
        """Return a dict with parsed args for the specified grid type"""
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError('RPNGrid: args should be of type dict')
        kv = self.allowedKeysVals().copy()
        if keys:
            if 'grtyp' in args.keys():
                raise ValueError('RPNGrid: cannot provide both keys and grtyp: '+repr(args))
            elif isinstance(keys, RPNMeta):
                kv['grtyp'] = keys.grtyp
                kv['ig14']  = (keys.ig1, keys.ig2, keys.ig3, keys.ig4)
                kv['shape'] = (keys.ni, keys.nj)
                if 'grid' in keys.__dict__.keys() and \
                        isinstance(keys.grid, RPNRec):
                    kv.update(keys.grid.__dict__)
            elif isinstance(keys, RPNGrid):
                kv.update(keys.__dict__)
        kv.update(args)
        if self.helper:
            kv.update(self.helper.parseArgs(keys, args))
        self.argsCheck(kv)
        return kv

    def argsCheck(self, d):
        """Check Grid params, raise an error if not ok"""
        grtyp = d['grtyp']
        shape = d['shape']
        ig14  = d['ig14']
        if not (type(grtyp) == type('')
            and type(shape) in (type([]), type(())) and len(shape)==2
            and (type(shape[0]) == type(shape[1]) == type(0))
            and type(ig14) in (type([]), type(())) and len(ig14)==4 \
            and (type(ig14[0]) == type(ig14[1]) == type(ig14[2]) == type(ig14[3]) == type(0)) ):
            raise ValueError('RPNGrid: invalid arg value')
        if self.helper:
            self.helper.argsCheck(d)

    def __init__(self, keys=None, **args):
        grtyp = self.getGrtyp(keys, args)
        if type(grtyp) == type(' '):
            grtyp = grtyp.upper()
        className = "RPNGrid{0}".format(grtyp)
        if grtyp in self.base_grtyp:
            className = "RPNGridBase"
        elif grtyp in self.ref_grtyp:
            className = "RPNGridRef"
        try:
            myClass = globals()[className]
            self.__dict__['helper'] = myClass()
        except:
            pass
        if not self.helper:
            raise ValueError('RPNGrid: unrecognize grtyp '+repr(grtyp))
        RPNParm.__init__(self, None, self.allowedKeysVals().copy(), {})
        self.update(self.parseArgs(keys, args))

    def interpol(self, fromData, fromGrid=None):
        """Interpolate (scalar) some gridded data to grid

        destData = myDestGrid.interpol(fromData, fromGrid)
        destRec  = myDestGrid.interpol(fromRec)

        Short for of calling:
        destData = myDestGrid.interpolVect(fromData, None, fromGrid)
        destRec  = myDestGrid.interpolVect(fromRec)

        See RPNGrid.interpolVect() methode for documentation
        """
        return self.interpolVect(fromData, None, fromGrid)

    #@static
    def interpolVectValidateArgs(self, fromDataX, fromDataY=None, fromGrid=None):
        """Check args for InterpolVect
        """
        recx  = None
        recy  = None
        isRec = False
        if (type(fromDataX) == numpy.ndarray
            and isinstance(fromGrid, RPNGrid)):
            recx   = RPNRec()
            recx.d = fromDataX
            try:
                recx.setGrid(fromGrid)
            except:
                raise ValueError('RPNGrid.interpolVect: fromGrid incompatible with fromDataX')
            if type(fromDataY) == numpy.ndarray:
                recy = RPNRec()
                recy.d = fromDataY
                try:
                    recy.setGrid(fromGrid)
                except:
                    raise ValueError('RPNGrid.interpolVect: fromGrid incompatible with fromDataY')
            elif fromDataY:
                raise TypeError('RPNGrid.interpolVect: fromDataY should be of same type as fromDataX')
        elif isinstance(fromDataX, RPNRec):
            isRec = True
            if fromGrid:
                raise TypeError('RPNGrid.interpolVect: cannot provide both an RPNRec for fromDataX and fromGrid')
            recx = fromDataX
            recx.setGrid()
            if isinstance(fromDataY, RPNRec):
                recy = fromDataY
                recy.setGrid()
                if recx.grid != recy.grid:
                    raise ValueError('RPNGrid.interpolVect: data X and Y should be on the same grid')
            elif fromDataY:
                raise TypeError('RPNGrid.interpolVect: fromDataY should be of same type as fromDataX')
        else:
            raise TypeError('RPNGrid.interpolVect: data should be of numpy.ndarray or RPNRec type')
        isVect = (recy != None)
        return (recx, recy, isVect, isRec)

    def getEzInterpArgs(self, isSrc):
        """Return the list of needed args for Fstdc.ezinterp (use helper)"""
        #TODO: may want to check helper, helper.getEzInterpArgs
        return self.helper.getEzInterpArgs(self.__dict__, isSrc)

##     def toScripGridPreComp(self, name=None):
##         """Return a Scrip grid instance for Precomputed addr&weights (use helper)"""
##         return self.helper.toScripGridPreComp(self.__dict__, name)

##     def toScripGrid(self, name=None):
##         """Return a Scrip grid instance (use helper)"""
##         if not('scripGrid' in self.__dict__.keys() and self.scripGrid):
##             self.__dict__['scripGrid'] = self.helper.toScripGrid(self.__dict__, name)
##         return self.scripGrid

##     def reshapeDataForScrip(self, data):
##         """Return reformated data suitable for SCRIP (use helper)"""
##         return self.helper.reshapeDataForScrip(self.__dict__, data)

##     def reshapeDataFromScrip(self, data):
##         """Inverse operation of reshapeDataForScrip (use helper)"""
##         return self.helper.reshapeDataFromScrip(self.__dict__, data)

    def getLatLon(self, xin=None, yin=None):
        """ Get the latitude and longitude coordinates of the given (x, y) pairs;
            If no pairs are specified, then return the latitude and longitude
            coordinates for every grid point."""

        # Step zero: if called with (None, None), build coordinate arrays
        if xin is None and yin is None:
            xin = numpy.zeros(self.shape, dtype=numpy.float32, order='F');
            yin = numpy.zeros(self.shape, dtype=numpy.float32, order='F');
            xin[:, :] = numpy.array(range(1, self.shape[0]+1)).reshape((self.shape[0], -1))
            yin[:, :] = numpy.array(range(1, self.shape[1]+1)).reshape((-1, self.shape[1]))

        # Step one: check to see if xin and yin are the right kind of arrays
        # we need: numpy arrays of dtype float32 that have contiguous allocation.
        # In order to be liberal about what we'll accept, we can do this as a
        # try/catch block, with 'catch' making a numpy array from scratch.
        try:
            # An explicit check for numpy.ndarray is necessary, because the
            # scalar types implement dtype and flags members that will
            # fool the following lines; that causes a segfault in C-code that
            # expects a legitimate array.
            assert(isinstance(xin, numpy.ndarray))
            assert(xin.dtype == numpy.float32)
            assert(xin.flags.c_contiguous or xin.flags.f_contiguous)
        except (AssertionError):
            # One of those conditions was not true, so copy the input
            # data to a numpy array.  Any further incompatibility,
            # such as for the datatype, will raise an exception that
            # propagates up.  For best compatibility with other routines,
            # we'll also default to Fortran-ordering.
            xin = numpy.array(xin, dtype=numpy.float32, order='F')

        # Step two: check whether the ordering of xin and yin match:
        try:
            assert(isinstance(yin, numpy.ndarray))
            assert(yin.dtype == numpy.float32)
            assert(xin.flags.c_contiguous == yin.flags.c_contiguous)
            assert(xin.flags.f_contiguous == yin.flags.f_contiguous)
        except (AssertionError):
            # Perform the same kind of allocation for yin as we might have
            # to do for xin.  Since we want the ordering to match xin but
            # don't care whether it's C or Fortran-ordering, we can use
            # Python's equivalent of the ternary operator.
            yin = numpy.array(yin, dtype=numpy.float32,
                            order=(xin.flags.c_contiguous and 'C' or 'F'))

        if (xin.shape != yin.shape):
            raise ValueError('Input arays xin and yin must have the same shape')

        # Get interpolation arguments
        ezia = self.getEzInterpArgs(False)

        # Call gdllfxy
        return Fstdc.gdllfxy(xin, yin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0'])

    def getXY(self, latin=None, lonin=None):
        """ Get the (x, y) coordinates corresponding to every given (lat, lon)
            input pair; this may be off the grid for coordinates that are
            themselves not on the grid.  If no input is provided, return the
            (x, y) arrays corresponding to every grid point."""
        # Step zero: if called with (None, None), build coordinate arrays
        if (latin is None and lonin is None):
            xin = numpy.zeros(self.shape, dtype=numpy.float32, order='F');
            yin = numpy.zeros(self.shape, dtype=numpy.float32, order='F');
            xin[:, :] = numpy.array(range(1, self.shape[0]+1)).reshape((self.shape[0], -1))
            yin[:, :] = numpy.array(range(1, self.shape[1]+1)).reshape((-1, self.shape[1]))
            return (xin, yin)

        # Step one: Perform the same set of input checks as getLatLon
        try:
            assert(isinstance(latin, numpy.ndarray))
            assert(latin.dtype == numpy.float32)
            assert(latin.flags.c_contiguous or latin.flags.f_contiguous)
        except (AssertionError):
            latin = numpy.array(latin, dtype=numpy.float32, order='F')

        # Step two: check whether the ordering of latin and lonin match:
        try:
            assert(isinstance(lonin, numpy.ndarray))
            assert(lonin.dtype == numpy.float32)
            assert(latin.flags.c_contiguous == lonin.flags.c_contiguous)
            assert(latin.flags.f_contiguous == lonin.flags.f_contiguous)
        except (AssertionError):
            lonin = numpy.array(lonin, dtype=numpy.float32,
                            order=(latin.flags.c_contiguous and 'C' or 'F'))

        if latin.shape != lonin.shape:
            raise ValueError('Input arays latin and lonin must have the same shape')

        # Get interpolation arguments
        ezia = self.getEzInterpArgs(False)

        # Call gdxyfll
        print('calling gdxyfll')
        return Fstdc.gdxyfll(latin, lonin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0'])

    def getWindSpdDir(self, uu, vv, latin=None, lonin=None):
        """ Convert the provided (UU, VV) wind pairs to (UV, WD) speed/direction
            pairs.  (UU, VV) must be:
                1) At provided (latin, lonin) coordinates, or
                2) Over the full grid (ni x nj) extent"""

        if latin is None and lonin is None: # Full grid conversion
            # Verify that uu and vv are full-grid numpy arrays
            try:
                assert(isinstance(uu, numpy.ndarray))
                assert(uu.dtype == numpy.float32)
                assert(uu.flags.f_contiguous == True)
            except AssertionError:
                uu = numpy.array(uu, dtype=numpy.float32, order='F')

            try:
                assert(isinstance(vv, numpy.ndarray))
                assert(vv.dtype == numpy.float32)
                assert(vv.flags.f_contiguous == True)
            except AssertionError:
                vv = numpy.array(vv, dtype=numpy.float32, order='F')

            # The array shapes must match this grid
            if uu.shape != self.shape or vv.shape != self.shape:
                raise ValueError('Full-grid winds UU and VV must match this '+
                                 'grid\'s shape ({0}, {1})'.
                                 format(self.shape[0], self.shape[1]))

            # Since we'll still be using a scattered-coordinate function to perform the conversion,
            # grab the lat/lon coordinates from getLatLon()
            (latin, lonin) = self.getLatLon()
        else:
            # Check inputs for consistency.  This is not necessary for full-grid
            # converstion because it was handled in the block above.
            try:
                assert(isinstance(uu, numpy.ndarray))
                assert(uu.dtype == numpy.float32)
                assert(uu.flags.f_contiguous or uu.flags.c_contiguous)
            except AssertionError:
                uu = numpy.array(uu, dtype=numpy.float32, order='F')
            orderchar = uu.flags.f_contiguous and 'F' or 'C' # Keep this around for other arrays

            try:
                assert(isinstance(vv, numpy.ndarray))
                assert(vv.dtype == numpy.float32)
                assert(vv.flags.f_contiguous == uu.flags.f_contiguous)
                assert(vv.flags.c_contiguous == uu.flags.c_contiguous)
            except AssertionError:
                vv = numpy.array(vv, dtype=numpy.float32, order=orderchar)
            if uu.shape != vv.shape:
                raise ValueError('Wind fields UU and VV must have the same shape')

            try:
                assert(isinstance(latin, numpy.ndarray))
                assert(latin.dtype == numpy.float32)
                assert(latin.flags.f_contiguous == uu.flags.f_contiguous)
                assert(latin.flags.c_contiguous == uu.flags.c_contiguous)
            except AssertionError:
                latin = numpy.array(latin, dtype=numpy.float32, order=orderchar)
            if latin.shape != uu.shape:
                raise ValueError('Coordinate array shape must match that of the wind fields')
            try:
                assert(isinstance(lonin, numpy.ndarray))
                assert(lonin.dtype == numpy.float32)
                assert(lonin.flags.f_contiguous == uu.flags.f_contiguous)
                assert(lonin.flags.c_contiguous == uu.flags.c_contiguous)
            except AssertionError:
                lonin = numpy.array(lonin, dtype=numpy.float32, order=orderchar)
            if lonin.shape != uu.shape:
                raise ValueError('Coordinate array shape must match that of the wind fields')

        # Now we have UU, VV, and coordinate arrays that match.  Grab the grid
        # interpolation parameters
        ezia = self.getEzInterpArgs(False)

        return(Fstdc.gdwdfuv(uu, vv, latin, lonin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0']))

    def getWindGrid(self, uv, wd, latin=None, lonin=None, xin=None, yin=None):
        """ Convert the provided (UV, WD) wind (speed/direction) pairs to grid-
            directed (UU, VV) form.  The provided (UV, WD) values must be:
                1) At given (latin, lonin) coordinates, or
                2) Over the full (ni x nj) grid extent"""
        if latin is None and lonin is None: # Full grid conversion
            try:
                assert(isinstance(uv, numpy.ndarray))
                assert(uv.dtype == numpy.float32)
                assert(uv.flags.f_contiguous == True)
            except AssertionError:
                uv = numpy.array(uv, dtype=numpy.float32, order='F')

            try:
                assert(isinstance(wd, numpy.ndarray))
                assert(wd.dtype == numpy.float32)
                assert(wd.flags.f_contiguous == True)
            except AssertionError:
                wd = numpy.array(wd, dtype=numpy.float32, order='F')

            if uv.shape != self.shape or wd.shape != self.shape:
                raise ValueError('Full-grid winds UV and WD must match this '+
                                 'grid\'s shape ({0}, {1})'.
                                 format(self.shape[0], self.shape[1]))

            (latin, lonin) = self.getLatLon()
        else:
            # Check inputs for consistency.  This is not necessary for full-grid
            # converstion because it was handled in the block above.
            try:
                assert(isinstance(uv, numpy.ndarray))
                assert(uv.dtype == numpy.float32)
                assert(uv.flags.f_contiguous or uv.flags.c_contiguous)
            except AssertionError:
                uv = numpy.array(uv, dtype=numpy.float32, order='F')
            orderchar = uv.flags.f_contiguous and 'F' or 'C' # Keep this around for other arrays

            try:
                assert(isinstance(wd, numpy.ndarray))
                assert(wd.dtype == numpy.float32)
                assert(wd.flags.f_contiguous == uv.flags.f_contiguous)
                assert(wd.flags.c_contiguous == uv.flags.c_contiguous)
            except AssertionError:
                wd = numpy.array(wd, dtype=numpy.float32, order=orderchar)
            if uv.shape != wd.shape:
                raise ValueError('Wind fields UU and VV must have the same shape')

            try:
                assert(isinstance(latin, numpy.ndarray))
                assert(latin.dtype == numpy.float32)
                assert(latin.flags.f_contiguous == uv.flags.f_contiguous)
                assert(latin.flags.c_contiguous == uv.flags.c_contiguous)
            except AssertionError:
                latin = numpy.array(latin, dtype=numpy.float32, order=orderchar)
            if latin.shape != uv.shape:
                raise ValueError('Coordinate array shape must match that of the wind fields')
            try:
                assert(isinstance(lonin, numpy.ndarray))
                assert(lonin.dtype == numpy.float32)
                assert(lonin.flags.f_contiguous == uv.flags.f_contiguous)
                assert(lonin.flags.c_contiguous == uv.flags.c_contiguous)
            except AssertionError:
                lonin = numpy.array(lonin, dtype=numpy.float32, order=orderchar)
            if lonin.shape != uv.shape:
                raise ValueError('Coordinate array shape must match that of the wind fields')

        ezia = self.getEzInterpArgs(False)

        return (Fstdc.gduvfwd(uv, wd, latin, lonin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0']))

    def interpolLatLon(self, latin, lonin, zz, vv=None):
        """ Interpolate the given 'zz' field to scattered (lat, lon) coordinates;
            if 'vv' is provided and not None, then vector interpolation (as
            grid-directed winds) will be performed."""
        # As per the previous functions, we have some array-verification to
        # perform.  This time, latin and lonin must be compatible with
        # each other (and contiguous), and zz/vv (if present) must match
        # the grid.

        # zz and vv have the strictest conditions, so start with those
        try:
            assert(isinstance(zz, numpy.ndarray))
            assert(zz.dtype == numpy.float32)
            assert(zz.flags.f_contiguous == True)
        except AssertionError:
            zz = numpy.array(zz, dtype=numpy.float32, order='F')

        if zz.shape != self.shape:
            raise ValueError('Full-grid field ZZ must match this '+
                             'grid\'s shape ({0}, {1})'.
                             format(self.shape[0], self.shape[1]))

        if vv is not None:
            try:
                assert(isinstance(vv, numpy.ndarray))
                assert(vv.dtype == numpy.float32)
                assert(vv.flags.f_contiguous == True)
            except AssertionError:
                vv = numpy.array(vv, dtype=numpy.float32, order='F')
            if vv.shape != self.shape:
                raise ValueError('Full-grid field VV must match this ' +
                                 'grid\'s shape ({0}, {1})'.
                                 format(self.shape[0], self.shape[1]))

        # Now for latin and lonin -- these are the same checks as getXY
        try:
            assert(isinstance(latin, numpy.ndarray))
            assert(latin.dtype == numpy.float32)
            assert(latin.flags.c_contiguous or latin.flags.f_contiguous)
        except AssertionError:
            latin = numpy.array(latin, dtype=numpy.float32, order='F')
        try:
            assert(isinstance(lonin, numpy.ndarray))
            assert(lonin.dtype == numpy.float32)
            assert(latin.flags.c_contiguous == lonin.flags.c_contiguous)
            assert(latin.flags.f_contiguous == lonin.flags.f_contiguous)
        except AssertionError:
            lonin = numpy.array(lonin, dtype=numpy.float32,
                            order=(latin.flags.c_contiguous and 'C' or 'F'))

        # Now, call the interpolation routine.  That routine itself checks for scalar or vector
        # interpolation, so we're free to pass vv=None if necessary.

        ezia = self.getEzInterpArgs(False)

        return (Fstdc.gdllval(zz, vv, latin, lonin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0']))

    def interpolXY(self, xin, yin, zz, vv=None):
        """ Interpolate the given 'zz' field to scattered (x, y) coordinates;
            if 'vv' is provided and not None, then vector interpolation (as
            grid-directed winds) will be performed."""
        # The structure of this method follows exactly from interpolLatLon,
        # starting with input verification

        # zz and vv have the strictest conditions, so start with those
        try:
            assert(isinstance(zz, numpy.ndarray))
            assert(zz.dtype == numpy.float32)
            assert(zz.flags.f_contiguous == True)
        except AssertionError:
            zz = numpy.array(zz, dtype=numpy.float32, order='F')

        if zz.shape != self.shape:
            raise ValueError('Full-grid field ZZ must match this ' +
                             'grid\'s shape ({0}, {1})'.
                             format(self.shape[0], self.shape[1]))

        if vv is not None:
            try:
                assert(isinstance(vv, numpy.ndarray))
                assert(vv.dtype == numpy.float32)
                assert(vv.flags.f_contiguous == True)
            except AssertionError:
                vv = numpy.array(vv, dtype=numpy.float32, order='F')
            if vv.shape != self.shape:
                raise ValueError('Full-grid field VV must match this ' +
                                 'grid\'s shape ({0}, {1})'.
                                 format(self.shape[0], self.shape[1]))

        try:
            assert(isinstance(xin, numpy.ndarray))
            assert(xin.dtype == numpy.float32)
            assert(xin.flags.c_contiguous or xin.flags.f_contiguous)
        except AssertionError:
            xin = numpy.array(xin, dtype=numpy.float32, order='F')
        try:
            assert(isinstance(yin, numpy.ndarray))
            assert(yin.dtype == numpy.float32)
            assert(xin.flags.c_contiguous == yin.flags.c_contiguous)
            assert(xin.flags.f_contiguous == yin.flags.f_contiguous)
        except AssertionError:
            yin = numpy.array(yin, dtype=numpy.float32,
                            order=(xin.flags.c_contiguous and 'C' or 'F'))

        # Now, call the interpolation routine.  That routine itself checks for scalar or vector
        # interpolation, so we're free to pass vv=None if necessary.

        ezia = self.getEzInterpArgs(False)

        return (Fstdc.gdxyval(zz, vv, xin, yin,
            ezia['shape'], ezia['grtyp'], ezia['g_ig14'],
            ezia['xy_ref'], ezia['hasRef'], ezia['ij0']))


    def interpolVect(self, fromDataX, fromDataY=None, fromGrid=None):
        """Interpolate some gridded scalar/vectorial data to grid
        """
        (recx, recy, isVect, isRec) = \
            self.interpolVectValidateArgs(fromDataX, fromDataY, fromGrid)
        recyd = None
        if isVect:
            recyd = recy.d
        (sg, dg) = (recx.grid, self)
        dataxy = (None, None)
        sg_a = sg.getEzInterpArgs(isSrc=True)
        dg_a = dg.getEzInterpArgs(isSrc=False)
        if sg_a and dg_a:
            a = RPNGridHelper.baseEzInterpArgs.copy()
            a.update(sg_a)
            sg_a = a
            a = RPNGridHelper.baseEzInterpArgs.copy()
            a.update(dg_a)
            dg_a = a
            dataxy = None
            dataxy = Fstdc.ezinterp(recx.d, recyd,
                    sg_a['shape'], sg_a['grtyp'], sg_a['g_ig14'],
                    sg_a['xy_ref'], sg_a['hasRef'], sg_a['ij0'],
                    dg_a['shape'], dg_a['grtyp'], dg_a['g_ig14'],
                    dg_a['xy_ref'], dg_a['hasRef'], dg_a['ij0'],
                    isVect)
            if isVect == 0:
                dataxy = (dataxy, None)
        else:
            raise TypeError('RPNGrid.interpolVect: Cannot perform interpolation between specified grids type')
##             #if verbose: print "using SCRIP"
##             if isVect!=1:
##                 #TODO: remove this exception when SCRIP.interp() does vectorial interp
##                 raise TypeError('RPNGrid.interpolVect: SCRIP.interp() Cannot perform vectorial interpolation yet!')
##             #Try to interpolate with previously computed addr&weights (if any)
##             sg_a = sg.toScripGridPreComp()
##             dg_a = dg.toScripGridPreComp()
##             try:
##                 scripObj = scrip.Scrip(sg_a, dg_a)
##             except:
##                 #Try while computing lat/lon and addr&weights
##                 sg_a = sg.toScripGrid()
##                 dg_a = dg.toScripGrid()
##             if sg_a and dg_a:
##                 scripObj = scrip.Scrip(sg_a, dg_a)
##                 datax  = sg.reshapeDataForScrip(recx.d)
##                 datax2 = scrip.scripObj.interp(datax)
##                 dataxy = (dg.reshapeDataFromScrip(datax2), None)
##             else:
##                 raise TypeError('RPNGrid.interpolVect: Cannot perform interpolation between specified grids type')
        if isRec:
            recx.d = dataxy[0]
            recx.setGrid(self)
            if isVect:
                recy.d = dataxy[1]
                recy.setGrid(self)
                return (recx, recy)
            else:
                return recx
        else:
            if isVect:
                return (dataxy[0], dataxy[1])
            else:
                return dataxy[0]

    def __repr__(self):
        return 'RPNGrid'+repr(self.__dict__)


class RPNGridBase(RPNGridHelper):
    """RPNGrid Helper class for RPNSTD-type grid description for basic projections
    """
    def parseArgs(self, keys, args):
        """Return a dict with parsed args for the specified grid type"""
        return {} #TODO: accept real values (xg14)

    #@static
    def argsCheck(self, d):
        """Check Grid params, raise an error if not ok"""
        if not d['grtyp'] in RPNGrid.base_grtyp:
            raise ValueError('RPNGridBase: invalid grtyp value:' +
                             repr(d['grtyp']))

    def getEzInterpArgs(self, keyVals, isSrc):
        """Return the list of needed args for Fstdc.ezinterp from the provided params"""
        a = RPNGridHelper.baseEzInterpArgs.copy()
        a['shape']  = keyVals['shape']
        a['grtyp']  = keyVals['grtyp']
        a['g_ig14'] = list(keyVals['ig14'])
        a['g_ig14'].insert(0, keyVals['grtyp'])
        return a

##     def toScripGrid(self, keyVals, name=None):
##         """Return a Scrip grid instance for the specified grid type"""
##         sg_a = self.getEzInterpArgs(keyVals, False)
##         doCorners = 1
##         (la, lo, cla, clo) = Fstdc.ezgetlalo(sg_a['shape'], sg_a['grtyp'], sg_a['g_ig14'], sg_a['xy_ref'], sg_a['hasRef'], sg_a['ij0'], doCorners)
##         if name is None:
##             name = self.toScripGridName(keyVals)
##         la  *= (numpy.pi/180.)
##         lo  *= (numpy.pi/180.)
##         cla *= (numpy.pi/180.)
##         clo *= (numpy.pi/180.)
##         return scrip.ScripGrid(name, (la, lo, cla, clo))


class RPNGridRef(RPNGridHelper):
    """RPNGrid Helper class for RPNSTD-type grid description for grid reference.

    Preferably use the generic RPNGrid class to indirectly get an instance
    """
    addAllowedKeysVals = {
        'xyaxis':(None, None),
        'g_ref':None
    }

    def parseArgs(self, keys, args):
        """Return a dict with parsed args for the specified grid type"""
        kv = {}
        if args is None:
            args = {}
        if type(args) != type({}):
            raise TypeError('RPNGridRef: args should be of type dict')
        #if keys is another grid or FstRec w/ grid
        #   or xyaxis was present in keys, args
        #   then it would already have been included
        #   only need to get xyaxis and g_ref from keys
        #   TODO: may not want to do this if they were provided in args or keys.grid
        if keys and isinstance(keys, RPNMeta):
            kv['xyaxis'] = keys.getaxis() # csubich -- proper spelling is without capital
            kv['g_ref']  = RPNGrid(kv['xyaxis'][0])
        for k in self.addAllowedKeysVals:
            if k in args.keys():
                kv[k] = args[k]
        return kv

    def argsCheck(self, d):
        """Check Grid params, raise an error if not ok"""
        xyaxis = d['xyaxis']
        if not (d['grtyp'] in RPNGrid.ref_grtyp
            and type(xyaxis) in (type([]), type(()))
            and len(xyaxis) == 2
            and isinstance(xyaxis[0], RPNRec)
            and isinstance(xyaxis[1], RPNRec)):
            raise ValueError('RPNGridRef: invalid value')

    def getEzInterpArgs(self, keyVals, isSrc):
        """Return the list of needed args for Fstdc.ezinterp from the provided params"""
        if keyVals['grtyp'] == 'Y' and isSrc:
            return None
        a = keyVals['g_ref'].getEzInterpArgs(isSrc)
        if a is None:
            return None
        a['grtyp']  = keyVals['grtyp']
        a['xy_ref'] = (keyVals['xyaxis'][0].d, keyVals['xyaxis'][1].d)
        a['hasRef'] = 1
        a['ij0'] = (1, 1)
        a['shape'] = keyVals['shape']
        if keyVals['grtyp'] == '#':
            a['ij0'] = keyVals['ig14'][2:]
        return a

##     def toScripGridName(self, keyVals):
##         """Return a hopefully unique grid name for the provided params"""
##         ij0 = (1, 1)
##         if keyVals['grtyp'] == '#':
##             ij0 = keyVals['ig14'][2:]
##         name = "grd%s%s-%i-%i-%i-%i-%i-%i-%i-%i" % (
##         keyVals['grtyp'], keyVals['g_ref'].grtyp,
##         keyVals['g_ref'].ig14[0], keyVals['g_ref'].ig14[1],
##         keyVals['g_ref'].ig14[2], keyVals['g_ref'].ig14[3],
##         keyVals['shape'][0], keyVals['shape'][1],
##         ij0[0], ij0[1])
##         return name

##     def toScripGrid(self, keyVals, name=None):
##         """Return a Scrip grid instance for the specified grid type"""
##         if name is None:
##             name = self.toScripGridName(keyVals)
##         sg_a = self.getEzInterpArgs(keyVals, False)
##         if sg_a is None:
##             #TODO: make this work for # grids and other not global JIM grids
##             scripGrid = keyVals['g_ref'].toScripGrid(keyVals['g_ref'].__dict__, name)
##         else:
##             doCorners = 1
##             (la, lo, cla, clo) = Fstdc.ezgetlalo(sg_a['shape'], sg_a['grtyp'], sg_a['g_ig14'], sg_a['xy_ref'], sg_a['hasRef'], sg_a['ij0'], doCorners)
##             la  *= (numpy.pi/180.)
##             lo  *= (numpy.pi/180.)
##             cla *= (numpy.pi/180.)
##             clo *= (numpy.pi/180.)
##             scripGrid = scrip.ScripGrid(name, (la, lo, cla, clo))
##         return scripGrid


class RPNRec(RPNMeta):
    """Standard file record, with data (ndarray class) and full set of descriptors (RPNMeta class).

    Examples
    --------
    >>> r = RPNRec()
    >>> r.d
    array([], dtype=float64)
    >>> r = RPNRec([1, 2, 3, 4])
    >>> r.d
    array([1, 2, 3, 4])
    >>> a = numpy.array([1, 2, 3, 4], order='F', dtype='float32')
    >>> r = RPNRec(a)
    >>> r.d
    array([ 1.,  2.,  3.,  4.], dtype=float32)
    >>> a[1] = 5
    >>> r.d #r.d is a reference to a, thus changing a changes a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r = RPNRec(a.copy(), RPNMeta(grtyp='X'))
    >>> r.d
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> a[1] = 9
    >>> r.d #r.d is a copy of a, thus changing a does not change a
    array([ 1.,  5.,  3.,  4.], dtype=float32)
    >>> r.grtyp
    'X'
    >>> r = RPNRec([1, 2, 3, 4])
    >>> r2 = RPNRec(r)
    >>> d = [x for x in r2.__dict__.items()]
    >>> d.sort()
    >>> d
    [('d', array([1, 2, 3, 4])), ('dateo', -1), ('datev', -1), ('datyp', -1), ('deet', -1), ('etiket', '            '), ('fileref', None), ('grid', None), ('grtyp', ' '), ('handle', -2), ('ig1', -1), ('ig2', -1), ('ig3', -1), ('ig4', -1), ('ip1', -1), ('ip2', -1), ('ip3', -1), ('nbits', -1), ('ni', -1), ('nj', -1), ('nk', -1), ('nom', '    '), ('npas', -1), ('nxt', 0), ('type', '  ')]
    >>> r.d[1] = 9 #r2 is a copy of r, thus this does not change r2.d
    >>> r2.d
    array([1, 2, 3, 4])

    @param data data part of the rec, can be a python list, numpy.ndarray or another RPNRec
    @param meta meta part of the record (RPNMeta), if data is an RPNRec it should not be provided
    @exceptions TypeError if arguments are not of valid type
    """
    def allowedKeysVals(self):
        """Return a dict of allowed Keys/Vals"""
        a = RPNMeta.allowedKeysVals(self).copy()
        a['d'] = None
        a['grid'] = None
        return a

    def __init__(self, data=None, meta=None):
        RPNMeta.__init__(self)
        if data is None:
            self.d = numpy.array([])
        elif type(data) == numpy.ndarray:
            self.d = data
        elif type(data) == type([]):
            self.d = numpy.array(data)
        elif isinstance(data, RPNRec):
            if meta:
                raise TypeError('RPNRec: cannot initialize with both an RPNRec and meta')
            self.d = data.d.copy()
            meta = RPNMeta(data)
        else:
            raise TypeError('RPNRec: cannot initialize data from arg #1')
        if meta:
            if isinstance(meta, RPNMeta):
                self.update(meta)
            elif type(meta) == type({}):
                self.update_by_dict(meta)
            else:
                raise TypeError('RPNRec: cannot initialize parameters from arg #2')

    def __setattr__(self, name, value):   # this method cannot create new attributes
        if name == 'd':
            if type(value) == numpy.ndarray:
                self.__dict__[name] = value
            else:
                raise TypeError('RPNRec: data should be an instance of numpy.ndarray')
        elif name == 'grid':
            if isinstance(value, RPNGrid):
                self.__dict__[name] = value
            else:
                raise TypeError('RPNRec: grid should be an instance of RPNGrid')
        else:
            RPNMeta.__setattr__(self, name, value)

    def interpol(self, togrid):
        """Interpolate RPNRec to another grid (horizontally)

        myRPNRec.interpol(togrid)
        @param togrid grid where to interpolate
        @exception ValueError if myRPNRec does not contain a valid grid desc
        @exception TypeError if togrid is not an instance of RPNGrid
        """
        if isinstance(togrid, RPNGrid):
            if not isinstance(self.grid, RPNGrid):
                self.setGrid()
            if self.grid:
                self.d = togrid.interpol(self.d, self.grid)
                self.setGrid(togrid)
            else:
                raise ValueError('RPNRec.interpol(togrid): unable to determine actual grid of RPNRec')
        else:
            raise TypeError('RPNRec.interpol(togrid): togrid should be an instance of RPNGrid')

    def setGrid(self, newGrid=None):
        """Associate a grid to the RPNRec (or try to get grid from rec metadata)

        myRPNRec.setGrid()
        myRPNRec.setGrid(newGrid)

        @param newGrid grid to associate to the record (RPNGrid)
        @exception ValueError if newGrid does not have same shape as rec data or if it's impossible to determine grid params
        @exception TypeError if newGrid is not an RPNGrid

        >>> r = RPNRec([1, 2, 3, 4], RPNMeta())
        >>> g = RPNGrid(grtyp='N', ig14=(1, 2, 3, 4), shape=(4, 1))
        >>> (g.grtyp, g.shape, g.ig14)
        ('N', (4, 1), (1, 2, 3, 4))
        >>> r.setGrid(g)
        >>> (r.grtyp, (r.ni, r.nj), (r.ig1, r.ig2, r.ig3, r.ig4))
        ('N', (4, 1), (1, 2, 3, 4))
        >>> (r.grid.grtyp, r.grid.shape, r.grid.ig14)
        ('N', (4, 1), (1, 2, 3, 4))
        """
        if newGrid:
            if isinstance(newGrid, RPNGrid):
                ni = max(self.d.shape[0], 1)
                nj = 1
                if len(self.d.shape) > 1:
                    nj = max(self.d.shape[1], 1)
                if (ni, nj) != newGrid.shape:
                    raise ValueError('RPNRec.setGrid(newGrid): rec data and newGrid do not have the same shape')
                else:
                    self.grid = newGrid
                    self.grtyp = newGrid.grtyp
                    (self.ig1, self.ig2, self.ig3, self.ig4) = newGrid.ig14
                    (self.ni, self.nj) = newGrid.shape
            else:
                raise TypeError('RPNRec.setGrid(newGrid): newGrid should be an instance of RPNGrid')
        else:
            self.grid = RPNGrid(self)
            if self.grid:
                self.grtyp = self.grid.grtyp
                (self.ig1, self.ig2, self.ig3, self.ig4) = self.grid.ig14
                (self.ni, self.nj) = self.grid.shape
            else:
                raise ValueError('RPNRec.setGrid(): unable to determine actual grid of RPNRec')

    def __repr__(self):
        kv = self.__dict__.copy()
        rd = repr(kv['d'])
        rg = repr(kv['grid'])
        del kv['d']
        del kv['grid']
        return 'RPNRec{meta='+repr(RPNMeta(kv))+', grid='+rg+', d='+rd+'}'

class RPNDate:
    """RPN STD Date representation

    myRPNDate = RPNDate(DATESTAMP)
    myRPNDate = RPNDate(YYYYMMDD, HHMMSShh)
    myRPNDate = RPNDate(myDateTime)
    myRPNDate = RPNDate(myRPNMeta)

    @param DATESTAMP CMC date stamp or RPNDate object
    @param YYYYMMDD  Int with Visual representation of YYYYMMDD
    @param HHMMSShh  Int with Visual representation of HHMMSShh
    @param myDateTime Instance of Python DateTime class
    @param myRPNMeta Instance RPNMeta with dateo, deet, npas properly set
    @exception TypeError if parameters are wrong type
    @exception ValueError if myRPNMeta

    >>> d1 = RPNDate(20030423, 11453500)
    >>> d1
    RPNDate(20030423, 11453500)
    >>> d2 = RPNDate(d1)
    >>> d2
    RPNDate(20030423, 11453500)
    >>> d2.incr(48)
    RPNDate(20030425, 11453500)
    >>> d1-d2
    -48.0
    >>> a = RPNMeta(dateo=d1.stamp, deet=1800, npas=3)
    >>> d3 = RPNDate(a)
    >>> d3
    RPNDate(20030423, 13153500)
    >>> utc = pytz.timezone("UTC")
    >>> d4 = datetime.datetime(2003, 4, 23, 11, 45, 35, 0, tzinfo=utc)
    >>> d5 = RPNDate(d4)
    >>> d5
    RPNDate(20030423, 11453500)
    >>> d6 = d5.toDateTime()
    >>> d6 == d4
    True
    """
    stamp = 0

    def __init__(self, word1, word2=-1):
        if isinstance(word1, datetime.datetime):
            (yyyy, mo, dd, hh, mn, ss, dummy, dummy2, dummy3) = word1.utctimetuple()
            cs = int(word1.microsecond/10000)
            word1 = yyyy*10000+mo*100+dd
            word2 = hh*1000000+mn*10000+ss*100+cs
        if isinstance(word1, RPNDate):
            self.stamp = word1.stamp
        elif isinstance(word1, RPNMeta):
            if word1.deet < 0 or word1.npas < 0 or word1.dateo <= 0:
                raise ValueError('RPNDate: Cannot compute date from RPNMeta')
            nhours = (1.*word1.deet*word1.npas)/3600.
            self.stamp = Fstdc.incdatr(word1.dateo, nhours)
        elif type(word1) == type(0):
            if word2 == -1:
                self.stamp = word1
            else:
                dummy = 0
#TODO: define named Cst for newdate in Fstdc
                (self.stamp, dummy1, dummy2) = \
                    Fstdc.newdate(dummy, word1, word2, Fstdc.NEWDATE_PRINT2STAMP)
        else:
            raise TypeError('RPNDate: arguments should be of type int')

    def __sub__(self, other):
        "Time difference between 2 dates"
        return Fstdc.difdatr(self.stamp, other.stamp)

    def incr(self, temps):
        """Increase Date by the specified number of hours

        @param temps Number of hours for the RPNDate to be increased
        @return self

        @exception TypeError if temps is not of int or real type
        """
        if type(temps) == type(1) or type(temps) == type(1.0):
            nhours = 0.0
            nhours = temps
            self.stamp = Fstdc.incdatr(self.stamp, nhours)
            return self
        else:
            raise TypeError('RPNDate.incr: argument should be int or real')

    def toDateTime(self):
        """Return the DateTime obj representing the RPNDate

        >>> myRPNDate = RPNDate(20030423, 11453600)
        >>> myDateTime = myRPNDate.toDateTime()
        >>> myDateTime
        datetime.datetime(2003, 4, 23, 11, 45, 35, tzinfo=<UTC>)

        #TODO: oups 1 sec diff!!!
        """
        word1 = word2 = 0
        (dummy, word1, word2) = Fstdc.newdate(self.stamp, word1, word2,
                                              Fstdc.NEWDATE_STAMP2PRINT)
        d = "{0:08d}.{1:08d}".format(word1, word2)
        yyyy = int(d[0:4])
        mo = int(d[4:6])
        dd = int(d[6:8])
        hh = int(d[9:11])
        mn = int(d[11:13])
        ss = int(d[13:15])
        cs = int(d[15:17])
        utc = pytz.timezone("UTC")
        return datetime.datetime(yyyy, mo, dd, hh, mn, ss, cs*10000, tzinfo=utc)

    def __repr__(self):
        word1 = word2 = 0
        (dummy, word1, word2) = Fstdc.newdate(self.stamp, word1, word2,
                                              Fstdc.NEWDATE_STAMP2PRINT)
        return "RPNDate({0:08d}, {1:08d})".format(word1, word2)


#TODO: make dateRange a sequence obj with .iter() methode to be ableto use it in a for statement

class RPNDateRange:
    """RPN STD Date Range representation

    RPNDateRange(DateStart, DateEnd, Delta)
    @param DateStart RPNDate start of the range
    @param DateEnd   RPNDate end of the range
    @param Delta     Increment of the range iterator, hours, real

    @exception TypeError if parameters are wrong type

    >>> d1 = RPNDate(20030423, 11453500)
    >>> d2 = RPNDate(d1)
    >>> d2.incr(48)
    RPNDate(20030425, 11453500)
    >>> dr = RPNDateRange(d1, d2, 6)
    >>> dr
    RPNDateRage(from:(20030423, 11453500), to:(20030425, 11453500), delta:6) at (20030423, 11453500)
    >>> dr.length()
    48.0
    >>> dr.next()
    RPNDate(20030423, 17453500)
    >>> dr = RPNDateRange(d1, d2, 36)
    >>> dr
    RPNDateRage(from:(20030423, 17453500), to:(20030425, 11453500), delta:36) at (20030423, 17453500)
    >>> dr.next()
    RPNDate(20030425, 05453500)
    >>> dr.next() #returns None because it is past the end of DateRange
    """
    #TODO: make this an iterator
    dateDebut = -1
    dateFin = -1
    delta = 0.0
    now = -1

    def __init__(self, debut=-1, fin=-1, delta=0.0):
        if isinstance(debut, RPNDate) and isinstance(fin, RPNDate) and \
                (type(delta) == type(1) or type(delta) == type(1.0)):
            self.dateDebut = debut
            self.now = debut
            self.dateFin = fin
            self.delta = delta
        else:
            raise TypeError('RPNDateRange: arguments type error RPNDateRange(RPNDate, RPNDate, Real)')

    def length(self):
        """Provide the duration of the date range
        @return Number of hours
        """
        return abs(self.dateFin-self.dateDebut)

    def lenght(self):
        """(deprecated, use length) Provide the duration of the date range
        Kept for backward compatibility, please use length()
        """
        return self.length()

    def remains(self):
        """Provide the number of hours left in the date range
        @return Number of hours left in the range
        """
        return abs(self.dateFin - self.now)

    def next(self):
        """Return the next date/time in the range (step of delta hours)
        @return next RPNDate, None if next date is beyond range
        """
        self.now.incr(self.delta)
        if (self.dateFin-self.now)*self.delta < 0.:
            return None
        return RPNDate(self.now)

    def reset(self):
        """Reset the RPNDateRange iterator to the range start date"""
        self.now = self.dateDebut

    def __repr__(self):
        d1 = repr(self.dateDebut)
        d2 = repr(self.dateFin)
        d0 = repr(self.now)
        return "RPNDateRage(from:{0}, to:{1}, delta:{2}) at {3}".\
                format(d1[7:27], d2[7:27], self.delta, d0[7:27])


FirstRecord = RPNMeta()
NextMatch = None


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
