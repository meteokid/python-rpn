#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn is a ctypes import of librmnshared.so
 
The librmn.proto python module includes ctypes prototypes for many
librmn C functions

 Warning:
    Please use with caution.
    The functions in this module are actual C funtions and
    must thus be called as such with appropriate argument typing and
    dereferencing.
    It is highly advised in a python program to prefer the use of the
    python wrapper found in
    * rpnpy.librmn.base
    * rpnpy.librmn.fstd98
    * rpnpy.librmn.interp
    * rpnpy.librmn.grids

 See Also:
    rpnpy.librmn.base
    rpnpy.librmn.fstd98
    rpnpy.librmn.interp
    rpnpy.librmn.grids
    rpnpy.librmn.const

 === EXTERNAL FUNCTIONS in primitive ===

    c_fclos(iun):
        Close file associated with unit iun.
        Proto:
           int c_fclos(int iun)
        Args:
           iun (int): (I) unit number
        Returns:
           int, zero if the connection is successful, non-zero otherwise

    c_fnom(iun, nom, ftype, lrec):
        Open a file and make the connection with a unit number.
        Proto:
           int c_fnom(int *iun, char *nom, char *ftype, int lrec)
        Args:
           iun   (int): (I/O) unit number
           nom   (str): (I) string containing the name of the file
           ftype (str): (I) string that contains the desired file attributes
           lrec  (int): (I) length of record(must be 0 except
                            if type contains D77)
        Returns:
           int, zero if the connection is successful, non-zero otherwise

    c_wkoffit(nom, l1):
        Return a code for the file type
        Proto:
           wordint c_wkoffit(char *nom, int l1) 
        Args:
           nom  (str) : (I) file path/name
           l1   (int) : (I) length of nom
        Returns:
           int, file type code

    c_crc32():
        Compute the Cyclic Redundancy Check (CRC)
        Proto:
           unsigned int crc32(unsigned int crc, const unsigned char *buf,
                              unsigned int lbuf)
        Args:
           crc  (int) : (I) initial crc
           buf        : (I) list of params to compute updated crc
                            (numpy.ndarray of type uint32)
           lbuf (int) : (I) length of buf*4
        Returns:
           int, Cyclic Redundancy Check number


 === EXTERNAL FUNCTIONS in base ===

    f_cigaxg(cgtyp, xg1, xg2, xg3, xg4, ig1, ig2, ig3, ig4)
        Encode real grid descriptors into ig1, ig2, ig3, ig4
        Proto:
           subroutine cigaxg(cgtyp, xg1, xg2, xg3, xg4, ig1, ig2, ig3, ig4)
            character(len=*) :: cgtyp
            integer :: ig1, ig2, ig3, ig4
            real :: xg1, xg2, xg3, xg4
        Args:
           in    - cgtyp - type de grille (voir ouvrir)
           out   - xg1   - ** descripteur de grille (reel),
           out   - xg2   -    igtyp = 'n', pi, pj, d60, dgrw
           out   - xg3   -    igtyp = 'l', lat0, lon0, dlat, dlon,
           out   - xg4   -    igtyp = 'a', 'b', 'g', xg1 = 0. global,
                                                         = 1. nord
                                                         = 2. sud **
                              igtyp = 'e', lat1, lon1, lat2, lon2
           in    - ig1   - descripteur de grille (entier) voir ouvrir
           in    - ig2   - descripteur de grille (entier) voir ouvrir
           in    - ig3   - descripteur de grille (entier) voir ouvrir
           in    - ig4   - descripteur de grille (entier) voir ouvrir

    f_cxgaig(cgtyp, ig1, ig2, ig3, ig4, xg1, xg2, xg3, xg4)
        encode real grid descriptors into ig1, ig2, ig3, ig4
        Proto:
            subroutine cxgaig(cgtyp, ig1, ig2, ig3, ig4, xg1, xg2, xg3, xg4)
            character(len=*) :: cgtyp
            integer :: ig1, ig2, ig3, ig4
            real :: xg1, xg2, xg3, xg4
        Args:
            in    - cgtyp - type de grille (voir ouvrir)
            out   - ig1   - descripteur de grille (entier) voir ouvrir
            out   - ig2   - descripteur de grille (entier) voir ouvrir
            out   - ig3   - descripteur de grille (entier) voir ouvrir
            out   - ig4   - descripteur de grille (entier) voir ouvrir
            in    - xg1   - ** descripteur de grille (reel),
            in    - xg2   -    igtyp = 'n', pi, pj, d60, dgrw
            in    - xg3   -    igtyp = 'l', lat0, lon0, dlat, dlon,
            in    - xg4   -    igtyp = 'a', 'b', 'g', xg1 = 0, global
                                                          = 1, nord
                                                          = 2, sud **
                               igtyp = 'e', lat1, lon1, lat2, lon2

    f_incdati(idate1, idate2, nhours)
        increase idate2 by nhours (idate1=idate2+nhours), rounded idate2,
        nhours to nearest hour
        Proto:
            subroutine incdati(idate1, idate2, nhours)
            integer idate1, idate2
            real*8  nhours
        Args:
            ... TODO ...
        Note:
            if incdat receive bad arguments,
            idate1=101010101 (1910/10/10 10z run 1)

    f_incdatr(idate1, idate2, nhours)
        increase idate2 by nhours (idate1=idate2+nhours)
        Proto:
            subroutine incdatr(idate1, idate2, nhours)
            integer idate1, idate2
            real*8  nhours
        Args:
            ... TODO ...
        Note:
            if incdat receive bad arguments,
            idate1=101010101 (1910/10/10 10z run 1)

    f_difdati(idate1, idate2, nhours)
        Compute date difference in hours, rounded idate2, nhours
        (nhours=idate1-idate2), rounded idate1, idate2 to nearest hour
        Proto:
            subroutine difdati(idate1, idate2, nhours)
            integer idate1, idate2
            real*8  nhours
        Args:
             ... TODO ...
       Note:
            if difdat receive bad arguments,
            idate1=101010101 (1910/10/10 10z run 1)

    f_difdatr(idate1, idate2, nhours)
        Compute date difference in hours, rounded idate2, nhours
        (nhours=idate1-idate2)
        Proto:
            subroutine difdatr(idate1, idate2, nhours)
            integer idate1, idate2
            real*8  nhours
        Args:
            ... TODO ...
        Note:
            if difdat receive bad arguments,
            idate1=101010101 (1910/10/10 10z run 1)

    f_NewDate_Options(value, command, value_len, command_len)
        Set/get option for newdate, incdatr, difdatr
        Proto:
           subroutine NewDate_Options(value, command)
              character*(*) value, command
        Args:
           value   (I/O): option and value to be set/get (str)
                          possible values:
                          if command == 'get':
                             'year'
                          if command == 'set':
                             'year=gregorian'
                             'year=365_day'
                             'year=360_day'
           command (I)  : type of operation (str)
                          possible values: 'set', 'get', 'unset'
        Note:
           A) Permits alternative calendar options, via either
              the NEWDATE_OPTIONS environment variable (which
              has precedence) or via appropriate "set" commands
           B) Also, returns calendar status via the "get" command
           C) The Get_Calendar_Status entry also return this
           The known calendars options are currently:
             gregorian
             365_day (no leap years) and
             360_day


    f_Ignore_LeapYear()
        Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr
        Equivalent to: NewDate_Options('year=365_day', 'set')
        Proto:
           subroutine Ignore_LeapYear()
        Args:
           None


    f_Accept_LeapYear()
        Set the 'no leap years' (365_day) option for newdate, incdatr, difdatr
        Equivalent to: NewDate_Options('year=gregorian', 'set')
        Proto:
           subroutine Accept_LeapYear()
        Args:
           None


    f_newdate(dat1, dat2, dat3, mode)
        converts dates between two of the following formats:
        printable date, cmc date-time stamp, true date
        Proto:
            function newdate(dat1, dat2, dat3, mode)
            integer newdate, dat1, dat2(*), dat3, mode
        Args:
            See Note below
        Note:
            mode can take the following values:
            -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7
            mode=1 : stamp to (true_date and run_number)
                out - dat1 - the truedate corresponding to dat2
                 in - dat2 - cmc date-time stamp (old or new style)
                out - dat3 - run number of the date-time stamp
                 in - mode - set to 1
            mode=-1 : (true_date and run_number) to stamp
                 in - dat1 - truedate to be converted
                out - dat2 - cmc date-time stamp (old or new style)
                 in - dat3 - run number of the date-time stamp
                 in - mode - set to -1
            mode=2 : printable to true_date
                out - dat1 - true_date
                 in - dat2 - date of the printable date (yyyymmdd)
                 in - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to 2
            mode=-2 : true_date to printable
                 in - dat1 - true_date
                out - dat2 - date of the printable date (yyyymmdd)
                out - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to -2
            mode=3 : printable to stamp
                out - dat1 - cmc date-time stamp (old or new style)
                 in - dat2 - date of the printable date (yyyymmdd)
                 in - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to 3
            mode=-3 : stamp to printable
                 in - dat1 - cmc date-time stamp (old or new style)
                out - dat2 - date of the printable date (yyyymmdd)
                out - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to -3
            mode=4 : 14 word old style date array to stamp and array(14)
                out - dat1 - cmc date-time stamp (old or new style)
                 in - dat2 - 14 word old style date array
                 in - dat3 - unused
                 in - mode - set to 4
            mode=-4 : stamp to 14 word old style date array
                 in - dat1 - cmc date-time stamp (old or new style)
                out - dat2 - 14 word old style date array
                 in - dat3 - unused
                 in - mode - set to -4
            mode=5    printable to extended stamp (year 0 to 10, 000)
                out - dat1 - extended date-time stamp (new style only)
                 in - dat2 - date of the printable date (yyyymmdd)
                 in - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to 5
            mode=-5   extended stamp (year 0 to 10, 000) to printable
                 in - dat1 - extended date-time stamp (new style only)
                out - dat2 - date of the printable date (yyyymmdd)
                out - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to -5
            mode=6 :  extended stamp to extended true_date (in hours)
                out - dat1 - the truedate corresponding to dat2
                 in - dat2 - cmc date-time stamp (old or new style)
                out - dat3 - run number, unused (0)
                 in - mode - set to 6
            mode=-6 : extended true_date (in hours) to extended stamp
                 in - dat1 - truedate to be converted
                out - dat2 - cmc date-time stamp (old or new style)
                 in - dat3 - run number, unused
                 in - mode - set to -6
            mode=7  - printable to extended true_date (in hours)
                out - dat1 - extended true_date
                 in - dat2 - date of the printable date (yyyymmdd)
                 in - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to 7
            mode=-7 : extended true_date (in hours) to printable
                 in - dat1 - extended true_date
                out - dat2 - date of the printable date (yyyymmdd)
                out - dat3 - time of the printable date (hhmmsshh)
                 in - mode - set to -7

 === EXTERNAL FUNCTIONS in fstd98 ===

    c_fstecr(field_in, work, npak, iun, date, deet, npas,
             ni, nj, nk, ip1, ip2, ip3,
             in_typvar, in_nomvar, in_etiket,
             in_grtyp, ig1, ig2, ig3, ig4,
             in_datyp_ori, rewrit)
        Writes record to file.
        Proto:
           int c_fstecr(word *field_in, void * work, int npak,
                        int iun, int date,
                        int deet, int npas,
                        int ni, int nj, int nk,
                        int ip1, int ip2, int ip3,
                        char *in_typvar, char *in_nomvar, char *in_etiket,
                        char *in_grtyp, int ig1, int ig2,
                        int ig3, int ig4,
                        int in_datyp_ori, int rewrit)
        Args:
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

        Note:
           librmn.c_fstecr.argtypes default to data of type _np.float32
           To write other data types you will need to redefine it with the
           appropriate type


    c_fst_edit_dir(handle, date, deet, npas,
                   ni, nj, nk, ip1, ip2, ip3,
                   in_typvar, in_nomvar, in_etiket,
                   in_grtyp, ig1, ig2, ig3, ig4, datyp)
        Edits the directory content of a RPN standard file.
        Proto:
           int c_fst_edit_dir(int handle,
                   unsigned int date, int deet, int npas,
                   int ni, int nj, int nk,
                   int ip1, int ip2, int ip3,
                   char *in_typvar, char *in_nomvar, char *in_etiket,
                   char *in_grtyp, int ig1, int ig2,
                   int ig3, int ig4, int datyp)
        Args:
           ... TODO ...
        Returns:
           int, zero successful, non-zero otherwise

    c_fsteff(handle)
        Deletes the record associated to handle.
        Proto:
           int c_fsteff(int handle)
        Args:
           handle (int) : (I) handle of the record to delete
        Returns:
           int, zero successful, non-zero otherwise

    c_fstfrm(iun)
        Closes a RPN standard file
        Proto:
           int c_fstfrm(int iun)
        Args:
           iun (int): (I) unit number
        Returns:
           int, zero successful, non-zero otherwise

    c_fstinf(iun, ni, nj, nk, datev, in_etiket,
             ip1, ip2, ip3, in_typvar, in_nomvar)
        Locate the next record that matches the research keys
        Proto:
            int c_fstinf(int iun, int *ni, int *nj, int *nk,
                         int datev, char *in_etiket,
                         int ip1, int ip2, int ip3,
                         char *in_typvar, char *in_nomvar)
        Args:
            iun (int): (I) unit number
            ... TODO ...
        Returns:
            int, ... TODO ...

    c_fstinfx(handle, iun, ni, nj, nk, datev, in_etiket,
              ip1, ip2, ip3, in_typvar, in_nomvar)
        Locate the next record that matches the research keys.
        The search begins at the position given by handle.
        Proto:
        int c_fstinfx(int handle, int iun, int *ni, int *nj, int *nk,
                      int datev, char *in_etiket,
                      int ip1, int ip2, int ip3,
                      char *in_typvar, char *in_nomvar)
        Args:
            handle (int): (I) handle record used as a starting point
            (<0 for file start)
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, ... TODO ...

    c_fstinl(iun, ni, nj, nk, datev, etiket, ip1, ip2, ip3, typvar, nomvar,
             liste, infon, nmax)
        Locates all the records that matches the research keys
        Proto:
            int c_fstinl(int iun, int *ni, int *nj, int *nk, int datev, 
                         char *etiket, int ip1, int ip2, int ip3,
                         char *typvar, char *nomvar,
                         word *liste, int *infon, int nmax)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

    c_fstlic(field, iun, niin, njin, nkin,
             datein, etiketin, ip1in, ip2in, ip3in,
             typvarin, nomvarin,
             ig1in, ig2in, ig3in, ig4in, grtypin)
        Search for a record that matches the research keys and
        check that the remaining parmeters match the record descriptors
        Proto:
            int c_fstlic(word *field, int iun, int niin, int njin, int nkin,
                         int datein, char *etiketin,
                         int ip1in, int ip2in, int ip3in,
                         char *typvarin, char *nomvarin,
                         int ig1in, int ig2in, int ig3in, int ig4in,
                         char *grtypin)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

    c_fstlir(field, iun, ni, nj, nk, datev, etiket,
             ip1, ip2, ip3, typvar, nomvar)
        Reads the next record that matches the research keys
        Proto:
            int c_fstlir(word *field, int iun, int *ni, int *nj, int *nk,
                         int datev, char *etiket,
                         int ip1, int ip2, int ip3, char *typvar, char *nomvar)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, key/handle to record, <0 on error

    c_fstlirx(field, handle, iun, ni, nj, nk, datev, etiket, ip1, ip2, ip3,
              typvar, nomvar)
        Reads the next record that matches the research keys.
        The search begins at the position given by handle.
        Proto:
            int c_fstlirx(word *field, int handle, int iun,
                          int *ni, int *nj, int *nk, int datev, char *etiket,
                          int ip1, int ip2, int ip3, char *typvar, char *nomvar)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, key/handle to record, <0 on error

    c_fstlis(field, iun, ni, nj, nk)
        Reads the next record that matches the last search criterias
        Proto:
            int c_fstlis(word *field, int iun, int *ni, int *nj, int *nk)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

    f_fstlnk(liste, n)
        Links a list of files together for search purpose
        Proto:
            ftnword f77name(fstlnk)(ftnword *liste, ftnword *f_n)
        Args:
            liste : (I) list of unit numbers (ndpointer(dtype=_np.int32))
            n     : (I) size of liste (int)
        Returns:
            int, zero successful, non-zero otherwise
    
    c_fstluk(field, handle, ni, nj, nk)
        Read the record at position given by handle.
        Proto:
            int c_fstluk(word *field, int handle, int *ni, int *nj, int *nk)
        Args:
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise
        Note:
            librmn.c_fstluk.argtypes default to data of type _np.float32
            To read other data types you will need to redefine it with the
            appropriate type

    c_fstmsq(field, iun, ni, nj, nk, datev, etiket,
             ip1, ip2, ip3, typvar, nomvar)
        Mask a portion of the research keys
        Proto:
            int c_fstmsq(int iun, int *mip1, int *mip2, int *mip3,
                         char *metiket, int getmode)
        Args:
            iun     (int): (I) unit number
            mip1    (int): (I/O) mask for vertical level
            mip2    (int): (I/O) mask for forecast hour
            mip3    (int): (I/O) mask for ip3
            metiket (str): (I/O) mask for label
            getmode (int): (I)   1: getmode, 0:set mode
        Returns:
            int, zero successful, non-zero otherwise

    c_fstnbr(iun)
        Returns the number of records of the file associated with unit
        Proto:
            int c_fstnbr(int iun)
        Args:
            iun (int): (I) unit number
        Returns:
            int, number of records of the file associated with unit

    c_fstnbrv(iun)
        Returns the number of valid records (excluding deleted records)
        Proto:
            int c_fstnbrv(int iun)
        Args:
            iun (int): (I) unit number
        Returns:
            int, number of validrecords of the file associated with unit

    c_fstopc(option, value, getmode)
        Prout or set a fstd or xdf global variable option.
        Proto:
            int c_fstopc(char *option, char *value, int getmode)
        Args:
            IN     option   (str) option name to be set/printed
            IN     value    (str) option value
            IN     getmode  (int) logical (1: get option, 0: set option)
        Returns:
            int, zero successful, non-zero otherwise

    c_fstopi(option, value, getmode)
        Prout or set a fstd or xdf global variable option.
        Proto:
            int c_fstopi(char *option, int value, int getmode)
        Args:
            IN     option   (str) option name to be set/printed
            IN     value    (int) option value
            IN     getmode  (int) logical (1: get option, 0: set option)
        Returns:
            int, zero successful, non-zero otherwise

    c_fstouv(iun, options)
        Opens a RPN standard file.
        Proto:
           int c_fstouv(int iun, char *options)
        Args:
           IN  iun     unit number associated to the file
           IN  options random or sequential access
        Returns:
           int, zero successful, non-zero otherwise

    c_fstprm(handle, dateo, deet, npas, ni, nj, nk, nbits, datyp,
             ip1, ip2, ip3, typvar, nomvar, etiket, 
             grtyp, ig1, ig2, ig3, ig4,
             swa, lng, dltf, ubc, extra1, extra2, extra3)
        Get all the description informations of the record.
        Proto:
            int c_fstprm(int handle,
                     int *dateo, int *deet, int *npas,
                     int *ni, int *nj, int *nk,
                     int *nbits, int *datyp, int *ip1,
                     int *ip2, int *ip3, char *typvar,
                     char *nomvar, char *etiket, char *grtyp,
                     int *ig1, int *ig2, int *ig3,
                     int *ig4, int *swa, int *lng,
                     int *dltf, int *ubc, int *extra1,
                     int *extra2, int *extra3)
        Args:
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

    c_fstsui(iun, ni, nj, nk)
        Finds the next record that matches the last search criterias
        Proto:
            int c_fstsui(int iun, int *ni, int *nj, int *nk)
        Args:
            iun    (int): (I) unit number
            ... TODO ...
        Returns:
            int, ... TODO ...

    c_fst_version()
        Returns package version number
        Proto:
            int c_fst_version()
        Returns:
            int, package version number

    c_fstvoi(iun, options)
        Prints out the directory content of a RPN standard file
        Proto:
            int c_fstvoi(int iun, char *options)
        Args:
            IN  iun     unit number associated to the file
            ... TODO ...
        Returns:
            int, zero successful, non-zero otherwise

    c_ip1_all(level, kind)
        Generates all possible coded ip1 values for a given level
        Proto:
            int c_ip1_all(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip
        Returns:
            int, ip1new on success, -1 on error
            
    c_ip2_all(level, kind)
        Generates all possible coded ip2 values for a given level
        Proto:
            int c_ip2_all(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip
        Returns:
            int, ip2new on success, -1 on error

    c_ip3_all(level, kind)
        Generates all possible coded ip3 values for a given level
        Proto:
            int c_ip3_all(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip
        Returns:
            int, ip3new on success, -1 on error

    c_ip1_val(level, kind)
        Generates coded ip1 value for a given level (shorthand for convip)
        Proto:
            int c_ip1_val(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip   
        Returns:
            int, ip1new on success, -1 on error

    c_ip2_val(level, kind)
        Generates coded ip2 value for a given level (shorthand for convip)
        Proto:
            int c_ip2_val(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip   
        Returns:
            int, ip2new on success, -1 on error

    c_ip3_val(level, kind)
        Generates coded ip3 value for a given level (shorthand for convip)
        Proto:
            int c_ip3_val(float level, int kind)
        Args:
            IN  level          ip level (float value)
            IN  kind           level kind as defined in convip   
        Returns:
            int, ip3new on success, -1 on error

    c_ip_is_equal(target, ip, ind)
        Compares different coded values of an ip for equality
        Proto:
            int ip_is_equal(int target, int ip, int ind)
        Args:
            IN target: must be first value in the table of coded value
                       to compare with
            IN ip    : current ip record value to compare
            IN ind   : index (1, 2 or 3)
                       representing ip1, ip2 or ip3 comparaisons
        Returns:
            int, ... TODO ...
        
 === EXTERNAL FUNCTIONS in fstd98/convip_plus and fstd98/convert_ip123 ===

    c_ConvertIp(ip, p, kind, mode)
        Codage/Decodage P, kind <-> IP pour IP1, IP2, IP3
        Args:
            ip   (int)  : (I/O) Valeur codee
            p    (float): (I/O) Valeur reelle
            kind (int)  : (I/O) Type de niveau
            mode (int)  : (I)   Mode de conversion
        Note: successeur de convip
            kind:
                0, p en metre rel. au niveau de la mer  (-20, 000 -> 100, 000)
                1, p est en sigma                       (0.0 -> 1.0)
                2, p est en pression (mb)               (0 -> 1100)
                3, p est un code arbitraire             (-4.8e8 -> 1.0e10)
                4, p est en metre rel. au niveau du sol (-20, 000 -> 100, 000)
                5, p est en coordonnee hybride          (0.0 -> 1.0)
                6, p est en coordonnee theta            (1 -> 200, 000)
                10, p represente le temps en heure      (0.0 -> 1.0e10)
                15, reserve (entiers)                                   
                17, p indice x de la matrice de conversion
                                                        (1.0 -> 1.0e10)
                    (partage avec kind=1 a cause du range exclusif
                21, p est en metres-pression (fact=1e4) (0 -> 1, 000, 000)
                    (partage avec kind=5 a cause du range exclusif)
            mode:
                -1, de IP -->  P
                0, forcer conversion pour ip a 31 bits
                   (default = ip a 15 bits) (initialisation call)
                +1, de P  --> IP
                +2, de P  --> IP en mode NEWSTYLE force a true
                +3, de P  --> IP en mode NEWSTYLE force a false

    c_ConvertIPtoPK(rp1, kind1, rp2, kind2, rp3, kind3, ip1v, ip2v, ip3v)
        Convert/decode ip1, ip2, ip3 to their kind + real value conterparts
        Proto:
            ConvertIPtoPK(RP1, kind1, RP2, kind2, RP3, kind3,
                          IP1V, IP2V, IP3V) result(status)
            integer(C_INT) :: status
            real(C_FLOAT),         intent(OUT) :: rp1, rp2, rp3
            integer(C_INT),        intent(OUT) :: kind1, kind2, kind3
            integer(C_INT), value, intent(IN)  :: ip1v, ip2v, ip3v
        Args:
            ip1v, ip2v, ip3v : ip values to be decoded
            rp1, kind1  : result of ip1v decoding
            rp2, kind2  : result of ip2v decoding
            rp3, kind3  : result of ip3v decoding
        Returns:
            int, 0 if ok, >0 on guessed the value, 32 on warning, 64 on error 

    c_ConvertPKtoIP(IP1, IP2, IP3, P1, kkind1, P2, kkind2, P3, kkind3)
        Convert/encode kind + real value into ip1, ip2, ip3
        Proto:
            ConvertPKtoIP(IP1, IP2, IP3, P1, kkind1, P2, kkind2,
                          P3, kkind3) result(status)
            integer(C_INT) :: status
            integer(C_INT),        intent(OUT) :: IP1, IP2, IP3
            real(C_FLOAT),  value, intent(IN)  :: P1, P2, P3
            integer(C_INT), value, intent(IN)  :: kkind1, kkind2, kkind3
        Args:
            p1, kkind1 : must be a level
            p2, kkind2 : should be a time but a level is accepted
                        (flagged as warning)
            p3, kkind3 : may be anything
            ip1, ip2, ip3 : will contain the encoded values in case of success,
                          and are undefined otherwise
        Returns:
            int, 0 if ok, >0 on guessed the value, 32 on warning, 64 on error 

    c_EncodeIp(ip1, ip2, ip3, rp1, rp2, rp3)
        Produce a valid (ip1, ip2, ip3) triplet from (real value, kind) pairs
        Proto:
            function encode_ip_0(ip1, ip2, ip3, rp1, rp2, rp3)
                     result(status) bind (c, name='encodeip')
            integer(C_INT) :: status
            integer(C_INT), intent(OUT) :: IP1, IP2, IP3
            type(FLOAT_IP), intent(IN)  :: RP1, RP2, RP3
        Args:
            RP1 must contain a level (or a pair of levels) in the atmosphere
            RP2 must contain  a time (or a pair of times)
            RP3 may contain anything, RP3.v2 will be ignored
                (if RP1 or RP2 contains a pair, RP3 is ignored)
            IP1, IP2, IP3 will contain the encoded values in case of success,
                and are undefined otherwise
        Returns:
            CONVERT_ERROR=32 in case of error, CONVERT_OK=0

    c_DecodeIp(RP1, RP2, RP3, IP1V, IP2V, IP3V)
        Produce valid (real value, kind) pairs from (ip1, ip2, ip3) triplet
        Proto:
            function decode_ip_0(RP1, RP2, RP3, IP1V, IP2V, IP3V)
                     result(status) BIND (C, name='DecodeIp')
            integer(C_INT) :: status
            integer(C_INT), value, intent(IN)  :: IP1V, IP2V, IP3V
            type(FLOAT_IP), intent(OUT) :: RP1, RP2, RP3
        Args:
            ip1/2/3 : should be encoded 'new style' but old style
                      encoding is accepted
            RP1   : will contain a level (or a pair of levels in
                    atmospheric ascending order) in the atmosphere
            RP2   : will contain a time (or a pair of times in ascending order)
            RP3.v2: will be the same as RP3.v1 (if RP1 or RP2 contains a pair,
                    RP3 is ignored)
        Returns:
            CONVERT_ERROR=32 in case of error, CONVERT_OK=0

    c_KindToString
        Translate kind integer code to 2 character string,
        gateway to Fortran kind_to_string
        Proto:
            void KindToString(int kind, char *s1, char *s2)
        Args:
            kind (int): (I) Valeur codee
            s1 (str): (O) first char
            s2 (str): (O) second char

 === EXTERNAL FUNCTIONS in fstd98/xdf98 ===

    c_xdflnk(liste, n)
        Links the list of random files together for record search purpose
        Proto:
            int c_xdflnk(word *liste, int n)
        Args:
            IN    liste    list of unit numbers associated to the files
            IN    n        number of files to be linked
        Returns:
            int, 0 on success, -1 on error
        Note:
            Use the first unit id in the list to refer to the linked files list

 === EXTERNAL FUNCTIONS in interp (ezscint) ===

    c_ezdefset(gdidout, gdidin)
        Defines a set of grids for interpolation
        gdid = c_ezdefset(gdidout, gdidin)
        Proto:
        wordint c_ezdefset(wordint gdout, wordint gdin)
        Args:
        
        Returns:
        int, gdid on success, -1 on error

    c_ezsint(zout, zin)
        Scalar interpolation
        Proto:
           wordint c_ezsint(ftnfloat *zout, ftnfloat *zin)
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezuvint(uuout, vvout, uuin, vvin)
        Vector interpolation, grid to grid
            ier = c_ezuvint(uuout, vvout, uuin, vvin)
        Proto:
           wordint c_ezuvint(ftnfloat *uuout, ftnfloat *vvout,
                             ftnfloat *uuin, ftnfloat *vvin)
        Args:
              ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezwdint(spdout, wdout, uuin, vvin)
        Vector interpolation, grid to speed/direction
        ier = c_ezwdint(spdout, wdout, uuin, vvin)
        Proto:
           wordint c_ezwdint(ftnfloat *uuout, ftnfloat *vvout,
                             ftnfloat *uuin, ftnfloat *vvin)
        Args:
            ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_gdll(gdid, lat, lon)
        Gets the latitude/longitude position of grid 'gdid'
        ier = c_gdll(gdid, lat, lon)
        Proto:
           wordint c_gdll(wordint gdid, ftnfloat *lat, ftnfloat *lon)
        Args:
            ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezgetval(option, value)
        Gets a numerical option from the package
        ier = c_ezgetval('option', value)
        Proto:
           wordint c_ezgetval(char *option, ftnfloat *fvalue)
        Args:
            ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezgetival(option, value)
        Gets a numerical option from the package
        ier = c_ezgetival('option', value)
        Proto:
           wordint c_ezgetival(char *option, wordint *ivalue)
        Args:
            ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezgetopt(option, value)
        Gets an option from the package (updated 03-2014)
        ier = c_ezgetopt('option', value)
        Proto:
            ... TODO ...
        Args:
            ... TODO ...   
        Returns:
           int, 0 on success, -1 on error

    c_gdxyfll(gdid, x, y, lat, lon, n)
        Returns the x-y positions of lat lon points on grid 'gdid'
        ier = c_gdxyfll(gdid, x, y, lat, lon, n)
        Proto:
           wordint c_gdxyfll_s(wordint gdid, ftnfloat *x, ftnfloat *y,
                               ftnfloat *lat, ftnfloat *lon, wordint n)
        Args:
           ... TODO ...   
        Returns:
           int, 0 on success, -1 on error

    c_gdllfxy(gdid, lat, lon, x, y, n)
        Returns the lat-lon coordinates of data located at positions x-y
        on grid GDID
        ier = c_gdllfxy(gdid, lat, lon, x, y, n)
        Proto:
           wordint c_gdllfxy(wordint gdid, ftnfloat *lat, ftnfloat *lon,
                             ftnfloat *x, ftnfloat *y, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_ezget_nsubgrids(super_gdid)
        Gets the number of subgrids from the 'U' (super) grid id
        nsubgrids = c_ezget_nsubgrids(super_gdid)
        Proto:
           wordint c_ezget_nsubgrids(wordint super_gdid)
        Args:
           super_gdid (int): id of the super grid
        Returns:
           int, number of sub grids associated with super_gdid on success,
                -1 on error

    c_ezget_subgridids(super_gdid, subgridids )
        Gets the list of grid ids for the subgrids in the 'U' grid (super_gdid).
        ier = c_ezget_subgridids(super_gdid, subgridids)
        Proto:
           wordint c_ezget_subgridids(wordint gdid, wordint *subgrid)
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error
   
    c_gdllsval(gdid, zout, zin, lat, lon, n)
        Scalar interpolation of points located at lat-lon coordinates.
        ier = c_gdllsval(gdid, zout, zin, lat, lon, n)
        Proto:
            wordint c_gdllsval(wordint gdid, ftnfloat *zout, ftnfloat *zin,
                               ftnfloat *lat, ftnfloat *lon, wordint n)
        Args:
            ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdxysval(gdid, zout, zin, x, y, n)
        Scalar intepolation of points located at x-y coordinates.
        ier = c_gdxysval(gdid, zout, zin, x, y, n)
        Proto:
           wordint c_gdxysval(wordint gdin, ftnfloat *zout, ftnfloat *zin,
                              ftnfloat *x, ftnfloat *y, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdllvval(gdid, uuout, vvout, uuin, vvin, lat, lon, n)
        Vector interpolation of points located at lat-lon coordinates,
        returned as grid components (UU and VV).
        ier = c_gdllvval(gdid, uuout, vvout, uuin, vvin, lat, lon, n)
        Proto:
           wordint c_gdllvval(wordint gdid, ftnfloat *uuout, ftnfloat *vvout,
                       ftnfloat *uuin, ftnfloat *vvin, 
                       ftnfloat *lat, ftnfloat *lon, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdxyvval(gdid, uuout, vvout, uuin, vvin, x, y, n)
        Vector interpolation of points located at x-y coordinates,
        returned as grid components (UU and VV).
        ier = c_gdxyvval(gdid, uuout, vvout, uuin, vvin, x, y, n)
        Proto:
           wordint c_gdxyvval(wordint gdin, ftnfloat *uuout, ftnfloat *vvout,
                       ftnfloat *uuin, ftnfloat *vvin,
                       ftnfloat *x, ftnfloat *y, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdllwdval(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
        Vector interpolation of points located at lat-lon coordinates,
        returned as speed and direction (UV and WD).
        ier = c_gdllwdval(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
        Proto:
           wordint c_gdllwdval(wordint gdid, ftnfloat *uuout, ftnfloat *vvout,
                       ftnfloat *uuin, ftnfloat *vvin, 
                       ftnfloat *lat, ftnfloat *lon, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error


    c_gdxywdval(gdin, uuout, vvout, uuin, vvin, x, y, n)
        Vector interpolation of points located at x-y coordinates,
        returned as speed and direction (UVand WD).
        Proto:
            wordint c_gdxywdval(wordint gdin, ftnfloat *uuout, ftnfloat *vvout,
                                ftnfloat *uuin, ftnfloat *vvin,
                                ftnfloat *x, ftnfloat *y, wordint n)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gduvfwd(gdid, uuout, vvout, spdin, wdin, lat, lon, n)
        Converts, on grid 'gdid', the direction/speed values at grid points
        to grid coordinates.
        ier = c_gduvfwd(gdid, uuout, vvout, spdin, wdin, lat, lon, n)
        Proto:
             wordint c_gduvfwd(wordint gdid,
                               ftnfloat *uugdout, ftnfloat *vvgdout,
                               ftnfloat *uullin, ftnfloat *vvllin,
                               ftnfloat *latin, ftnfloat *lonin, wordint npts)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdwdfuv(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
        Converts, on grid 'gdid', the grid winds at grid points speed/direction
        ier = c_gdwdfuv(gdid, spdout, wdout, uuin, vvin, lat, lon, n)
        Proto:
            wordint c_gdwdfuv(wordint gdid, ftnfloat *spd_out, ftnfloat *wd_out,
                              ftnfloat *uuin, ftnfloat *vvin, 
                              ftnfloat *latin, ftnfloat *lonin, wordint npts)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_gdsetmask(gdid, mask)
        Associates a permanent mask with grid 'gdid'
        ier = c_gdsetmask(gdid, mask)
        Proto:
            int c_gdsetmask(int gdid, int *mask)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error   

    c_gdgetmask(gdid, mask)
        Returns the mask associated with grid 'gdid'
        ier = c_gdgetmask(gdid, mask)
        Proto:
            int c_gdgetmask(int gdid, int *mask)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error   

    c_ezsint_mdm(zout, mask_out, zin, mask_in)
        Scalar interpolation, using the source field and an associated mask.
        Returns the interpolated field and an interpolated mask.
        ier = c_ezsint_mdm(zout, mask_out, zin, mask_in)
        Proto:
            int c_ezsint_mdm(float *zout, int *mask_out, float *zin,
                             int *mask_in)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_ezuvint_mdm(uuout, vvout, mask_out, uuin, vvin, mask_in)
        Vector interpolation, using the source field and an associated mask.
        Returns the interpolated winds and an interpolated mask.
        ier = c_ezuvint_mdm(uuout, vvout, mask_out, uuin, vvin, mask_in)
        Proto:
            int c_ezuvint_mdm(float *uuout, float *vvout, int *mask_out,
                              float *uuin, float *vvin, int *mask_in)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_ezsint_mask(mask_out, mask_in)
        Interpolation of the source mask to the output mask using the
        current (gdin, gdout) set
        ier = c_ezsint_mask(mask_out, mask_in)
        Proto:
            int c_ezsint_mask(int *mask_out, int *mask_in)
        Args:
           ... TODO ...  
        Returns:
           int, 0 on success, -1 on error

    c_ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
        Universal grid definition. Applicable to all cases.
        gdid = c_ezqkdef(ni, nj, grtyp, ig1, ig2, ig3, ig4, iunit)
        Proto:
           wordint c_ezqkdef(wordint ni, wordint nj, char *grtyp,
                 wordint ig1, wordint ig2, wordint ig3, wordint ig4,
                 wordint iunit)
        Args:
           ... TODO ... 
        Returns:
           int, gdid on success, <0 on error

    c_ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
        Generic grid definition except for 'U' grids (with necessary
        positional parameters taken from the calling arguments)
        gdid = c_ezgdef_fmem(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
        Proto:
           wordint c_ezgdef_fmem(wordint ni, wordint nj,
                   char *grtyp, char *grref,
                   wordint ig1, wordint ig2, wordint ig3, wordint ig4,
                   ftnfloat *ax, ftnfloat *ay)
        Args:
           ... TODO ...
        Returns:
           int, grid id on success, -1 on error

    c_ezgdef_supergrid(ni, nj, grtyp, grref, vercode, nsubgrids, subgridid)
        U grid definition (which associates to a list of concatenated
        subgrids in one record)
        gdid = c_ezgdef_supergrid(ni, nj, grtyp, grref, vercode,
                                  nsubgrids, subgridid)
        Proto:
           wordint c_ezgdef_supergrid(wordint ni, wordint nj,
                   char *grtyp, char *grref,
                   wordint vercode, wordint nsubgrids, wordint *subgrid)
        Args:
           ... TODO ...
        Returns:
           int, super grid id on success, -1 on error

    c_ezgdef(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
        Generic grid definition
        (obsolete - consider using ezqkdef orezgdef_fmem)
        gdid = c_ezgdef(ni, nj, grtyp, grref, ig1, ig2, ig3, ig4, ax, ay)
        Proto:
            wordint c_ezgdef(wordint ni, wordint nj, char *grtyp, char *grref,
                             wordint ig1, wordint ig2, wordint ig3, wordint ig4,
                             ftnfloat *ax, ftnfloat *ay);
        Args:
           ... TODO ...
        Returns:
           int, gtid id on success, -1 on error

    c_ezgprm(gdid, grtyp, ni, nj, ig1, ig2, ig3, ig4)
        Get current grid parameters
        ier = c_ezgprm(gdid, grtyp, ni, nj, ig1, ig2, ig3, ig4)
        Proto:
           wordint c_ezgprm(wordint gdid, char *grtyp, wordint *ni, wordint *nj,
                            wordint *ig1, wordint *ig2, wordint *ig3,
                            wordint *ig4);
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezgxprm(gdid, ni, nj, grtyp, ig1, ig2, ig3, ig4, grref, 
              ig1ref, ig2ref, ig3ref, ig4ref)
        Get extended grid parameters
        ier = c_ezgxprm(gdid, ni, nj, grtyp, ig1, ig2, ig3, ig4, grref,
                        ig1ref, ig2ref, ig3ref, ig4ref)
        Proto:
           wordint c_ezgxprm(wordint gdid, wordint *ni, wordint *nj, 
                char *grtyp, wordint *ig1, wordint *ig2, 
                wordint *ig3, wordint *ig4,
                char *grref, wordint *ig1ref, wordint *ig2ref,
                wordint *ig3ref, wordint *ig4ref);
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezgfstp(gdid, nomvarx, typvarx, etikx, nomvary, typvary, etiky, ip1,
              ip2, ip3, dateo, deet, npas, nbits)
        Get the standard file attributes of the positional records
        ier = c_ezgfstp(gdid, nomvarx, typvarx, etikx, nomvary, typvary,
                        etiky, ip1, ip2, ip3, dateo, deet, npas, nbits)
        Proto:
            wordint c_ezgfstp(wordint gdid,
                      char *nomvarx, char *typvarx, char *etiketx,
                      char *nomvary, char *typvary, char *etikety,
                      wordint *ip1, wordint *ip2, wordint *ip3,
                      wordint *dateo, wordint *deet, wordint *npas,
                      wordint *nbits);
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_gdgaxes(gdid, ax, ay)
        Gets the deformation axes of the 'Z' grid
        ier = c_gdgaxes(gdid, ax, ay)
        Proto:
            wordint c_gdgaxes(wordint gdid, ftnfloat *ax, ftnfloat *ay)
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_gdxpncf(gdid, i1, i2, j1, j2)
        Gets the expansion coefficients used to expand grid 'gdid'
        ier = c_gdxpncf(gdid, i1, i2, j1, j2)
        Proto:
            wordint c_gdxpncf(wordint gdin, wordint *i1, wordint *i2,
                              wordint *j1, wordint *j2)
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_gdgxpndaxes(gdid, ax, ay)
        Gets the deformation axes of the 'Z' grid on an expanded
        grid ax(i1:i2), ay(j1:j2)
        ier = c_gdgxpndaxes(gdid, ax, ay)
        Proto:
            wordint c_gdgxpndaxes(wordint gdid, ftnfloat *ax, ftnfloat *ay)
        Args:
           ... TODO ...
        Returns:
           int, 0 on success, -1 on error

    c_ezsetival(option, value)
        Sets an integer numerical option for the package (updated 03-2014)
        ier = c_ezsetival('option', value)
        Proto:
           wordint c_ezsetival(char *option, ftnfloat fvalue)
        Args:
           option (str) :
           value  (int) :
        Returns:
           int, 0 on success, -1 on error

    c_ezsetopt(option, value)
        Sets an option for the package (updated 03-2014)
        ier = c_ezsetopt('option', 'value')
        Proto:
           wordint c_ezsetopt(char *option, char *value)
        Args:
           option (str) :
           value  (str) :
        Returns:
        int, 0 on success, -1 on error
    
    c_ezsetval(option, value)
        Sets a floating point numerical option for the package
        ier = c_ezsetval('option', value)
        Proto:
           wordint c_ezsetval(char *option, ftnfloat fvalue)
        Args:
           option (str)   :
           value  (float) :
        Returns:
           int, 0 on success, -1 on error
    
    c_gdrls(gdid)
        Frees a previously allocated grid
        ier = c_gdrls(gdid)
        Proto:
        wordint c_gdrls(wordint gdin)
        Args:
        gid (int) : grid id to be released
        Returns:
        int, 0 on success, -1 on error

   ... TODO ...

"""

import ctypes as _ct
import numpy  as _np
import numpy.ctypeslib as _npc

from . import librmn

## Convert function name with Fortran name mangling
f77name = lambda x: str(x) + '_'
f77name.__doc__ = "Convert function name with Fortran name mangling"

## f77name = lambda x: getattr(, '_'+x)()
## def callMethod(o, name):
##     getattr(o, name)()

class FLOAT_IP(_ct.Structure):
    """
    A structure to hold level values and kind with support for a value range

    FLOAT_IP(v1, v2, kind)

    Args and Attributes:
       v1  : (float) 1st value of the IP
       v2  : (float) 2nd value of the IP
       kind: (int)   IP kind
    
    Examples:
    >>> p  = FLOAT_IP(100., 100., rpnpy.librmn.const.LEVEL_KIND_PMB)
    >>> dp = FLOAT_IP(100., 200., rpnpy.librmn.const.LEVEL_KIND_PMB)

    See Also:
       c_ConvertIp
       c_ConvertIPtoPK
       c_ConvertPKtoIP
       c_EncodeIp
       c_DecodeIp
       c_KindToString
    """
    _fields_ = [("v1", _ct.c_float),
                ("v2", _ct.c_float),
                ("kind", _ct.c_int)]

    def __str__(self):
        return "FLOAT_IP(%f, %f, %d)" % (self.v1, self.v2, self.kind)
    def __repr__(self):
        return "FLOAT_IP(%f, %f, %d)" % (self.v1, self.v2, self.kind)

    def toList(self):
        """
        Returns a tuple with FLOAT_IP's 3 attributes: v1, v2, kind
        
        Returns:
           (v1,v2,kind)
        """
        return (self.v1, self.v2, self.kind)


#--- primitives -----------------------------------------------------

librmn.c_fclos.argtypes = (_ct.c_int, )
librmn.c_fclos.restype  = _ct.c_int
c_fclos = librmn.c_fclos

librmn.c_fnom.argtypes = (_ct.POINTER(_ct.c_int), _ct.c_char_p,
                          _ct.c_char_p, _ct.c_int)
librmn.c_fnom.restype  = _ct.c_int
c_fnom = librmn.c_fnom

librmn.c_wkoffit.argtypes = (_ct.c_char_p, _ct.c_int)
librmn.c_wkoffit.restype  = _ct.c_int
c_wkoffit = librmn.c_wkoffit

librmn.crc32.argtypes = (
    _ct.c_uint,
    _npc.ndpointer(dtype=_np.uint32),
    _ct.c_uint
    )
## librmn.crc32.argtypes = (
##     _ct.c_int,
##     _npc.ndpointer(dtype=_np.uint32),
##     _ct.c_int
##     )
librmn.crc32.restype  = _ct.c_uint
c_crc32 = librmn.crc32

#--- base -----------------------------------------------------------

# Fortran function a provided with a formal interface in librmn_c.c

librmn.cigaxg_.argtypes = (_ct.c_char_p,
                          _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
                          _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
                          _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                          _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
## f_cigaxg = f77name(librmn.cigaxg)
f_cigaxg = librmn.cigaxg_


librmn.cxgaig_.argtypes = (_ct.c_char_p,
                          _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                          _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                          _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
                          _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float))
## f_cxgaig = f77name(librmn.cxgaig)
f_cxgaig = librmn.cxgaig_


librmn.incdati_.argtypes = (_ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                            _ct.POINTER(_ct.c_double))
## f_incdati = f77name(librmn.incdati)
f_incdati = librmn.incdati_


librmn.incdatr_.argtypes = (_ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                            _ct.POINTER(_ct.c_double))
## f_incdatr = f77name(librmn.incdatr)
f_incdatr = librmn.incdatr_


librmn.difdati_.argtypes = (_ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                            _ct.POINTER(_ct.c_double))
## f_difdati = f77name(librmn.difdati)
f_difdati = librmn.difdati_


librmn.difdatr_.argtypes = (_ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                            _ct.POINTER(_ct.c_double))
## f_difdatr = f77name(librmn.difdatr)
f_difdatr = librmn.difdatr_


librmn.newdate_options_.argtypes = (_ct.c_char_p, _ct.c_char_p, _ct.c_int,
                                    _ct.c_int)
## f_newdate_options = f77name(librmn.newdate_options)
f_newdate_options = librmn.newdate_options_


librmn.ignore_leapyear_.argtypes = []
## f_newdate_options = f77name(librmn.ignore_leapyear)
f_ignore_leapyear = librmn.ignore_leapyear_


librmn.accept_leapyear_.argtypes = []
## f_newdate_options = f77name(librmn.accept_leapyear)
f_accept_leapyear = librmn.accept_leapyear_


librmn.newdate_.argtypes = (_ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                            _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
librmn.newdate_.restype  = _ct.c_int
## f_newdate = f77name(librmn.newdate)
f_newdate = librmn.newdate_


#--- fstd98/fstd98 --------------------------------------------------

librmn.c_fstecr.argtypes = (
    _npc.ndpointer(dtype=_np.float32), _npc.ndpointer(dtype=_np.float32),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int)
librmn.c_fstecr.restype  = _ct.c_int
c_fstecr = librmn.c_fstecr


librmn.c_fst_edit_dir.argtypes = (
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
    _ct.c_char_p, _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int
    )
librmn.c_fst_edit_dir.restype  = _ct.c_int
c_fst_edit_dir = librmn.c_fst_edit_dir


librmn.c_fsteff.argtypes = (_ct.c_int, )
librmn.c_fsteff.restype  = _ct.c_int
c_fsteff = librmn.c_fsteff


librmn.c_fstfrm.argtypes = (_ct.c_int, )
librmn.c_fstfrm.restype  = _ct.c_int
c_fstfrm = librmn.c_fstfrm


librmn.c_fstinf.argtypes = (
    _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p
    )
librmn.c_fstinf.restype  = _ct.c_int
c_fstinf = librmn.c_fstinf


librmn.c_fstinfx.argtypes = (
    _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p
    )
librmn.c_fstinfx.restype  = _ct.c_int
c_fstinfx = librmn.c_fstinfx


librmn.c_fstinl.argtypes = (
    _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p,
    _npc.ndpointer(dtype=_np.intc), _ct.POINTER(_ct.c_int), _ct.c_int
    )
librmn.c_fstinl.restype  = _ct.c_int
c_fstinl = librmn.c_fstinl


librmn.c_fstlic.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p,
    _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_char_p
    )
librmn.c_fstlic.restype  = _ct.c_int
c_fstlic = librmn.c_fstlic


librmn.c_fstlir.argtypes = (
    _npc.ndpointer(dtype=_np.float32), _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p
    )
librmn.c_fstlir.restype  = _ct.c_int
c_fstlir = librmn.c_fstlir


librmn.c_fstlirx.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int,
    _ct.c_char_p, _ct.c_char_p
    )
librmn.c_fstlirx.restype  = _ct.c_int
c_fstlirx = librmn.c_fstlirx


librmn.c_fstlis.argtypes = (
    _npc.ndpointer(dtype=_np.float32), _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int)
    )
librmn.c_fstlis.restype  = _ct.c_int
c_fstlis = librmn.c_fstlis


librmn.fstlnk_.argtypes = (_npc.ndpointer(dtype=_np.int32),
                           _ct.POINTER(_ct.c_int))
## f_fstlnk = f77name(librmn.fstlnk)
librmn.fstlnk_.restype  = _ct.c_int
f_fstlnk = librmn.fstlnk_


librmn.c_fstluk.argtypes = (
    _npc.ndpointer(dtype=_np.float32), _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int)
    )
librmn.c_fstluk.restype  = _ct.c_int
c_fstluk = librmn.c_fstluk


librmn.c_fstmsq.argtypes = (
    _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_char_p, _ct.c_int
    )
librmn.c_fstmsq.restype  = _ct.c_int
c_fstmsq = librmn.c_fstmsq


librmn.c_fstnbr.argtypes = (_ct.c_int, )
librmn.c_fstnbr.restype  = _ct.c_int
c_fstnbr = librmn.c_fstnbr


librmn.c_fstnbrv.argtypes = (_ct.c_int, )
librmn.c_fstnbrv.restype  = _ct.c_int
c_fstnbrv = librmn.c_fstnbrv


librmn.c_fstopc.argtypes = (_ct.c_char_p, _ct.c_char_p, _ct.c_int)
librmn.c_fstopc.restype  = _ct.c_int
c_fstopc = librmn.c_fstopc

librmn.c_fstopi.argtypes = (_ct.c_char_p, _ct.c_int, _ct.c_int)
librmn.c_fstopi.restype  = _ct.c_int
c_fstopi = librmn.c_fstopi


librmn.c_fstouv.argtypes = (_ct.c_int, _ct.c_char_p)
librmn.c_fstouv.restype  = _ct.c_int
c_fstouv = librmn.c_fstouv


librmn.c_fstprm.argtypes = (
    _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.c_char_p,
    _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int)
    )
librmn.c_fstprm.restype  = _ct.c_int
c_fstprm = librmn.c_fstprm


librmn.c_fstsui.argtypes = (
    _ct.c_int,
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int)
    )
librmn.c_fstsui.restype  = _ct.c_int
c_fstsui = librmn.c_fstsui


librmn.c_fst_version.argtypes = []
librmn.c_fst_version.restype  = _ct.c_int
c_fst_version = librmn.c_fst_version


librmn.c_fstvoi.argtypes = (_ct.c_int, _ct.c_char_p)
librmn.c_fstvoi.restype  = _ct.c_int
c_fstvoi = librmn.c_fstvoi


librmn.c_ip1_all.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip1_all.restype  = _ct.c_int
c_ip1_all = librmn.c_ip1_all


librmn.c_ip2_all.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip2_all.restype  = _ct.c_int
c_ip2_all = librmn.c_ip2_all


librmn.c_ip3_all.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip3_all.restype  = _ct.c_int
c_ip3_all = librmn.c_ip3_all


librmn.c_ip1_val.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip1_val.restype  = _ct.c_int
c_ip1_val = librmn.c_ip1_val


librmn.c_ip2_val.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip2_val.restype  = _ct.c_int
c_ip2_val = librmn.c_ip2_val


librmn.c_ip3_val.argtypes = (_ct.c_float, _ct.c_int)
librmn.c_ip3_val.restype  = _ct.c_int
c_ip3_val = librmn.c_ip3_val


librmn.ip_is_equal.argtypes = (_ct.c_int, _ct.c_int, _ct.c_int)
librmn.ip_is_equal.restype  = _ct.c_int
c_ip_is_equal = librmn.ip_is_equal


#--- fstd98/convip_plus & convert_ip123 ---------------------------------


librmn.ConvertIp.argtypes = (
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_int), _ct.c_int
    )
c_ConvertIp = librmn.ConvertIp


librmn.ConvertIPtoPK.argtypes = (
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_int),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_int, _ct.c_int
    )
librmn.ConvertIPtoPK.restype  = _ct.c_int
c_ConvertIPtoPK = librmn.ConvertIPtoPK


librmn.ConvertPKtoIP.argtypes = (
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.c_int, _ct.c_float,
    _ct.c_int, _ct.c_float,
    _ct.c_int, _ct.c_float
    )
librmn.ConvertPKtoIP.restype  = _ct.c_int
c_ConvertPKtoIP = librmn.ConvertPKtoIP


librmn.EncodeIp.argtypes = (
    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
    _ct.POINTER(FLOAT_IP), _ct.POINTER(FLOAT_IP), _ct.POINTER(FLOAT_IP)
    )
librmn.EncodeIp.restype  = _ct.c_int
c_EncodeIp = librmn.EncodeIp


#TODO: ctypes struct
librmn.DecodeIp.argtypes = (
    _ct.POINTER(FLOAT_IP), _ct.POINTER(FLOAT_IP), _ct.POINTER(FLOAT_IP),
    _ct.c_int, _ct.c_int, _ct.c_int
    )
c_DecodeIp = librmn.DecodeIp


librmn.KindToString.argtypes = (_ct.c_int, _ct.c_char_p, _ct.c_char_p)
c_KindToString = librmn.KindToString


#--- fstd98/xdf98 ---------------------------------------------------

librmn.c_xdflnk.argtypes = (_npc.ndpointer(dtype=_np.intc), _ct.c_int)
librmn.c_xdflnk.restype  = _ct.c_int
c_xdflnk = librmn.c_xdflnk


#---- interp (ezscint) ----------------------------------------------

librmn.c_ezqkdef.argtypes = (
    _ct.c_int, _ct.c_int, _ct.c_char_p,
    _ct.c_int, _ct.c_int,
    _ct.c_int, _ct.c_int, _ct.c_int
    )
librmn.c_ezqkdef.restype  = _ct.c_int
c_ezqkdef = librmn.c_ezqkdef


librmn.c_ezdefset.argtypes = (_ct.c_int, _ct.c_int)
librmn.c_ezdefset.restype  = _ct.c_int
c_ezdefset = librmn.c_ezdefset


librmn.c_ezsint.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32)
    )
librmn.c_ezsint.restype  = _ct.c_int
c_ezsint = librmn.c_ezsint


librmn.c_ezuvint.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32)
    )
librmn.c_ezuvint.restype  = _ct.c_int
c_ezuvint = librmn.c_ezuvint


librmn.c_ezwdint.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32)
    )
librmn.c_ezwdint.restype  = _ct.c_int
c_ezwdint = librmn.c_ezwdint


librmn.c_gdll.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32)
    )
librmn.c_gdll.restype  = _ct.c_int
c_gdll = librmn.c_gdll


librmn.c_ezsetval.argtypes = (_ct.c_char_p, _ct.c_float)
librmn.c_ezsetval.restype  = _ct.c_int
c_ezsetval = librmn.c_ezsetval


librmn.c_ezsetival.argtypes = (_ct.c_char_p, _ct.c_int)
librmn.c_ezsetival.restype  = _ct.c_int
c_ezsetival = librmn.c_ezsetival


librmn.c_ezgetval.argtypes = (_ct.c_char_p, _ct.POINTER(_ct.c_float))
librmn.c_ezgetval.restype  = _ct.c_int
c_ezgetval = librmn.c_ezgetval


librmn.c_ezgetival.argtypes = (_ct.c_char_p, _ct.POINTER(_ct.c_int))
librmn.c_ezgetival.restype  = _ct.c_int
c_ezgetival = librmn.c_ezgetival


librmn.c_ezsetopt.argtypes = (_ct.c_char_p, _ct.c_char_p)
librmn.c_ezsetopt.restype  = _ct.c_int
c_ezsetopt = librmn.c_ezsetopt


librmn.c_ezgetopt.argtypes = (_ct.c_char_p, _ct.c_char_p)
librmn.c_ezgetopt.restype  = _ct.c_int
c_ezgetopt = librmn.c_ezgetopt


librmn.c_gdrls.argtypes = (_ct.c_int, )
librmn.c_gdrls.restype  = _ct.c_int
c_gdrls = librmn.c_gdrls


librmn.c_gdxyfll.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdxyfll.restype  = _ct.c_int
c_gdxyfll = librmn.c_gdxyfll


librmn.c_gdllfxy.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdllfxy.restype  = _ct.c_int
c_gdllfxy = librmn.c_gdllfxy


librmn.c_gdllsval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdllsval.restype  = _ct.c_int
c_gdllsval = librmn.c_gdllsval


librmn.c_gdxysval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdxysval.restype  = _ct.c_int
c_gdxysval = librmn.c_gdxysval


librmn.c_gdllvval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdllvval.restype  = _ct.c_int
c_gdllvval = librmn.c_gdllvval


librmn.c_gdxyvval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdxyvval.restype  = _ct.c_int
c_gdxyvval = librmn.c_gdxyvval


librmn.c_gdllwdval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdllwdval.restype  = _ct.c_int
c_gdllwdval = librmn.c_gdllwdval



librmn.c_gdxywdval.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdxywdval.restype  = _ct.c_int
c_gdxywdval = librmn.c_gdxywdval


librmn.c_gduvfwd.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gduvfwd.restype  = _ct.c_int
c_gduvfwd = librmn.c_gduvfwd


librmn.c_gdwdfuv.argtypes = (
    _ct.c_int,
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _ct.c_int
    )
librmn.c_gdwdfuv.restype  = _ct.c_int
c_gdwdfuv = librmn.c_gdwdfuv


librmn.c_gdsetmask.argtypes = (_ct.c_int, _npc.ndpointer(dtype=_np.intc))
librmn.c_gdsetmask.restype  = _ct.c_int
c_gdsetmask = librmn.c_gdsetmask


librmn.c_gdgetmask.argtypes = (_ct.c_int, _npc.ndpointer(dtype=_np.intc))
librmn.c_gdgetmask.restype  = _ct.c_int
c_gdgetmask = librmn.c_gdgetmask


#Note: Not in librmn at the moment
##  = librmn.
## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
## """Scalar interpolation, using the mask associated by the
##    function 'gdsetmask' (*** currently not implemented)
## ier = ezsint_m(zout, zin)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


#Note: Not in librmn at the moment
##  = librmn.
## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
## """Vector interpolation, using the mask associated by the
##    function 'gdsetmask' (*** currently not implemented)
## ier = ezuvint_m(uuout, vvout, uuin, vvin)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


librmn.c_ezsint_mdm.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.intc),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.intc)
    )
librmn.c_ezsint_mdm.restype  = _ct.c_int
c_ezsint_mdm = librmn.c_ezsint_mdm


librmn.c_ezuvint_mdm.argtypes = (
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.intc),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.float32),
    _npc.ndpointer(dtype=_np.intc)
    )
librmn.c_ezuvint_mdm.restype  = _ct.c_int
c_ezuvint_mdm = librmn.c_ezuvint_mdm


librmn.c_ezsint_mask.argtypes = (
    _npc.ndpointer(dtype=_np.intc),
    _npc.ndpointer(dtype=_np.intc)
    )
librmn.c_ezsint_mask.restype  = _ct.c_int
c_ezsint_mask = librmn.c_ezsint_mask


librmn.c_ezgdef_fmem.argtypes = (_ct.c_int, _ct.c_int, _ct.c_char_p,
                                 _ct.c_char_p,_ct.c_int, _ct.c_int,
                                 _ct.c_int, _ct.c_int,
                                 _npc.ndpointer(dtype=_np.float32),
                                 _npc.ndpointer(dtype=_np.float32))
librmn.c_ezgdef_fmem.restype  = _ct.c_int
c_ezgdef_fmem = librmn.c_ezgdef_fmem


librmn.c_ezgdef_supergrid.argtypes = (
    _ct.c_int, _ct.c_int, _ct.c_char_p, _ct.c_char_p,
    _ct.c_int, _ct.c_int, _npc.ndpointer(dtype=_np.intc)
    )
librmn.c_ezgdef_supergrid.restype  = _ct.c_int
c_ezgdef_supergrid = librmn.c_ezgdef_supergrid


## librmn.c_ezgdef.argtypes = (_ct.c_int, _ct.c_int, _ct.c_char_p,
##                     _ct.c_char_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
##                     _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float))
## librmn.c_ezgdef.restype  = _ct.c_int
## c_ezgdef = librmn.c_ezgdef


librmn.c_ezgprm.argtypes = (_ct.c_int, _ct.c_char_p,
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
librmn.c_ezgprm.restype  = _ct.c_int
c_ezgprm = librmn.c_ezgprm


librmn.c_ezgxprm.argtypes = (_ct.c_int, _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int),
                    _ct.c_char_p,
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.c_char_p,
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
librmn.c_ezgxprm.restype  = _ct.c_int
c_ezgxprm = librmn.c_ezgxprm


librmn.c_ezgfstp.argtypes = (_ct.c_int,
            _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
            _ct.c_char_p, _ct.c_char_p, _ct.c_char_p,
            _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
            _ct.POINTER(_ct.c_int),
            _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
            _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
librmn.c_ezgfstp.restype  = _ct.c_int
c_ezgfstp = librmn.c_ezgfstp


librmn.c_gdgaxes.argtypes = (_ct.c_int,
                             _npc.ndpointer(dtype=_np.float32),
                             _npc.ndpointer(dtype=_np.float32))
librmn.c_gdgaxes.restype  = _ct.c_int
c_gdgaxes = librmn.c_gdgaxes


librmn.c_ezget_nsubgrids.argtypes = (_ct.c_int, )
librmn.c_ezget_nsubgrids.restype  = _ct.c_int
c_ezget_nsubgrids = librmn.c_ezget_nsubgrids


librmn.c_ezget_subgridids.argtypes = (_ct.c_int, _npc.ndpointer(dtype=_np.intc))
librmn.c_ezget_subgridids.restype  = _ct.c_int
c_ezget_subgridids = librmn.c_ezget_subgridids


librmn.c_gdxpncf.argtypes = (_ct.c_int,
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int),
                    _ct.POINTER(_ct.c_int), _ct.POINTER(_ct.c_int))
librmn.c_gdxpncf.restype  = _ct.c_int
c_gdxpncf = librmn.c_gdxpncf


librmn.c_gdgxpndaxes.argtypes = (_ct.c_int,
                                 _npc.ndpointer(dtype=_np.float32),
                                 _npc.ndpointer(dtype=_np.float32))
librmn.c_gdgxpndaxes.restype  = _ct.c_int
c_gdgxpndaxes = librmn.c_gdgxpndaxes


#Note: Not in librmn at the moment
## librmn.c_gdxpngd.argtypes = (_ct.c_int,
##                     _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float))
## librmn.c_gdxpngd.restype  = _ct.c_int
## c_gdxpngd = librmn.c_gdxpngd
## """Gets the expanded grid.
## ier = c_gdxpngd(gdid, zxpngd, zin)
## Proto:
##    wordint c_gdxpngd(wordint gdin, ftnfloat *zxpnded, ftnfloat *zin)
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Nearest neighbor interpolation from a regular or irregular grid.
## call ez_rgdint_0(zo, px, py, npts, z, ni, j1, j2)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """

 
## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bilinear interpolation from a regular grid,
##    without wrap around the borders.
## call ez_rgdint_1_nw(zo, px, py, npts, z, ni, j1, j2)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bilinear interpolation from a regular grid, with wrap around the borders.
## call ez_rgdint_1_w(zo, px, py, npts, z, ni, j1, j2, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bicubic interpolation from a regular grid,
##    without wrap around the borders.
## call ez_rgdint_3_nw(zo, px, py, npts, z, ni, j1, j2)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bicubic interpolation from a regular grid, with wrap around the borders.
## call ez_rgdint_3_w(zo, px, py, npts, z, ni, j1, j2, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """

 
## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bicubic interpolation from a regular grid,
##    without wrap around the borders.
## call ez_rgdint_3_wnnc(zo, px, py, npts, z, ni, j1, j2, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bicubic interpolation from a gaussian grid,
##    without wrap around the borders.
## call ez_gggdint_nw(zo, px, py, npts, ay, z, i1, i2, j1, j2)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Bicubic interpolation from a gaussian grid, with wrap around the borders.
## call ez_gggdint_w(zo, px, py, npts, ay, z, ni, j1, j2, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Interpolation from a gaussian grid, without wrap around the borders.
## call ez_irgdint_1_nw(zo, px, py, npts, ax, ay, z, ni, nj)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Interpolation from a irregular grid, with wrap around the borders.
## call ez_irgdint_1_w(zo, px, py, npts, ax, ay, z, ni, nj, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Interpolation from a irregular grid, without wrap around the borders.
## ez_irgdint_3_nw(zo, px, py, npts, ax, ay, cx, cy, z, i1, i2, j1, j2)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Interpolation from an irregular grid, with wrap around the borders.
## ez_irgdint_3_w(zo, px, py, npts, ax, ay, cx, cy, z, ni, nj, wrap)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """


## librmn..argtypes = ()
## librmn..restype  = _ct.c_int
##  = librmn.
## """Interpolation from a irregular grid,
##    without wrap around the borders and without the Newton coefficients.
## call ez_irgdint_nws(zo, px, py, npts, ax, ay, z, i1, i2, j1, j2, degree)
## Proto:
## Args:
## Returns:
##    int, 0 on success, -1 on error
## """

#--- fstd98/burp98.c + c_burp_c -------------------------------------




#---- template_utils ------------------------------------------------

#define WB_FORTRAN_REAL 1
#define WB_FORTRAN_INT 2
#define WB_FORTRAN_CHAR 3
#define WB_FORTRAN_BOOL 4
#define WB_OPTION_SET(options, option) (0 .ne. iand(options, option))

#define WB_IS_ARRAY 4096
#define WB_REWRITE_AT_RESTART 2048
#define WB_REWRITE_MANY 1024
#define WB_REWRITE_UNTIL 512
#define WB_REWRITE_NONE 256
#define WB_DEFAULT WB_REWRITE_NONE
#define WB_READ_ONLY_ON_RESTART 128
#define WB_INITIALIZED 64
#define WB_BADVAL 32
#define WB_HAS_RULES 16
#define WB_IS_LOCAL 8
#define WB_CREATED_BY_RESTART 4
#define WB_NOTINITIALIZED 2
#define WB_CREATE_ONLY 1

#define WB_STRICT_DICTIONARY 2
#define WB_ALLOW_DEFINE 1
#define WB_FORBID_DEFINE 0

#define WB_MSG_DEBUG -1
#define WB_MSG_INFO 0
#define WB_MSG_WARN 1
#define WB_MSG_ERROR 2
#define WB_MSG_SEVERE 3
#define WB_MSG_FATAL 4

#define WB_IS_OK(errcode) (errcode >= 0)
#define WB_IS_ERROR(errcode) (errcode < 0)
#define WB_OK 0
#define WB_ERROR -1
#define WB_ERR_NAMETOOLONG -1000
#define WB_ERR_NOTFOUND -1001
#define WB_ERR_READONLY -1002
#define WB_ERR_WRONGTYPE -1003
#define WB_ERR_WRONGDIMENSION -1004
#define WB_ERR_ALLOC -1005
#define WB_ERR_NOTYPE -1006
#define WB_ERR_NOMEM -1007
#define WB_ERR_NOVAL -1008
#define WB_ERR_BADVAL -1009
#define WB_ERR_WRONGSTRING -1010
#define WB_ERR_CKPT -1011
#define WB_ERR_REDEFINE -1012
#define WB_ERR_BIG -1013
#define WB_ERR_SYNTAX -1014
#define WB_ERR_OPTION -1015
#define WB_ERR_READ -1016

#define WB_MAXSTRINGLENGTH 520
#define WB_MAXNAMELENGTH 27

## /* set verbosity level (C callable) */
## int c_wb_verbosity(int level)

## /* create a new whiteboard instance */
##  WhiteBoard *c_wb_new(){

## /* get rid of a whiteboard instance */
## int c_wb_free(WhiteBoard *WB) {

## /*
##   get the data associated with a whiteboard entry 
##   name   : pointer to character string containing name of key
##            (length MUST be supplied in Lname)
##   Type   : character value R/I/L/C , key type  real/inetger/logical/character
##   Ltype  : length in bytes of each key element 4/8 for R/I/L,
##            1->WB_MAXSTRINGLENGTH for character strings
##   value  : pointer to where data is to be returned
##            (everything considered as unsigned bytes)
##   Nval   : number of elements that can be stored into value
##   Lname  : length of key pointed to by name (FORTRAN style)
## */
## int c_wb_get(WhiteBoard *WB, unsigned char *name, char Type,
##              int Ltype, unsigned char *value, int Nval, int Lname){

## /*
##   put entry into whiteboard
##   name   : pointer to character string containing name of key
##           (length MUST be supplied in Lname)
##   Type   : character value R/I/L/C , key type  real/inetger/logical/character
##   Ltype  : length in bytes of each key element 4/8 for R/I/L,
##            1->WB_MAXSTRINGLENGTH for character strings
##   value  : pointer to data asssociated with key
##            (everything considered as unsigned bytes)
##   Nval   : number of elements (0 means a scalar) (1 or more means an array)
##   Options: options associated with name
##   Lname  : length of key pointed to by name (FORTRAN style)
## */
## int c_wb_put(WhiteBoard *WB, unsigned char *name, char Type, int Ltype,
##              unsigned char *value, int Nval, int Options, int Lname){

## /* read a dictionary or user directive file */
## int c_wb_read(WhiteBoard *WB, char *filename, char *package, char *section,
##               int Options, int Lfilename, int Lpackage, int Lsection){

## /*
##  basic whiteboard check/action routine, the Whiteboard "swiss knife"
##   name   : pointer to character string containing name of key
##            (length MUST be supplied in Lname)
##   OptionsMask: options mask to be tested,
##                call Action if OptionsMask&options  is non zero
##   Lname  : length of key pointed to by name (FORTRAN style)
##   printflag: print messages if nonzero
##   Action:  routine to be called if name and OptionsMask &options  is non zero
##   blinddata: pointer to be passed to Action routine as a second argument
##              (first argument is matching line)
## */
## typedef int (*ACTION)(LINE *, void *);
## int c_wb_check(WhiteBoard *WB, unsigned char *name, int OptionsMask,
##                int Lname, int printflag, ACTION Action, void *blinddata ){

## /* write checkpoint file */
## int c_wb_checkpoint(){

## /* read whiteboard checkpoint file */
## int c_wb_reload(){


## /* FORTRAN callable subroutine to get emtadata associated
##    with a whiteboard name */
## wordint f77_name(f_wb_get_meta)(WhiteBoard **WB, unsigned char *name,
##                                 wordint *elementtype, wordint *elementsize,
##                                 wordint *elements, wordint *options,
##                                 F2Cl lname){
##    int Elementtype, Elementsize, Elements;
##    LINE *line;
##    PAGE *page;
##    int Lname=lname;
##    int status;
## /*
##    int Options=0;
## */
##    status = c_wb_lookup(*WB, name, &Elementtype, &Elementsize,
##                         &Elements, &line, &page, 0, Lname);
##    /* no screaming if not found */

## /* get a list of keys matching a name */
## wordint f77_name(f_wb_get_keys)(WhiteBoard **WB, unsigned char *labels,
##                                 wordint *nlabels, unsigned char *name,
##                                 F2Cl llabels, F2Cl lname){
##    int Lname = lname;
##    int status;
##    KEYS keys;

##    keys.UserKeyArray = labels;
##    keys.UserMaxLabels = *nlabels;
##    keys.UserKeyLength = llabels;
##    status = c_wb_check(*WB, name, -1, Lname, message_level<=WB_MSG_INFO,
##                        CopyKeyName, &keys) ;
##    return (status);
## }


# =========================================================================

if __name__ == "__main__":
    print c_fst_version()
    
# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
