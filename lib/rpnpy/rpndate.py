#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module RPNDate contains the classes used to manipulate
RPN Standard Files date format
"""
import datetime
import pytz

#import rpnpy_version
import rpnpy.librmn.all as _rmn

class RPNDate(object):
    """
    RPN STD Date representation

    myRPNDate = RPNDate(DATESTAMP)
    myRPNDate = RPNDate(DATESTAMP0, deet=DEET, nstep=NSTEP)
    myRPNDate = RPNDate(YYYYMMDD, HHMMSShh)
    myRPNDate = RPNDate(YYYYMMDD, HHMMSShh, deet=DEET, nstep=NSTEP)
    myRPNDate = RPNDate(myDateTime, deet=DEET, nstep=NSTEP)
    ## myRPNDate = RPNDate(myRPNMeta)

    Args:
        DATESTAMP  : CMC date stamp or RPNDate object [Int]
        DATESTAMP0 : date0 CMC date stamp or RPNDate object [Int]
        DEET       : Time step in Sec
        NSTEP      : Number of steps
        YYYYMMDD   : Visual representation of YYYYMMDD [Int]
        HHMMSShh   : Visual representation of HHMMSShh [Int]
        myDateTime : Instance of Python DateTime class
        myRPNMeta  : Instance RPNMeta with dateo, deet, npas properly set
    Raises:
        TypeError  if parameters are wrong type
        ValueError if myRPNMeta
    Attibutes:
        dateo : Date of origin [CMC date time stamp format]
        datev : Date of validity [CMC date time stamp format]
        stamp : Date of validity [CMC date time stamp format] (alias to datev)
        dt    : Time step [seconds]
        nstep : Step number [int]

    Examples:
    >>> d1 = RPNDate(20030423, 11453500)
    >>> print('# {0}'.format(repr(d1)))
    # RPNDate(20030423, 11453500)
    >>> d2 = RPNDate(d1)
    >>> print('# {0}'.format(repr(d2)))
    # RPNDate(20030423, 11453500)
    >>> d2 += 48
    >>> print('# {0}'.format(repr(d2)))
    # RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)
    >>> print('# {} = {} + {} * {}'.format(d2.datev, d2.dateo, d2.dt, d2.nstep))
    # 307134583 = 307091383 + 3600.0 * 48.0
    >>> print('# {0}'.format(repr(d1-d2)))
    # -48.0
    >>> utc = pytz.timezone("UTC")
    >>> d4 = datetime.datetime(2003, 4, 23, 11, 45, 35, 0, tzinfo=utc)
    >>> d5 = RPNDate(d4)
    >>> print('# {0}'.format(repr(d5)))
    # RPNDate(20030423, 11453500)
    >>> d6 = d5.toDateTime()
    >>> print('# {0}'.format(repr(d6 == d4)))
    # True
    >>> print('# {0}'.format(str(d5)))
    # 20030423.11453500

    See Also:
        RPNDateRange
        rpnpy.librmn.base.newdate
        rpnpy.librmn.base.incdatr
        rpnpy.librmn.base.difdatr
    """

    def __init__(self, mydate, hms=None, dt=None, nstep=None):
        self.__updated = 1
        self.__datev   = 0
        self.__dict__['dateo'] = 0
        self.__dict__['dt']    = 0
        self.__dict__['nstep'] = 0
        if isinstance(mydate, datetime.datetime):
            (yyyy, mo, dd, hh, mn, ss, dummy, dummy2, dummy3) = \
                mydate.utctimetuple()
            cs = int(mydate.microsecond//10000)
            mydate = yyyy*10000+mo*100+dd
            hms = hh*1000000+mn*10000+ss*100+cs
            RPNDate.__init__(self, mydate, hms, dt=dt, nstep=nstep)
        elif isinstance(mydate, RPNDate):
            self.dateo = mydate.dateo
            self.dt  = mydate.dt
            self.nstep = mydate.nstep
        elif not type(mydate) == type(0):
            try:
                RPNDate.__init__(self, mydate.dateo, dt=mydate.deet,
                                 nstep=mydate.npas)
            except:
                raise TypeError('RPNDate: Cannot instanciate with arg of type :'
                                 + str(type(mydate)))
        else:
            if hms is None:
                self.dateo = mydate
            else:
                if not type(hms) == type(0):
                    raise TypeError('RPNDate: arguments should be of type int')
                dummy = 0
                self.dateo = _rmn.newdate(_rmn.NEWDATE_PRINT2STAMP, mydate, hms)
        if not dt is None:
            self.dt = dt
        if not nstep is None:
            self.nstep = nstep
        self.__update(1)


    def __update(self, force=0):
        "Update datev if needed"
        if self.__updated == 0 or force == 1:
            nhours = float(self.dt * self.nstep) / 3600.
            self.__datev = _rmn.incdatr(self.dateo, nhours)
            self.__updated = 1


    def __getattr__(self, name):
        if name in ['datev', 'stamp']:
            self.__update()
            return self.__datev
        return super(RPNDate, self).__getattr__(name)


    def __setattr__(self, name, value):
        if name in ['datev', 'stamp']:
            raise ValueError('RPNDate: Cannot set '+name)
        tmp = super(RPNDate, self).__setattr__(name, value)
        if name in ['dateo', 'dt', 'nstep']:
            self.__update(1)
        return tmp

    def __delattr__(self, name):
        if name in ['datev', 'stamp', 'dateo', 'dt', 'nstep']:
            raise ValueError('RPNDate: Cannot delete '+name)
        return super(RPNDate, self).__delattr__(name)


    def __coerce__(self, other):
        return None


    def __cmp__(self, other):
        if not isinstance(other, RPNDate):
            raise TypeError('RPNDate cannot compare to non RPNDate')
        if self.datev == other.datev:
            return 0
        else:
            return int((self - other)*3600.)


    def __sub__(self, other):
        "Time difference between 2 dates [hours] or Decrease time by nhours"
        if isinstance(other, RPNDate):
            return _rmn.difdatr(self.datev, other.datev)
        elif type(other) == type(1) or type(other) == type(1.0):
            nhours = -other
            mydate = RPNDate(self)
            mydate += nhours
            return mydate
        else:
            raise TypeError('RPNDate: Cannot substract object of type ' +
                            str(type(other)))


    def __add__(self, nhours):
        "Increase time by nhours"
        mydate = RPNDate(self)
        mydate += nhours
        return mydate


    def __isub__(self, nhours):
        nhours2 = -nhours
        ## self += nhours2
        ## return self
        return self.__iadd__(self, nhours2)


    def __iadd__(self, nhours):
        "Increase time by nhours"
        if type(nhours) == type(1) or type(nhours) == type(1.0):
            if self.dt == 0:
                self.dt = 3600.
            # nsteps = float(nhours)*3600. / float(self.dt)
            self.nstep += float(nhours)*3600. / float(self.dt)
            return self
        else:
            raise TypeError('RPNDate: Cannot add object of type ' +
                            str(type(nhours)))


    def update(self, dateo=None, dt=None, nstep=None):
        """
        Update the RPNDate instance attributes

        Args:
            dateo : date0 CMC date stamp or RPNDate object [Int]
            dt    : Time step in Sec
            nstep : Number of steps
        Returns:
            None
        """
        if not dt is None:
            self.dt = dt
        if not nstep is None:
            self.nstep = nstep
        if not dateo is None:
            RPNDate.__init__(self, dateo, dt=dt, nstep=nstep)
        self.__update(1)


    def incr(self, nhours):
        """
        Increase Date by the specified number of hours

        Args:
            nhours : Number of hours for the RPNDate to be increased
        Returns:
            self
        Raises:
            TypeError if nhours is not of int or real type
        """
        self += nhours
        return self


    def toDateTime(self):
        """
        Return the DateTime obj representing the RPNDate

        Returns:
            DateTime obj representing the RPNDate

        Examples:
        >>> myRPNDate = RPNDate(20030423, 11453600)
        >>> myDateTime = myRPNDate.toDateTime()
        >>> myDateTime
        datetime.datetime(2003, 4, 23, 11, 45, 35, tzinfo=<UTC>)

        #TODO: oups 1 sec diff!!!
        """
        ymd = hms = 0
        (ymd, hms) = _rmn.newdate(_rmn.NEWDATE_STAMP2PRINT, self.datev)
        d = "{0:08d}.{1:08d}".format(ymd, hms)
        yyyy = int(d[0:4])
        mo = int(d[4:6])
        dd = int(d[6:8])
        hh = int(d[9:11])
        mn = int(d[11:13])
        ss = int(d[13:15])
        cs = int(d[15:17])
        utc = pytz.timezone("UTC")
        return datetime.datetime(yyyy, mo, dd, hh, mn, ss, cs*10000, tzinfo=utc)


    def toDateO(self):
        """
        Return RPNDate updated so that dateo=datev (dt=nstep=0)

        Returns:
            Self

        Examples:
        >>> d1 = RPNDate(20030423, 11000000, dt=1800, nstep=4)
        >>> print('# {0}'.format(repr(d1)))
        # RPNDate(20030423, 13000000) ; RPNDate(20030423, 11000000, dt=  1800.0, nstep=     4.0)
        >>> d2 = d1.toDateO()
        >>> print('# {0}'.format(repr(d2)))
        # RPNDate(20030423, 13000000)
        """
        RPNDate.update(self, dateo=self.datev, dt=0, nstep=0)
        return self

    def __repr__(self):
        ymd = hms = 0
        (ymd, hms) = _rmn.newdate(_rmn.NEWDATE_STAMP2PRINT, self.datev)
        if self.dt == 0:
            return "RPNDate({0:08d}, {1:08d})".format(ymd, hms)
        else:
            ymd0 = hms0 = 0
            (ymd0, hms0) = _rmn.newdate(_rmn.NEWDATE_STAMP2PRINT, self.dateo)
            return "RPNDate({0:08d}, {1:08d}) ; RPNDate({2:08d}, {3:08d}, dt={4:8.1f}, nstep={5:8.1f})".format(ymd, hms, ymd0, hms0, self.dt, self.nstep)

    def __str__(self):
        ymd = hms = 0
        (ymd, hms) = _rmn.newdate(_rmn.NEWDATE_STAMP2PRINT, self.datev)
        return "{0:08d}.{1:08d}".format(ymd, hms)


#TODO: make dateRange a sequence obj with .iter() methode to be ableto use it in a for statement

class RPNDateRange(object):
    """
    RPN STD Date Range representation

    RPNDateRange(DateStart, DateEnd, Delta)

    Args:
        DateStart RPNDate start of the range
        DateEnd   RPNDate end of the range
        Delta     Increment of the range iterator, hours, real
    Raises:
        TypeError if parameters are wrong type
    Attibutes:
        dateDebut : Range start date
        now       : Present date in the iterator
        delta     : Date iterator increment
        dateFin   : Range end date

    Examples:
    >>> d1 = RPNDate(20030423, 11453500)
    >>> d2 = d1 + 48
    >>> print('# {0}'.format(repr([d1, d2])))
    # [RPNDate(20030423, 11453500), RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)]
    >>> dr = RPNDateRange(d1, d2, 6)
    >>> print('# {0}'.format(repr(dr)))
    # RPNDateRage(from:(20030423, 11453500), to:(20030425, 11453500), delta:6) at (20030423, 11453500)
    >>> l = dr.length() #1
    >>> print('# {0}'.format(repr(l)))
    # 48.0
    >>> x = dr.next() #1
    >>> print('# {0}'.format(repr(x)))
    # RPNDate(20030423, 17453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=     6.0)
    >>> print('# from:{}; to:{}; now:{}; delta:{}'.format(dr.dateDebut.datev, dr.dateFin.datev, dr.now.datev, dr.delta))
    # from:307091383; to:307134583; now:307096783; delta:6
    >>> print('# {0}'.format(repr([d1, d2])))
    # [RPNDate(20030423, 11453500), RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)]
    >>> dr = RPNDateRange(d1, d2, 36)
    >>> print('# {0}'.format(repr(dr)))
    # RPNDateRage(from:(20030423, 11453500), to:(20030425, 11453500), delta:36) at (20030423, 11453500)
    >>> x = dr.next() #2
    >>> print('# {0}'.format(repr(x)))
    # RPNDate(20030424, 23453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    36.0)
    >>> dr.next() #raise
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "rpndate.py", line 313, in next
        raise StopIteration
    StopIteration
    >>> print('# {0}'.format(repr([d1, d2])))  #3 make sure d1, d2 where not changed
    # [RPNDate(20030423, 11453500), RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)]
    >>> dr = RPNDateRange(d1, d2, 12)
    >>> print('# {0}'.format(repr(dr))) #3
    # RPNDateRage(from:(20030423, 11453500), to:(20030425, 11453500), delta:12) at (20030423, 11453500)
    >>> for d4 in dr: print('# {0}'.format(repr(d4)))
    # RPNDate(20030423, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=     0.0)
    # RPNDate(20030423, 23453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    12.0)
    # RPNDate(20030424, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    24.0)
    # RPNDate(20030424, 23453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    36.0)
    # RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)
    >>> for d4 in dr: print('# {0}'.format(repr(d4))) #iterator test 2 (should start over)
    # RPNDate(20030423, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=     0.0)
    # RPNDate(20030423, 23453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    12.0)
    # RPNDate(20030424, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    24.0)
    # RPNDate(20030424, 23453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    36.0)
    # RPNDate(20030425, 11453500) ; RPNDate(20030423, 11453500, dt=  3600.0, nstep=    48.0)

    See Also:
        RPNDate
        rpnpy.librmn.base.newdate
        rpnpy.librmn.base.incdatr
        rpnpy.librmn.base.difdatr
    """
    dateDebut = -1
    dateFin = -1
    delta = 0.0
    now = -1

    def __init__(self, debut=-1, fin=-1, delta=0.0):
        if isinstance(debut, RPNDate) and isinstance(fin, RPNDate) and \
            (type(delta) == type(1) or type(delta) == type(1.0)):
            self.dateDebut = RPNDate(debut)
            self.now       = RPNDate(debut)
            self.dateFin   = RPNDate(fin)
            self.delta     = delta
        else:
            raise TypeError('RPNDateRange: arguments type error RPNDateRange(RPNDate, RPNDate, Real)')

    def length(self):
        """
        Returns the duration of the date range

        Returns:
            Number of hours
        """
        return abs(self.dateFin-self.dateDebut)

    def lenght(self):
        """
        (deprecated, use length) Provide the duration of the date range
        Kept for backward compatibility, please use length()

        Returns:
            Number of hours
        """
        return self.length()

    def remains(self):
        """
        Provide the number of hours left in the date range

        Returns:
            Number of hours left in the range
        """
        return abs(self.dateFin-self.now)

    def next(self):
        """
        Return the next date/time in the range (step of delta hours)

        Returns:
            next RPNDate, None if next date is beyond range
        """
        self.now.incr(self.delta)
        if (self.dateFin-self.now)*self.delta < 0.:
            raise StopIteration
        return RPNDate(self.now)

    def reset(self):
        """
        Reset the RPNDateRange iterator to the range start date

        Returns:
            None
        """
        self.now = self.dateDebut

    def __repr__(self):
        d1 = str(self.dateDebut).replace('.', ', ')
        d2 = str(self.dateFin).replace('.', ', ')
        d0 = str(self.now).replace('.', ', ')
        return "RPNDateRage(from:({0}), to:({1}), delta:{2}) at ({3})". \
            format(d1, d2, self.delta, d0)

    def __iter__(self):
        tmp = RPNDateRange(self.dateDebut, self.dateFin, self.delta)
        tmp.now = tmp.now - tmp.delta
        return tmp

    def __next__(self):
        return self.next()

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
