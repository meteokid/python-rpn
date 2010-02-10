"""Module RPNDate contains the classes used to manipulate RPN Standard Files date format

    @author: Mario Lepine <mario.lepine@ec.gc.ca>
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import datetime
import pytz

import rpn_version
import Fstdc

from rpnmeta import RPNMeta

class RPNDate:
    """RPN STD Date representation

    myRPNDate = RPNDate(DATESTAMP)
    myRPNDate = RPNDate(YYYYMMDD,HHMMSShh)
    myRPNDate = RPNDate(myDateTime)
    myRPNDate = RPNDate(myRPNMeta)

    @param DATESTAMP CMC date stamp or RPNDate object
    @param YYYYMMDD  Int with Visual representation of YYYYMMDD
    @param HHMMSShh  Int with Visual representation of HHMMSShh
    @param myDateTime Instance of Python DateTime class
    @param myRPNMeta Instance RPNMeta with dateo,deet,npas properly set
    @exception TypeError if parameters are wrong type
    @exception ValueError if myRPNMeta

    >>> d1 = RPNDate(20030423,11453500)
    >>> d1
    RPNDate(20030423,11453500)
    >>> d2 = RPNDate(d1)
    >>> d2
    RPNDate(20030423,11453500)
    >>> d2.incr(48)
    RPNDate(20030425,11453500)
    >>> d1-d2
    -48.0
    >>> a = RPNMeta(dateo=d1.stamp,deet=1800,npas=3)
    >>> d3 = RPNDate(a)
    >>> d3
    RPNDate(20030423,13153500)
    >>> utc = pytz.timezone("UTC")
    >>> d4 = datetime.datetime(2003,04,23,11,45,35,0,tzinfo=utc)
    >>> d5 = RPNDate(d4)
    >>> d5
    RPNDate(20030423,11453500)
    >>> d6 = d5.toDateTime()
    >>> d6 == d4
    True
    """
    stamp = 0

    def __init__(self,word1,word2=-1):
        if isinstance(word1,datetime.datetime):
            (yyyy,mo,dd,hh,mn,ss,dummy,dummy2,dummy3) = word1.utctimetuple()
            cs = int(word1.microsecond/10000)
            word1 = yyyy*10000+mo*100+dd
            word2 = hh*1000000+mn*10000+ss*100+cs
        if isinstance(word1,RPNDate):
            self.stamp = word1.stamp
        elif isinstance(word1,RPNMeta):
            if word1.deet<0 or word1.npas<0 or word1.dateo<=0 :
                raise ValueError, 'RPNDate: Cannot compute date from RPNMeta'
            nhours = (1.*word1.deet*word1.npas)/3600.
            self.stamp=Fstdc.incdatr(word1.dateo,nhours)
        elif type(word1) == type(0):
            if (word2 == -1):
                self.stamp = word1
            else:
                dummy=0
#TODO: define named Cst for newdate in Fstdc
                (self.stamp,dummy1,dummy2) = Fstdc.newdate(dummy,word1,word2,3)
        else:
            raise TypeError, 'RPNDate: arguments should be of type int'

    def __sub__(self,other):
        "Time difference between 2 dates"
        return(Fstdc.difdatr(self.stamp,other.stamp))

    def incr(self,temps):
        """Increase Date by the specified number of hours

        @param temps Number of hours for the RPNDate to be increased
        @return self

        @exception TypeError if temps is not of int or real type
        """
        if ((type(temps) == type(1)) or (type(temps) == type(1.0))):
            nhours = 0.0
            nhours = temps
            self.stamp=Fstdc.incdatr(self.stamp,nhours)
            return(self)
        else:
            raise TypeError,'RPNDate.incr: argument should be int or real'

    def toDateTime(self):
        """Return the DateTime obj representing the RPNDate

        >>> myRPNDate = RPNDate(20030423,11453600)
        >>> myDateTime = myRPNDate.toDateTime()
        >>> myDateTime
        datetime.datetime(2003, 4, 23, 11, 45, 35, tzinfo=<UTC>)

        #TODO: oups 1 sec diff!!!
        """
        word1 = word2 = 0
        (dummy,word1,word2) = Fstdc.newdate(self.stamp,word1,word2,-3)
        d = "%8.8d.%8.8d" % (word1, word2)
        yyyy = int(d[0:4])
        mo = int(d[4:6])
        dd = int(d[6:8])
        hh = int(d[9:11])
        mn = int(d[11:13])
        ss = int(d[13:15])
        cs = int(d[15:17])
        utc = pytz.timezone("UTC")
        return datetime.datetime(yyyy,mo,dd,hh,mn,ss,cs*10000,tzinfo=utc)

    def __repr__(self):
        word1 = word2 = 0
        (dummy,word1,word2) = Fstdc.newdate(self.stamp,word1,word2,-3)
        return "RPNDate(%8.8d,%8.8d)" % (word1, word2)


#TODO: make dateRange a sequence obj with .iter() methode to be ableto use it in a for statement

class RPNDateRange:
    """RPN STD Date Range representation

    RPNDateRange(DateStart,DateEnd,Delta)
    @param DateStart RPNDate start of the range
    @param DateEnd   RPNDate end of the range
    @param Delta     Increment of the range iterator, hours, real

    @exception TypeError if parameters are wrong type

    >>> d1 = RPNDate(20030423,11453500)
    >>> d2 = RPNDate(d1)
    >>> d2.incr(48)
    RPNDate(20030425,11453500)
    >>> dr = RPNDateRange(d1,d2,6)
    >>> dr
    RPNDateRage(from:(20030423,11453500), to:(20030425,11453500), delta:6) at (20030423,11453500)
    >>> dr.lenght()
    48.0
    >>> dr.next()
    RPNDate(20030423,17453500)
    >>> dr = RPNDateRange(d1,d2,36)
    >>> dr
    RPNDateRage(from:(20030423,17453500), to:(20030425,11453500), delta:36) at (20030423,17453500)
    >>> dr.next()
    RPNDate(20030425,05453500)
    >>> dr.next() #returns None because it is past the end of DateRange
    """
    #TODO: make this an iterator
    dateDebut=-1
    dateFin=-1
    delta=0.0
    now=-1

    def __init__(self,debut=-1,fin=-1,delta=0.0):
        if isinstance(debut,RPNDate) and isinstance(fin,RPNDate) and ((type(delta) == type(1)) or (type(delta) == type(1.0))):
            self.dateDebut=debut
            self.now=debut
            self.dateFin=fin
            self.delta=delta
        else:
            raise TypeError,'RPNDateRange: arguments type error RPNDateRange(RPNDate,RPNDate,Real)'

    def lenght(self):
        """Provide the duration of the date range
        @return Number of hours
        """
        return abs(self.dateFin-self.dateDebut)

    def remains():
        """Provide the number of hours left in the date range
        @return Number of hours left in the range
        """
        return abs(self.dateFin-self.now)

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
        self.now=self.dateDebut

    def __repr__(self):
        d1 = repr(self.dateDebut)
        d2 = repr(self.dateFin)
        d0 = repr(self.now)
        return "RPNDateRage(from:%s, to:%s, delta:%d) at %s" % (d1[7:27],d2[7:27],self.delta,d0[7:27])


if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
