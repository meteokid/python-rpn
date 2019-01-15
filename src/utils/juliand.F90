!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
      REAL FUNCTION JULIAND( DEET,NPAS,IDATIM )
!
      implicit none
#include <arch_specific.hf>
!
      INTEGER IDATIM(14),NPAS
      REAL DEET
!
!Author
!            B. Bilodeau - may 1993
!
!Revision
! 001        B. Bilodeau - June 1998 - Newdate
! 002        B. Dugas    - June 2000 - Calculate NDAYS in REAL*8
! 003        B. Dugas    - July 2010 - Replace JDATEC,DATEC by incdatr,difdatr in
!                                      order to support a "no leap year" option
!
!Object
!            This function calculates the number of days
!            since the beginning of the current year,
!            using the analysis date, the number of time
!            steps since that date and the length of
!            the time step.
!
!            The calculations are precise to whitin
!            half a day. The returned real value has
!            a null fractional part.
!
!Arguments
!
!            - Input -
! DEET       Length of the time step in seconds
! NPAS       Number of time steps since the beginning of the
!            time integration.
! IDATIM     Valid CMC date time group (initial date of the run)
!
!
!
      integer dat1,dat2, ier, is1,is2, JoursA
      integer DebutAnnee,AujourdHui,AnneeSuivante
      real(8) Heures,HeuresC,HeuresA

      integer  NEWDATE
      external NEWDATE,difdatr,incdatr

!     find current date
      Heures = int( (DEET*DBLE( NPAS ))/3600._8 )
      call incdatr( AujourdHui,idatim(14),Heures )

!     convert current date to a printable format
      ier = newdate( AujourdHui, is1,is2, -3 )

!     dat1,dat2 correspond to 00Z Jan 01 of the current year
      dat1 = ((is1/10000)*100+1)*100+1 ; dat2 = 0

      if (is1 == dat1 .and. is2 == dat2) then
         JULIAND = 1
         return
      endif

!     CMC date time group of Jan 01 of the current year
      ier = newdate( DebutAnnee, dat1,dat2, +3 )

!     number of hours since the beginning of the current year
      call difdatr( AujourdHui,DebutAnnee,HeuresC )

!     dat1,dat2 now correspond to 00Z Jan 01 of the next year
      dat1 = (((is1/10000)+1)*100+1)*100+1 ; dat2 = 0

!     CMC date time group of Jan 01 of the next year
      ier = newdate( AnneeSuivante, dat1,dat2, +3 )

      call difdatr( AnneeSuivante,DebutAnnee,HeuresA )

!     number of days in the current year+1
      JoursA =  nint( HeuresA/24 )

!     ACTUAL DAY OF THE YEAR
      JULIAND = nint( HeuresC/24 )+1
      if (JULIAND > JoursA) JULIAND = 1
!
      RETURN
      END
