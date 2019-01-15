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
!** S/P SURF_PRECIP3
!
      SUBROUTINE SURF_PRECIP3 ( T, TLC, TLS, TLCS, TSC, TSS, &
                        TSCS, RAINRATE, SNOWRATE, FNEIGE, FIP,N)
      use phy_options
      implicit none
#include <arch_specific.hf>
!
      INTEGER N
      REAL T(N), TLC(N), TLS(N), TSC(N), TSS(N)
      REAL TLCS(N), TSCS(N), RAINRATE(N), SNOWRATE(N)
      REAL FNEIGE(N),FIP(N)
!
!Author
!          S. Belair (November 1998)
!Revisions
! 001      B. Bilodeau (January 2001)
!          Automatic arrays
! 002      B. Bilodeau and S. Belair (July 2001)
!          Change conditions for option 0
! 003      S. Belair (February 2004)
!          Bug correction related to partition of total
!          precipitation rate into liquid and solid.
! 004      L. Spacek (Aug 2004) - cloud clean-up
!          elimination of ISTCOND=2,6,7,8 ICONVEC=4
! 005      D. Talbot (March 2005)
!          Modification to the previous correction
!          so that it does not apply to consun
! 006      A-M.  Leduc (April 2006). Add TLCS and TSCS to the TOTRATE.
!                      Change the option 1 to use FNEIGE. CHANGE name to
!                      SURF_PRECIP2.
! 007     A-M. Leduc. A. Plante. (Nov 2007) add FIP. change s/r name to surf_precip3
!                change calculation of snowrate to include all solid precipitation
!                with f_solid.
!
!Object
!          Determine the phase of precipitation
!          reaching the surface
!
!Arguments
!
!          - Input -
! T        low-level air temperature
! TLC      total liquid "convective" precipitation rate
! TLS      total liquid "stratiform" precipitation rate
! TLCS     total liquid "shallow convective" precipitation rate
! TSC      total solid  "convective" precipitation rate
! TSS      total solid  "stratiform" precipitation rate
! TSCS     total solid  "shallow convective" precipitation rate
! FNEIGE   snow fraction from Bourge1 subroutine
! FIP      Fraction of liquid precipitation in the form of ice pellets.
!
!          - Output -
! RAINRATE rate of precipitation at the surface (liquid)
! SNOWRATE rate of precipitation at the surface (solid)
!
!
include "thermoconsts.inc"
!
!
      INTEGER I, OPTION
      REAL F_SOLID
!
!
!
!MODULES
!
!***********************************************************************
!     AUTOMATIC ARRAYS
!***********************************************************************
!
      REAL, dimension(N) :: TOTRATE
      REAL, dimension(N) :: EXPLI
!
!***********************************************************************
!
!
!
!
!
!                    Sum of the precipitation rate at
!                    the surface
!
      DO I=1,N
        RAINRATE(I) = 0.
        SNOWRATE(I) = 0.
        TOTRATE (I) = TLC(I)+TLS(I)+TLCS(I)+TSC(I)+TSS(I)+TSCS(I)
      END DO
!
!
!
!                    OPTIONS:  Depending on the explicit scheme
!                              used during the integration,
!                              we have three options for specifying
!                              the phase of precipitation reaching
!                              the ground
!
      if ((stcond=='NIL') .or. (stcond=='CONDS')) then
        OPTION=0
      else if ((stcond=='NEWSUND') .or. (stcond=='CONSUN')) then
        OPTION=1
      else
        OPTION=2
      endif
!
!
!
      IF (OPTION.EQ.0) THEN
!
!                    FIRST OPTION:
!                    The phase of the precipitation reaching
!                    the surface is simply determined from
!                    the low-level air temperature:
!
!                     T > 0.  then rain
!                     T < 0.  then snow
!
        DO I=1,N
          IF (T(I).GE.TCDK) THEN
            RAINRATE(I) = TOTRATE(I)
            SNOWRATE(I) = 0.
          ELSE IF (T(I).LT.TCDK) THEN
            RAINRATE(I) = 0.
            SNOWRATE(I) = TOTRATE(I)
          END IF
        END DO
!
!
!
      ELSE IF (OPTION.EQ.1) THEN
!
!                    SECOND OPTION:
!                    The phase of precipitation at the surface
!                    is determined using the bourge snow fraction
!                    fneige and ice pellet fraction fip.
!
        DO I=1,N
!
           F_SOLID     = FNEIGE(I) + ( 1.-FNEIGE(I) )* FIP(I)
           SNOWRATE(I) = TOTRATE(I) * F_SOLID
           RAINRATE(I) = TOTRATE(I) * (1.-F_SOLID)
!
        END DO
!
!
      ELSE
!
!                    THIRD OPTION:
!                    The phase of precipitation at the surface
!                    is determined by results from the "explicit"
!                    condensation scheme (MIXED-PHASE, EXMO, KONG-YAU).
!                    It was noted in the LAM2.5 that KONG-YAU
!                    could produce a non zero TSS over Quebec in summer
!                    and therefore OPTION 1 was no longer appropriate
!
        DO I=1,N
          EXPLI(I) = TLS(I) + TSS(I)
!
!
!                    If "stratiform" (stable condensation) is greater
!                    than 0, then the explicit scheme determines if
!                    the precipitation is rain or snow
!
          IF (EXPLI(I).GE.1.E-10) THEN
            RAINRATE(I) = TLS(I)/EXPLI(I) * TOTRATE(I)
            SNOWRATE(I) = TSS(I)/EXPLI(I) * TOTRATE(I)
!
!                    If "stratiform" precipitation is null, then
!                    we use low-level air temperature to specify
!                    the phase of precip.
!
          ELSE IF (T(I).GE.TCDK) THEN
            RAINRATE(I) = TOTRATE(I)
            SNOWRATE(I) = 0.0
          ELSE IF (T(I).LT.TCDK) THEN
            RAINRATE(I) = 0.
            SNOWRATE(I) = TOTRATE(I)
          END IF
!
!
        END DO
!
!
      END IF
!
!
!
!
      RETURN
      END
