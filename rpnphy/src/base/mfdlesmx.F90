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
!**S/R MFSLESMX  -  MIXED PHASE SATURATION VAPOR PRESSURE CALCULATION
!
      SUBROUTINE MFDLESMX(RES,TT,FF,DF,NI,NK)
!
      implicit none
#include <arch_specific.hf>
!
      INTEGER NI,NK
      REAL RES(NI,NK),TT(NI,NK),FF(NI,NK),DF(NI,NK)
!
!Author
!          A. Plante (May 2003), based on FDLESMX from J. Mailhot
!
!Revision
!
!Object
!          To calculate mixed phase saturation vapor pressure
!
!Arguments
!
!          - Output -
! RES      mixed phase saturation vapor pressure
!
!          - Input -
! TT       temperature
! FF       ice fraction
! DF       value of derivative w/r to T for saturation w/r to ice or liquid
! NI       horizontal dimension
! NK       vertical dimension
!
!*
!
include "thermoconsts.inc"
!
      INTEGER I,K
!
!***********************************************************************
!     AUTOMATIC ARRAYS
!***********************************************************************
!
      REAL*8, dimension(NI,NK) :: WORK8
      REAL*8, dimension(NI,NK) :: FOEWAD
      REAL*8, dimension(NI,NK) :: FESID
      REAL*8, dimension(NI,NK) :: FESMXD
!
!***********************************************************************
!
include "dintern.inc"
include "fintern.inc"
!
!MODULES
!
      DO K=1,NK
         DO I=1,NI
            FOEWAD(I,K)=FOEWAF(TT(I,K))
            FESID(I,K) =FESIF(TT(I,K))
         ENDDO
      ENDDO
      CALL VEXP(FOEWAD,FOEWAD,NI*NK)
      CALL VEXP(FESID,FESID,NI*NK)
      DO K=1,NK
         DO I=1,NI
            FOEWAD(I,K)=FOMULT(FOEWAD(I,K))
            FESID(I,K) =FOMULT(FESID(I,K))
            FESMXD(I,K)=FESMXX(FF(I,K),FESID(I,K),FOEWAD(I,K))
         ENDDO
      ENDDO
      DO K=1,NK
      DO I=1,NI
      RES(I,K)=&
      FDLESMXX(TT(I,K),FF(I,K),DF(I,K),FOEWAD(I,K),FESID(I,K),FESMXD(I,K))
      ENDDO
      ENDDO
!
      RETURN
      END
