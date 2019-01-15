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
!** S/P BLCLOUD3
!
      SUBROUTINE BLCLOUD3 (U, V, T, TVE, QV, QC, FNN, &
                           S, SW, PS, DUDZ2, RI, DTHV, &
                           AT2M, AT2E, &
                           N, M, NK)
!
      implicit none
#include <arch_specific.hf>
!
!
      INTEGER N, M, NK
      REAL U(M,NK), V(M,NK), T(M,NK), TVE(N,NK), QV(M,NK)
      REAL QC(N,NK), FNN(N,NK), S(N,NK), SW(N,NK), PS(N)
      REAL DUDZ2(N,NK), RI(N,NK), DTHV(N,NK)
      REAL AT2M(N,NK), AT2E(N,NK)
!
!Author
!          J. Mailhot (Nov 2000)
!
!Revision
! 001      A.-M. Leduc (Oct 2001) Automatic arrays
! 002      J. Mailhot (Jun 2002) Change calling sequence and rename BLCLOUD1
! 003      J. Mailhot (Feb 2003) Change calling sequence and rename BLCLOUD2
! 004      L. Spacek (Dec 2007) - add "vertical staggering" option
! 005                             change the name to blcloud3

!Object
!          Calculate the boundary layer buoyancy parameters (virtual potential
!          temperature, buoyancy flux) and the vertical shear squared.
!
!Arguments
!
!          - Input -
! U        east-west component of wind
! V        north-south component of wind
! T        temperature (thetal when ISTEP=2)
! TVE      virtual temperature on 'E' levels
! QV       specific humidity (total water content QW = QV + QC)
! QC       boundary layer cloud water content
! FNN      flux enhancement factor (fN) * cloud fraction (N)
! S        sigma levels
! PS       surface pressure (in Pa)
! AT2M     coefficients for interpolation of T,Q to momentum levels
! AT2E     coefficients for interpolation of T,Q to energy levels
!
!
!          - Output -
! DUDZ2    vertical shear of the wind squared (on 'E' levels)
! RI       Richardson number - in gradient form - (on 'E' levels)
! DTHV     buoyancy flux term - in gradient form - (on 'E' levels)
!
!          - Input -
! N        horizontal dimension
! M        first dimension of U,V, T and QV
! NK       vertical dimension
!
!
!Notes
!          Implicit (i.e. subgrid-scale) cloudiness scheme for unified
!             description of stratiform and shallow, nonprecipitating
!             cumulus convection appropriate for a low-order turbulence
!             model based on Bechtold et al.:
!            - Bechtold and Siebesma 1998, JAS 55, 888-895
!            - Cuijpers and Bechtold 1995, JAS 52, 2486-2490
!            - Bechtold et al. 1995, JAS 52, 455-463
!            - Bechtold et al. 1992, JAS 49, 1723-1744
!          The boundary layer cloud properties (cloud fraction, cloud water
!            content) are computed in the companion S/R CLSGS.
!
!
!IMPLICITS
!
include "thermoconsts.inc"
!
!*
!
      INTEGER J, K, ITOTAL
!
!
!
!*********************************************************
!     AUTOMATIC ARRAYS
!*********************************************************

      REAL, dimension(N,NK) :: THL
      REAL, dimension(N,NK) :: QW
      REAL, dimension(N,NK) :: ALPHA
      REAL, dimension(N,NK) :: BETA
      REAL, dimension(N,NK) :: A
      REAL, dimension(N,NK) :: B
      REAL, dimension(N,NK) :: C
      REAL, dimension(N,NK) :: DZ
      REAL, dimension(N,NK) :: DQWDZ
      REAL, dimension(N,NK) :: DTHLDZ
      REAL, dimension(N,NK) :: COEFTHL
      REAL, dimension(N,NK) :: COEFQW
      REAL, dimension(N,NK) :: FICELOCAL
!
!*********************************************************
!
!
!MODULES
!
      EXTERNAL DVRTDF,THERMCO2,FICEMXP
!
!------------------------------------------------------------------------
!
!
!
!       0.     Preliminaries
!       --------------------
!
!
      CALL FICEMXP(FICELOCAL,A,B,T,N,N,NK)
!
!
!       1.     Thermodynamic coefficients
!       ---------------------------------
!
!
      CALL THERMCO2 (T, QV, QC, SW, PS, T, FICELOCAL, FNN, &
                     THL, QW, A, B, C, ALPHA, BETA, &
                     0, .TRUE., N, M, NK)
!
!       2.     The buoyant parameters,
!                                             (cf. BS 1998 eq. 4)
      DO K=1,NK
      DO J=1,N
!                                              put thv in DUDZ2 temporarily
        DUDZ2(J,K) = THL(J,K) + ALPHA(J,K)*QW(J,K) + BETA(J,K)*QC(J,K)
        COEFTHL(J,K) = 1.0 + DELTA*QW(J,K) &
                                 - BETA(J,K)*B(J,K)*FNN(J,K)
        COEFQW(J,K) = ALPHA(J,K) + BETA(J,K)*A(J,K)*FNN(J,K)
      END DO
      END DO
!
!
!
!
!       3.     Vertical derivative of THL and QW
!       ----------------------------------------
!
      DO K=1,NK-1
      DO J=1,N
        DZ(J,K) = -RGASD*TVE(J,K)*ALOG( S(J,K+1)/S(J,K) ) / GRAV
      END DO
      END DO
!
      DO J=1,N
        DZ(J,NK) = 0.0
      END DO
!
      CALL TOTHERMO(THL, THL, AT2M,AT2M,N,NK+1,NK,.false.)
      CALL TOTHERMO(QW,  QW,  AT2M,AT2M,N,NK+1,NK,.false.)

      CALL DVRTDF ( DTHLDZ, THL, DZ, N, N, N, NK)
      CALL DVRTDF ( DQWDZ, QW, DZ, N, N, N, NK)
!
!
!       4.     The buoyancy flux and vertical shear squared
!       -----------------------------------------------------------------------
!
!
!
      CALL TOTHERMO(COEFTHL, COEFTHL, AT2E,AT2E,N,NK+1,NK,.true.)
      CALL TOTHERMO(COEFQW,  COEFQW,  AT2E,AT2E,N,NK+1,NK,.true.)
      CALL TOTHERMO(DUDZ2,   DUDZ2,   AT2E,AT2E,N,NK+1,NK,.true.)
!
      DO K=1,NK-1
      DO J=1,N
!                                              coefficients on 'E' levels
        DTHV(J,K) = ( COEFTHL(J,K)*DTHLDZ(J,K) &
                    + COEFQW(J,K)*DQWDZ(J,K) ) &
                  * ( GRAV / DUDZ2(J,K) )
      END DO
      END DO
!
      DO J=1,N
        DTHV(J,NK) = 0.0
      END DO
!
!                                              vertical shear squared
      CALL DVRTDF ( A, U, DZ, N, N, M, NK)
      CALL DVRTDF ( B, V, DZ, N, N, M, NK)
!
      DO K=1,NK
      DO J=1,N
        DUDZ2(J,K) = A(J,K)**2 + B(J,K)**2
        RI(J,K) = DTHV(J,K)/(DUDZ2(J,K)+1.E-6)
      END DO
      END DO
!
      DO J=1,N
        RI(J,NK) = 0.0
      END DO
!
!
      RETURN
      END
