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
!**S/P MIXLEN3
!
!
      SUBROUTINE MIXLEN3( ZN, th, EN, zz, H, SIGMA, PS, N, NK)
!
!
      implicit none
#include <arch_specific.hf>
!
      INTEGER N,NK
!
      REAL ZN(N,NK), th(N,NK)
      REAL EN(N,NK), zz(N,NK)
      REAL SIGMA(N,NK), PS(N)
      REAL H(N)
!
!
!
!Author
!          S. Belair (November 1996)
!
!Revision
! 001      J. Mailhot (July 1999) - version using the virtual potential
!                                   temperature; change name to MIXLEN1
! 002      J. Mailhot (Sept 1999) - clipping of RIF maximum for computation of ZE
! 003      S. Belair (Oct 1999)   - staggerred values for the virtual
!                                   potential temperature and the heights
! 004      J. Mailhot (July 2000) - correct limits for solution of quadratic eqn.
! 005      J. Mailhot (Aug 2000) - add relaxation option (RELAX = .T. or .F.)
! 006      S. Belair, J. Mailhot (March 2001)
!                                  blend between local (i.e.,
!                                  Bougeault-Lacarrere) and
!                                  background (i.e., input) mixing and
!                                  dissipation lengths
! 007      A-M Leduc  (Oct 2001)   - Automatic arrays
! 008      J. Mailhot (May 2002) - restrict local mixing to convective case
! 009      S. Belair, J. Mailhot (June 2002) - use fixed heights for blend
!                                              and remove stability considerations
! 010      S. Belair (Jan 2003)   -reorganization and modification of bougeault
!                                   mixlen1--->mixlen2
! 011      B. Bilodeau (Aug 2003) - IBM conversion (scalar version)
! 012      S. Belair (March 2004) - Relax the mixing length towards the
!                                   Blackadar value in the upper troposphere
!                                   (i.e., between plow and phigh)
! 002      L. Spacek (Dec 2007)   - all calculations on energy levels
!
!          C. Girard              - memory optimization
!
!Object
!           Calculates the mixing length ZN and the dissipation
!           length ZE based on the Bougeault and Lacarrere method.
!
!Arguments
!                        -Output-
!
! ZN        mixing length
!
!                         -Input-
!
! ZN        mixing length at t- (only if RELAX = .TRUE.)
! th        virtual potential temperature
! EN        turbulent kinetic energy
! zz        height of the sigma levels
! H         height of the boundary layer (unused)
! N         horizontal dimension
! NK        vertical dimension
!
!
      real, parameter :: ZMIX=500.,PLOW=550E2,PHIGH=450E2

      integer :: j,k,ki,kf,kk,stat
      integer, dimension(n) :: slk
      real :: gravinv,recbeta,enlocal
      real :: pres,delthk,slope,delen,delzup,delzdown,buoy,buoysum,znblac
      real, dimension(nk) ::lup,ldown

      integer, external :: neark
!
!***********************************************************************
!
include "thermoconsts.inc"
include "dintern.inc"
include "fintern.inc"

      GRAVINV = 1./GRAV
      stat = neark(sigma,ps,1000.,n,nk,slk) !determine "surface layer" vertical index for buoyancy coefficient

      DO J=1,N
!
!                              surface buoyancy term BETA
!
      RECBETA = th(J,slk(j))*GRAVINV
!
      DO KI=2,NK-1
!
        ENLOCAL = MIN( EN(J,KI), 4. )
!
!                      --------- FIND THE UPWARD MIXING LENGTH
!                                (LUP)

        KF=2
        buoy=0.0
        DO K=KI,2,-1
          buoysum=buoy
          buoy=buoy + 0.5*( zz(J,K-1)-zz(J,K) ) * &
           ( th(J,K-1) + th(J,K) - 2.*th(J,KI) )
          IF (buoy.GT.ENLOCAL*RECBETA) THEN
             KF = K
             GOTO 10
          ENDIF
        END DO
!
10      LUP(KI)    = zz(J,KF)-zz(J,KI)
        DELTHK     = th(J,KF)-th(J,KI)
        SLOPE      = (th(J,KF-1)-th(J,KF))/(zz(J,KF-1)-zz(J,KF))
        SLOPE      = MAX( SLOPE, 1.E-6 )
        DELEN      = ENLOCAL*RECBETA - buoysum
        DELZUP     = -DELTHK+SQRT(MAX(0.0,DELTHK*DELTHK+2.*SLOPE*DELEN))
        DELZUP     = DELZUP / SLOPE     
        LUP(KI)    = MAX( LUP(KI) + DELZUP , 1. )
!
!
!                             Same work but for the downward
!                             free path
!
        KF=NK-1
        buoy=0.0
        DO K=KI,NK-1
          buoysum=buoy
          buoy=buoy + 0.5*( zz(J,K) - zz(J,K+1) ) * &
               ( 2.*th(J,KI) - th(J,K) - th(J,K+1) )
          IF (buoy.GT.ENLOCAL*RECBETA) THEN
             KF = K
             GO TO 11
          ENDIF
        END DO
!
11      LDOWN(KI)   = zz(J,KI) - zz(J,KF)
        DELTHK      = th(J,KI) - th(J,KF)
        SLOPE       = (th(J,KF)-th(J,KF+1))/(zz(J,KF)-zz(J,KF+1))
        SLOPE       = MAX( SLOPE , 1.E-6 )
        DELEN       = ENLOCAL*RECBETA - buoysum
        DELZDOWN    = -DELTHK+SQRT(MAX(0.0,DELTHK*DELTHK+2.*SLOPE*DELEN))
        DELZDOWN    = DELZDOWN / SLOPE     
        LDOWN(KI)   = LDOWN(KI) + DELZDOWN     
        LDOWN(KI)   = MIN( LDOWN(KI), zz(J,KI) )
        LDOWN(KI)   = MAX( LDOWN(KI), 1. )
!
      END DO
!
      DO K=1,NK
      KK=min(NK-1,max(2,K))
!
!                            Calculate the mixing length ZN
!                            and the dissipation length from the
!                            LUP and LDOWN results
!
        znblac=ZN(J,K)
!
        ZN(J,K) = MIN( LUP(KK), LDOWN(KK) )
        ZN(J,K) = MIN(  ZN(J,K), zz(J,K) )
!
!
!                            Blending of the mixing and dissipation lengths
!                            between the local values (i.e., Bougeaut-
!                            Lacarrere calculations) and background values
!                            (i.e., from input arguments)
!                            Restrict local mixing to convective case
!                            This blending is done near the surface (below
!                            ZMIX), and in the upper troposphere (above PHIGH,
!                            with a linear transition between PLOW and PHIGH).
!
!
        IF ( zz(J,K).LT.ZMIX ) &
          ZN(J,K) = ZNBLAC+zz(J,K)/ZMIX*(ZN(J,K)-ZNBLAC)
!
        PRES   = SIGMA(J,K) * PS(J)
        ZN(J,K) = ZNBLAC + &
                  ( MIN( MAX( PRES, PHIGH ) , PLOW ) - PHIGH ) &
                * ( ZN(J,K) - ZNBLAC ) / ( PLOW - PHIGH )
!
        ZN(J,K) = MAX( ZN(J,K), 1.E-6 )
!
      END DO
!
      END DO
!
!
      RETURN
      END
