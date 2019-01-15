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
!**S/P pbl_atmflux
!
!
      SUBROUTINE pbl_atmflux (R,VAR,TVAR,KCOEF,GAMMA, &
                              ALPHA,BETA,PS,TSTAG,QSTAG,TAU,SG,SELOC, &
                              C,D,TYPVAR,N,NK,TRNCH)
!
!
      implicit none
#include <arch_specific.hf>
!
      INTEGER N,NK,TRNCH
      REAL R(N,NK+1),VAR(N,NK),TVAR(N,NK),KCOEF(N,NK),TX
      REAL GAMMA(N,NK)
      REAL ALPHA(N), BETA(N), PS(N)
      REAL TSTAG(N,NK), QSTAG(N,NK), TAU, SG(N,NK), SELOC(N,NK)
      REAL C(N,NK), D(N,NK)
      INTEGER TYPVAR
!
!
!Author
!          S. Belair (February 1996)
!
!Revision
! 001      S. Belair (Oct 1996) - Include the countergradient term
! 002      L. Spacek (Dec 2007) - add "vertical staggering" option
!                                 change the name to atmflux1
!
!
!Object
!        Calculate the atmospheric fluxes for heat, vapour,
!        and momentum.
!
!Arguments
!                        -Output-
!
! R         Resulting atmospheric flux
!           For U and V:  rho (w'v') = rho Km dv/ds
!           For Theta:    rho cp (w'theta')
!           For qv:       rho L (w'qv')
!
!                         -Input-
!
! VAR       Variable at t (U,V,Theta, or qv)
! TVAR      Time tendency of the variable
! KCOEF     Vertical diffusion coefficient in sigma form
! GAMMA     Counter-gradient term (non-zero only for theta and hu)
! ALPHA     inhomogeneous bottom boundary condition
! BETA      homogeneous bottom boundary condition
! PS        Surface pressure
! T         Temperature
! Q         Specific humidity
! TAU       Timestep
! SG        Sigma levels
! SELOC     Staggered sigma levels
! C         Work field
! D         Work field
! TYPVAR    Type of variable to treat
!           '0'  --->  U
!           '1'  --->  V
!           '2'  --->  Q
!           '3'  --->  Theta
! N,NK      Horizontal and vertical dimensions
!
!*
!
      INTEGER J,K
      REAL A
!
!
!
include "thermoconsts.inc"
include "dintern.inc"
include "fintern.inc"
!
!
!                                  Calculate VAR(t+1) (put into C)
      DO K=1,NK
        DO J=1,N
          C(J,K) = VAR(J,K) + TVAR(J,K)*TAU
        END DO
      END DO
!
!                                  Vertical derivative of VAR(t+1)
!                                  (Values on staggered levels).
!                                  Put into D.
!
      CALL DIFUVD4N(D, C, SG, N, NK)
!
!                                  Product K * dVAR/ds (into R).
!                                  Values on staggered levels.
!                                  CAREFUL:  We must divide by
!                                  a factor A = g sigma / R T
!                                  (on staggered levels again)
!
!
      DO K=1,NK-1
        DO J=1,N
          A = ( RGASD*TSTAG(J,K) ) / ( GRAV*SELOC(J,K) )
          R(J,K) = KCOEF(J,K) * D(J,K) * A
!
!
!
!                                  Add the countergradient part of the
!                                  flux:
!                                         = K * gamma / A**2
!
          IF (TYPVAR.EQ.2) THEN
             R(J,K) = R(J,K) + KCOEF(J,K) * GAMMA(J,K) * A
             GAMMA(J,K) = GAMMA(J,K) / A
          END IF
          IF (TYPVAR.EQ.3) &
             R(J,K) = R(J,K) + KCOEF(J,K) * &
                      GAMMA(J,K) * A * A
        END DO
      END DO
!
!                                  The same product for the
!                                  lowest level NK (actually,
!                                  NK is one less than the number
!                                  of model levels).
!                                  Km*dVAR/ds = alfa + beta*u(t+1)
!                                  Note:  Need to multiply by A also
!
      DO J=1,N
        A = ( RGASD*TSTAG(J,NK) ) / ( GRAV*SELOC(J,NK) )  !if it's SELOC, then it should be E-level, right?
        R(J,NK)   = (ALPHA(J) + BETA(J)*C(J,NK) ) * A
        IF (TYPVAR.EQ.2) &
              GAMMA(J,NK) = GAMMA(J,NK) / A
      END DO
!
!                                  Multiply by the air density
!                                  rho = p / (R Tv) = s ps / RTv
!                                  CAREFUL:  the calculations must
!                                            be on the SELOC levels,
!                                            but the variables T and
!                                            Q are defined on SG levels.
!
!                                  For all the levels except NK, NK+1
      DO K=1,NK-1
        DO J=1,N
          R(J,K) = SELOC(J,K)*PS(J) / &
                   ( RGASD * FOTVT(TSTAG(J,K),QSTAG(J,K)) ) &
                 * R(J,K)
        END DO
      END DO
!
!                                  For NK and NK+1
!
      DO J=1,N
          R(J,NK) = SELOC(J,NK)*PS(J) / &
                   ( RGASD * FOTVT(TSTAG(J,NK),QSTAG(J,NK)) ) &
                 *   R(J,NK)
          R(J,NK+1) = R(J,NK)
      END DO
!
!
!                                  At this point, R contains
!                                  rho (w'var').  For the temperature
!                                  and the specific humidity, however,
!                                  we need to multiply by cp and L,
!                                  respectively.
!
!                                  Furthermore, a factor (T/theta)
!                                  is kept for the fluxes of sensible
!                                  heat (because we want w'T', and
!                                  not w'theta')
!
      DO K=1,NK
        DO J=1,N
          IF (TYPVAR.EQ.2) R(J,K) = CHLC * R(J,K)
          IF (TYPVAR.EQ.3) &
            R(J,K) = CPD  * R(J,K) * &
                   ( SELOC(J,K)*PS(J) / 100000. )**0.286
        END DO
      END DO
!
      DO J=1,N
        R(J,NK+1) = R(J,NK)
      END DO
!
!
!                                  Time series for the fluxes
!
      IF (TYPVAR.EQ.0) &
        CALL serxst2(R, 'F5', TRNCH, N, NK+1, 0.0, 1.0, -1)
      IF (TYPVAR.EQ.1) &
        CALL serxst2(R, 'F6', TRNCH, N, NK+1, 0.0, 1.0, -1)
      IF (TYPVAR.EQ.2) &
        CALL serxst2(R, 'F3', TRNCH, N, NK+1, 0.0, 1.0, -1)
      IF (TYPVAR.EQ.3) &
        CALL serxst2(R, 'F4', TRNCH, N, NK+1, 0.0, 1.0, -1)
!
!
      RETURN
      END
