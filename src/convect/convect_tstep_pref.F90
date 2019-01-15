!     ######################################################################
      SUBROUTINE CONVECT_TSTEP_PREF( KLON, KLEV,                           &
                                   & PU, PV, PPRES, PZ, PDXDY, KLCL, KCTL, &
                                   & PTIMEA, PPREF )
!     ######################################################################

!!**** Routine to compute convective advection time step and precipitation
!!     efficiency
!!
!!
!!    PURPOSE
!!    -------
!!      The purpose of this routine is to determine the convective
!!      advection time step PTIMEC as a function of the mean ambient
!!      wind as well as the precipitation efficiency as a function
!!      of wind shear and cloud base height.
!!
!!
!!**  METHOD
!!    ------
!!
!!
!!    EXTERNAL
!!    --------
!!     None
!!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!     Module YOE_CONVPAREXT
!!          JCVEXB, JCVEXT     ! extra levels on the vertical boundaries
!!
!!    REFERENCE
!!    ---------
!!
!!      Book1,2 of documentation
!!      Fritsch and Chappell, 1980, J. Atmos. Sci.
!!      Kain and Fritsch, 1993, Meteor. Monographs, Vol.
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    07/11/95
!!   Last modified  04/10/97
!-------------------------------------------------------------------------------


!*       0.    DECLARATIONS
!              ------------

USE YOE_CONVPAREXT
#include "tsmbkind.cdk"


IMPLICIT NONE
#include <arch_specific.hf>

!*       0.1   Declarations of dummy arguments :

INTEGER_M, INTENT(IN)                    :: KLON   ! horizontal dimension
INTEGER_M, INTENT(IN)                    :: KLEV   ! vertical dimension
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PPRES  ! pressure (Pa)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PU     ! grid scale horiz. wind u
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PV     ! grid scale horiz. wind v
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PZ     ! height of model layer (m)
REAL_B, DIMENSION(KLON),      INTENT(IN) :: PDXDY  ! grid area (m^2)
INTEGER_M, DIMENSION(KLON),   INTENT(IN) :: KLCL   ! lifting condensation level index
INTEGER_M, DIMENSION(KLON),   INTENT(IN) :: KCTL   ! cloud top level index

REAL_B, DIMENSION(KLON),      INTENT(OUT):: PTIMEA ! advective time period
REAL_B, DIMENSION(KLON),      INTENT(OUT):: PPREF  ! precipitation efficiency


!*       0.2   Declarations of local variables KLON

INTEGER_M :: IIE, IKB, IKE                      ! horizontal + vertical loop bounds
INTEGER_M :: JI                                 ! horizontal loop index
INTEGER_M :: JK, JKLC, JKP5, JKCT               ! vertical loop index

INTEGER_M, DIMENSION(KLON)  :: IP500       ! index of 500 hPa levels
REAL_B, DIMENSION(KLON)     :: ZCBH        ! cloud base height
REAL_B, DIMENSION(KLON)     :: ZWORK1, ZWORK2, ZWORK3  ! work arrays


!-------------------------------------------------------------------------------

!        0.3   Set loop bounds
!              ---------------

IIE = KLON
IKB = 1 + JCVEXB
IKE = KLEV - JCVEXT


!*       1.     Determine vertical index for 500 hPa levels
!               ------------------------------------------


IP500(:) = IKB
DO JK = IKB, IKE
    WHERE ( PPRES(:,JK) >= 500.E2_JPRB ) IP500(:) = JK
ENDDO


!*       2.     Compute convective time step
!               ----------------------------

            ! compute wind speed at LCL, 500 hPa, CTL

DO JI = 1, IIE
   JKLC = KLCL(JI)
   JKP5 = IP500(JI)
   JKCT = KCTL(JI)
   ZWORK1(JI) = SQRT( PU(JI,JKLC) * PU(JI,JKLC) +           &
              &       PV(JI,JKLC) * PV(JI,JKLC)  )
   ZWORK2(JI) = SQRT( PU(JI,JKP5) * PU(JI,JKP5) +           &
              &       PV(JI,JKP5) * PV(JI,JKP5)  )
   ZWORK3(JI) = SQRT( PU(JI,JKCT) * PU(JI,JKCT) +           &
              &       PV(JI,JKCT) * PV(JI,JKCT)  )
ENDDO

ZWORK2(:) = MAX( 0.1_JPRB, 0.5_JPRB * ( ZWORK1(:) + ZWORK2(:) ) )

PTIMEA(:) = SQRT( PDXDY(:) ) / ZWORK2(:)


!*       3.     Compute precipitation efficiency
!               -----------------------------------

!*       3.1    Precipitation efficiency as a function of wind shear
!               ----------------------------------------------------

ZWORK2(:) = SIGN( 1._JPRB, ZWORK3(:) - ZWORK1(:) )
DO JI = 1, IIE
    JKLC = KLCL(JI)
    JKCT = KCTL(JI)
    ZWORK1(JI) = ( PU(JI,JKCT) - PU(JI,JKLC) )  *          &
               & ( PU(JI,JKCT) - PU(JI,JKLC) )  +          &
               & ( PV(JI,JKCT) - PV(JI,JKLC) )  *          &
               & ( PV(JI,JKCT) - PV(JI,JKLC) )
    ZWORK1(JI) = 1.E3_JPRB * ZWORK2(JI) * SQRT( ZWORK1(JI) ) /  &
               & MAX( 1.E-2_JPRB, PZ(JI,JKCT) - PZ(JI,JKLC) )
ENDDO

PPREF(:)  = 1.591_JPRB + ZWORK1(:) * ( -.639_JPRB + ZWORK1(:)       &
          &              * (  9.53E-2_JPRB - ZWORK1(:) * 4.96E-3_JPRB ) )
PPREF(:)  = MAX( .4_JPRB, MIN( PPREF(:), .70_JPRB ) )

!*       3.2    Precipitation efficiency as a function of cloud base height
!               ----------------------------------------------------------

DO JI = 1, IIE
   JKLC = KLCL(JI)
   ZCBH(JI)   = MAX( 3._JPRB, ( PZ(JI,JKLC) - PZ(JI,IKB) ) * 3.281E-3_JPRB )
ENDDO
ZWORK1(:) = .9673_JPRB + ZCBH(:) * ( -.7003_JPRB + ZCBH(:) * ( .1622_JPRB + &
          &   ZCBH(:) *  ( -1.2570E-2_JPRB + ZCBH(:) * ( 4.2772E-4_JPRB -   &
          &   ZCBH(:) * 5.44E-6_JPRB ) ) ) )
ZWORK1(:) = MAX( .4_JPRB, MIN( .70_JPRB, _ONE_/ ( _ONE_ + ZWORK1(:) ) ) )

!*       3.3    Mean precipitation efficiency is used to compute rainfall
!               ----------------------------------------------------------

PPREF(:) = 0.5_JPRB * ( PPREF(:) + ZWORK1(:) )


END SUBROUTINE CONVECT_TSTEP_PREF

