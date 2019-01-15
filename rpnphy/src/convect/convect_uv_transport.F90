!#######################################################################
  SUBROUTINE CONVECT_UV_TRANSPORT( KLON, KLEV, PU, PV, PUC, PVC,       &
                                 & KDPL, KPBL, KLCL, KCTL, KLFS, KDBL, &
                                 & PUMF, PUER, PUDR, PDMF, PDER, PDDR, &
                                 & PTIMEC, PDXDY, PMIXF, PLMASS, PWSUB,&
                                 & KFTSTEPS )
!#######################################################################

!!**** Compute  modified horizontal wind components due to convective event
!!
!!
!!    PURPOSE
!!    -------
!!      The purpose of this routine is to determine convective adjusted
!!      horizontal wind components u and v
!!      The final convective tendencies can then be evaluated in the main
!!      routine DEEP_CONVECT by (PUC-PU)/PTIMEC
!!
!!
!!**  METHOD
!!    ------
!!      Identical to the computation of the conservative variables in the
!!      tracer routine but includes pressure term
!!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module YOMCST
!!          RG                 ! gravity constant
!!
!!     Module YOE_CONVPAREXT
!!          JCVEXB, JCVEXT     ! extra levels on the vertical boundaries
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!
!!      Original    11/02/02
!!
!-------------------------------------------------------------------------------


!*       0.    DECLARATIONS
!              ------------

USE YOMCST
USE YOE_CONVPAR
USE YOE_CONVPAREXT
#include "tsmbkind.cdk"

IMPLICIT NONE
#include <arch_specific.hf>

!*       0.1   Declarations of dummy arguments :

INTEGER_M,                INTENT(IN) :: KLON     ! horizontal dimension
INTEGER_M,                INTENT(IN) :: KLEV     ! vertical dimension

REAL_B,DIMENSION(KLON,KLEV),INTENT(IN) :: PU     ! horizontal wind in x (m/s)
REAL_B,DIMENSION(KLON,KLEV),INTENT(IN) :: PV     ! horizontal wind in x (m/s)
REAL_B,DIMENSION(KLON,KLEV),INTENT(OUT):: PUC    ! convective adjusted value of u (m/s)
REAL_B,DIMENSION(KLON,KLEV),INTENT(OUT):: PVC    ! convective adjusted value of v (m/s)

INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KDPL   ! index for departure level
INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KPBL   ! index for top of source layer
INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KLCL   ! index lifting condens. level
INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KCTL   ! index for cloud top level
INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KLFS   ! index for level of free sink
INTEGER_M, DIMENSION(KLON), INTENT(IN) :: KDBL   ! index for downdraft base level

REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PUMF ! updraft mass flux (kg/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PUER ! updraft entrainment (kg/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PUDR ! updraft detrainment (kg/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PDMF ! downdraft mass flux (kg/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PDER ! downdraft entrainment (kg/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PDDR ! downdraft detrainment (kg/s)

REAL_B, DIMENSION(KLON),     INTENT(IN) :: PTIMEC! convection time step
REAL_B, DIMENSION(KLON),     INTENT(IN) :: PDXDY ! grid area (m^2)
REAL_B, DIMENSION(KLON),     INTENT(IN) :: PMIXF ! mixed fraction at LFS
REAL_B, DIMENSION(KLON,KLEV),INTENT(IN) :: PLMASS! mass of model layer (kg)
REAL_B, DIMENSION(KLON,KLEV),INTENT(IN) :: PWSUB ! envir. compensating subsidence(Pa/s)
INTEGER_M,                   INTENT(IN) :: KFTSTEPS  ! maximum fractional time steps


!*       0.2   Declarations of local variables :

INTEGER_M :: IIE, IKB, IKE  ! horizontal + vertical loop bounds
INTEGER_M :: IKS            ! vertical dimension
INTEGER_M :: JI             ! horizontal loop index
INTEGER_M :: JK, JKP        ! vertical loop index
INTEGER_M :: JSTEP          ! fractional time loop index
INTEGER_M :: JKLD, JKLP, JKMAX ! loop index for levels

INTEGER_M, PARAMETER             :: IUV = 2    ! for u and v
REAL_B, DIMENSION(KLON,KLEV)     :: ZOMG       ! compensat. subsidence (Pa/s)
REAL_B, DIMENSION(KLON,KLEV,IUV) :: ZUUV, ZDUV ! updraft/downdraft values
REAL_B, DIMENSION(KLON)          :: ZTIMEC     ! fractional convective time step
REAL_B, DIMENSION(KLON,KLEV)     :: ZTIMC      ! 2D work array for ZTIMEC
REAL_B, DIMENSION(KLON,KLEV,IUV) :: ZUVMFIN, ZUVMFOUT
                                   ! work arrays for environm. compensat. mass
REAL_B, DIMENSION(KLON,IUV)      :: ZWORK1, ZWORK2, ZWORK3

!-------------------------------------------------------------------------------

!*       0.3   Compute loop bounds
!              -------------------

IIE    = KLON
IKB    = 1 + JCVEXB
IKS    = KLEV
IKE    = KLEV - JCVEXT
JKMAX  = MAXVAL( KCTL(:) )


!*      2.      Updraft computations
!               --------------------

ZUUV(:,:,:) = _ZERO_

!*      2.1     Initialization  at LCL
!               ----------------------------------

DO JI = 1, IIE
    JKLD = KDPL(JI)
    JKLP = KPBL(JI)
    ZWORK1(JI,1) = _HALF_ * ( PU(JI,JKLD) + PU(JI,JKLP) )
    ZWORK1(JI,2) = _HALF_ * ( PV(JI,JKLD) + PV(JI,JKLP) )
ENDDO

!*      2.2     Final updraft loop
!               ------------------

DO JK = MINVAL( KDPL(:) ), JKMAX
JKP = JK + 1

     DO JI = 1, IIE
       IF ( KDPL(JI) <= JK .AND. KLCL(JI) > JK ) THEN
            ZUUV(JI,JK,1) = ZWORK1(JI,1)
            ZUUV(JI,JK,2) = ZWORK1(JI,2)
       END IF

       IF ( KLCL(JI) - 1 <= JK .AND. KCTL(JI) > JK ) THEN
                            ! instead of passive tracers equations
                            ! wind equations also include pressure term
           ZUUV(JI,JKP,1) = ( PUMF(JI,JK) * ZUUV(JI,JK,1) +                   &
                            &   PUER(JI,JKP) * PU(JI,JK) )  /                 &
                            & ( PUMF(JI,JKP) + PUDR(JI,JKP) + 1.E-7_JPRB ) +  &
                            &   XUVDP * ( PU(JI,JKP) - PU(JI,JK) ) 
           ZUUV(JI,JKP,2) = ( PUMF(JI,JK) * ZUUV(JI,JK,2) +                   &
                            &   PUER(JI,JKP) * PV(JI,JK) )  /                 &
                            & ( PUMF(JI,JKP) + PUDR(JI,JKP) + 1.E-7_JPRB ) +  &
                            &   XUVDP * ( PV(JI,JKP) - PV(JI,JK) ) 
       ENDIF
     ENDDO

ENDDO

!*      3.      Downdraft computations
!               ----------------------

ZDUV(:,:,:) = _ZERO_

!*      3.1     Initialization at the LFS
!               -------------------------

ZWORK1(:,:) = SPREAD( PMIXF(:), DIM=2, NCOPIES=IUV )
DO JI = 1, IIE
     JK = KLFS(JI)
     ZDUV(JI,JK,1) = ZWORK1(JI,1) * PU(JI,JK) +                          &
                    &            ( _ONE_ - ZWORK1(JI,1) ) * ZUUV(JI,JK,1)
     ZDUV(JI,JK,2) = ZWORK1(JI,2) * PV(JI,JK) +                          &
                    &            ( _ONE_ - ZWORK1(JI,2) ) * ZUUV(JI,JK,2)
ENDDO

!*      3.2     Final downdraft loop
!               --------------------

DO JK = MAXVAL( KLFS(:) ), IKB + 1, -1
JKP = JK - 1
    DO JI = 1, IIE
      IF ( JK <= KLFS(JI) .AND. JKP >= KDBL(JI) ) THEN
       ZDUV(JI,JKP,1) = ( ZDUV(JI,JK,1) * PDMF(JI,JK) -                  &
                        &     PU(JI,JK) * PDER(JI,JKP) ) /               &
                        & ( PDMF(JI,JKP) - PDDR(JI,JKP) - 1.E-7_JPRB ) + & 
                        &   XUVDP * ( PU(JI,JKP) - PU(JI,JK) ) 
       ZDUV(JI,JKP,2) = ( ZDUV(JI,JK,2) * PDMF(JI,JK) -                  &
                        &     PV(JI,JK) *  PDER(JI,JKP) ) /              &
                        & ( PDMF(JI,JKP) - PDDR(JI,JKP) - 1.E-7_JPRB ) + &
                        &   XUVDP * ( PV(JI,JKP) - PV(JI,JK) ) 
      ENDIF
    ENDDO
ENDDO


!*      4.      Final closure (environmental) computations
!               ------------------------------------------

PUC(:,IKB:IKE) = PU(:,IKB:IKE) ! initialize adjusted envir. values
PVC(:,IKB:IKE) = PV(:,IKB:IKE) ! initialize adjusted envir. values

DO JK = IKB, IKE
   ZOMG(:,JK) = PWSUB(:,JK) * PDXDY(:) / RG ! environmental subsidence
ENDDO

ZTIMEC(:) = PTIMEC(:) / REAL( KFTSTEPS ) ! adjust  fractional time step
                                         ! to be an integer multiple of PTIMEC
WHERE ( PTIMEC(:) < _ONE_ ) ZTIMEC(:) = _ZERO_
ZTIMC(:,:)= SPREAD( ZTIMEC(:), DIM=2, NCOPIES=IKS )

ZUVMFIN(:,:,:)   = _ZERO_
ZUVMFOUT(:,:,:)  = _ZERO_


DO JSTEP = 1, KFTSTEPS ! Enter the fractional time step loop

      DO JK = IKB + 1, JKMAX
      JKP = MAX( IKB + 1, JK - 1 )
        DO JI = 1, IIE
        IF ( JK <= KCTL(JI) )  THEN
          ZWORK3(JI,1) = ZOMG(JI,JK)
          ZWORK1(JI,1) = SIGN( _ONE_, ZWORK3(JI,1) )
          ZWORK2(JI,1) = _HALF_ * ( _ONE_ + ZWORK1(JI,1) )
          ZWORK1(JI,1) = _HALF_ * ( _ONE_ - ZWORK1(JI,1) )
          ZUVMFIN(JI,JK,1)  = - ZWORK3(JI,1) * PUC(JI,JKP) * ZWORK1(JI,1)
          ZUVMFOUT(JI,JK,1) =   ZWORK3(JI,1) * PUC(JI,JK)  * ZWORK2(JI,1)
          ZUVMFIN(JI,JK,2)  = - ZWORK3(JI,1) * PVC(JI,JKP) * ZWORK1(JI,1)
          ZUVMFOUT(JI,JK,2) =   ZWORK3(JI,1) * PVC(JI,JK)  * ZWORK2(JI,1)
          ZUVMFIN(JI,JKP,1) = ZUVMFIN(JI,JKP,1) + ZUVMFOUT(JI,JK,1) * ZWORK2(JI,1)
          ZUVMFIN(JI,JKP,2) = ZUVMFIN(JI,JKP,2) + ZUVMFOUT(JI,JK,2) * ZWORK2(JI,1)
          ZUVMFOUT(JI,JKP,1)= ZUVMFOUT(JI,JKP,1)+ ZUVMFIN(JI,JK,1)  * ZWORK1(JI,1)
          ZUVMFOUT(JI,JKP,2)= ZUVMFOUT(JI,JKP,2)+ ZUVMFIN(JI,JK,2)  * ZWORK1(JI,1)
        END IF
        END DO
      END DO

       DO JK = IKB + 1, JKMAX
        DO JI = 1, IIE
        IF ( JK <= KCTL(JI) ) THEN
         PUC(JI,JK) = PUC(JI,JK) + ZTIMC(JI,JK) / PLMASS(JI,JK) *  (       &
                   &   ZUVMFIN(JI,JK,1) + PUDR(JI,JK) * ZUUV(JI,JK,1) +    &
                   &   PDDR(JI,JK) * ZDUV(JI,JK,1) - ZUVMFOUT(JI,JK,1) -   &
                   &   ( PUER(JI,JK) + PDER(JI,JK) ) * PU(JI,JK)    )
         PVC(JI,JK) = PVC(JI,JK) + ZTIMC(JI,JK) / PLMASS(JI,JK) *  (       &
                   &   ZUVMFIN(JI,JK,2) + PUDR(JI,JK) * ZUUV(JI,JK,2) +    &
                   &   PDDR(JI,JK) * ZDUV(JI,JK,2) - ZUVMFOUT(JI,JK,2) -   &
                   &   ( PUER(JI,JK) + PDER(JI,JK) ) * PV(JI,JK)    )
        END IF
        END DO
       END DO

ENDDO ! Exit the fractional time step loop


END SUBROUTINE CONVECT_UV_TRANSPORT

