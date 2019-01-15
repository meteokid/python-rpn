!-------------------------------------------------------------------------------
!   ############################################################################
    SUBROUTINE BKF_SHALLOW ( KLON, KLEV, KIDIA, KFDIA, KBDIA, KTDIA,           &
                         & PDTCONV,  KICE,                                     &
                         & OSETTADJ, PTADJS,                                   &
                         & PPABS, PZZ,                                         &
                         & PT, PRV, PRC, PRI, PU, PV, PW,                      &
                         & OUVTRANSS,PCLOUD,PURC,PURI,                         &
                         & PTTENS, PRVTENS, PUTENS, PVTENS,                    &
                         & OCHTRANS, KCH1, PCH1, PCH1TEN,                      &
                         & PKSHAL                                              ) 
!   ############################################################################
!
!!**** Interface routine to the fast Meso-NH convection code developed for ECMWF/ARPEGE
!!     having a structure typical for operational routines
!!     
!!     Transformations necessary to call deep+ shallow code
!!     - skip input vertical arrays/levels : bottom=1, top=KLEV
!!     - transform specific humidities in mixing ratio
!!
!!
!!    PURPOSE
!!    -------
!!      The routine interfaces the MNH convection code as developed for operational
!!      forecast models like ECMWF/ARPEGE or HIRLAM with the typical Meso-NH array structure
!!      Calls the deep and/or shallow convection routine
!!
!!
!!**  METHOD
!!    ------
!!     Returns one tendency for shallow+deep convection but each part can
!!     be activated/desactivated separately
!!     For deep convection one can enable up to 3 additional ensemble members
!!     - this substantially improves the smoothness of the scheme and reduces
!!       allows for runs with different cloud radii (entrainment rates) and
!!       reduces the arbitrariness inherent to convective trigger condition
!!      
!!     
!!
!!    EXTERNAL
!!    --------
!!    CONVECT_SHALLOW
!!    SU_CONVPAR, SU_CONVPAR1
!!    SUCST:   ECMWF/ARPEGE routine
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    11/12/98
!!      modified    20/03/2002 by P. Marquet : transformed for ARPEGE/Climat
!!                             (tsmbkind.h, REAL_B, INTEGER_M, _JPRB, "& &",
!!                              _ZERO_, _ONE_, _HALF_)
!!      modified    11/04/O2 allow for ensemble of deep updrafts/downdrafts
!!
!!    REFERENCE
!!    ---------
!!    Bechtold et al., 2001, Quart. J. Roy. Meteor. Soc., Vol 127, pp 869-886: 
!!           A mass flux convection scheme for regional and global models.
!!
!-------------------------------------------------------------------------------
!Revisions in RPN
! 001   -PV-jan2015 - do not aggregate deep and shallow tendencies; add output of shallow tendencies
! 002   -PV-apr2015 - output cloud top and base heights for deep only
! 003   -JY-Jul2015 - separate shallow convection from convection.ftn90 (which has both deep and shallow convection in bechtold scheme) !
! note: kcltop and kclbas are non-flipped, must be flipped for use ; chem transport needs a bit of work to get tendency out
!*       0.    DECLARATIONS
!              ------------
!
!
#include "tsmbkind.cdk"
IMPLICIT NONE
#include <arch_specific.hf>
!
!*       0.1   Declarations of dummy arguments :
!
!
INTEGER_M,                    INTENT(IN)   :: KLON   ! horizontal dimension
INTEGER_M,                    INTENT(IN)   :: KLEV   ! vertical dimension
INTEGER_M,                    INTENT(IN)   :: KIDIA  ! value of the first point in x
INTEGER_M,                    INTENT(IN)   :: KFDIA  ! value of the last point in x
INTEGER_M,                    INTENT(IN)   :: KBDIA  ! vertical  computations start at
!                                                    ! KBDIA that is at least 1
INTEGER_M,                    INTENT(IN)   :: KTDIA  ! vertical computations can be
                                                     ! limited to KLEV + 1 - KTDIA
                                                     ! default=1
REAL_B,                       INTENT(IN)   :: PDTCONV! Interval of time between two
                                                     ! calls of the deep convection
                                                     ! scheme
INTEGER_M,                    INTENT(IN)   :: KICE   ! flag for ice ( 1 = yes, 
                                                     !                0 = no ice )
LOGICAL,                      INTENT(IN)   :: OSETTADJ ! logical to set convective
						     ! adjustment time by user
REAL_B,                       INTENT(IN)   :: PTADJS ! user defined shal. adjustment time (s)
!
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PT     ! grid scale T at time t  (K)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PRV    ! grid scale water vapor  (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PRC    ! grid scale r_c (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PRI    ! grid scale r_i (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PU     ! grid scale horiz. wind u (m/s) 
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PV     ! grid scale horiz. wind v (m/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PW     ! grid scale vertical velocity (m/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PPABS  ! grid scale pressure (Pa)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PZZ    ! geopotential (m2/s2) 
!   
!
LOGICAL,                      INTENT(IN)        :: OUVTRANSS! shallow
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)       :: PCLOUD   ! cloud fraction shallow
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)       :: PURC    ! grid-scale liquid cond (kg/kg) 
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)       :: PURI    ! grid-scale ice cond (kg/kg) 
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT)     :: PTTENS   ! convective temperat. tendency (K/s) shallow
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT)     :: PRVTENS  ! convective r_v tendency (1/s)  shallow
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT)     :: PUTENS   ! convecctive u tendency (m/s^2) shallow
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT)     :: PVTENS   ! convecctive v tendency (m/s^2) shallow

! convective Chemical Tracers:
LOGICAL,                      INTENT(IN)        :: OCHTRANS ! flag to compute convective
                                                            ! transport for chemical tracer
INTEGER_M,                    INTENT(IN)        :: KCH1     ! number of species
REAL_B, DIMENSION(KLON,KLEV,KCH1), INTENT(IN)   :: PCH1     ! grid scale chemical species
REAL_B, DIMENSION(KLON,KLEV,KCH1), INTENT(INOUT):: PCH1TEN  ! chemical convective tendency
                                                            ! (1/s)
!					 
! Diagnostic variables:

!INTEGER_M, DIMENSION(KLON),   INTENT(INOUT) :: KCLTOP ! cloud top level (number of model level)
!INTEGER_M, DIMENSION(KLON),   INTENT(INOUT) :: KCLBAS ! cloud base level(number of model level)
                                                      ! they are given a value of
                                                      ! 0 if no convection

!
REAL_B, DIMENSION(KLON),      INTENT(OUT)   :: PKSHAL ! shallow convective counter
!						 
!*       0.2   Declarations of local variables :
!
INTEGER_M  :: JI, JK, JKP, JN  ! loop index
!
!
! Local arrays (upside/down) necessary for change of ECMWF arrays to convection arrays
REAL_B, DIMENSION(KLON,KLEV) :: ZKKFC
REAL_B, DIMENSION(KLON,KLEV) :: ZT     ! grid scale T at time t  (K)
REAL_B, DIMENSION(KLON,KLEV) :: ZRV    ! grid scale water vapor  (kg/kg)
REAL_B, DIMENSION(KLON,KLEV) :: ZRC    ! grid scale r_c mixing ratio (kg/kg)
REAL_B, DIMENSION(KLON,KLEV) :: ZRI    ! grid scale r_i mixing ratio (kg/kg)
REAL_B, DIMENSION(KLON,KLEV) :: ZU     ! grid scale horiz. wind u (m/s) 
REAL_B, DIMENSION(KLON,KLEV) :: ZV     ! grid scale horiz. wind v (m/s)
REAL_B, DIMENSION(KLON,KLEV) :: ZW     ! grid scale vertical velocity (m/s)
REAL_B, DIMENSION(KLON,KLEV) :: ZW1    ! perturbed vertical velocity for ensemble (m/s)
REAL_B, DIMENSION(KLON,KLEV) :: ZPABS  ! grid scale pressure (Pa)
REAL_B, DIMENSION(KLON,KLEV) :: ZZZ    ! height of model layer (m) 
!
REAL_B, DIMENSION(KLON,KLEV,KCH1):: ZCH1     ! grid scale chemical species
REAL_B, DIMENSION(KLON,KLEV,KCH1):: ZCH1TEN  ! chemical convective tendency

common/cnvmain/ZCLOUD,ZTTENS, ZRVTENS, ZRCTENS, ZRITENS, ZUMFS,   ZURVS, ZURC, ZURI, &
ZUTENS, ZVTENS, ZCH1TENS, ICLBASS, ICLTOPS
!$OMP THREADPRIVATE(/cnvmain/)
#define ALLOCATABLE POINTER
!
! special for shallow convection
REAL_B, DIMENSION(:,:), ALLOCATABLE   :: ZCLOUD,ZTTENS, ZRVTENS, ZRCTENS, ZRITENS, &
                                         & ZUMFS, ZURVS, ZURC,ZURI,            &
                                         & ZUTENS, ZVTENS
REAL_B, DIMENSION(:,:,:), ALLOCATABLE :: ZCH1TENS
INTEGER_M, DIMENSION(:), ALLOCATABLE  :: ICLBASS, ICLTOPS
!
!*       0.5   Declarations of additional Ensemble fields:
!
!
! special for ERA40
REAL_B, DIMENSION(:,:),   ALLOCATABLE :: ZUDRS
!-------------------------------------------------------------------------------
!
!
!*       .9   Setup fundamental thermodunamical/physical constants using ECMWF/ARPEGE routine
!             ------------------------------------------------------------------------------
!
     CALL SUCST(54,20020211,0,0)
!
!
!*       1.   Allocate 2D (horizontal, vertical) arrays and additional ensemble arrays
!             ------------------------------------------------------------------------
!
    ALLOCATE( ZCLOUD(KLON,KLEV) )
    ALLOCATE( ZTTENS(KLON,KLEV) )
    ALLOCATE( ZUTENS(KLON,KLEV) )
    ALLOCATE( ZVTENS(KLON,KLEV) ) 
    ALLOCATE( ZRVTENS(KLON,KLEV) ) 
    ALLOCATE( ZRCTENS(KLON,KLEV) )
    ALLOCATE( ZRITENS(KLON,KLEV) ) 
    ALLOCATE( ZCH1TENS(KLON,KLEV,KCH1) ) 
    ALLOCATE( ZUMFS(KLON,KLEV) )
    ALLOCATE( ZURC(KLON,KLEV) )
    ALLOCATE( ZURI(KLON,KLEV) )
    ALLOCATE( ZURVS(KLON,KLEV) )
    ALLOCATE( ICLBASS(KLON) )
    ALLOCATE( ICLTOPS(KLON) )

    ALLOCATE( ZUDRS(KLON,KLEV) )

!    KCLTOP(:)  = 1 ! set default value when no convection
!    KCLBAS(:)  = 1 ! can be changed  depending on user
!
    PCLOUD = 0.0
    PURC   = 0.0
    PURI   = 0.0
!
!
!
!*       2.   Flip arrays upside-down as  first vertical level in convection is 1
!             --------------------------------------------------------------------
!
DO JK = 1, KLEV
   JKP = KLEV - JK + 1
   DO JI = KIDIA, KFDIA
      ZPABS(JI,JKP) = PPABS(JI,JK)
      ZZZ(JI,JKP)   = PZZ(JI,JK)
      ZT(JI,JKP)    = PT(JI,JK)
      ZRV(JI,JKP)   = PRV(JI,JK) / ( _ONE_ - PRV(JI,JK) ) ! transform specific humidity
      ZRC(JI,JKP)   = PRC(JI,JK) / ( _ONE_ - PRC(JI,JK) ) ! in mixing ratio
      ZRI(JI,JKP)   = PRI(JI,JK) / ( _ONE_ - PRI(JI,JK) ) 
      ZU(JI,JKP)    = PU(JI,JK)
      ZV(JI,JKP)    = PV(JI,JK)
      ZW(JI,JKP)    = PW(JI,JK) 
   END DO
END DO
IF ( OCHTRANS ) THEN
   DO JK = 1, KLEV
      JKP = KLEV - JK + 1
      DO JN = 1, KCH1
         DO JI = KIDIA, KFDIA
            ZCH1(JI,JKP,JN) = PCH1(JI,JK,JN)
         END DO
      END DO
   END DO
END IF
! 
!             ----------------------------
!
!*       3.  Call shallow convection routine
!             -------------------------------
!
    CALL SU_CONVPAR 
    CALL SU_CONVPAR_SHAL
!
    CALL CONVECT_SHALLOW( KLON, KLEV, KIDIA, KFDIA, KBDIA, KTDIA,        &
                           & PDTCONV, KICE, OSETTADJ, PTADJS,            &
                           & ZPABS, ZZZ,                                 &
                           & ZT, ZRV, ZRC, ZRI, ZW,                      &
                           & ZTTENS, ZRVTENS, ZRCTENS, ZRITENS,          &
                           & ICLTOPS, ICLBASS, ZUMFS, ZURVS,             &
                           & ZCLOUD,ZURC,ZURI,                           &
                           & OCHTRANS, KCH1, ZCH1, ZCH1TENS              &
                           &,ZUDRS, PKSHAL, OUVTRANSS, ZU, ZV,           &
                           & ZUTENS, ZVTENS                              )  
!
!            ---------------------------------------------------------
!
!
!*       4.  Reflip arrays to ECMWF/ARPEGE vertical structure
!            change mixing ratios to specific humidity

DO JK = 1, KLEV
   JKP = KLEV - JK + 1
   DO JI = KIDIA, KFDIA
      PTTENS(JI,JK) = ZTTENS(JI,JKP)
      PRVTENS(JI,JK) = ZRVTENS(JI,JKP) ! / ( _ONE_ + ZRV(JI,JKP) ) ** 2
      PUTENS(JI,JK) = ZUTENS(JI,JKP)
      PVTENS(JI,JK) = ZVTENS(JI,JKP)
      PCLOUD(JI,JK) = ZCLOUD(JI,JKP)
      PURC(JI,JK) = ZURC(JI,JKP)
      PURI(JI,JK) = ZURI(JI,JKP)
   END DO
END DO

IF ( OCHTRANS ) THEN
   DO JK = 1, KLEV
      JKP = KLEV - JK + 1
      DO JN = 1, KCH1
         DO JI = KIDIA, KFDIA
            PCH1TEN(JI,JK,JN) = ZCH1TENS(JI,JKP,JN)
         END DO
      END DO
   END DO
END IF
   
!*       7.  Deallocate local arrays
!
       DEALLOCATE( ICLBASS )
       DEALLOCATE( ICLTOPS )
       DEALLOCATE( ZUMFS )
       DEALLOCATE( ZURVS )
       DEALLOCATE( ZURC )
       DEALLOCATE( ZURI )
       DEALLOCATE( ZCH1TENS ) 
       DEALLOCATE( ZRCTENS )
       DEALLOCATE( ZRITENS ) 
       DEALLOCATE( ZCLOUD )
       DEALLOCATE( ZTTENS )
       DEALLOCATE( ZUTENS )
       DEALLOCATE( ZVTENS )
       DEALLOCATE( ZRVTENS ) 

       DEALLOCATE( ZUDRS )
!
!
END SUBROUTINE BKF_SHALLOW
!
!----------------------------------------------------------------------------
