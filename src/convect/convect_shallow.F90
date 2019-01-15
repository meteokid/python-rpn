!############################################################################
 SUBROUTINE CONVECT_SHALLOW( KLON, KLEV, KIDIA, KFDIA, KBDIA, KTDIA,        &
                           & PDTCONV, KICE, OSETTADJ, PTADJS,               &
                           & PPABST, PZZ,                                   &
                           & PTT, PRVT, PRCT, PRIT, PWT,                    &
                           & PTTEN, PRVTEN, PRCTEN, PRITEN,                 &
                           & KCLTOP, KCLBAS, PUMF, PURV,                    &
                           & PCLOUD,PURCOUT,PURIOUT,                        &
                           & OCH1CONV, KCH1, PCH1, PCH1TEN,                 &
                           & PUDR, PKSHAL, OUVCONV, PUT, PVT, PUTEN, PVTEN  ) 
!############################################################################

!!**** Monitor routine to compute all convective tendencies by calls
!!     of several subroutines.
!!
!!
!!    PURPOSE
!!    -------
!!      The purpose of this routine is to determine the convective
!!      tendencies. The routine first prepares all necessary grid-scale
!!      variables. The final convective tendencies are then computed by
!!      calls of different subroutines.
!!
!!
!!**  METHOD
!!    ------
!!      We start by selecting convective columns in the model domain through
!!      the call of routine TRIGGER_FUNCT. Then, we allocate memory for the
!!      convection updraft and downdraft variables and gather the grid scale
!!      variables in convective arrays.
!!      The updraft and downdraft computations are done level by level starting
!!      at the  bottom and top of the domain, respectively.
!!      All computations are done on MNH thermodynamic levels. The depth
!!      of the current model layer k is defined by DP(k)=P(k-1)-P(k)
!!
!!
!!
!!    EXTERNAL
!!    --------
!!    CONVECT_TRIGGER_SHAL
!!    CONVECT_SATMIXRATIO
!!    CONVECT_UPDRAFT_SHAL
!!        CONVECT_CONDENS
!!        CONVECT_MIXING_FUNCT
!!    CONVECT_CLOSURE_SHAL
!!        CONVECT_CLOSURE_THRVLCL
!!        CONVECT_CLOSURE_ADJUST_SHAL
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module YOMCST
!!          RG                   ! gravity constant
!!          RPI                  ! number Pi
!!          RATM                 ! reference pressure
!!          RD, RV               ! gaz  constants for dry air and water vapor
!!          RCPD, RCPV           ! specific heat for dry air and water vapor
!!          RALPW, RBETW, RGAMW  ! constants for water saturation pressure
!!          RTT                  ! triple point temperature
!!          RLVTT, RLSTT         ! vaporization, sublimation heat constant
!!          RCW, RCS             ! specific heat for liquid water and ice
!!
!!      Module YOE_CONVPAREXT
!!          JCVEXB, JCVEXT       ! extra levels on the vertical boundaries
!!
!!      Module YOE_CONVPAR
!!          XA25                 ! reference grid area
!!          XCRAD                ! cloud radius
!!
!!
!!    REFERENCE
!!    ---------
!!
!!      Bechtold, 1997 : Meso-NH scientific  documentation (31 pp)
!!      Fritsch and Chappell, 1980, J. Atmos. Sci., Vol. 37, 1722-1761.
!!      Kain and Fritsch, 1990, J. Atmos. Sci., Vol. 47, 2784-2801.
!!      Kain and Fritsch, 1993, Meteor. Monographs, Vol. 24, 165-170.
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    26/03/96
!!   Peter Bechtold 15/11/96 replace theta_il by enthalpy
!!         "        10/12/98 changes for ARPEGE
!-------------------------------------------------------------------------------


!*       0.    DECLARATIONS
!              ------------

USE YOMCST
USE YOE_CONVPAREXT
USE YOE_CONVPAR_SHAL
!use yomlun,only : nulout
#include "tsmbkind.cdk"


IMPLICIT NONE
#include <arch_specific.hf>

!*       0.1   Declarations of dummy arguments :


INTEGER_M,                    INTENT(IN) :: KLON     ! horizontal dimension
INTEGER_M,                    INTENT(IN) :: KLEV     ! vertical dimension
INTEGER_M,                    INTENT(IN) :: KIDIA    ! value of the first point in x
INTEGER_M,                    INTENT(IN) :: KFDIA    ! value of the last point in x
INTEGER_M,                    INTENT(IN) :: KBDIA    ! vertical  computations start at
!                                                    ! KBDIA that is at least 1
INTEGER_M,                    INTENT(IN) :: KTDIA    ! vertical computations can be
                                                     ! limited to KLEV + 1 - KTDIA
                                                     ! default=1
REAL_B,                       INTENT(IN) :: PDTCONV  ! Interval of time between two
                                                     ! calls of the deep convection
                                                     ! scheme
INTEGER_M,                    INTENT(IN) :: KICE     ! flag for ice ( 1 = yes,
                                                     !                0 = no ice )
LOGICAL,                      INTENT(IN) :: OSETTADJ ! logical to set convective
                                                     ! adjustment time by user
REAL_B,                       INTENT(IN) :: PTADJS   ! user defined adjustment time (s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PTT      ! grid scale temperature at (K)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PRVT     ! grid scale water vapor (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PRCT     ! grid scale r_c  (kg/kg)"
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PRIT     ! grid scale r_i (kg/kg)"
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PWT      ! grid scale vertical
                                                     ! velocity (m/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PPABST   ! grid scale pressure (Pa)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN) :: PZZ      ! height of model layer (m)

REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PTTEN  ! convective temperature
                                                     ! tendency (K/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PRVTEN ! convective r_v tendency (1/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PRCTEN ! convective r_c tendency (1/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PRITEN ! convective r_i tendency (1/s)
INTEGER_M, DIMENSION(KLON),   INTENT(INOUT):: KCLTOP ! cloud top level
INTEGER_M, DIMENSION(KLON),   INTENT(INOUT):: KCLBAS ! cloud base level
                                                     ! they are given a value of
                                                     ! 0 if no convection
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PUMF   ! updraft mass flux (kg/s m2)
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)  :: PURV   ! updraft water vapor (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)  :: PURCOUT  ! normalized mixing ratio of updraft cloud water (kg/kg)
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)  :: PURIOUT  ! normalized mixing ratio of updraft cloud ice   (kg/kg)

LOGICAL,                      INTENT(IN) :: OCH1CONV ! include tracer transport
INTEGER_M,                    INTENT(IN) :: KCH1     ! number of species
REAL_B, DIMENSION(KLON,KLEV,KCH1), INTENT(IN) :: PCH1! grid scale chemical species
REAL_B, DIMENSION(KLON,KLEV,KCH1), INTENT(INOUT):: PCH1TEN! species conv. tendency (1/s)

! for ERA40
REAL_B, DIMENSION(KLON,KLEV), INTENT(OUT)  :: PUDR   ! updraft detrainment rate (kg/s m3)
REAL_B, DIMENSION(KLON),      INTENT(OUT)  :: PKSHAL ! shallow convective counter
!
LOGICAL,                      INTENT(IN)   :: OUVCONV! include wind transport (Cu friction)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PUT    ! grid scale horiz. wind u (m/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(IN)   :: PVT    ! grid scale horiz. wind v (m/s)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PUTEN  ! convective u tendency (m/s^2)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PVTEN  ! convective v tendency (m/s^2)
REAL_B, DIMENSION(KLON,KLEV), INTENT(INOUT):: PCLOUD ! shallow cloud fraction (%)

!*       0.2   Declarations of local fixed memory variables :

INTEGER_M  :: ITEST, ICONV, ICONV1    ! number of convective columns
INTEGER_M  :: IIB, IIE                ! horizontal loop bounds
INTEGER_M  :: IKB, IKE                ! vertical loop bounds
INTEGER_M  :: IKS                     ! vertical dimension
INTEGER_M  :: JI, JL                  ! horizontal loop index
INTEGER_M  :: JN                      ! number of tracers
INTEGER_M  :: JK, JKP, JKM            ! vertical loop index
INTEGER_M  :: IFTSTEPS                ! only used for chemical tracers
REAL_B     :: ZEPS, ZEPSA, ZEPSB      ! R_d / R_v, R_v / R_d, RCPV / RCPD - ZEPSA
REAL_B     :: ZCPORD, ZRDOCP          ! C_p/R_d,  R_d/C_p

LOGICAL, DIMENSION(KLON,KLEV)        :: GTRIG3 ! 3D logical mask for convection
LOGICAL, DIMENSION(KLON)             :: GTRIG  ! 2D logical mask for trigger test
REAL_B,  DIMENSION(KLON,KLEV)        :: ZTHT, ZSTHV, ZSTHES  ! grid scale theta, theta_v
REAL_B,  DIMENSION(KLON)             :: ZWORK2, ZWORK2B ! work array

common/shallow/ IDPL    ,IPBL    ,ILCL    ,IETL    ,ICTL    ,ILFS    ,&
ISDPL   ,ISPBL   ,ISLCL   ,ZSTHLCL ,ZSTLCL  ,ZSRVLCL ,ZSWLCL  ,ZSZLCL  ,&
ZSTHVELCL,ZSDXDY  ,ZZ      ,ZPRES   ,ZDPRES  ,ZW      ,ZTT     ,ZTH     ,&
ZTHV    ,ZTHL    ,ZTHES, ZTHEST ,ZRW     ,ZRV     ,ZRC     ,ZRI     ,&
ZDXDY   ,ZUMF    ,ZUER    ,ZUDR    ,ZUTHL   ,ZUTHV   ,ZWU,   ZURW    ,ZURC    ,&
ZURI    ,ZMFLCL  ,ZCAPE   ,ZTHLCL  ,ZTLCL   ,ZRVLCL  ,ZWLCL   ,ZZLCL   ,&
ZTHVELCL,ZDMF    ,ZDER    ,ZDDR    ,ZLMASS  ,ZTIMEC  ,ZTHC    ,ZRVC    ,&
ZRCC    ,ZRIC    ,ZWSUB   , GTRIG1  , GWORK   , IINDEX, IJINDEX, IJSINDEX,&
IJPINDEX, ZCPH    , ZLV, ZLS, ZCH1    , ZCH1C   , ZWORK3  , GTRIG4, &
ZU      ,ZV      ,ZUC     ,ZVC
!$OMP THREADPRIVATE(/shallow/)
#define ALLOCATABLE POINTER
!*       0.2   Declarations of local allocatable  variables :

INTEGER_M, DIMENSION(:),ALLOCATABLE  :: IDPL    ! index for parcel departure level
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: IPBL    ! index for source layer top
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: ILCL    ! index for lifting condensation level
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: IETL    ! index for zero buoyancy level
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: ICTL    ! index for cloud top level
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: ILFS    ! index for level of free sink

INTEGER_M, DIMENSION(:), ALLOCATABLE :: ISDPL   ! index for parcel departure level
INTEGER_M, DIMENSION(:),ALLOCATABLE  :: ISPBL   ! index for source layer top
INTEGER_M, DIMENSION(:), ALLOCATABLE :: ISLCL   ! index for lifting condensation level
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSTHLCL ! updraft theta at LCL
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSTLCL  ! updraft temp. at LCL
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSRVLCL ! updraft rv at LCL
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSWLCL  ! updraft w at LCL
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSZLCL  ! LCL height
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSTHVELCL! envir. theta_v at LCL
REAL_B, DIMENSION(:), ALLOCATABLE    :: ZSDXDY  ! grid area (m^2)

! grid scale variables
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZZ      ! height of model layer (m)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZPRES   ! grid scale pressure
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZDPRES  ! pressure difference between
                                                ! bottom and top of layer (Pa)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZW      ! grid scale vertical velocity on theta grid
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTT     ! temperature
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTH     ! grid scale theta
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTHV    ! grid scale theta_v
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTHL    ! grid scale enthalpy (J/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTHES, ZTHEST ! grid scale saturated theta_e
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRW     ! grid scale total water (kg/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRV     ! grid scale water vapor (kg/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRC     ! grid scale cloud water (kg/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRI     ! grid scale cloud ice (kg/kg)
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZDXDY   ! grid area (m^2)

! updraft variables
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUMF    ! updraft mass flux (kg/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUER    ! updraft entrainment (kg/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUDR    ! updraft detrainment (kg/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUTHL   ! updraft enthalpy (J/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUTHV   ! updraft theta_v (K)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZWU     ! updraft vertical velocity (m/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZURW    ! updraft total water (kg/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZURC    ! updraft cloud water (kg/kg)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZURI    ! updraft cloud ice   (kg/kg)
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZMFLCL  ! cloud base unit mass flux(kg/s)
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZCAPE   ! available potent. energy
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZTHLCL  ! updraft theta at LCL
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZTLCL   ! updraft temp. at LCL
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZRVLCL  ! updraft rv at LCL
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZWLCL   ! updraft w at LCL
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZZLCL   ! LCL height
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZTHVELCL! envir. theta_v at LCL

! downdraft variables
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZDMF    ! downdraft mass flux (kg/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZDER    ! downdraft entrainment (kg/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZDDR    ! downdraft detrainment (kg/s)

! closure variables
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZLMASS  ! mass of model layer (kg)
REAL_B, DIMENSION(:),   ALLOCATABLE  :: ZTIMEC  ! advective time period

REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZTHC    ! conv. adj. grid scale theta
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRVC    ! conv. adj. grid scale r_w
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRCC    ! conv. adj. grid scale r_c
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZRIC    ! conv. adj. grid scale r_i
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZWSUB   ! envir. compensating subsidence (Pa/s)

LOGICAL,   DIMENSION(:), ALLOCATABLE  :: GTRIG1  ! logical mask for convection
LOGICAL,   DIMENSION(:), ALLOCATABLE  :: GWORK   ! logical work array
INTEGER_M, DIMENSION(:), ALLOCATABLE  :: IINDEX, IJINDEX, IJSINDEX, IJPINDEX!hor.index
REAL_B,    DIMENSION(:), ALLOCATABLE  :: ZCPH    ! specific heat C_ph
REAL_B,    DIMENSION(:), ALLOCATABLE  :: ZLV, ZLS! latent heat of vaporis., sublim.
REAL_B                                :: ZES     ! saturation vapor mixng ratio
REAL_B                                :: ZW1     ! work variables

! Chemical Tracers:
REAL_B,  DIMENSION(:,:,:), ALLOCATABLE:: ZCH1    ! grid scale chemical specy (kg/kg)
REAL_B,  DIMENSION(:,:,:), ALLOCATABLE:: ZCH1C   ! conv. adjust. chemical specy 1
REAL_B,  DIMENSION(:,:),   ALLOCATABLE:: ZWORK3  ! conv. adjust. chemical specy 1
LOGICAL, DIMENSION(:,:,:), ALLOCATABLE:: GTRIG4  ! logical mask
! for U, V transport:
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZU      ! grid scale horiz. u component on theta grid
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZV      ! grid scale horiz. v component on theta grid
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZUC     ! horizontal wind u (m/s)
REAL_B, DIMENSION(:,:), ALLOCATABLE  :: ZVC     ! horizontal wind v (m/s)
!-------------------------------------------------------------------------------


!*       0.3    Compute loop bounds
!               -------------------

IIB    = KIDIA
IIE    = KFDIA
JCVEXB = MAX( 0, KBDIA - 1 )
IKB    = 1 + JCVEXB
IKS    = KLEV
JCVEXT = MAX( 0, KTDIA - 1)
IKE    = IKS - JCVEXT

!PV code related to refresh as a fct of counter is useless (for now) for shallow convection, commented out

!*       0.5    Update convective counter ( where KCOUNT > 0
!               convection is still active ).
!               ---------------------------------------------

GTRIG(IIB:IIE) = .TRUE.
ITEST          = COUNT( GTRIG(:) )

!IF ( ITEST == 0 ) RETURN



!*       0.7    Reset convective tendencies to zero if convective
!               counter becomes negative
!               -------------------------------------------------

GTRIG3(:,:) = SPREAD( GTRIG(:), DIM=2, NCOPIES=IKS )

!WHERE ( GTRIG3(:,:) )
    PTTEN(:,:)  = _ZERO_
    PRVTEN(:,:) = _ZERO_
    PRCTEN(:,:) = _ZERO_
    PRITEN(:,:) = _ZERO_
    PUTEN(:,:)  = _ZERO_
    PVTEN(:,:)  = _ZERO_
    PUMF(:,:)   = _ZERO_
    PURV(:,:)   = _ZERO_
    PURCOUT(:,:)  = _ZERO_
    PURIOUT(:,:)  = _ZERO_
    PUDR(:,:)   = _ZERO_
    PCLOUD (:,:) =_ZERO_
!END WHERE

!WHERE ( GTRIG(:) )
    KCLTOP(:)  = 1 
    KCLBAS(:)  = 1
    PKSHAL(:)  = 0.
!END WHERE

IF ( OCH1CONV ) THEN
    ALLOCATE( GTRIG4(KLON,KLEV,KCH1) )
    GTRIG4(:,:,:) = SPREAD( GTRIG3(:,:), DIM=3, NCOPIES=KCH1 )
    WHERE( GTRIG4(:,:,:) ) PCH1TEN(:,:,:) = _ZERO_
    DEALLOCATE( GTRIG4 )
ENDIF


!*       1.     Initialize  local variables
!               ----------------------------

ZEPS   = RD  / RV
ZEPSA  = RV  / RD
ZEPSB  = RCPV / RCPD - ZEPSA
ZCPORD = RCPD / RD
ZRDOCP = RD  / RCPD


!*       1.1    Set up grid scale theta, theta_v, theta_es
!               ------------------------------------------

ZTHT  (:,:) = 300._JPRB
ZSTHV (:,:) = 300._JPRB
ZSTHES(:,:) = 400._JPRB

DO JK = IKB, IKE
DO JI = IIB, IIE
   IF ( PPABST(JI,JK) > 40.E2_JPRB ) THEN
      ZTHT(JI,JK)  = PTT(JI,JK) * ( RATM / PPABST(JI,JK) ) ** ZRDOCP
      ZSTHV(JI,JK) = ZTHT(JI,JK) * ( _ONE_ + ZEPSA * PRVT(JI,JK) ) /         &
                   & ( _ONE_ + PRVT(JI,JK) + PRCT(JI,JK) + PRIT(JI,JK) )

          ! use conservative Bolton (1980) formula for theta_e
          ! it is used to compute CAPE for undilute parcel ascent
          ! For economical reasons we do not use routine CONVECT_SATMIXRATIO here

      ZES = EXP( RALPW - RBETW / PTT(JI,JK) - RGAMW * LOG( PTT(JI,JK) ) )
      ZES = MIN( _ONE_, ZEPS * ZES / ( PPABST(JI,JK) - ZES ) )
      ZSTHES(JI,JK) = PTT(JI,JK) * ( ZTHT(JI,JK) / PTT(JI,JK) ) **           &
                   &  ( _ONE_ - 0.28_JPRB * ZES )                            &
                   &    * EXP( ( 3374.6525_JPRB / PTT(JI,JK) - 2.5403_JPRB ) &
                   &    * ZES * ( _ONE_ + 0.81_JPRB * ZES )  )
   ENDIF
ENDDO
ENDDO



!*       2.     Test for convective columns and determine properties at the LCL
!               --------------------------------------------------------------

!*       2.1    Allocate arrays depending on number of model columns that need
!               to be tested for convection (i.e. where no convection is present
!               at the moment.
!               --------------------------------------------------------------

  ALLOCATE( ZPRES(ITEST,IKS) )
  ALLOCATE( ZZ(ITEST,IKS) )
  ALLOCATE( ZW(ITEST,IKS) )
  ALLOCATE( ZTH(ITEST,IKS) )
  ALLOCATE( ZTHV(ITEST,IKS) )
  ALLOCATE( ZTHEST(ITEST,IKS) )
  ALLOCATE( ZRV(ITEST,IKS) )
  ALLOCATE( ZSTHLCL(ITEST) )
  ALLOCATE( ZSTLCL(ITEST) )
  ALLOCATE( ZSRVLCL(ITEST) )
  ALLOCATE( ZSWLCL(ITEST) )
  ALLOCATE( ZSZLCL(ITEST) )
  ALLOCATE( ZSTHVELCL(ITEST) )
  ALLOCATE( ISDPL(ITEST) )
  ALLOCATE( ISPBL(ITEST) )
  ALLOCATE( ISLCL(ITEST) )
  ALLOCATE( ZSDXDY(ITEST) )
  ALLOCATE( GTRIG1(ITEST) )
  ALLOCATE( IINDEX(KLON) )
  ALLOCATE( IJSINDEX(ITEST) )

  DO JI = 1, KLON
    IINDEX(JI) = JI
  ENDDO

  IJSINDEX(:) = PACK( IINDEX(:), MASK=GTRIG(:) )

  DO JK = IKB, IKE
  DO JI = 1, ITEST
    JL = IJSINDEX(JI)
    ZPRES(JI,JK)  = PPABST(JL,JK)
    ZZ(JI,JK)     = PZZ(JL,JK)
    ZTH(JI,JK)    = ZTHT(JL,JK)
    ZTHV(JI,JK)   = ZSTHV(JL,JK)
    ZTHEST(JI,JK) = ZSTHES(JL,JK)
    ZRV(JI,JK)    = MAX( _ZERO_, PRVT(JL,JK) )
    ZW(JI,JK)     = PWT(JL,JK)
  ENDDO
  ENDDO
  DO JI = 1, ITEST
    JL = IJSINDEX(JI)
    ZSDXDY(JI)    = XA25
  ENDDO

!*       2.2    Compute environm. enthalpy and total water = r_v + r_i + r_c
!               and envir. saturation theta_e
!               ------------------------------------------------------------


!*       2.3    Test for convective columns and determine properties at the LCL
!               --------------------------------------------------------------

  ISLCL(:) = MAX( IKB, 2 )   ! initialize DPL PBL and LCL
  ISDPL(:) = IKB
  ISPBL(:) = IKB

  CALL CONVECT_TRIGGER_SHAL(  ITEST, KLEV,                              &
                           &  ZPRES, ZTH, ZTHV, ZTHEST,                 &
                           &  ZRV, ZW, ZZ, ZSDXDY,                      &
                           &  ZSTHLCL, ZSTLCL, ZSRVLCL, ZSWLCL, ZSZLCL, &
                           &  ZSTHVELCL, ISLCL, ISDPL, ISPBL, GTRIG1    )

  DEALLOCATE( ZPRES )
  DEALLOCATE( ZZ )
  DEALLOCATE( ZTH )
  DEALLOCATE( ZTHV )
  DEALLOCATE( ZTHEST )
  DEALLOCATE( ZRV )
  DEALLOCATE( ZW )


!*       3.     After the call of TRIGGER_FUNCT we allocate all the dynamic
!               arrays used in the convection scheme using the mask GTRIG, i.e.
!               we do calculus only in convective columns. This corresponds to
!               a GATHER operation.
!               --------------------------------------------------------------

  ICONV = COUNT( GTRIG1(:) )
  IF ( ICONV == 0 )  THEN
      DEALLOCATE( ZSTHLCL )
      DEALLOCATE( ZSTLCL )
      DEALLOCATE( ZSRVLCL )
      DEALLOCATE( ZSWLCL )
      DEALLOCATE( ZSZLCL )
      DEALLOCATE( ZSTHVELCL )
      DEALLOCATE( ZSDXDY )
      DEALLOCATE( ISLCL )
      DEALLOCATE( ISDPL )
      DEALLOCATE( ISPBL )
      DEALLOCATE( GTRIG1 )
      DEALLOCATE( IINDEX )
      DEALLOCATE( IJSINDEX )
      RETURN   ! no convective column has been found, exit CONVECT_SHALLOW 
  ENDIF

   ! vertical index variables

   ALLOCATE( IDPL(ICONV) )
   ALLOCATE( IPBL(ICONV) )
   ALLOCATE( ILCL(ICONV) )
   ALLOCATE( ICTL(ICONV) )
   ALLOCATE( IETL(ICONV) )

   ! grid scale variables

   ALLOCATE( ZZ(ICONV,IKS) )
   ALLOCATE( ZPRES(ICONV,IKS) )
   ALLOCATE( ZDPRES(ICONV,IKS) )
   ALLOCATE( ZTT(ICONV, IKS) )
   ALLOCATE( ZTH(ICONV,IKS) )
   ALLOCATE( ZTHV(ICONV,IKS) )
   ALLOCATE( ZTHL(ICONV,IKS) )
   ALLOCATE( ZTHES(ICONV,IKS) )
   ALLOCATE( ZRV(ICONV,IKS) )
   ALLOCATE( ZRC(ICONV,IKS) )
   ALLOCATE( ZRI(ICONV,IKS) )
   ALLOCATE( ZRW(ICONV,IKS) )
   ALLOCATE( ZDXDY(ICONV) )
   ALLOCATE( ZU(ICONV,IKS) )
   ALLOCATE( ZV(ICONV,IKS) )

   ! updraft variables

   ALLOCATE( ZUMF(ICONV,IKS) )
   ALLOCATE( ZUER(ICONV,IKS) )
   ALLOCATE( ZUDR(ICONV,IKS) )
   ALLOCATE( ZUTHL(ICONV,IKS) )
   ALLOCATE( ZUTHV(ICONV,IKS) )
   ALLOCATE( ZWU(ICONV,IKS) )
   ALLOCATE( ZURW(ICONV,IKS) )
   ALLOCATE( ZURC(ICONV,IKS) )
   ALLOCATE( ZURI(ICONV,IKS) )
   ALLOCATE( ZTHLCL(ICONV) )
   ALLOCATE( ZTLCL(ICONV) )
   ALLOCATE( ZRVLCL(ICONV) )
   ALLOCATE( ZWLCL(ICONV) )
   ALLOCATE( ZMFLCL(ICONV) )
   ALLOCATE( ZZLCL(ICONV) )
   ALLOCATE( ZTHVELCL(ICONV) )
   ALLOCATE( ZCAPE(ICONV) )

   ! work variables

   ALLOCATE( IJINDEX(ICONV) )
   ALLOCATE( IJPINDEX(ICONV) )
   ALLOCATE( ZCPH(ICONV) )
   ALLOCATE( ZLV(ICONV) )
   ALLOCATE( ZLS(ICONV) )


!*           3.1    Gather grid scale and updraft base variables in
!                   arrays using mask GTRIG
!                   ---------------------------------------------------

    GTRIG(:)      = UNPACK( GTRIG1(:), MASK=GTRIG(:), FIELD=.FALSE. )
    IJINDEX(:)    = PACK( IINDEX(:), MASK=GTRIG(:) )

    DO JK = IKB, IKE
    DO JI = 1, ICONV
         JL = IJINDEX(JI)
         ZZ(JI,JK)     = PZZ(JL,JK)
         ZPRES(JI,JK)  = PPABST(JL,JK)
         ZTT(JI,JK)    = PTT(JL,JK)
         ZTH(JI,JK)    = ZTHT(JL,JK)
         ZTHES(JI,JK)  = ZSTHES(JL,JK)
         ZRV(JI,JK)    = MAX( _ZERO_, PRVT(JL,JK) )
         ZRC(JI,JK)    = MAX( _ZERO_, PRCT(JL,JK) )
         ZRI(JI,JK)    = MAX( _ZERO_, PRIT(JL,JK) )
         ZTHV(JI,JK)   = ZSTHV(JL,JK)
         ZU(JI,JK)     = PUT(JL,JK)
         ZV(JI,JK)     = PVT(JL,JK)
    ENDDO
    ENDDO

    DO JI = 1, ITEST
       IJSINDEX(JI) = JI
    ENDDO
    IJPINDEX(:) = PACK( IJSINDEX(:), MASK=GTRIG1(:) )
    DO JI = 1, ICONV
         JL = IJPINDEX(JI)
         IDPL(JI)      = ISDPL(JL)
         IPBL(JI)      = ISPBL(JL)
         ILCL(JI)      = ISLCL(JL)
         ZTHLCL(JI)    = ZSTHLCL(JL)
         ZTLCL(JI)     = ZSTLCL(JL)
         ZRVLCL(JI)    = ZSRVLCL(JL)
         ZWLCL(JI)     = ZSWLCL(JL)
         ZZLCL(JI)     = ZSZLCL(JL)
         ZTHVELCL(JI)  = ZSTHVELCL(JL)
         ZDXDY(JI)     = ZSDXDY(JL)
    ENDDO

    ALLOCATE( GWORK(ICONV) )
    GWORK(:)  = PACK( GTRIG1(:),  MASK=GTRIG1(:) )
    DEALLOCATE( GTRIG1 )
    ALLOCATE( GTRIG1(ICONV) )
    GTRIG1(:) = GWORK(:)

    DEALLOCATE( GWORK )
    DEALLOCATE( IJPINDEX )
    DEALLOCATE( ISDPL )
    DEALLOCATE( ISPBL )
    DEALLOCATE( ISLCL )
    DEALLOCATE( ZSTHLCL )
    DEALLOCATE( ZSTLCL )
    DEALLOCATE( ZSRVLCL )
    DEALLOCATE( ZSWLCL )
    DEALLOCATE( ZSZLCL )
    DEALLOCATE( ZSTHVELCL )
    DEALLOCATE( ZSDXDY )


!*           3.2    Compute pressure difference
!                   ---------------------------------------------------

        ZDPRES(:,IKB) = _ZERO_
        DO JK = IKB + 1, IKE
            ZDPRES(:,JK)  = ZPRES(:,JK-1) - ZPRES(:,JK)
        ENDDO

!*           3.3   Compute environm. enthalpy and total water = r_v + r_i + r_c
!                  ----------------------------------------------------------

        DO JK = IKB, IKE, 1
            ZRW(:,JK)  = ZRV(:,JK) + ZRC(:,JK) + ZRI(:,JK)
            ZCPH(:)    = RCPD + RCPV * ZRW(:,JK)
            ZLV(:)     = RLVTT + ( RCPV - RCW ) * ( ZTT(:,JK) - RTT ) ! compute L_v
            ZLS(:)     = RLSTT + ( RCPV - RCS ) * ( ZTT(:,JK) - RTT ) ! compute L_i
            ZTHL(:,JK) = ZCPH(:) * ZTT(:,JK) + ( _ONE_ + ZRW(:,JK) ) * RG * ZZ(:,JK) &
                       & - ZLV(:) * ZRC(:,JK) - ZLS(:) * ZRI(:,JK)
        ENDDO

        DEALLOCATE( ZCPH )
        DEALLOCATE( ZLV )
        DEALLOCATE( ZLS )


!*           4.     Compute updraft properties
!                   ----------------------------

!*           4.1    Set mass flux at LCL ( here a unit mass flux with w = 1 m/s )
!                   -------------------------------------------------------------

         ZDXDY(:)  = XA25
         ZMFLCL(:) = XA25 * 1.e-2_JPRB


     CALL CONVECT_UPDRAFT_SHAL( ICONV, KLEV,                                     &
                              & KICE, ZPRES, ZDPRES, ZZ, ZTHL, ZTHV, ZTHES, ZRW, &
                              & ZTHLCL, ZTLCL, ZRVLCL, ZWLCL, ZZLCL, ZTHVELCL,   &
                              & ZMFLCL, GTRIG1, ILCL, IDPL, IPBL,                &
                              & ZUMF, ZUER, ZUDR, ZUTHL, ZUTHV, ZWU,             &
                              & ZURW, ZURC, ZURI, ZCAPE, ICTL, IETL                    )



!*           4.2    In routine UPDRAFT GTRIG1 has been set to false when cloud
!                   thickness is smaller than 3 km
!                   -----------------------------------------------------------


     ICONV1 = COUNT(GTRIG1)

     IF ( ICONV1 > 0 )  THEN

!*       4.3    Allocate memory for downdraft variables
!               ---------------------------------------

! downdraft variables

        ALLOCATE( ZDMF(ICONV,IKS) )
        ALLOCATE( ZDER(ICONV,IKS) )
        ALLOCATE( ZDDR(ICONV,IKS) )
        ALLOCATE( ILFS(ICONV) )
        ALLOCATE( ZLMASS(ICONV,IKS) )
        ZDMF(:,:) = _ZERO_
        ZDER(:,:) = _ZERO_
        ZDDR(:,:) = _ZERO_
        ILFS(:)   = IKB
        DO JK = IKB, IKE
           ZLMASS(:,JK)  = ZDXDY(:) * ZDPRES(:,JK) / RG  ! mass of model layer
        ENDDO
        ZLMASS(:,IKB) = ZLMASS(:,IKB+1)

! closure variables

        ALLOCATE( ZTIMEC(ICONV) )
        ALLOCATE( ZTHC(ICONV,IKS) )
        ALLOCATE( ZRVC(ICONV,IKS) )
        ALLOCATE( ZRCC(ICONV,IKS) )
        ALLOCATE( ZRIC(ICONV,IKS) )
        ALLOCATE( ZWSUB(ICONV,IKS) )


!*           5.     Compute downdraft properties
!                   ----------------------------

        ZTIMEC(:) = XCTIME_SHAL
        IF ( OSETTADJ ) ZTIMEC(:) = PTADJS

!*           7.     Determine adjusted environmental values assuming
!                   that all available buoyant energy must be removed
!                   within an advective time step ZTIMEC.
!                   ---------------------------------------------------

       CALL CONVECT_CLOSURE_SHAL( ICONV, KLEV,                         &
                                & ZPRES, ZDPRES, ZZ, ZDXDY, ZLMASS,    &
                                & ZTHL, ZTH, ZRW, ZRC, ZRI, GTRIG1,    &
                                & ZTHC, ZRVC, ZRCC, ZRIC, ZWSUB,       &
                                & ILCL, IDPL, IPBL, ICTL,              &
                                & ZUMF, ZUER, ZUDR, ZUTHL, ZURW,       &
                                & ZURC, ZURI, ZCAPE, ZTIMEC, IFTSTEPS  )




!*           8.     Determine the final grid-scale (environmental) convective
!                   tendencies and set convective counter
!                   --------------------------------------------------------


!*           8.1    Grid scale tendencies
!                   ---------------------

          ! in order to save memory, the tendencies are temporarily stored
          ! in the tables for the adjusted grid-scale values


      DO JK = IKB, IKE
         ZTHC(:,JK) = ( ZTHC(:,JK) - ZTH(:,JK) ) / ZTIMEC(:)             &
         & * ( ZPRES(:,JK) / RATM ) ** ZRDOCP ! change theta in temperature
         ZRVC(:,JK) = ( ZRVC(:,JK) - ZRW(:,JK) + ZRC(:,JK) + ZRI(:,JK) ) &
         &                                        / ZTIMEC(:)


         ZRCC(:,JK) = ( ZRCC(:,JK) - ZRC(:,JK) ) / ZTIMEC(:)
         ZRIC(:,JK) = ( ZRIC(:,JK) - ZRI(:,JK) ) / ZTIMEC(:)
      ENDDO


!*           8.2    Apply conservation correction
!                   -----------------------------

          ! adjustment at cloud top to smooth discontinuous profiles at PBL inversions
          ! (+ - - tendencies for moisture )


       DO JI = 1, ICONV
          JK = ICTL(JI)
          JKM= MAX(1,ICTL(JI)-1)
          JKP= MAX(1,ICTL(JI)-2)
          ZRVC(JI,JKM) = ZRVC(JI,JKM) + .5_JPRB * ZRVC(JI,JK)
          ZRCC(JI,JKM) = ZRCC(JI,JKM) + .5_JPRB * ZRCC(JI,JK)
          ZRIC(JI,JKM) = ZRIC(JI,JKM) + .5_JPRB * ZRIC(JI,JK)
          ZTHC(JI,JKM) = ZTHC(JI,JKM) + .5_JPRB * ZTHC(JI,JK)
          ZRVC(JI,JKP) = ZRVC(JI,JKP) + .3_JPRB * ZRVC(JI,JK)
          ZRCC(JI,JKP) = ZRCC(JI,JKP) + .3_JPRB * ZRCC(JI,JK)
          ZRIC(JI,JKP) = ZRIC(JI,JKP) + .3_JPRB * ZRIC(JI,JK)
          ZTHC(JI,JKP) = ZTHC(JI,JKP) + .3_JPRB * ZTHC(JI,JK)
          ZRVC(JI,JK)  = .2_JPRB * ZRVC(JI,JK)  
          ZRCC(JI,JK)  = .2_JPRB * ZRCC(JI,JK) 
          ZRIC(JI,JK)  = .2_JPRB * ZRIC(JI,JK)
          ZTHC(JI,JK)  = .2_JPRB * ZTHC(JI,JK)
       END DO
         

          ! compute vertical integrals - fluxes
       
       JKM = MAXVAL( ICTL(:) )
       ZWORK2(:) = _ZERO_
       ZWORK2B(:)= _ZERO_
       DO JK = IKB+1, JKM
         JKP = JK + 1
         DO JI = 1, ICONV
           ZW1 = _HALF_ *  (ZPRES(JI,JK-1) - ZPRES(JI,JKP)) / RG
           ZWORK2(JI) = ZWORK2(JI) + ( ZRVC(JI,JK) + ZRCC(JI,JK) + ZRIC(JI,JK) ) *   & ! moisture
              &                   ZW1
           ZWORK2B(JI) = ZWORK2B(JI) + (                                             & ! enthalpy
              &                        ( RCPD + RCPV * ZRW(JI,JK) )* ZTHC(JI,JK)   - &
              &  ( RLVTT + ( RCPV - RCW ) * ( ZTT(JI,JK) - RTT ) ) * ZRCC(JI,JK)   - &
              &  ( RLSTT + ( RCPV - RCS ) * ( ZTT(JI,JK) - RTT ) ) * ZRIC(JI,JK) ) * & !pv-bugfix-put rcs rather than rcw 
              &                   ZW1
         ENDDO
       ENDDO


          ! Budget error (integral must be zero)

       DO JI = 1, ICONV
           JKP = ICTL(JI) 
           IF ( ICTL(JI) > IKB+1 ) THEN
              ZW1 = RG / ( ZPRES(JI,IKB) - ZPRES(JI,JKP) - &
                           &_HALF_*(ZDPRES(JI,IKB+1) - ZDPRES(JI,JKP+1)) )
              ZWORK2(JI) =  ZWORK2(JI) * ZW1
              ZWORK2B(JI)= ZWORK2B(JI) * ZW1
           END IF
       ENDDO

          ! Apply uniform correction

       DO JK = JKM, IKB+1, -1
         DO JI = 1, ICONV
           IF ( ICTL(JI) > IKB+1 .AND. JK <= ICTL(JI) ) THEN
              ! ZW1 = ABS(ZRVC(JI,JK)) +  ABS(ZRCC(JI,JK)) +  ABS(ZRIC(JI,JK)) + 1.E-12_JPRB
                ZRVC(JI,JK) = ZRVC(JI,JK) - ZWORK2(JI)                                ! moisture
              ! ZRVC(JI,JK) = ZRVC(JI,JK) - ABS(ZRVC(JI,JK))/ZW1*ZWORK2(JI)           ! moisture
              ! ZRCC(JI,JK) = ZRCC(JI,JK) - ABS(ZRCC(JI,JK))/ZW1*ZWORK2(JI)
              ! ZRIC(JI,JK) = ZRIC(JI,JK) - ABS(ZRIC(JI,JK))/ZW1*ZWORK2(JI)
!PV                ZTHC(JI,JK) = ZTHC(JI,JK) + ZWORK2B(JI) / ( RCPD + RCPV * ZRW(JI,JK) )! energy
!PV        this correction must be checked energy vs enthalphy???
                ZTHC(JI,JK) = ZTHC(JI,JK) - ZWORK2B(JI) / ( RCPD + RCPV * ZRW(JI,JK) )
           END IF
         ENDDO
       ENDDO


          ! extend tendencies to first model level

    ! DO JI = 1, ICONV
    !    ZWORK2(JI)    = ZDPRES(JI,IKB+1) + ZDPRES(JI,IKB+2)
    !    ZTHC(JI,IKB)  = ZTHC(JI,IKB+1) * ZDPRES(JI,IKB+2)/ZWORK2(JI)
    !    ZTHC(JI,IKB+1)= ZTHC(JI,IKB+1) * ZDPRES(JI,IKB+1)/ZWORK2(JI)
    !    ZRVC(JI,IKB)  = ZRVC(JI,IKB+1) * ZDPRES(JI,IKB+2)/ZWORK2(JI)
    !    ZRVC(JI,IKB+1)= ZRVC(JI,IKB+1) * ZDPRES(JI,IKB+1)/ZWORK2(JI)
    ! END DO


              ! execute a "scatter"= pack command to store the tendencies in
              ! the final 2D tables

      DO JK = IKB, IKE
      DO JI = 1, ICONV
         JL = IJINDEX(JI)
         PTTEN(JL,JK)   = ZTHC(JI,JK)
         PRVTEN(JL,JK)  = ZRVC(JI,JK)
         PRCTEN(JL,JK)  = ZRCC(JI,JK)
         PRITEN(JL,JK)  = ZRIC(JI,JK)

         PURV(JL,JK)    = ZURW(JI,JK) - ZURC(JI,JK) - ZURI(JI,JK)
      ENDDO
      ENDDO


!                   Cloud base and top levels
!                   -------------------------

     ILCL(:) = MIN( ILCL(:), ICTL(:) )
     DO JI = 1, ICONV
        JL = IJINDEX(JI)
        KCLTOP(JL) = ICTL(JI)
        KCLBAS(JL) = ILCL(JI)
        IF(GTRIG1(JI)) PKSHAL(JL) = 1.
     ENDDO


!*           8.7    Compute convective tendencies for Tracers
!                   ------------------------------------------

  IF ( OCH1CONV ) THEN

       ALLOCATE( ZCH1(ICONV,IKS,KCH1) )
       ALLOCATE( ZCH1C(ICONV,IKS,KCH1) )
       ALLOCATE( ZWORK3(ICONV,KCH1) )

       DO JK = IKB, IKE
       DO JI = 1, ICONV
          JL = IJINDEX(JI)
          ZCH1(JI,JK,:) = PCH1(JL,JK,:)
       ENDDO
       ENDDO

      CALL CONVECT_CHEM_TRANSPORT( ICONV, KLEV, KCH1, ZCH1, ZCH1C,          &
                                &  IDPL, IPBL, ILCL, ICTL, ILFS, ILFS,      &
                                &  ZUMF, ZUER, ZUDR, ZDMF, ZDER, ZDDR,      &
                                &  ZTIMEC, ZDXDY, ZDMF(:,1), ZLMASS, ZWSUB, &
                                &  IFTSTEPS )

       DO JK = IKB, IKE
       DO JN = 1, KCH1
          ZCH1C(:,JK,JN) = ( ZCH1C(:,JK,JN)- ZCH1(:,JK,JN) ) / ZTIMEC(:)
       ENDDO
       ENDDO


!*           8.8    Apply conservation correction
!                   -----------------------------

          ! Compute vertical integrals

       JKM = MAXVAL( ICTL(:) )
       ZWORK3(:,:) = _ZERO_
       DO JK = IKB, JKM+1
         JKP = JK + 1
         DO JI = 1, ICONV
           ZWORK3(JI,:) = ZWORK3(JI,:) + ZCH1C(JI,JK,:) *                    &
                        &   (ZPRES(JI,JK) - ZPRES(JI,JKP)) / RG
         ENDDO
       ENDDO

          ! Mass error (integral must be zero)

       DO JI = 1, ICONV
           JKP = ICTL(JI) + 1
           IF ( ICTL(JI) > IKB+1 ) THEN
              ZW1 = RG / ( ZPRES(JI,IKB) - ZPRES(JI,JKP) - &
                       & _HALF_*(ZDPRES(JI,IKB+1) - ZDPRES(JI,JKP+1)) )
              ZWORK3(JI,:) = ZWORK3(JI,:) * ZW1
           END IF
       ENDDO

          ! Apply uniform correction but assure positive mass at each level

       DO JK = JKM, IKB, -1
         DO JI = 1, ICONV
           IF ( ICTL(JI) > IKB+1 .AND. JK <= ICTL(JI) ) THEN
                ZCH1C(JI,JK,:) = ZCH1C(JI,JK,:) - ZWORK3(JI,:)
              ! ZCH1C(JI,JK,:) = MAX( ZCH1C(JI,JK,:), -ZCH1(JI,JK,:)/ZTIMEC(JI) )
           ENDIF
         ENDDO
       ENDDO

          ! extend tendencies to first model level

    !  DO JI = 1, ICONV
    !     ZWORK2(JI) = ZDPRES(JI,IKB+1) + ZDPRES(JI,IKB+2)
    !  ENDDO
    !  DO JN = 1, KCH1
    !  DO JI = 1, ICONV
    !    ZCH1(JI,IKB,JN)  = ZCH1(JI,IKB+1,JN) * ZDPRES(JI,IKB+2)/ZWORK2(JI)
    !    ZCH1(JI,IKB+1,JN)= ZCH1(JI,IKB+1,JN) * ZDPRES(JI,IKB+1)/ZWORK2(JI)
    !  ENDDO
    !  ENDDO

       DO JK = IKB, IKE
       DO JI = 1, ICONV
          JL = IJINDEX(JI)
          PCH1TEN(JL,JK,:) = ZCH1C(JI,JK,:)
       ENDDO
       ENDDO
  ENDIF

!
!*           8.9    Compute convective tendencies for wind
!                   --------------------------------------

  IF ( OUVCONV ) THEN

       ALLOCATE( ZUC(ICONV,IKS) )
       ALLOCATE( ZVC(ICONV,IKS) )

       CALL CONVECT_UV_TRANSPORT( ICONV, KLEV, ZU, ZV, ZUC, ZVC,            &
                                & IDPL, IPBL, ILCL, ICTL, ILFS, ILFS,       &
                                & ZUMF, ZUER, ZUDR, ZDMF, ZDER, ZDDR,       &
                                & ZTIMEC, ZDXDY, ZDMF(:,1), ZLMASS, ZWSUB,  &
                                & IFTSTEPS )

       DO JK = IKB, IKE
          ZUC(:,JK) = ( ZUC(:,JK)- ZU(:,JK) ) / ZTIMEC(:)
          ZVC(:,JK) = ( ZVC(:,JK)- ZV(:,JK) ) / ZTIMEC(:)
       ENDDO

!*           8.91    Apply conservation correction
!                   -----------------------------

          ! Compute vertical integrals

       JKM = MAXVAL( ICTL(:) )
       ZWORK2(:) = _ZERO_
       ZWORK2B(:)= _ZERO_
       DO JK = IKB+1, JKM
         JKP = JK + 1
         DO JI = 1, ICONV
           ZW1 = _HALF_ *  (ZPRES(JI,JK-1) - ZPRES(JI,JKP)) / RG
           ZWORK2(JI) = ZWORK2(JI) + ZUC(JI,JK) * ZW1
           ZWORK2B(JI)= ZWORK2B(JI)+ ZVC(JI,JK) * ZW1
         ENDDO
       ENDDO

          !  error (integral must be zero)

       DO JI = 1, ICONV
         JKP = ICTL(JI) + 1
         ZW1 = RG / ( ZPRES(JI,IKB) - ZPRES(JI,JKP) - &
                     & _HALF_*(ZDPRES(JI,IKB+1) - ZDPRES(JI,JKP+1)) )
         ZWORK2(JI) = ZWORK2(JI) * ZW1
         ZWORK2B(JI)= ZWORK2B(JI)* ZW1
       ENDDO

          ! Apply uniform correction 

       DO JK = JKM, IKB, -1
         DO JI = 1, ICONV
           ZUC(JI,JK) = ZUC(JI,JK) - ZWORK2(JI)
           ZVC(JI,JK) = ZVC(JI,JK) - ZWORK2B(JI)
         ENDDO
       ENDDO
!
          ! extend tendencies to first model level
 
    ! DO JI = 1, ICONV
    !    ZWORK2(JI) = ZDPRES(JI,IKB+1) + ZDPRES(JI,IKB+2)
    !    ZUC(JI,IKB)  = ZUC(JI,IKB+1) * ZDPRES(JI,IKB+2)/ZWORK2(JI)
    !    ZUC(JI,IKB+1)= ZUC(JI,IKB+1) * ZDPRES(JI,IKB+1)/ZWORK2(JI)
    !    ZVC(JI,IKB)  = ZVC(JI,IKB+1) * ZDPRES(JI,IKB+2)/ZWORK2(JI)
    !    ZVC(JI,IKB+1)= ZVC(JI,IKB+1) * ZDPRES(JI,IKB+1)/ZWORK2(JI)
    ! ENDDO

       DO JK = IKB, IKE
       DO JI = 1, ICONV
          JL = IJINDEX(JI)
          PUTEN(JL,JK)   = ZUC(JI,JK)
          PVTEN(JL,JK)   = ZVC(JI,JK)
       ENDDO
       ENDDO
  ENDIF

!*           9.     Write up- and downdraft mass fluxes
!                   -----------------------------------

    DO JK = IKB, IKE
       ZUMF(:,JK)  = ZUMF(:,JK) / ZDXDY(:) ! Mass flux per unit area
    ENDDO
    DO JK = IKB+1, IKE
       ZUDR(:,JK)  = ZUDR(:,JK) / ( ZDXDY(:) * ZDPRES(:,JK) )! detrainment for ERA40
    ENDDO

    ZWORK2(:) = _ONE_
    DO JK = IKB, IKE
    DO JI = 1, ICONV
       JL = IJINDEX(JI)
       IF ( KCLTOP(JL) <= IKB+1 ) ZWORK2(JL) = _ZERO_
       PUMF(JL,JK) = ZUMF(JI,JK) * ZWORK2(JL)
!
!  Diagnose cloud coverage 
         ZW1 = ZUTHV(JI,JK)*RD /ZPRES(JI,JK)
       IF (ZWU(JI,JK) > 1.E-2_JPRB  ) THEN
         ZW1 = ZUTHV(JI,JK)*RD /ZPRES(JI,JK)
         PCLOUD(JL,JK) = PUMF(JL,JK)/( ZW1* ZWU(JI,JK))
         PCLOUD(JL,JK) = MAX(0.0, PCLOUD(JL,JK))
       ENDIF

! Cloud liquid and ice water mixing ratio in updraft, normalized by convective cloud
       PURCOUT(JL,JK) = ZURC(JI,JK)*PCLOUD(JL,JK)
       PURIOUT(JL,JK) = ZURI(JI,JK)*PCLOUD(JL,JK)

       PUDR(JL,JK) = ZUDR(JI,JK) * ZWORK2(JL)
       PURV(JL,JK) = PURV(JL,JK) * ZWORK2(JL)
    ENDDO
    ENDDO


!*           10.    Deallocate all local arrays
!                   ---------------------------

! downdraft variables

      DEALLOCATE( ZDMF )
      DEALLOCATE( ZDER )
      DEALLOCATE( ZDDR )
      DEALLOCATE( ILFS )
      DEALLOCATE( ZLMASS )
!
!   closure variables
!
      DEALLOCATE( ZTIMEC )
      DEALLOCATE( ZTHC )
      DEALLOCATE( ZRVC )
      DEALLOCATE( ZRCC )
      DEALLOCATE( ZRIC )
      DEALLOCATE( ZWSUB )
!
       IF ( OCH1CONV ) THEN
           DEALLOCATE( ZCH1 )
           DEALLOCATE( ZCH1C )
           DEALLOCATE( ZWORK3 )
       ENDIF
       IF ( OUVCONV ) THEN
           DEALLOCATE( ZUC )
           DEALLOCATE( ZVC )
       ENDIF
!
    ENDIF
!
!    vertical index
!
    DEALLOCATE( IDPL )
    DEALLOCATE( IPBL )
    DEALLOCATE( ILCL )
    DEALLOCATE( ICTL )
    DEALLOCATE( IETL )
!
! grid scale variables
!
    DEALLOCATE( ZZ )
    DEALLOCATE( ZPRES )
    DEALLOCATE( ZDPRES )
    DEALLOCATE( ZTT )
    DEALLOCATE( ZTH )
    DEALLOCATE( ZTHV )
    DEALLOCATE( ZTHL )
    DEALLOCATE( ZTHES )
    DEALLOCATE( ZRW )
    DEALLOCATE( ZRV )
    DEALLOCATE( ZRC )
    DEALLOCATE( ZRI )
    DEALLOCATE( ZDXDY )
    DEALLOCATE( ZU )
    DEALLOCATE( ZV )
!
! updraft variables
!
    DEALLOCATE( ZUMF )
    DEALLOCATE( ZUER )
    DEALLOCATE( ZUDR )
    DEALLOCATE( ZUTHL )
    DEALLOCATE( ZUTHV )
    DEALLOCATE( ZWU )
    DEALLOCATE( ZURW )
    DEALLOCATE( ZURC )
    DEALLOCATE( ZURI )
    DEALLOCATE( ZTHLCL )
    DEALLOCATE( ZTLCL )
    DEALLOCATE( ZRVLCL )
    DEALLOCATE( ZWLCL )
    DEALLOCATE( ZZLCL )
    DEALLOCATE( ZTHVELCL )
    DEALLOCATE( ZMFLCL )
    DEALLOCATE( ZCAPE )
!
! work arrays
!
    DEALLOCATE( IINDEX )
    DEALLOCATE( IJINDEX )
    DEALLOCATE( IJSINDEX )
    DEALLOCATE( GTRIG1 )
!
!
END SUBROUTINE CONVECT_SHALLOW
