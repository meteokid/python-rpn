MODULE YOE_CONVECT


#include "tsmbkind.cdk"
 
IMPLICIT NONE
 
SAVE
! maybe specifiy values later in su0phy.F90
CHARACTER*8:: CCUMFSCHEM 

INTEGER_M :: NBDIA            ! starting level (from bottom=1) for convection computat.
INTEGER_M :: NENSM            ! number of additional ensemble members for deep
INTEGER_M :: NCH1  ! number of chemical tracers
INTEGER_M :: NICE             ! use ice computations or only liquid (1=with ice 0 without)

LOGICAL   :: LDEEP ! LMFPEN   ! switch for deep convection
LOGICAL   :: LSHAL ! LSHCV    ! switch for shallow convection
LOGICAL   :: LDOWN ! LMFDD    ! take or not convective downdrafts into account 
LOGICAL   :: LREFRESH         ! refresh convective tendencies at every time step
LOGICAL   :: LSETTADJ         ! logical to set convective adjustment time by user
LOGICAL   :: LUVTRANS         ! flag to compute convective transport for hor. wind
LOGICAL   :: LCHTRANS         ! flag to compute convective transport for chemical tracers
REAL_B    :: RTADJD           ! user defined deep  adjustment time (s) (if LSETTADJ)
REAL_B    :: RTADJS           ! user defined shallow adjustment time (s) (if LSETTADJ)
LOGICAL   :: LDIAGCONV2D      ! logical for convection 2D diagnostics
LOGICAL   :: LDIAGCONV3D      ! logical for convection 3D diagnostics

END MODULE YOE_CONVECT
