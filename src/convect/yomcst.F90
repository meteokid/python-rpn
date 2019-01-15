MODULE YOMCST


#include "tsmbkind.cdk"

IMPLICIT NONE

SAVE

!     ------------------------------------------------------------------

!*    Common of physical constants
!     You will find the meanings in the annex 1 of the documentation

! A1.0 Fundamental constants
REAL_B :: RPI
REAL_B :: RCLUM
REAL_B :: RHPLA
REAL_B :: RKBOL
REAL_B :: RNAVO
! A1.1 Astronomical constants
REAL_B :: RDAY
REAL_B :: REA
REAL_B :: REPSM
REAL_B :: RSIYEA
REAL_B :: RSIDAY
REAL_B :: ROMEGA
! A1.2 Geoide
REAL_B :: RA
REAL_B :: RG
REAL_B :: R1SA
! A1.3 Radiation
REAL_B :: RSIGMA
REAL_B :: RI0
! A1.4 Thermodynamic gas phase
REAL_B :: R
REAL_B :: RMD
REAL_B :: RMV
REAL_B :: RMO3
REAL_B :: RD
REAL_B :: RV
REAL_B :: RCPD
REAL_B :: RCPV
REAL_B :: RCVD
REAL_B :: RCVV
REAL_B :: RKAPPA
REAL_B :: RETV
! A1.5,6 Thermodynamic liquid,solid phases
REAL_B :: RCW
REAL_B :: RCS
! A1.7 Thermodynamic transition of phase
REAL_B :: RLVTT
REAL_B :: RLSTT
REAL_B :: RLVZER
REAL_B :: RLSZER
REAL_B :: RLMLT
REAL_B :: RTT
REAL_B :: RATM
REAL_B :: RDT
! A1.8 Curve of saturation
REAL_B :: RESTT
REAL_B :: RALPW
REAL_B :: RBETW
REAL_B :: RGAMW
REAL_B :: RALPS
REAL_B :: RBETS
REAL_B :: RGAMS
REAL_B :: RALPD
REAL_B :: RBETD
REAL_B :: RGAMD


!    ------------------------------------------------------------------
END MODULE YOMCST
