MODULE YOMRIP


#include "tsmbkind.cdk"

IMPLICIT NONE

SAVE

!     ------------------------------------------------------------------

!*    Real time related variables (updated in UPDTIM)

!     NINDAT : run initial date in the form AAAAMMDD
!     NSSSSS : initial time in seconds (e.g. for 12h, 43200)
!     RTIMST : ABSOLUTE TIME OF THE MODEL AT START

!     NSTADD : NUMBER OF DAYS SINCE START OF THE MODEL
!     NSTASS : NUMBER OF SECONDS since start of model modulo(86400)
!     RSTATI : NUMBER OF SECONDS SINCE START OF THE MODEL
!     RTIMTR : ABSOLUTE TIME OF THE MODEL

!     RHGMT  : GMT TIME OF THE MODEL  (BETWEEN 0 AND 86400)
!     REQTIM : EQUATION OF TIME

!     ------------------------------------------------------------------

!*    Sun related variables (updated in UPDTIM)
!     RSOVR  : TRUE SOLAR TIME (GMT+EQUATION OF TIME)

!     RDEASO : DISTANCE EARTH-SUN
!     RDECLI : DECLINATION
!     RWSOVR : IN RADIANS, TRUE SOLAR TIME (GMT+EQUATION OF TIME)
!              THIS ANGLE IS ALSO PI - (LONGITUDE OF THE POINT
!              WHERE THE SUN IS AT ZENITH)
!     RIP0   : I0 WEIGHTED BY THE DISTANCE EARTH-SUN

!     RCODEC : COSINE OF THE DECLINATION
!     RSIDEC :   SINE OF THE DECLINATION

!     RCOVSR : COSINE OF TRUE SOLAR TIME
!     RSIVSR :   SINE OF TRUE SOLAR TIME


!     ------------------------------------------------------------------

!*    Moon related variables (updated in UPDTIM)

!     RDECLU : LUNAR DECLINATION
!     RTMOLT : IN RADIANS, TRUE LUNAR TIME (GMT+EQUATION OF TIME)
!              THIS ANGLE IS ALSO PI - (LONGITUDE OF THE POINT
!              WHERE THE MOON IS AT ZENITH)
!     RIP0LU : LUNAR I0 (DOWNWARD TOA LUNAR FLUX)

!     RCODECLU : COSINE OF THE LUNAR DECLINATION
!     RSIDECLU :   SINE OF THE LUNAR DECLINATION

!     RCOVSRLU : COSINE OF TRUE LUNAR TIME
!     RSIVSRLU :   SINE OF TRUE LUNAR TIME

!     ------------------------------------------------------------------

!*    Time step related variables

!     RDTSA  : TDT  /RA
!     RDTSA2 : RDTSA**2
!     RDTS62 : RDTSA**2/6
!     RDTS22 : RDTSA**2/2

!     RTDT   : TDT


INTEGER_M :: NINDAT
INTEGER_M :: NSSSSS
INTEGER_M :: NSTADD
INTEGER_M :: NSTASS
REAL_B :: RTIMST
REAL_B :: RSTATI
REAL_B :: RTIMTR
REAL_B :: RHGMT
REAL_B :: REQTIM
REAL_B :: RSOVR
REAL_B :: RDEASO
REAL_B :: RDECLI
REAL_B :: RWSOVR
REAL_B :: RIP0
REAL_B :: RCODEC
REAL_B :: RSIDEC
REAL_B :: RCOVSR
REAL_B :: RSIVSR
REAL_B :: RDTSA
REAL_B :: RDTSA2
REAL_B :: RDTS62
REAL_B :: RDTS22
REAL_B :: RTDT
REAL_B :: RDECLU
REAL_B :: RTMOLT
REAL_B :: RIP0LU
REAL_B :: RCODECLU
REAL_B :: RSIDECLU
REAL_B :: RCOVSRLU
REAL_B :: RSIVSRLU
!     ------------------------------------------------------------------
END MODULE YOMRIP
