!     ######################
      MODULE YOE_CONVPAREXT
!     ######################

#include "tsmbkind.cdk"

IMPLICIT NONE

INTEGER_M, SAVE :: JCVEXB ! start vertical computations at
                          ! 1 + JCVEXB = 1 + ( KBDIA - 1 )
INTEGER_M, SAVE :: JCVEXT ! limit vertical computations to
                          ! KLEV - JCVEXT = KLEV - ( KTDIA - 1 )

END MODULE YOE_CONVPAREXT

