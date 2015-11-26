      subroutine out_cfile3
      implicit none
#include <arch_specific.hf>
!
#include "out3.cdk"
#include "out.cdk"
!
      integer, external :: fstfrm
      integer err
      real dummy
!
!----------------------------------------------------------------------
!
      call out_fstecr3 ( dummy,dummy,dummy,dummy,dummy,dummy,&
                         dummy,dummy,dummy,dummy,dummy,dummy,&
                         dummy,dummy,dummy, .true. )

      if ( (Out3_iome .ge. 0) .and. (Out_unf .gt. 0) ) then
         err = fstfrm(Out_unf)
         call fclos(Out_unf)
         Out_unf = 0
      endif

 102  format (' FST FILE UNIT=',i3,' FILE = ',a,' IS CLOSED')
!----------------------------------------------------------------------
      return
      end
