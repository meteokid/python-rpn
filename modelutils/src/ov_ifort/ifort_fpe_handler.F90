
!#TODO: move to stub-able, arch specific lib
subroutine fpe_setup
   use ifport
   implicit none
   interface
      subroutine fpe_handler(signo, siginfo)
         integer(4), intent(in) :: signo, siginfo
      end subroutine fpe_handler
   end interface
   integer :: ir
   ir = ieee_handler('set', 'invalid',  fpe_handler)
   ir = ieee_handler('set', 'division', fpe_handler)
   ir = ieee_handler('set', 'overflow', fpe_handler)
!!$   ir = ieee_handler('set', 'undeflow', fpe_handler)
!!$   ir = ieee_handler('set', 'inexact',  fpe_handler)
   return
end subroutine fpe_setup


subroutine fpe_handler(sig, code)
   use ifport
   implicit none
   integer :: sig, code
   if (code == FPE$INVALID .or. FPE$ZERODIVIDE .or. FPE$OVERFLOW) then
      call msg_buffer_flush()
   endif
!!$   if (code == FPE$UNDERFLOW) print *,'occurred divide by zero.'
!!$   if (code == FPE$INEXACT) print *,'occurred divide by zero.'
   call abort()
end subroutine fpe_handler
