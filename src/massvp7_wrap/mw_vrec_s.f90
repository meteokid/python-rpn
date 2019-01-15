
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   subroutine mw_vrec_s(z,x,n)

  ! Environment setup for a call to the corresponding vector MASS function
  use, intrinsic :: xlf_fp_util
  implicit none
  

  ! Argument list
        real*8 z,x
      integer*4 n


  ! Construction-time additions
  

  ! Internal variables
  integer(fpscr_kind) :: round_mode,fpscr_reset

  ! Save current state of FPSCR
  fpscr_reset = get_fpscr()

  ! Disable all FP traps and set rounding mode to nearest
  call clr_fpscr_flags(FP_ENBL_SUMM)
  round_mode = set_round_mode(FP_RND_RN)

  ! Vector accelerated math operation
  call vrec(z,x,n)

  ! Restore previous state of FPSCR
  fpscr_reset = set_fpscr(fpscr_reset)

end subroutine mw_vrec_s
