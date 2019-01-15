
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   integer*4 function mw_vpopcnt8_s(x,n)

  ! Environment setup for a call to the corresponding vector MASS function
  use, intrinsic :: xlf_fp_util
  implicit none
  

  ! Argument list
        integer*8 x
      integer*4 n


  ! Construction-time additions
  integer*4, external :: vpopcnt8


  ! Internal variables
  integer(fpscr_kind) :: round_mode,fpscr_reset

  ! Save current state of FPSCR
  fpscr_reset = get_fpscr()

  ! Disable all FP traps and set rounding mode to nearest
  call clr_fpscr_flags(FP_ENBL_SUMM)
  round_mode = set_round_mode(FP_RND_RN)

  ! Vector accelerated math operation
  mw_vpopcnt8_s = vpopcnt8(x,n)

  ! Restore previous state of FPSCR
  fpscr_reset = set_fpscr(fpscr_reset)

end function mw_vpopcnt8_s
