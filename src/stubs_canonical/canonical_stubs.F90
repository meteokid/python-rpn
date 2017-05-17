integer function canonical_nml (F_namelistf_S, F_unout, F_dcmip_L, F_wil_L)
      implicit none
      character* (*) F_namelistf_S
      logical F_dcmip_L,F_wil_L
      integer F_unout
      canonical_nml = 1
      F_dcmip_L= .false. ; F_wil_L= .false.
      return
      end
subroutine canonical_cases
return
end
subroutine canonical_coriolis
return
end
subroutine canonical_terminator_0
return
end
subroutine canonical_terminator_1
return
end
subroutine canonical_terminator_2
return
end
subroutine wil_set
return
end
subroutine wil_init
return
end
subroutine dcmip_set
return
end
subroutine dcmip_init
return
end
subroutine dcmip_vrd_set
return
end
subroutine dcmip_setl_set
return
end
real*8 function dcmip_mult_X()
dcmip_mult_X=-1.
return
end
integer function dcmip_div_X()
dcmip_div_X=-1.
return
end
