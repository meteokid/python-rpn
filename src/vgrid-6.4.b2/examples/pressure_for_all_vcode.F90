!/* libdescrip - Vertical grid descriptor library for FORTRAN programming
! * Copyright (C) 2016  Direction du developpement des previsions nationales
! *                     Centre meteorologique canadien
! *
! * This library is free software; you can redistribute it and/or
! * modify it under the terms of the GNU Lesser General Public
! * License as published by the Free Software Foundation,
! * version 2.1 of the License.
! *
! * This library is distributed in the hope that it will be useful,
! * but WITHOUT ANY WARRANTY; without even the implied warranty of
! * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! * Lesser General Public License for more details.
! *
! * You should have received a copy of the GNU Lesser General Public
! * License along with this library; if not, write to the
! * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
! * Boston, MA 02111-1307, USA.
program pressure_for_all_vcode

   use vGrid_Descriptors, only: vgd_new, vgd_levels, vgd_putopt, VGD_ERROR

   implicit none
   integer :: i, comp_pres
   integer, parameter :: nfiles=11 
   character(len=200), dimension(nfiles) :: files=(/&
        "../tests/data_Linux/dm_1001_from_model_run",&
        "../tests/data_Linux/dm_1002_from_model_run",&
        "../tests/data_Linux/dm_2001_from_editfst",&
        "../tests/data_Linux/dm_5001_from_model_run",&
        "../tests/data_Linux/dm_5002_from_model_run",&
        "../tests/data_Linux/dm_5002_from_model_run",&
        "../tests/data_Linux/dm_5003_from_model_run",&
        "../tests/data_Linux/dm_5004_from_model_run",&
        "../tests/data_Linux/dm_5005_from_model_run",&
        "../tests/data_Linux/dm_5100_from_model_run",&
        "../tests/data_Linux/dm_5999_from_model_run"&
        /)
   
   if( vgd_putopt("ALLOW_SIGMA",.true.) == VGD_ERROR )then
      print*,'Error with vgd_putopt on ALLOW_SIGM'
      call exit(1)
   endif

   do i=1, nfiles
      if( comp_pres(files(i),i) == VGD_ERROR )then
         call exit(1)
      endif
   enddo

end program pressure_for_all_vcode

!=======================================================================
!=======================================================================
!=======================================================================

integer function comp_pres(F_file, ind) result(status)
   use vGrid_Descriptors, only: vgrid_descriptor, vgd_new, vgd_get, &
        vgd_levels, &
        VGD_OK, VGD_ERROR, VGD_LEN_RFLD, VGD_LEN_RFLS, VGD_NO_REF_NOMVAR
   implicit none
   integer, intent(in) :: ind
   character(len=*), intent(in) :: F_file
   ! Local variables
   integer :: lu, fnom, fstouv, fstluk, fstfrm, fstinf, fclos, ier, key ,&
        k, ni, nj, nk, ni2, nj2, nk2, nijk, similar_vec_real
   integer, dimension(:), pointer :: ip1_list
   real, dimension(:,:), pointer :: rfld_2d, rfls_2d
   real, dimension(:,:,:), pointer :: levels, levels2
   character(len=VGD_LEN_RFLD) :: rfld_S, rfls_S
   type (vgrid_descriptor) :: vgd

   nullify(ip1_list, rfld_2d, rfls_2d, levels)

   status = VGD_ERROR
   lu = 10 + ind
   print*,'Treating file ',trim(F_file)
   if( fnom(lu, F_file, 'RND', 0) < 0 )then
      print*,'ERROR: with fnom on ', trim(F_file)
      return
   endif
   if( fstouv(lu, 'RND+R/O') == 0 )then
      print*,'ERROR: no record in file ', trim(F_file)
      return
   endif   
   if( vgd_new(vgd, lu, 'fst') == VGD_ERROR )then
      print*,'Error with vgd_new on file ', trim(F_file)
      call exit(1)
   endif

   ! Read RFLD ?
   ier = vgd_get(vgd, "RFLD", rfld_S, .true.);
   if( rfld_S == VGD_NO_REF_NOMVAR )then
      print*,"   The current Vcode has no RFLD field"
      ! Get grid size from TT
       key = fstinf( lu, ni, nj, nk, -1, " ", -1, -1, -1, " ", "TT");
       if( key < 0 )then
          print*,"Problem getting info for TT"
          return
       endif
       ! Allocate the surface field rfld_2d since it will be used to get the horizontal preblem size
       ! in the vgrid library. But the value in this surface field will not be used.
       allocate(rfld_2d(ni,nj))
       rfld_2d=0.
   else
      print*,"   RFLD='",rfld_S,"'";
      key = fstinf( lu, ni, nj, nk, -1, " ", -1, -1, -1, " ", rfld_S);
      if( key < 0 )then
         print*,"Problem getting info for ", rfld_S
         return
      endif
      allocate(rfld_2d(ni,nj))
      if( fstluk( rfld_2d, key, ni, nj, nk ) < 0 )then
         print*,"Problem with fstluk for ",rfld_S
         return
      endif
      if( rfld_S == "P0  " ) rfld_2d = rfld_2d*100.
   endif   

   ! Read RFLS ?
   ier = vgd_get(vgd, "RFLS", rfls_S, .true.);
   if( rfls_S == VGD_NO_REF_NOMVAR )then
      print*,"   The current Vcode has no RFLS field"     
   else
      print*,"   RFLS='",rfls_S,"'";
      key = fstinf( lu, ni2, nj2, nk2, -1, " ", -1, -1, -1, " ", rfls_S);
      if( key < 0 )then
         print*,"Problem getting info for ", rfls_S
         return
      endif
      if( ni2 /= ni .or. nj2 /= nj .or. nk2 /= nk )then
         print*,'ERROR: size problem with ', rfls_S
         return
      endif
      allocate(rfls_2d(ni,nj))
      if( fstluk( rfls_2d, key, ni, nj, nk ) < 0 )then
         print*,"Problem with fstluk for ",rfls_S
         return
      endif
      if( rfls_S == "P0LS" ) rfls_2d = rfls_2d*100.
   end if

   ! Compute pressure for momentum level
   if( vgd_get(vgd, "VIPM", ip1_list) )then
      print*,"Error with vgd_get VIPM"
      return
   endif   
   nk = size(ip1_list)
   nijk=ni*nj*nk
   allocate(levels(ni,nj,nk))
   if( vgd_levels(vgd, sfc_field=rfld_2d, sfc_field_ls=rfls_2d, ip1_list=ip1_list, levels=levels) == VGD_ERROR) then
      print*,'Error with vgd_levels'
      call exit(1)
   endif

   ! Compare computed pressure with PX in file
   allocate(levels2(ni,nj,nk))
   do k=1,nk
      key = fstinf(lu, ni2, nj2, nk2, -1, " ", ip1_list(k), -1, -1, " ", "PX")
      if( key < 0 )then
         print*,"Problem getting info for PX for ip1=", ip1_list(k)
         return
      endif
      if( ni2 /= ni .or. nj2 /= nj )then
         print*,"Size problem with PPX for ip1=", ip1_list(k)
         return
      endif
      ier = fstluk(levels2(1,1,k), key, ni2, nj2, nk2 )
   enddo
   levels2 = levels2*100.
   if( similar_vec_real(levels, nijk, levels2, nijk) == 0 )then
      print*,">>>> pressure is the same"
   else
      print*,">>>> pressure differs"
   endif
   
   if( associated(rfld_2d) ) deallocate(rfld_2d)
   if( associated(rfls_2d) ) deallocate(rfls_2d)
   deallocate(levels)

   status = fstfrm(lu)
   status = fclos(lu)
   status = VGD_OK

end function comp_pres

integer function similar_vec_real(vec1, n1, vec2, n2) result(status)
   use vGrid_Descriptors, only : VGD_ERROR, VGD_OK
   implicit none
   integer :: n1, n2
   real, dimension(n1) :: vec1
   real, dimension(n2) :: vec2
   ! Local variables 
   integer :: i

   if( n1 /= n2 ) then
      status = -2
      return
   endif
   do i=1, n1
      if( abs(vec1(i)) < 1.e-37 ) then
         if( abs(vec2(i)) > 1.e-37 ) then
            print*,"Vector differs: val1=", vec1(i), ", val2=", vec2(i)
            return
         endif
      else
         if ( abs(vec1(i)-vec2(i))/vec1(i) > 1.e-5 )then
            print*,"Vector differs: val1=", vec1(i), ", val2=", vec2(i)
            status = -1
            return
         endif
      endif
   end do   
   status = 0
end function similar_vec_real
