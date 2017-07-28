!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                       Environnement Canada
! This library is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, version 2.1 of the License. This library is
! distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
! PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
! You should have received a copy of the GNU Lesser General Public License
! along with this library; if not, write to the Free Software Foundation, Inc.,
! 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!---------------------------------- LICENCE END ---------------------------------

subroutine adv_cfl_lam3 ( F_x_in, F_y_in, F_z_in, i0, in, j0, jn, &
                          F_ni, F_nj, k0, F_nk, F_lev_S )
   use adv_cfl
   use glb_ld
      use adv_grid
      use adv_interp
      use outgrid
   implicit none
#include <arch_specific.hf>
   !@objective Compute the courrant numbers for this time step
   !@arguments
   character(len=1) :: F_lev_S         !I, m/t: momentum/thermo level
   integer          :: F_ni,F_nj,F_nk  !I, dims for F_x/y/z
   integer          :: i0, in, j0, jn, k0  !I, Scope of operator
   real, dimension(F_ni,F_nj,F_nk) :: &
        F_x_in, F_y_in, F_z_in         !I, upstream positions
   !@author Vivian Lee         October 2002
   !@revisions
!*@/

   integer, save :: numproc=0, myproc=0
   integer, allocatable,save :: iwk(:,:,:)
   real(8), allocatable,save :: wk_8(:,:)

   integer :: i, j, k, cfl_i(3,3), err, iproc, imax,jmax,kmax
   real*8 :: x_cfl, y_cfl, z_cfl, xy_cfl, xyz_cfl, max_cfl_8, cfl_8(3)
   real*8, dimension(:),pointer :: p_bsz_8, p_dlz_8

   integer :: npex,npey,medomm,mex,mey,sizex,sizey, &
        ismaster, mymaster, mybloc, myblocx,myblocy,blocme
   character(len=12) :: domname
   !---------------------------------------------------------------------
   if (numproc == 0) then
      call rpn_comm_carac(npex,npey,myproc,medomm,mex,mey,sizex,sizey, &
           ismaster, mymaster, mybloc, myblocx,myblocy,blocme,domname)
      numproc = npex * npey
      !TODO: check if ??? myproc=medomm ???
      !- or do this
      ! err = rpn_comm_mype(myproc,mex,mey)
      ! err = 1
      ! call rpn_comm_allreduce(err,sum, 1, RPN_COMM_INT, 'MPI_SUM', RPN_COMM_GRID, err2)
      allocate(iwk(3,3,numproc), wk_8(3,numproc))
   endif

   if (F_lev_S == 'm') then
      p_bsz_8 => adv_bsz_8%m
      p_dlz_8 => adv_dlz_8%m
   else
      p_bsz_8 => adv_bsz_8%t
      p_dlz_8 => adv_dlz_8%t
   endif


   adv_cfl_8(:  ) = 0.D0
   adv_cfl_i(:,:) = 0

   !     Compute the largest horizontal courrant number
   imax = 0
   jmax = 0
   kmax = 0
   max_cfl_8 = 0.D0

   do k=k0,F_nk
      do j=j0,jn
         do i=i0,in
            x_cfl = (abs(F_x_in(i,j,k)-adv_xx_8(i)))/adv_dlx_8(1)
            y_cfl = (abs(F_y_in(i,j,k)-adv_yy_8(j)))/adv_dly_8(1)
            xy_cfl= sqrt(x_cfl*x_cfl + y_cfl*y_cfl)
            if (xy_cfl > max_cfl_8) then
               imax = i
               jmax = j
               kmax = k
               max_cfl_8 = xy_cfl
            endif
         enddo
      enddo
   enddo

   cfl_8(1)   = max_cfl_8
   cfl_i(1,1) = imax + l_i0 - 1
   cfl_i(2,1) = jmax + l_j0 - 1
   cfl_i(3,1) = kmax

   !     Compute the largest vertical courrant number
   imax = 0
   jmax = 0
   kmax = 0
   max_cfl_8 = 0.D0

   do k=k0,F_nk
      do j=j0,jn
         do i=i0,in
            z_cfl = (abs(F_z_in(i,j,k)-p_bsz_8(k-1)))/p_dlz_8(k-2)
            if (z_cfl > max_cfl_8) then
               imax = i
               jmax = j
               kmax = k
               max_cfl_8 = z_cfl
            endif
         enddo
      enddo
   enddo

   cfl_8(2)   = max_cfl_8
   cfl_i(1,2) = imax + l_i0 - 1
   cfl_i(2,2) = jmax + l_j0 - 1
   cfl_i(3,2) = kmax

   !     Calculate the largest 3D courrant number
   imax = 0
   jmax = 0
   kmax = 0
   max_cfl_8 = 0.D0

   do k=k0,F_nk
      do j=j0,jn
         do i=i0,in
            x_cfl = (abs(F_x_in(i,j,k)-adv_xx_8(i)))/adv_dlx_8(1)
            y_cfl = (abs(F_y_in(i,j,k)-adv_yy_8(j)))/adv_dly_8(1)
            z_cfl = (abs(F_z_in(i,j,k)-p_bsz_8(k-1)))/p_dlz_8(k-2)
            xyz_cfl= sqrt(x_cfl*x_cfl + y_cfl*y_cfl + z_cfl*z_cfl)
            if (xyz_cfl>max_cfl_8) then
               imax = i
               jmax = j
               kmax = k
               max_cfl_8=xyz_cfl
            endif
         enddo
      enddo
   enddo

   cfl_8(3)   = max_cfl_8
   cfl_i(1,3) = imax + l_i0 - 1
   cfl_i(2,3) = jmax + l_j0 - 1
   cfl_i(3,3) = kmax

   call RPN_COMM_gather(cfl_8,3,"MPI_DOUBLE_PRECISION",wk_8,3, &
        "MPI_DOUBLE_PRECISION",0,"GRID", err)
   call RPN_COMM_gather(cfl_i,9,"MPI_INTEGER",iwk,9, &
        "MPI_INTEGER",0,"GRID", err)

   if (myproc == 0) then
      imax = iwk(1,1,1)
      jmax = iwk(2,1,1)
      kmax = iwk(3,1,1)
      max_cfl_8 = wk_8(1,1)
      do iproc = 2, numproc
         if (wk_8(1,iproc)>max_cfl_8) then
            imax = iwk(1,1,iproc)
            jmax = iwk(2,1,iproc)
            kmax = iwk(3,1,iproc)
            max_cfl_8 = wk_8(1,iproc)
         endif
      end do
      adv_cfl_8(1)   = max_cfl_8
      adv_cfl_i(1,1) = imax
      adv_cfl_i(2,1) = jmax
      adv_cfl_i(3,1) = kmax

      imax = iwk(1,2,1)
      jmax = iwk(2,2,1)
      kmax = iwk(3,2,1)
      max_cfl_8 = wk_8(2,1)
      do iproc = 2, numproc
         if (wk_8(2,iproc) > max_cfl_8) then
            imax = iwk(1,2,iproc)
            jmax = iwk(2,2,iproc)
            kmax = iwk(3,2,iproc)
            max_cfl_8 = wk_8(2,iproc)
         endif
      end do
      adv_cfl_8(2)   = max_cfl_8
      adv_cfl_i(1,2) = imax
      adv_cfl_i(2,2) = jmax
      adv_cfl_i(3,2) = kmax

      imax = iwk(1,3,1)
      jmax = iwk(2,3,1)
      kmax = iwk(3,3,1)
      max_cfl_8 = wk_8(3,1)
      do iproc = 2, numproc
         if (wk_8(3,iproc)>max_cfl_8) then
            imax = iwk(1,3,iproc)
            jmax = iwk(2,3,iproc)
            kmax = iwk(3,3,iproc)
            max_cfl_8 = wk_8(3,iproc)
         endif
      end do
      adv_cfl_8(3)   = max_cfl_8
      adv_cfl_i(1,3) = imax
      adv_cfl_i(2,3) = jmax
      adv_cfl_i(3,3) = kmax
   endif
   !---------------------------------------------------------------------
   return
end subroutine adv_cfl_lam3
