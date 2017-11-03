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

subroutine adv_trilin_ijk ( F_x, F_y, F_z, F_capz, F_ii, F_jj, F_kk,        &
                            F_lcx, F_lcy, F_lcz, F_bsx_8, F_bsy_8, F_bsz_8, &
                            F_diz_8, F_z00_8, F_iimax,                      &
                            F_num, i0, in, j0, jn, k0, F_nk )
      use glb_ld
      use adv_grid
      use adv_interp
      use ver
      use outgrid
   implicit none
#include "arch_specific.hf"
!
   !@objective Optimized tri-linear interpolation with SETINT inside
!
   !@arguments
   integer, intent(in) :: F_nk                              !I, number of vertical levels
   integer, intent(in) :: F_num                             !I, dims of position fields
   integer, intent(in) :: i0,in,j0,jn,k0                    !I, scope ofthe operator
   integer, intent(in) :: F_iimax
   integer,dimension(F_num), intent(inout) :: F_ii, F_jj, F_kk !I/O, localisation indices
   real,dimension(F_num), intent(inout)    :: F_capz           !I/O, precomputed displacements along the z-dir
   real,dimension(F_num), intent(in)       :: F_x, F_y, F_z    !I, x,y,z positions
   integer,dimension(*)  :: F_lcx,F_lcy,F_lcz
   real*8, intent(in) ::  F_z00_8
   real*8, dimension(*), intent(in) :: F_bsx_8,F_bsy_8
   real*8, dimension( 0:*), intent(in) :: F_bsz_8
   real*8, dimension(-1:*), intent(in) :: F_diz_8
!
   !@author Valin, Tanguay
   !@revisions
   ! v3_20 -Valin & Tanguay -  initial version
   ! v3_21 -Tanguay M.      -  evaluate min-max vertical CFL as function of k
   ! v4_10 -Plante A.       -  Replace single locator vector with 3 vectors.
   ! v5.0.a12-Aider R.      -  Adapted to height vertical coordinate


   integer :: i, j, k, ii, jj, kk, n, n0, sig
   real    :: capz
   real*8  :: rri, rrj, rrk

   !---------------------------------------------------------------------

! Vertical variable type:  Height--> sig <0 , Pressure --> sig >0
    sig=int((Ver_z_8%m(l_nk)-Ver_z_8%m(1))/(abs(  Ver_z_8%m(l_nk)-Ver_z_8%m(1) )))

!$omp parallel do private(i,j,k,n,n0,ii,jj,kk,rri,rrj,rrk,capz)
    do k=k0,F_nk
       do j=j0,jn

         n0 = (k-1)*l_ni*l_nj + (j-1)*l_ni
         do i=i0,in
            n = n0 + i

            rri= F_x(n)
            ii = int((rri - adv_x00_8) * adv_ovdx_8)
            ii = F_lcx(ii+1) + 1
            if (rri < F_bsx_8(ii)) ii = ii - 1
            F_ii(n) = max(1,min(ii,F_iimax))

            rrj= F_y(n)
            jj = int((rrj - adv_y00_8) * adv_ovdy_8)
            jj = F_lcy(jj+1) + 1
            if (rrj < F_bsy_8(jj)) jj = jj - 1
            F_jj(n) = max(adv_haloy,min(jj,adv_jjmax))

            rrk= F_z(n)
            kk = int((rrk - F_z00_8) * adv_ovdz_8*sig)
            kk = F_lcz(kk+1)

            rrk = rrk - F_bsz_8(kk)
            if (real(sig)*rrk < 0.) kk = kk - 1

            capz = rrk * F_diz_8(kk)
            if (real(sig)*rrk < 0.) capz = 1. + capz

            !- We keep F_capz, otherwise we would need rrk
            F_capz(n) = capz
            F_kk(n) = kk
         enddo

       enddo
    enddo
!$omp end parallel do

   !---------------------------------------------------------------------

   return
end subroutine adv_trilin_ijk




