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
#include "stop_mpi.h"


!/@*
subroutine adx_trilin_turbo3 (F_out, F_in, F_dt, &
                              F_x, F_y, F_capz, F_ii, F_jj, F_kk, &
                              F_bsx_8, F_bsy_8, F_xbc_8, F_ybc_8, &
                              F_num, i0, in, j0, jn, k0, F_nk)
   implicit none
#include <arch_specific.hf>
!
   !@objective Optimized tri-linear interpolation with SETINT inside
!
   !@arguments
   integer :: F_nk                      !I, number of vertical levels
   integer :: F_num                     !I, dims of position fields
   integer :: i0,in,j0,jn,k0            !I, scope ofthe operator
   real    :: F_dt                      !I, multiplicative constant (1. or timestep lenght)
   integer,dimension(F_num) :: &
        F_ii, F_jj, F_kk                !I, localisation indices
   real,dimension(F_num) :: &
        F_capz, &                       !I, precomputed displacements along the z-dir
        F_x, F_y                        !I, x,y positions 
   real,dimension(*)     :: F_in        !I, field to interpolate
   real,dimension(F_num) :: F_out       !O, F_dt * result of interpolation
   real*8, dimension(*)  :: F_bsx_8,F_bsy_8,F_xbc_8, F_ybc_8
!
   !@author Valin, Tanguay  
   !@revisions
   ! v3_20 -Valin & Tanguay -  initial version 
   ! v3_21 -Tanguay M.      -  evaluate min-max vertical CFL as function of k 
   ! v4_10 -Plante A.       -  Replace single locator vector with 3 vectors.
   ! V4_14 -Plante A.       - Scope on loop k
!*@/

#undef __ADX_DIMS__
#include "adx_dims.cdk"

   integer :: n, n0, o1, o2
   integer :: i, j, k, ii, jj, kk
   real    :: capx, capy, capz
   real*8  :: rri, rrj, rrk, prf1, prf2, prf3, prf4

   !---------------------------------------------------------------------

!$omp parallel do private(n,n0,ii,jj,kk,rri,rrj,rrk,capx,capy,capz,o1,o2,prf1,prf2,prf3,prf4)
   DO_K: do k=k0,F_nk
      DO_J: do j=j0,jn

         n0 = (k-1)*adx_mlnij + (j-1)*adx_mlni
         do i=i0,in
            n = n0 + i

            ii = F_ii(n)
            jj = F_jj(n)
            kk = F_kk(n)

            rri= F_x(n)
            rrj= F_y(n)

            o1 = (kk)*adx_nijag + (jj-adx_int_j_off-1)*adx_nit + (ii-adx_int_i_off)
            o2 = o1 + adx_nit

            !- x interpolation
            capx = (rri-F_bsx_8(ii)) *F_xbc_8(ii)

            prf1 = (1. - capx) * F_in(o1) + capx * F_in(o1+1)
            prf2 = (1. - capx) * F_in(o2) + capx * F_in(o2+1)

            o1 = o1 + adx_nijag
            o2 = o2 + adx_nijag

            prf3 = (1. - capx) * F_in(o1) + capx * F_in(o1+1)
            prf4 = (1. - capx) * F_in(o2) + capx * F_in(o2+1)

            !- y interpolation
            capy = (rrj-F_bsy_8(jj)) *F_ybc_8(jj)  

            prf1 = (1. - capy) * prf1 + capy  * prf2
            prf2 = (1. - capy) * prf3 + capy  * prf4

            !- z interpolation
            capz = F_capz(n)
            F_out(n) = ((1. - capz) * prf1 + capz  * prf2) * F_dt
         enddo

      enddo DO_J
   enddo DO_K
!$omp end parallel do

   !---------------------------------------------------------------------

   return
end subroutine adx_trilin_turbo3
