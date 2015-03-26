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

#include "model_macros_f.h"

!/@*
subroutine adx_trilin_turbo4 (F_out   , F_in    , F_dt    ,            &
                              F_n     , F_capx  , F_capy  , F_capz  ,  &
                              F_x     , F_y     , F_z     ,            &
                              F_ii    , F_jj    , F_kk    ,            &
                              F_lcx   , F_bsx_8 , F_dix_8 ,            &
                              F_lcy   , F_bsy_8 , F_diy_8 ,            &
                              F_lcz   , F_bsz_8 , F_diz_8 ,            &
                              F_num,i0,in,j0,jn,k0,kn,F_nkm)

   implicit none
#include <arch_specific.hf>

   !@objective tri-linear interpolation

   !@arguments
   integer :: F_num, F_n(F_num),i0,in,j0,jn,kn,k0,F_nkm
   real :: F_dt, F_in(*)
   real :: F_out(F_num), F_capx(F_num), F_capy(F_num), F_capz(F_num)
   integer,dimension(F_num) :: F_ii,F_jj,F_kk
   real,dimension(F_num) :: &
        F_x, F_y, F_z    !I, upstream position coor
 
   integer,dimension(*) :: F_lcx,F_lcy,F_lcz
   real*8, dimension(*) :: F_bsx_8,F_bsy_8
   real*8, dimension(*) :: F_dix_8,F_diy_8
   real*8, dimension(0:2*F_nkm)    :: F_bsz_8
   real*8, dimension(-1:2*F_nkm+2) :: F_diz_8

   !______________________________________________________________________
   !              |                                                 |     |
   ! NAME         | DESCRIPTION                                     | I/O |
   !--------------|-------------------------------------------------|-----|
   !              |                                                 |     |
   ! F_out        | F_dt * result of interpolation                  |  o  |
   ! F_in         | field to interpolate                            |  i  |
   !              |                                                 |     |
   ! F_dt         | multiplicative constant (1.0 or timestep lenght)|  i  |
   !              |                                                 |     |
   ! F_n          | positions in the 3D volume of interpolation     |  i  |
   !              | boxes                                           |     |
   !              |                                                 |     |
   ! F_capx       | \                                               |  i  |
   ! F_capy       |   precomputed displacements                     |  i  |
   ! F_capz       | / along the x,y,z directions                    |  i  |
   !              |                                                 |     |
   ! F_num        | number of points to interpolate                 |  i  |
   !______________|_________________________________________________|_____|
   !
   !@author  alain patoine

   !@revisions
   ! v3_00 - Desgagne & Lee    - Lam configuration
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v4_14 - Plante A. - Scope on loop k
   ! v4_40 - Tanguay M.        - Revision TL/AD

   !*@/
#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_interp.cdk"
#include "adx_nosetint.cdk"

   integer :: n, o1, o2, i, j, k, ii, jj, kk, ij, n0
   real*8  :: prf1, prf2, prf3, prf4
   real*8  :: prd_8, prdt_8, p_z00_8
   !---------------------------------------------------------------------

   p_z00_8 = adx_verZ_8%m(0)

!$omp parallel do private(n,o1,o2,prf1,prf2,prf3,prf4,prd_8,ii,prdt_8,jj,kk,ij)

   do k=k0,kn
      do j=j0,jn

         n0 = (k-1)*adx_mlnij + ((j-1)*adx_mlni)

         if ( Adx_hor_L ) then

         do i=i0,in

            n = n0 + i

            prd_8 = dble(F_x(n))
            ii = (prd_8 - adx_x00_8) * adx_ovdx_8
            ii = F_lcx(ii+1) + 1
            ii = max(2,ii)
            ii = min(ii,adx_gni+2*adx_halox-2)

            prdt_8 = prd_8 - F_bsx_8(ii)
            if (prdt_8 < 0.0) then
               ii = max(2,ii - 1)
               prdt_8 = prd_8 - F_bsx_8(ii)
            endif

            F_ii  (n) = ii
            F_capx(n) = prdt_8 * F_dix_8(ii)

            prd_8 = dble(F_y(n))
            jj = (prd_8 - adx_y00_8) * adx_ovdy_8
            jj = F_lcy(jj+1) + 1
            jj = max(adx_haloy,jj)
            jj = min(jj,adx_gnj+adx_haloy)

            prdt_8 = prd_8 - F_bsy_8(jj)
            if (prdt_8 < 0.0) then
               jj = max(adx_haloy,jj - 1)
               prdt_8 = prd_8 - F_bsy_8(jj)
            endif

            F_jj  (n) = jj 
            F_capy(n) = prdt_8 * F_diy_8(jj)

            kk = F_kk(n)  

            ij = (jj-adx_int_j_off-1)*adx_nit + (ii-adx_int_i_off)

            F_n(n) = kk*adx_nijag + ij

         enddo

         endif

         if ( Adx_ver_L ) then

         do i=i0,in

            n = n0 + i

            ii = F_ii(n)

            jj = F_jj(n) 

            prd_8 = dble(F_z(n))
            kk = (prd_8 - p_z00_8) * adx_ovdz_8
            kk = F_lcz(kk+1)
            prd_8 = prd_8 - F_bsz_8(kk)
            if (prd_8 < 0.0) kk = kk - 1

            F_kk  (n) = kk
            F_capz(n) = prd_8 * F_diz_8(kk)
            if (prd_8 < 0.0) F_capz(n) = 1.0 + F_capz(n)

            ij = (jj-adx_int_j_off-1)*adx_nit + (ii-adx_int_i_off)

            F_n(n) = kk*adx_nijag + ij

         enddo

         endif

         do i=i0,in

            n = n0 + i

            o1 = F_n(n)
            o2 = o1 + adx_nit

            !- x interpolation
            prf1 = (1.0 - F_capx(n)) * F_in(o1) + F_capx(n) * F_in(o1+1)
            prf2 = (1.0 - F_capx(n)) * F_in(o2) + F_capx(n) * F_in(o2+1)

            o1 = o1 + adx_nijag
            o2 = o2 + adx_nijag

            prf3 = (1.0 - F_capx(n)) * F_in(o1) + F_capx(n) * F_in(o1+1)
            prf4 = (1.0 - F_capx(n)) * F_in(o2) + F_capx(n) * F_in(o2+1)

            !- y interpolation
            prf1 = (1.0 - F_capy(n)) * prf1 + F_capy(n)  * prf2
            prf2 = (1.0 - F_capy(n)) * prf3 + F_capy(n)  * prf4

            !- z interpolation
            F_out(n) = ( (1.0 - F_capz(n)) * prf1 + F_capz(n)  * prf2 ) * F_dt

         enddo

      enddo
   enddo
!$omp end parallel do

   Adx_hor_L = .false.
   Adx_ver_L = .false.

   !---------------------------------------------------------------------
   return
end subroutine adx_trilin_turbo4

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

subroutine adx_trilin_turbo2()
   call stop_mpi(STOP_ERROR,'adx_trilin_turbo2','called a stub')
   return
end subroutine adx_trilin_turbo2
