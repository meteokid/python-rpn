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

!**s/p psadj_LAM - Adjust surface pressure for conservation when LAM using Flux calculations based on Aranami et al. (2015)  

      subroutine psadj_LAM (F_cub_o,F_cub_i,Minx,Maxx,Miny,Maxy,F_nk,k0)

      use gmm_vt1
      implicit none

#include <arch_specific.hf>

      !Arguments
      !---------
      integer,           intent(in) :: Minx,Maxx,Miny,Maxy                !I, Dimension H
      integer,           intent(in) :: k0                                 !I, Scope of operator
      integer,           intent(in) :: F_nk                               !I, Number of vertical levels
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)   :: F_cub_o !I: High-order SL solution FLUX_out 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)   :: F_cub_i !I: High-order SL solution FLUX_in 

!Author Monique Tanguay
!
!revision
! v4_XX - Tanguay M.        - initial MPI version
!
!**/
#include "glb_ld.cdk"
#include "gmm.hf"
#include "geomg.cdk"
#include "vt0.cdk"
#include "cstv.cdk"
#include "lun.cdk"
#include "psadj.cdk"

      type(gmm_metadata) :: mymeta
      real, pointer, dimension(:,:,:) :: tr
      real*8,dimension(Minx:Maxx,Miny:Maxy,1:l_nk):: pr_m_8,pr_t_8
      real*8,dimension(Minx:Maxx,Miny:Maxy)       :: pr_p0_1_8,pr_p0_0_8,pr_p0_dry_8,pr_fl_dry_8
      real,  dimension(Minx:Maxx,Miny:Maxy,1:l_nk):: work,sumq
      integer :: err,i,j,k,iteration,istat
      real*8 l_avg_8(2),g_avg_8(2),g_avg_ps_dry_1_8,g_avg_ps_dry_0_8,g_avg_fl_dry_0_8
      character(len= 9) communicate_S

      !---------------------------------------------------------------------

      if (Lun_out>0) write(Lun_out,*) ''
      if (Lun_out>0) write(Lun_out,*) '--------------------------------------'
      if (Lun_out>0) write(Lun_out,*) 'PSADJ LAM is done for DRY AIR (REAL*8)'
      if (Lun_out>0) write(Lun_out,*) '--------------------------------------'

      iteration = 1 

      communicate_S = "GRID"

      !---------------------
      !Treat TIME T1
      !---------------------

         istat = gmm_get(gmmk_st1_s,st1,mymeta)

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 (pr_m_8,pr_t_8,pr_p0_1_8,st1,l_minx,l_maxx,l_miny,l_maxy,l_nk)

         !Compute dry surface pressure on CORE
         !------------------------------------
         call sumhydro (sumq,l_minx,l_maxx,l_miny,l_maxy,l_nk,'P')

         istat = gmm_get('TR/HU:P',tr)

!$omp parallel shared(sumq,tr,pr_p0_dry_8)
!$omp do
         do k=k0,l_nk
            sumq(1:l_ni,1:l_nj,k) = sumq(1:l_ni,1:l_nj,k) + tr(1:l_ni,1:l_nj,k) 
         end do
!$omp end do

!$omp do
         do j=1+pil_s,l_nj-pil_n
            pr_p0_dry_8(:,j) = 0.0d0
            do k=k0,l_nk-1
            do i=1+pil_w,l_ni-pil_e
                pr_p0_dry_8(i,j)= pr_p0_dry_8(i,j) + &
                    (1.-sumq(i,j,k))*(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
            enddo
            end do
            do i=1+pil_w,l_ni-pil_e
               pr_p0_dry_8(i,j)= pr_p0_dry_8(i,j) + &
                    (1.-sumq(i,j,l_nk))*(pr_p0_1_8(i,j) - pr_m_8(i,j,l_nk))
            end do
         end do
!$omp enddo
!$omp end parallel

         l_avg_8(1) = 0.0d0

         !Estimate dry air mass on CORE
         !-----------------------------
         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e

            l_avg_8(1) = l_avg_8(1) + pr_p0_dry_8(i,j) * Geomg_area_8(i,j) * Geomg_mask_8(i,j)

         enddo
         enddo

         call RPN_COMM_allreduce (l_avg_8,g_avg_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

         g_avg_ps_dry_1_8 = g_avg_8(1) * PSADJ_scale_8

  800    continue

      !---------------------
      !Treat TIME T0
      !---------------------

         istat = gmm_get(gmmk_st0_s,st0,mymeta)

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 (pr_m_8,pr_t_8,pr_p0_0_8,st0,l_minx,l_maxx,l_miny,l_maxy,l_nk)

         !Compute dry surface pressure on CORE and FLUX on NEST+CORE
         !----------------------------------------------------------
         call sumhydro (sumq,l_minx,l_maxx,l_miny,l_maxy,l_nk,'M')

         istat = gmm_get('TR/HU:M',tr)

!$omp parallel shared(sumq,tr,pr_p0_dry_8,pr_fl_dry_8)
!$omp do
         do k=k0,l_nk
            sumq(1:l_ni,1:l_nj,k) = sumq(1:l_ni,1:l_nj,k) + tr(1:l_ni,1:l_nj,k) 
         end do
!$omp end do

!$omp do
         do j=1+pil_s,l_nj-pil_n
            pr_p0_dry_8(:,j) = 0.0d0
            do k=k0,l_nk-1
            do i=1+pil_w,l_ni-pil_e
               pr_p0_dry_8(i,j)= pr_p0_dry_8(i,j) + &
                    (1.-sumq(i,j,k))*(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
            enddo
            end do
            do i=1+pil_w,l_ni-pil_e
               pr_p0_dry_8(i,j)= pr_p0_dry_8(i,j) + &
                    (1.-sumq(i,j,l_nk))*(pr_p0_0_8(i,j) - pr_m_8(i,j,l_nk))
            end do
         end do
!$omp enddo

!$omp do
         do j=1,l_nj
            pr_fl_dry_8(:,j) = 0.0d0
            do k=k0,l_nk-1
            do i=1,l_ni
               pr_fl_dry_8(i,j)= pr_fl_dry_8(i,j) + &
                    (1.-sumq(i,j,k))*(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))*(F_cub_i(i,j,k) - F_cub_o(i,j,k)) 
            enddo
            end do
            do i=1,l_ni
               pr_fl_dry_8(i,j)= pr_fl_dry_8(i,j) + &
                    (1.-sumq(i,j,l_nk))*(pr_p0_0_8(i,j) - pr_m_8(i,j,l_nk))*(F_cub_i(i,j,k) - F_cub_o(i,j,k))
            end do
         end do
!$omp enddo
!$omp end parallel

         !Estimate dry air mass on CORE
         !-----------------------------
         l_avg_8(1) = 0.0d0

         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e

            l_avg_8(1) = l_avg_8(1) + pr_p0_dry_8(i,j) * Geomg_area_8(i,j) * Geomg_mask_8(i,j)

         enddo
         enddo

         !Estimate mass FLUX
         !------------------
         l_avg_8(2) = 0.0d0

         do j=1,l_nj
         do i=1,l_ni

            l_avg_8(2) = l_avg_8(2) + pr_fl_dry_8(i,j) * Geomg_area_8(i,j) * Geomg_mask_8(i,j)

         enddo
         enddo

         call RPN_COMM_allreduce (l_avg_8,g_avg_8,2,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

         g_avg_ps_dry_0_8 = g_avg_8(1) * PSADJ_scale_8
         g_avg_fl_dry_0_8 = g_avg_8(2) * PSADJ_scale_8

         !Correct surface pressure in order to preserve dry air mass on CORE when taking mass FLUX into account   
         !-----------------------------------------------------------------------------------------------------
         pr_p0_0_8 = pr_p0_0_8 + (g_avg_ps_dry_1_8 - g_avg_ps_dry_0_8) + g_avg_fl_dry_0_8

         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            st0(i,j)= log(pr_p0_0_8(i,j)/Cstv_pref_8)
         end do
         end do

         iteration = iteration + 1

         if (iteration<4) goto 800

      !---------------------
      !Diagnostics
      !---------------------

      if (Lun_out>0) then
         write(Lun_out,*)    ''
         write(Lun_out,*)    '------------------------------------------------------'
         write(Lun_out,1004) 'PSADJ_LAM: C=',g_avg_ps_dry_0_8,' F=',g_avg_fl_dry_0_8
         write(Lun_out,*)    '------------------------------------------------------'
         write(Lun_out,*)    ''
      endif

      return

1004 format(1X,A13,E19.12,A3,E19.12)

      end subroutine psadj_LAM 
