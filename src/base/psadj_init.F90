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

!**s/r psadj_init - Estimate area and air mass at initial time

      subroutine psadj_init (F_kount)
      use gmm_vt1
      use gmm_geof
      use grid_options
      use gem_options
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: F_kount

#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "dcst.cdk"
#include "psadj.cdk"
#include "cstv.cdk"

      integer err,i,j,istat
      logical, save :: done = .false.
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy,1:l_nk):: pr_m_8,pr_t_8 
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy)       :: pr_p0_1_8,pr_p0_w_1_8 
      real*8 l_avg_8,x_avg_8
      character(len= 9) communicate_S
!
!     ---------------------------------------------------------------
!
      if ( Schm_psadj == 0 ) then
         if ( Schm_psadj_print_L ) goto 999
         return
      endif

      if ( Schm_psadj <= 2 ) then
         if ( Cstv_dt_8*F_kount <= Iau_period ) goto 999
         if ( ( Schm_psadj == 1 ) .and. done )  goto 999
      endif

      done= .true.

      !Estimate area
      !-------------
      istat = gmm_get(gmmk_fis0_s,fis0)
      l_avg_8 = 0.0d0
      x_avg_8 = 0.0d0
      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         l_avg_8 = l_avg_8 + Geomg_area_8(i,j) * Geomg_mask_8(i,j)
         if(fis0(i,j).gt.1.) &
            x_avg_8 = x_avg_8 + Geomg_area_8(i,j) * Geomg_mask_8(i,j)
      enddo
      enddo

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      call RPN_COMM_allreduce (l_avg_8, PSADJ_scale_8, 1, &
        "MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      call RPN_COMM_allreduce (x_avg_8, PSADJ_fact_8, 1, &
        "MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      PSADJ_fact_8  = PSADJ_fact_8/PSADJ_scale_8
      PSADJ_scale_8 = 1.0d0/PSADJ_scale_8
      PSADJ_fact_8  =(1.0d0-(1.0d0-PSADJ_fact_8)*Cstv_psadj_8)/PSADJ_fact_8

      istat = gmm_get(gmmk_st1_s,st1)

      !Obtain pressure levels
      !----------------------
      call calc_pressure_8 ( pr_m_8, pr_t_8, pr_p0_1_8, st1, &
                             l_minx,l_maxx,l_miny,l_maxy,l_nk )

      !Compute dry surface pressure at initial time (- Cstv_pref_8)
      !------------------------------------------------------------
      if (Schm_psadj==2) then

         call dry_sfc_pressure_8 ( pr_p0_w_1_8, pr_m_8, pr_p0_1_8, &
                                   l_minx,l_maxx,l_miny,l_maxy,l_nk,'P' )

      !Compute wet surface pressure at initial time (- Cstv_pref_8)
      !------------------------------------------------------------
      elseif (Schm_psadj==1) then

         pr_p0_w_1_8(1:l_ni,1:l_nj) = pr_p0_1_8(1:l_ni,1:l_nj) - Cstv_pref_8

      endif

      !Store air mass at initial time
      !------------------------------
      l_avg_8 = 0.0d0
      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         l_avg_8 = l_avg_8 + pr_p0_w_1_8(i,j) * Geomg_area_8(i,j) &
                                                 * Geomg_mask_8(i,j) 
      enddo
      enddo

      call RPN_COMM_allreduce ( l_avg_8,PSADJ_g_avg_ps_initial_8,&
                                1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      PSADJ_g_avg_ps_initial_8 = PSADJ_g_avg_ps_initial_8 &
                               * PSADJ_scale_8

  999 continue

      if (Schm_psadj_print_L) call stat_psadj (1,"BEFORE DYNSTEP") 
!
!     ---------------------------------------------------------------
!

 1000 format(1X,A12,1X,A7,1X,E19.12,1X,A16)

      return
      end
