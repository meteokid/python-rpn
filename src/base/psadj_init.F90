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

!**s/r psadj_init - Estimate area and dry air mass at initial time

      subroutine psadj_init
      implicit none
#include <arch_specific.hf>

#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "schm.cdk"
#include "vt1.cdk"
#include "grd.cdk"
#include "dcst.cdk"
#include "psadj.cdk"

      integer err,i,j,k,istat
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy,1:l_nk):: &
                                                    pr_m_8,pr_t_8
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy)       :: &
                  pr_p0_1_8,pr_p0_0_8,pr_p0_dry_1_8,pr_p0_dry_0_8 
      real*8 l_avg_8,g_avg_ps_dry_0_8
      real*8,parameter :: QUATRO_8 = 4.0d0
      character(len= 9) communicate_S
!
!     ---------------------------------------------------------------
!
      if (.not.Schm_psadj_L) return

!Estimate area

      l_avg_8 = 0.0d0
      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         l_avg_8 = l_avg_8 + Geomg_area_8(i,j) * Geomg_mask_8(i,j)
      enddo
      enddo

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      call RPN_COMM_allreduce (l_avg_8, PSADJ_scale_8, 1, &
        "MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      PSADJ_scale_8 = 1.0d0/PSADJ_scale_8

      if (Grd_yinyang_L) then
         istat = gmm_get(gmmk_st1_s,st1)

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 ( pr_m_8, pr_t_8, pr_p0_1_8, st1, &
                                l_minx,l_maxx,l_miny,l_maxy,l_nk )

         !Compute dry surface pressure (- Cstv_pref_8)
         !--------------------------------------------
         call dry_sfc_pressure_8 ( pr_p0_dry_1_8, pr_m_8, pr_p0_1_8, &
                                   l_minx,l_maxx,l_miny,l_maxy,l_nk,'P' )

         !Estimate dry air mass at initial time
         !-------------------------------------
         l_avg_8 = 0.0d0
         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            l_avg_8 = l_avg_8 + pr_p0_dry_1_8(i,j) * Geomg_area_8(i,j) &
                                                   * Geomg_mask_8(i,j) 
         enddo
         enddo

         call RPN_COMM_allreduce ( l_avg_8,PSADJ_g_avg_ps_dry_initial_8,&
                    1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

         PSADJ_g_avg_ps_dry_initial_8 = PSADJ_g_avg_ps_dry_initial_8 &
                                      * PSADJ_scale_8
      endif
!
!     ---------------------------------------------------------------
!
      return
      end
