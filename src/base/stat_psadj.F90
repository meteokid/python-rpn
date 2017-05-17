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

!**s/r stat_psadj - Print dry/wet air mass

      subroutine stat_psadj (F_time,F_comment_S)
      use gmm_vt1
      use gmm_vt0
      use grid_options
      implicit none

      integer,          intent(in) :: F_time       !I, Time 0 or Time 1
      character(len=*), intent(in) :: F_comment_S  !I, Comment

#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "lun.cdk"
#include "cstv.cdk"
#include "ptopo.cdk"

      real, pointer, dimension (:,:) :: w2d 
      integer err,i,j,istat
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy,1:l_nk):: pr_m_8,pr_t_8 
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy)       :: pr_p0_8,pr_p0_dry_8 
      real*8 l_avg_8(2),g_avg_ps_8(2),scale_8
      character(len= 9) communicate_S
      character(len= 1) in_S
      character(len= 7) time_S

      !---------------------------------------------------------------

      !Estimate area
      !-------------
      l_avg_8(1) = 0.0d0
      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         l_avg_8(1) = l_avg_8(1) + Geomg_area_8(i,j) * Geomg_mask_8(i,j)
      enddo
      enddo

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      call RPN_COMM_allreduce (l_avg_8, scale_8, 1, &
        "MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      scale_8 = 1.0d0/scale_8

      if (Grd_yinyang_L) then

         if (F_time==1) istat = gmm_get(gmmk_st1_s,w2d) 
         if (F_time==0) istat = gmm_get(gmmk_st0_s,w2d) 

         if (F_time==1) in_S = 'P'  
         if (F_time==0) in_S = 'M'  

         if (F_time==1) time_S = "TIME T1"
         if (F_time==0) time_S = "TIME T0"

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 ( pr_m_8, pr_t_8, pr_p0_8, w2d, &
                                l_minx,l_maxx,l_miny,l_maxy,l_nk )

         !Compute dry surface pressure (- Cstv_pref_8)
         !--------------------------------------------
         call dry_sfc_pressure_8 ( pr_p0_dry_8, pr_m_8, pr_p0_8, &
                                   l_minx,l_maxx,l_miny,l_maxy,l_nk,in_S)

         !Add Cstv_pref_8 to dry surface pressure
         !---------------------------------------
         pr_p0_dry_8(:,:) = pr_p0_dry_8(:,:) + Cstv_pref_8

         !Compute dry air mass
         !--------------------
         l_avg_8(1) = 0.0d0
         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            l_avg_8(1) = l_avg_8(1) + pr_p0_dry_8(i,j) * Geomg_area_8(i,j) &
                                                       * Geomg_mask_8(i,j) 
         enddo
         enddo

         !Compute wet air mass
         !--------------------
         l_avg_8(2) = 0.0d0
         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            l_avg_8(2) = l_avg_8(2) + pr_p0_8(i,j) * Geomg_area_8(i,j) &
                                                   * Geomg_mask_8(i,j)   
         enddo
         enddo

         call RPN_COMM_allreduce ( l_avg_8,g_avg_ps_8,&
                    2,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

         g_avg_ps_8(:) = g_avg_ps_8(:) * scale_8 

         if (Lun_out>0) write(Lun_out,1000) "DRY air mass",time_S,g_avg_ps_8(1),F_comment_S,'PANEL=',Ptopo_couleur
         if (Lun_out>0) write(Lun_out,1000) "WET air mass",time_S,g_avg_ps_8(2),F_comment_S,'PANEL=',Ptopo_couleur

      endif

      !---------------------------------------------------------------

 1000 format(1X,A12,1X,A7,1X,E19.12,1X,A16,1X,A6,I1)

      return
      end
