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

!**s/r psadj - Adjust surface pressure for conservation
!
      subroutine psadj
      implicit none
#include <arch_specific.hf>

!author
!     Andre Plante from hzd_main      
!
!revision
! v4_05 - Lepine M.         - VMM replacement with GMM
! v4_50 - Qaddouri-PLante   - YY version
! v4_70 - Tanguay M.        - dry air pressure conservation
! v4_80 - Tanguay M.        - REAL*8 with iterations and psadj LAM 
!	
#include "gmm.hf"
#include "glb_ld.cdk"
#include "cstv.cdk"
#include "geomg.cdk"
#include "schm.cdk"
#include "vt0.cdk"
#include "vt1.cdk"
#include "ptopo.cdk"
#include "grd.cdk"
#include "lun.cdk"
#include "tr3d.cdk"
#include "dcst.cdk"
#include "psadj.cdk"
#include "rstr.cdk"

      type(gmm_metadata) :: mymeta
      integer err,i,j,k,n,istat,iteration
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy,1:l_nk):: pr_m_8,pr_t_8
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy)       :: pr_p0_1_8,pr_p0_0_8,pr_p0_dry_1_8,pr_p0_dry_0_8 
      real*8 l_avg_8,g_avg_ps_dry_0_8
      real*8,parameter :: QUATRO_8 = 4.0d0, ONE_8 = 1.0d0
      character(len= 9) communicate_S
      logical,save :: done_L=.FALSE.
!
!     ---------------------------------------------------------------
!
      if (.not.Schm_psadj_L) return

      if (G_lam.and..not.Grd_yinyang_L) then
         if (Rstri_rstn_L) call gem_error(-1,'psadj','PSADJ NOT AVAILABLE FOR LAMs in RESTART mode')
         if (.not. Schm_adxlegacy_L) then
            call adv_psadj_LAM_0
         else
            call adx_psadj_LAM_0
         endif
         return
      endif

! for GU and GY
      if (Lun_out>0) write(Lun_out,*) ''
      if (Lun_out>0) write(Lun_out,*) '----------------------------------'
      if (Lun_out>0) write(Lun_out,*) 'PSADJ is done for DRY AIR (REAL*8)'
      if (Lun_out>0) write(Lun_out,*) '----------------------------------'
      if (Lun_out>0) write(Lun_out,*) ''

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      istat = gmm_get(gmmk_st0_s,st0,mymeta)

      do n= 1, 3

      !Obtain pressure levels
      !----------------------
      call calc_pressure_8 (pr_m_8,pr_t_8,pr_p0_0_8,st0,l_minx,l_maxx,l_miny,l_maxy,l_nk)

      !Compute dry surface pressure (- Cstv_pref_8)
      !--------------------------------------------
      call dry_sfc_pressure_8 (pr_p0_dry_0_8,pr_m_8,pr_p0_0_8,l_minx,l_maxx,l_miny,l_maxy,l_nk,'M')

      !Estimate dry air mass 
      !---------------------
      l_avg_8 = 0.0d0
      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         l_avg_8 = l_avg_8 + pr_p0_dry_0_8(i,j) * Geomg_area_8(i,j) * Geomg_mask_8(i,j)
      enddo
      enddo

      call RPN_COMM_allreduce (l_avg_8,g_avg_ps_dry_0_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err)

      g_avg_ps_dry_0_8 = g_avg_ps_dry_0_8 * PSADJ_scale_8

      !Correct surface pressure in order to preserve dry air mass    
      !----------------------------------------------------------
      pr_p0_0_8 = pr_p0_0_8 + (PSADJ_g_avg_ps_dry_initial_8 - g_avg_ps_dry_0_8)

      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         st0(i,j)= log(pr_p0_0_8(i,j)/Cstv_pref_8)
      end do
      end do

      end do
!
!     ---------------------------------------------------------------
!
      return
      end
