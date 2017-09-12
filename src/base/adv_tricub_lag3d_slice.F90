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

!**s/p adv_tricub_lag3d_slice: Conservative Semi-Lagrangian advection based on SLICE Zerroukat et al.(2002)

      subroutine adv_tricub_lag3d_slice (F_cub_rho,F_in_rho,      &
                                         F_x_usm,F_y_usm,F_z_usm, & !POSITIONS USM
                                         F_x_svm,F_y_svm,F_z_svm, & !POSITIONS SVM
                                         F_num,F_ni,k0,F_nk,F_lev_S)


      use adv_slice_storage
      use grid_options
      use gem_options
      use geomh
      use glb_ld
      use lun
      use ver
      use adv_grid
      use tracers
      use adv_interp
      use outgrid
      use ptopo
      implicit none

#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev_S ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num     ! number points
      integer, intent(in) :: F_ni      ! dimension of horizontal fields
      integer, intent(in) :: F_nk ! number of vertical levels
      integer, intent(in) :: k0   ! scope of operator
      real,dimension(F_num), intent(in)  :: F_x_usm, F_y_usm, F_z_usm ! interpolation target x,y,z coordinates USM
      real,dimension(F_num), intent(in)  :: F_x_svm, F_y_svm, F_z_svm ! interpolation target x,y,z coordinates SVM
      real,dimension(*),     intent(in)  :: F_in_rho                  ! field to interpolate (scaled by density)
      real,dimension(F_num), intent(out) :: F_cub_rho                 ! High-order SL solution (scaled by density)

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer :: n,i,j,k,ii,jj,kk,kkmax,idxk,idxjk,o2, &
                 i0_e,in_e,j0_e,jn_e,i0_c,in_c,j0_c,jn_c
      real*8 :: p_z00_8

      real*8 :: mass_of_area_8
      external  mass_of_area_8

      logical slice_old_L
!
!---------------------------------------------------------------------
!
      if (Grd_yinyang_L)       call handle_error (-1,'ADV_TRICUB_LAG3D_SLICE','SLICE not available for GY')
      if (.NOT.Schm_autobar_L) call handle_error (-1,'ADV_TRICUB_LAG3D_SLICE','SLICE not available for BAROCLINE')

      call timing_start2 (86, 'SLICE', 38)

      slice_old_L = .false.

      if (.NOT.slice_old_L.and.Tr_SLICE_rebuild<3) call handle_error(-1,'ADV_TRICUB_LAG3D_SLICE','SLICE VERSION 2: Tr_SLICE_rebuild not available')

      if (Lun_out>0.and.slice_old_L) then
         write(Lun_out,901)
      elseif(Lun_out>0) then
         write(Lun_out,902)
      endif

      kkmax   = l_nk - 1
      p_z00_8 = Ver_z_8%m(0)
      if (F_lev_S == 'm') then
         p_lcz     => adv_lcz%m
         p_bsz_8   => adv_bsz_8%m
         p_zabcd_8 => adv_zabcd_8%m
         p_zbacd_8 => adv_zbacd_8%m
         p_zcabd_8 => adv_zcabd_8%m
         p_zdabc_8 => adv_zdabc_8%m
         p_zbc_8   => adv_zbc_8%m
      else if (F_lev_S == 't') then
         p_lcz     => adv_lcz%t
         p_bsz_8   => adv_bsz_8%t
         p_zabcd_8 => adv_zabcd_8%t
         p_zbacd_8 => adv_zbacd_8%t
         p_zcabd_8 => adv_zcabd_8%t
         p_zdabc_8 => adv_zdabc_8%t
         p_zbc_8   => adv_zbc_8%t
      else if (F_lev_S == 'x') then
         p_lcz     => adv_lcz%x
         p_bsz_8   => adv_bsz_8%x
         p_zabcd_8 => adv_zabcd_8%x
         p_zbacd_8 => adv_zbacd_8%x
         p_zcabd_8 => adv_zcabd_8%x
         p_zdabc_8 => adv_zdabc_8%x
         p_zbc_8   => adv_zbc_8%x
      endif

      i0_c = 1+pil_w
      in_c = l_ni-pil_e
      j0_c = 1+pil_s
      jn_c = l_nj-pil_n

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      !-------------------------------------
      if (Tr_do_only_once_each_timestep_L) then
      !-------------------------------------

         allocate(  x_LEFT_iecv_8(0:l_ni,1:l_nj,k0:F_nk),  z_LEFT_iecv_8(0:l_ni,1:l_nj,k0:F_nk), &
                  x_CENTRE_iecv_8(1:l_ni,0:l_nj,k0:F_nk),z_CENTRE_iecv_8(1:l_ni,0:l_nj,k0:F_nk))

         allocate(ys_location(1:l_nj),yv_location(0:l_nj),b_sn_location(0:l_ni))

         !OBSOLETE
         !--------
         allocate (x_usm_8(0:l_ni,1:l_nj  ,k0:F_nk),y_usm_8(0:l_ni,1:l_nj,k0:F_nk),z_usm_8(0:l_ni,1:l_nj,k0:F_nk), &
                   x_svm_8(1:l_ni,0:l_nj  ,k0:F_nk),y_svm_8(1:l_ni,0:l_nj,k0:F_nk),z_svm_8(1:l_ni,0:l_nj,k0:F_nk))

         if (Lctl_step==1) then
            allocate (z_bve_8(0:l_ni,0:l_nj+1,k0:F_nk+1))
         end if

         allocate (i_bw(l_nj),i_be(l_nj),j_ps_(k0:F_nk),j_pn_(k0:F_nk))

         allocate (c1_w(l_nj,k0:F_nk),c1_e(l_nj,k0:F_nk))

         if (Lctl_step==1) then

             allocate (s_LEFT_ecv_8 (0:l_ni),s_LEFT_iecv_8(0:l_ni),rho_LEFT_ecv_8 (0:l_ni), &
                       s_LEFT_ilcv_8(0:l_nj),s_LEFT_lcv_8 (0:l_nj),rho_LEFT_ilcv_8(0:l_nj), &
                       m_ecv_X_8    (1:l_ni),ds_ecv_8     (1:l_ni),slope_rho_X_8  (1:l_ni), &
                       m_ilcv_Y_8   (1:l_nj),ds_ilcv_8    (1:l_nj),slope_rho_Y_8  (1:l_nj), &
                       m_iecv_X_8   (1:l_ni),m_lcv_Y_8    (1:l_nj),                         &
                       iecv_location(0:l_ni),lcv_location (0:l_nj))

             allocate (dist_LEFT_ecv_8 (0:l_ni,1:l_nj,k0:F_nk),dist_LEFT_iecv_8(0:l_ni,1:l_nj,k0:F_nk), &
                       dist_LEFT_ilcv_8(1:l_ni,0:l_nj,k0:F_nk),dist_LEFT_lcv_8 (1:l_ni,0:l_nj,k0:F_nk))

         endif

         if (Lctl_step==1) then

            allocate (c1_s(l_ni,k0:F_nk),c1_n(l_ni,k0:F_nk))

         endif

#include "adv_tricub_lag3d_slice_loop_1.cdk"

         deallocate (x_usm_8,y_usm_8,z_usm_8,x_svm_8,y_svm_8,z_svm_8)

         deallocate (x_LEFT_iecv_8,z_LEFT_iecv_8,x_CENTRE_iecv_8,z_CENTRE_iecv_8,ys_location,yv_location,b_sn_location)

         deallocate (j_ps_,j_pn_,c1_w,c1_e)

      !--------------------------------------
      endif !END do_only_once_each_timestep_L
      !--------------------------------------

      Tr_do_only_once_each_timestep_L = .FALSE.

      allocate (m_ijk_8 (1:l_ni,1:l_nj,k0:F_nk),m_ecv_8(1:l_ni,1:l_nj,k0:F_nk), &
                m_iecv_8(1:l_ni,1:l_nj,k0:F_nk),m_lcv_8(1:l_ni,1:l_nj,k0:F_nk))

      m_ijk_8 = 999. ; m_ecv_8  = 999. ; m_iecv_8 = 999. ; m_lcv_8 = 999.0

      !NEEDED to compute the initial mass distribution
      !-----------------------------------------------
      offi_ = Ptopo_gindx(1,Ptopo_myproc+1)-1
      offj_ = Ptopo_gindx(3,Ptopo_myproc+1)-1

#include "adv_tricub_lag3d_slice_loop_2.cdk"

      deallocate (m_ijk_8,m_ecv_8,m_iecv_8,m_lcv_8)

      call timing_stop (86)
!
!---------------------------------------------------------------------
!
      return

910 format(2X,A34,E20.12)
901 format(/,1X,'WE USE SLICE VERSION 1',/)
902 format(/,1X,'WE USE SLICE VERSION 2',/)

      end subroutine adv_tricub_lag3d_slice
