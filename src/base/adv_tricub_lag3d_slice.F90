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
                                         F_num,k0,F_nk,F_lev_S)

      implicit none

#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev_S ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num ! number points
      integer, intent(in) :: F_nk ! number of vertical levels
      integer, intent(in) :: k0   ! scope of operator
      real,dimension(F_num), intent(in)  :: F_x_usm, F_y_usm, F_z_usm ! interpolation target x,y,z coordinates USM
      real,dimension(F_num), intent(in)  :: F_x_svm, F_y_svm, F_z_svm ! interpolation target x,y,z coordinates SVM
      real,dimension(*),     intent(in)  :: F_in_rho                  ! field to interpolate (scaled by density)
      real,dimension(F_num), intent(out) :: F_cub_rho                 ! High-order SL solution (scaled by density)

      !@author Monique Tanguay

      !@revisions
      ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation

#include "adv_grid.cdk"
#include "adv_interp.cdk"
#include "adv_slice_storage.cdk"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "grd.cdk"
#include "ver.cdk"
#include "ptopo.cdk"
#include "schm.cdk"
#include "lun.cdk"
#include "lctl.cdk"

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer :: n,i,j,k,ii,jj,kk,iL,iR,kkmax,idxk,idxjk,o1,o2,o3,o4, &
                 i0_e,in_e,j0_e,jn_e,i0_c,in_c,j0_c,jn_c
      real*8 :: p_z00_8 
!     
!---------------------------------------------------------------------
!     
      if (Grd_yinyang_L)       call handle_error (-1,'ADV_TRICUB_LAG3D_SLICE','SLICE not available for GY')
      if (.NOT.Schm_autobar_L) call handle_error (-1,'ADV_TRICUB_LAG3D_SLICE','SLICE not available for BAROCLINE')

      call timing_start2 (33, 'ADV_LAG3D', 31)
 
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
      if (Adv_do_only_once_each_timestep_L) then 
      !------------------------------------- 

         allocate (x_usm_8(0:l_ni,1:l_nj  ,k0:F_nk+1),y_usm_8(0:l_ni,1:l_nj,k0:F_nk+1),z_usm_8(0:l_ni,1:l_nj,k0:F_nk+1), &
                   x_svm_8(1:l_ni,0:l_nj  ,k0:F_nk+1),y_svm_8(1:l_ni,0:l_nj,k0:F_nk+1),z_svm_8(1:l_ni,0:l_nj,k0:F_nk+1), &
                   x_bin_8(0:l_ni,1:l_nj  ,k0:F_nk+1),z_bin_8(0:l_ni,1:l_nj,k0:F_nk+1),                                  &
                   x_cin_8(1:l_ni,0:l_nj  ,k0:F_nk+1),z_cin_8(1:l_ni,0:l_nj,k0:F_nk+1),                                  &
                   z_bve_8(0:l_ni,0:l_nj+1,k0:F_nk+1))

         if (Lctl_step==1) &  
         allocate (sx_b1_8(0:l_ni,l_nj,k0:F_nk+1),sx_b2_8(0:l_ni,l_nj,k0:F_nk+1)) !TO BE REVISED

         if (Lctl_step==1) &  
         allocate (c1_w(l_nj,k0:F_nk+1),c1_e(l_nj,k0:F_nk+1), & !TO BE REVISED
                   c2_w(l_nj,k0:F_nk+1),c2_e(l_nj,k0:F_nk+1))

         if (Lctl_step==1) &
         allocate (sy_b1_8(0:l_nj,l_ni,k0:F_nk+1),sy_b2_8(0:l_nj,l_ni,k0:F_nk+1)) !TO BE REVISED

         if (Lctl_step==1) &
         allocate (c1_s(l_ni,k0:F_nk+1),c1_n(l_ni,k0:F_nk+1), & !TO BE REVISED
                   c2_s(l_ni,k0:F_nk+1),c2_n(l_ni,k0:F_nk+1))

#include "adv_tricub_lag3d_slice_loop_1.cdk"

         deallocate (x_usm_8,y_usm_8,z_usm_8,x_svm_8,y_svm_8,z_svm_8,x_bin_8,z_bin_8,x_cin_8,z_cin_8,z_bve_8)

      !-------------------------------------- 
      endif !END do_only_once_each_timestep_L
      !-------------------------------------- 

      Adv_do_only_once_each_timestep_L = .FALSE.

      allocate (m_ijk_8(1:l_ni,1:l_nj,k0:F_nk),m_cew_8(1:l_ni,1:l_nj,k0:F_nk), & !TO BE REVISED
                m_cns_8(1:l_ni,1:l_nj,k0:F_nk),m_cve_8(1:l_ni,1:l_nj,k0:F_nk))

      m_ijk_8 = 999.
      m_cew_8 = 999.
      m_cns_8 = 999.
      m_cve_8 = 999.

      allocate (w_casc00_8(1:l_nj,k0:F_nk+1)) !TO BE REVISED
      allocate (w_casc11_8(1:l_ni,k0:F_nk+1)) !TO BE REVISED

      allocate (mi1_8     (1:l_ni),          mi2_8     (1:l_ni), &
                xi1_8     (0:l_ni),          xi2_8     (0:l_ni), &
                w_casc2a_8(1:l_nj,k0:F_nk+1),w_casc2b_8(1:l_nj,k0:F_nk+1)) !TO BE REVISED

      allocate (mj1_8     (1:l_nj),          mj2_8     (1:l_nj), &
                yj1_8     (0:l_nj),          yj2_8     (0:l_nj), &
                w_casc3a_8(1:l_ni,k0:F_nk+1),w_casc3b_8(1:l_ni,k0:F_nk+1)) !TO BE REVISED

      w_casc00_8 = 0.0d0
      w_casc2a_8 = 0.0d0
      w_casc2b_8 = 0.0d0
      w_casc3a_8 = 0.0d0
      w_casc3b_8 = 0.0d0
      w_casc11_8 = 0.0d0

      !NEEDED to compute the initial mass distribution
      !-----------------------------------------------
      offi_ = Ptopo_gindx(1,Ptopo_myproc+1)-1
      offj_ = Ptopo_gindx(3,Ptopo_myproc+1)-1

#include "adv_tricub_lag3d_slice_loop_2.cdk"

      !-----------  
      !DIAGNOSTICS
      !-----------  
      if (G_lam.and..NOT.Grd_yinyang_L) then

          c_area_8 = 0.0d0
          gc_area_8 = 0.0d0

          do j=j0_c,jn_c
          do i=i0_c,in_c
             c_area_8 = c_area_8 + Geomg_area_8(i,j)
          enddo
          enddo

          call rpn_comm_ALLREDUCE(c_area_8,gc_area_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err )

          scale_mass_8 = 1.0d0/gc_area_8

      else

          call handle_error(-1,'TRICUB','TRICUB: SCALING NOT DONE')

      endif

      casc00_8 = 0.0d0
      casc2a_8 = 0.0d0
      casc2b_8 = 0.0d0
      casc3a_8 = 0.0d0
      casc3b_8 = 0.0d0
      casc11_8 = 0.0d0

      do k=k0,F_nk
         do j=j0_c,jn_c
            casc00_8 = w_casc00_8(j,k) + casc00_8
            casc2a_8 = w_casc2a_8(j,k) + casc2a_8
            casc2b_8 = w_casc2b_8(j,k) + casc2b_8
         enddo
      enddo

      do k=k0,F_nk
         do i=i0_c,in_c
            casc3a_8 = w_casc3a_8(i,k) + casc3a_8
            casc3b_8 = w_casc3b_8(i,k) + casc3b_8
            casc11_8 = w_casc11_8(i,k) + casc11_8
          enddo
      enddo

      write(Lun_out,910) 'CASC00',casc00_8*scale_mass_8
      write(Lun_out,910) 'CASC2A',casc2a_8*scale_mass_8
      write(Lun_out,910) 'CASC2B',casc2b_8*scale_mass_8
      write(Lun_out,910) 'CASC3A',casc3a_8*scale_mass_8
      write(Lun_out,910) 'CASC3B',casc3b_8*scale_mass_8
      write(Lun_out,910) 'CASC11',casc11_8*scale_mass_8

      deallocate (m_ijk_8,m_cew_8,m_cns_8,m_cve_8)

      deallocate (mi1_8,mi2_8,xi1_8,xi2_8,w_casc00_8,w_casc2a_8,w_casc2b_8, &
                  mj1_8,mj2_8,yj1_8,yj2_8,w_casc3a_8,w_casc3b_8,w_casc11_8)

      call timing_stop (33)
!     
!---------------------------------------------------------------------
!     
      return

910 format(1X,A34,E20.12)

      end subroutine adv_tricub_lag3d_slice
