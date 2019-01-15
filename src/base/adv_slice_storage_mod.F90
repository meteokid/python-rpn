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

module adv_slice_storage
   implicit none
   public
   save

   !STORAGE: LOCAL CONSERVATION SLICE Zerroukat et al.(2002)
   !--------------------------------------------------------

   !REVISED
   !-------
   real*8,  pointer, dimension(:) :: s_LEFT_ecv_8 =>null(),s_LEFT_iecv_8=>null(),rho_LEFT_ecv_8 =>null(), &
                                     s_LEFT_ilcv_8=>null(),s_LEFT_lcv_8 =>null(),rho_LEFT_ilcv_8=>null(), &
                                     m_ecv_X_8    =>null(),ds_ecv_8     =>null(),slope_rho_X_8  =>null(), &
                                     m_ilcv_Y_8   =>null(),ds_ilcv_8    =>null(),slope_rho_Y_8  =>null(), &
                                     m_iecv_X_8   =>null(),m_lcv_Y_8    =>null()

   integer, pointer, dimension(:) :: iecv_location=>null(),lcv_location=>null(),ys_location=>null(),yv_location=>null(), &
                                     b_sn_location=>null()

   real*8,  pointer, dimension(:,:,:) :: m_ijk_8,m_ecv_8,m_iecv_8,m_lcv_8

   real*8,  pointer, dimension(:,:,:) :: x_LEFT_iecv_8  =>null(),z_LEFT_iecv_8  =>null()
   real*8,  pointer, dimension(:,:,:) :: x_CENTRE_iecv_8=>null(),z_CENTRE_iecv_8=>null()

   real*8,save, pointer, dimension(:,:,:) :: dist_LEFT_ecv_8,dist_LEFT_iecv_8,dist_LEFT_ilcv_8,dist_LEFT_lcv_8

   integer jm,jp

   real*8, pointer,     dimension(:,:,:)   :: x_usm_8=>null(),y_usm_8=>null(),z_usm_8=>null(), &
                                              x_svm_8=>null(),y_svm_8=>null(),z_svm_8=>null()

   integer,parameter :: ext = 0

   !OBSOLETE
   !--------

   integer j_ps,j_pn
   integer,pointer,     dimension(:)       :: i_bw   =>null(),i_be   =>null()

   integer,pointer,     dimension(:)       :: j_ps_  =>null(),j_pn_  =>null()
   integer,save, pointer, dimension(:,:)   :: c1_w   =>null(),c1_e   =>null()

   integer,save, pointer, dimension(:,:)   :: c1_s   =>null(),c1_n   =>null()
   real*8  slope_8

   integer offi_,offj_,L1,R1,L2,R2

   real*8, pointer, dimension(:,:,:) :: z_bve_8  =>null()

   real*8   distance_GC_8
   external distance_GC_8

end module adv_slice_storage
