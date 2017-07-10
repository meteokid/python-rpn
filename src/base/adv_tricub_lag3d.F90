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

      subroutine adv_tricub_lag3d (F_cub, F_mono, F_lin, F_min, F_max,         &
                                   F_in, F_in_rho, F_conserv_local,            &
                                   F_cub_o, F_in_o, F_cub_i, F_in_i, F_flux_n, &
                                   F_x, F_y, F_z,                              &
                                   F_x_usm, F_y_usm, F_z_usm,                  &
                                   F_x_svm, F_y_svm, F_z_svm,                  &
                                   F_num, F_nind, ii, F_k0, F_nk,              &
                                   F_mono_L,  F_conserv_L, F_lev)
      implicit none
#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num ! number points
      integer, intent(in) :: F_nk ! number of vertical levels
      integer, intent(in) :: F_k0 ! scope of operator
      logical, intent(in) :: F_mono_L ! .true. monotonic interpolation
      logical, intent(in) :: F_conserv_L ! .true. conservative interpolation
      integer, intent(in) :: F_flux_n    ! 0=NO FLUX; 1=TRACER+FLUX; 2=FLUX only
      integer, intent(in) :: F_conserv_local !I, > 0 if Local Conservation
      real,dimension(F_num), intent(in)  :: F_x, F_y, F_z ! interpolation target x,y,z coordinates
      real,dimension(F_num), intent(in)  :: F_x_usm,F_y_usm,F_z_usm ! interpolation target x,y,z coordinates USM
      real,dimension(F_num), intent(in)  :: F_x_svm,F_y_svm,F_z_svm ! interpolation target x,y,z coordinates SVM
      real,dimension(*),     intent(in)  :: F_in          ! field to interpolate
      real,dimension(*),     intent(in)  :: F_in_rho      ! field to interpolate (scaled by density)
      real,dimension(*),     intent(in)  :: F_in_o,F_in_i ! field to interpolate (FLUX_out/FLUX_in)
      real,dimension(F_num), intent(out) :: F_cub ! High-order SL solution
      real,dimension(F_num), intent(out) :: F_mono! High-order monotone SL solution
      real,dimension(F_num), intent(out) :: F_lin ! Low-order SL solution
      real,dimension(F_num), intent(out) :: F_min ! MIN over cell
      real,dimension(F_num), intent(out) :: F_max ! MAX over cell
      real,dimension(F_num), intent(out) :: F_cub_o ! High-order SL solution FLUX_out
      real,dimension(F_num), intent(out) :: F_cub_i ! High-order SL solution FLUX_in
      integer , intent(in) :: F_nind
      integer , dimension(F_nind*4), intent(in)  :: ii            ! pre-computed indices to be used in: adv_tricub_lag3d_loop

   !@revisions
   !  2012-05,  Stephane Gaudreault: code optimization
   !  2016-01,  Monique Tanguay    : GEM4 Mass-Conservation
   !  2017-01,  A. Qaddouri        : Correction F_lin
   !@objective Tri-cubic interp: Lagrange 3d (Based on adx_tricub v3.1.1) (MASS-CONSERVATION)

#include "adv.cdk"
#include "adv_grid.cdk"
#include "adv_interp.cdk"
#include "glb_ld.cdk"

      logical :: zcubic_L
      integer :: n0, nx, ny, nz, m1, o1, o2, o3, o4 , &
                 kkmax, n, id

      real*8  :: a1, a2, a3, a4, b1, b2, b3, b4, &
              c1, c2, c3, c4, d1, d2, d3, d4, &
              p1, p2, p3, p4
      real*8  :: rri,rrj,rrk,ra,rb,rc,rd
      real*8  :: prf1,prf2,capx,capy,capz
      real    :: prmin,prmax,za
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      real*8 :: triprd,zb,zc,zd
      triprd(za,zb,zc,zd)=(za-zb)*(za-zc)*(za-zd)
!
!---------------------------------------------------------------------
!
      if ( trim(Adv_component_S) == 'TRAJ' ) then
         call timing_start2 (37, 'ADV_LAG3D', 34)
      else if ( trim(Adv_component_S) == 'INTP_RHS' ) then
         call timing_start2 (38, 'ADV_LAG3D', 31)
      else
         call timing_start2 (39, 'ADV_LAG3D', 27)
      endif

      kkmax   = F_nk - 1
    if (F_lev == 'm') then
      p_bsz_8   => adv_bsz_8%m
      p_zabcd_8 => adv_zabcd_8%m
      p_zbacd_8 => adv_zbacd_8%m
      p_zcabd_8 => adv_zcabd_8%m
      p_zdabc_8 => adv_zdabc_8%m
      p_zbc_8   => adv_zbc_8%m
    else if (F_lev  == 't') then
      p_bsz_8   => adv_bsz_8%t
      p_zabcd_8 => adv_zabcd_8%t
      p_zbacd_8 => adv_zbacd_8%t
      p_zcabd_8 => adv_zcabd_8%t
      p_zdabc_8 => adv_zdabc_8%t
      p_zbc_8   => adv_zbc_8%t
    else if (F_lev == 'x') then
      p_bsz_8   => adv_bsz_8%x
      p_zabcd_8 => adv_zabcd_8%x
      p_zbacd_8 => adv_zbacd_8%x
      p_zcabd_8 => adv_zcabd_8%x
      p_zdabc_8 => adv_zdabc_8%x
      p_zbc_8   => adv_zbc_8%x
     endif


      if (F_flux_n == 2) goto 10

      if (.NOT.F_conserv_L) then

#undef ADV_CONSERV

         if (F_mono_L) then
#define ADV_MONO

#include "adv_tricub_lag3d_loop.cdk"

         else
#undef ADV_MONO

#include "adv_tricub_lag3d_loop.cdk"

         endif

      else

         !No local conservation: Standard cubic interpolation with MIN/MAX/LIN for Bermejo-Conde/ILMC
         !-------------------------------------------------------------------------------------------
         if (F_conserv_local==0) then
#define ADV_CONSERV
#define ADV_MONO
#include "adv_tricub_lag3d_loop.cdk"

         !Conservative Semi-Lagrangian advection based on SLICE Zerroukat et al.(2002)
         !----------------------------------------------------------------------------
         elseif (F_conserv_local==1) then

            call adv_tricub_lag3d_slice (F_cub, F_in_rho,           &
                                         F_x_usm, F_y_usm, F_z_usm, & !POSITIONS USM
                                         F_x_svm, F_y_svm, F_z_svm, & !POSITIONS SVM
                                         F_num, l_ni, l_nj, F_k0, F_nk, F_lev)
         else

            call handle_error(-1,'ADV_TRICUB_LAG3D','Current F_conserv_local NOT AVAILABLE')

         endif

      endif

   10 continue

      !-------------------------
      !Estimate FLUX_out/FLUX_in
      !-------------------------
      if (F_flux_n>0) call adv_tricub_lag3d_flux (F_cub_o, F_in_o, F_cub_i, F_in_i, &
                                                  F_x, F_y, F_z, F_num, F_k0, F_nk, F_lev)   ! todo:  optimiser

      if ( trim(Adv_component_S) == 'TRAJ' ) then
         call timing_stop (37)
      else if ( trim(Adv_component_S) == 'INTP_RHS' ) then
         call timing_stop (38)
      else
         call timing_stop (39)
      endif
!
!---------------------------------------------------------------------
!
      return
      end subroutine adv_tricub_lag3d
