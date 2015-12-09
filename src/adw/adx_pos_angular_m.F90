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
#include "constants.h"
#include "msg.h"
#include "stop_mpi.h"
!/@*
subroutine adx_pos_angular_m (F_nb_iter    ,        &
                              F_px  ,F_py  ,F_pz  , &
                              F_u   ,F_v   ,F_w   , &
                              F_ua  ,F_va  ,F_wa  , F_wdm, &
                              F_xth ,F_yth ,F_zth , &
                              F_aminx, F_amaxx, F_aminy, F_amaxy, &
                              F_ni, F_nj, k0, F_nk, F_nk_winds)
   implicit none
#include <arch_specific.hf>
!
   !@objective calculate upstream positions at th and t1 using angular displacement
!
   !@arguments
   integer, intent(in) :: F_nb_iter                                                                     ! total number of iterations for traj
   integer, intent(in) :: F_aminx, F_amaxx, F_aminy, F_amaxy                                            ! wind fields array bounds
   integer, intent(in) :: F_ni, F_nj                                                                    ! dims of position fields
   integer, intent(in) :: F_nk, F_nk_winds                                                              ! nb levels, nb of winds levels
   integer, intent(in) :: k0                                                                            ! scope of the operation k0 to F_nk
   real, dimension(F_ni,F_nj,F_nk), intent(out) :: F_px  , F_py  , F_pz                                 ! upstream positions valid at t1
   real, dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk_winds), intent(in),target:: F_u   , F_v   , F_w ! real destag winds, may be on super grid
   real, dimension(F_ni,F_nj,F_nk), intent(in),target:: F_ua,   F_va,   F_wa                            ! Arival winds
   real, dimension(F_ni,F_nj,F_nk), intent(inout) :: F_xth , F_yth , F_zth                              ! upwind longitudes at central time
   real, dimension(F_ni,F_nj,F_nk), intent(inout) :: F_wdm
!
   !@author Stéphane Gaudreault, Claude Girard
   !@revisions
   !v4_80 - Tanguay M.    - FLUX calculations
!*@/

#include "adx_nml.cdk"
#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_dyn.cdk"
#include "adx_poles.cdk"
#include "schm.cdk"

#include "glb_ld.cdk"
#include "adx_interp.cdk"
#include "adx_tracers.cdk"

   logical,parameter :: CLIP_TRAJ = .true.
   logical,parameter :: DO_W      = .false.
   logical,parameter :: DO_UV     = .true.
   real,   parameter :: DTH_1     = 1.

   integer :: i, j, k, iter
   integer :: i0,in,j0,jn
   real    :: dth
   real, dimension(F_ni,F_nj,F_nk) :: u_d,v_d,w_d

   real,   dimension(:,:,:), pointer :: dummy3d
   real :: ztop_bound, zbot_bound
   real*8 :: inv_cy_8

   !---------------------------------------------------------------------

   call msg(MSG_DEBUG,'adx_pos_m_lam')

   dummy3d => F_w

   dth  = 0.5 * adx_dt_8

   ztop_bound=adx_verZ_8%m(0)
   zbot_bound=adx_verZ_8%m(F_nk+1)

   if (Adx_extension_L) then
      call adx_get_ij0n_ext (i0,in,j0,jn)
   else
      call adx_get_ij0n (i0,in,j0,jn)
   endif

   do iter = 1, F_nb_iter

      call adx_int_winds_lam (u_d,v_d, F_u,F_v, F_xth,F_yth,F_zth, &
                              DTH_1,CLIP_TRAJ, DO_UV, i0,in,j0,jn, &
                              F_ni,F_nj,F_aminx, F_amaxx, F_aminy, &
                              F_amaxy,k0,F_nk,F_nk_winds)

      if(adx_trapeze_L.or.Schm_step_settls_L) then
!$omp parallel private (inv_cy_8)
!$omp do
         do k=k0,F_nk
            do j=j0,jn
               inv_cy_8 = 1.d0 / adx_cy_8(j)
               do i=i0,in
                  F_xth(i,j,k) = Adx_xx_8(i) - (u_d(i,j,k)/cos(F_yth(i,j,k)) + F_ua(i,j,k) * inv_cy_8) * dth
                  F_yth(i,j,k) = Adx_yy_8(j) - (v_d(i,j,k) + F_va(i,j,k)) * dth
               end do
            end do
         enddo
!$omp enddo
!$omp end parallel
      else
!$omp parallel
!$omp do
         do k=k0,F_nk
            do j=j0,jn
               do i=i0,in
                  F_xth(i,j,k) = Adx_xx_8(i) - u_d(i,j,k)/cos(F_yth(i,j,k)) * dth
                  F_yth(i,j,k) = Adx_yy_8(j) - v_d(i,j,k) * dth
               end do
            end do
         enddo
!$omp enddo
!$omp end parallel
      endif

      !- 3D interpol of zeta dot and new upstream pos along zeta

      call adx_int_winds_lam (w_d,v_d, F_w,dummy3d, F_xth,F_yth,F_zth, &
                              DTH_1,CLIP_TRAJ, DO_W, i0,in,j0,jn,      &
                              F_ni,F_nj,F_aminx, F_amaxx, F_aminy,     &
                              F_amaxy,k0,F_nk,F_nk_winds)

!$omp parallel
      if(adx_trapeze_L.or.Schm_step_settls_L) then
!$omp do
         do k = max(1,k0),F_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) = adx_verZ_8%m(k) - (w_d(i,j,k)+F_wa(i,j,k))*dth
                  F_zth(i,j,k) = min(zbot_bound,max(F_zth(i,j,k),ztop_bound))
               enddo
            enddo
         enddo
!$omp enddo
      else
!$omp do
         do k = max(1,k0),F_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) = adx_verZ_8%m(k) - 2.D0*dth*w_d(i,j,k)
                  F_zth(i,j,k) = min(zbot_bound,max(F_zth(i,j,k),ztop_bound))
                  F_zth(i,j,k) = 0.5D0*(F_zth(i,j,k) + adx_verZ_8%m(k))
               enddo
            enddo
         enddo
!$omp enddo
      endif
!$omp end parallel

   end do

   ! Departure point

   if (adx_trapeze_L.or.Schm_step_settls_L) then
      ! nothing to do ...
      F_px = F_xth
      F_py = F_yth
      F_pz = F_zth
      F_wdm = w_d
   else
!$omp parallel
!$omp do
      do k=k0,F_nk
         do j=j0,jn
            do i=i0,in
               F_px(i,j,k) = Adx_xx_8(i) - u_d(i,j,k)/cos(F_yth(i,j,k)) * adx_dt_8
               F_py(i,j,k) = Adx_yy_8(j) - v_d(i,j,k) * adx_dt_8
               F_pz(i,j,k) = F_zth(i,j,k) - adx_verZ_8%m(k)
               F_pz(i,j,k) = Adx_verZ_8%m(k) + 2.0 * F_pz(i,j,k)
               F_pz(i,j,k) = min(zbot_bound,max(F_pz(i,j,k),ztop_bound))
            end do
         end do
      enddo
!$omp enddo
!$omp end parallel
   endif

   call msg(MSG_DEBUG,'adx_pos_m_lam [end]')

   !---------------------------------------------------------------------

   return
end subroutine adx_pos_angular_m
