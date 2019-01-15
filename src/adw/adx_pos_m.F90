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
subroutine adx_pos_m( F_nb_iter    ,        &
                            F_px  ,F_py  ,F_pz  , &
                            F_u   ,F_v   ,F_w   , &
                            F_ua  ,F_va  ,F_wa  , F_wdm, &
                            F_xth ,F_yth ,F_zth , &
                            F_xcth,F_ycth,F_zcth, &
                            F_xct1,F_yct1,F_zct1, &
                            F_aminx, F_amaxx, F_aminy, F_amaxy, &
                            F_ni, F_nj, k0, F_nk, F_nk_winds)
   implicit none
#include <arch_specific.hf>
!
   !@objective calculate upstream positions at th and t1 using great circles
!
   !@arguments
   integer :: F_nb_iter          !I, total number of iterations for traj
   integer :: F_aminx, F_amaxx, F_aminy, F_amaxy !I, wind fields array bounds
   integer :: F_ni, F_nj         !I, dims of position fields
   integer :: F_nk, F_nk_winds   !I, nb levels, nb of winds levels
   integer :: k0                 !I, scope of the operation k0 to F_nk
   real, dimension(F_ni,F_nj,F_nk) :: &
        F_px  , F_py  , F_pz     !O, upstream positions valid at t1
   real,dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk_winds),target::&
        F_u   , F_v   , F_w      !I, real destag winds, may be on super grid
   real,dimension(F_ni,F_nj,F_nk),target::&
        F_ua,   F_va,   F_wa, F_wdm     !O, Arival winds
   real, dimension(F_ni,F_nj,F_nk) :: &
        F_xth , F_yth , F_zth,&  !I/O, upwind longitudes at central time
        F_xcth, F_ycth, F_zcth,& !O, upwind cartesian positions at central time
        F_xct1, F_yct1, F_zct1   !O, upstream cartesian positions at t1
!
   !@author alain patoine
   !@revisions
   ! v2_31 - Desgagne M.    - removed stkmemw
   ! v2_31 - Tanguay M.     - gem stop if adx_fro_a.gt.0 in anal mode
   ! v3_00 - Desgagne & Lee - Lam configuration
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v3_20 - Valin & Tanguay - Optimized SETINT/TRILIN
   ! v3_20 - Gravel S.       - Change test a lower and upper boundaries
   ! v3_20 - Tanguay M.      - Improve alarm when points outside advection grid
   ! v3_20 - Dugas B.        - correct calculation for LAM when Glb_pil gt 7
   ! v3_21 - Lee V.          - bug correction, F_yth should not be modified.
   ! v4_05 - Lepine M.       - VMM replacement with GMM
   ! V4_10 - Plante A.       - Support to thermodynamic positions.
   ! V4_14 - Plante A.       - Do not compute position in top pilot zone
   ! v4_40 - Lee/Qaddouri    - add Adw_gccsa_L option for trajsp,trajex
!*@/

#include "adx_nml.cdk"
#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_dyn.cdk"
#include "adx_poles.cdk"

#include "glb_ld.cdk"
#include "adx_interp.cdk"

   real*8, parameter :: PDP_8 = 1.D0 + 1.D-6
   real*8, parameter :: PDM_8 = 1.D0 - 1.D-6
   logical,parameter :: CLIP_TRAJ = .true.
   logical,parameter :: DO_W      = .false.
   logical,parameter :: DO_UV     = .true.
   real,   parameter :: DTH_1     = 1.

   integer :: i, j, k, iter, ioff
   integer :: i0,in,j0,jn
   real    :: dth
   real, dimension(F_ni,F_nj,F_nk) :: u_d,v_d,w_d

   real,   dimension(:,:,:), pointer :: dummy3d
   real :: ztop_bound, zbot_bound

   !---------------------------------------------------------------------

   call msg(MSG_DEBUG,'adx_pos_m')

   dummy3d => F_w

   dth  = 0.5 * adx_dt_8

   ztop_bound=adx_verZ_8%m(0)
   zbot_bound=adx_verZ_8%m(F_nk+1)

   call adx_get_ij0n (i0,in,j0,jn)

   DO_ITER: do iter = 1, F_nb_iter

      !- 3d interpol of u and v winds and new upstream pos along x and y
      if (adx_lam_L) then
         call adx_int_winds_lam (u_d,v_d, F_u,F_v, F_xth,F_yth,F_zth, &
                            DTH_1,CLIP_TRAJ, DO_UV, i0,in,j0,jn, &
                            F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0,F_nk,F_nk_winds)
      else
         call adx_int_winds_glb (u_d,v_d, F_u,F_v, F_xth,F_yth,F_zth, &
                            DTH_1,DO_UV, i0,in,j0,jn, &
                            F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0,F_nk,F_nk_winds)
      endif

      if(adx_trapeze_L) then
         call adx_traj_trapeze (F_xth,F_yth, F_xct1,F_yct1,F_zct1, &
                                u_d,v_d,F_ua,F_va,dth,i0,in,j0,jn,k0)
      else
         call adx_trajsp2 (F_xth,F_yth, F_xcth,F_ycth,F_zcth, u_d,v_d, &
                                adx_cx_8,adx_cy_8,adx_sx_8,adx_sy_8, &
                                    dth,i0,in,j0,jn,k0,adx_lni,adx_lnj)
      endif

      !- 3D interpol of zeta dot and new upstream pos along zeta

      if (adx_lam_L) then
         call adx_int_winds_lam (w_d,v_d, F_w,dummy3d, F_xth,F_yth,F_zth, &
                            DTH_1,CLIP_TRAJ, DO_W, i0,in,j0,jn, &
                            F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0,F_nk,F_nk_winds)
      else
         call adx_int_winds_glb (w_d,v_d, F_w,dummy3d, F_xth,F_yth,F_zth, &
                            DTH_1,DO_W, i0,in,j0,jn, &
                            F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0,F_nk,F_nk_winds)
      endif

!$omp parallel private(k,j,i) shared (u_d,v_d,w_d)
      if(adx_trapeze_L) then
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
      else ! mid-point rule !
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
   enddo DO_ITER

   if(adx_trapeze_L)then
      F_px=F_xth
      F_py=F_yth
      F_pz=F_zth
      F_wdm=w_d
   else
       call adx_trajex2 (F_px, F_py, F_xct1,F_yct1,F_zct1, &
                        F_xcth,F_ycth,F_zcth,i0,in,j0,jn,k0)

!$omp parallel private(k,j,i)
!$omp do
      do k = k0,F_nk
         do j = j0,jn
         do i = i0,in
            F_pz(i,j,k) = F_zth(i,j,k) - adx_verZ_8%m(k)
            F_pz(i,j,k) = Adx_verZ_8%m(k) + 2.0 * F_pz(i,j,k)
            F_pz(i,j,k) = min(zbot_bound,max(F_pz(i,j,k),ztop_bound))
         enddo
         enddo
      enddo
!$omp enddo
!$omp end parallel
   endif

   call msg(MSG_DEBUG,'adx_pos_m [end]')

   !---------------------------------------------------------------------

   return
end subroutine adx_pos_m
