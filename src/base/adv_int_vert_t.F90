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
!
      subroutine adv_int_vert_t ( F_xt, F_yt, F_zt, F_xm, F_ym, F_zm, &
                                  F_wat, F_wdm, F_ni,F_nj, F_nk     , &
                                  F_k0, i0,in,j0,jn, F_cubic_L )
      use grid_options
      use gem_options
      use adv_options, only: adv_slt_winds
      use glb_ld
      use cstv
      use ver
      use adv_grid
      use outgrid
      use gmm_itf_mod
      use gmm_pw, only: gmmk_pw_uslt_s, gmmk_pw_vslt_s, pw_uslt, pw_vslt
      use dcst, only: Dcst_inv_rayt_8
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: F_ni,F_nj, F_nk,F_k0
      integer, intent(in) :: i0,in,j0,jn
      real, dimension(F_ni,F_nj,F_nk), intent(out) :: F_xt,F_yt,F_zt
      real, dimension(F_ni,F_nj,F_nk), intent(in) :: F_xm,F_ym,F_zm
      real, dimension(F_ni,F_nj,F_nk), intent(in) :: F_wat,F_wdm
      logical, intent(in) :: F_cubic_L

!authors
!     A. Plante & C. Girard
!
!object
!     see id section
!
!arguments
!______________________________________________________________________
!              |                                                 |     |
! NAME         | DESCRIPTION                                     | I/O |
!--------------|-------------------------------------------------|-----|
!              |                                                 |     |
! F_xt         | upwind longitudes for themodynamic level        |  o  |
! F_yt         | upwind latitudes for themodynamic level         |  o  |
! F_zt         | upwind height for themodynamic level            |  o  |
! F_xm         | upwind longitudes for momentum level            |  i  |
! F_ym         | upwind latitudes for momentum level             |  i  |
! F_zm         | upwind height for momentum level                |  i  |
!______________|_________________________________________________|_____|


#include "constants.h"

      integer :: i,j,k,k00,err
      real*8, dimension(2:F_nk-2) :: w1, w2, w3, w4
      real*8, dimension(i0:in,F_nk) :: wdt
      real*8 :: lag3, hh, x, x1, x2, x3, x4, ww, wp, wm, inv_cy_8
      real :: zmin_bound, zmax_bound, z_bottom
      real*8, parameter :: two = 2.0d0, half=0.5d0

      lag3( x, x1, x2, x3, x4 ) = &
        ( ( x  - x2 ) * ( x  - x3 ) * ( x  - x4 ) )/ &
        ( ( x1 - x2 ) * ( x1 - x3 ) * ( x1 - x4 ) )

!
!---------------------------------------------------------------------
!
!***********************************************************************
! Note : extra computation are done in the pilot zone if
!        (Lam_gbpil_t != 0) for coding simplicity
!***********************************************************************
!

      zmin_bound = Ver_zmin_8
      zmax_bound = Ver_zmax_8
      z_bottom   = Ver_z_8%m(F_nk+1)

! Prepare parameters for cubic intepolation
      do k=2,F_nk-2
          hh = Ver_z_8%t(k)
          x1 = Ver_z_8%m(k-1)
          x2 = Ver_z_8%m(k  )
          x3 = Ver_z_8%m(k+1)
          x4 = Ver_z_8%m(k+2)
          w1(k) = lag3( hh, x1, x2, x3, x4 )
          w2(k) = lag3( hh, x2, x1, x3, x4 )
          w3(k) = lag3( hh, x3, x1, x2, x4 )
          w4(k) = lag3( hh, x4, x1, x2, x3 )
      enddo

      k00=max(F_k0-1,1)

!$omp parallel private(i,j,k,wdt,ww,wp,wm) &
!$omp          shared(k00, w1, w2, w3, w4, &
!$omp                 zmin_bound,zmax_bound,z_bottom)
!$omp do
  do j=j0,jn

!     Fill non computed upstream positions with zero to avoid math exceptions
!     in the case of top piloting
      do k=1,k00-1
         do i=i0,in
           F_xt(i,j,k)=0.0
           F_yt(i,j,k)=0.0
           F_zt(i,j,k)=0.0
         end do
      enddo
      inv_cy_8 = 1d0 / adv_cy_8(j)

!     Retrieve wind information at the lowest thermodynamic level if available
!     from the surface layer scheme
      nullify(pw_uslt,pw_vslt)
      if (adv_slt_winds) then
         err = gmm_get(gmmk_pw_uslt_s, pw_uslt)
         err = gmm_get(gmmk_pw_vslt_s, pw_vslt)
      endif

      do k=k00,F_nk-1
         if(F_cubic_L.and.k >= 2.and.k <= F_nk-2)then
           !Cubic
            do i=i0,in
               F_xt(i,j,k) = w1(k)*F_xm(i,j,k-1)+ &
                             w2(k)*F_xm(i,j,k  )+ &
                             w3(k)*F_xm(i,j,k+1)+ &
                             w4(k)*F_xm(i,j,k+2)
               F_yt(i,j,k) = w1(k)*F_ym(i,j,k-1)+ &
                             w2(k)*F_ym(i,j,k  )+ &
                             w3(k)*F_ym(i,j,k+1)+ &
                             w4(k)*F_ym(i,j,k+2)
            enddo
         else
           !Linear
            do i=i0,in
               F_xt(i,j,k) = (F_xm(i,j,k )+F_xm (i,j,k+1))*half
               F_yt(i,j,k) = (F_ym(i,j,k )+F_ym (i,j,k+1))*half
            enddo
         endif
      enddo

      if(Schm_trapeze_L) then
         !working with displacements for the vertical position

          do k=k00,F_nk-1
            do i=i0,in
              if(k >= 2.and.k <= F_nk-2)then
                  !Cubic
                   wdt(i,k) = &
                       w1(k)*F_wdm(i,j,k-1)+ &
                       w2(k)*F_wdm(i,j,k  )+ &
                       w3(k)*F_wdm(i,j,k+1)+ &
                       w4(k)*F_wdm(i,j,k+2)
              else
                   !Linear
                   wdt(i,k) = (F_wdm(i,j,k)+F_wdm(i,j,k+1))*0.5d0
              endif

              F_zt(i,j,k)=Ver_z_8%t(k) - Cstv_dtzD_8*  wdt(i,  k) &
                                       - Cstv_dtzA_8*F_wat(i,j,k)
              F_zt(i,j,k)=max(F_zt(i,j,k),zmin_bound)
              F_zt(i,j,k)=min(F_zt(i,j,k),zmax_bound)
            enddo
          enddo

          ! The lowest thermodynamic level is half way between the surface
          ! and the lowest momentum level
          if (associated(pw_uslt) .and. associated(pw_vslt)) then
             ! Use surface layer winds for horizontal displacements
             do i=i0,in
                F_xt(i,j,F_nk) = adv_xx_8(i) - Dcst_inv_rayt_8 * Cstv_dt_8 * pw_uslt(i,j)*inv_cy_8
                F_yt(i,j,F_nk) = adv_yy_8(j) - Dcst_inv_rayt_8 * Cstv_dt_8 * pw_vslt(i,j)
             enddo
          else
             ! Extrapolate horizontal positions downwards
             ww=Ver_wmstar_8(F_nk)
             wp=(Ver_z_8%t(F_nk  )-Ver_z_8%m(F_nk-1))*Ver_idz_8%t(F_nk-1)
             wm=1.d0-wp
             do i=i0,in
                F_xt(i,j,F_nk)=wp*F_xm(i,j,F_nk)+wm*F_xm(i,j,F_nk-1)
                F_yt(i,j,F_nk)=wp*F_ym(i,j,F_nk)+wm*F_ym(i,j,F_nk-1)
             enddo
          endif
          ! Interpolate vertical positions
          do i=i0,in
             F_zt(i,j,F_nk)= Ver_z_8%t(F_nk)-ww*(Cstv_dtD_8*  wdt(i,  F_nk-1) &
                  +Cstv_dtA_8*F_wat(i,j,F_nk-1))
             F_zt(i,j,F_nk)= min(F_zt(i,j,F_nk),zmax_bound)
             F_zt(i,j,F_nk)= max(F_zt(i,j,F_nk),zmin_bound)
          enddo

      else
          !working directly with positions
          do k=k00,F_nk-1
             do i=i0,in
                if(k >= 2.and.k <= F_nk-2)then
                  !Cubic
                   F_zt(i,j,k)= &
                        w1(k)*F_zm(i,j,k-1)+ &
                        w2(k)*F_zm(i,j,k  )+ &
                        w3(k)*F_zm(i,j,k+1)+ &
                        w4(k)*F_zm(i,j,k+2)
                else
                  !Linear
                   F_zt(i,j,k) = (F_zm(i,j,k)+F_zm(i,j,k+1))*half
                endif
                ! Must stay in domain
                   F_zt(i,j,k)=max(F_zt(i,j,k),zmin_bound)
                   F_zt(i,j,k)=min(F_zt(i,j,k),zmax_bound)
             end do
          end do
      ! For last thermodynamic level, positions in the horizontal are those
      ! of the momentum levels; no displacement allowed in the vertical
      ! at bottum. At top vertical displacement is obtian from linear inter.
      ! and is bound to first thermo level.
          do i=i0,in
             F_xt(i,j,F_nk) = F_xm(i,j,F_nk)
             F_yt(i,j,F_nk) = F_ym(i,j,F_nk)
             F_zt(i,j,F_nk) = z_bottom
          enddo

      endif

  enddo
!$omp enddo
!$omp end parallel


!
!--------------------------------------------------------------
!
   return
   end subroutine adv_int_vert_t
