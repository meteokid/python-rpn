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
      implicit none
#include <arch_specific.hf>

      integer :: F_ni,F_nj, F_nk,F_k0
      integer i0,in,j0,jn,k00
      real, dimension(F_ni,F_nj,F_nk) :: F_xt,F_yt,F_zt
      real, dimension(F_ni,F_nj,F_nk) :: F_xm,F_ym,F_zm
      real, dimension(F_ni,F_nj,F_nk) :: F_wat,F_wdm
      logical :: F_cubic_L

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
#include "adv_grid.cdk"
#include "cstv.cdk"

      integer :: i,j,k
      real*8, dimension(2:F_nk-2) :: w1, w2, w3, w4
      real*8, dimension(i0:in,F_nk) :: wdt
      real*8 :: lag3, hh, x, x1, x2, x3, x4, ww, wp, wm
      real :: ztop_bound, zbot_bound

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
      ztop_bound=adv_verZ_8%m(0)
      zbot_bound=adv_verZ_8%m(F_nk+1)

! Prepare parameters for cubic intepolation
      do k=2,F_nk-2
         hh = adv_verZ_8%t(k)
         x1 = adv_verZ_8%m(k-1)
         x2 = adv_verZ_8%m(k  )
         x3 = adv_verZ_8%m(k+1)
         x4 = adv_verZ_8%m(k+2)
         w1(k) = lag3( hh, x1, x2, x3, x4 )
         w2(k) = lag3( hh, x2, x1, x3, x4 )
         w3(k) = lag3( hh, x3, x1, x2, x4 )
         w4(k) = lag3( hh, x4, x1, x2, x3 )
      enddo

      ww=(adv_verZ_8%m(F_nk+1)-adv_verZ_8%t(F_nk))/(adv_verZ_8%m(F_nk+1)-adv_verZ_8%t(F_nk-1))
      wp=(adv_verZ_8%t(F_nk)-adv_verZ_8%m(F_nk-1))/(adv_verZ_8%m(F_nk)-adv_verZ_8%m(F_nk-1))
      wm=(adv_verZ_8%m(F_nk)-adv_verZ_8%t(F_nk))/(adv_verZ_8%m(F_nk)-adv_verZ_8%m(F_nk-1))

      k00=max(F_k0-1,1)

!$omp parallel private(i,j,k,wdt)
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

      do k=k00,F_nk-1
         if(F_cubic_L.and.k.ge.2.and.k.le.F_nk-2)then
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

              !working with displacements for the vertical positions
               wdt(i,k) =    w1(k)*F_wdm(i,j,k-1)+ &
                             w2(k)*F_wdm(i,j,k  )+ &
                             w3(k)*F_wdm(i,j,k+1)+ &
                             w4(k)*F_wdm(i,j,k+2)
            enddo
         else
           !Linear
            do i=i0,in
               F_xt(i,j,k) = (F_xm(i,j,k )+F_xm (i,j,k+1))*0.5d0
               F_yt(i,j,k) = (F_ym(i,j,k )+F_ym (i,j,k+1))*0.5d0
               wdt (i,k)   = (F_wdm(i,j,k)+F_wdm(i,j,k+1))*0.5d0
            enddo
         endif

         do i=i0,in
            F_zt(i,j,k)=adv_verZ_8%t(k)-(wdt(i,k)+F_wat(i,j,k))*cstv_dt_8*0.5d0
            F_zt(i,j,k)=max(F_zt(i,j,k),ztop_bound)
            F_zt(i,j,k)=min(F_zt(i,j,k),zbot_bound)
         enddo

      enddo

     !for the last level
      do i=i0,in
        !extrapolating horizontal positions
         F_xt(i,j,F_nk)=wp*F_xm(i,j,F_nk)+wm*F_xm(i,j,F_nk-1)
         F_yt(i,j,F_nk)=wp*F_ym(i,j,F_nk)+wm*F_ym(i,j,F_nk-1)

        !interpolating vertical positions
         F_zt(i,j,F_nk)= adv_verZ_8%t(F_nk)-ww*(wdt(i,F_nk-1)+&
                         F_wat(i,j,F_nk-1))*cstv_dt_8*0.5d0
         F_zt(i,j,F_nk)= min(F_zt(i,j,F_nk),zbot_bound)
      enddo

      enddo
!$omp enddo
!$omp end parallel
!     
!---------------------------------------------------------------------
! 
      return
      end subroutine adv_int_vert_t
