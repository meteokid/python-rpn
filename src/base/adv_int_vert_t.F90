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
      subroutine adv_int_vert_t ( F_xt, F_yt, F_zt, F_xtn, F_ytn, F_ztn, F_xm, F_ym, F_zm, &
                                  F_wat, F_wdm, F_ni,F_nj, F_nk     , &
                                  F_k0, i0,in,j0,jn, F_cubic_L )
      implicit none
#include <arch_specific.hf>

      integer :: F_ni,F_nj, F_nk,F_k0
      integer i0,in,j0,jn,k00
      real, dimension(F_ni,F_nj,F_nk) :: F_xt,F_yt,F_zt
      real, dimension(F_ni,F_nj)      :: F_xtn,F_ytn,F_ztn
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
#include "ver.cdk"
#include "cstv.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "ptopo.cdk"

      integer :: i,j,k
      integer :: BCS_BASE
      integer :: n,cnt,nc,nc1,nc2,sum_cnt,totaln,err
      real*8  two, half, EPS_8
      real  :: xpos,ypos
      real*8, dimension(2:F_nk-2) :: w1, w2, w3, w4
      real*8, dimension(i0:in,F_nk) :: wdt
      real*8 :: lag3, hh, x, x1, x2, x3, x4, ww, wp, wm
      real :: ztop_bound, zbot_bound
      real :: minposx,maxposx,minposy,maxposy
      real :: prct
      parameter (two = 2.0, half=0.5, EPS_8 = 1.D-5)
     
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

      ztop_bound=Ver_z_8%m(0)
      zbot_bound=Ver_z_8%m(F_nk+1)

       BCS_BASE= 4
      if (Grd_yinyang_L) BCS_BASE = 3
      minposx = adv_xx_8(adv_lminx+1) + EPS_8
      if (l_west)  minposx = adv_xx_8(1+BCS_BASE) + EPS_8
      maxposx = adv_xx_8(adv_lmaxx-1) - EPS_8
      if (l_east)  maxposx = adv_xx_8(F_ni-BCS_BASE) - EPS_8
      minposy = adv_yy_8(adv_lminy+1) + EPS_8
      if (l_south) minposy = adv_yy_8(1+BCS_BASE) + EPS_8
      maxposy = adv_yy_8(adv_lmaxy-1) - EPS_8
      if (l_north) maxposy = adv_yy_8(F_nj-BCS_BASE) - EPS_8

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

cnt=0

!$omp parallel private(i,j,k,wdt,ww,wp,wm,nc,nc1,nc2)  
nc = 0
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
               xpos = w1(k)*F_xm(i,j,k-1)+ &
                             w2(k)*F_xm(i,j,k  )+ &
                             w3(k)*F_xm(i,j,k+1)+ &
                             w4(k)*F_xm(i,j,k+2)
               ypos = w1(k)*F_ym(i,j,k-1)+ &
                             w2(k)*F_ym(i,j,k  )+ &
                             w3(k)*F_ym(i,j,k+1)+ &
                             w4(k)*F_ym(i,j,k+2)
              F_xt(i,j,k) = min(max(xpos,minposx),maxposx)
              F_yt(i,j,k) = min(max(ypos,minposy),maxposy)
              ! Clipp traj stat
              nc=nc+min(1,max(0,ceiling(abs(F_xt(i,j,k)-xpos)+abs(F_yt(i,j,k)-ypos))))
            enddo
         else
           !Linear
            do i=i0,in
               xpos = (F_xm(i,j,k )+F_xm (i,j,k+1))*half
               ypos = (F_ym(i,j,k )+F_ym (i,j,k+1))*half
               F_xt(i,j,k) = min(max(xpos,minposx),maxposx)
               F_yt(i,j,k) = min(max(ypos,minposy),maxposy)
               ! Clipp traj stat
              nc=nc+min(1,max(0,ceiling(abs(F_xt(i,j,k)-xpos)+abs(F_yt(i,j,k)-ypos))))
            enddo
         endif
      enddo
  
   if(Schm_trapeze_L) then
         !working with displacements for the vertical position
 
       do k=k00,F_nk-1            
         do i=i0,in
           if(k.ge.2.and.k.le.F_nk-2)then
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
            F_zt(i,j,k)=max(F_zt(i,j,k),ztop_bound)
            F_zt(i,j,k)=min(F_zt(i,j,k),zbot_bound)
         enddo
      enddo
 
        !for the last level when at the surface
         wp=(Ver_z_8%m(F_nk+1)-Ver_z_8%m(F_nk-1))*Ver_idz_8%t(F_nk-1)
         wm=1.d0-wp
          do i=i0,in
           !extrapolating horizontal positions downward
           xpos = wp*F_xm(i,j,F_nk)+wm*F_xm(i,j,F_nk-1)
           ypos = wp*F_ym(i,j,F_nk)+wm*F_ym(i,j,F_nk-1)
           F_xt(i,j,F_nk) = min(max(xpos,minposx),maxposx)
           F_yt(i,j,F_nk) = min(max(ypos,minposy),maxposy)
           ! Clipp traj stat
           nc=nc+min(1,max(0,ceiling(abs(F_xt(i,j,F_nk)-xpos)+abs(F_yt(i,j,F_nk)-ypos))))

           !vertical position
           F_zt(i,j,F_nk)=zbot_bound
          enddo

          !for the last level when half way between surface and last momentum level
           ww=Ver_wmstar_8(F_nk)
           wp=(Ver_z_8%t(F_nk  )-Ver_z_8%m(F_nk-1))*Ver_idz_8%t(F_nk-1)
           wm=1.d0-wp
          do i=i0,in
           !extrapolating horizontal positions downward
           F_xtn(i,j)=wp*F_xm(i,j,F_nk)+wm*F_xm(i,j,F_nk-1)
           F_ytn(i,j)=wp*F_ym(i,j,F_nk)+wm*F_ym(i,j,F_nk-1)
           F_xtn(i,j) = min(max(F_xtn(i,j),minposx),maxposx)
           F_ytn(i,j) = min(max(F_ytn(i,j),minposy),maxposy)

           !interpolating vertical positions     
           F_ztn(i,j)= Ver_z_8%t(F_nk)-ww*(Cstv_dtD_8*  wdt(i,  F_nk-1) &
                                       +Cstv_dtA_8*F_wat(i,j,F_nk-1))
           F_ztn(i,j)= min(F_ztn(i,j),zbot_bound)
          enddo
     
   else     
        !working directly with positions
         do k=k00,F_nk-1
            do i=i0,in
               if(k.ge.2.and.k.le.F_nk-2)then
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
               F_zt(i,j,k)=max(F_zt(i,j,k),ztop_bound)
               F_zt(i,j,k)=min(F_zt(i,j,k),zbot_bound)
            end do
         end do
      ! For last thermodynamic level, positions in the horizontal are those  
      ! of the momentum levels; no displacement allowed in the vertical      
      ! at bottum. At top vertical displacement is obtian from linear inter. 
      ! and is bound to first thermo level.                                  
          do i=i0,in
             F_xt(i,j,F_nk) = F_xm(i,j,F_nk)
             F_yt(i,j,F_nk) = F_ym(i,j,F_nk)
             F_zt(i,j,F_nk) = zbot_bound
             F_xtn(i,j) = F_xm(i,j,F_nk)
             F_ytn(i,j) = F_ym(i,j,F_nk)
             F_ztn(i,j) = zbot_bound
          enddo

   endif

enddo
!$omp enddo
!$omp atomic
cnt=cnt+nc
!$omp end parallel
     
! Trajectory Clipping stat   
  call adv_print_cliptrj_s (cnt,F_ni,F_nj,F_nk,k00,'INTERP '//trim('t'))


!
!--------------------------------------------------------------
!
   return
   end subroutine adv_int_vert_t
