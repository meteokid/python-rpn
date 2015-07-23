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
      subroutine adv_thermo2mom (F_fld_m, F_fld_t, F_ni,F_nj,F_nk, F_minx, F_maxx, F_miny, F_maxy)

implicit none

#include <arch_specific.hf>
      integer, intent(in) :: F_ni,F_nj,F_nk
      integer, intent(in) :: F_minx,F_maxx,F_miny,F_maxy
		real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk),intent(in) :: F_fld_t
		real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk), intent(out) :: F_fld_m

#include "glb_ld.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "ver.cdk"

      integer :: i,j,k,km2, i0,j0,in,jn
		real*8  :: xx, x1, x2, x3, x4, w1, w2, w3, w4
		real*8  :: zd_z_8(F_nk+1)

#define lag3(xx, x1, x2, x3, x4)  ((((xx) - (x2)) * ((xx) - (x3)) * ((xx) - (x4)))/( ((x1) - (x2)) * ((x1) - (x3)) * ((x1) - (x4))))

!
!     ---------------------------------------------------------------
!      
      zd_z_8(1) = Cstv_Ztop_8
      zd_z_8(2:F_nk+1) = Ver_z_8%t(1:F_nk)

!$omp parallel private(i0,in,j0,jn,xx,x1,x2,x3,x4,&
!$omp                  i,j,k,w1,w2,w3,w4)
      i0 = 1
      in = F_ni
      j0 = 1
      jn = F_nj
      if (G_lam .and. .not. Grd_yinyang_L) then
         if (l_west)  i0 = 3
         if (l_east)  in = F_ni - 1
         if (l_south) j0 = 3
         if (l_north) jn = F_nj - 1
      endif

!$omp do
      do k=2,F_nk-1
         xx = Ver_z_8%m(k)
         x1 = zd_z_8(k-1)
         x2 = zd_z_8(k)
         x3 = zd_z_8(k+1)
         x4 = zd_z_8(k+2)
         w1 = lag3(xx, x1, x2, x3, x4)
         w2 = lag3(xx, x2, x1, x3, x4)
         w3 = lag3(xx, x3, x1, x2, x4)
         w4 = lag3(xx, x4, x1, x2, x3)

! zdt=0 is not present in vector but this allow to use this
! boundary condition anyway.
         km2=max(1,k-2)

         if(k.eq.2) then
            w1=0.d0
         end if

         do j = j0, jn
            do i = i0, in
               F_fld_m(i,j,k)= &
               w1*F_fld_t(i,j,km2) + w2*F_fld_t(i,j,k-1)  + &
               w3*F_fld_t(i,j,k  ) + w4*F_fld_t(i,j,k+1)
            enddo
         enddo
      enddo
!$omp enddo

!- Note zdot at top = 0
      k = 1
      w2 = (zd_z_8(k)-Ver_z_8%m(k)) / (zd_z_8(k)-zd_z_8(k+1))

!$omp do
      do j = j0, jn
         do i = i0, in
            F_fld_m(i,j,1) = w2 * F_fld_t(i,j,1)
         enddo
      enddo
!$omp enddo

!- Note  zdot at surface = 0
      k = F_nk
      w1 = (Ver_z_8%m(k)-zd_z_8(k+1)) / (zd_z_8(k)-zd_z_8(k+1))

!$omp do
      do j = j0, jn
         do i = i0, in
            F_fld_m(i,j,F_nk) = w1 * F_fld_t(i,j,F_nk-1)
         enddo
      enddo
!$omp enddo

!$omp end parallel
!
!     ---------------------------------------------------------------
!      
      end subroutine adv_thermo2mom
