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

!**s/r xyfil - Filtering of a field (x-y filter)
!
      subroutine xyfil ( F_d, F_s, F_w1, F_co,  &
                         F_pole_L, Minx,Maxx,Miny,Maxy,Nk)
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, Nk
      logical F_pole_L
      real F_d (Minx:Maxx,Miny:Maxy,Nk), F_s(Minx:Maxx,Miny:Maxy,Nk), &
           F_w1(Minx:Maxx,Miny:Maxy,Nk), F_co

!author
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v2_31 - Desgagne M.       - remove stkmemw
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_d           O              output field 
! F_s           I              input  field 
! F_w1          -              work   field 
! F_co          I              filtering coeficient 
!                              ( 0.0 <= F_co <= 0.5)
! F_whx_8       I              grid point spacings on x-axis
! F_why_8       I              grid point spacings on y-axis
! F_pole_L      I              field includes poles if .true.
!----------------------------------------------------------------

#include "glb_ld.cdk"
#include "dcst.cdk"

!      real w2hx(l_minx:l_maxx), w2hy(l_miny:l_maxy)

      real prmean1, prmean2, prcom1
      integer i, j, k, i0, in, j0, jn
      real*8, parameter :: half=0.5d0
      

!
!     ---------------------------------------------------------------
!
      prcom1 = 1. - F_co
!
!
      call rpn_comm_xch_halo (F_s,l_minx,l_maxx,l_miny,l_maxy,&
           l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
!   INTERPOLATION ALONG X
!
!************************
      i0 = 1 + pil_w
      if ((l_west).and.(G_lam)) i0 = i0+1
      in = l_niu - pil_e
      j0 = 1 + pil_s
      if (l_south) j0 = j0+1
      jn = l_njv - pil_n
!
      do k=1,Nk
         do j= 1+pil_s, l_nj-pil_n 
         do i=i0,in
            F_w1(i,j,k)= F_co * (F_s(i-1,j,k)+F_s(i+1,j,k))*half &
                                + prcom1 * F_s (i,j,k)
         end do
         end do
      end do
      if (G_lam) then
      if (l_west) then
         do k=1,Nk
         do j= 1+pil_s, l_nj-pil_n 
            F_w1(i0-1,j,k) = F_s(i0-1,j,k)
         end do
         end do
      endif
      if (l_east) then
         do k=1,Nk
         do j= 1+pil_s, l_nj-pil_n 
            F_w1(in+1,j,k) = F_s(in+1,j,k)
         end do
         end do
      endif
      endif
!
      call rpn_comm_xch_halo (F_w1,l_minx,l_maxx,l_miny,l_maxy, &
           l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
!   INTERPOLATION ALONG Y
!
!*************************
      do k=1,Nk
        do j=j0,jn
        do i= 1+pil_w, l_ni-pil_e 
           F_d(i,j,k)= F_co * (F_w1(i,j-1,k)+F_w1(i,j+1,k))*half &
                              + prcom1 * F_w1(i,j,k)
        end do
        end do
      end do

      if (l_south) then
         do k=1,Nk
         do i= 1+pil_w, l_ni-pil_e 
            F_d(i,j0-1,k) = F_w1(i,j0-1,k)
         end do
         end do
      endif
      if (l_north) then
         do k=1,Nk
         do i= 1+pil_w, l_ni-pil_e 
            F_d(i,jn+1,k) = F_w1(i,jn+1,k)
         end do
         end do
      endif
!
!     ---------------------------------------------------------------
!
      return
      end
