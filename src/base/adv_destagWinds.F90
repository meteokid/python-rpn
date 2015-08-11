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

      subroutine adv_destagWinds (F_uth, F_vth, minx,maxx,miny,maxy,nk)
      implicit none
#include <arch_specific.hf>
      
      integer minx,maxx,miny,maxy,nk
      real F_uth(minx:maxx,miny:maxy,nk), F_vth(minx:maxx,miny:maxy,nk)

      !@revisions
      ! v4_40 - Qaddouri/Lee      - Yin-Yang, to exchange unstaggered winds

      !@objective unstagger wind components (Interpolate to geopotential grid)

#include "glb_ld.cdk"
#include "dcst.cdk"
#include "grd.cdk"
#include "inuvl.cdk"

      !- CUBIC LAGRANGE INTERPOLATION COEFFICIENTS from AND to U and V grids
      !      real*8 inuvl_wxux3_8(l_minx:l_maxx,4) ! coef for U to PHI-grid
      !      real*8 inuvl_wyvy3_8(l_miny:l_maxy,4) ! coef for V to PHI-grid

      integer :: i,j,k,i0,in,j0,jn,i0u,inu,j0v,jnv,jext
      real, dimension(:,:,:), allocatable :: uu,vv
      real*8  :: inv_rayt_8
      real*8, parameter ::  alpha1=-1.d0/16.d0 , alpha2=9.d0/16.d0
!     
!---------------------------------------------------------------------
! 
      inv_rayt_8 = 1.D0 / Dcst_rayt_8

      call rpn_comm_xch_halo(F_uth,l_minx,l_maxx,l_miny,l_maxy,&
           l_niu,l_nj,l_nk,G_halox,G_haloy,G_periodx,G_periody,G_niu,0)
      call rpn_comm_xch_halo(F_vth,l_minx,l_maxx,l_miny,l_maxy,&
           l_ni,l_njv,l_nk,G_halox,G_haloy,G_periodx,G_periody,G_ni,0)

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj

      i0u = 1
      inu = l_ni
      j0v = 0
      jnv = l_nj

         jext=2
         if (l_west)  i0u = 1    + jext
         if (l_east)  inu = l_ni - jext
         if (l_south) j0v = 1    + jext
         if (l_north) jnv = l_nj - jext

       !- Interpolate advection winds to geopotential grid

      allocate (uu(l_minx:l_maxx,l_miny:l_maxy,l_nk),vv(l_minx:l_maxx,l_miny:l_maxy,l_nk))

!$omp parallel private(i,j,k)
!$omp do
      do k=1,l_nk

         do j = j0, jn
            do i = i0u, inu
               uu(i,j,k) = ( F_uth(i-2,j,k) + F_uth(i+1,j,k) )*alpha1 &
                          +( F_uth(i  ,j,k) + F_uth(i-1,j,k) )*alpha2
                         
            enddo
         enddo

         do j = j0v, jnv
            do i = i0, in
               vv(i,j,k) = inuvl_wyvy3_8(j,1)*F_vth(i,j-2,k) + inuvl_wyvy3_8(j,2)*F_vth(i,j-1,k) &
                         + inuvl_wyvy3_8(j,3)*F_vth(i,j  ,k) + inuvl_wyvy3_8(j,4)*F_vth(i,j+1,k)
            enddo
         enddo

     enddo 
!$omp enddo

      
!$omp do
         do k = 1,l_nk
            do j = j0,jn
               do i = i0u,inu
                  F_uth(i,j,k) = inv_rayt_8 * uu(i,j,k)
               enddo
            enddo
            do j = j0v,jnv
               do i = i0,in
                  F_vth(i,j,k) = inv_rayt_8 * vv(i,j,k)
               enddo
            enddo
         enddo
!$omp enddo
      
!$omp end parallel

      deallocate (uu,vv)
!     
!---------------------------------------------------------------------
!     
 
      return
      end subroutine adv_destagWinds
