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

!**s/r hz_div_damp

      subroutine hz_div_damp ( F_du,F_dv, F_u, F_v, &
                              i0u,inu,j0u,jnu,i0v,inv,j0v,jnv, &
                              i0,in,j0,jn,Minx,Maxx,Miny,Maxy,Nk )
      use gem_options
      use tdpack
      implicit none
#include <arch_specific.hf>

      real, dimension(Minx:Maxx,Miny:Maxy,NK),   intent (INOUT) :: F_du, F_dv
      real, dimension(Minx:Maxx,Miny:Maxy,NK),   intent (IN)    :: F_u, F_v
      integer, intent(IN) :: i0u,inu,j0u,jnu,i0v,inv,j0v,jnv
      integer, intent(IN) :: i0,in,j0,jn,Minx,Maxx,Miny,Maxy,Nk
!author
!   Claude Girard
!
#include "geomg.cdk"
#include "cstv.cdk"

      integer i,j,k
      real div(Minx:Maxx,Miny:Maxy,Nk)
      real*8 kdiv_damp,kdiv_damp_max
!
!     ---------------------------------------------------------------
!
      kdiv_damp_max=0.25*(rayt_8*Geomg_hx_8)**2/Cstv_dt_8
      kdiv_damp=Hzd_div_damp*kdiv_damp_max/Cstv_bA_m_8

      do k=1,Nk
         do j=j0v,jnv+1
         do i=i0u,inu+1
            div(i,j,k) = (F_u (i,j,k)-F_u (i-1,j,k))*geomg_invDXM_8(j) &
                       + (F_v (i,j,k)*geomg_cyM_8(j)-F_v (i,j-1,k)*geomg_cyM_8(j-1))*geomg_invDYM_8(j)
         enddo
         enddo
      enddo

      do k =1, Nk
         do j=j0u,jnu
         do i=i0u,inu
            F_du(i,j,k)=F_du(i,j,k)+kdiv_damp*(div(i+1,j,k)-div(i,j,k))*geomg_invDXMu_8(j)
         enddo
         enddo
         do j=j0v,jnv
         do i=i0v,inv
            F_dv(i,j,k)=F_dv(i,j,k)+kdiv_damp*(div(i,j+1,k)-div(i,j,k))*geomg_invDYMv_8(j)
         enddo
         enddo
      enddo

!     ---------------------------------------------------------------

!
      return
      end subroutine hz_div_damp
