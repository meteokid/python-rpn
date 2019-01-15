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

!**s/r hzd_theta_gu_pole

      subroutine hzd_theta_gu_pole( th, Minx,Maxx,Miny,Maxy,Nk )
      use hzd_ctrl
      implicit none
#include <arch_specific.hf>

      integer                                , intent(IN    ) :: &
                                             Minx,Maxx,Miny,Maxy,Nk
      real, dimension(Minx:Maxx,Miny:Maxy,Nk), intent (INOUT) :: th

#include "glb_ld.cdk"
#include "hzd.cdk"

      integer j
      real th_tend(l_ni,l_nj,G_nk)
      real*8 :: w(G_nj)
!
!     _________________________________________________________________
!
      w= 1.d0
      do j = 1, Hzd_theta_njpole_gu_only
         w(j) = dble(j-1)/dble(Hzd_theta_njpole_gu_only)
      end do
      do j = G_nj-Hzd_theta_njpole_gu_only+1, G_nj
         w(j) = max(0.d0,dble(G_nj-j)/dble(Hzd_theta_njpole_gu_only))
      end do
!        Original value before the diffusion (undiffused value)
      th_tend(1:l_ni,1:l_nj,1:G_nk) = th(1:l_ni,1:l_nj,1:G_nk)
      
!        We compute diffused value everywhere 
      call hzd_ctrl4 ( th, 'S', l_minx,l_maxx,l_miny,l_maxy,G_nk)

!        Computation of the increment due to the diffusion
      th_tend(1:l_ni,1:l_nj,1:G_nk) = th(1:l_ni,1:l_nj,1:G_nk) - &
                                      th_tend(1:l_ni,1:l_nj,1:G_nk) 
!        Weighting from  nj_pole in function of the distance to the pole
!        At pole, we have weight = 0, we want th = th_original
!        Between pole and njpole, we have weight = [0,1], we want th = weight * INCREMENT_diffusion + th_original
!        At njpole and further away from pole, we have weight = 1, we want th = th_diffused

      do j=1,l_nj
         th(1:l_ni,j,1:G_nk) = (w(l_j0+j-1)-1.d0)*th_tend(1:l_ni,j,1:G_nk) &
                               + th(1:l_ni,j,1:G_nk)
      enddo
!
!     _________________________________________________________________
!
      return
      end
