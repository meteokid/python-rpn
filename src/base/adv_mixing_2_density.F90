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

!**s/p adv_mixing_2_density : F_kind=1=Mixing ratio to Density; F_kind=2=Density to Mixing ratio

      subroutine adv_mixing_2_density (F_tr_mixing,F_tr_density,F_minx,F_maxx,F_miny,F_maxy,F_nk,F_kind,F_time)

      use glb_ld
      implicit none

#include <arch_specific.hf>

      integer :: F_minx,F_maxx,F_miny,F_maxy,F_nk,F_kind,F_time

      real F_tr_mixing(F_minx:F_maxx,F_miny:F_maxy,F_nk),F_tr_density(F_minx:F_maxx,F_miny:F_maxy,F_nk)

      !@author Monique Tanguay

      !@revisions
      ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation


      !---------------------------------------------------------------------
      integer i,j,k
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk) :: density,bidon
      !---------------------------------------------------------------------

      call get_density (density,bidon,F_time,F_minx,F_maxx,F_miny,F_maxy,F_nk,1)

      if (F_kind==1) then

!$omp parallel do private(i,j,k)
         do k=1,F_nk
         do j=1,l_nj
         do i=1,l_ni
            F_tr_density(i,j,k) = F_tr_mixing(i,j,k) * density(i,j,k)
         enddo
         enddo
         enddo
!$omp end parallel do

      endif

      if (F_kind==2) then

!$omp parallel do private(i,j,k)
         do k=1,F_nk
         do j=1,l_nj
         do i=1,l_ni
            F_tr_mixing(i,j,k) = F_tr_density(i,j,k) / density(i,j,k)
         enddo
         enddo
         enddo
!$omp end parallel do

      endif

      !---------------------------------------------------------------------

      return
end subroutine adv_mixing_2_density
