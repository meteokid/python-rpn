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

!**s/r dry_sfc_pressure - Compute dry air surface pressure
!
      subroutine dry_sfc_pressure (F_drysfcp0, presT, p0T, &
                                   Minx,Maxx,Miny,Maxy,Nk,F_timelevel_S)
      use glb_ld
      use cstv
      use gmm_itf_mod
      implicit none
#include <arch_specific.hf>

      character*1 F_timelevel_S
      integer Minx,Maxx,Miny,Maxy,Nk
      real F_drysfcp0(Minx:Maxx,Miny:Maxy),presT(Minx:Maxx,Miny:Maxy,Nk),&
           p0T(Minx:Maxx,Miny:Maxy)

!author
!     Michel Desgagne --  fall 2014
!
!revision
! v4_70 - M. Desgagne      - Initial version


      integer i,j,k,istat
      real, dimension(Minx:Maxx,Miny:Maxy,Nk) :: sumq
      real, pointer, dimension(:,:,:)         :: tr
!     ________________________________________________________________
!
      call sumhydro (sumq,Minx,Maxx,Miny,Maxy,Nk,F_timelevel_S)

      istat = gmm_get('TR/HU:'//F_timelevel_S,tr)

!$omp parallel shared(sumq,tr,F_drysfcp0,presT,p0T)

!$omp do
      do k=1,Nk
         sumq(1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)= &
         sumq(1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)+ &
         tr  (1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)
      end do
!$omp enddo
!$omp do
      do j=1+pil_s,l_nj-pil_n
         F_drysfcp0(:,j) = 0.
         do k=1,Nk-1
         do i=1+pil_w,l_ni-pil_e
            F_drysfcp0(i,j)= F_drysfcp0(i,j) + &
                 (1.-sumq(i,j,k))*(presT(i,j,k+1) - presT(i,j,k))
         enddo
         end do
         do i=1+pil_w,l_ni-pil_e
            F_drysfcp0(i,j)= F_drysfcp0(i,j) + &
                 (1.-sumq(i,j,Nk))*(p0T(i,j) - presT(i,j,Nk)) - Cstv_pref_8
         end do
      end do
!$omp enddo

!$omp end parallel

!     ________________________________________________________________
!
      return
      end

