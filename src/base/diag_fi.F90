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

!**s/r diag_fi - Computes geopotential by vertically integrating the temperature

      subroutine diag_fi ( F_fi, F_s, F_t, F_q, F_fis, &
                           Minx,Maxx,Miny,Maxy, Nk, i0,in,j0,jn )
      implicit none
#include <arch_specific.hf>
    
      integer Minx,Maxx,Miny,Maxy,Nk,i0,in,j0,jn
      real F_fi (Minx:Maxx,Miny:Maxy,Nk+1),F_q(Minx:Maxx,Miny:Maxy,2:Nk+1)
      real F_s  (Minx:Maxx,Miny:Maxy)     ,F_t(Minx:Maxx,Miny:Maxy,Nk  )
      real F_fis(Minx:Maxx,Miny:Maxy)

!author
!
! Andre Plante july 2006.
!
!revision
!
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_50 - Desgagne M 	    - re-visit interface
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_fi         O    - geopotential
! F_s          I    - log(pis/pistars)
! F_t          I    - temperature
! F_fis        I    - surface geopotential

#include "glb_ld.cdk"
#include "dcst.cdk"
#include "ver.cdk"
#include "schm.cdk"
 
      integer i,j,k,kq,kmq
      real*8, parameter :: one = 1.d0, half = .5d0
      real*8  qbar,w1
!
!     ---------------------------------------------------------------
!
!$omp parallel private(qbar,w1,kq,kmq)

!$omp do 
      do j=j0,jn
         do i=i0,in
            F_fi(i,j,G_nk+1)= F_fis(i,j)
         end do
      end do
!$omp enddo

      if (Schm_hydro_L) then

!$omp do 
      do j=j0,jn
         do k= G_nk,1,-1
            w1= Dcst_rgasd_8*Ver_dz_8%t(k)
            do i= i0,in
               F_fi(i,j,k)= F_fi(i,j,k+1)+w1*F_t(i,j,k)*(one+Ver_dbdz_8%t(k)*F_s(i,j))
            end do
         end do
      end do
!$omp enddo

      else

!$omp do 
      do j=j0,jn
         do k= G_nk,1,-1
            kq = max(2,k)
            kmq= max(2,k-1)
            w1= Dcst_rgasd_8*Ver_dz_8%t(k)
            do i= i0,in
               qbar=Ver_wpstar_8(k)*F_q(i,j,k+1)+Ver_wmstar_8(k)*half*(F_q(i,j,kq)+F_q(i,j,kmq))
               qbar=Ver_wp_8%t(k)*qbar+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k)
               F_fi(i,j,k)= F_fi(i,j,k+1)+w1*F_t(i,j,k)*exp(-qbar)*(one+Ver_dbdz_8%t(k)*F_s(i,j))
            end do
         end do
      end do
!$omp enddo

      endif
!$omp end parallel
!
!     ---------------------------------------------------------------
!
      return
      end
