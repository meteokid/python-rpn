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

!**s/r intm2t - downward vertical integration of variable ne momentum level.
!

!
      subroutine intm2t (F_gg_t, F_ff_m, F_con, F_hz_m_8, Minx,Maxx,Miny,Maxy,Nk, &
                         F_i0,F_in,F_j0,F_jn,F_k0)
!
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy, Nk,F_i0,F_in,F_j0,F_jn,F_k0
      real F_gg_t(Minx:Maxx,Miny:Maxy,Nk+1),F_ff_m(Minx:Maxx,Miny:Maxy,Nk),F_con
      real*8  F_hz_m_8(Nk)
!
!author
!     A. Plante - cmc - aug 2005 (based on hatoprg.ftn from J. Cote)
!
!revision
!
!object  DOWNWARD vertical integration for momentum variable. The output is on
!        the thermodynamic levels.
!
!                 / z(sfc) 
!        G   =  C |        F dz 
!         k       / z(k)
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_gg_t      O          resulting vertical integration (on thermo levels)
!  F_ff_m      I          input field to be integrated   (on momentum levels)
!  F_con       I          mutiplication coefficient 
!  F_hz_m_8    I          intervals in z-direction (on momentum levels)
!----------------------------------------------------------------
!
 
#include "glb_ld.cdk"
!
      integer i, j, k
      real w1(Minx:Maxx,Miny:Maxy,2)
      real*8  ccc 
!
!notes
!     The computation is done by horizontal slices 
!*
!     __________________________________________________________________
!
      do j=F_j0,F_jn
      do i=F_i0,F_in
         w1(i,j,1) = 0.0
      end do
      end do 

      do k=G_nk,1,-1
         ccc = F_con*F_hz_m_8(k)
         do j=F_j0,F_jn
         do i=F_i0,F_in
            w1(i,j,2) = ccc*F_ff_m(i,j,k)
            F_gg_t(i,j,k+1) = w1(i,j,1)
            w1(i,j,1) = w1(i,j,1) + w1(i,j,2)
         end do
         end do
      end do

      do j=F_j0,F_jn
      do i=F_i0,F_in
         F_gg_t(i,j,1) = w1(i,j,1)
      end do
      end do

      return
      end
