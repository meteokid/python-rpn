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

!**s/r var_topo - varies topography incrementally from analysis
!                 topography to target topography

      subroutine var_topo2 (F_topo, F_step, Minx,Maxx,Miny,Maxy)
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy
      real F_topo (Minx:Maxx,Miny:Maxy), F_step

#include "gmm.hf"
#include "glb_ld.cdk"
#include "p_geof.cdk" 
#include "dcst.cdk"
#include "vtopo.cdk"

      integer i,j, gmmstat
      real*8, parameter :: one = 1.0d0
      real*8, parameter :: two = 2.0d0
      real*8  lt, pio2, f, a, b
!     __________________________________________________________________
!
      gmmstat = gmm_get(gmmk_topo_low_s , topo_low )
      gmmstat = gmm_get(gmmk_topo_high_s, topo_high)

      lt   = Vtopo_ndt
      pio2 = Dcst_pi_8 / two

      f = max(0.,min(F_step-Vtopo_start,real(Vtopo_ndt)))
      b = f / lt
      b = (cos(pio2 * (one-b) ))**2
      a = one - b
      do j= 1, l_nj 
      do i= 1, l_ni
         F_topo (i,j) = a*topo_low(i,j) + b*topo_high(i,j)
      end do
      end do
!     __________________________________________________________________
!
      return
      end
