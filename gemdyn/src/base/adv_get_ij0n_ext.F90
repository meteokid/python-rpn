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

!**s/p adv_get_ij0n_ext: Establish scope of extended advection operations

      subroutine adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      implicit none

#include <arch_specific.hf>

      integer :: i0_e,j0_e,in_e,jn_e

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation
      ! v4_87 - Tanguay M.        - Adjust extension

#include "glb_ld.cdk"
#include "adv_grid.cdk"
#include "grd.cdk"

      !---------------------------------------------------------------------
      integer :: jext
      !---------------------------------------------------------------------

      i0_e = 1
      in_e = l_ni
      j0_e = 1
      jn_e = l_nj

      jext = Grd_maxcfl + 1

      if (l_west)  i0_e =    1 + pil_w - jext
      if (l_east)  in_e = l_ni - pil_e + jext
      if (l_south) j0_e =    1 + pil_s - jext
      if (l_north) jn_e = l_nj - pil_n + jext

      !---------------------------------------------------------------------

      return
end subroutine adv_get_ij0n_ext
