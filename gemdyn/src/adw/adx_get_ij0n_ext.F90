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
!/@*
!**s/p adx_get_ij0n_ext: Establish scope of extended advection operations if LAM (Based on adx_get_ij0n)

subroutine adx_get_ij0n_ext(i0,in,j0,jn)

   implicit none

#include <arch_specific.hf>

   integer :: i0,j0,in,jn

   !@author Monique Tanguay

   !@revisions
   !v4_80 - Tanguay M.        - GEM4 Mass-Conservation and FLUX calculations

#include "adx_dims.cdk"
   !*@/
   !---------------------------------------------------------------------
   integer :: jext

   i0 = 1
   in = adx_mlni
   j0 = 1
   jn = adx_mlnj

   if (adx_lam_L) then
     !jext=1
      jext=adx_maxcfl
      if (adx_yinyang_L) jext=2
      if (adx_is_west)  i0 =            adx_pil_w - jext
      if (adx_is_east)  in = adx_mlni - adx_pil_e + jext
      if (adx_is_south) j0 =            adx_pil_s - jext
      if (adx_is_north) jn = adx_mlnj - adx_pil_n + jext
   endif

   !---------------------------------------------------------------------
   return
end subroutine adx_get_ij0n_ext
