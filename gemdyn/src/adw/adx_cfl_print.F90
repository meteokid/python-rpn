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
subroutine adx_cfl_print()
   implicit none
#include <arch_specific.hf>
   !@objective Print precomputed CFL and reset stats
   !@author  Stephane Chamberland, 2010-01
   !@revisions
#include "adx_cfl.cdk"
   !*@/
   real :: cfl
   !---------------------------------------------------------------------
   cfl = sngl(adx_cfl_8(1))
   write (6,99) 'x,y',adx_cfl_i(1,1),adx_cfl_i(2,1), adx_cfl_i(3,1),cfl
   cfl = sngl(adx_cfl_8(2))
   write (6,99) 'z'  ,adx_cfl_i(1,2),adx_cfl_i(2,2), adx_cfl_i(3,2),cfl
   cfl = sngl(adx_cfl_8(3))
   write (6,99) '3D' ,adx_cfl_i(1,3),adx_cfl_i(2,3), adx_cfl_i(3,3),cfl
   adx_cfl_8 (:  ) = 0.d0
   adx_cfl_i (:,:) = 0

99 format(' MAX COURANT NUMBER:  ', a3,': [(',i4,',',i4,',',i4,') ',f12.5,']')

   !---------------------------------------------------------------------
   return
end subroutine adx_cfl_print
