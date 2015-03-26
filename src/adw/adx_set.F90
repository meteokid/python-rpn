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
#include "constants.h"
#include "msg.h"

!/@*
subroutine adx_set()
   implicit none
#include <arch_specific.hf>
   !@objective Sets different advection parameters
   !@arguments
   !
   !@author alain patoine
   !@revisions
   ! v3_20 - Gravel & Valin & Tanguay - Lagrange 3D and optimized SETINT/TRILIN
   ! v4_10 - Plante A.                - Add support to thermo upstream pos
!*@/
#include "adx_dims.cdk"
   !---------------------------------------------------------------------
   call msg(MSG_DEBUG,'adx_set')

   call adx_get_params()
   call adx_set_grid()

   call adx_set_interp()

   if (.not.adx_lam_L) then
      call adx_set_interp_pole()
   endif

   call msg(MSG_DEBUG,'adx_set: end')
   !---------------------------------------------------------------------
   return
end subroutine adx_set

