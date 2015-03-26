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
subroutine adx_set_interp_pole()
   implicit none
#include <arch_specific.hf>
   !@objective set 1-D interpolation of grid reflexion across the pole 
   !@author  alain patoine
   !@revisions
   !*@/
#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_poles.cdk"
   integer :: istat, i, i2
   !---------------------------------------------------------------------
   allocate( &
        adx_iln(adx_gni), &
        adx_lnr_8(adx_gni), &
        stat = istat)
   call handle_error_l(istat==0,'adx_set_interp_pole','problem allocating mem')

   do i = 1,adx_gni
      if (adx_xg_8(i) < adx_xg_8(1) + CONST_PI_8) then
         adx_lnr_8(i) = adx_xg_8(i) + CONST_PI_8
      else
         adx_lnr_8(i) = adx_xg_8(i) - CONST_PI_8
      endif
      do i2 = 1,adx_gni     
         adx_iln(i) = i2
         if (adx_lnr_8(i) >= adx_xg_8(i2) .and.  &
              adx_lnr_8(i) < adx_xg_8(i2+1) ) exit
      enddo
   enddo
   !---------------------------------------------------------------------
   return
end subroutine adx_set_interp_pole
