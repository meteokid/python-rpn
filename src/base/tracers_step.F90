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

!**s/r tracers_step
!
      subroutine tracers_step (F_water_tracers_only_L)
   
use adv_tracers_interpol_mod  , only :adv_tracers_interp_lam

   implicit none
#include <arch_specific.hf>
#include "glb_ld.cdk"
      logical, intent(IN) :: F_water_tracers_only_L

!author
!     Michel Desgagne --- summer 2014
!
!revision
! v4_70 - Desgagne M.       - initial version 

!     _________________________________________________________________
!
      if (.not. F_water_tracers_only_L) &
           call stat_mass_tracers (1,"BEFORE ADVECTION")
     
       if (G_lam .and. .not. Advection_lam_legacy) then
     
        call adv_tracers_interp_lam (F_water_tracers_only_L)    
      
       else
        call adx_tracers_interp (F_water_tracers_only_L)    
       endif

      if (.not. F_water_tracers_only_L) &
           call stat_mass_tracers (0,"AFTER ADVECTION")

!     _________________________________________________________________
!
      return
      end
