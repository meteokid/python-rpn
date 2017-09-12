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
module dynkernel_options
   implicit none
   public
   save

   !# Main selector for dynamical kernel
   !# * "DYNAMICS_FISL_P" : Fully implicit SL in pressure
   !# * "DYNAMICS_FISL_H" : Fully implicit SL in height
   !# * "DYNAMICS_EXPO_H" : Exponential integrators in height
   character(len=32) :: Dynamics_Kernel_S = 'DYNAMICS_FISL_P'
   namelist /dyn_kernel/ Dynamics_Kernel_S
contains

   !/@*
   function dynkernel_options_init() result(F_istat)
      implicit none
      !@object Additional initialisation steps before reading the nml
      integer :: F_istat
      !*@/
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      !----------------------------------------------------------------
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      !----------------------------------------------------------------
      return
   end function dynkernel_options_init

end module dynkernel_options
