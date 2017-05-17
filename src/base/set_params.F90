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

!**   s/r set_params - initialize some constant parameters

      subroutine set_params
      use gem_options
      implicit none
#include <arch_specific.hf>

#include "dcst.cdk"
#include "cstv.cdk"
#include "ver.cdk"

      real*8, parameter :: zero=0.d0, one=1.d0
!
!     ---------------------------------------------------------------

      Cstv_tau_8   = Cstv_dt_8 * Cstv_bA_8
      Cstv_invT_8  = one/Cstv_tau_8
      Cstv_Beta_8  = (one-Cstv_bA_8)/Cstv_bA_8

      Cstv_tau_m_8   = Cstv_dt_8 * Cstv_bA_m_8
      Cstv_invT_m_8  = one/Cstv_tau_m_8
      Cstv_Beta_m_8  = (one-Cstv_bA_m_8)/Cstv_bA_m_8      

!     Parameters for the nonhydrostatic case
      Cstv_tau_nh_8   = Cstv_dt_8 * Cstv_bA_nh_8
      Cstv_invT_nh_8  = one/Cstv_tau_nh_8
      Cstv_Beta_nh_8  = (one-Cstv_bA_nh_8)/Cstv_bA_nh_8

      if (Schm_advec.eq.1) then ! traditional advection
         Cstv_dtA_8  = Cstv_dt_8 * 0.5d0
         Cstv_dtzA_8 = Cstv_dt_8 * 0.5d0 
      endif
      if (Schm_advec.eq.2) then ! consistant advection
         Cstv_dtA_8  = Cstv_tau_m_8
         Cstv_dtzA_8 = Cstv_tau_8
      endif
      Cstv_dtD_8  = Cstv_dt_8 - Cstv_dtA_8
      Cstv_dtzD_8 = Cstv_dt_8 - Cstv_dtzA_8

      if (Schm_advec.eq.0) then ! no advection
         Cstv_dtA_8  = 0.d0
         Cstv_dtD_8  = 0.d0
         Cstv_dtzA_8 = 0.d0
         Cstv_dtzD_8 = 0.d0
      endif

      Ver_igt_8    = Cstv_invT_8/Dcst_grav_8
      Ver_ikt_8    = Cstv_invT_m_8/Dcst_cappa_8
      if(Schm_hydro_L) Ver_igt_8=zero
      ! Modified epsilon
      Ver_igt2_8   = Cstv_rE_8*Ver_igt_8*(Cstv_invT_nh_8/Dcst_grav_8)
      Ver_igt_8    = Cstv_rE_8*Ver_igt_8 ! Modified epsilon
!
!     ---------------------------------------------------------------
!
      return
      end
