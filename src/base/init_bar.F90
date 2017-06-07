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

!**s/r init_bar - prepare data for autobarotropic runs (Williamson cases)

      subroutine init_bar ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo,&
                            Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                            F_trprefix_S, F_trsuffix_S, F_datev )
      use dynkernel_options
      use gmm_geof
      use inp_mod
      use gmm_pw
      use gem_options
      implicit none

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_v (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_w (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_t (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_s (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q (Mminx:Mmaxx,Mminy:Mmaxy,Nk+1), &
           F_topo(Mminx:Mmaxx,Mminy:Mmaxy)

      !object
      !============================================================
      !     prepare data for autobarotropic runs (Williamson cases)
      !============================================================

#include "gmm.hf"
#include "glb_ld.cdk"
#include "tr3d.cdk"
#include "cstv.cdk"

      !---------------------------------------------------------------

      integer istat
      real, dimension (:,:,:), pointer :: hu
      real, dimension (Mminx:Mmaxx,Mminy:Mmaxy,Nk) :: gz_t

      !---------------------------------------------------------------

      if (Vtopo_L)      call handle_error (-1,'INIT_BAR','Vtopo_L not available YET')

      if (Schm_sleve_L) call handle_error (-1,'INIT_BAR','  SLEVE not available YET')

      !Setup Williamson Case 7: The 21 December 1978 Initial conditions are read
      !-------------------------------------------------------------------------
      call inp_data ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo,&
                      Mminx,Mmaxx,Mminy,Mmaxy, Nk, .true.       ,&
                      F_trprefix_S, F_trsuffix_S, F_datev )

      !Initialize T/ZD/W/Q
      !-------------------
      F_t = Cstv_Tstr_8 ; F_zd = 0. ; F_w = 0. ; F_q = 0.

      !Initialize HU
      !-------------
      istat = gmm_get ('TR/HU:P',hu)

      hu = 0.

      !Initialize d(Zeta)dot and dz/dt
      !-------------------------------
      Inp_zd_L = .TRUE.
      Inp_w_L  = .TRUE.

      !Prepare initial conditions (staggered u-v,gz,s,topo) for Williamson cases
      !-------------------------------------------------------------------------
      call wil_init (F_u,F_v,gz_t,F_s,F_topo,Mminx,Mmaxx,Mminy,Mmaxy,Nk)

      !Required for CASE5 LAM version
      !------------------------------
      istat = gmm_get (gmmk_topo_low_s , topo_low )
      istat = gmm_get (gmmk_topo_high_s, topo_high)

      topo_high(1:l_ni,1:l_nj) =    F_topo(1:l_ni,1:l_nj)
      topo_low (1:l_ni,1:l_nj) = topo_high(1:l_ni,1:l_nj)

      !Estimate U-V and T on scalar grids
      !----------------------------------
      istat = gmm_get (gmmk_pw_uu_plus_s, pw_uu_plus)
      istat = gmm_get (gmmk_pw_vv_plus_s, pw_vv_plus)
      istat = gmm_get (gmmk_pw_tt_plus_s, pw_tt_plus)

      call hwnd_stag ( pw_uu_plus,pw_vv_plus,F_u,F_v, &
                       Mminx,Mmaxx,Mminy,Mmaxy,Nk,.false. )

      pw_tt_plus = F_t

      if (trim(Dynamics_Kernel_S) == 'DYNAMICS_EXPO_H') then
         call exp_init_bar ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, gz_t, F_topo,&
                             Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                             F_trprefix_S, F_trsuffix_S, F_datev )
      end if

      return

      !---------------------------------------------------------------

      end
