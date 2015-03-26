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

!**s/r hzd_main - applies horizontal diffusion on a given set of fields
!
      subroutine hzd_main 
      implicit none
#include <arch_specific.hf>

!author
!     Joseph-Pierre Toviessi ( after version v1_03 of dif )
!
!revision
! v2_00 - Desgagne M.       - initial MPI version 
! v2_10 - Qaddouri&Desgagne - higher order diffusion operator
! v2_21 - Desgagne M.       - new call to horwavg
! v2_30 - Edouard  S.       - adapt for vertical hybrid coordinate
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_01 - Toviessi J. P.    - add call hzd_ho_parite
! v3_02 - Desgagne M.       - correction for non-hydrostatic version
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_20 - Tanguay M.        - Introduce Hzd_hzdmain_n_L
! v3_21 - Desgagne M.       - added explicit horiz diff.
! v4_xx - Gravel, S.        - adapt to vertical staggering
! v3_30 - Tanguay M.        - activate Hzd_type_S='HO_EXP' 
! v4_04 - Girard-Plante     - Diffuse only real winds, zdot and theta.
!                           - Move psadj code in new s/r psadj
! v4_05 - Plante A.         - Diffusion of w all the time
! v4_05 - Lepine M.         - VMM replacement with GMM
! v4_15 - Desgagne M.       - refonte majeure
! v4_40 - Plante A.         - Equatorial_sponge
! v4_50 - Desgagne M.       - New control switches

#include "gmm.hf"
#include "glb_ld.cdk"
#include "hzd.cdk"
#include "tr3d.cdk"
#include "vspng.cdk"
#include "schm.cdk"
#include "eq.cdk"
#include "vt1.cdk"

      logical switch_on_UVW, switch_on_TR, switch_on_vrtspng, &
              switch_on_eqspng, switch_on_THETA
      integer i,istat
      real, pointer, dimension(:,:,:) :: tr
!     _________________________________________________________________
!
      switch_on_UVW     = Hzd_lnr      .gt.0.
      switch_on_TR      = Hzd_lnr_tr   .gt.0.
      switch_on_THETA   = Hzd_lnr_theta.gt.0.
      switch_on_vrtspng = Vspng_nk     .ge.1
      switch_on_eqspng  = Eq_nlev      .gt.1

      istat = gmm_get(gmmk_ut1_s,ut1)
      istat = gmm_get(gmmk_vt1_s,vt1)
      istat = gmm_get(gmmk_zdt1_s,zdt1)
      istat = gmm_get(gmmk_tt1_s,tt1)
      istat = gmm_get(gmmk_wt1_s,wt1)

      call itf_ens_hzd ( ut1,vt1,tt1, l_minx,l_maxx,l_miny,l_maxy, G_nk )
!     
!**************************************
!  3. Horizontal diffusiion on theta  *
!**************************************
!
      if ( switch_on_THETA ) then
         call timing_start ( 61, 'HZD_theta' )
         call hzd_theta
         call timing_stop  ( 61 )
      endif

      if ( switch_on_TR ) then
         call timing_start ( 62, 'HZD_theta' )
         do i=1, Tr3d_ntr
            if (Tr3d_hzd(i)) then
               nullify (tr)
               istat = gmm_get('TR/'//trim(Tr3d_name_S(i))//':P',tr)
               if (istat.eq.0) call hzd_ctrl3 (tr, 'S_TR', G_nk)
            endif
         end do
         call timing_stop  ( 62 )
      endif
!
!***************************
!  1. Horizontal diffusion *
!***************************
!
      if ( switch_on_UVW ) then
         call timing_start ( 63, 'HORDIFF' )
         call hzd_ctrl3 ( ut1, 'U', G_nk)
         call hzd_ctrl3 ( vt1, 'V', G_nk)
         call hzd_ctrl3 (zdt1, 'S', G_nk)
         call hzd_ctrl3 ( wt1, 'S', G_nk)
         call timing_stop ( 63 )
      endif
!     
!***********************
!  2. Vertical sponge  *
!***********************
!
      if ( switch_on_vrtspng ) then
         call timing_start ( 65, 'V_SPNG' )
         call vspng_drv3 (ut1, vt1, zdt1, wt1, tt1, &
                          l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call timing_stop ( 65 )
      endif
!     
!*************************
!  3. Equatorial sponge  *
!*************************
!
      if ( switch_on_eqspng ) then
         call timing_start ( 67, 'EQUA_SPNG')
         call eqspng_drv (ut1,vt1,l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call timing_stop ( 67 )
      endif

      call itf_ens_hzd ( ut1,vt1,tt1, l_minx,l_maxx,l_miny,l_maxy, G_nk )

! Update pw_* variables

      if (switch_on_UVW   .or. switch_on_vrtspng .or. &
          switch_on_THETA .or. switch_on_TR ) &
         call pw_update_GPW
      if (switch_on_UVW .or. switch_on_vrtspng .or. switch_on_eqspng)&
         call pw_update_UV
      if (switch_on_THETA .or. switch_on_TR .or. switch_on_vrtspng  )&
         call pw_update_T
!
!     _________________________________________________________________
!
      return
      end
