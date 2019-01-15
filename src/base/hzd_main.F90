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
      use hzd_ctrl
      implicit none
#include <arch_specific.hf>

!author
!     Joseph-Pierre Toviessi ( after version v1_03 of dif )
!
!revision
! v4_40 - Plante A.         - Equatorial_sponge
! v4_50 - Desgagne M.       - New control switches
! v4_80 - Lee&Qaddouri      - Add DEL N horizontal diffusion
! v4_80 - Gaudreault S.     - Nonlinear Smagorinsky diffusion

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "hzd.cdk"
#include "tr3d.cdk"
#include "vspng.cdk"
#include "eq.cdk"
#include "vt1.cdk"
#include "schm.cdk"

      logical switch_on_UVW, switch_on_TR, switch_on_vrtspng    , &
              switch_on_eqspng, switch_on_THETA, switch_on_smago, &
              switch_on_smagoTH
      integer i,istat
      real, pointer, dimension(:,:,:) :: tr
!
!-------------------------------------------------------------------
!
      if (Lun_debug_L) write (Lun_out,1000)

      call timing_start2 ( 60, 'HZD_main', 1 )
      switch_on_UVW     = Hzd_lnr       > 0.
      switch_on_TR      =(Hzd_lnr_tr    > 0.).and.any(Tr3d_hzd)
      switch_on_THETA   = Hzd_lnr_theta > 0.
      switch_on_vrtspng = Vspng_nk      >=1
      switch_on_eqspng  = Eq_nlev       > 1
      switch_on_smago   = Hzd_smago_param > 0.
      switch_on_smagoTH = switch_on_smago.and.(Hzd_smago_prandtl > 0.)

      istat = gmm_get(gmmk_ut1_s,ut1)
      istat = gmm_get(gmmk_vt1_s,vt1)
      istat = gmm_get(gmmk_zdt1_s,zdt1)
      istat = gmm_get(gmmk_tt1_s,tt1)
      istat = gmm_get(gmmk_wt1_s,wt1)

      call itf_ens_hzd ( ut1,vt1,tt1, l_minx,l_maxx,l_miny,l_maxy, G_nk )

!**********************************
!  Horizontal diffusion on theta  *
!**********************************

      if ( switch_on_THETA ) then
         call timing_start2 ( 61, 'HZD_theta', 60 )
         call hzd_theta
         call timing_stop  ( 61 )
      endif

!**********************************
!  Horizontal diffusion on tracers*
!**********************************

      if ( switch_on_TR ) then
         call timing_start2 ( 62, 'HZD_tracers', 60 )
         do i=1, Tr3d_ntr
            if (Tr3d_hzd(i)) then
               nullify (tr)
               istat = gmm_get('TR/'//trim(Tr3d_name_S(i))//':P',tr)
               if (istat.eq.0) call hzd_ctrl4 &
                     (tr, 'S_TR', l_minx,l_maxx,l_miny,l_maxy,G_nk)
            endif
         end do
         call timing_stop  ( 62 )
      endif

!************************
! Smagorinsky diffusion *
!************************

      if (switch_on_smago) then
         call timing_start2 ( 63, 'HZD_smago', 60 )
         call hzd_smago ( ut1, vt1, zdt1, tt1, &
                          l_minx, l_maxx, l_miny, l_maxy, G_nk)
         call timing_stop ( 63 )
      end if

!************************
!  Horizontal diffusion *
!************************

      if ( switch_on_UVW ) then
         call timing_start2 ( 64, 'HZD_bkgrnd', 60 )
         call hzd_ctrl4 ( ut1, vt1, l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call hzd_ctrl4 (zdt1, 'S', l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call hzd_ctrl4 ( wt1, 'S', l_minx,l_maxx,l_miny,l_maxy,G_nk)
         if (Schm_nologT_L.and.Hzd_xidot_L) then
            istat = gmm_get(gmmk_xdt1_s,xdt1)
            call hzd_ctrl4 ( xdt1, 'S', l_minx,l_maxx,l_miny,l_maxy,G_nk)
         endif
         call timing_stop ( 64 )
      endif
!
!********************
!  Vertical sponge  *
!********************
!
      if ( switch_on_vrtspng ) then
         call timing_start2 ( 65, 'V_SPNG', 60 )
         call vspng_drv3 (ut1, vt1, zdt1, wt1, tt1, &
                          l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call timing_stop ( 65 )
      endif

!**********************
!  Equatorial sponge  *
!**********************

      if ( switch_on_eqspng ) then
         call timing_start2 ( 67, 'EQUA_SPNG', 60)
         call eqspng_drv (ut1,vt1,l_minx,l_maxx,l_miny,l_maxy,G_nk)
         call timing_stop ( 67 )
      endif

      call itf_ens_hzd ( ut1,vt1,tt1, l_minx,l_maxx,l_miny,l_maxy, G_nk )

! Update pw_* variables

      call timing_start2 ( 69, 'PW_UPDATE_HZD', 60)
      if (switch_on_UVW   .or. switch_on_vrtspng .or.             &
          switch_on_THETA .or. switch_on_TR .or. switch_on_smago) &
         call pw_update_GPW
      if (switch_on_UVW .or. switch_on_vrtspng .or. &
          switch_on_eqspng .or. switch_on_smago)    &
         call pw_update_UV
      if (switch_on_THETA .or. switch_on_smagoTH .or. &
          switch_on_TR .or. switch_on_vrtspng  )      &
         call pw_update_T
      call timing_stop ( 69 )

      call timing_stop ( 60 )

1000  format(3X,'MAIN HORIZONTAL DIFFUSION : (S/R HZD_MAIN)')
!
!-------------------------------------------------------------------
!
      return
      end
