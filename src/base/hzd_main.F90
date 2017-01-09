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

!**s/r hzd_main - Main controler for horizontal diffusion

      subroutine hzd_main
      use hzd_ctrl
      use hzd_exp
      implicit none
#include <arch_specific.hf>

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "hzd.cdk"
#include "tr3d.cdk"
#include "vspng.cdk"
#include "eq.cdk"
#include "vt1.cdk"
#include "schm.cdk"
#include "crg.cdk"

      logical switch_on_UVW, switch_on_TR, switch_on_vrtspng_UVT    , &
              switch_on_vrtspng_W, switch_on_eqspng, switch_on_THETA, &
              switch_on_smago, switch_on_smagoTH
      integer i,j,k,istat
      integer, save :: depth
      real*8 pis2,fract
      real, dimension(:)    , pointer, save :: weight=> null()
      real, dimension(:,:,:), pointer       :: tr
!
!-------------------------------------------------------------------
!
      if (Lun_debug_L) write (Lun_out,1000)

      call timing_start2 ( 60, 'HZD_main', 1 )
      switch_on_UVW         = Hzd_lnr       > 0.
      switch_on_TR          =(Hzd_lnr_tr    > 0.).and.any(Tr3d_hzd)
      switch_on_THETA       = Hzd_lnr_theta > 0.
      switch_on_vrtspng_UVT = Vspng_nk      >=1
      switch_on_vrtspng_W   = Vspng_nk      >=1
      switch_on_eqspng      = Eq_nlev       > 1
      switch_on_smago       = Hzd_smago_param > 0.
      switch_on_smagoTH     = switch_on_smago .and. &
                              (Hzd_smago_prandtl > 0.)

      if(hzd_in_rhs_L)      switch_on_THETA       =.false.
      if(hzd_in_rhs_L)      switch_on_UVW         =.false.
      if(top_spng_in_rhs_L) switch_on_vrtspng_UVT =.false.
      if(eqspng_in_rhs_L)   switch_on_eqspng      =.false.
      if(smago_in_rhs_L)    switch_on_smago       =.false.

      istat = gmm_get(gmmk_ut1_s,ut1)
      istat = gmm_get(gmmk_vt1_s,vt1)
      istat = gmm_get(gmmk_zdt1_s,zdt1)
      istat = gmm_get(gmmk_tt1_s,tt1)
      istat = gmm_get(gmmk_wt1_s,wt1)

      call itf_ens_hzd (ut1,vt1,tt1, l_minx,l_maxx,l_miny,l_maxy, G_nk)

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
         call timing_stop ( 64 )
      endif

!********************
!  Vertical sponge  *
!********************

      if ( switch_on_vrtspng_UVT ) then
         call timing_start2 ( 65, 'V_SPNG', 60 )
         call hzd_exp_deln ( ut1,  'U', l_minx,l_maxx,l_miny,l_maxy,&
                             Vspng_nk, F_VV=vt1, F_type_S='VSPNG' )
         call hzd_exp_deln ( tt1, 'M', l_minx,l_maxx,l_miny,l_maxy,&
                             Vspng_nk, F_type_S='VSPNG' )
         call timing_stop ( 65 )
      endif
      if ( switch_on_vrtspng_W ) then
         call timing_start2 ( 65, 'V_SPNG', 60 )
         if (Vspng_riley_L) then
            if (.not. associated(weight)) then
               pis2 = acos(0.0d0)
               depth= Vspng_nk+1
               allocate (weight(depth-1))
               weight(1)= 0.
               do k=2, depth-1
                  fract= (dble(k) - dble(depth))/ dble(depth-1)
                  weight(k)= cos(pis2*fract)**2
               end do
            endif
            do k=1, depth-1
               do j=1, l_nj
                  do i=1, l_ni
                     zdt1(i,j,k) = zdt1(i,j,k)*weight(k)
                      wt1(i,j,k) =  wt1(i,j,k)*weight(k)
                  end do
               end do
            end do
         else
            call hzd_exp_deln ( zdt1, 'M', l_minx,l_maxx,l_miny,l_maxy,&
                                Vspng_nk, F_type_S='VSPNG' )
            call hzd_exp_deln ( wt1, 'M', l_minx,l_maxx,l_miny,l_maxy,&
                                Vspng_nk, F_type_S='VSPNG' )
         endif
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

      call timing_start2 ( 69, 'PW_UPDATE', 60)
      if (switch_on_UVW   .or. switch_on_vrtspng_W .or.           &
          switch_on_THETA .or. switch_on_TR .or. switch_on_smago) &
         call pw_update_GPW
      if (switch_on_UVW .or. switch_on_vrtspng_UVT .or. &
          switch_on_eqspng .or. switch_on_smago)        &
         call pw_update_UV
      if (switch_on_THETA .or. switch_on_smagoTH .or. &
          switch_on_TR .or. switch_on_vrtspng_UVT  )  &
         call pw_update_T
      call timing_stop ( 69 )

      call timing_stop ( 60 )

 1000 format(3X,'MAIN HORIZONTAL DIFFUSION : (S/R HZD_MAIN)')
!
!-------------------------------------------------------------------
!
      return
      end
