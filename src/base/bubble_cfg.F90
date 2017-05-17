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

!**s/r bubble_cfg
!
      integer function bubble_cfg (unf)
      use step_options
      use gem_options
      use grid_options
      implicit none
#include <arch_specific.hf>
      integer unf
!!$#include "cstv.cdk" !#This is already included in theonml.cdk
!!$#include "out3.cdk" !#This is already included in theonml.cdk
#include "theonml.cdk"

      integer i, j, k, idatx, longueur
      real*8 delta_8, zmt_8(maxhlev)
      real*8 c1_8,Exner_8,height_8,pres_8,p1000hPa_8
      real aa
!
!     ---------------------------------------------------------------
      Dcst_rayt_8=Dcst_rayt_8*0.1d0 ! an accuracy problem
!
      stat_liste(1)='URT1'
      stat_liste(2)='VRT1'
      stat_liste(3)='WT1'
      stat_liste(4)='ZDT1'
      stat_liste(5)='TT1'
      stat_liste(6)='ST1'
      stat_liste(7)='QT1'

      p1000hPa_8=100000.d0
      bubble_cfg = -1
      Step_gstat = 1
      hyb = -1.

      Grd_typ_S='GU'
      Grd_rot_8 = 0.0
      Grd_rot_8(1,1) = 1.
      Grd_rot_8(2,2) = 1.
      Grd_rot_8(3,3) = 1.
      Grd_roule=.false.
      Grd_bsc_base = 5
      Grd_bsc_ext1 = 0
      Grd_maxcfl   = 3
      Grd_bsc_adw  = Grd_maxcfl  + Grd_bsc_base
      Grd_extension= Grd_bsc_adw + Grd_bsc_ext1
!
      Grd_xlat1=0.
      Grd_xlon1=180.
      Grd_xlat2=0.
      Grd_xlon2=270.

      Cstv_bA_8 = 0.5
      Cstv_bA_m_8 = 0.5
      Cstv_bA_nh_8 = 0.5
      Cstv_tstr_8 = 303.16

      Cstv_dt_8 = 5.0
      Step_total=120

      Hyb_rcoef = 1.0

      Lam_ctebcs_L=.true.

      hdif_lnr = 0.
      hdif_pwr = 6

      Schm_trapeze_L=.true.
      Schm_cub_traj_L=.true.
      Schm_phyms_L = .false.
      Schm_psadj = 0

      G_halox=3

      Zblen_L        = .false.
      Zblen_spngtt_L = .false.
      Zblen_spngthick=0.

      Out3_close_interval_S = '999h'
      Out3_close_interval = 999
      Out3_unit_S = 'HOU'


      Out3_etik_s  = 'BUBBLE'
      Schm_hydro_L = .false.

      Grd_ni = 101
      Grd_nj = 1
      G_nk   = 99

      Grd_dx = 10.  ! meters

      bubble_domain_top = 1000.  ! meters
      bubble_theta= 303.16  ! 30 Celsius
      bubble_rad  = 25  ! number of grid points
      bubble_ictr = (Grd_ni-1)/2 + Grd_extension
      bubble_kctr = G_nk-bubble_rad-1

      Lam_blend_H = 0

      Hgc_gxtyp_s='E'
      call cxgaig ( Hgc_gxtyp_S,Hgc_ig1ro,Hgc_ig2ro,Hgc_ig3ro,Hgc_ig4ro, &
                              Grd_xlat1,Grd_xlon1,Grd_xlat2,Grd_xlon2 )

      rewind (unf)
      read ( unf, nml=bubble_cfgs, end = 9220, err=9000)
      go to 9221
 9220 if (Ptopo_myproc.eq.0) write( Lun_out,9230) Theo_case_S
 9221 continue

      Grd_dx = (Grd_dx/Dcst_rayt_8)*(180./Dcst_pi_8)  ! in degrees

!   adjust dimensions to include piloted area (pil_n, s, w, e)

      Glb_pil_n = Grd_extension
      Glb_pil_s = Glb_pil_n ; Glb_pil_w=Glb_pil_n ; Glb_pil_e=Glb_pil_n

      pil_w= 0 ; pil_n= 0 ; pil_e= 0 ; pil_s= 0
      if (l_west ) pil_w= Glb_pil_w
      if (l_north) pil_n= Glb_pil_n
      if (l_east ) pil_e= Glb_pil_e
      if (l_south) pil_s= Glb_pil_s
      Lam_pil_w= Glb_pil_w
      Lam_pil_n= Glb_pil_n
      Lam_pil_e= Glb_pil_e
      Lam_pil_s= Glb_pil_s
!
      Grd_ni   = Grd_ni + 2*Grd_extension
      Grd_nj   = Grd_nj + 2*Grd_extension
      Grd_jref = (Grd_nj+1 )/2
      Zblen_hmin = bubble_domain_top

      if(hyb(1).lt.0) then
        !isentropic case
         c1_8=Dcst_grav_8/(Dcst_cpd_8*bubble_theta)
         Exner_8=1.d0-c1_8*bubble_domain_top
         Cstv_ptop_8 = Exner_8**(1.d0/Dcst_cappa_8)*p1000hPa_8
!        Uniform distribution of levels in terms of height
         do k=1,G_nk
            height_8=bubble_domain_top*(1.d0-(dble(k)-.5d0)/G_nk)
            Exner_8=1.d0-c1_8*height_8
            pres_8=Exner_8**(1.d0/Dcst_cappa_8)*p1000hPa_8
            hyb(k)=(pres_8-Cstv_ptop_8)/(p1000hPa_8-Cstv_ptop_8)
            print*,'hyb(k)=',hyb(k)
         enddo

!        denormalize
         do k=1,G_nk
            hyb(k) = hyb(k) + (1.-hyb(k))*Cstv_ptop_8/p1000hPa_8
         enddo
      else
         do k=1024,1,-1
         if(hyb(k).lt.0) G_nk=k-1
         enddo
      endif

      Grd_dy = Grd_dx
      Grd_x0_8=0.
      Grd_xl_8=Grd_x0_8 + (Grd_ni -1) * Grd_dx
      Grd_y0_8= - (Grd_jref-1) * Grd_dy
      Grd_yl_8=Grd_y0_8 + (Grd_nj -1) * Grd_dy
      if ( (Grd_x0_8.lt.  0.).or.(Grd_y0_8.lt.-90.).or. &
           (Grd_xl_8.gt.360.).or.(Grd_yl_8.gt. 90.) ) then
          if (Ptopo_myproc.eq.0) write (Lun_out,9600) &
              Grd_x0_8,Grd_y0_8,Grd_xl_8,Grd_yl_8
          return
       endif
      call datp2f( idatx, Step_runstrt_S)
      Out3_date= idatx

      G_halox = min(G_halox,Grd_ni-1)
      G_haloy = G_halox

      bubble_cfg = 1

      step_dt=Cstv_dt_8

      Fcst_rstrt_S = 'step,9999999'
      Fcst_bkup_S  = 'step,9999999'
      return

 9000 write (Lun_out, 9100)
!     ---------------------------------------------------------------
 9100 format (/,' NAMELIST mtn_cfgs INVALID FROM FILE: model_settings'/)
 9230 format (/,' Default setup will be used for :',a/)
 9500 format (/1x,'From subroutine mtn_cfg:', &
              /1x,'wrong value for model top')
 9600 format (/1x,'From subroutine mtn_cfg:', &
              /1x,'wrong lam grid configuration  ')
      end

