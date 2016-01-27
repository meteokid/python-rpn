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

!**s/r gem_nml

      integer function gem_nml (F_namelistf_S)
      implicit none
#include <arch_specific.hf>

      character* (*) F_namelistf_S

!author
!     M. Desgagne    - Summer  2006
!
!revision
! v3_30 - Desgagne M.       - initial version
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_03 - Tanguay M.        - Williamson's cases
! v4_05 - Desgagne M.       - Add Lam_blendoro_L and Lam_cascsfc_L
! v4_05 - McTaggart-Cowan R.- Remove time series settings
! v4_05 - Girard C.         - Open top option
! v4_05 - Plante A.         - Vertical blending option
! v4_4  - Plante A.         - Equatorial sponge
!
!object
!  Default configuration and reading namelists gem_cfgs and grdc

#include "nml.cdk"

      integer, external :: fnom

      character*64 dumc_S
      integer i,k,nrec,err,unf
!
!-------------------------------------------------------------------
!
      gem_nml = -1

      if ((F_namelistf_S.eq.'print').or.(F_namelistf_S.eq.'PRINT')) then
         gem_nml = 0
         if ( Lun_out.ge.0) write (Lun_out,nml=gem_cfgs_p)
         if ( Lun_out.ge.0 .and. &
             Grdc_ndt.gt.0) write (lun_out,nml=grdc_p)
         return
      endif

      G_halox = 4
      G_haloy = G_halox

      Iau_cutoff    = 6.
      Iau_interval  = -1.
      Iau_period    = -1.
      Iau_tracers_S = ''
      Iau_weight_S  = 'constant'
      Iau_ninblocx  = 1
      Iau_ninblocy  = 1

      Init_balgm_L   = .false.
      Init_dfwin_L   = .true.
      Init_dflength_S= '5p'
      Init_dfpl_S    = '6h'
      Init_dftr_L    = .false.

      Schm_hydro_L  = .false.
      Schm_hzdadw_L = .false.
      Schm_topo_L   = .true.
      Schm_itcn     = 2
      Schm_itnlh    = 2
      Schm_itraj    = 2
      Schm_nblendyy = -1
      Schm_Tlift    = 0
      Schm_MTeul    = 0
      Schm_advec    = 2
      Schm_settls_L   = .false.
      Schm_capa_var_L = .false.
      Schm_cub_Coriolis_L = .false.
      Schm_superwinds_L  = .true.
      Schm_adcub_L    = .true.
      Schm_psadj_L    = .false.
      Schm_source_ps_L= .false.
      Schm_autobar_L  = .false.
      Schm_bitpattern_L = .false.
      Schm_wload_L     = .false.
      Schm_adxlegacy_L = .false.
      Schm_cub_traj_L  = .true.
      Schm_trapeze_L   = .true.
      Schm_lift_ltl_L  = .true.

      Lam_blend_H_func_S = 'COS2'
      Lam_blend_H   = 10
      Lam_blend_T   =  0
      Lam_gbpil_T   = -1
      Lam_hint_S    = 'CUB_LAG'
      Lam_ctebcs_L  = .false.
      Lam_toptt_L   = .false.
      Lam_0ptend_L  = .true.
      Lam_blendoro_L= .true.
      Lam_current_S = '20000101.000000'
      Lam_acidtest_L= .false.

      Spn_nudging_S = ' '
      Spn_start_lev = 1.0
      Spn_up_const_lev = 0.0
      Spn_trans_shape_S = 'LINEAR'
      Spn_relax_hours = 10.
      Spn_cutoff_scale_small=100.
      Spn_cutoff_scale_large=300.
      Spn_step=21600
      Spn_weight_L = .false.
      Spn_wt_pwr=2

      Zblen_L   = .false.

      Cstv_dt_8    = 900
      Cstv_bA_8    = 0.6
      Cstv_bA_nh_8 = 0.5
      Cstv_rE_8    = 1.d0
      Cstv_tstr_8  = 240.0

      Lctl_rxstat_S   = 'LCL_4'
      Lctl_debug_L    = .false.

      Grd_rcoef(1) = 1.0
      Grd_rcoef(2) = 1.0

      hyb = -1.

      sol_fft_L     = .true.
      sol_type_S    = 'DIRECT'
      Sol3D_krylov_S = 'FGMRES'
      sol2D_precond_S = 'JACOBI'
      sol3D_precond_S = 'JACOBI'
      sol_im        = 15
      sol_fgm_maxits= 200
      sol_fgm_eps   = 1.d-07
      sol_yyg_maxits= 40
      sol_yyg_eps   = 1.d-04

      Hzd_difva_L    = .false.
      Hzd_prof_S     = "NIL"
      Hzd_pwr        = -1
      Hzd_lnr        = -1.
      Hzd_pwr_theta  = -1
      Hzd_lnr_theta  = -1.
      Hzd_pwr_tr     = -1
      Hzd_lnr_tr     = -1.
      Hzd_div_damp   = -1.
      Hzd_type_S     = 'HO_EXP9P'
      Hzd_xidot_L    = .false.
      Hzd_smago_L    = .false.
      Hzd_smago_param= -1
      Hzd_theta_njpole_gu_only= -1

      Vspng_nk       = 0
      Vspng_coeftop  = -1.
      Vspng_njpole   = 3
      Vspng_zmean_L  = .false.

      Zdot_divHLM_L  = .false.

      Vtopo_start = -1
      Vtopo_ndt   = 0

      Tr3d_list_S   = ''
      Tr3d_ntr      = 0
      Tr3d_anydate_L= .false.

! The default here is NO modulation (weigh is 1.0 everywhere)
! Activation can be done with P_lmvd_weigh_high_lat=0.
      P_lmvd_weigh_high_lat =  1.0
      P_lmvd_weigh_low_lat  =  1.0
      P_lmvd_high_lat       = 30.0
      P_lmvd_low_lat        =  5.0

      perturb_nbits=0
      perturb_npts=10

      Eq_sponge=0.

      Inp_blacklist_S = ''
      Inp_npes  = 1
      Out3_npes = 1

      Out3_lieb_levels = 0.
      Out3_lieb_conv   = 0.1
      Out3_lieb_maxite = 100

      Out3_etik_S    = 'GEMDM'
      Out3_close_interval_S= "" !# Out3_closestep_S= ""
      Out3_postproc_fact = 0    !# Out3_postfreq_S= ''
      Out3_ndigits   = 3
      Out3_ip3       = 0
      Out3_nbitg     = 16
      Out3_linbot    = 0
      Out3_cliph_L   = .false.
      Out3_flipit_L  = .false.
!
!     Williamson's cases section
!     --------------------------
      Williamson_case     = 0
      Williamson_alpha    = 0.0

      Grdc_xlat1  = Grd_xlat1
      Grdc_xlon1  = Grd_xlon1
      Grdc_xlat2  = Grd_xlat2
      Grdc_xlon2  = Grd_xlon2
      Grdc_ni     = 0
      Grdc_nj     = 0
      grdc_dx     = -1.
      grdc_dy     = -1.
      Grdc_Hblen  = 10
      Grdc_maxcfl = 1
      Grdc_nfe    = ''
      Grdc_start_S= ''
      Grdc_end_S  = ''
      Grdc_initphy_L = .false.
      Grdc_nbits  = 32
      Grdc_trnm_S = '@#$%'

      stat_liste = ''

      if (F_namelistf_S .ne. '') then

         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', nrec) .ne. 0) goto 9110
         rewind(unf)
         read (unf, nml=gem_cfgs, end = 9120, err=9120)
         rewind(unf)

         rewind(unf)
         read  (unf, nml=williamson, end = 1001, err = 9125)
 1001    continue
         rewind(unf)

         read (unf, nml=grdc,     end = 1000, err=9130)
 1000    call fclos (unf)

      endif

      call low2up (Lctl_rxstat_S ,dumc_S)
      Lctl_rxstat_S = dumc_S

      gem_nml = 1
      goto 9999

 9110 if (Lun_out.gt.0) then
         write (Lun_out, 9050) trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9120 call fclos (unf)
      if (Lun_out.ge.0) then
         write (Lun_out, 9150) 'gem_cfgs',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9125 if (Lun_out.gt.0) then
          write (Lun_out, 9525)
          write (Lun_out, 8000)
      endif
      goto 9999

 9130 call fclos (unf)
      if (Lun_out.ge.0) then
         write (Lun_out, 9150) 'grdc',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif

 8000 format (/,'========= ABORT IN S/R gem_nml.f ============='/)
 9050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 9150 format (/,' NAMELIST ',A,' INVALID IN FILE: ',A/)
 9525 format (/,' NAMELIST williamson INVALID IN FILE: model_settings '/)
!
!-------------------------------------------------------------------
!
 9999 return
      end
