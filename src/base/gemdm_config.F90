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

!**s/r gemdm_config - Establish final model configuration
!
#include <msg.h>

      integer function gemdm_config ( )
      use timestr_mod, only: timestr_parse,timestr2step,timestr2sec
      use mu_jdate_mod, only: mu_set_leap_year, MU_JDATE_LEAP_IGNORED
      implicit none
#include <arch_specific.hf>

!author
!     M. Desgagne    - Summer 2006
!
!revision
! v3_30 - Desgagne M.       - initial version
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_03 - Lee/Desgagne      - ISST, add Grdc_maxcfl, Step_maxcfl, elim Grdc_pil
! v4_04 - Plante A.         - Remove offline mode
! v4_05 - Desgagne M.       - Throw warning and return for digital filter in LAM
! v4_05 - McTaggart-Cowan R.- Set Lun_sortie_s and Schm_opentop_L
! v4_10 - Tanguay M.        - Adjust digital filter when LAM

#include <rmnlib_basics.hf>
#include <WhiteBoard.hf>
#include "dcst.cdk"
#include "nml.cdk"
#include "step.cdk"
#include "out.cdk"
#include "tracers.cdk"

      character*16  dumc_S, datev
      character*256 fln_S
      logical wronghyb
      integer i, k, nrec, ipcode, ipkind, istat, err
      real    pcode,deg_2_rad,sec
      real*8  dayfrac, sec_in_day
      parameter (sec_in_day=86400.0d0)
!
!-------------------------------------------------------------------
!
      gemdm_config = -1

      if (Step_runstrt_S == 'NIL') then
         if (lun_out>0) then
            write (Lun_out, 6005)
            write (Lun_out, 8000)
         endif
         return
      endif

      if (.not.Step_leapyears_L) then
         call Ignore_LeapYear ()
         call mu_set_leap_year (MU_JDATE_LEAP_IGNORED)
         if (Lun_out>0) write(Lun_out,6010)
      endif

      dayfrac= - (Step_initial * Step_dt / sec_in_day)
      call incdatsd (datev, Step_runstrt_S, dayfrac)
      call datp2f ( Out3_date, datev )
      if (lun_out>0) write (Lun_out,6007) Step_runstrt_S, datev
      call datp2f ( Step_CMCdate0, Step_runstrt_S)

      Lun_debug_L = (Lctl_debug_L.and.Ptopo_myproc.eq.0)

!     Use newcode style:
      call convip ( ipcode, pcode, ipkind, 0, ' ', .false. )

      Level_kind_ip1 = 5
      Level_version  = 5

      if(Schm_Tlift.eq.1)then
         call gem_error ( -1, 'gemdm_config', &
                          'review for toplev removed and Tlift' )
         Level_version  = 3
      endif

      if (G_lam) then
         Glb_pil_n = Grd_extension
         Glb_pil_s=Glb_pil_n ; Glb_pil_w=Glb_pil_n ; Glb_pil_e=Glb_pil_n
         if (Grd_yinyang_L) then
            Lam_blend_H  = 0
            Lam_ctebcs_L = .true.
            Lam_blendoro_L = .false.
         else
            Lam_blend_H = max(0,Lam_blend_H)
         endif
         Lam_blend_Hx = Lam_blend_H ; Lam_blend_Hy = Lam_blend_H
         if (Schm_theoc_L) Lam_blend_Hy = 0
         Lam_tdeb = 999999.0
      endif
      if ((G_lam).and.(.not.Grd_yinyang_L)) then
         Lam_gbpil_T = max(0,Lam_gbpil_T)
         Lam_blend_T = max(0,Lam_blend_T)
      else
         Lam_gbpil_T = 0
         Lam_blend_T = 0
      endif

      !  Forcing some options under Schm_adxlegacy_L=.t.
      if (Schm_adxlegacy_L) then
         Schm_cub_traj_L = .false.
         Schm_trapeze_L  = .false.
         Schm_lift_ltl_L = .false.
      else
         ! Schm_adxlegacy_L=.t. is mandatory for GU grids
         if (.not. G_lam) then
            if (lun_out>0) write (Lun_out, 9203)
            return
         endif
         Schm_lift_ltl_L = .false.
         if ( (Schm_trapeze_L .or. Schm_step_settls_L) .and. .not.Schm_autobar_L ) Schm_lift_ltl_L = .true.
      endif

      deg_2_rad = Dcst_pi_8/180.

      P_lmvd_high_lat = min(90.,abs(P_lmvd_high_lat))
      P_lmvd_low_lat  = min(P_lmvd_high_lat,abs(P_lmvd_low_lat))
      P_lmvd_weigh_low_lat  = max(0.,min(1.,P_lmvd_weigh_low_lat ))
      P_lmvd_weigh_high_lat = max(0.,min(1.,P_lmvd_weigh_high_lat))

      Out_rewrit_L   = .false.

      if (Out3_close_interval_S == " ") Out3_close_interval_S= '1H'
      err = timestr_parse(Out3_close_interval,Out3_unit_S,Out3_close_interval_S)
      if (Out3_close_interval <= 0.) Out3_close_interval_S= '1H'
      err = timestr_parse(Out3_close_interval,Out3_unit_S,Out3_close_interval_S)
      if (.not.RMN_IS_OK(err)) then
         if (lun_out>0) write (Lun_out, *) "(gemdm_config) Invalid Out3_close_interval_S="//trim(Out3_close_interval_S)
         return
      endif
      Out3_postproc_fact = max(0,Out3_postproc_fact)

      if(Out3_nbitg .lt. 0) then
         if (lun_out>0) write (Lun_out, 9154)
         Out3_nbitg=16
      endif
      Out3_lieb_nk = 0
      do i = 1, MAXELEM
         if(Out3_lieb_levels(i) .le. 0. ) goto 81
         Out3_lieb_nk = Out3_lieb_nk + 1
      enddo
 81   continue

      Cstv_dt_8     = dble(Step_dt)

!     Counting # of vertical levels specified by user
      G_nk = 0
      do k = 1, maxhlev
         if (hyb(k) .lt. 0.) exit
         G_nk = k
      enddo
      if (lun_out>0) &
      write(lun_out,'(2x,"hyb="/5(f12.9,","))') hyb(1:g_nk)

      if (Schm_autobar_L) Schm_phyms_L= .false.

      call set_zeta2( hyb, G_nk )

      Schm_nith = G_nk

      err= 0
      err= min( timestr2step (Grdc_ndt,   Grdc_nfe    , Step_dt), err)
      err= min( timestr2step (Grdc_start, Grdc_start_S, Step_dt), err)
      err= min( timestr2step (Grdc_end,   Grdc_end_S  , Step_dt), err)
      if (err.lt.0) return

      if (Grdc_ndt   > -1) Grdc_ndt  = max( 1, Grdc_ndt)
      if (Grdc_start <  0) Grdc_start= Lctl_step
      if (Grdc_end   <  0) Grdc_end  = Step_total

      Grdc_maxcfl = max(1,Grdc_maxcfl)
      Grdc_pil    = Grdc_maxcfl + Grd_bsc_base + Grd_bsc_ext1
      Grdc_ni     = Grdc_ni   + 2*Grdc_pil
      Grdc_nj     = Grdc_nj   + 2*Grdc_pil
      Grdc_iref   = Grdc_iref +   Grdc_pil
      Grdc_jref   = Grdc_jref +   Grdc_pil

      call low2up  (Lam_hint_S ,dumc_S)
      Lam_hint_S= dumc_S

      call low2up  (sol_type_S ,dumc_S)
      sol_type_S = trim(dumc_S)
      call low2up  (sol2D_precond_S ,dumc_S)
      sol2D_precond_S = trim(dumc_S)
      call low2up  (sol3D_precond_S ,dumc_S)
      sol3D_precond_S = trim(dumc_S)

      if ( (Sol_type_S /= 'ITERATIVE_2D') .and. &
           (Sol_type_S /= 'ITERATIVE_3D') .and. &
           (Sol_type_S /= 'DIRECT'   ) ) then
         if (lun_out>0) write (Lun_out, 9200) Sol_type_S
         return
      endif

      if (Sol_type_S(1:9) == 'ITERATIVE') then
         if (Sol2D_precond_S /= 'JACOBI') then
            if (lun_out>0) write (Lun_out, 9201) Sol2D_precond_S
            return
         endif
      endif

      if (Sol_type_S == 'ITERATIVE_3D') then
         if (Sol3D_krylov_S /= 'FGMRES' .and. Sol3D_krylov_S /= 'FBICGSTAB') then
            if (Lun_out > 0) then
               write(Lun_out, *) 'ABORT: WRONG CHOICE OF KRYLOV METHOD FOR 3D ITERATIVE SOLVER: Sol3D_krylov_S =', Sol3D_krylov_S
            end if
            return
         end if

         if (Sol3D_precond_S /= 'JACOBI') then
            if (lun_out>0) then
               write (Lun_out, *) 'ABORT: WRONG CHOICE OF PRE-CONDITIONNER FOR 3D ITERATIVE SOLVER: Sol3D_precond_S =', Sol3D_precond_S
            end if
            return
         endif
      end if

      if ((Hzd_smago_param > 0.) .and. (.not. G_lam)) then
         if (lun_out>0) then
            write (Lun_out, *) 'ABORT: Smagorinksy-type horizontal diffusion only available for LAM/YY'
         end if
         return
      end if

      if ((Hzd_smago_prandtl > 0.) .and. (Hzd_smago_param <= 0.)) then
         if (lun_out>0) then
            write (Lun_out, *) 'ABORT: Hzd_smago_param must be set to a positive value', Hzd_smago_param
         end if
         return
      end if

      G_ni  = Grd_ni
      G_nj  = Grd_nj

      G_niu = G_ni
      G_njv = G_nj - 1

      if (G_lam) then
         G_niu= G_ni - 1
! Additional temporary check for Schm_psadj_L in LAM config. 
         if ( .not.(Grd_yinyang_L) .and. Schm_psadj_L .and. &
              .not.(Schm_psadj_lam_L) )then
            if (lun_out>0) write (Lun_out, 6700) 
            return
         endif
      endif

!     Check for open top (piloting and blending)
      Schm_opentop_L  = .false.
      if (Lam_gbpil_T > 0) then
         Schm_opentop_L = .true.
         if(lun_out>0.and.Vspng_nk/=0)write (Lun_out, 9570)
         Vspng_nk       = 0
      endif
      if (Lam_gbpil_T <= 0 .and. Lam_blend_T > 0) then
         if (lun_out>0)write (Lun_out, 9580)
         return
      endif


!     Check for modified epsilon/ super epsilon
      if ( Cstv_rE_8 /= 1.0d0 ) then
         if (Cstv_Tstr_8 .lt. 0. ) then
             if (lun_out>0) write (Lun_out, 9680) 'Cstv_rE_8'
            return
         endif

         if (Schm_opentop_L) then
            if (lun_out>0) write (Lun_out, 9681) 'Cstv_rE_8'
            return
         endif
      endif

      if ( Cstv_bA_nh_8 .lt. Cstv_bA_8) then
         Cstv_bA_nh_8=Cstv_bA_8
      endif


!     Check for incompatible use of IAU and DF
      if (Iau_period > 0. .and. Init_balgm_L) then
         if (Lun_out>0) then
            write (Lun_out, 7040)
            write (Lun_out, 8000)
         endif
         return
      endif

      !#TODO: check for Iau_ninblocx, Iau_ninblocy compat with topology

!     Some common setups for AUTOBAROTROPIC runs
      if (Schm_autobar_L) then
         if (lun_out>0) write (Lun_out, 6100)
         Schm_hydro_L   = .true.
         if(Williamson_case/=7) Schm_topo_L = .false.
         if (Williamson_case.eq.3.or.Williamson_case.eq.4) then
             if (lun_out>0) write (Lun_out, 6300)
             return
         endif
         if (Williamson_case>2.and.Williamson_alpha/=0.) then
            if (lun_out>0) write (Lun_out, 6400)
            return
         endif
         if (Williamson_case.eq.2.and. Grd_yinyang_L .and. &
             Williamson_alpha/=0.0 .and. Williamson_alpha/=90.0 ) then
            if (lun_out>0) write (Lun_out, 6500)
            return
         endif
         Williamson_alpha=Williamson_alpha*(Dcst_pi_8/180.0)
      endif
      Tr_2D_3D_L = .FALSE.
      if (Schm_autobar_L.and.Williamson_case==1) then
          Tr_2D_3D_L = .TRUE.
          if (lun_out>0) write (Lun_out, 7021)
      endif

      err= 0
      err= min( timestr2step (Init_dfnp, Init_dflength_S, Step_dt), err)
      err= min( timestr2sec  (sec,       Init_dfpl_S,     Step_dt), err)
      if (err.lt.0) return

      Init_halfspan = -9999

      if (Init_dfnp <= 0) then
         if (Init_balgm_L .and. Lun_out > 0) write(Lun_out,6602)
         Init_balgm_L = .false.
      endif
      Init_dfpl_8 = sec
      if (Init_balgm_L) then
         Init_dfnp   = max(2,Init_dfnp)
         if ( mod(Init_dfnp,2) /= 1 ) Init_dfnp= Init_dfnp+1
         Init_halfspan = (Init_dfnp - 1) / 2
      else
         Init_dfnp   = -9999
         Init_halfspan = -9999
      endif
      if (.not.Rstri_rstn_L) then
         Init_mode_L= .false.
         if (Init_balgm_L) Init_mode_L= .true.
      endif

      if (Lctl_debug_L) then
         call msg_set_minMessageLevel(MSG_INFOPLUS)
!        istat = gmm_verbosity(GMM_MSG_DEBUG)
!        istat = wb_verbosity(WB_MSG_DEBUG)
!        call msg_set_minMessageLevel(MSG_DEBUG)
         call handle_error_setdebug(Lctl_debug_L)
      endif

      if (Hzd_lnr_theta .lt. 0.) then
         if (lun_out>0) write (Lun_out, 6200)
         Hzd_lnr_theta = Hzd_lnr
      endif
      if (Hzd_pwr_theta .lt. 0 ) then
         if (lun_out>0) write (Lun_out, 6201)
         Hzd_pwr_theta = Hzd_pwr
      endif

      gemdm_config = 1

      Vtopo_L = (Vtopo_ndt .gt. 0)

 6005 format (/' Starting time Step_runstrt_S not specified'/)
 6007 format (/X,48('#'),/,2X,'STARTING DATE OF THIS RUN IS : ',a16,/ &
                           2X,'00H FORECAST WILL BE VALID AT: ',a16,/ &
               X,48('#')/)
 6010 format (//'  =============================='/&
                '  = Leap Years will be ignored ='/&
                '  =============================='//)
 6100 format (//'  =============================='/&
               '  AUTO-BAROTROPIC RUN:          '/&
                '  Schm_topo_L set to .false. '/&
                '  =============================='//)
 6200 format ( ' Applying DEFAULT FOR THETA HOR. DIFFUSION Hzd_lnr_theta= Hzd_lnr')
 6201 format ( ' Applying DEFAULT FOR THETA HOR. DIFFUSION Hzd_pwr_theta= Hzd_pwr')
 6300 format (/' Williamson case 3 and 4 are not available'/)
 6400 format (/' Williamson Alpha(in deg) must be 0.0 for cases greater than 2 '/)
 6500 format (/' Williamson case 2: Alpha(in deg) must be 0.0 or 90.0 for Grd_yinyang '/)
 6602 format (/' WARNING: Init_dfnp is <= 0; Settings Init_balgm_L=.F.'/)
 6700  format (/'Schm_psadj_L option in LAM config NOT fully tested yet.'&
               /'To continue using this option confirm your intent with Schm_psadj_lam_L=.true'/)
 7021 format (//'  ==========================='/&
                '  ACADEMIC 2D or 3D Advection'/&
                '  ==========================='//)
 7040 format (/' OPTION Init_balgm_L=.true. NOT AVAILABLE if applying IAU (Iau_period>0)'/)
 8000 format (/,'========= ABORT IN S/R GEMDM_CONFIG ============='/)
 9154 format (/,' Out3_nbitg IS NEGATIVE, VALUE will be set to 16'/)
 9200 format (/'ABORT: WRONG CHOICE OF SOLVER for Helmholtz problem: Sol_type_S =',a/)
 9201 format (/'ABORT: WRONG CHOICE OF PRE-CONDITIONNER FOR 2D ITERATIVE SOLVER: Sol2D_precond_S =',a/)
 9203 format (/,'ABORT: WRONG CHOICE OF ADVECTION FOR GU: set Schm_adxlegacy_L = .true.'/)
 9570 format (/,'WARNING: Vspng_nk set to zero since top piloting is used'/)
 9580 format (/,'ABORT: Non zero Lam_blend_T cannot be used without top piloting'/)
 9680 format (/,'ABORT: ',a,' cannot be less than 1.0 for T*<0'/)
 9681 format (/,'ABORT: ',a,' cannot be less than 1.0 for OPEN_TOP scheme'/)
!
!-------------------------------------------------------------------
!
      return
      end

