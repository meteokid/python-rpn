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

!**s/r gem_run - Performs the integration of the model
!
      subroutine gem_run (F_rstrt_L)
      implicit none
#include <arch_specific.hf>

      logical F_rstrt_L

!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_rstrt_L     O         Is a restart required
!----------------------------------------------------------------

#include "init.cdk"
#include "lun.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "rstr.cdk"
#include "schm.cdk"
#include "lctl.cdk"
#include "grd.cdk"

      logical, external :: gem_muststop
      integer, external :: model_timeout_alarm
      character*16 datev
      integer stepf,seconds_since,last_step
      real*8 dayfrac, sec_in_day
      parameter (sec_in_day=86400.0d0)
!
!     ---------------------------------------------------------------
!
      dayfrac = dble(Step_kount) * Cstv_dt_8 / sec_in_day
      call incdatsd (datev,Step_runstrt_S,dayfrac)

      if (Lun_out.gt.0) write (6,900) datev

      call blocstat (.true.)

      call gemtim4 ( Lun_out, 'STARTING TIME LOOP', .false. )

      stepf= Step_total
      if (Init_mode_L) stepf= Init_dfnp-1
      last_step= Step_initial + stepf

      F_rstrt_L = .false.
      if ( .not. Rstri_rstn_L ) then
         call out_outdir (Step_total)
         if (Step_kount.eq.0) then
            call iau_apply2 (Step_kount)
            if ( Schm_phyms_L ) then
               call pw_shuffle
               call pw_update_GPW
               call pw_update_UV
               call pw_update_T
               call itf_phy_step (0,Lctl_step)
            endif
         endif
         call out_dyn (.true., .true.)
         if (gem_muststop (stepf)) goto 999
      endif

      do while (Step_kount .lt. stepf)

         seconds_since= model_timeout_alarm(Step_alarm)

         Lctl_step= Lctl_step + 1  ;  Step_kount= Step_kount + 1
         if (Lun_out.gt.0) write (Lun_out,1001) Lctl_step,last_step

         call out_outdir (Step_total)

         call pw_shuffle

         call dynstep

         call out_dyn (.false., .true.) ! casc output

         if ( Schm_phyms_L ) call itf_phy_step (Step_kount, Lctl_step)

         call hzd_main

         call iau_apply2 (Step_kount)

         if (Grd_yinyang_L) call yyg_xchng_all

         if ( Init_mode_L ) call digflt ! digital filter

         call out_dyn (.true., .false.) ! regular output

         call blocstat (.false.)

         if (Lun_out.gt.0) write(Lun_out,3000) Lctl_step

         call save_restart
         
         F_rstrt_L= gem_muststop (stepf)

         if (F_rstrt_L) exit

      end do

 999  seconds_since= model_timeout_alarm(Step_alarm)

      if (Lun_out.gt.0) write(Lun_out,4000) Lctl_step

 900  format (/'STARTING THE INTEGRATION WITH THE FOLLOWING DATA: VALID ',a)
 1001 format(/,'DYNAMICS: PERFORMING TIMESTEP #',I9,' OUT OF ',I9, &
             /,'=========================================================')
 3000 format(/,'THE TIME STEP ',I8,' IS COMPLETED')
 4000 format(/,'GEM_RUN: END OF THE TIME LOOP AT TIMESTEP',I8, &
             /,'===================================================')
!
!     ---------------------------------------------------------------
!
      return
      end
