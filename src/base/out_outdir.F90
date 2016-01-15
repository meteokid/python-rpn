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

      subroutine out_outdir
      use iso_c_binding
      use timestr_mod, only: timestr_prognum,timestr_unitfact
      implicit none
#include <arch_specific.hf>

!AUTHOR   Michel Desgagne  - Summer 2015
!
!REVISION
! v4_80 - Desgagne M.      - Initial version

#include <rmnlib_basics.hf>
#include "cstv.cdk"
#include "grd.cdk"
#include "lctl.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "ptopo.cdk"
#include "lun.cdk"
#include "path.cdk"
#include "step.cdk"
#include "init.cdk"
#include <clib_interface_mu.hf>
      include "rpn_comm.inc"
      
      character(len=1024),save :: dirstep_S=' ', diryy_S=' ', dirbloc_S=' ', &
                                  FMT=' ', last_S=' '
      character*16 datev
      character*10 postjob_S
      character*7  blocxy_S
      character*2  digits_S
      integer err,last_step_post,flag_step_post,stepno,timing,ndigits, &
              remainder,prognum,prognum1,upperlimit
      real :: interval
      real,   parameter :: eps=1.e-12
      real*8, parameter :: OV_day = 1.0d0/86400.0d0
      real*8  dayfrac,fatc_8
!
!----------------------------------------------------------------------
!
      upperlimit = Step_total

      call out_steps

      if ( Init_mode_L .and. (Step_kount.gt.Init_halfspan) ) return

      write (blocxy_S,'(I3.3,"-",I3.3)') Ptopo_mycol, Ptopo_myrow

      if (Out3_postproc_fact <= 0) then
         last_step_post = upperlimit
         flag_step_post = upperlimit
         Out_post_L = .false.
      else
         interval = Out3_close_interval * Out3_postproc_fact
         stepno = max(Step_kount,1)
         err = timestr_prognum(prognum ,Out3_unit_S,interval,Out_dateo,&
                               float(Out_deet),stepno  ,Out_endstepno)
         err = timestr_prognum(prognum1,Out3_unit_S,interval,Out_dateo,&
                               float(Out_deet),stepno+1,Out_endstepno)
         Out_post_L = (prognum1 > prognum .or. stepno == Out_endstepno)

         if (Out3_unit_S(1:3) == 'MON') then
            last_step_post = prognum
         else
            fatc_8 = timestr_unitfact(Out3_unit_S,Cstv_dt_8)
            last_step_post = nint(dble(prognum) * fatc_8)
            last_step_post = min(last_step_post+Step_delay,upperlimit)
         endif
      endif

! These next few lines will serve soon in establishing
! a self adjustable lenght for last_S which will replace postjob_S
!      ndigits=1
!      remainder=Step_total/10
!      do while (remainder > 0)
!         remainder=remainder/10
!         ndigits = ndigits + 1
!      end do
!      write (digits_S,'(i2)') ndigits
!      FMT="(i"//trim(digits_S)//"."//trim(digits_S)//")"
!      write (last_S,trim(FMT)) last_step_post

      if (last_step_post.ge.0) then
         write (postjob_S,'(i10.10)') last_step_post
      else
         write (postjob_S,'(i10.9) ') last_step_post
      endif

      Out_laststep_S = 'laststep_'//postjob_S
      Out_dirname_S  = trim(Path_output_S)//'/'//Out_laststep_S

      ! PE0 is responsible for creating shared subdir structure
      if (dirstep_S /= Out_dirname_S) then
         dirstep_S = Out_dirname_S        
         if (Ptopo_myproc == 0 .and. Ptopo_couleur == 0) then
            err = clib_mkdir(trim(Out_dirname_S))
            if (Lun_out>0) write(Lun_out,1001) trim(Out_laststep_S),Step_kount
         endif
      endif
      
      ! Wait for Grid PE0 to be finished subdir creation
      call rpn_comm_barrier (RPN_COMM_ALLGRIDS, err)

      ! Each io pe now creates an independent subdir for outputs
      Out_dirname_S = trim(Out_dirname_S)//'/'//blocxy_S
      err = CLIB_OK
      if (Out3_iome .ge. 0 .and. dirbloc_S /= Out_dirname_S &
                           .and. Ptopo_couleur == 0) then
         dirbloc_S = Out_dirname_S
         err = clib_mkdir ( trim(Out_dirname_S) )
         err = clib_isdir ( trim(Out_dirname_S) )
      endif

      call gem_error (err,'out_outdir','unable to create output directory structure')

      Out_dateo = Out3_date
      if ( lctl_step .lt. 0 ) then  ! adjust Out_dateo because ip2=npas=0
         dayfrac = dble(lctl_step-Step_delay) * Cstv_dt_8 * OV_day
         call incdatsd (datev,Step_runstrt_S,dayfrac)
         call datp2f   (Out_dateo,datev)
      endif

      Out_ip2  = int (dble(lctl_step) * Out_deet / 3600. + eps)
      Out_ip2  = max (0, Out_ip2  )
      Out_npas = max (0, Lctl_step)

      Out_ip3  = 0
      if (Out3_ip3.eq.-1) Out_ip3 = max (0, Lctl_step)
      if (Out3_ip3.gt.0 ) Out_ip3 = Out3_ip3

      Out_typvar_S = 'P'
      if (Lctl_step.lt.0) Out_typvar_S = 'I'
      
 1001 format (' OUT_OUTDIR: DIRECTORY output/',a,' was created at timestep: ',i9)
!
!----------------------------------------------------------------------
!
      return
      end
