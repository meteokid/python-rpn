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

!**s/r out_outdir - Establishes output directory base on Lctl_step and Out3_postfreq
!

      subroutine out_outdir (upperlimit)
      use timestr_mod, only: timestr_prognum,timestr_unitfact
      implicit none
#include <arch_specific.hf>

      integer upperlimit

!AUTHOR   Michel Desgagne  - Summer 2010
!
!REVISION
! v4_14 - Desgagne M.      - Initial version

#include <rmnlib_basics.hf>
#include "cstv.cdk"
#include "grd.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "ptopo.cdk"
#include "lun.cdk"
#include "path.cdk"
#include "step.cdk"
#include <clib_interface_mu.hf>
      include "rpn_comm.inc"
      
!!$      real(RDOUBLE),parameter :: SECDAY  = 86400.0d0
!!$      real(RDOUBLE),parameter :: SECHOUR = 3600.0d0
!!$      real(RDOUBLE),parameter :: SECMIN  = 60.0d0

      character(len=1024),save :: dirstep_S=' ', diryy_S=' ', dirbloc_S=' ', &
                                  FMT=' ', last_S=' '
      character*10 postjob_S
      character*7  blocxy_S
      character*2  digits_S
      logical flag
      integer err,last_step_post,flag_step_post,stepno,timing,ndigits,remainder,prognum,prognum1
      real :: interval
      real(RDOUBLE) :: fatc_8
!
!----------------------------------------------------------------------
!
      call out_steps

      write (blocxy_S ,'(I3.3,"-",I3.3)') Out_myblocx, Out_myblocy

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

      if (Grd_yinyang_L) then
         ! Wait for multigrid PE0 to be finished subdir creation
         call rpn_comm_barrier(RPN_COMM_MULTIGRID, err)

         Out_dirname_S = trim(Out_dirname_S)//'/'//trim(Grd_yinyang_S)
         if (diryy_S /= Out_dirname_S) then
            diryy_S = Out_dirname_S
            if (Ptopo_myproc == 0) then
               err = clib_mkdir(trim(Out_dirname_S))
            endif
         endif
      endif
      
      ! Wait for Grid PE0 to be finished subdir creation
      call rpn_comm_barrier(RPN_COMM_BLOCMASTER, err)

      ! Each block master now creates an independent subdir for outputs
      Out_dirname_S = trim(Out_dirname_S)//'/'//blocxy_S
      err = CLIB_OK
      if (Ptopo_blocme == 0 .and. dirbloc_S /= Out_dirname_S) then
         dirbloc_S = Out_dirname_S
         err = clib_mkdir ( trim(Out_dirname_S) )
         err = clib_isdir ( trim(Out_dirname_S) )
      endif

      call handle_error(err,'out_outdir','unable to create output directory structure')

 1001 format (' OUT_OUTDIR: DIRECTORY output/',a,' was created at timestep: ',i9)
!
!----------------------------------------------------------------------
!
      return
      end
