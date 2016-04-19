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
!**s/r gem_muststop

      logical function gem_muststop (F_finalstep)
      use iso_c_binding
      use timestr_mod, only: timestr_isstep, TIMESTR_MATCH
      implicit none
#include <arch_specific.hf>

      integer F_finalstep

#include <clib_interface_mu.hf>
#include "lctl.cdk"
#include "step.cdk"
#include "lun.cdk"
#include "init.cdk"
#include "path.cdk"
#include "cstv.cdk"
#include "out.cdk"
#include "ptopo.cdk"
      include "rpn_comm.inc"

      character*2048 filen,filen_link,append
      logical flag,pe0_master_L,output_L,finalstep_L,end_of_run_L
      integer err,unf
      real*8 timeleft,hugetype
!
!     ---------------------------------------------------------------
!
      call timing_start2 ( 70, 'MUSTOP', 1 )
      pe0_master_L = (Ptopo_myproc.eq.0) .and. (Ptopo_couleur.eq.0)
      filen      = trim(Path_output_S)//'/output_ready_MASTER'
      filen_link = trim(Path_output_S)//'/output_ready'
      output_L= .false. ; unf= 474
      finalstep_L = Step_kount.eq.F_finalstep
      end_of_run_L= (finalstep_L.and.(.not.Init_mode_L))

      if ( Out_post_L .or. end_of_run_L ) then
         output_L= .true.
         if (pe0_master_L) then
            append=''
            if (finalstep_L .and. Ptopo_last_domain_L) append='^last'
            open  ( unf,file=filen,access='SEQUENTIAL',&
                    form='FORMATTED',position='APPEND' )
            write (unf,'(3(a))') 'NORMAL_OUTPUT ','NA ',trim(Out_laststep_S)//trim(append)
            close (unf)
         endif
      endif
      
      ! Send a signal to gem_monitor_output
      if ( output_L .and. (.not.finalstep_L) ) then

         call rpn_comm_barrier (RPN_COMM_ALLGRIDS, err)
         if (pe0_master_L) then
            err = clib_symlink ( trim(filen), trim(filen_link) )
            write (6,1001) trim(Out_laststep_S),lctl_step      
         endif

      endif

      ! Get timeleft to determine if we can continue
      if (pe0_master_L) then
         filen=trim(Path_basedir_S)//'/time_left'

         open (unf,file=trim(filen),access='SEQUENTIAL',&
               status='OLD',iostat=err,form='FORMATTED')
         if (err.eq.0) then
            read (unf,'(e)',end=33,err=33) timeleft
33          close(unf)
         else
            timeleft= huge(hugetype)
         endif
         flag = (timeleft.lt.Step_maxwall)
      endif

      call RPN_COMM_bcast (flag, 1, "MPI_LOGICAL",0,"MULTIGRID",err)
      gem_muststop = flag
      
      if ( (Step_kount.gt.0) .and. .not. &
            (Init_mode_L .and. (Step_kount.ge.Init_halfspan)) ) &
           gem_muststop = gem_muststop .or. &
           timestr_isstep ( Fcst_rstrt_S, Step_CMCdate0, real(Cstv_dt_8),&
                            Step_kount ) == TIMESTR_MATCH

      gem_muststop= gem_muststop .and. .not.end_of_run_L

      if (gem_muststop) call wrrstrt ()
      
      call gemtim4 ( Lun_out, 'CURRENT TIMESTEP', .false. )
      call timing_stop (70)

 1001 format (' OUT_LAUNCHPOST: DIRECTORY output/',a, &
              ' was released for postprocessing at timestep: ',i9)
 1002 format (' SAVING A RESTART AT TIMESTEP: ',i7,' valid: ',a)
!
!     ---------------------------------------------------------------
!
      return
      end
