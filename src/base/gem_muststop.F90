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
!
      logical function gem_muststop (F_finalstep)
      use timestr_mod, only: timestr_isstep, TIMESTR_MATCH
      implicit none
#include <arch_specific.hf>
!
      integer F_finalstep
!
!author
!     Michel Desgagne  -  Spring 2015
!
!revision
! v4_80 - Desgagne M.       - Initial version
!
#include <clib_interface_mu.hf>
#include <WhiteBoard.hf>
#include "lctl.cdk"
#include "step.cdk"
#include "lun.cdk"
#include "grd.cdk"
#include "init.cdk"
#include "path.cdk"
#include "cstv.cdk"
#include "out.cdk"
#include "ptopo.cdk"
      include "rpn_comm.inc"

      character*2048 timeleft_S,filen,filen_link,append,dirname_S,cmd
      character*16   datev
      character*3 mycol_S,myrow_S
      logical flag,pe0_master_L,output_L,finalstep_L,end_of_run_L
      logical, save :: done=.false. , last_domain_L
      integer err,unf
      real*8 timeleft,hugetype,dayfrac
      real*8, parameter :: OV_day = 1.d0/86400.0d0
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,3000) Lctl_step

      call timing_start ( 34, 'MUSTOP' )
      pe0_master_L = (Ptopo_myproc.eq.0) .and. (Ptopo_couleur.eq.0)
      filen      = trim(Path_output_S)//'/output_ready_MASTER'
      filen_link = trim(Path_output_S)//'/output_ready'
      output_L= .false. ; unf= 474
      finalstep_L = Step_kount.eq.F_finalstep
      end_of_run_L= (finalstep_L.and.(.not.Init_mode_L))
      if (.not.done) then
         err= wb_get('model/last_domain',last_domain_L)
         done= .true.
      endif

      if ( timestr_isstep (Fcst_bkup_S, Step_CMCdate0, real(Cstv_dt_8), &
                           Step_kount) == TIMESTR_MATCH ) then
         call wrrstrt ()
         dayfrac = dble(Step_kount) * Cstv_dt_8 * OV_day
         call incdatsd (datev,Step_runstrt_S,dayfrac)
         dirname_S= trim(Path_output_S)//'/'//Out_laststep_S//'/restart_'//trim(datev)
         if (Grd_yinyang_L) &
         dirname_S=trim(dirname_S)//'/'//trim(Grd_yinyang_S)
         if (Ptopo_myproc.eq.0) then
            err= clib_mkdir_r ( trim(dirname_S) )
            call mkdir_gem ( trim(dirname_S), Ptopo_npex, Ptopo_npey )
         endif
         call rpn_comm_Barrier("grid", err)
         write(mycol_S,'(i3.3)') Ptopo_mycol
         write(myrow_S,'(i3.3)') Ptopo_myrow
         cmd='mv *_restart '//trim(dirname_S)//'/'//mycol_S//'-'//myrow_S
         call system (cmd)
!yet another solution...
!err = clib_glob(filelist,nfiles,'_restart',maxnfiles)
!do ifile = 1,nfiles
!    istat = clib_basename(filelist(ifile),myname)
!    pathnew = trim(newdir)//'/'//trim(myname)
!    istat = clib_rename(filelist(ifile),pathnew)
!enddo

         if (pe0_master_L) write (6,1002) Lctl_step,datev
      endif
      
      if ( Out_post_L .or. end_of_run_L ) then
         output_L= .true.
         if (pe0_master_L) then
            append=''
            if (finalstep_L .and. last_domain_L) append='^last'
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
      gem_muststop = flag .or. &
           timestr_isstep ( Fcst_rstrt_S, Step_CMCdate0, real(Cstv_dt_8),&
                            Step_kount ) == TIMESTR_MATCH

      gem_muststop= gem_muststop .and. .not.end_of_run_L

      if (gem_muststop) call wrrstrt ()
      
      call gemtim4 ( Lun_out, 'CURRENT TIMESTEP', .false. )
      call timing_stop (34)

 1001 format (' OUT_LAUNCHPOST: DIRECTORY output/',a, &
              ' was released for postprocessing at timestep: ',i9)
 1002 format (' SAVING A RESTART AT TIMESTEP: ',i7,' valid: ',a)
 3000 format(/,'THE TIME STEP ',I8,' IS COMPLETED')
!
!     ---------------------------------------------------------------
!
      return
      end
