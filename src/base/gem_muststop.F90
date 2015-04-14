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
#include "init.cdk"
#include "path.cdk"
#include "cstv.cdk"
#include "modconst.cdk"
#include "out.cdk"
#include "ptopo.cdk"
      include "rpn_comm.inc"

      character*1024 timeleft_S,filen,filen_link,append
      character*16   datev
      logical flag,pe0_master_L,bkup_L,output_L,finalstep_L
      logical, save :: done=.false. , last_domain_L
      integer err,unf
      real*8 timeleft,hugetype,dayfrac
      real*8, parameter :: OV_day = 1.d0/86400.0d0
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,3000) Lctl_step

      pe0_master_L = (Ptopo_myproc.eq.0) .and. (Ptopo_couleur.eq.0)
      filen      = trim(Path_output_S)//'/output_ready_MASTER'
      filen_link = trim(Path_output_S)//'/output_ready'
      bkup_L= .false. ; output_L= .false. ; unf= 474
      finalstep_L= Step_kount.eq.F_finalstep
      if (.not.done) then
         err= wb_get('model/last_domain',last_domain_L)
         done= .true.
      endif

      if (mod(Step_kount,Step_bkup).eq.0) then
         bkup_L= .true.
         call wrrstrt ()
         dayfrac = dble(Step_kount) * Cstv_dt_8 * OV_day
         call incdatsd (datev,Mod_runstrt_S,dayfrac)
         
         if (pe0_master_L) then
            write (6,1002) Lctl_step,datev
            open  ( unf,file=filen,access='SEQUENTIAL',&
                    form='FORMATTED',position='APPEND' )
            write (unf,'(3(a))') 'SAVE_RESTART ', trim(datev), ' '//trim(Out_laststep_S)
            close (unf)
         endif
      endif
      
      if ( Out_post_L .or. (finalstep_L.and.(.not.Init_mode_L)) ) then
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
      
      if ( (output_L .or. bkup_L) .and. (.not.finalstep_L) ) then

         call rpn_comm_barrier (RPN_COMM_ALLGRIDS, err)

         if (pe0_master_L) then
            err = clib_symlink ( trim(filen), trim(filen_link) )
            write (6,1001) trim(Out_laststep_S),lctl_step      
         endif

      endif

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
      call RPN_COMM_bcast (flag, 1, "MPI_LOGICAL",0,"allgrids",err)

      gem_muststop = mod(Step_kount,Step_rsti).eq.0 .or. flag

      if (.not.Init_mode_L) &
      gem_muststop = gem_muststop .and. finalstep_L

      if (gem_muststop .and. (.not.bkup_L) ) call wrrstrt ()
      
      call gemtim4 ( Lun_out, 'CURRENT TIMESTEP', .false. )

! TODO: itf_phy_restart must be revisited ????
!      if ( Lctl_step.eq.Step_spinphy ) call itf_phy_restart ('W', .true.)

 1001 format (' OUT_LAUNCHPOST: DIRECTORY output/',a, &
              ' was released for postprocessing at timestep: ',i9)
 1002 format (' SAVING A RESTART AT TIMESTEP: ',i7,' valid: ',a)
 3000 format(/,'THE TIME STEP ',I8,' IS COMPLETED')
!
!     ---------------------------------------------------------------
!
      return
      end
