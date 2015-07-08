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

!**s/r step_nml - Read namelist time

      integer function step_nml (F_namelistf_S)
      use timestr_mod
      implicit none
#include <arch_specific.hf>

      character* (*) F_namelistf_S

!authors    Michel Desgagne - Spring 2011
! 
!revision
! v4_40 - Desgagne M.       - initial MPI version
!
!object
!  Default configuration and reading namelist 'step'

#include <rmnlib_basics.hf>
#include "lun.cdk"
#include "grd.cdk"
#include "lctl.cdk"
#include "rstr.cdk"
#include "step.cdk"

      integer nrec,unf,err
      real :: sec
!
!-------------------------------------------------------------------
!
      step_nml = -1

      if ((F_namelistf_S.eq.'print').or.(F_namelistf_S.eq.'PRINT')) then
         step_nml = 0
         if (Lun_out.gt.0) write (6  ,nml=step)
         return
      endif

! Defaults values for ptopo namelist variables

      Step_runstrt_S = 'NIL'
      Fcst_start_S   = ''
      Fcst_end_S     = ''
      Fcst_nesdt_S   = ''
      Fcst_gstat_S   = ''
      Fcst_rstrt_S   = ''
      Fcst_bkup_S    = ''
      Fcst_spinphy_S = ''
      Fcst_alarm_S = ''

      Step_dt          = -1.0
      Step_leapyears_L = .true.

      if (F_namelistf_S .ne. '') then
         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', nrec) .ne. 0) then
            if (Lun_out.ge.0) write (Lun_out, 7050) trim( F_namelistf_S )
            goto 9999
         endif
         rewind(unf)
         read (unf, nml=step, end = 9120, err=9130)
         goto 9000
      endif

 9120 if (Lun_out.ge.0) write (Lun_out, 7060) trim( F_namelistf_S )
      goto 9999
 9130 if (Lun_out.ge.0) write (Lun_out, 7070) trim( F_namelistf_S )
      goto 9999

 9000 if (Step_dt .lt. 0.) then
         if (Lun_out.gt.0) write(Lun_out,*)  &
                    ' Step_dt must be specified in namelist &step'
         goto 9999
      endif

      err= 0

      if ( Fcst_start_S  == '' ) Fcst_start_S  = '0H'
      if ( Fcst_end_S    == '' ) Fcst_end_S    = Fcst_start_S

      err= min( timestr2step (Step_initial, Fcst_start_S, Step_dt), err)
      err= min( timestr2step (Step_total  , Fcst_end_S  , Step_dt), err)
      err= min( timestr2step (Step_nesdt  , Fcst_nesdt_S, Step_dt), err)
      Step_total= Step_total - Step_initial

      if ( Fcst_rstrt_S  == '' ) then
         write(Fcst_rstrt_S,'(a,i6)') 'step,',Step_total+1
      else
         err = timestr_check(Fcst_rstrt_S)
      endif
      if ( Fcst_bkup_S  == '' ) then
         write(Fcst_bkup_S,'(a,i6)') 'step,',Step_total+1
      else
         err = timestr_check(Fcst_bkup_S)
      endif

      if ( Fcst_gstat_S  == '' ) then
         Step_gstat= Step_total-Step_initial+1
      else
         err= min( timestr2step (Step_gstat, Fcst_gstat_S, Step_dt), err)
      endif
      if ( Fcst_spinphy_S  == '' ) then
         Step_spinphy= Step_total-Step_initial+1
      else
         err= min( timestr2step (Step_spinphy, Fcst_spinphy_S, Step_dt), err)
      endif
      if ( Fcst_alarm_S  == '' ) then
         Step_alarm= 600
      else
         err= min( timestr2step (Step_alarm, Fcst_alarm_S, Step_dt), err)
      endif

      if (err.lt.0) goto 9999

      Step_delay= Step_initial

      if (.not.Rstri_rstn_L) Lctl_step= Step_initial

      if ( (Step_nesdt .le. 0) .and. (Grd_typ_S(1:1).eq.'L') ) then
         if (Lun_out.gt.0) write(Lun_out,*)  &
                    ' Step_nesdt must be specified in namelist &step'
         goto 9999
      else
         Step_nesdt= Step_nesdt * Step_dt
      endif

      step_nml = 1

 7050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 7060 format (/,' Namelist &step NOT AVAILABLE in FILE: ',a/)
 7070 format (/,' NAMELIST &step IS INVALID IN FILE: ',a/)

 9999 err = fclos (unf)
!
!-------------------------------------------------------------------
!
      return
      end function step_nml
