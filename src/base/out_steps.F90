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
!**s/r out_steps -

      subroutine out_steps
      implicit none
#include <arch_specific.hf>

!author
!     Michel Desgagne  -  Winter 2013
!
!revision
! v4_50 - Desgagne M.       - Initial version

#include "dimout.cdk"
#include "init.cdk"
#include "lctl.cdk"
#include "step.cdk"
#include "out_listes.cdk"

      integer istep,step0,stepf
      integer, save :: marker
      logical, save :: done = .false., dgflt_L = .true.
!
!     ---------------------------------------------------------------
!
      if (.not. done) then
         nullify(outd_sorties,outp_sorties,outc_sorties)
         marker= Lctl_step - 1
      endif

      if ( (.not.Init_mode_L) .and. dgflt_L) marker= Lctl_step - 1
      dgflt_L = Init_mode_L

      if (marker .ge. Lctl_step) return

      step0 = Lctl_step
      stepf = Lctl_step + 50
      marker= stepf
        
      if (associated(outd_sorties)) deallocate (outd_sorties)
      if (associated(outp_sorties)) deallocate (outp_sorties)
      if (associated(outc_sorties)) deallocate (outc_sorties)
      allocate (outd_sorties(0:MAXSET,step0:stepf))
      allocate (outp_sorties(0:MAXSET,step0:stepf))
      allocate (outc_sorties(0:MAXSET,step0:stepf))

      outd_sorties(0,:)= 0 ; outp_sorties(0,:)= 0 ; outc_sorties(0,:)= 0
      do istep = step0, stepf
         if (.not.(Init_mode_L .and. (istep+Step_initial).ge.Init_halfspan)) &
         call out_thistep (outd_sorties(0,istep),istep,MAXSET,'DYN')
         if (Init_mode_L .and. (istep+Step_initial).ge.Init_halfspan+1) cycle
         call out_thistep (outp_sorties(0,istep),istep,MAXSET,'PHY')
         call out_thistep (outc_sorties(0,istep),istep,MAXSET,'CHM')
      end do

      done = .true.
!
!     ---------------------------------------------------------------
!
      return
      end
