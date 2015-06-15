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

!**s/r itf_phy_step - Apply the physical processes: CMC/RPN package
!
      subroutine itf_phy_step ( F_step_kount, F_lctl_step )
      use phy_itf, only: phy_input,phy_step
      implicit none
#include <arch_specific.hf>

      integer, intent(IN) :: F_step_kount, F_lctl_step

!authors 
!     Desgagne, McTaggart-Cowan, Chamberland -- Spring 2014
!
!revision
! v4_70 - authors          - initial version
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_step_kount  I          step count
! F_lctl_step   I          step number
!----------------------------------------------------------------

#include "lun.cdk"
#include "ptopo.cdk"
#include "path.cdk"
#include "rstr.cdk"

      integer,external :: itf_phy_prefold_opr

      integer err
!
!     ---------------------------------------------------------------
!
      if ( Rstri_user_busper_L .and. (F_step_kount.eq.0) ) return

      call timing_start2 ( 40, 'PHYSTEP', 1 )

      if (Lun_out.gt.0) write (Lun_out,1001) F_lctl_step

      if (F_step_kount == 0) call itf_phy_geom4 (err)

      !call pw_glbstat('DEBUG')

      call itf_phy_copy ()

      call rpn_comm_bloc (Ptopo_ninblocx,Ptopo_ninblocy)

      call timing_start2 ( 45, 'PHY_input', 40)
      err = phy_input ( itf_phy_prefold_opr,F_step_kount,Path_phyincfg_S,&
                        Path_phy_S, 'GEOPHY/Gem_geophy.fst' )
      call handle_error (err,'itf_phy_step','Problem with phy_input') 
      call timing_stop  ( 45 )

      call rpn_comm_bloc ( Ptopo_nblocx  ,Ptopo_nblocy )

      call pe_rebind (Ptopo_nthreads_phy,(Ptopo_myproc.eq.0).and. &
                                         (F_step_kount    .eq.0)  )

      call timing_start2 ( 46, 'PHY_step', 40)
      err = phy_step ( F_step_kount, F_lctl_step )
      call timing_stop  ( 46 )

      call handle_error (err,'itf_phy_step','Problem with phy_step') 

      call pe_rebind (Ptopo_nthreads_dyn,(Ptopo_myproc.eq.0).and. &
                                         (F_step_kount    .eq.0) )

      call timing_start2 ( 47, 'PHY_update', 40)
      call itf_phy_update3 ( F_step_kount > 0 )
      call timing_stop  ( 47 )

      call timing_start2 ( 48, 'PHY_output', 40)
      call itf_phy_output2 ( F_lctl_step )
      call timing_stop  ( 48 )

      call timing_stop ( 40 )

 1001 format(/,'PHYSICS : PERFORMING TIMESTEP #',I9, &
             /,'========================================')
!
!     ---------------------------------------------------------------
!
      return
      end subroutine itf_phy_step
