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

!s/r wrrstrt - Write the restart file
!
      subroutine wrrstrt ()
      use phy_itf, only: phy_restart
      implicit none
#include <arch_specific.hf>
!
!author
!     M. Desgagne - Mars 2000
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v2_10 - Desgagne M.       - introduce WA files
! v2_21 - Dugas B.          - adapt to climate mode
! v2_30 - Corbeil L.        - Added writing of pres_surf pres_top
! v2_31 - Desgagne M.       - Add Tr2d tracers
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_21 - Valcke, S.        - Oasis coupling: Removed wawrit of c_cplg_step
! v3_21 - Lee V.            - Remove Tr2d tracers
! v3_30 - Desgagne & Winger - Write one global binary restart file if required
! v3_30 - Desgagne M.       - restart for coupling
! v3_31 - Desgagne M.       - new coupling interface to OASIS
! v3_31 - Desgagne M.       - restart with physics BUSPER
! v4_05 - Lepine M.         - VMM replacement with GMM

#include "gmm.hf"
#include "lun.cdk"
#include "init.cdk"
#include "step.cdk"
#include "lctl.cdk"
#include "tr3d.cdk"
#include "ntr2mod.cdk"

      include "rpn_comm.inc"

      integer, external :: fnom,fclos

      integer ier,gmmstat,me,howmany,newcomm,i
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,2000) Lctl_step

      call split_on_hostid (RPN_COMM_comm('GRID'),me,howmany,newcomm)

      call timing_start ( 33, 'RESTART' )
      do i=0,howmany-1
         if (i.eq.me) then

            Lun_rstrt = 0
            ier = fnom (Lun_rstrt,'gem_restart','SEQ+UNF',0)

            write(Lun_rstrt) Lctl_step,Step_kount,Init_mode_L
            write(Lun_rstrt)  NTR_runstrt_S, NTR_horzint_L       , &
                      NTR_Tr3d_name_S,NTR_Tr3d_wload,NTR_Tr3d_hzd, &
                      NTR_Tr3d_mono,NTR_Tr3d_mass,NTR_Tr3d_ntr

            ier = fclos(Lun_rstrt)  

            !        Write Gmm-files

            gmmstat = gmm_checkpoint_all(GMM_WRIT_CKPT)

            ier = phy_restart ('W', .false.)

         endif

         call mpi_barrier (newcomm,ier)

      end do
      call timing_stop (33)

 2000 format(/,'WRITING A RESTART FILE AT TIMESTEP #',I8, &
             /,'=========================================')
!
!     ---------------------------------------------------------------
!      
      return
      end
