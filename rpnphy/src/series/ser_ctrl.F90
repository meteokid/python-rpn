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

!**s/r ser_ctrl  - suspend or resume time series/grid point diagnostics
!                 extractions
!
      subroutine ser_ctrl2(BUSPER_3d, esp_busper, &
                           phydim_ni, phydim_nj, F_stepno         , &
                           F_delt, date, satuco )
      implicit none
#include <arch_specific.hf>

      logical satuco
      integer esp_busper, phydim_ni, phydim_nj, F_stepno, date(14)
      real    F_delt
      real, dimension(esp_busper, phydim_nj) :: BUSPER_3d

!author
!     Andre Methot - aug 94 - v0_14
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v3_30 - Winger K.         - correct time series handling in climate mode
!
!object
!         This subroutine suspend or resume time series/grid point
!         diagnostics extractions IF it is required.
!
!     This subroutine was designed to skip extractions during the
!     second half of digital filter initialization.
!     It is called a each timestep.
!
!     The routine is also used to stop time series extractions after
!     a user given timestep.

#include <WhiteBoard.hf>
#include "series.cdk"
      logical, save :: done  = .false.
      integer ier
      real nhours
!
!     ---------------------------------------------------------------
!
      if (.not.done) then
         call ser_geopf2(BUSPER_3d, esp_busper, &
                         phydim_ni, phydim_nj, date, satuco)
      endif

      done= .true.
      P_serg_sroff_L= .false.
      if (F_stepno .eq. (P_serg_serstp+1)) P_serg_sroff_L= .true.

      call serset ('KOUNT',  F_stepno, 1, ier)
      nhours= float(F_stepno)*F_delt/3600.
      call serset ('HEURE',  nhours,   1, ier)  
!
!     ---------------------------------------------------------------
!      
      return
      end
