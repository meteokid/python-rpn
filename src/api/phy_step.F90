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

module phy_step_mod
   use phy_options, only: phy_init_ctrl, PHY_CTRL_INI_OK, dyninread_list_s, delt, date, satuco
   use phy_typedef, only: PHY_NONE
   private
   public :: phy_step

contains

  !/@*
  function phy_step (F_stepcount, F_stepdriver) result(F_istat)
    use str_mod, only: str_concat
    use physlb_mod, only: physlb1
    use cpl_itf   , only: cpl_step
    use phybus, only: entbus, perbus, dynbus, volbus
    implicit none

    !@objective Apply the physical processes: CMC/RPN package
    integer, intent(in) :: F_stepcount     !Step kount incrementing from 0
    integer, intent(in) :: F_stepdriver    !Step number of the driving model
    integer :: F_istat                     !Return status (RMN_OK or RMN_ERR)

    !@authors Desgagne, Chamberland, McTaggart-Cowan, Spacek -- Spring 2014

    !@revision
    !*@/
#include <rmnlib_basics.hf>
#include <arch_specific.hf>
#include <WhiteBoard.hf>
#include <msg.h>
    include "phygrd.cdk"
    include "physteps.cdk"

    integer, external :: phyput_input_param, sfc_get_input_param

    integer, save :: pslic

    logical :: series_L, srsus_L
    integer :: istat, type1, sizeof1, ntr, options1
    character(len=256) :: str256

    !---------------------------------------------------------------
    F_istat = RMN_ERR
    if (phy_init_ctrl == PHY_NONE) then
       F_istat = PHY_NONE
       return
    else if (phy_init_ctrl /= PHY_CTRL_INI_OK) then
       call msg(MSG_ERROR,'(phy_step) Physics not properly initialized.')
       return
    endif

    if (F_stepcount == 0) then
       ntr = 0
       istat = wb_get_meta('itf_phy/READ_TRACERS', type1, sizeof1, ntr, options1)
       if (WB_IS_OK(istat) .and. ntr > 0) then
          allocate(dyninread_list_s(ntr))
          istat = wb_get('itf_phy/READ_TRACERS', dyninread_list_s, ntr)
          if (.not.WB_IS_OK(istat)) then
             call msg (MSG_ERROR, &
                  '(phy_step) Unable to retrieve tracer WB entry')
             return
          endif
          call str_concat(str256, dyninread_list_s, ', ')
          call msg(MSG_INFO,'(phy_step) List of read tracers: '//trim(str256))
      else
          call msg(MSG_INFO,'(phy_step) No list of read tracers found')
          ntr = 0
          allocate(dyninread_list_s(1))
          dyninread_list_s = ''
       endif
    endif

    istat = wb_get('model/series/P_serg_srsus_L' , srsus_L)
    if (istat /= WB_OK) call msg(MSG_ERROR,'(wb P_serg_srsus_L)')
    series_L = (srsus_L .and. (F_stepcount >= 1))

    if (series_L) &
         call ser_ctrl2(perbus, size(perbus,1), phydim_ni, &
                        phydim_nj, F_stepcount, delt, date, satuco)

    pslic = 0
    step_kount  = F_stepcount
    step_driver = F_stepdriver
    istat = WB_OK
    istat = min(phyput_input_param(),istat)
    istat = min(sfc_get_input_param(),istat)
    if (istat /= WB_OK) call msg(MSG_ERROR,'(phy_step)')

    call cpl_step(F_stepcount, F_stepdriver)

!$omp parallel
    call physlb1(entbus, dynbus, perbus, volbus, &
         size(entbus,1), size(dynbus,1), size(perbus,1), size(volbus,1), &
         F_stepcount, phydim_ni, phydim_nj, phydim_nk, pslic)
!$omp end parallel

    if (series_L) call ser_out(F_stepcount == 1, date, satuco)

    call phystats(F_stepcount, delt)

    F_istat = RMN_OK
    !---------------------------------------------------------------
    return
  end function phy_step

end module phy_step_mod

