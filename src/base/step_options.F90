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
module step_options
   implicit none
   public
   save
   
   !# Starting date for model run  (yyyymmdd.hhmmss)
   character(len=16) :: Step_runstrt_S = 'NIL'
   namelist /step/ Step_runstrt_S 
   
   !# Starting date for model run slice (yyyymmdd.hhmmss)
   character(len=16) :: Fcst_start_S =  ' '
   namelist /step/ Fcst_start_S
   
   !# End date for model run slice (yyyymmdd.hhmmss)
   character(len=16) :: Fcst_end_S =  ' '
   namelist /step/ Fcst_end_S
   
   !# Read nesting data every Fcst_nesdt_S
   character(len=16) :: Fcst_nesdt_S = ' '
   namelist /step/ Fcst_nesdt_S
   
   !# Output global stat (glbstat) every Fcst_gstat_S
   character(len=16) :: Fcst_gstat_S = ' '
   namelist /step/ Fcst_gstat_S
   
   !# Save a restart file + stop every Fcst_rstrt_S
   character(len=16) :: Fcst_rstrt_S = ' '
   namelist /step/ Fcst_rstrt_S 
   
   !# Save a restart file + continue every Fcst_bkup_S
   character(len=16) :: Fcst_bkup_S = 'NIL'
   namelist /step/ Fcst_bkup_S
   
   !# 
   character(len=16) :: Fcst_spinphy_S = ' '
   namelist /step/ Fcst_spinphy_S
   
   !# Save a restart file + continue at that time
   character(len=16) :: Fcst_bkup_additional_S = 'NIL'
   namelist /step/ Fcst_bkup_additional_S
      
   !# Setting for Fortran alarm time
   integer :: Step_alarm = 600
   namelist /step/ Step_alarm
   
   !# Account for leap years
   logical :: Step_leapyears_L = .true.
   namelist /step/ Step_leapyears_L
   
   !# Length of model timestep (sec)
   real*8  :: Step_dt = -1.
   namelist /step/ Step_dt

   ! Internal variables NOT in step namelist

   integer Step_total, Step_gstat, Step_delay, Step_spinphy, &
           Step_kount, Step_CMCdate0           , &
           Step_initial, Step_bkup_additional

   real*8  Step_nesdt, Step_maxwall

contains

   function step_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      
      return
   end function step_options_init

end module step_options
