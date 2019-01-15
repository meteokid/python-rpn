!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

module cnv_options
   implicit none 
   public
   save

   !# Number of species for convective transport (never tested)
   integer           :: bkf_kch         = 0
   namelist /convection_cfgs/ bkf_kch

   !# Number of additional ensemble members (max 3) for deep bkf convection
   integer           :: bkf_kens        = 0
   namelist /convection_cfgs/ bkf_kens

   !# Take ice phase into account in deep bkf (yes=1)
   integer           :: bkf_kice        = 1
   namelist /convection_cfgs/ bkf_kice

   !# Limit vertical computation by ktdia-1 levels
   integer           :: bkf_ktdia       = 1
   namelist /convection_cfgs/ bkf_ktdia

   !# Activate convective transport of species for deep and shallow bkf
   logical           :: bkf_lch1conv    = .false.
   namelist /convection_cfgs/ bkf_lch1conv

   !# Allow downdrafts in deep bkf
   logical           :: bkf_ldown       = .true.
   namelist /convection_cfgs/ bkf_ldown

   !# Force re-calculation of deep bkf at every timestep
   logical           :: bkf_lrefresh    = .false.
   namelist /convection_cfgs/ bkf_lrefresh

   !# Set convective timescales for deep and shallow
   logical           :: bkf_lsettadj    = .true.
   namelist /convection_cfgs/ bkf_lsettadj

   !# Activate shallow convective momentum transport
   logical           :: bkf_lshalm      = .false.
   namelist /convection_cfgs/ bkf_lshalm

   !# Deep bkf timescale (s) if bkf_lsettadj= .false.
   real              :: bkf_xtadjd      = 3600.
   namelist /convection_cfgs/ bkf_xtadjd

   !# Shallow bkf timescale (s)if bkf_lsettadj= .false.
   real              :: bkf_xtadjs      = 3600.
   namelist /convection_cfgs/ bkf_xtadjs

   !# Deep convection scheme name
   !# * 'NIL     ' :
   !# * 'SEC     ' :
   !# * 'OLDKUO  ' :
   !# * 'KFC     ' :
   !# * 'BECHTOLD' :
   character(len=16) :: deep            = 'nil'
   namelist /convection_cfgs/ deep
   character(len=*), parameter :: DEEP_OPT(13) = (/ &
        'NIL     ', 'SEC     ', 'MANABE  ', 'OLDKUO  ', 'FCP     ', &
        'KFC     ', 'KUOSTD  ', 'KUOSYM  ', 'KUOSUN  ', 'RAS     ', &
        'FCPKUO  ', 'KFCKUO2 ', 'BECHTOLD' &
        /)
!!$   character(len=*), parameter :: DEEP_OPT(5) = (/ &
!!$        'NIL     ', 'SEC     ', 'OLDKUO  ', &
!!$        'KFC     ', 'BECHTOLD' &
!!$        /)

   !# Minimum depth of conv. updraft for KFC  trigger (m)
   real              :: kfcdepth        = 4000.
   namelist /convection_cfgs/ kfcdepth

   !# Total forced detrainment in KFC scheme
   real              :: kfcdet          = 0.
   namelist /convection_cfgs/ kfcdet

   !# Init. level of forced detrainment in KFC scheme
   real              :: kfcdlev         = 0.5
   namelist /convection_cfgs/ kfcdlev

   !# generate wind tendencies in KFC or deep BKF if .true.
   logical           :: kfcmom          = .false.
   namelist /convection_cfgs/ kfcmom

   !# Compute production terms for Kain-Fritsch scheme
   logical           :: kfcprod         = .false.
   namelist /convection_cfgs/ kfcprod

   !# Initial convective updraft radius in KFC scheme(m)
   real              :: kfcrad          = 1500.
   namelist /convection_cfgs/ kfcrad

   !# Varies convective timescale as a function of CAPE for Kain-Fritsch scheme<br>
   !# KFCTAUCAPE = time1, time2, cmean, dcape
   !# * time1 (s): max kfctimec 
   !# * time2 (s): min kfctimec 
   !# * cmean (J/Kg): cape value at which kfctimec will be mean of time1 and time2 
   !# * dcape (J/Kg): decrease in kfctimec from time1 to time2 will occur over range cmean-dcape to cmean+dcape 
   real              :: kfctaucape(4)   = (/-1., -1., -1., -1./)
   namelist /convection_cfgs/ kfctaucape

   !# Time interval for refresh of tendencies in Kain-Fritsch scheme (s)  
   real              :: kfctimea        = 3600.
   namelist /convection_cfgs/ kfctimea

   !# Convective time scale in Kain-Fritsch(s)
   real              :: kfctimec        = 3600.
   namelist /convection_cfgs/ kfctimec

   !# Trigger parameter of Kain-Fritsch convection scheme (WKLCL).
   !# Trigger parameter will increase from kfctrig4(3) to kfctrig4(4) [m/s]
   !# between timestep kfctrig4(1) and timestep kfctrig4(2)
   real              :: kfctrig4(4)     = (/0., 0., 0.05, 0.05/)
   namelist /convection_cfgs/ kfctrig4

   !# Nominal resolution for which KFCTRIG4 is set.
   !# This is inactive if value <= 0.
   real              :: kfctriga        = -1.0
   namelist /convection_cfgs/ kfctriga

   !# Over land and lakes we keep the value set by the "ramp" above over sea water:
   !# * for |lat| >= TRIGLAT(2) we keep value set by the "ramp" KFCTRIG4
   !# * for |lat| <= TRIGLAT(1) we use the new value KFCTRIGL [m/s]
   !# * and linear interpolation in between TRIGLAT(1) and TRIGLAT(2)
   real              :: kfctrigl        = 0.05
   namelist /convection_cfgs/ kfctrigl

   !# Logical key for variation of the trigger function depending on latitude and land-sea-lake mask
   logical            :: kfctriglat      = .false.
   namelist /convection_cfgs/ kfctriglat

   !# Switch for shallow convection
   !# * 'NIL'
   !# * 'KTRSNT'
   !# * 'KTRSNT_MG'
   !# * 'BECHTOLD'
   character(len=16) :: shal            = 'nil'
   namelist /convection_cfgs/ shal
   character(len=*), parameter :: SHAL_OPT(4) = (/ &
        'KTRSNT_MG', 'KTRSNT', 'BECHTOLD', 'NIL' &
        /)

   !# Over land and lakes we keep the value set by the "ramp" above over sea water:
   !# * for |lat| >= TRIGLAT(2) we keep value set by the "ramp" KFCTRIG4
   !# * for |lat| <= TRIGLAT(1) we use the new value KFCTRIGL
   !# * and linear interpolation in between TRIGLAT(1) and TRIGLAT(2)
   real              :: triglat(2)      = 0.0
   namelist /convection_cfgs/ triglat

contains

   function cnv_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      F_istat = RMN_OK
      return
   end function cnv_options_init

end module cnv_options
