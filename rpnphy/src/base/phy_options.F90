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

module phy_options
   use phy_typedef, only: PHY_ERROR
   implicit none
   public
   save
   
   real,    parameter :: FACTDT            = 1.

   integer, parameter :: LEVMAX            = 200 !# NOMBRE MAXIMUM DE NIVEAUX POUR LA PHYSIQUE

   integer, parameter :: PHY_CTRL_NML_OK   = 1001
   integer, parameter :: PHY_CTRL_INI_OK   = 1003

   integer, parameter :: OPT_OPTIX_OLD     = 1
   integer, parameter :: OPT_OPTIX_NEW     = 2


   logical           :: chemistry    = .false.
   logical           :: climat       = .false.
   character(len=16) :: conv_shal    = 'NIL'
   character(len=16) :: convec       = 'NIL'
   integer           :: cw_rad       = 0
   integer           :: date(14)     = 0
   real              :: delt         = 0.
   character(len=4),  pointer :: dyninread_list_s(:) => NULL()
   logical           :: dynout       = .false.
   logical           :: impflx       = .false.
   logical           :: inincr       = .false.
   integer           :: ioptix       = OPT_OPTIX_OLD
   integer           :: kntrad       = 1
   logical           :: offline      = .false.
   character(len=1024) :: ozone_file_s = 'NIL'
   integer           :: phy_init_ctrl = PHY_ERROR
   logical           :: reduc        = .false.
   character(len=16) :: schmsol      = 'ISBA'
   logical           :: tdiaglim    = .false.
   integer           :: tlift        = 0
   logical           :: vstag        = .false.


   !# Time length (hours) for special time accumulated physics variables
   integer           :: acchr        = 0
   namelist /physics_cfgs/ acchr
   namelist /physics_cfgs_p/ acchr

   !# Turbulent kinetic energy advect. is active if .true.
   logical           :: advectke     = .false.
   namelist /physics_cfgs/ advectke
   namelist /physics_cfgs_p/ advectke

   !# Surface heat flux from oceans is active if .true.
   logical           :: chauf        = .true.
   namelist /physics_cfgs/ chauf
   namelist /physics_cfgs_p/ chauf

   !# Obtain estimate of surface wind gusts if .true.
   logical           :: diag_twind   = .false.
   namelist /physics_cfgs/ diag_twind
   namelist /physics_cfgs_p/ diag_twind

   !# Diffuse vertical motion if .true.
   logical           :: diffuw       = .false.
   namelist /physics_cfgs/ diffuw
   namelist /physics_cfgs_p/ diffuw

   !# Surface friction is active if .true.
   !# Uses Schuman-Newell lapse rate if .false.
   logical           :: drag         = .true.
   namelist /physics_cfgs/ drag
   namelist /physics_cfgs_p/ drag

   !# Minimal value for TKE in stable case (for 'CLEF')
   real              :: etrmin2      = 1.E-4
   namelist /physics_cfgs/ etrmin2
   namelist /physics_cfgs_p/ etrmin2

   !# Surface evaporation is active if .true.
   logical           :: evap         = .true.
   namelist /physics_cfgs/ evap
   namelist /physics_cfgs_p/ evap

   !# Boundary layer processes
   !# * 'NIL    ': no vertical diffusion
   !# * 'CLEF   ': non-cloudy boundary layer formulation
   !# * 'MOISTKE': cloudy boundary layer formulation
   !# * 'SURFACE':
   !# * 'PHYSIMP':
   character(len=16) :: fluvert      = 'NIL'
   namelist /physics_cfgs/ fluvert
   namelist /physics_cfgs_p/ fluvert
   character(len=*), parameter :: FLUVERT_OPT(5) = (/ &
        'NIL    ', 'SURFACE', 'PHYSIMP', 'CLEF   ', 'MOISTKE' &
        /)

   !# (MOISTKE only) Apply factor fnn_reduc 
   !# * .false.: everywhere 
   !# * .true.: over water only
   logical           :: fnn_mask     = .false.
   namelist /physics_cfgs/ fnn_mask
   namelist /physics_cfgs_p/ fnn_mask

   !# (MOISTKE only) Reduction factor (between 0. and 1.) to be applied to the 
   !# parameter FNN (turbulent flux enhancement due to boundary layer clouds) 
   real              :: fnn_reduc    = 1.
   namelist /physics_cfgs/ fnn_reduc
   namelist /physics_cfgs_p/ fnn_reduc

   !# (CLEF+CONRES only) Non-dimensional parameter (must be >= 1.) that controls 
   !# the value of the flux enhancement factor in CONRES 
   real              :: fnnmod       = 2.
   namelist /physics_cfgs/ fnnmod
   namelist /physics_cfgs_p/ fnnmod

   !# Use Fomichev radiation code if .true.
   logical           :: fomic        = .false.
   namelist /physics_cfgs/ fomic
   namelist /physics_cfgs_p/ fomic

   !# Gravity wave drag formulation
   !# * 'GWD86': gravity wave drag + low-level blocking
   !# * 'GWD95': gravity wave drag (1995 version)
   character(len=16) :: gwdrag       = 'NIL'
   namelist /physics_cfgs/ gwdrag
   namelist /physics_cfgs_p/ gwdrag
   character(len=*), parameter :: GWDRAG_OPT(3) = (/ &
        'NIL  ', 'GWD86', 'GWD95' &
        /)

   !# Number of times the 3-point filter will be applied to smooth the GW flux profiles 
   integer           :: hines_flux_filter = 0
   namelist /physics_cfgs/ hines_flux_filter
   namelist /physics_cfgs_p/ hines_flux_filter

   !# Consider heating from non-orog. drag if = 1
   integer           :: iheatcal     = 0
   namelist /physics_cfgs/ iheatcal
   namelist /physics_cfgs_p/ iheatcal

   !# Allow overwrite of agregated ilmo value by ctmdiag
   logical           :: ilmodiag     = .true.
   namelist /physics_cfgs/ ilmodiag
   namelist /physics_cfgs_p/ ilmodiag

   !# Comma-separated list of diagnostic level inputs to read.
   !# Default: indiag_list_s(1) = 'DEFAULT LIST', 
   !# expanded to: UU, VV, TT, HU + all dynamic Tracers
   character(len=32) :: indiag_list_s(128) = ' '
   namelist /physics_cfgs/ indiag_list_s

   !# Initialize water content and cloud fraction seen by radiation for time 0 if .true. 
   logical           :: inilwc       = .false.
   namelist /physics_cfgs/ inilwc
   namelist /physics_cfgs_p/ inilwc

   !# Update ozone climatology during the run
   logical           :: intozot      = .false.
   namelist /physics_cfgs/ intozot
   namelist /physics_cfgs_p/ intozot

   !# Time between full radiation calculation (units D,H,M,S,P)
   character(len=16) :: kntrad_S     = ''
   namelist /physics_cfgs/ kntrad_S
   namelist /physics_cfgs_p/ kntrad_S

   !# Compute ice fraction in KTRSNT_MG if .true.
   logical           :: kticefrac    = .true.
   namelist /physics_cfgs/ kticefrac
   namelist /physics_cfgs_p/ kticefrac

   !# Compute lightning diagnostics if .true. 
   !# (currently for Milbrandt-Yau microphysics only)
   logical           :: lightning_diag = .false.
   namelist /physics_cfgs/ lightning_diag
   namelist /physics_cfgs_p/ lightning_diag

   !# Add methane oxydation as source of humidity in the stratosphere if .true.
   logical           :: lmetox       = .false.
   namelist /physics_cfgs/ lmetox
   namelist /physics_cfgs_p/ lmetox

   !# Mixing length calc. scheme
   !# * 'BLAC62': mixing length calc. using Blackadar
   !# * 'BOUJO ': mixing length calc. using Bougeault
   character(len=16) :: longmel      = 'BLAC62'
   namelist /physics_cfgs/ longmel
   namelist /physics_cfgs_p/ longmel
   character(len=*), parameter :: LONGMEL_OPT(2) = (/ &
        'BLAC62', 'BOUJO ' &
        /)

   !# Time length (hours) for special time averaged physics variables
   integer           :: moyhr        = 0
   namelist /physics_cfgs/ moyhr
   namelist /physics_cfgs_p/ moyhr

   !# Number of frozen hydrometeor categories to use in the P3 microphysics 
   !# scheme (currently limited to <5)
   integer           :: mp_p3_ncat   = 1
   namelist /physics_cfgs/ mp_p3_ncat
   namelist /physics_cfgs_p/ mp_p3_ncat

   !# Switch for airmass type (1 = maritime, 2 = continental)
   integer           :: my_ccntype   = 1
   namelist /physics_cfgs/ my_ccntype
   namelist /physics_cfgs_p/ my_ccntype

   !# Double-moment for cloud (for 'mp_my' only)
   logical           :: my_dblmom_c  = .true.
   namelist /physics_cfgs/ my_dblmom_c
   namelist /physics_cfgs_p/ my_dblmom_c

   !# Double-moment for graupel (for 'mp_my' only)
   logical           :: my_dblmom_g  = .true.
   namelist /physics_cfgs/ my_dblmom_g
   namelist /physics_cfgs_p/ my_dblmom_g

   !# Double-moment for hail (for 'mp_my' only)
   logical           :: my_dblmom_h  = .true.
   namelist /physics_cfgs/ my_dblmom_h
   namelist /physics_cfgs_p/ my_dblmom_h

   !# Double-moment for ice (for 'mp_my' only)
   logical           :: my_dblmom_i  = .true.
   namelist /physics_cfgs/ my_dblmom_i
   namelist /physics_cfgs_p/ my_dblmom_i

   !# Double-moment for rain (for 'mp_my' only)
   logical           :: my_dblmom_r  = .true.
   namelist /physics_cfgs/ my_dblmom_r
   namelist /physics_cfgs_p/ my_dblmom_r

   !# Double-moment for snow (for 'mp_my' only)
   logical           :: my_dblmom_s  = .true.
   namelist /physics_cfgs/ my_dblmom_s
   namelist /physics_cfgs_p/ my_dblmom_s

   !# Compute MY Diagnostic fields if .true.
   logical           :: my_diagon    = .true.
   namelist /physics_cfgs/ my_diagon
   namelist /physics_cfgs_p/ my_diagon

   !# Ice-phase switched on if .true.
   logical           :: my_iceon     = .true.
   namelist /physics_cfgs/ my_iceon
   namelist /physics_cfgs_p/ my_iceon

   !# Initialize the number concentration for each category
   logical           :: my_initn     = .true.
   namelist /physics_cfgs/ my_initn
   namelist /physics_cfgs_p/ my_initn

   !# Autoconversion (cloud to rain) switched on
   logical           :: my_rainon    = .true.
   namelist /physics_cfgs/ my_rainon
   namelist /physics_cfgs_p/ my_rainon

   !# Sedimentation switched on
   logical           :: my_sedion    = .true.
   namelist /physics_cfgs/ my_sedion
   namelist /physics_cfgs_p/ my_sedion

   !# Snow initiation switched on
   logical           :: my_snowon    = .true.
   namelist /physics_cfgs/ my_snowon
   namelist /physics_cfgs_p/ my_snowon

   !# Parameters for three-component freezing term
   real              :: my_tc3comp(3)= (/0.,-10.,0./)
   namelist /physics_cfgs/ my_tc3comp
   namelist /physics_cfgs_p/ my_tc3comp

   !# Warm-phase switched on
   logical           :: my_warmon    = .true.
   namelist /physics_cfgs/ my_warmon
   namelist /physics_cfgs_p/ my_warmon

   !# Physic input blocking along X
   integer           :: ninblocx     = 1
   namelist /physics_cfgs/ ninblocx
   namelist /physics_cfgs_p/ ninblocx

   !# Physic input blocking along Y
   integer           :: ninblocy     = 1
   namelist /physics_cfgs/ ninblocy
   namelist /physics_cfgs_p/ ninblocy

   !# Hines non-orographic GWD scheme is active if .true.
   logical           :: non_oro      = .false.
   namelist /physics_cfgs/ non_oro
   namelist /physics_cfgs_p/ non_oro

   !# Pressure (in Pa) that defines the bottom emission level for gravity waves 
   real              :: non_oro_pbot = 61000.0
   namelist /physics_cfgs/ non_oro_pbot
   namelist /physics_cfgs_p/ non_oro_pbot

   !# Number of timesteps for which surface fluxes "FC" and "FV" are 
   !# gradually set from 0 to their full value in a "slow start fashion" 
   !# at the beginning of a time integration
   integer           :: nsloflux     = 0
   namelist /physics_cfgs/ nsloflux
   namelist /physics_cfgs_p/ nsloflux

   !# Vectoc lenght physics memory space folding for openMP
   integer           :: p_runlgt     = -1
   namelist /physics_cfgs/ p_runlgt
   namelist /physics_cfgs_p/ p_runlgt

   !# Run with a coupled high vertical resolution boundary layer (PBL).
   logical           :: pbl_coupled  = .false.
   namelist /physics_cfgs/ pbl_coupled
   namelist /physics_cfgs_p/ pbl_coupled

   !# Run with a modified closure for the dissipation length scale
   !# * 'NIL'
   !# * 'LIM50'
   character(len=16) :: pbl_diss     = 'NIL'
   namelist /physics_cfgs/ pbl_diss
   namelist /physics_cfgs_p/ pbl_diss
   character(len=*), parameter :: PBL_DISS_OPT(2) = (/ &
        'LIM50', 'NIL  ' &
        /)

   !# Limit the vertical extent of the PBL scheme to below the model level listed here
   integer           :: pbl_ktop     = 1
   namelist /physics_cfgs/ pbl_ktop
   namelist /physics_cfgs_p/ pbl_ktop

   !# Use the mixing length to average the Richardson number profile of (potentially) 
   !# many layers to derive a "background" Ri estimate
   logical           :: pbl_ribkg    = .false.
   namelist /physics_cfgs/ pbl_ribkg
   namelist /physics_cfgs_p/ pbl_ribkg

   !# Richardson num. critical values for hysteresis
   real              :: pbl_ricrit(2)= 1.
   namelist /physics_cfgs/ pbl_ricrit
   namelist /physics_cfgs_p/ pbl_ricrit

   !# 
   !# * 'NIL     ':
   !# * 'CONRES  ':
   !# * 'SHALOW  ':
   !# * 'SHALODQC':
   !# * 'GELEYN  ':
   character(len=16) :: pbl_shal     = 'NIL'
   namelist /physics_cfgs/ pbl_shal
   namelist /physics_cfgs_p/ pbl_shal
   character(len=*), parameter :: PBL_SHAL_OPT(5) = (/ &
        'SHALODQC', 'SHALOW', 'CONRES', 'GELEYN', 'NIL' &
        /)

   !# Time splitting (number of steps per model step) for the coupled PBL scheme
   integer           :: pbl_tsplit   = 1
   namelist /physics_cfgs/ pbl_tsplit
   namelist /physics_cfgs_p/ pbl_tsplit

   !# Use of 2*dt in KTE calculation of moistke
   logical           :: pbl_tkediff2dt = .false.
   namelist /physics_cfgs/ pbl_tkediff2dt
   namelist /physics_cfgs_p/ pbl_tkediff2dt

   !# Relaxation timescale (s) for mixing length smoothing
   real              :: pbl_zntau    = 7200.
   namelist /physics_cfgs/ pbl_zntau
   namelist /physics_cfgs_p/ pbl_zntau

   !# Number of layers to split model layers into for the high vertical 
   !# resolution coupled PBL
   integer           :: pbl_zsplit   = 1
   namelist /physics_cfgs/ pbl_zsplit
   namelist /physics_cfgs_p/ pbl_zsplit

   !# Scheme to determine precipitation type
   !# * 'NIL     ': no call to bourge
   !# * 'BOURGE  ': use Bourgouin algorithm (bourge1) to determine precip. types.
   !# * 'BOURGE3D':
   character(len=16) :: pcptype      = 'NIL'
   namelist /physics_cfgs/ pcptype
   namelist /physics_cfgs_p/ pcptype
   character(len=*), parameter :: PCPTYPE_OPT(3) = (/ &
        'NIL     ', 'BOURGE  ', 'BOURGE3D' &
        /)

   !# Use double presision for physic statistics output
   logical           :: phystat_dble_l = .false.
   namelist /physics_cfgs/ phystat_dble_l
   namelist /physics_cfgs_p/ phystat_dble_l

   !# Physic statistics output for 3d varables:
   !# * .false. : mean, var, min and max for the whole 3d fiels
   !# * .true.  : mean, var, min and max are done for each levels independently
   logical           :: phystat_2d_l = .false.
   namelist /physics_cfgs/ phystat_2d_l
   namelist /physics_cfgs_p/ phystat_2d_l

   !# Physic statistics output Frequency 
   character(len=16) :: phystat_freq_S = '0h'
   namelist /physics_cfgs/ phystat_freq_S
   namelist /physics_cfgs_p/ phystat_freq_S

   !# Physic statistics output: bus variable list that should be included in physics 
   !# "block" stats. Possible values:
   !# * Long varnames
   !# * Short varnames
   !# * 'ALLVARS=EDPV': all variables from E, D, P, V buses (any combination of the 4 letters); 
   character(len=32) :: phystat_list_s(1024) = ' '
   namelist /physics_cfgs/ phystat_list_s
!!$   namelist /physics_cfgs_p/ phystat_list_s

   !# CFC11 bckgrnd atmospheric concentration (PPMV)
   real              :: qcfc11       = 0.280
   namelist /physics_cfgs/ qcfc11
   namelist /physics_cfgs_p/ qcfc11

   !# CFC12 bckgrnd atmospheric concentration (PPMV)
   real              :: qcfc12       = 0.530
   namelist /physics_cfgs/ qcfc12
   namelist /physics_cfgs_p/ qcfc12

   !# CH4 bckgrnd atmospheric concentration (PPMV)
   real              :: qch4         = 1.783
   namelist /physics_cfgs/ qch4
   namelist /physics_cfgs_p/ qch4

   !# CO2 bckgrnd atmospheric concentration (PPMV)
   real              :: qco2         = 380.
   namelist /physics_cfgs/ qco2
   namelist /physics_cfgs_p/ qco2

   !# N2O bckgrnd atmospheric concentration (PPMV)
   real              :: qn2o         = 0.3186
   namelist /physics_cfgs/ qn2o
   namelist /physics_cfgs_p/ qn2o

   !# format of radiation files to be read
   !# * 'STD': RPN standard file
   !# * 'UNF': unformatted
   character(len=16) :: radfiles     = 'STD'
   namelist /physics_cfgs/ radfiles
   namelist /physics_cfgs_p/ radfiles
   character(len=*), parameter :: RADFILES_OPT(2) = (/ &
        'STD', 'UNF' &
        /)

   !# Radiation fixes near the model top(for newrad only) if .true.
   logical           :: radfix       = .true.
   namelist /physics_cfgs/ radfix
   namelist /physics_cfgs_p/ radfix

   !# Vertical smoothing on radiative fluxes(for newrad only) if .true.
   logical           :: radfltr      = .true.
   namelist /physics_cfgs/ radfltr
   namelist /physics_cfgs_p/ radfltr

   !# Radiation scheme
   !# * 'NIL      ': no radiation scheme
   !# * 'OLDRAD   ': simple radiation scheme
   !# * 'NEWRAD   ': complete radiation scheme
   !# * 'CCCMARAD ': most advanced radiation scheme
   !# * 'CCCMARAD2': most advanced radiation scheme v2
   character(len=16) :: radia        = 'NIL'
   namelist /physics_cfgs/ radia
   namelist /physics_cfgs_p/ radia
   character(len=*), parameter :: RADIA_OPT(5) = (/ &
        'NIL      ', 'OLDRAD   ', 'NEWRAD   ', 'CCCMARAD ', 'CCCMARAD2' &
        /)

   !# List of levels on which IR and VIS radiation calculations are 
   !# performed (to save on CPU time) (for newrad only)
   integer           :: radnivl(LEVMAX+1) = 0
   namelist /physics_cfgs/ radnivl
   
   !# Key for activation of the radiation along slopes
   logical           :: radslope     = .false.
   namelist /physics_cfgs/ radslope
   namelist /physics_cfgs_p/ radslope

   !# Additional output for low level refraction
   logical           :: refract      = .false.
   namelist /physics_cfgs/ refract
   namelist /physics_cfgs_p/ refract

   !# Launching level value of GW RMS wind (m/s) from non-orographic origin
   real              :: rmscon       = 1.0
   namelist /physics_cfgs/ rmscon
   namelist /physics_cfgs_p/ rmscon

   !# water/ice phase for saturation calc. if .true.; 
   !# water phase only for saturation calc. if .false.
   logical           :: satuco       = .true.
   namelist /physics_cfgs/ satuco
   namelist /physics_cfgs_p/ satuco

   !# Sets the minimum value of the drag coefficient in the orographic 
   !# blocking scheme.
   real              :: sgo_cdmin    = 1.0
   namelist /physics_cfgs/ sgo_cdmin
   namelist /physics_cfgs_p/ sgo_cdmin

   !# Turns on/off the non-linear amplification factor (depending on wind 
   !# direction) of the drag coefficient in the orographic blocking scheme
   logical           :: sgo_nldirfac = .true.
   namelist /physics_cfgs/ sgo_nldirfac
   namelist /physics_cfgs_p/ sgo_nldirfac

   !# Turns on/off the amplification factor (due to stability) of the drag 
   !# coefficient in the orographic blocking scheme
   logical           :: sgo_stabfac  = .true.
   namelist /physics_cfgs/ sgo_stabfac
   namelist /physics_cfgs_p/ sgo_stabfac

   !# Run ISCCP cloud simulator (cccmarad only) if .true.
   logical           :: simisccp     = .false.
   namelist /physics_cfgs/ simisccp
   namelist /physics_cfgs_p/ simisccp

   !# Condensation scheme name
   !# * 'NIL       ' : No explicit condensation scheme used
   !# * 'CONDS     ' :
   !# * 'CONSUN    ' :
   !# * 'NEWSUND   ' :
   !# * 'MP_MY2_OLD' :
   !# * 'MP_MY2    ' :
   !# * 'MP_P3     ' :
   character(len=16) :: stcond       = 'NIL'
   namelist /physics_cfgs/ stcond
   namelist /physics_cfgs_p/ stcond
   character(len=*), parameter :: STCOND_OPT(7) = (/ &
        'NEWSUND   ', 'CONSUN    ', 'CONDS     ', 'MP_MY2_OLD',   &
        'MP_MY2    ', 'MP_P3     ', 'NIL       ' &
        /)

   !# Special treatment of stratosphere; 
   !# if .true. ignore convection/condensation tendencies where pressure is lower 
   !# than topc or specific humidity is lower than minq as specified in nocld.cdk 
   logical           :: stratos      = .false.
   namelist /physics_cfgs/ stratos
   namelist /physics_cfgs_p/ stratos

   !# Factor used in the gwd formulation = 1/(LENGTH SCALE)
   real              :: taufac       = 8.E-6
   namelist /physics_cfgs/ taufac
   namelist /physics_cfgs_p/ taufac

   !# Run the physics in test harness mode
   logical           :: test_phy     = .false.
   namelist /physics_cfgs/ test_phy
   namelist /physics_cfgs_p/ test_phy

   !# Use correct vertical levels (thermo) and offset (none) for 
   !# Bougeault-Lacarrere mixing length calculation
   logical           :: tmp_boujo_height_corr = .false.
   namelist /physics_cfgs/ tmp_boujo_height_corr
   namelist /physics_cfgs_p/ tmp_boujo_height_corr

   !# (newrad only) Use TT(12000) instead of skin temp in downward IR 
   !# flux calculation if .true.
   logical           :: ts_flxir     = .false.
   namelist /physics_cfgs/ ts_flxir
   namelist /physics_cfgs_p/ ts_flxir


contains

   function phy_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      indiag_list_s(1) = 'DEFAULT LIST'
      return
   end function phy_options_init

end module phy_options
