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
module gem_options
   implicit none
   public
   save

   integer, parameter :: MAXHLEV = 1024
   integer, parameter :: IAU_MAX_TRACERS = 250
   integer, parameter :: MAX_BLACKLIST = 250
   integer, parameter :: MAXELEM_mod = 60
   integer,parameter  :: STAT_MAXN = 500

   !# SL off-centering parameter for hydrostatic
   real*8 :: Cstv_bA_8 = 0.6
   namelist /gem_cfgs  / Cstv_bA_8
   namelist /gem_cfgs_p/ Cstv_bA_8

   !# SL off-centering parameter for the momentum equations
   real*8 :: Cstv_bA_m_8 = 0.6
   namelist /gem_cfgs  / Cstv_bA_m_8      
   namelist /gem_cfgs_p/ Cstv_bA_m_8

   !# SL off-centering parameter for nonhydrostatic
   real*8 :: Cstv_bA_nh_8 = 0.5
   namelist /gem_cfgs  / Cstv_bA_nh_8      
   namelist /gem_cfgs_p/ Cstv_bA_nh_8

   !# T* basic state temperature (K) 
   real*8 :: Cstv_Tstr_8 = 240.0
   namelist /gem_cfgs  / Cstv_tstr_8
   namelist /gem_cfgs_p/ Cstv_tstr_8

   !# Parameter controlling modified epsilon (Ver_epsi_8) [nonhydrostatic part]
   real*8 :: Cstv_rE_8 = 1.d0
   namelist /gem_cfgs  / Cstv_rE_8   
   namelist /gem_cfgs_p/ Cstv_rE_8

   !# another reference pressure
   real*8 :: Cstv_pSref_8 = 1.d5
   namelist /gem_cfgs  / Cstv_pSref_8
   namelist /gem_cfgs_p/ Cstv_pSref_8

   !# Fraction of adjustment to be given to the ocean
   real*8 :: Cstv_psadj_8 = 1.d0
   namelist /gem_cfgs  / Cstv_psadj_8
   namelist /gem_cfgs_p/ Cstv_psadj_8

   !# number of points for the halo on X
   integer :: G_halox = 4
   namelist /gem_cfgs  / G_halox
   namelist /gem_cfgs_p/ G_halox

   !# number of points for the halo on Y
   integer :: G_haloy = 4
   namelist /gem_cfgs  / G_haloy
   namelist /gem_cfgs_p/ G_haloy

   !# Heap memory will be painted ti NaN using an array wrk01(G_ni,G_nj,Heap_nk)
   integer :: Heap_nk = -1
   namelist /gem_cfgs  / Heap_nk
   namelist /gem_cfgs_p/ Heap_nk

   !# array of model levels ,  0.0 < HYB < 1.0 
   real, dimension(MAXHLEV) :: hyb = -1
   namelist /gem_cfgs/ Hyb

   !# pair of coefficients (min,max) to control the flattenning of the 
   !# vertical coordinate 
   real, dimension(2):: Hyb_rcoef = (/1., 1./)
   namelist /gem_cfgs  / Hyb_rcoef
   namelist /gem_cfgs_p/ Hyb_rcoef

   !# Horizontal diffusion if activated, will be applied to the following
   !# variables  : Horizontal winds, ZDot, W, tracers (_tr var)

   !# Compute horizontal diffusion within RHS
   logical :: Hzd_in_rhs_L = .false.
   namelist /gem_cfgs  / Hzd_in_rhs_L
   namelist /gem_cfgs_p/ Hzd_in_rhs_L

   !# Background 2 delta-x removal ratio - range(0.0-1.0)
   real :: Hzd_lnR = -1.
   namelist /gem_cfgs  / Hzd_lnr
   namelist /gem_cfgs_p/ Hzd_lnr

   !# Theta 2 delta-x removal ratio - range(0.0-1.0). 
   real :: Hzd_lnr_theta = -1.
   namelist /gem_cfgs  / Hzd_lnr_theta
   namelist /gem_cfgs_p/ Hzd_lnr_theta

   !# Tracers 2 delta-x removal ratio - range(0.0-1.0)
   real :: Hzd_lnR_tr = -1.
   namelist /gem_cfgs  / Hzd_lnr_tr
   namelist /gem_cfgs_p/ Hzd_lnr_tr

   !# Order of the background diffusion operator
   !# 2, 4, 6, 8
   integer :: Hzd_pwr = -1
   namelist /gem_cfgs  / Hzd_pwr
   namelist /gem_cfgs_p/ Hzd_pwr

   !# Order of the background diffusion operator on theta
   !# 2, 4, 6, 8 
   integer :: Hzd_pwr_theta = -1.
   namelist /gem_cfgs  / Hzd_pwr_theta
   namelist /gem_cfgs_p/ Hzd_pwr_theta

   !# Order of the background diffusion operator on tracers
   integer :: Hzd_pwr_tr = -1.
   namelist /gem_cfgs  / Hzd_pwr_tr
   namelist /gem_cfgs_p/ Hzd_pwr_tr

   !# Fraction of the maximum divergence damping - range(0.0-1.0)
   integer :: Hzd_div_damp = -1.
   namelist /gem_cfgs  / Hzd_div_damp
   namelist /gem_cfgs_p/ Hzd_div_damp

   !# Main Smagorinsky control parameter
   real :: Hzd_smago_param= -1.
   namelist /gem_cfgs  / Hzd_smago_param
   namelist /gem_cfgs_p/ Hzd_smago_param

   !# Apply Smago diffusion on theta using Hzd_smago_param/Hzd_smago_prandtl parameter
   real :: Hzd_smago_prandtl = -1.
   namelist /gem_cfgs  / Hzd_smago_prandtl
   namelist /gem_cfgs_p/ Hzd_smago_prandtl

   !# Apply Smago diffusion on HU using Hzd_smago_param/Hzd_smago_prandtl_hu parameter
   real :: Hzd_smago_prandtl_hu = -1.
   namelist /gem_cfgs  / Hzd_smago_prandtl_hu
   namelist /gem_cfgs_p/ Hzd_smago_prandtl_hu

   !# 
   real :: Hzd_smago_lnr = 0.
   namelist /gem_cfgs  / Hzd_smago_lnr
   namelist /gem_cfgs_p/ Hzd_smago_lnr
   !# 
   real :: Hzd_smago_min_lnr = -1.
   namelist /gem_cfgs  / Hzd_smago_min_lnr
   namelist /gem_cfgs_p/ Hzd_smago_min_lnr
   !# 
   real :: Hzd_smago_bot_lev = 0.7
   namelist /gem_cfgs  / Hzd_smago_bot_lev
   namelist /gem_cfgs_p/ Hzd_smago_bot_lev
   !# 
   real :: Hzd_smago_top_lev = 0.4
   namelist /gem_cfgs  / Hzd_smago_top_lev
   namelist /gem_cfgs_p/ Hzd_smago_top_lev
   !# 
   logical :: Hzd_smago_theta_nobase_L = .false.
   namelist /gem_cfgs  / Hzd_smago_theta_nobase_L
   namelist /gem_cfgs_p/ Hzd_smago_theta_nobase_L
   !# 
   real :: Hzd_smago_fric_heat = 0.
   namelist /gem_cfgs  / Hzd_smago_fric_heat
   namelist /gem_cfgs_p/ Hzd_smago_fric_heat

   !# Coefficients that multiply KM to simulate sponge layer near the top 
   !# of the model. Warning! if this parameter is used, the EPONGE in the 
   !# physics namelist should be removed.
   real, dimension(1000) :: Eq_sponge = 0.
   namelist /gem_cfgs/ Eq_sponge

   !# Latitudinal ramping of equatorial sponge
   logical :: Eq_ramp_L = .false.
   namelist /gem_cfgs  / Eq_ramp_L
   namelist /gem_cfgs_p/ Eq_ramp_L

   !# The following variables are to help compute latitudinal modulation of
   !# vertical diffusion coefficient on momentum by fitting a cubic btwn
   !# values P_lmvd_weigh_low_lat and P_lmvd_weigh_high_lat at latitudes 
   !# P_lmvd_low_lat and P_lmvd_high_lat

   !# Multiplication factor of P_pbl_spng at latitude P_lmvd_high_lat
   real :: P_lmvd_weigh_high_lat = 1.0
   namelist /gem_cfgs  / P_lmvd_weigh_high_lat
   namelist /gem_cfgs_p/ P_lmvd_weigh_high_lat

   !# Multiplication factor of P_pbl_spng at latitude P_lmvd_low_lat
   real :: P_lmvd_weigh_low_lat = 1.0
   namelist /gem_cfgs  / P_lmvd_weigh_low_lat
   namelist /gem_cfgs_p/ P_lmvd_weigh_low_lat

   !# Latitude at which the multiplication factor becomes P_lmvd_weigh_high_lat
   real :: P_lmvd_high_lat = 30.0
   namelist /gem_cfgs  / P_lmvd_high_lat
   namelist /gem_cfgs_p/ P_lmvd_high_lat

   !# latitude at which the multiplication factor becomes P_lmvd_weigh_low_lat
   real :: P_lmvd_low_lat =  5.0
   namelist /gem_cfgs  / P_lmvd_low_lat
   namelist /gem_cfgs_p/ P_lmvd_low_lat

   !Iau

   !# Filter cutoff period for Iau_weight_S='sin' in hours
   real :: Iau_cutoff    = 6.
   namelist /gem_cfgs/ Iau_cutoff
   namelist /gem_cfgs_p/ Iau_cutoff

   !# The number of seconds between increment fields
   real :: Iau_interval = -1.
   namelist /gem_cfgs/ Iau_interval
   namelist /gem_cfgs_p/ Iau_interval

   !# The number of seconds over which IAU will be  will be run 
   !# (typically the length of the assimilation window).  
   !# Default < 0 means that no IAUs are applied.
   real :: Iau_period = -1.
   namelist /gem_cfgs/ Iau_period
   namelist /gem_cfgs_p/ Iau_period

   !# An optional list of tracers to be incremented.
   character(len=4), dimension(IAU_MAX_TRACERS) :: Iau_tracers_S = ' '
   namelist /gem_cfgs/ Iau_tracers_S

   !# The type of weighting function to be applied to the analysis increments:
   !# * 'constant' (default) uniform increments
   !# * 'sin' DF-style weights (Fillion et al. 1995)
   character(len=64) :: Iau_weight_S = 'constant'
   namelist /gem_cfgs/ Iau_weight_S
   namelist /gem_cfgs_p/ Iau_weight_S

   !# Input PE blocking along npex
   integer :: Iau_ninblocx = 1
   namelist /gem_cfgs/ Iau_ninblocx
   namelist /gem_cfgs_p/ Iau_ninblocx

   !# Input PE blocking along npey
   integer :: Iau_ninblocy = 1
   namelist /gem_cfgs/ Iau_ninblocy
   namelist /gem_cfgs_p/ Iau_ninblocy

!Init

   !# true -> Digital filter initialization is performed
   logical :: Init_balgm_L   = .false.
   namelist /gem_cfgs  / Init_balgm_L
   namelist /gem_cfgs_p/ Init_balgm_L

   !# true -> Windowing is applied
   logical :: Init_dfwin_L   = .true.
   namelist /gem_cfgs  / Init_dfwin_L
   namelist /gem_cfgs_p/ Init_dfwin_L

   !# number of points for digital filter (equals the number of timesteps +1)
   character(len=16) :: Init_dflength_S = '5p'
   namelist /gem_cfgs  / Init_dflength_S
   namelist /gem_cfgs_p/ Init_dflength_S

   !# period limit of digital filter units D,H,M,S
   character(len=16) :: Init_dfpl_S = '6h'
   namelist /gem_cfgs/ Init_dfpl_S
   namelist /gem_cfgs_p/ Init_dfpl_S

   !# * true -> passive tracers digitally filtered
   !# * false-> passive tracers set to result obtained at mid-period during initialization period (no filtering)
   logical :: Init_dftr_L = .false.
   namelist /gem_cfgs  / Init_dftr_L
   namelist /gem_cfgs_p/ Init_dftr_L

   !# Number of PEs to use for input
   integer :: Inp_npes  = 1
   namelist /gem_cfgs  / Inp_npes
   namelist /gem_cfgs_p/ Inp_npes

   !# List of variables to NOT process during input
   character(len=4), dimension(MAX_BLACKLIST) :: Inp_blacklist_S = ' '
   namelist /gem_cfgs/ Inp_blacklist_S

   !# Type of vertical interpolation scheme
   character(len=8) :: Inp_vertintype_tracers_S = 'cubic'
   namelist /gem_cfgs  / Inp_vertintype_tracers_S
   namelist /gem_cfgs_p/ Inp_vertintype_tracers_S

   !Lam

   !# Number of levels for top piloting
   integer :: Lam_gbpil_T = -1
   namelist /gem_cfgs  / Lam_gbpil_T
   namelist /gem_cfgs_p/ Lam_gbpil_T

   !# Number of points for horizontal blending
   integer :: Lam_blend_H = 10
   namelist /gem_cfgs  / Lam_blend_H
   namelist /gem_cfgs_p/ Lam_blend_H

   !# Number of levels for top blending
   integer :: Lam_blend_T = 0
   namelist /gem_cfgs  / Lam_blend_T
   namelist /gem_cfgs_p/ Lam_blend_T

   !# True-> for blending to zero the physics tendency in blending area
   logical :: Lam_0ptend_L = .true.
   namelist /gem_cfgs  / Lam_0ptend_L
   namelist /gem_cfgs_p/ Lam_0ptend_L

   !# True-> to force constant (fixed) boundary conditions
   logical :: Lam_ctebcs_L = .false.
   namelist /gem_cfgs  / Lam_ctebcs_L
   namelist /gem_cfgs_p/ Lam_ctebcs_L

   !# Type of horizontal interpolation to model grid 
   !# * 'CUB_LAG'
   !# * 'LINEAR'
   !# * 'NEAREST'
   character(len=16) :: Lam_hint_S = 'CUB_LAG'
   namelist /gem_cfgs  / Lam_hint_S
   namelist /gem_cfgs_p/ Lam_hint_S

   !# True-> The plane of the top temperature layer is completely 
   !# overwritten from the 2D pilot data
   logical :: Lam_toptt_L = .false.
   namelist /gem_cfgs  / Lam_toptt_L
   namelist /gem_cfgs_p/ Lam_toptt_L

   !# True-> to blend the model topography with the pilot topography
   logical :: Lam_blendoro_L = .true.
   namelist /gem_cfgs  / Lam_blendoro_L
   namelist /gem_cfgs_p/ Lam_blendoro_L

   !# True->to print more information to std output
   logical :: Lctl_debug_L = .false.
   namelist /gem_cfgs  / Lctl_debug_L
   namelist /gem_cfgs_p/ Lctl_debug_L

   !# precistion in print glbstats
   !# * 'LCL_4'
   !# * 'LCL_8'
   character(len=6) :: Lctl_rxstat_S = 'LCL_4'
   namelist /gem_cfgs  / Lctl_rxstat_S
   namelist /gem_cfgs_p/ Lctl_rxstat_S

   !Out3

   !# True-> to clip humidity variables on output 
   logical :: Out3_cliph_L = .false.
   namelist /gem_cfgs  / Out3_cliph_L
   namelist /gem_cfgs_p/ Out3_cliph_L

   !# Interval of output file name change
   character(len=16) :: Out3_close_interval_S = ' '
   namelist /gem_cfgs  / Out3_close_interval_S
   namelist /gem_cfgs_p/ Out3_close_interval_S

   !# 'etiket' used for output fields
   character(len=12) :: Out3_etik_S = 'GEMDM'
   namelist /gem_cfgs  / Out3_etik_S
   namelist /gem_cfgs_p/ Out3_etik_S

   !# Vertical interpolation scheme for output
   character(len=12) :: Out3_vinterp_type_S = 'linear'
   namelist /gem_cfgs  / Out3_vinterp_type_S
   namelist /gem_cfgs_p/ Out3_vinterp_type_S

   !# Default value for IP3 is 0, -1 for IP3 to contain step number, 
   !# >0 for given IP3
   integer :: Out3_ip3 = 0
   namelist /gem_cfgs  / Out3_ip3
   namelist /gem_cfgs_p/ Out3_ip3

   !# Number of layers close to the bottom of the model within which a 
   !# linear interpolation of GZ will be performed
   integer :: Out3_linbot = 0
   namelist /gem_cfgs  / Out3_linbot
   namelist /gem_cfgs_p/ Out3_linbot

   !# Packing factor used for all variables except for those defined in 
   !# Out_xnbits_s
   integer :: Out3_nbitg = 16
   namelist /gem_cfgs  / Out3_nbitg
   namelist /gem_cfgs_p/ Out3_nbitg

   !# Minimum of digits used to represent output units
   integer :: Out3_ndigits = 3
   namelist /gem_cfgs  / Out3_ndigits
   namelist /gem_cfgs_p/ Out3_ndigits

   !# List of levels for underground extrapolation
   real, dimension(MAXELEM_mod) :: Out3_lieb_levels = 0.
   namelist /gem_cfgs/ Out3_lieb_levels

   !# Maximum number of iterations for the Liebman procedure
   integer :: Out3_lieb_maxite = 100
   namelist /gem_cfgs  / Out3_lieb_maxite
   namelist /gem_cfgs_p/ Out3_lieb_maxite

   !# number of iterations to exchange halo for the Liebman procedure
   integer :: Out3_liebxch_iter = 4
   namelist /gem_cfgs  / Out3_liebxch_iter
   namelist /gem_cfgs_p/ Out3_liebxch_iter

   !# Precision criteria for the Liebman procedure
   real :: Out3_lieb_conv = 0.1
   namelist /gem_cfgs  / Out3_lieb_conv
   namelist /gem_cfgs_p/ Out3_lieb_conv

   !# Sortie jobs lauched every Out3_postproc_fact*Out3_close_interval_S
   integer :: Out3_postproc_fact = 0
   namelist /gem_cfgs  / Out3_postproc_fact
   namelist /gem_cfgs_p/ Out3_postproc_fact

   !# Total number of PEs for output using MFV collector
   integer :: Out3_npes = 1
   namelist /gem_cfgs/ Out3_npes
   namelist /gem_cfgs_p/ Out3_npes

   !# Total number of PEs along npex for output using MID collector
   integer :: Out3_npex = -1
   namelist /gem_cfgs  / Out3_npex
   namelist /gem_cfgs_p/ Out3_npex

   !# Total number of PEs along npey for output using MID collector
   integer :: Out3_npey = -1
   namelist /gem_cfgs  / Out3_npey
   namelist /gem_cfgs_p/ Out3_npey

   !# Number of bits to perturb on initial conditions
   integer :: perturb_nbits = 0
   namelist /gem_cfgs  / perturb_nbits
   namelist /gem_cfgs_p/ perturb_nbits

   !# Stride for perturbation on initial conditions
   integer :: perturb_npts = 10
   namelist /gem_cfgs  / perturb_npts
   namelist /gem_cfgs_p/ perturb_npts

   !Schm

   !# * True-> cubic interpolation
   !# * False-> linear interpolation
   logical :: Schm_adcub_L = .true.
   namelist /gem_cfgs  / Schm_adcub_L
   namelist /gem_cfgs_p/ Schm_adcub_L

   !# True-> auto barotropic option
   logical :: Schm_autobar_L = .false.
   namelist /gem_cfgs  / Schm_autobar_L
   namelist /gem_cfgs_p/ Schm_autobar_L

   !# * True-> hydrostatic
   !# * False-> non-hydrostatic
   logical :: Schm_hydro_L = .false.
   namelist /gem_cfgs  / Schm_hydro_L
   namelist /gem_cfgs_p/ Schm_hydro_L

   !# True-> horizontal diffusion of momentum at each CN iteration
   logical :: Schm_hzdadw_L = .false.
   namelist /gem_cfgs  / Schm_hzdadw_L
   namelist /gem_cfgs_p/ Schm_hzdadw_L

   !# Number of iterations for Crank-Nicholson
   integer :: Schm_itcn = 2
   namelist /gem_cfgs  / Schm_itcn
   namelist /gem_cfgs_p/ Schm_itcn

   !# Number of iterations to solve non-linear Helmholtz problem
   integer :: Schm_itnlh  = 2
   namelist /gem_cfgs  / Schm_itnlh
   namelist /gem_cfgs_p/ Schm_itnlh

   !# Number of iterations to compute trajectories
   integer :: Schm_itraj = 2
   namelist /gem_cfgs  / Schm_itraj
   namelist /gem_cfgs_p/ Schm_itraj

   !# *  -1: no blending between Yin and Yang
   !# *   0: blending at init only
   !# * > 0: blending at every nblendyy timestep
   integer :: Schm_nblendyy = -1
   namelist /gem_cfgs  / Schm_nblendyy
   namelist /gem_cfgs_p/ Schm_nblendyy

   !# * 0 -> No conservation of surface pressure
   !# * 1 -> Conservation of Total air mass Pressure
   !# * 2 -> Conservation of Dry air mass Pressure with forced Schm_source_ps_L=.true.
   integer :: Schm_psadj = 0
   namelist /gem_cfgs  / Schm_psadj
   namelist /gem_cfgs_p/ Schm_psadj

   !# correction at each timestep due to sources and sinks of specific humidity
   logical :: Schm_source_ps_L = .false.
   namelist /gem_cfgs  / Schm_source_ps_L
   namelist /gem_cfgs_p/ Schm_source_ps_L

   !# True-> print dry/wet air masses
   logical :: Schm_psadj_print_L = .false.
   namelist /gem_cfgs  / Schm_psadj_print_L
   namelist /gem_cfgs_p/ Schm_psadj_print_L

   !# Confirmation to use psadj with a LAM configuration
   logical :: Schm_psadj_lam_L = .false.
   namelist /gem_cfgs  / Schm_psadj_lam_L
   namelist /gem_cfgs_p/ Schm_psadj_lam_L

   !# True-> 
   logical :: Schm_dry_mixing_ratio_L = .false.
   namelist /gem_cfgs  / Schm_dry_mixing_ratio_L
   namelist /gem_cfgs_P/ Schm_dry_mixing_ratio_L

   !# True-> use SLEVE vertical coordinate
   logical :: Schm_sleve_L = .false.
   namelist /gem_cfgs  / Schm_sleve_L
   namelist /gem_cfgs_P/ Schm_sleve_L

   !# True-> averaging B and C in SLEVE scheme
   logical :: Schm_bcavg_L = .true.
   namelist /gem_cfgs  / Schm_bcavg_L
   namelist /gem_cfgs_P/ Schm_bcavg_L

   !# True-> to use topography
   logical :: Schm_Topo_L = .true.
   namelist /gem_cfgs  / Schm_Topo_L
   namelist /gem_cfgs_p/ Schm_Topo_L

   !# True-> variable cappa in thermodynamic equation
   logical :: Schm_capa_var_L = .false.
   namelist /gem_cfgs  / Schm_capa_var_L
   namelist /gem_cfgs_p/ Schm_capa_var_L

   !# * 0   ->          NO advection
   !# * 1   -> traditional advection
   !# * 2   -> consistant advection with respect to off-centering
   integer :: Schm_advec = 1
   namelist /gem_cfgs  / Schm_advec
   namelist /gem_cfgs_p/ Schm_advec

   !# True-> Eulerian treatment of mountains in the continuity equation 
   logical :: Schm_eulmtn_L = .false.
   namelist /gem_cfgs  / Schm_eulmtn_L
   namelist /gem_cfgs_p/ Schm_eulmtn_L

   !# True-> Modify slightly code behaviour to ensure bitpattern 
   !# reproduction in restart mode using FST file
   logical :: Schm_bitpattern_L = .false.
   namelist /gem_cfgs/ Schm_bitpattern_L

   !# Apply water loading in the calculations
   logical :: Schm_wload_L = .false.
   namelist /gem_cfgs  / Schm_wload_L
   namelist /gem_cfgs_p/ Schm_wload_L

   !# Use cubic interpolation in trajectory computation
   logical :: Schm_cub_traj_L = .true.
   namelist /gem_cfgs  / Schm_cub_traj_L
   namelist /gem_cfgs_p/ Schm_cub_traj_L

   !# Use trapezoidal average for advection winds
   logical :: Schm_trapeze_L = .true.
   namelist /gem_cfgs  / Schm_trapeze_L
   namelist /gem_cfgs_p/ Schm_trapeze_L

  !Sol

   !# Type of solver
   !# * 'ITERATIF'
   !# * 'DIRECT'
   character(len=26) :: sol_type_S = 'DIRECT'
   namelist /gem_cfgs  / Sol_type_S
   namelist /gem_cfgs_p/ Sol_type_S

   !# * True-> use FFT solver if possible
   !# * False-> use MXMA solver (slower,less precise)
   logical :: sol_fft_L = .true.
   namelist /gem_cfgs/ Sol_fft_L
   namelist /gem_cfgs_p/ Sol_fft_L

   !# Epsilon convergence criteria for none Yin-Yang iterative solver
   real*8 :: sol_fgm_eps   = 1.d-07
   namelist /gem_cfgs  / Sol_fgm_eps
   namelist /gem_cfgs_p/ Sol_fgm_eps

   !# Epsilon convergence criteria for the Yin-Yang iterative solver
   real*8 :: sol_yyg_eps   = 1.d-04
   namelist /gem_cfgs  / Sol_yyg_eps
   namelist /gem_cfgs_p/ Sol_yyg_eps

   !# maximum number of iterations allowed for none Yin-Yang iterative solver
   integer :: sol_fgm_maxits= 200
   namelist /gem_cfgs  / Sol_fgm_maxits
   namelist /gem_cfgs_p/ Sol_fgm_maxits

   !# maximum number of iterations allowed for the Yin-Yang iterative solver
   integer :: sol_yyg_maxits= 40
   namelist /gem_cfgs  / Sol_yyg_maxits
   namelist /gem_cfgs_p/ Sol_yyg_maxits

   !# size of Krylov subspace in iterative solver - should not exceed 100
   integer :: sol_im = 15
   namelist /gem_cfgs  / Sol_im
   namelist /gem_cfgs_p/ Sol_im

   !# 2D preconditioner for iterative solver
   character(len=26) :: sol2D_precond_S = 'JACOBI'
   namelist /gem_cfgs  / Sol2D_precond_S
   namelist /gem_cfgs_p/ Sol2D_precond_S

   !# 3D preconditioner for iterative solver
   character(len=26) :: sol3D_precond_S = 'JACOBI'
   namelist /gem_cfgs  / Sol3D_precond_S
   namelist /gem_cfgs_p/ Sol3D_precond_S

   !# Krylov method for 3d iterative solver
   character(len=26) :: Sol3D_krylov_S = 'FGMRES'
   namelist /gem_cfgs  / Sol3D_krylov_S
   namelist /gem_cfgs_p/ Sol3D_krylov_S

   !Spn

   !# Spectral nudging list of variables (eg. 'UVT' or 'UV') 
   character(len=16) :: Spn_nudging_S = ' '
   namelist /gem_cfgs  / Spn_nudging_S
   namelist /gem_cfgs_p/ Spn_nudging_S

   !# Nudging profile lower end in hyb level (eg. 1.0 or 0.8) 
   !# If use 0.8, the profile will be set zero when hyb > 0.8
   real :: Spn_start_lev = 1.0
   namelist /gem_cfgs  / Spn_start_lev
   namelist /gem_cfgs_p/ Spn_start_lev

   !# Nudging profile upper end in hyb level (eg. 0.0 or 0.2)
   !# If use 0.2, the profile wll be set 1.0 when hyb < 0.2
   real :: Spn_up_const_lev = 0.0
   namelist /gem_cfgs  / Spn_up_const_lev
   namelist /gem_cfgs_p/ Spn_up_const_lev

   !# Nudging profile transition shape('COS2' or 'LINEAR')
   !# Set the shape between Spn_start_lev and Spn_up_const_lev
   character(len=16) :: Spn_trans_shape_S = 'LINEAR'
   namelist /gem_cfgs  / Spn_trans_shape_S
   namelist /gem_cfgs_p/ Spn_trans_shape_S

   !# Nudging relaxation timescale (eg. 10 hours ) 
   real :: Spn_relax_hours = 10.
   namelist /gem_cfgs  / Spn_relax_hours
   namelist /gem_cfgs_p/ Spn_relax_hours

   !# The filter will be set zero for smaller scales (in km)
   real :: Spn_cutoff_scale_large = 300.
   namelist /gem_cfgs  / Spn_cutoff_scale_large
   namelist /gem_cfgs_p/ Spn_cutoff_scale_large

   !# The filter will be set 1.0 for larger scales (in km) between 
   !# Spn_cutoff_scale_small and Spn_cutoff_scale_large, 
   !# the filter will have a COS2 transition.
   real :: Spn_cutoff_scale_small = 100.
   namelist /gem_cfgs  / Spn_cutoff_scale_small
   namelist /gem_cfgs_p/ Spn_cutoff_scale_small

   !# Nudging interval in seconds (eg. 1800, means nudging is performed 
   !# every every 30 minutes) 
   integer :: Spn_step = 21600
   namelist /gem_cfgs  / Spn_step
   namelist /gem_cfgs_p/ Spn_step

   !# Nudging weight in temporal space (.true. or .false.). 
   !# If the driving fields are available every 6 hours and Spn_step is 
   !# set to 30 minutes then nudging will have more weight every six hours 
   !# when the driving fields are available
   logical :: Spn_weight_L = .false.
   namelist /gem_cfgs  / Spn_weight_L
   namelist /gem_cfgs_p/ Spn_weight_L

   !# The weight factor when Spn_weight_L=.true. 
   !# (The weigh factor is COS2**(Spn_wt_pwr), Spn_wt_pwr could  be set as 
   !# 0, 2, 4, 6. If Spn_wt_pwr = 2, weight factor is COS2)
   integer :: Spn_wt_pwr = 2
   namelist /gem_cfgs  / Spn_wt_pwr
   namelist /gem_cfgs_p/ Spn_wt_pwr

   !# list of variables to do blocstat. 
   !# Any gmm variable name, or predefine lists : 
   !# * 'ALL'
   !# * 'ALL_DYN_T0'
   !# * 'ALL_DYN_T1'
   !# * 'ALL_TR_T0'
   !# * 'ALL_TR_T1'
   character(len=32) :: stat_liste(STAT_MAXN) = ' '
   namelist /gem_cfgs/ stat_liste

   !# list of tracers to be read from analyse
   character(len=512), dimension(500) :: Tr3d_list_S = ' '
   namelist /gem_cfgs/ Tr3d_list_S

   !# Override for default tracers attributes
   character(len=512) :: Tr3d_default_s = ' '
   namelist /gem_cfgs/ Tr3d_default_s

   !# True-> tracers validity time does not have to match analysis
   logical :: Tr3d_anydate_L= .false.
   namelist /gem_cfgs  / Tr3d_anydate_L
   namelist /gem_cfgs_p/ Tr3d_anydate_L

   !# Vspng: 
   !# Vertical sponge if activated, will be applied to the following 
   !# variables: Horizontal Wind,  Temperature (Top level), Zdot, W

   !# Top coefficient for del-2 diffusion (m2/s)
   real :: Vspng_coeftop = -1.
   namelist /gem_cfgs  / Vspng_coeftop
   namelist /gem_cfgs_p/ Vspng_coeftop

   !# Number of levels from the top of the model 
   integer :: Vspng_nk = 0
   namelist /gem_cfgs  / Vspng_nk
   namelist /gem_cfgs_p/ Vspng_nk

   !# True-> Riley diffusion on vertical motion on Vspng_nk levels
   logical :: Vspng_riley_L= .false.
   namelist /gem_cfgs  / Vspng_riley_L
   namelist /gem_cfgs_p/ Vspng_riley_L

   !Vtopo

   !# Time at which to start evolving topography toward target
   character(len=16) :: Vtopo_start_S = ''
   namelist /gem_cfgs  / Vtopo_start_S
   namelist /gem_cfgs_p/ Vtopo_start_S

   !# On which length of time to evolve topography
   character(len=16) :: Vtopo_length_S = ''
   namelist /gem_cfgs  / Vtopo_length_S
   namelist /gem_cfgs_p/ Vtopo_length_S

   !# True-> apply vertical sponge to momentum, and divergence
   logical :: Zblen_L = .false.
   namelist /gem_cfgs  / Zblen_L  
   namelist /gem_cfgs_p/ Zblen_L  

   !# True-> apply vertical sponge also to temperature
   logical :: Zblen_spngtt_L = .false.
   namelist /gem_cfgs  / Zblen_spngtt_L
   namelist /gem_cfgs_p/ Zblen_spngtt_L

   !# height (in meters) of lower boundary for sponge
   real :: Zblen_hmin = 0.
   namelist /gem_cfgs  / Zblen_hmin
   namelist /gem_cfgs_p/ Zblen_hmin

   !# thickness (in meters) of vertical sponge
   real :: Zblen_spngthick = 0.
   namelist /gem_cfgs  / Zblen_spngthick
   namelist /gem_cfgs_p/ Zblen_spngthick

   !# True-> divergence high level modulation in initial computation of Zdot
   logical :: Zdot_divHLM_L = .false.
   namelist /gem_cfgs  / Zdot_divHLM_L
   namelist /gem_cfgs_p/ Zdot_divHLM_L

   character(len=16) Lam_current_S, Lam_previous_S
   logical Init_mode_L, Lam_wgt0, Vtopo_L
   logical Schm_testcases_L,Schm_canonical_dcmip_L,Schm_opentop_L,&
           Schm_canonical_williamson_L, Schm_testcases_adv_L,&
           Schm_phyms_L, Schm_theoc_L
   integer Eq_nlev, Init_dfnp, Init_halfspan
   integer Lctl_step, Lam_blend_Hx, Lam_blend_Hy
   integer Schm_nith, stat_nombre
   integer Vtopo_start, Vtopo_ndt, Vspng_niter
   real, dimension(:,:), pointer :: eponmod
   real, dimension(:  ), pointer :: coef,cm,cp
   real, dimension(:  ), pointer :: Init_dfco
   real*8 Init_dfpl_8, Lam_tdeb, Lam_tfin
   real*8, dimension(:), pointer :: Vspng_coef_8

contains

   function gem_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      
      return
   end function gem_options_init

end module gem_options
