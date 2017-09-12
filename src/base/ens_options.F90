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
module ens_options
   implicit none
   public
   save

   integer, parameter :: MAX2DC=6

   !# Switch to activate generation of Markov chains, use of SKEB
   !# and use of PTP
   logical :: Ens_conf = .false.
   namelist /ensembles/ Ens_conf

   !# Switch to activate the
   !# in PTP & SKEB
   logical :: Ens_iau_mc = .false.
   namelist /ensembles/ Ens_iau_mc

   !# Switch to activate the first initialisation of Matkov chaines
   !# in PTP & SKEB
   logical :: Ens_first_init_mc = .false.
   namelist /ensembles/ Ens_first_init_mc

   !# Switch to activate SKEB
   !# (3D MARKOV CHAINES)
   logical :: Ens_skeb_conf = .false.
   namelist /ensembles/ Ens_skeb_conf

   !# switch to print global stat related to Markov chains, SKEB and PTP
   !# (3D MARKOV CHAINES)
   logical :: Ens_stat = .false.
   namelist /ensembles/ Ens_stat

   !# switch to do the calculation of the divergence due to SKEB forcing
   !# (3D MARKOV CHAINES)
   logical :: Ens_skeb_div = .false.
   namelist /ensembles/ Ens_skeb_div

   !# switch to do SKEB calculation based on diffusion
   !# (3D MARKOV CHAINES)
   logical :: Ens_skeb_dif = .false.
   namelist /ensembles/ Ens_skeb_dif

   !# switch to do SKEB calculation based on gravity wave drag
   !# (3D MARKOV CHAINES)
   logical :: Ens_skeb_gwd = .false.
   namelist /ensembles/ Ens_skeb_gwd

   !# Seed of the random number generator usually we put DAY and member number
   !# (3D MARKOV CHAINES)
   integer :: Ens_mc_seed = -1
   namelist /ensembles/ Ens_mc_seed

   !# number of longitudes of the gaussian grid used for the 3D Markov chains
   !# (in the SKEB calculation)
   !# (3D MARKOV CHAINES)
   integer :: Ens_skeb_nlon = 16
   namelist /ensembles/ Ens_skeb_nlon

   !# number of latitudes of the gaussian grid used  for the 3D Markov chains
   !# (used in the SKEB calculation)
   !# (3D MARKOV CHAINES)
   integer :: Ens_skeb_nlat = 8
   namelist /ensembles/ Ens_skeb_nlat

   !#
   !# (3D MARKOV CHAINES)
   integer :: Ens_skeb_ncha = 1
   namelist /ensembles/ Ens_skeb_ncha

   !# low wave number truncation limit used in 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   integer :: Ens_skeb_trnl = 2
   namelist /ensembles/ Ens_skeb_trnl

   !# high wave number truncation limit used in 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   integer :: Ens_skeb_trnh = 8
   namelist /ensembles/ Ens_skeb_trnh

   !# maximum value of the 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_max = 0.
   namelist /ensembles/ Ens_skeb_max

   !# minimum value of the 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_min = 0.
   namelist /ensembles/ Ens_skeb_min

   !# std. dev. value for the 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_std = 0.
   namelist /ensembles/ Ens_skeb_std

   !# decorrelation time (seconds) for 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_tau = 0.
   namelist /ensembles/ Ens_skeb_tau

   !# value of stretch for 3D Markov chain (used by SKEB)
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_str = 0.
   namelist /ensembles/ Ens_skeb_str

   !# coefficient Alpha for momentum in SKEB
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_alph = 0.
   namelist /ensembles/ Ens_skeb_alph

   !# coefficient Alpha for temperature in SKEB
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_alpt = 0.
   namelist /ensembles/ Ens_skeb_alpt

   !# coefficient for Gaussian filter used in SKEB
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_bfc = 1.0e-01
   namelist /ensembles/ Ens_skeb_bfc

   !# wavelength for Gaussian filter in SKEB
   !# (3D MARKOV CHAINES)
   real :: Ens_skeb_lam = 2.0e+05
   namelist /ensembles/ Ens_skeb_lam

   !# no. of longitudes for 2D Markov chains
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_nlon(MAX2DC) = 16
   namelist /ensembles/Ens_ptp_nlon

   !# no. of latitudes for 2D Markov chains
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_nlat(MAX2DC) = 8
   namelist /ensembles/ Ens_ptp_nlat

   !# number of 2d Markov chains
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_ncha = 1
   namelist /ensembles/ Ens_ptp_ncha

   !# low wave number horizontal truncation limit for 2D Markov chains
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_trnl(MAX2DC) = 1
   namelist /ensembles/ Ens_ptp_trnl

   !# high wave number horizontal truncation limit for 2D Markov chains
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_trnh(MAX2DC) = 8
   namelist /ensembles/ Ens_ptp_trnh

   !# (ignored) Ens_ptp_l = Ens_ptp_trnh-Ens_ptp_trnl+1
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_l(MAX2DC) = 0
   namelist /ensembles/ Ens_ptp_l

   !# (ignored) Ens_ptp_m = Ens_ptp_trnh+1
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_m(MAX2DC) = 0
   namelist /ensembles/ Ens_ptp_m

   !# (ignored) Ens_ptp_lmax = maxval(Ens_ptp_l)
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_lmax = 0
   namelist /ensembles/ Ens_ptp_lmax

   !# (ignored) Ens_ptp_mmax = maxval(Ens_ptp_m)
   !# (2D MARKOV CHAINES)
   integer :: Ens_ptp_mmax = 0
   namelist /ensembles/ Ens_ptp_mmax

   !# minimum value of the 2D Markov chain
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_min(MAX2DC) = 0.0
   namelist /ensembles/ Ens_ptp_min

   !# maximum value of the 2D Markov chains
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_max(MAX2DC) = 0.0
   namelist /ensembles/ Ens_ptp_max

   !# standard deviation value for 2D Markov chains
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_std(MAX2DC) = 0.0
   namelist /ensembles/ Ens_ptp_std

   !# decorrelation time (seconds) for 2D Markov chains
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_tau(MAX2DC) = 0.0
   namelist /ensembles/ Ens_ptp_tau

   !# value of stretch for Markov chains
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_str(MAX2DC) = 0.0
   namelist /ensembles/ Ens_ptp_str

   !# switch to activate PTP (perturb tendencies of physics)
   !# (2D MARKOV CHAINES)
   logical :: Ens_ptp_conf = .false.
   namelist /ensembles/ Ens_ptp_conf

   !# upper value of transition zone of vertical envelope in sigma for PTP
   !# (above that full perturbation)
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_env_u = 1.0
   namelist /ensembles/ Ens_ptp_env_u

   !# bottom value of transition zone of vertical envelope in sigma for PTP
   !# (below that no perturbation)
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_env_b = 1.0
   namelist /ensembles/ Ens_ptp_env_b

   !# CAPE value in Kain-Fritsch scheme to stop perturbing the physical
   !# tendencies
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_cape = 0.0
   namelist /ensembles/ Ens_ptp_cape

   !# TLC value (convective precipitation) in Kuo (OLDKUO) scheme to stop
   !# perturbing the physical tendencies
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_tlc = 0.0
   namelist /ensembles/ Ens_ptp_tlc

   !# vertical velocity value (m/s) above which we stop perturbing the
   !# physical tendencies
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_crit_w = 100.0
   namelist /ensembles/ Ens_ptp_crit_w

   !# factor of reduction of the perturbation the physical tendencies (in PTP)
   !# when convection occurs
   !# (2D MARKOV CHAINES)
   real :: Ens_ptp_fac_reduc = 0.0
   namelist /ensembles/ Ens_ptp_fac_reduc

contains

   function ens_options_init() result(F_istat)
      implicit none
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.

      return
   end function ens_options_init

end module ens_options
