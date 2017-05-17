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
module grid_options
   implicit none
   public
   save

   !# Type of grid described using 2 characters:
   !# * "GY" : Global Yin-Yang
   !# * "LU" : LAM    Uniform
   character(len=2) ::  Grd_typ_S = 'GY'
   namelist /grid/ Grd_typ_S

   !# Number of points along NI
   integer :: Grd_ni = 0
   namelist /grid/ Grd_ni

   !# Number of points along NJ
   integer :: Grd_nj = 0
   namelist /grid/ Grd_nj

   !# Max Supported Courrant number;
   !# Pilot area=Grd_maxcfl +Grd_bsc_base+Grd_bsc_ext1
   integer :: Grd_maxcfl = 1
   namelist /grid/ Grd_maxcfl

   !# (LU only) Mesh length (resolution) in x-direction (degrees) 
   real  :: Grd_dx = 0.
   namelist /grid/ Grd_dx

   !# (LU only) Mesh length (resolution) in y-direction (degrees) 
   real  :: Grd_dy = 0.
   namelist /grid/ Grd_dy

   !# Latitude on rotated grid of reference point, Grd_iref,Grd_jref (degrees) 
   real  :: Grd_latr = 0.
   namelist /grid/ Grd_latr

   !# Longitude on rotated grid of reference point, Grd_iref,Grd_jref (degrees) 
   real  :: Grd_lonr = 180.
   namelist /grid/ Grd_lonr

   !# (GY only) Overlap extent along latitude axis for GY grid (degrees)
   real  :: Grd_overlap = 0.
   namelist /grid/ Grd_overlap

   !# Geographic latitude of the center of the computational domain (degrees)
   real  :: Grd_xlon1 = 180.
   namelist /grid/ Grd_xlon1

   !# Geographic longitude of the center of the computational domain (degrees)
   real  :: Grd_xlat1 = 0.
   namelist /grid/ Grd_xlat1

   !# Geographic longitude of a point on the equator of the computational domain
   !# east of Grd_xlon1,Grd_xlat1  (degrees)
   real  :: Grd_xlon2 = 270.
   namelist /grid/ Grd_xlon2

   !# Geographic latitude of a point on the equator of the computational domain
   !# east of  Grd_xlon1,Grd_xlat1  (degrees)
   real  :: Grd_xlat2 = 0.
   namelist /grid/ Grd_xlat2

   character*12 Grd_yinyang_S
   logical Grd_roule, Grd_yinyang_L
   integer Grd_bsc_base, Grd_bsc_adw, Grd_bsc_ext1, Grd_extension
   integer Grd_iref, Grd_jref
   integer Grd_ndomains, Grd_mydomain
   integer Grd_local_gid,Grd_lclcore_gid,Grd_global_gid,Grd_lphy_gid
   integer Grd_lphy_i0,Grd_lphy_in,Grd_lphy_j0,Grd_lphy_jn,Grd_lphy_ni,Grd_lphy_nj
   real*8 Grd_rot_8(3,3), Grd_x0_8, Grd_xl_8, Grd_y0_8, Grd_yl_8

contains

   !/@*
   function grid_options_init() result(F_istat)
      implicit none
      !@object Additional initialisation steps before reading the nml
      integer :: F_istat
      !*@/
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      !----------------------------------------------------------------
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      !----------------------------------------------------------------
      return
   end function grid_options_init

end module grid_options
