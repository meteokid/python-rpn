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
module grdc_options
   implicit none
   public
   save
   
   integer, parameter :: MAX_TRNM = 1000

   !# x horizontal resolution of target cascade grid (degrees) 
   real  :: Grdc_dx = -1.
   namelist /grdc/ Grdc_dx
   namelist /grdc_p/ Grdc_dx

   !# y horizontal resolution of target cascade grid (degrees) 
   real  :: Grdc_dy = -1.
   namelist /grdc/ Grdc_dy
   namelist /grdc_p/ Grdc_dy

   !# Number of points for the blending zone (Hblen_x)
   integer :: Grdc_Hblen = 10
   namelist /grdc/ Grdc_Hblen
   namelist /grdc_p/ Grdc_Hblen

   !# TRUE to dump out permanent bus for cascade mode
   logical :: Grdc_initphy_L = .false.
   namelist /grdc/ Grdc_initphy_L
   namelist /grdc_p/ Grdc_initphy_L

   !# Latitude on rotated grid of ref point, Grdc_iref,Grdc_jref (degrees) 
   real  :: Grdc_latr = 0.
   namelist /grdc/ Grdc_latr
   namelist /grdc_p/ Grdc_latr

   !# Longitude on rotated grid of ref point, Grdc_iref,Grdc_jref (degrees) 
   real  :: Grdc_lonr = 180.
   namelist /grdc/ Grdc_lonr
   namelist /grdc_p/ Grdc_lonr

   !# Max Supported Courrant number;
   !# Pilot area=Grdc_maxcfl +Grdc_bsc_base+Grdc_bsc_ext1
   integer :: Grdc_maxcfl = 1
   namelist /grdc/ Grdc_maxcfl
   namelist /grdc_p/ Grdc_maxcfl

   !# Number of bits for the packing factor 
   integer :: Grdc_nbits = 32
   namelist /grdc/ Grdc_nbits
   namelist /grdc_p/ Grdc_nbits

   !# Nesting interval specified with digits ending with one character 
   !# for the units:
   !# * S : seconds
   !# * D : days
   !# * M : minutes
   !# * H : hours 
   character(len=15) :: Grdc_nfe = ' '
   namelist /grdc/ Grdc_nfe
   namelist /grdc_p/ Grdc_nfe

   !# Number of points along X
   integer :: Grdc_ni = 0
   namelist /grdc/ Grdc_ni
   namelist /grdc_p/ Grdc_ni

   !# Number of points along Y
   integer :: Grdc_nj = 0
   namelist /grdc/ Grdc_nj
   namelist /grdc_p/ Grdc_nj

   !# Time string (units D, H, M or S) from the start of the run to 
   !# start producing the cascade files 
   character(len=15) :: Grdc_start_S = ' '
   namelist /grdc/ Grdc_start_S
   namelist /grdc_p/ Grdc_start_S

   !# Time string (units D, H, M or S) from the start of the run to 
   !# stop producing the cascade files
   character(len=15) :: Grdc_end_S = ' '
   namelist /grdc/ Grdc_end_S
   namelist /grdc_p/ Grdc_end_S

   !# List of tracers to be written from piloting run
   character(len=4) :: Grdc_trnm_S(MAX_TRNM) = '@#$%'
   namelist /grdc/ Grdc_trnm_S

   integer Grdc_iref,Grdc_jref,Grdc_gid,Grdc_gif,Grdc_gjd,Grdc_gjf, &
           Grdc_ndt,Grdc_pil,Grdc_ntr,Grdc_start,Grdc_end
   real    Grdc_xlat1,Grdc_xlon1,Grdc_xlat2,Grdc_xlon2
   real*8  Grdc_xp1,Grdc_yp1

contains

   function grdc_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      
      return
   end function grdc_options_init

end module grdc_options
