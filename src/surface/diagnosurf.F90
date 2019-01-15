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

subroutine diagnosurf3(m, ni, nk, trnch, itask, kount)
   use phy_itf, only: phy_get
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>

   integer, intent(in) :: m, ni, nk, itask, trnch, kount

   !@Author  B. Bilodeau (Sept 1999)
   !@Revisions
   ! 001      B. Bilodeau (Nov 2000) - New comdeck phybus.cdk
   ! 002      B. Dugas               - Add S5
   ! 003      L. Spacek   (Oct 2010) - Replace UD/VD by UDWE/VDSN
   !                                   Add WSPD/WD (speed/diresction)
   !@Object Time-series and zonal diagnostics extractions
   !       of surface-related variables
   !@Arguments
   !          - Input -
   ! M        horizontal dimensions of fields
   ! NI       number of elements processed in the horizontal
   !          (could be a subset of M in future versions)
   !          (not used for the moment)
   ! NK       vertical dimension
   ! TRNCH    row number
   ! ITASK    task number
   logical, external :: series_isstep

#define PTR1D(NAME2,IDX) busptr(vd%NAME2%i)%ptr(1+IDX,trnch)

   integer ierget, it, istat
   real, pointer, dimension(:,:) :: tmpptr
   real, pointer, dimension(:) :: zinsmavg

   if (.not.series_isstep(' ')) return

   it = itask

   ! Extract time series and zonal diagnostics on 1 levels
!!$   call sersetm('KA', trnch, 1)

   !     inverse of Monin-Obukhov length
   call serxst2(PTR1D(ILMO, (indx_agrege-1)*NI), 'IL', trnch, ni, 1, 0., 1., -1)
   !     specific humidity at the surface
   call serxst2(PTR1D(QSURF, (indx_agrege-1)*NI), 'QG', trnch, ni, 1, 0.0, 1., -1)

   !     glacier surface temperature
   call serxst2(PTR1D(TGLACIER, NI), '9I', trnch, ni, 1, 0.0, 1., -1)

   !     marine ice temperature (3 levels)
   call serxst2(PTR1D(TMICE, NI), '7I', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(TMICE, 2*NI), '8I', trnch, ni, 1, 0.0, 1., -1)

   !     snow thickness (soil, glacier and marine ice)
   call serxst2(PTR1D(SNODP, (indx_soil-1)*NI), 'S1', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(SNODP, (indx_glacier-1)*NI), 'S2', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(SNODP, (indx_ice-1)*NI), 'S4', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(SNODP, (indx_agrege-1)*NI), 'S5', trnch, ni, 1, 0.0, 1., -1)

   !     surface albedo (soil, glacier, marine ice, water and average)
   call serxst2(PTR1D(ALVIS, (indx_soil-1)*NI), 'XS', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(ALVIS, (indx_glacier-1)*NI), 'XG', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(ALVIS, (indx_water-1)*NI), 'XW', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(ALVIS, (indx_ice-1)*NI), 'XI', trnch, ni, 1, 0.0, 1., -1)
   call serxst2(PTR1D(ALVIS, (indx_agrege-1)*NI), 'AL', trnch, ni, 1, 0.0, 1., -1)

   !     soil temperature
   call serxst2(PTR1D(TSURF, 0), 'TS', trnch, ni, 1, 0.0, 1., -1) !J8

   !     deep soil temperature
   call serxst2(PTR1D(TSOIL, NI), 'TP', trnch, ni, 1, 0.0, 1., -1)

   !     radiative surface temperature
   call serxst2(PTR1D(TSRAD, 0), 'G3', trnch, ni, 1, 0.0, 1., -1) !TG

   !     diagnostic U component of the wind at screen level
   call serxst2(PTR1D(UDIAG, 0), 'UDWE', trnch, ni, 1, 0.0, 1., -1) !UD
   call serxst2(PTR1D(UDIAG, 0), 'WSPD', trnch, ni, 1, 0.0, 1., -1) !UD

   !     diagnostic V component of the wind at screen level
   call serxst2(PTR1D(VDIAG, 0), 'VDSN', trnch, ni, 1, 0.0, 1., -1) !VD
   call serxst2(PTR1D(VDIAG, 0), 'WD', trnch, ni, 1, 0.0, 1., -1) !VD

   !     soil moisture content
   call serxst2(PTR1D(WSOIL, 0), 'WG', trnch, ni, 1, 0.0, 1., -1) !I1

   !     deep soil moisture content
   call serxst2(PTR1D(WSOIL, NI), 'WR', trnch, ni, 1, 0.0, 1., -1)

   IF_ISBA: if (schmsol == 'ISBA') then              ! isba

      !#TODO: fix this, may cause an omp crash on Linux/ifort
!!$      nullify(zinsmavg, tmpptr)
!!$      istat = phy_get(tmpptr, 'insmavg', F_npath='V', F_bpath='DPVE', &
!!$           F_quiet=.true., F_folded=.true.)
!!$      if (associated(tmpptr)) then
!!$         zinsmavg(1:ni) => tmpptr(:, trnch)
!!$         call serxst2(zinsmavg, 'MA', trnch, ni, 0., 1., -1)
!!$      endif
      
      !     liquid water stored on canopy
      call serxst2(PTR1D(WVEG, 0), 'C5', trnch, ni, 1, 0.0, 1., -1) !I3

      !     mass of snow cover
      call serxst2(PTR1D(SNOMA, 0), 'C6', trnch, ni, 1, 0.0, 1., -1) !I5

      !     snow albedo
      call serxst2(PTR1D(SNOAL, 0), 'C7', trnch, ni, 1, 0.0, 1., -1) !I6

      !     snow density
      call serxst2(PTR1D(SNORO, 0), 'C8', trnch, ni, 1, 0.0, 1., -1) !7S

      !     net radiation
      call serxst2(PTR1D(RNET_S, 0), 'C9', trnch, ni, 1, 0.0, 1., -1) !NR

      !     latent heat flux over bare ground
      call serxst2(PTR1D(LEG, 0), 'D3', trnch, ni, 1, 0.0, 1., -1) !L2

      !     latent heat flux over vegetation
      call serxst2(PTR1D(LEV, 0), 'D4', trnch, ni, 1, 0.0, 1., -1) !LV

      !     latent heat flux over snow
      call serxst2(PTR1D(LES, 0), 'D5', trnch, ni, 1, 0.0, 1., -1) !LS

      !     direct latent heat flux from vegetation leaves
      call serxst2(PTR1D(LER, 0), 'D6', trnch, ni, 1, 0.0, 1., -1) !LR

      !     latent heat of evapotranspiration
      call serxst2(PTR1D(LETR, 0), 'D7', trnch, ni, 1, 0.0, 1., -1) !LT

      !     runoff
      call serxst2(PTR1D(OVERFL, 0), 'E2', trnch, ni, 1, 0.0, 1., -1) !RO

      !     drainage
      call serxst2(PTR1D(DRAIN, 0), 'E3', trnch, ni, 1, 0.0, 1., -1) !DR

      !     fraction of the grid covered by snow
      call serxst2(PTR1D(PSN, 0), 'E5', trnch, ni, 1, 0.0, 1., -1) !5P

      !     fraction of bare ground covered by snow
      call serxst2(PTR1D(PSNG, 0), 'E6', trnch, ni, 1, 0.0, 1., -1) !3P

      !     fraction of vegetation covered by snow
      call serxst2(PTR1D(PSNV, 0), 'E7', trnch, ni, 1, 0.0, 1., -1) !4P

      !     stomatal resistance
      call serxst2(PTR1D(RST, 0), 'E8', trnch, ni, 1, 0.0, 1., -1) !R1

      !     specific humidity of the surface
      call serxst2(PTR1D(HUSURF, 0), 'E9', trnch, ni, 1, 0.0, 1., -1) !FH

      !     Halstead coefficient (relative humidity of veg. canopy)
      call serxst2(PTR1D(HV, 0), 'G1', trnch, ni, 1, 0.0, 1., -1) !HV

      !     soil volumetric ice content
      call serxst2(PTR1D(ISOIL, 0), 'G4', trnch, ni, 1, 0.0, 1., -1) !I2

      !     liquid water in snow
      call serxst2(PTR1D(WSNOW, 0), 'G5', trnch, ni, 1, 0.0, 1., -1) !I4

      !     liquid precip. rate
      call serxst2(PTR1D(RAINRATE, 0), 'G6', trnch, ni, 1, 0.0, 1., -1) !U1

      !     solid precip. rate
      call serxst2(PTR1D(SNOWRATE, 0), 'G7', trnch, ni, 1, 0.0, 1., -1) !U3

   endif IF_ISBA

   return
end subroutine diagnosurf3
