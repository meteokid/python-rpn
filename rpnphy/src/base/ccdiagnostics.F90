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

subroutine ccdiagnostics(fbus,fsiz,vbus,vsiz, &
                         zcte,zcqe,ps, &
                         trnch,ni,nk,icpu,kount)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>

  integer               :: fsiz,vsiz,trnch,ni,nk,icpu,kount
  real,dimension(fsiz)  :: fbus(fsiz)
  real,dimension(vsiz)  :: vbus(vsiz)
  real,dimension(ni)    :: ps
  real,dimension(ni,nk) :: zcte,zcqe

!Author
!          L.Spacek, November 2011
!
!Revisions
! 001
!
!Object
!          Convection/condensation diagnostics
!
!Arguments
!
!          - Input -
! dsiz     dimension of dbus
! fsiz     dimension of fbus
! vsiz     dimension of vbus
! zcte     convective temperature tendency
! zcqe     convective humidity tendency
! ps       surface pressure
! trnch    slice number
! ni       horizontal running length
! nk       vertical dimension
! icpu     cpu number executing slice "trnch"
! kount    timestep number
!
!          - Input/Output -
! dbus     dynamics input field
! fbus     historic variables for the physics
! vbus     physics tendencies and other output fields from the physics
!
!Implicites

  include "thermoconsts.inc"
  logical, external :: series_isstep

  integer               :: i,k,ik,istat
  real                  :: tcel,frac,airdenm1
  real,dimension(ni)    :: temp1,temp2,vis_lowest
  real,dimension(ni,nk) :: liquid, solid


!  convert to liquid-equivalent precipitation rates:
!  (divide by density of water [1000 kg m-3])
  fbus(tsc:tsc    +ni-1) = fbus(tsc:tsc     +ni-1) * 1.e-03
  fbus(tss:tss    +ni-1) = fbus(tss:tss     +ni-1) * 1.e-03
  fbus(tlc:tlc    +ni-1) = fbus(tlc:tlc     +ni-1) * 1.e-03
  fbus(tls:tls    +ni-1) = fbus(tls:tls     +ni-1) * 1.e-03
  fbus(rckfc:rckfc+ni-1) = fbus(rckfc:rckfc +ni-1) * 1.e-03

! note - precipitation rates are not zeroed at step 0 for micorphysics schemes
  if ( (kount.eq.0) .and. (stcond(1:2) /= 'MP') ) then
!        mettre a zero les taux de precipitations
     do i=0,ni-1
        fbus(tlc     +i) = 0.
        fbus(tlcs    +i) = 0.
        fbus(tls     +i) = 0.
        fbus(tsc     +i) = 0.
        fbus(tscs    +i) = 0.
        fbus(tss     +i) = 0.
        fbus(rckfc   +i) = 0.
     end do
  endif

  if (.not.series_isstep(' ')) return

!     diagnostics                                   *

!     stratiformes clouds
  call serxst2(vbus(flagmxp), 'FG', trnch, ni, nk, 0., 1., -1)
  call serxst2(fbus(fxp),     'NS', trnch, ni, nk, 0., 1., -1)

  if (.not.any(convec == (/&
       'NIL   ', &
       'SEC   ', &
       'MANABE'  &           
       /))) then
!        tendances convectives
     call serxst2(zcte, 'TK', trnch, ni, nk, 0.0, 1., -1)
     call serxst2(zcqe, 'QK', trnch, ni, nk, 0.0, 1., -1)
  endif

  if ( (stcond=='NEWSUND') .or. (stcond=='CONSUN') ) then
!        precipitation flux
     call serxst2(vbus(rnflx),  'WF', trnch, ni, nk, 0., 1., -1)
     call serxst2(vbus(snoflx), 'SF', trnch, ni, nk, 0., 1., -1)

!        thickness and water path
     call serxst2(vbus(icw),  'IE', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(iwv),  'IH', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(lwp2), 'IC', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(iwp),  'II', trnch, ni, 1, 0., 1., -1)
     !
  else if (convec == 'OLDKUO') then

     call serxst2(fbus(fbl),  'NC', trnch, ni, nk, 0.0, 1., -1)
     
  endif

  if (stcond(1:6)=='MP_MY2') then
     call serxst2(fbus(tls_rn1), 'RRN1', trnch, ni, 1, 0., 1., -1)
     call serxst2(fbus(tls_rn2), 'RRN2', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tls_fr1), 'RFR1', trnch, ni, 1, 0., 1., -1)
     call serxst2(fbus(tls_fr2), 'RFR2', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tss_sn1), 'RSN1', trnch, ni, 1, 0., 1., -1)
     call serxst2(fbus(tss_sn2), 'RSN2', trnch, ni, 1, 0., 1., -1)
     call serxst2(fbus(tss_sn3), 'RSN3', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tss_pe1), 'RPE1', trnch, ni, 1, 0., 1., -1)
     call serxst2(fbus(tss_pe2), 'RPE2', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tss_pe2l),'RPEL', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tss_snd), 'RSND', trnch, ni, 1, 0., 1., -1)

     call serxst2(fbus(tsrad),   'TG',   trnch, ni, 1, 0., 1., -1)

     call serxst2(vbus(h_cb), 'H_CB', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(h_ml), 'H_ML', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(h_m2), 'H_M2', trnch, ni, 1, 0., 1., -1)
     call serxst2(vbus(h_sn), 'H_SN', trnch, ni, 1, 0., 1., -1)

!   prepare only lowest-level component visibilities for series and zonal avg
     do i=1, ni
        vis_lowest(i) = vbus(vis+(nk-1)*ni+i-1)
     enddo
     call serxst2(vis_lowest, 'VIS', trnch, ni, 1, 0., 1., -1)
     do i=1, ni
        vis_lowest(i) = vbus(vis1+(nk-1)*ni+i-1)
     enddo
     call serxst2(vis_lowest, 'VIS1', trnch, ni, 1, 0., 1., -1)
     do i=1, ni
        vis_lowest(i) = vbus(vis2+(nk-1)*ni+i-1)
     enddo
     call serxst2(vis_lowest, 'VIS2', trnch, ni, 1, 0., 1., -1)
     do i=1, ni
        vis_lowest(i) = vbus(vis3+(nk-1)*ni+i-1)
     enddo
     call serxst2(vis_lowest, 'VIS3', trnch, ni, 1, 0., 1., -1)

     !
  endif

end subroutine ccdiagnostics
