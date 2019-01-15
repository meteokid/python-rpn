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
!** S/P PREP_CW
subroutine prep_cw2 (f, fsiz, d, dsiz, v, vsiz, &
     ficebl, kount, trnch, task, ni, nk)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>

  integer fsiz, dsiz, vsiz, ni, n, nk
  integer kount, trnch, task
  real, target :: f(fsiz), d(dsiz), v(vsiz)
  real ficebl(ni,nk)

!Author
!          L. Spacek (Oct 2004)
!
!Revisions
!     001      see version 5.5.0 for previous history
!
!Object
!          Save water contents and cloudiness in the permanent bus
!
!Arguments
!
!          - Input -
! dsiz     dimension of d
! fsiz     dimension of f
! vsiz     dimension of v
! ficebl   fraction of ice
! kount    index of timestep
! trnch    number of the slice
! task     task number
! ni       horizontal dimension
! nk       vertical dimension
!
!          - Output -
!
!          - Input/Output -
! d        dynamic             bus
! f        permanent variables bus
! v        volatile (output)   bus
!***********************************************************************

  integer ik, i, k
  real, target,dimension(ni,nk) :: zero
  real, pointer, dimension(:,:) :: zfbl, zfdc, zfsc, zftot, zfxp, ziwc, zlwc, &
       zqcplus, zqiplus, zqldi, zqlsc, &
       zqsdi, zqssc, zqtbl, zsnow, zqi_cat1, zqi_cat2, zqi_cat3, zqi_cat4

  zero   (1:ni,1:nk) =  0.0
  zfbl   (1:ni,1:nk) => f(fbl:)
  zfdc   (1:ni,1:nk) => f(fdc:)
  zftot  (1:ni,1:nk) => f(ftot:)
  zfxp   (1:ni,1:nk) => f(fxp:)
  zlwc   (1:ni,1:nk) => f(lwc:)
  ziwc   (1:ni,1:nk) => f(iwc:)
  zqcplus(1:ni,1:nk) => d(qcplus:)
  zqtbl  (1:ni,1:nk) => f(qtbl:)
  if (qiplus > 0) zqiplus(1:ni,1:nk) => d(qiplus:)
!
  if (convec == 'KFC'.or.convec == 'BECHTOLD') then
     zqldi  (1:ni,1:nk) => f(qldi:)
     zqsdi  (1:ni,1:nk) => f(qsdi:)
  else
     zqldi => zero(1:ni,1:nk)
     zqsdi => zero(1:ni,1:nk)
  endif
!
  if (conv_shal /= 'NIL') then
     zqlsc  (1:ni,1:nk) => v(qlsc:)
     zqssc  (1:ni,1:nk) => v(qssc:)
  else
     zqlsc => zero(1:ni,1:nk)
     zqssc => zero(1:ni,1:nk)
  endif
!
  if (conv_shal /= 'NIL') then
     zfsc   (1:ni,1:nk) => f(fsc:)
  else
     zfsc => zero(1:ni,1:nk)
  endif
!
!
!     Apply the convection/condensation tendencies
!     ------------------------------------------
!
!     Cloud water
!     ------------
!
  if (stcond/='NIL'.and.stcond/='CONDS/') then
!
!     Copy qcplus into the permanent bus. Il will be used during
!     next timestep by the radiation
!
     do k=1,nk
        do i=1,ni
           zlwc(i,k) =  zqcplus(i,k)
        enddo
     enddo
  endif
!     If we have the CONSUN scheme, the cloud water from MoisTKE has
!     priority over the cloud water from the grid-scale scheme.
!     In this Case, everything goes into the LWC (for liquid cloud water)
!     variable.
!
  if (stcond=='CONSUN'.and.fluvert=='MOISTKE') then
     do k=1,nk
        do i=1,ni
           if(zqtbl(i,k)>zqcplus(i,k))then
              zlwc(i,k) = zqtbl(i,k)
              zfxp(i,k) = zfbl(i,k)
           endif
        enddo
     enddo
  endif


!     MoisTKE has priority over the microphysics scheme

  if (stcond(1:2)=='MP') then

     if (stcond(1:6)=='MP_MY2') then
        zlwc = zqcplus
     zsnow(1:ni,1:nk) => d(qnplus:)
        ziwc = zqiplus + zsnow
     elseif (stcond=='MP_P3') then
        zlwc = zqcplus
        zqi_cat1(1:ni,1:nk) => d(i1qtplus:)
        ziwc = zqi_cat1
        if (mp_p3_ncat >= 2) then
           zqi_cat2(1:ni,1:nk) => d(i2qtplus:)
           ziwc = ziwc + zqi_cat2
        endif
        if (mp_p3_ncat >= 3) then
           zqi_cat3(1:ni,1:nk) => d(i3qtplus:)
           ziwc = ziwc + zqi_cat3
        endif
        if (mp_p3_ncat >= 4) then
           zqi_cat4(1:ni,1:nk) => d(i4qtplus:)
           ziwc = ziwc + zqi_cat4
        endif
     endif

     if(fluvert=='MOISTKE') then
        do k=1,nk
           do i=1,ni
              if (zqtbl(i,k).gt.zlwc(i,k)+ziwc(i,k)) then
                 zlwc(i,k) =  zqtbl(i,k) * (1.0 - ficebl(i,k) )
                 ziwc(i,k) =  zqtbl(i,k) * ficebl(i,k)
                 zfxp(i,k) =  zfbl (i,k)
              endif
           enddo
        enddo
     endif
  endif

!     Add the cloud water (liquid and solid) coming from shallow and deep
!     cumulus clouds (only for the Kuo Transient and Kain-Fritsch schemes).
!     Note that no conditions are used for these calculations ...
!     qldi, qsdi, and qlsc, qssc are zero if these schemes are not used.
!     Also note that qldi, qsdi, qlsc and qssc are NOT IN-CLOUD values
!     (multiplication done in kfcp4 and ktrsnt)
!
!     For grid-scale schemes (e.g., Sundqvist, Milbrandt-Yau, P3) all
!     the cloud water is put in LWC (and will be partition later in the
!     radiation subroutines)
!
  if (stcond(1:2)=='MP') then
     zlwc = zlwc + zqldi + zqlsc
     ziwc = ziwc + zqsdi + zqssc
  else
     zlwc = zlwc + zqldi + zqsdi + zqlsc + zqssc
  endif


!     Combine explicit and Implicit clouds using the random overlap
!     approximation:
!         FXP is for grid-scale condensation
!              (but can also include the PBL clouds of MoisTKE)
!         FDC is for deep convection clouds
!              (always defined as necessary for condensation too)
!         FSC is for the shallow convection clouds
!
!
  do k=1,nk
     do i=1,ni
        zftot(i,k) = min(  max( &
             1. - (1.-zfxp(i,k))*(1.-zfdc(i,k))*(1.-zfsc(i,k)) , &
             0.       )  , &
             1.)
     enddo
  enddo
end subroutine prep_cw2
