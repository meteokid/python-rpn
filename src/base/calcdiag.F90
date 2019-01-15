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

!/@*
subroutine calcdiag(d,f,v,dsiz,fsiz,vsiz,dt,trnch,kount,ni,nk)
   !@Object Calculates averages and accumulators of tendencies and diagnostics
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Arguments
   !          - Input/Output -
   ! d        dynamic bus
   ! f        permanent bus
   !          - input -
   ! v        volatile (output) bus
   !          - input -
   ! dsiz     dimension of d
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! trnch    slice number
   ! kount    timestep number
   ! dt       length of timestep
   ! n        horizontal running length
   ! nk       vertical dimension

   integer dsiz,fsiz,vsiz,trnch,kount,ni,nk
   real dt
   real, target :: d(dsiz),f(fsiz), v(vsiz)

   !@Author B. Bilodeau Feb 2003
   !*@/

#include <WhiteBoard.hf>
   include "thermoconsts.inc"
   include "surface.cdk"
   include "physteps.cdk"

   logical :: lmoyhr, laccum, lreset, lavg
   integer :: i,k,moyhr_steps
   real :: moyhri,uvs(ni)
   real :: tempo, tempo2, sol_stra, sol_conv, liq_stra, liq_conv
   real, dimension(ni) :: vmod,vdir,th_air

   include "calcdiag_ptr.cdk"
   include "calcdiag_ptr_as.cdk"
   !----------------------------------------------------------------

   lmoyhr = (moyhr > 0)
   laccum = (lmoyhr .or. dynout)
   lreset = .false.
   lavg   = .false.
   if (lmoyhr) then
      lreset = (mod((step_driver-1),moyhr) == 0)
      lavg   = (mod(step_driver,moyhr) == 0)
   endif

   !****************************************************************
   !     Derived screen-level fields
   !     ---------------------------

   IF_FLUVERT: if (.not.(kount == 0 .and. &
      any(fluvert == (/'MOISTKE', 'CLEF   '/)))) then

      !  Screen level relative humidity
      call mfohr4(v(rhdiag), f(qdiag), f(tdiag), f(pmoins), ni, 1, ni ,satuco)
      !  Clip the screen level relative humidity to a range from 0-1
      do i = 1, ni
         zrhdiag(i) = max(min(zrhdiag(i), 1.0), 0.)
      enddo

      !     Screen level dewpoint depression
      call mhuaes3(v(esdiag), f(qdiag), f(tdiag), f(pmoins), .false., ni, 1, ni)
      !     Screen level dewpoint
      do i = 1,ni
         ztdew(i) = ztdiag(i) - zesdiag(i)
      enddo

   endif IF_FLUVERT


   !****************************************************************
   !     PRECIPITATION RATES AND ACCUMULATIONS
   !     -------------------------------------

   !# Compute the averaging interval (inverse), with all available data
   !  averaged when moving through driver step = 0
   moyhri = 1.
   if (lmoyhr) then
      moyhr_steps = moyhr
      if (step_driver == 0 .and. kount > 0) moyhr_steps = min(moyhr,kount)
      moyhri = 1./float(moyhr_steps)
   endif

   !# Set precipitation accumulators to zero at the beginning and after every
   !  acchr hours, and by default (acchr=0) as the model step goes through 0.
   IF_RESET_PRECIP: if (kount == 0 .or. &
        (acchr > 0 .and. mod((step_driver-1),acchr) == 0) .or. &
        (acchr == 0 .and. step_driver-1 == 0)) then
      do i=1,ni
         zasc (i) = 0.
         zascs(i) = 0.
         zalc (i) = 0.
         zalcs(i) = 0.
         zass (i) = 0.
         zals (i) = 0.
         zpc  (i) = 0.
         zpy  (i) = 0.
         zpz  (i) = 0.
         zae  (i) = 0.
         zpr  (i) = 0.
         zazr (i) = 0.
         zrn  (i) = 0.
         zaip (i) = 0.
         zsn  (i) = 0.
      enddo
      if  (stcond(1:6) == 'MP_MY2')  then
         do i = 1,ni
            zals_rn1 (i) = 0.
            zals_rn2 (i) = 0.
            zals_fr1 (i) = 0.
            zals_fr2 (i) = 0.
            zass_sn1 (i) = 0.
            zass_sn2 (i) = 0.
            zass_sn3 (i) = 0.
            zass_pe1 (i) = 0.
            zass_pe2 (i) = 0.
            zass_pe2l(i) = 0.
            zass_snd (i) = 0.
            zass_mx  (i) = 0.
            zass_s2l (i) = 0.
         enddo
      endif
   endif IF_RESET_PRECIP

   IF_KOUNT_NOT_0: if (kount /= 0) then
      !DIAGNOSTICS ON PRECIPITATION TYPE.
      if (pcptype == 'BOURGE') then
         call bourge2(v(fneige),v(fip),d(tplus),d(sigw),f(pmoins),ni,nk-1)
      else if (pcptype == 'BOURGE3D')then
         call bourge1_3d(f(fneige),f(fip),d(tplus),d(sigw),f(pmoins),ni,nk-1)
      endif

      IF_BOURG3D: if (pcptype == 'BOURGE3D') then

         !AZR3D: ACCUMULATION DES PRECIPITATIONS VERGLACLACANTES EN 3D
         !AIP3D: ACCUMULATION DES PRECIPITATIONS RE-GELEES EN 3D
         if (stcond == 'CONSUN') then
            do k = 1,nk-1
               do i = 1, ni
                  ! Flux de consun1
                  tempo = (zsnoflx(i,k)+zrnflx(i,k))*.001
                  sol_stra = max(0.,zfneige2d(i,k)*tempo)
                  liq_stra = max(0.,tempo-sol_stra)
                  ! Flux de kfc
                  tempo = (zkfcrf(i,k)+zkfcsf(i,k))*.001
                  sol_conv = max(0.,zfneige2d(i,k)*tempo)
                  liq_conv = max(0.,tempo-sol_conv)

                  tempo = liq_stra+liq_conv

                  if (ztplus(i,k) < tcdk) then
                     zazr3d(i,k) = zazr3d(i,k) + (1.-zfip2d(i,k))*tempo*dt
                  endif

                  zaip3d(i,k) = zaip3d(i,k) + zfip2d(i,k)*tempo*dt
               enddo
            enddo
         endif
      endif IF_BOURG3D

!VDIR NODEP
      DO_NI: do i = 1,ni

         !taux des precipitations de la convection profonde
         zry(i) = ztsc(i) + ztlc(i)

         !taux des precipitations de la convection restreinte
         zrz(i) = ztscs(i) + ztlcs(i)

         !taux des precipitations implicites
         zrc(i) = zry(i) + zrz(i)

         !Precipitation types from Milbrandt-Yau cloud microphysics scheme:

         IF_MY2: if (stcond(1:6) == 'MP_MY2')  then

            !tls:  rate of liquid precipitation (sum of rain and drizzle)
            ztls(i) = ztls_rn1(i) + ztls_rn2(i) + ztls_fr1(i) + ztls_fr2(i)

            !tss:  rate of solid precipitation (sum of all fozen precipitation types)
            ztss(i) = ztss_sn1(i) + ztss_sn2(i) +ztss_sn3(i) &
                 + ztss_pe1(i) + ztss_pe2(i)

            !tss_mx:  rate of mixed precipitation (liquid and solid simultaneously [>0.01 mm/h each])
            if (ztls(i) > 2.78e-9 .and. ztss(i) > 2.78e-9) then   ![note: 2.78e-9 m/s = 0.01 mm/h]
               ztss_mx(i) = ztls(i) + ztss(i)
            else
               ztss_mx(i) = 0.
            endif

            !als_rn1:  accumulation of liquid drizzle
            zals_rn1(i) = zals_rn1(i) + ztls_rn1(i) * dt

            !als_rn2:  accumulation of liquid rain
            zals_rn2(i) = zals_rn2(i) + ztls_rn2(i) * dt

            !als_fr1:  accumulation of freezing drizzle
            zals_fr1(i) = zals_fr1(i) + ztls_fr1(i) * dt

            !als_fr2:  accumulation of freezing rain
            zals_fr2(i) = zals_fr2(i) + ztls_fr2(i) * dt

            !ass_sn1:  accumulation of ice crystals
            zass_sn1(i) = zass_sn1(i) + ztss_sn1(i) * dt

            !ass_sn2:  accumulation of snow
            zass_sn2(i) = zass_sn2(i) + ztss_sn2(i) * dt

            !ass_sn3:  accumulation of graupel
            zass_sn3(i) = zass_sn3(i) + ztss_sn3(i) * dt

            !ass_pe1:  accumulation of ice pellets
            zass_pe1(i) = zass_pe1(i) + ztss_pe1(i) * dt

            !ass_pe2:  accumulation of hail
            zass_pe2(i) = zass_pe2(i) + ztss_pe2(i) * dt

            !ass_pe2l:  accumulation of hail (large only)
            zass_pe2l(i) = zass_pe2l(i) + ztss_pe2l(i) * dt

            !ass_snd:  accumulation of total unmelted snow (i+s+g)
            zass_snd(i) = zass_snd(i) + ztss_snd(i) * dt

            !ass_mx:  accumulation of mixed precipitation
            zass_mx(i) = zass_mx(i) + ztss_mx(i) * dt

            !  ass_s2l:  solid-to-liquid ratio for accumulated total snow (i+s+g)
            TEMPO =  zass_sn1(i)+zass_sn2(i)+zass_sn3(i)
            if (TEMPO > 1.e-6) then
               zass_s2l(i) = zass_snd(i)/TEMPO
            else
               zass_s2l(i) = 0.
            endif

         endif IF_MY2

         !taux des precipitations, grid-scale condensation scheme
         zrr(i) = ztss(i) + ztls(i)

         !taux total
         zrt(i) = zrc(i) + zrr(i)

         !asc : accumulation des precipitations solides de la convection profonde
         zasc(i) = zasc(i) + ztsc(i) * dt

         !ascs : accumulation des precipitations solides de la convection restreinte
         zascs(i) = zascs(i) + ztscs(i) * dt

         !alc : accumulation des precipitations liquides de la convection profonde
         zalc(i) = zalc(i) + ztlc(i) * dt

         !alcs : accumulation des precipitations liquides de la convection restreinte
         zalcs(i) = zalcs(i) + ztlcs(i) * dt

         !ass : accumulation of total solid precipitation, grid-scale condensation scheme
         zass(i) = zass(i) + ztss(i) * dt

         !als : accumulation des precipitations liquides, grid-scale condensation scheme
         zals(i) = zals(i) + ztls(i) * dt

         !pc : accumulation des precipitations implicites
         zpc(i) = zalc(i) + zasc(i) + zalcs(i) + zascs(i)

         !py : accumulation des precipitations de la convection profonde
         zpy(i) = zalc(i) + zasc(i)

         !pz : accumulation des precipitations de la convection restreinte
         zpz(i) = zalcs(i) + zascs(i)

         !ae : accumulation des precipitations, grid-scale condensation scheme
         zae(i) = zals(i) + zass(i)

         !pr : accumulation des precipitations totales
         zpr(i) = zpc(i) + zae(i)

         if ( pcptype == 'BOURGE3D' .and. &
             (stcond == 'CONSUN' .or. stcond(1:6) == 'MP_MY2') ) then
            tempo    = ztls(i)+ztss(i)
            sol_stra = max(0.,zfneige2d(i,nk)*tempo)
            liq_stra = max(0.,tempo-sol_stra)
            tempo    = (ztlc(i)+ztlcs(i))+(ztsc(i)+ztscs(i))
            sol_conv = max(0.,zfneige2d(i,nk)*tempo)
            liq_conv = max(0.,tempo-sol_conv)
         elseif ( pcptype == 'BOURGE' .and. &
                  (stcond == 'CONSUN' .or. stcond(1:6) == 'MP_MY2') ) then
            tempo    = ztls(i)+ztss(i)
            sol_stra = max(0.,zfneige1d(i)*tempo)
            liq_stra = max(0.,tempo-sol_stra)
            tempo    = (ztlc(i)+ztlcs(i))+(ztsc(i)+ztscs(i))
            sol_conv = max(0.,zfneige1d(i)*tempo)
            liq_conv = max(0.,tempo-sol_conv)
         elseif (pcptype == 'NIL') then
            sol_stra = ztss(i)
            liq_stra = ztls(i)
            sol_conv = ztsc(i)+ztscs(i)
            liq_conv = ztlc(i)+ztlcs(i)
         else
            tempo    = ztls(i)+ztss(i)
            sol_stra = max(0.,zfneige1d(i)*tempo)
            liq_stra = max(0.,tempo-sol_stra)
            tempo    = (ztlc(i)+ztlcs(i))+(ztsc(i)+ztscs(i))
            sol_conv = max(0.,zfneige1d(i)*tempo)
            liq_conv = max(0.,tempo-sol_conv)
         endif

         !RN : ACCUMULATION DES PRECIPITATIONS de PLUIE
         !AZR: ACCUMULATION DES PRECIPITATIONS VERGLACLACANTES
         !AIP: ACCUMULATION DES PRECIPITATIONS RE-GELEES
         !SN : ACCUMULATION DES PRECIPITATIONS de neige


         MY2_EXPLICIT: if ( stcond(1:6) == 'MP_MY2' .and. pcptype == 'NIL') then

            sol_stra = ztss(i)
            liq_stra = ztls(i)
            tempo    = ztlc(i)+ztlcs(i)+ztsc(i)+ztscs(i)  !total (deep+shallow) convection
            sol_conv = max(0.,zfneige1d(i)*tempo)
            liq_conv = max(0.,tempo-sol_conv)
            tempo2   = zazr(i)
            zazr(i)  =  zazr(i) + (ztls_fr1(i) + ztls_fr2(i))*dt  !from MY2
            zrn (i)  =  zrn (i) + (ztls_rn1(i) + ztls_rn2(i))*dt  !from MY2
            !Add diagnostic portion of rain/freezing rain from convective schemes:
            if  (ztplus(i,nk) < tcdk) then
               zazr(i) = zazr(i) + (1.-zfip1d(i))*liq_conv*dt  !from cvt schemes
            else
               zrn (i) = zrn (i) + (1.-zfip1d(i))*liq_conv*dt  !from cvt schemes
            endif
            if (zazr(i) > tempo2) zzrflag(i) = 1.
            zaip(i) = zaip(i)          &
                 + ztss_pe1(i)*dt      &              !from MY2
                 + zfip1d(i)*tempo*dt                 !from convective schemes
            !note: Hail from MY2 (tss_pe2) is not included in total ice pellets (aip)
            zsn(i)  = zsn(i)                                 &
                 + (ztss_sn1(i)+ztss_sn2(i)+ztss_sn3(i))*dt  & !from MY2
                 + sol_conv*dt                                 !from convective schemes
            !note: contribution to SN from M-Y scheme is from ice, snow,
            !      and graupel, instantaneous solid-to-liquid ratio for
            !      pcp rate total snow (i+s+g):
            TEMPO =  ztss_sn1(i)+ztss_sn2(i)+ztss_sn3(i)
            if (TEMPO > 1.e-12) then
               ztss_s2l(i) = ztss_snd(i)/TEMPO
            else
               ztss_s2l(i) = 0.
            endif

         else

            !diagnostic partitioning of precipitation types:
            ! (total precipitation from all schemes)
            ! BOURGE and NIL
            if (pcptype == 'BOURGE' .or. pcptype == 'NIL')then
               tempo =  liq_stra+liq_conv
               if  (ztplus(i,nk) < tcdk) then
                  zazr(i) = zazr(i) + (1.-zfip1d(i))*tempo*dt
                  if (tempo > 0.) zzrflag(i) = 1.
               else
                  zrn(i) = zrn(i)  + (1.-zfip1d(i))*tempo*dt
               endif
               zaip(i) = zaip(i) + zfip1d(i)*tempo*dt
               zsn (i) = zsn (i) + (sol_stra+sol_conv)*dt
               !   BOURGE3D
            else if (pcptype == 'BOURGE3D')then
               tempo = liq_stra+liq_conv
               if  (ztplus(i,nk) < tcdk) then
                  zazr(i) = zazr(i) + (1.-zfip2d(i,nk))*tempo*dt
                  if (tempo > 0.) zzrflag(i) = 1.
               else
                  zrn(i) = zrn(i)  + (1.-zfip2d(i,nk))*tempo*dt
               endif
               zaip(i) = zaip(i) + zfip2d(i,nk)*tempo*dt
               zsn(i) = zsn(i) + (sol_stra+sol_conv)*dt
            endif

         endif MY2_EXPLICIT

      end do DO_NI

   endif IF_KOUNT_NOT_0

   if (refract) then
      call refractivity(d,f,v,dsiz,fsiz,vsiz,ni,nk)
   endif

   if (lightning_diag) then
      call lightning(d,f,v,dsiz,fsiz,vsiz,ni,nk)
   endif

   !****************************************************************
   !     AVERAGES
   !     --------

   !set averages to zero every moyhr hours
   IF_ACCUM_0: if (laccum) then

      !pre-calculate screen wind modulus

      do i = 1,ni
         uvs(i) = zudiag(i)*zudiag(i) &
              + zvdiag(i)*zvdiag(i)
      end do

      call VSSQRT(uvs,uvs,ni)

      IF_RESET_0: if (kount == 0 .or. lreset) then

         do i = 1, ni
            zfcmy   (i) = 0.0
            zfvmy   (i) = 0.0
            zkshalm (i) = 0.0
            ziwvm   (i) = 0.0
            ztlwpm  (i) = 0.0
            ztiwpm  (i) = 0.0
            zhrsmax (i) = zrhdiag(i)
            zhrsmin (i) = zrhdiag(i)
            zhusavg (i) = 0.0
            ztdiagavg(i)= 0.0
            zp0avg  (i) = 0.0
            zuvsavg (i) = 0.0
            zuvsmax (i) = uvs(i)
         end do

         do k = 1,nk
            do i = 1,ni
               !minimum and maximum temperature
               zttmin(i,k) = ztplus(i,k)
               zttmax(i,k) = ztplus(i,k)
               !minimum and maximum temperature tendencies
               ztadvmin(i,k) = ztadv(i,k)
               ztadvmax(i,k) = ztadv(i,k)
            end do
         enddo

         do k=1,nk-1
            do i = 1, ni

               zccnm   (i,k) = 0.0
               ztim    (i,k) = 0.0
               zt2m    (i,k) = 0.0
               zugwdm  (i,k) = 0.0
               zvgwdm  (i,k) = 0.0
               zugnom  (i,k) = 0.0
               zvgnom  (i,k) = 0.0
               ztgnom  (i,k) = 0.0
               zudifvm (i,k) = 0.0
               zvdifvm (i,k) = 0.0
               ztdifvm (i,k) = 0.0
               zqdifvm (i,k) = 0.0
               ztadvm  (i,k) = 0.0
               zuadvm  (i,k) = 0.0
               zvadvm  (i,k) = 0.0
               zqadvm  (i,k) = 0.0
               zqmetoxm(i,k) = 0.0
               zhushalm(i,k) = 0.0
               ztshalm (i,k) = 0.0
               zlwcm   (i,k) = 0.0
               ziwcm   (i,k) = 0.0
               zlwcradm(i,k) = 0.0
               ziwcradm(i,k) = 0.0
               zcldradm(i,k) = 0.0
               ztphytdm(i,k) = 0.0
               zhuphytdm(i,k)= 0.0
               zuphytdm(i,k) = 0.0
               zvphytdm(i,k) = 0.0

               !see condensation/convesction for the calculation of the
               !averages of the following arrays
               zzctem  (i,k) = 0.0
               zzstem  (i,k) = 0.0
               zzcqem  (i,k) = 0.0
               zzcqcem (i,k) = 0.0
               zzsqem  (i,k) = 0.0
               zzsqcem (i,k) = 0.0

            end do
         enddo

         !#Note: all convec str in the if below must have same len
         if (any(convec == (/ &
              'KFC     ',     &
              'BECHTOLD'/))) then

            do i = 1, ni
               zcapekfcm (i) = 0.0
               zwumkfcm  (i) = 0.0
               zzbaskfcm (i) = 0.0
               zztopkfcm (i) = 0.0
               zkkfcm    (i) = 0.0
            end do

            do k = 1,nk
               do i = 1, ni
                  ztfcpm  (i,k) = 0.0
                  zhufcpm (i,k) = 0.0
                  zqckfcm (i,k) = 0.0
                  zumfkfcm(i,k) = 0.0
                  zdmfkfcm(i,k) = 0.0
                  zudcm   (i,k) = 0.0
                  zvdcm   (i,k) = 0.0
                  zuscm   (i,k) = 0.0
                  zvscm   (i,k) = 0.0
               end do
            enddo

         endif

      endif IF_RESET_0

   endif IF_ACCUM_0


   IF_ACCUM_1: if (laccum .and. kount /= 0) then

      do i = 1, ni

         zfcmy    (i) = zfcmy   (i) + zfc_ag(i)
         zfvmy    (i) = zfvmy   (i) + zfv_ag(i)
         zkshalm  (i) = zkshalm (i) + zkshal(i)
         ziwvm    (i) = ziwvm   (i) + ziwv  (i)
         ztlwpm   (i) = ztlwpm  (i) + ztlwp (i)
         ztiwpm   (i) = ztiwpm  (i) + ztiwp (i)

         zhrsmax  (i) = max(zhrsmax  (i) , zrhdiag (i))
         zhrsmin  (i) = min(zhrsmin  (i) , zrhdiag (i))
         zhusavg  (i) =     zhusavg  (i) + zqdiag  (i)
         ztdiagavg(i) =     ztdiagavg(i) + ztdiag  (i)
         zp0avg   (i) =     zp0avg   (i) + zpplus  (i)
         zuvsavg  (i) =     zuvsavg  (i) + uvs     (i)
         zuvsmax  (i) = max(zuvsmax  (i) , uvs     (i))

         if (lavg) then
            zfcmy    (i) = zfcmy    (i) * moyhri
            zfvmy    (i) = zfvmy    (i) * moyhri
            zkshalm  (i) = zkshalm  (i) * moyhri
            ziwvm    (i) = ziwvm    (i) * moyhri
            ztlwpm   (i) = ztlwpm   (i) * moyhri
            ztiwpm   (i) = ztiwpm   (i) * moyhri

            zhusavg  (i) = zhusavg  (i) * moyhri
            ztdiagavg(i) = ztdiagavg(i) * moyhri
            zp0avg   (i) = zp0avg   (i) * moyhri
            zuvsavg  (i) = zuvsavg  (i) * moyhri
         endif

      end do

      do k = 1, nk
         do i = 1, ni
            !minimum and maximum temperature
            zttmin  (i,k) = min(zttmin  (i,k), ztplus(i,k))
            zttmax  (i,k) = max(zttmax  (i,k), ztplus(i,k))
            !minimum and maximum temperature tendencies
            ztadvmin(i,k) = min(ztadvmin(i,k), ztadv (i,k))
            ztadvmax(i,k) = max(ztadvmax(i,k), ztadv (i,k))
         end do
      enddo

      do k = 1, nk-1
         do i = 1, ni

            zccnm   (i,k) = zccnm   (i,k) + zftot(i,k)
            ztim    (i,k) = ztim    (i,k) + zti  (i,k)
            zt2m    (i,k) = zt2m    (i,k) + zt2  (i,k)
            zugwdm  (i,k) = zugwdm  (i,k) + zugwd(i,k)
            zvgwdm  (i,k) = zvgwdm  (i,k) + zvgwd(i,k)
            zugnom  (i,k) = zugnom  (i,k) + zugno(i,k)
            zvgnom  (i,k) = zvgnom  (i,k) + zvgno(i,k)
            ztgnom  (i,k) = ztgnom  (i,k) + ztgno(i,k)
            zudifvm (i,k) = zudifvm (i,k) + zudifv(i,k)
            zvdifvm (i,k) = zvdifvm (i,k) + zvdifv(i,k)
            ztdifvm (i,k) = ztdifvm (i,k) + ztdifv(i,k)
            zqdifvm (i,k) = zqdifvm (i,k) + zqdifv(i,k)
            ztadvm  (i,k) = ztadvm  (i,k) + ztadv (i,k)
            zuadvm  (i,k) = zuadvm  (i,k) + zuadv (i,k)
            zvadvm  (i,k) = zvadvm  (i,k) + zvadv (i,k)
            zqadvm  (i,k) = zqadvm  (i,k) + zqadv (i,k)
            zqmetoxm(i,k) = zqmetoxm(i,k) + zqmetox(i,k)
            zhushalm(i,k) = zhushalm(i,k) + zhushal(i,k)
            ztshalm (i,k) = ztshalm (i,k) + ztshal(i,k)
            zlwcm   (i,k) = zlwcm   (i,k) + zlwc(i,k)
            ziwcm   (i,k) = ziwcm   (i,k) + ziwc(i,k)
            zlwcradm(i,k) = zlwcradm(i,k) + zlwcrad(i,k)
            ziwcradm(i,k) = ziwcradm(i,k) + ziwcrad(i,k)
            zcldradm(i,k) = zcldradm(i,k) + zcldrad(i,k)
            ztphytdm(i,k) = ztphytdm(i,k) + ztphytd(i,k)
            zhuphytdm(i,k)= zhuphytdm(i,k)+ zhuphytd(i,k)
            zuphytdm(i,k) = zuphytdm(i,k) + zuphytd(i,k)
            zvphytdm(i,k) = zvphytdm(i,k) + zvphytd(i,k)

            IF_AVG_1: if (lavg) then
               zccnm   (i,k) = zccnm   (i,k) * moyhri
               ztim    (i,k) = ztim    (i,k) * moyhri
               zt2m    (i,k) = zt2m    (i,k) * moyhri
               zugwdm  (i,k) = zugwdm  (i,k) * moyhri
               zvgwdm  (i,k) = zvgwdm  (i,k) * moyhri
               zugnom  (i,k) = zugnom  (i,k) * moyhri
               zvgnom  (i,k) = zvgnom  (i,k) * moyhri
               ztgnom  (i,k) = ztgnom  (i,k) * moyhri
               zudifvm (i,k) = zudifvm (i,k) * moyhri
               zvdifvm (i,k) = zvdifvm (i,k) * moyhri
               ztdifvm (i,k) = ztdifvm (i,k) * moyhri
               zqdifvm (i,k) = zqdifvm (i,k) * moyhri
               ztadvm  (i,k) = ztadvm  (i,k) * moyhri
               zuadvm  (i,k) = zuadvm  (i,k) * moyhri
               zvadvm  (i,k) = zvadvm  (i,k) * moyhri
               zqadvm  (i,k) = zqadvm  (i,k) * moyhri
               zqmetoxm(i,k) = zqmetoxm(i,k) * moyhri
               zzctem  (i,k) = zzctem  (i,k) * moyhri
               zzcqem  (i,k) = zzcqem  (i,k) * moyhri
               zzcqcem (i,k) = zzcqcem (i,k) * moyhri
               zzstem  (i,k) = zzstem  (i,k) * moyhri
               zzsqem  (i,k) = zzsqem  (i,k) * moyhri
               zzsqcem (i,k) = zzsqcem (i,k) * moyhri
               zhushalm(i,k) = zhushalm(i,k) * moyhri
               ztshalm (i,k) = ztshalm (i,k) * moyhri
               zlwcm   (i,k) = zlwcm   (i,k) * moyhri
               ziwcm   (i,k) = ziwcm   (i,k) * moyhri
               zlwcradm(i,k) = zlwcradm(i,k) * moyhri
               ziwcradm(i,k) = ziwcradm(i,k) * moyhri
               zcldradm(i,k) = zcldradm(i,k) * moyhri
               ztphytdm(i,k) = ztphytdm(i,k) * moyhri
               zhuphytdm(i,k)= zhuphytdm(i,k)* moyhri
               zuphytdm(i,k) = zuphytdm(i,k) * moyhri
               zvphytdm(i,k) = zvphytdm(i,k) * moyhri
            endif IF_AVG_1

         end do
      end do

      !#Note: all convec str in the if below must have same len
      if (any(convec == (/ &
           'KFC     ',     &
           'BECHTOLD'/))) then
         do k = 1, nk-1
            do i = 1, ni
               ztfcpm  (i,k) = ztfcpm  (i,k) + ztfcp (i,k)
               zhufcpm (i,k) = zhufcpm (i,k) + zhufcp(i,k)
               zqckfcm (i,k) = zqckfcm (i,k) + zqckfc(i,k)
               zumfkfcm(i,k) = zumfkfcm(i,k) + zumfkfc(i,k)
               zdmfkfcm(i,k) = zdmfkfcm(i,k) + zdmfkfc(i,k)
               zudcm   (i,k) = zudcm   (i,k) + zufcp  (i,k)
               zvdcm   (i,k) = zvdcm   (i,k) + zvfcp  (i,k)
            enddo
         enddo
         if (associated(ztusc)) zuscm(:,1:nk-1) = zuscm(:,1:nk-1) + ztusc(:,1:nk-1)
         if (associated(ztvsc)) zvscm(:,1:nk-1) = zvscm(:,1:nk-1) + ztvsc(:,1:nk-1)
         IF_AVG_2: if (lavg) then
            do k = 1, nk-1
               do i = 1, ni
                  ztfcpm  (i,k) = ztfcpm  (i,k) * moyhri
                  zhufcpm (i,k) = zhufcpm (i,k) * moyhri
                  zqckfcm (i,k) = zqckfcm (i,k) * moyhri
                  zumfkfcm(i,k) = zumfkfcm(i,k) * moyhri
                  zdmfkfcm(i,k) = zdmfkfcm(i,k) * moyhri
                  zudcm   (i,k) = zudcm   (i,k) * moyhri
                  zvdcm   (i,k) = zvdcm   (i,k) * moyhri
                  zuscm   (i,k) = zuscm   (i,k) * moyhri
                  zvscm   (i,k) = zvscm   (i,k) * moyhri
               end do
            end do
         endif IF_AVG_2

         do i=1, ni
            zcapekfcm (i) = zcapekfcm(i) + zcapekfc(i)
            zwumkfcm  (i) = zwumkfcm(i) + zwumaxkfc(i)
            zzbaskfcm (i) = zzbaskfcm(i) + zzbasekfc(i)
            zztopkfcm (i) = zztopkfcm(i) + zztopkfc(i)
            zkkfcm    (i) = zkkfcm(i) + zkkfc(i)

            if (lavg) then
               zcapekfcm (i) = zcapekfcm (i) * moyhri
               zwumkfcm(i) = zwumkfcm(i) * moyhri
               zzbaskfcm(i) = zzbaskfcm(i) * moyhri
               zztopkfcm(i) = zztopkfcm(i) * moyhri
               zkkfcm(i) = zkkfcm(i) * moyhri
            endif

         end do
      endif

   endif IF_ACCUM_1


   !****************************************************************
   !     ACCUMULATORS
   !     ------------

   !Set accumulators to zero at the beginning and after every acchr hours,
   !and by default (acchr=0) as the model step goes through 0.
   IF_RESET_ACCUMULATORS: if (kount == 0. .or. (acchr > 0 .and. mod((step_driver-1),acchr) == 0) .or. &
        (acchr == 0 .and. step_driver-1 == 0)) then
      do i = 1,ni
         zrainaf(i) = 0.
         zsnowaf(i) = 0.
      enddo
      if (radia /= 'NIL' .or. fluvert == 'SURFACE') then
         do i = 1,ni
            zeiaf    (i) = 0.
            zevaf    (i) = 0.
            zfiaf    (i) = 0.
            zfsaf    (i) = 0.
            zivaf    (i) = 0.
            zntaf    (i) = 0.
            zflusolaf(i) = 0.
         enddo
      endif
      if (radia(1:8) == 'CCCMARAD') then
         do i=1,ni
            zclbaf    (i) = 0.
            zcltaf    (i) = 0.
            zcstaf    (i) = 0.
            zcsbaf    (i) = 0.
            zfsdaf    (i) = 0.
            zfsfaf    (i) = 0.
            zfsiaf    (i) = 0.
            zfsvaf    (i) = 0.
            zparraf   (i) = 0.
         enddo
      endif
      if (fluvert /= 'NIL') then
         do i = 1,ni
            zsuaf (i) = 0.
            zsvaf (i) = 0.
            zfqaf (i) = 0.
            zsiaf (i) = 0.
            zflaf (i) = 0.
            zfcaf (i) = 0.
            zfvaf (i) = 0.
         enddo
      endif
      if (lightning_diag) then
         do i = 1,ni
            zafoudre (i) = 0.
         enddo
      endif
   endif IF_RESET_ACCUMULATORS

   IF_KOUNT_NOT_0b: if (kount /= 0) then

!VDIR NODEP
      DO_NI_ACC: do i = 1,ni

         !Accumulation of precipitation (in m)

         zrainaf(i) = zrainaf(i) + zrainrate(i)*dt
         zsnowaf(i) = zsnowaf(i) + zsnowrate(i)*dt

         if (radia /= 'NIL' .or. fluvert == 'SURFACE') then
            zeiaf    (i) = zeiaf (i) + zei  (i) * dt
            zevaf    (i) = zevaf (i) + zev  (i) * dt
            zfiaf    (i) = zfiaf (i) + zfdsi(i) * dt
            zfsaf    (i) = zfsaf (i) + zfdss(i) * dt
            zivaf    (i) = zivaf (i) + ziv  (i) * dt
            zntaf    (i) = zntaf (i) + znt  (i) * dt
            zflusolaf(i) = zflusolaf(i) + &
                 zflusolis(i) * dt
         endif

         !Accumulation of sfc and toa net clear sky fluxes, available with cccmarad

         if (radia(1:8) == 'CCCMARAD') then
            zclbaf    (i) = zclbaf (i) + zclb  (i) * dt
            zcltaf    (i) = zcltaf (i) + zclt  (i) * dt
            zcstaf    (i) = zcstaf (i) + zcstt (i) * dt
            zcsbaf    (i) = zcsbaf (i) + zcsb  (i) * dt
            zfsdaf    (i) = zfsdaf (i) + zfsd  (i) * dt
            zfsfaf    (i) = zfsfaf (i) + zfsf  (i) * dt
            zfsiaf    (i) = zfsiaf (i) + zfsi  (i) * dt
            zfsvaf    (i) = zfsvaf (i) + zfsv  (i) * dt
            zparraf    (i) = zparraf (i) + zparr  (i) * dt
         endif

         if (fluvert /= 'NIL') then
            zsuaf (i) = zsuaf (i) + zustress(i) * dt
            zsvaf (i) = zsvaf (i) + zvstress(i) * dt
            zfqaf (i) = zfqaf (i) + zfq  (i) * dt
            zsiaf (i) = zsiaf (i) + zfnsi(i) * dt
            zflaf (i) = zflaf (i) + zfl  (i) * dt
            zfcaf (i) = zfcaf (i) + &
                 zfc_ag(i) * dt
            zfvaf (i) = zfvaf (i) + &
                 zfv_ag(i) * dt
         endif


         if (lightning_diag) then
            !Accumulation of lightning threat (in number of flashes/m2)
            zafoudre(i) = zafoudre(i) + zfoudre(i) *  dt
         end if

      end do DO_NI_ACC

   endif IF_KOUNT_NOT_0B

   !# For output purpose, diag level values needs to be copied into nk level of corresponding dynbus var
   zhuplus(:,nk) = zqdiag
   ztplus(:,nk)  = ztdiag
   zuplus(:,nk)  = zudiag
   zvplus(:,nk)  = zvdiag
   !----------------------------------------------------------------
   return
end subroutine calcdiag
