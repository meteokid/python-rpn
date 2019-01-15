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
!** S/P CCCMARAD - DRIVER ROUTINE FOR RADIATION

      subroutine ccc2_cccmarad (d, dsiz, f, fsiz, v, vsiz , &
                           temp, qq, ps, sig, &
                           tau, kount, icpu , &
                           trnch , n , m , nk , nkp, &
                           liqwcin, icewcin, liqwpin, icewpin, cldfrac)
      use sfclayer_mod, only: sl_prelim,sl_sfclayer,SL_OK
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>

      integer dsiz, fsiz, kount, trnch, vsiz, n, m, nk, nkp
      integer it, icpu, istat
      real, target :: d(dsiz), f(fsiz), v(vsiz)
      real temp(m,nkp), qq(m,nk), ps(n), sig(n,nk+1)
      real liqwcin(n,nk), icewcin(n,nk), cldfrac(n*nk)
      real liqwpin(m,nk), icewpin(m,nk)
      real tau

!Authors
!        p. vaillancourt, d. talbot, j. li, rpn, cmc, cccma; (may 2006)
!
!Revisions
! 001      see version 5.5.0 for previous history
!
!Object
!        prepares all inputs for radiative transfer scheme
!        (cloud optical properties, trace gases, ozone, aerosols..)
!        executes ccc radiative transfer for infrared and solar radiation
!
!Arguments
!
!          - input/output -
! f        field of permanent physics variables
! fsiz     dimension of f
!
!          - input -
! temp     temperature
! qq       specific humidity
! ps       surface pressure
! sig      sigma levels
! tau      timestep
! kount    number of timesteps
! icpu     task number
! kntrad   frequency of call for infra-red radiation
! trnch    index of the vertical plane (ni*nk) for which
!          calculations are to be done.
! n        horizontal dimension
! m        1st dimension of t and q
! nk       number of layers
! nkp      number of flux levels (nk+1)
! liqwcin  in-cloud liquid water content (kg/kg)
! icewcin  in-cloud ice    water content (kg/kg)
! liqwpin  in-cloud liquid water path (g/m^2)
! icewpin  in-cloud ice    water path (g/m^2)
! cldfrac  cloud fraction (0.-1.)
!
! Notes
!          cccmarad produces:
!          infra-red rate (ti) of cooling
!          shortwave rate (t2) of heating
!          shortwave flux to ground (fdss)
!          infra-red flux to ground (fdsi)
!          infra-red flux to the top of the atmosphere (ei)
!          shortwave flux to the top of the atmosphere (ev)
!          planetary albedo (ap=ev/incident solar flux)
!
! BEWARE :
! Remove comments to the lines at the end of preintp if pressure at model
! top is less than .0005 Pa
! When pressure a model top is less than 10 hPa then minor bands are used
! These variables change values for different topology but do not impact
! on validation for different topology : maxc lev1 ncum ncdm in cldifm
! mcont               in raddriv
! lstart              in qozon3

!Implicites

      include "thermoconsts.inc"

#include "surface.cdk"
#include "clefcon.cdk"
#include "ozopnt.cdk"
#include "radiation.cdk"
#include "nbsnbl.cdk"
#include "ccc_tracegases.cdk"
#include "tables.cdk"

! ISCCP
#include "mcica.cdk"
      logical, external :: series_isstep

      real, parameter :: seuil = 1.e-3

      external ccc2_ckdlw,ccc2_ckdsw,ccc_dataero,ccc_tracedata

      real, external :: juliand

      real :: julien, r0r

!     pointeurs des variables volatiles de la radiation
!     determines par une routine de gestion de memoire
!
!*********************************************************
!     Automatic arrays
!*********************************************************
!
      logical,dimension(n)     :: thold
      integer,dimension(n)     :: p7,p8
      real,dimension(n)        :: p1,p3,p4,p5,p6,pbl,albpla,fdl,ful,fslo,&
                                  rmu0,v1,ws,ws_vs,cosas_vs
      real,dimension(nkp)      :: p10,p11
      real,dimension(n*npcl)   :: p2
      real,dimension(n,nkp)    :: shtj,tfull,s_qrt
      real,dimension(n,nk)     :: co2,ch4,an2o,f11,af12,f113,f114,o2
      real,dimension(n,nbs)    :: salb
      real,dimension(n,nk,5)   :: tauae
      real,dimension(n,nk,nbs) :: exta,exoma,exomga,fa,taucs,omcs,gcs
      real,dimension(n,nk,nbl) :: absa,taucl,omcl,gcl
!
!*******************************************************
      integer*8 ncsec_deb, ncsec_now, timestep, csec_in_day, day_reminder
      real*8 hz_8
      real hz, hzp, ptop, ptopoz, alwcap, fwcap, albrmu
      integer i, k, l, iuv
      logical lcsw, lclw, aerosolback
      integer il1,il2
      character(len=1) :: niuv

      real dummy1(n),dummy2(n),dummy3(n),dummy4(n)
      real, dimension(n) :: vmod2,vdir,th_air,my_tdiag,my_udiag,my_vdiag

!
! isccp
!
      real :: &
       liqwcin_s(n,nk,nx_loc),  &! subcolumns of cloud liquid water
       icewcin_s(n,nk,nx_loc)    ! subcolumns of cloud ice water
!
      real :: &
       sigma_qcw(n,nk),         &! std. dev. of cloud water/mean cloud water
       rlc_cf(n,nk),            &! decorelation length for cloud amount (km)
       rlc_cw(n,nk),            &! decorrelation length for cloud condensate(km)
       cldtot(n)                 ! total cloud fraction as computed using
                                 ! stochastic cloud generator
!
      integer :: &
       ncldy(n),                &! number of cloudy subcolumns
       iseed(n)                  ! integer pseudo-random number seed
!
      real :: &
       rseed                     ! real pseudo-random number seed
!
#include "solcons.cdk"
!
      data lcsw, lclw, aerosolback / .true., .true., .true./
!
include "cccmarad_ptr.cdk"
!
include "cccmarad_ptr_as.cdk"
!
!  use integer variables instead of actual integers
!
      il1=1
      il2=n
!
      it = icpu
!
      csec_in_day=8640000
      timestep= nint(tau*100.)
      ncsec_deb= date(5)*360000 + DATE(6)
      ncsec_now= ncsec_deb + kount*timestep
      day_reminder= mod(ncsec_now, csec_in_day)
      hz_8 = day_reminder / 360000.0d0
      hz   = hz_8
!
!...  redefine co2, ch4, n2o, f11 and f12 concentrations
!... following corresponding parameters from /OPTIONR/
!
      co2_ppm = qco2     * 1.e-6
      rmco2   =  co2_ppm * 44d0     / 28.97
!
      ch4_ppm = qch4     * 1.e-6
      rmch4   =  ch4_ppm * 16.00d0  / 28.97
!
      n2o_ppm = qn2o     * 1.e-6
      rmn2o   =  n2o_ppm * 44.00d0  / 28.97
!
      f11_ppm = qcfc11   * 1.e-9
      rmf11   = f11_ppm  * 137.37d0 / 28.97
!
      f12_ppm = qcfc12   * 1.e-9
      rmf12   = f12_ppm  * 120.91d0 / 28.97
!
      do k = 1, nk
        do i = 1, n
          co2(i,k)=rmco2
          ch4(i,k)=rmch4
          an2o(i,k)=rmn2o
          f11(i,k)=rmf11
          af12(i,k)=rmf12
          f113(i,k)=rmf113
          f114(i,k)=rmf114
          o2(i,k)=rmo2
        enddo
      enddo

      zt2   = 0.0
      zfdss = 0.0
      zev   = 0.0
      zflusolis = 0.0
      zfsd  = 0.0
      zfsf  = 0.0
      zfsv  = 0.0
      zfsi  = 0.0
      zparr = 0.0
!
!...    calculate the variation of solar constant
!
        julien = juliand(tau, kount, date)
        alf = julien / 365. * 2 * pi
        r0r = solcons(alf)
!
!...    cosine of solar zenith angle at greenwich hour
!
        call suncos1(rmu0, dummy1, dummy2, dummy3, dummy4, n, &
                     f(dlat), f(dlon), hz, julien, date, .false.)
!
!...    calculate cloud optical properties and dependent diagnostic
!       cloud variables
!...    such as cloud cover, effective and true; cloud top temp and pressure
!...    called every timestep
!
        call cldoppro3 (taucs, omcs, gcs, taucl, omcl, gcl, &
                       f(topthw), f(topthi), f(ecc),f(tcc), &
                       f(eccl), f(eccm), f(ecch), &
                       v(ctp), v(ctt), liqwcin, icewcin, &
                       liqwpin, icewpin, cldfrac, &
                       temp, sig, ps, f(mg), f(ml), m, &
                       n, nk, nkp)


!...  pour les pas de temps radiatifs
      if (kount == 0 .or. mod(kount-1, kntrad) .eq. 0)          then
!
! Local estimate of screen level temperature and wind
        if (kount == 0 )          then
            my_tdiag = ztdiag
            my_udiag = zudiag
            my_vdiag = zvdiag
        else
            i = sl_prelim(ztmoins(:,nk),zhumoins(:,nk),zumoins(:,nk),zvmoins(:,nk), &
            zp0_moins,zzusl,spd_air=vmod2,dir_air=vdir,min_wind_speed=VAMIN)
            if (i /= SL_OK) then
             print*, 'Aborting in cccmarad() because of error returned by sl_prelim()'
             stop
            endif

            th_air = ztmoins(:,nk)*zsigt(:,nk)**(-cappa)
            i = sl_sfclayer(th_air,zhumoins(:,nk),vmod2,vdir,zzusl,zztsl,ztsrad,zqsurf, &
            zz0_ag,zz0t_ag,zdlat,zfcor,hghtm_diag=10.,hghtt_diag=1.5,t_diag=my_tdiag, &
            u_diag=my_udiag,v_diag=my_vdiag)
            if (i /= SL_OK) then
              print*, 'Aborting in cccmarad() because of error returned by sl_sfclayer()'
              stop
            endif
        endif
!
!...    calculte sigma(shtj) and temperature(tfull) at flux levels
!
        do i = 1, n
           s_qrt(i,1) = sig(i,1) / sig(i,2)
           s_qrt(i,nkp) = 1.0
! The following line extrapolates the temperature above model top
! for moon layer temperature
!           tfull(i,1) = 0.5 * (3.0 * temp(i,1) - temp(i,2))
! The following line assumes temperature is isothermal above model top
! This assumption must also be imposed in raddriv (see calc of a1(i,5)
! and planck subroutines
           tfull(i,1) =  temp(i,1)
! Choose boundary condition for LW down flux very near the surface
! High vertical resolution SCM tests suggest that average of 2m and ground temperature causes less problems
           tfull(i,nkp) = 0.5*(my_tdiag(i)+f(tsrad+i-1))
!           tfull(i,nkp) = my_tdiag(i)
!           tfull(i,nkp) = f(tsrad+i-1)
        enddo
        do k = 2, nk
          do i = 1, n
            s_qrt(i,k) = sig(i,k-1) * sig(i,k)
            tfull(i,k) = 0.5 * (temp(i,k-1) + temp(i,k))
          enddo
        enddo

        call vssqrt (shtj,s_qrt,n*nkp)

        do i = 1, n
          shtj(i,1)  = sig(i,1) * shtj(i,1)
        enddo
!
!
!...    calculate aerosol optical properties
!
        do i = 1, n
            pbl(i) = 1500.0
        enddo
        call ccc_aerooppro (tauae,exta,exoma,exomga,fa,absa, &
                        temp,shtj,sig,ps,f(dlat),f(mg), f(ml),pbl, &
                        aerosolback,il1, il2, n, nk, nkp )
!
!...    from ozone zonal monthly climatology: interpolate to proper date
!       and grid, calculate total amount above model top (ptop)

        call radfac3 (f(o3s),f(oztoit),sig,nkp,nk,npcl,f(dlat),ps,n,n, &
                      nkp, p2, p3, p4, p5, p6, p7, p8, p10, p11, nlacl, &
                      goz(fozon), goz(clat), goz(pref))
!
!       must modify oztoit to fit the needs of raddriv who expects an average
!       mixing ratio rather than an integral (convert cm back to kg/kg)
        do i = 1, n
!          ptop = sig(i,1)*ps(i)
           ptopoz = -10.0
!          look for ozone reference pressure level closest to model top
           do k = 0, npcl-1
              if (goz(pref+k) .lt. std_p_prof(1)) then
                  ptopoz = goz(pref+k)
              endif
           enddo
           if (ptopoz.gt.0.0) zoztoit(i)=zoztoit(i)* &
                                         grav*2.144e-2/ptopoz
        enddo
!
!...    calculate cosine of solar zenith angle at kount + kntrad - 1
!
        julien = juliand(tau, kount + kntrad - 1, date)
        ncsec_now= ncsec_deb + (kount + kntrad - 1)*timestep
	day_reminder= mod(ncsec_now, csec_in_day)
        hz_8 = day_reminder / 360000.0d0
        hzp = hz_8

        call suncos1(f(cosas), dummy1, dummy2, dummy3, dummy4, n, &
                     f(dlat), f(dlon), hzp, julien, date, .false.)
!
        do i = 1, n
!...      albedo (6% to 80%), temporally set the same for all 4 band
          salb(i,1) = amax1 (amin1 (zalvis_ag(i), 0.80), 0.06)
!         f(salb6z+i-1) = salb(i,1)
          zsalb6z(i) = 0.0
          do l = 2, nbs
             salb(i,l) = salb(i,1)
          enddo
!
!...      adjust the cosine of solar zenith angle to radition call time
          zcosas(i) = (rmu0(i) + zcosas(i)) * 0.5
!
        enddo

!
!----------------------------------------------------------------------------------
!       open water albedo adjusted for solar angle and white caps,
!       fwcap is fraction of white caps, alwcap is albedo of white caps
!       ws is the 10m wind speed, f(cosas) is cosine of solar zenith angle,
!       albrmu is albedo corrected for solar zenith angle
!       ref for white cap effect is : monahan et al., 1980, jpo, 10,2094-2099
!       ref for solar angle dependence  : taylor et al., 1996, qjrms,122,839-861
!       danger: if this code is accepted, it should migrate to where the
!       agregated albedo is calculated. furthermore, the 10m wind speed
!       should be recalculated when needed, present ws comes from the beginning
!       of the previous time step, rather than the end
!----------------------------------------------------------------------------------

        ws_vs=my_udiag*my_udiag+my_vdiag*my_vdiag

        call vspown1(ws, ws_vs, 1.705, n)
        call vspown1(cosas_vs, zcosas, 1.4, n)
        alwcap = 0.3
        do i = 1, n
!       au pas de temps zero f(glsea) n est pas defini car
!       la radiation est faite avant la sfc
           if (zmg(i) .le. 0.01 .and. zglsea(i) .le. 0.01 .and. &
               zml(i) .le. 0.01 .and. zcosas(i) .gt. seuil ) then
             fwcap      = amin1 (3.84e-06 * ws(i), 1.0)
             albrmu     = 0.037 / (1.1 * cosas_vs(i) + 0.15)
             salb(i,1)  = (1.-fwcap) * albrmu + fwcap * alwcap
             salb(i,1)  = amax1 (amin1 (salb(i,1), 0.80), 0.03)
             zsalb6z(i) = salb(i,1)
             do l = 2, nbs
                salb(i,l) = salb(i,1)
             enddo
           endif
        enddo
!
        if (simisccp) then
!
! ISCCP
!
! seed random number generator
!
           do i = 1, n
!
! generate the random number based on local latitude,longitude,hour
! and julien day.  created so that the size of the seed should not
! exceed 2^31-1.  if it does then there will be problems.
!
              rseed = 1.0e5*((zdlat(i)+(pi/2.0))*2.0*pi+ zdlon(i)) &
                    + hz*1.0e6 &
                    + julien*100.0
!
              iseed(i) = int(rseed)
!
           end do
!
!          call random_seed(generator=2) ! specific to ibm
           call random_seed(put=iseed)
!
! define the cloud overlap parameters and horizontal variability
!
           call prep_mcica(rlc_cf, rlc_cw, sigma_qcw, cldfrac, n,il1,il2,nk)
!
! generate sub-olumns of liquid and ice water contents
!
           call mcica_cld_gen(cldfrac, liqwcin, icewcin, rlc_cf, rlc_cw, &
                              sigma_qcw, temp, sig, ps, n, il1, il2, nk, &
                              ncldy, liqwcin_s, icewcin_s, cldtot)
!
! call the ISCCP simulator
!
           call isccp_sim_driver( &
                        f(itp), f(ictp), f(itau), f(icep), f(itcf),  &! output
                        f(isun), &
                        liqwcin_s, icewcin_s, ps, sig, shtj,         &! input
                        il1, il2, n, nk, nkp, &
                        f(cosas), f(tsrad), temp, qq, f(mg), f(ml))
!
        endif
!
!       actual call to the Li & Barker (2005) radiation
!
        call ccc2_raddriv (f(fsg),f(fsd0),f(fsf0),f(fsv0),f(fsi0), &
                      v(fatb),v(fadb),v(fafb),v(fctb),v(fcdb),v(fcfb), &
                      albpla,fdl,ful,f(t20), f(ti), &
                      f(cstt),f(csb),f(clt),f(clb),f(parr0), &
                      f(fluxds0),f(fluxus0),f(fluxdl),f(fluxul), &
                      fslo, f(fsamoon), ps, shtj, sig, &
                      tfull, temp, f(tsrad), f(o3s),f(oztoit), &
                      qq, co2, ch4, an2o, f11, &
                      af12,f113,f114,o2,f(cosas), r0r, salb, taucs, &
                      omcs, gcs, taucl, omcl, gcl, &
                      cldfrac, tauae, exta, exoma, exomga, &
                      fa, absa, lcsw, lclw, &
                      il1, il2, n, nk, nkp)
!
!       ti (t2): infrared (solar) cooling (heating) rate
!       fdsi (fdss): infrared (solar) downward flux at surface.
!       ei (ev): infrared (solar) upward flux at toa
!       ap: albedo planetaire.
!
        thold=(zcosas .gt. seuil .and. rmu0 .gt. seuil)
!
        do 1100 i = 1, n
          zfdsi(i)  = fdl(i)
          zei(i)    = ful(i)
          zfdss0(i) = zfsg(i)
          zev0(i)   = CONSOL2 * r0r * zcosas(i) * albpla(i)
!
!...      moduler les flux et les taux par le cosinus de l'angle solaire.
!...      rapport des cosinus : angle actuel sur angle moyen.
!
          v1(i) = rmu0(i) / zcosas(i)
          v1(i) = min(v1(i),2.0)
          zvv1(i)= v1(i)
          if(thold(i))then
            zfdss(i)     = zfdss0(i)         * v1(i)
            zev(i)       = zev0(i)           * v1(i)
            zflusolis(i) = (zfsd0(i)+zfsf0(i))  * v1(i)
            zfsd(i)      = zfsd0(i)          * v1(i)
            zfsf(i)      = zfsf0(i)          * v1(i)
            zfsv(i)      = zfsv0(i)          * v1(i)
            zfsi(i)      = zfsi0(i)          * v1(i)
            zparr(i)     = zparr0(i)         * v1(i)
            zfluxds(i,nkp)       = zfluxds0(i,nkp)       * v1(i)
            zfluxus(i,nkp)       = zfluxus0(i,nkp)       * v1(i)
          endif
 1100   continue
!
        do 1200 k=1,nk
           do 1200 i=1,n
             if(thold(i))then
                zt2(i,k)     = zt20(i,k) * v1(i)
                zfluxds(i,k) = zfluxds0(i,k) * v1(i)
                zfluxus(i,k) = zfluxus0(i,k) * v1(i)
             endif
 1200   continue

!where (zcosas.gt. seuil .and. rmu0 .gt. seuil)
!
!
!...    in case mod(kount-1,kntrad) non zero
!
      else
!
!...    ajustement du solaire aux pas non multiples de kntrad par
!       modulation avec cosinus de l'angle solaire
!
!...    moduler par le cosinus de l'angle solaire. mettre a zero les
!       valeurs appropriees de fdss, ev et t2.
!
       thold=(zcosas .gt. seuil .and. rmu0 .gt. seuil)
       do 1300 i=1,n
          v1(i) = rmu0(i) / zcosas(i)
          v1(i) = min(v1(i),2.0)
          zvv1(i)= v1(i)
        if(thold(i))then
            zfdss(i)     = zfdss0(i)         * v1(i)
            zev(i)       = zev0(i)           * v1(i)
            zflusolis(i) = (zfsd0(i)+zfsf0(i))  * v1(i)
            zfsd(i)      = zfsd0(i)          * v1(i)
            zfsf(i)      = zfsf0(i)          * v1(i)
            zfsv(i)      = zfsv0(i)          * v1(i)
            zfsi(i)      = zfsi0(i)          * v1(i)
            zparr(i)     = zparr0(i)         * v1(i)
        endif
!
 1300   continue
!
       do 1400 k=1,nk
       do 1400 i=1,n
         if(thold(i))then
            zt2(i,k) = zt20(i,k) * v1(i)
         endif
1400   continue

       do 1450 k=1,nkp
       do 1450 i=1,n
         if(thold(i))then
            zfluxds(i,k) = zfluxds0(i,k) * v1(i)
            zfluxus(i,k) = zfluxus0(i,k) * v1(i)
         endif
1450   continue

!
!...    end of radiation loop
      endif
!
      do i=1,n
        zcang(i) = rmu0(i)
!
!       iv represente le flux entrant au sommet de l'atmosphere
!       if below ensures iv is zero when sun is set
!
        if(thold(i))then
          ziv(i) = CONSOL2 * r0r * rmu0(i)
        else
          ziv(i) = 0.0
        endif
!
        if(ziv(i).gt. 1.0) then
          zap(i) = zev(i) / ziv(i)
        else
          zap(i) = 0.
        endif
!
        p1(i) = ziv(i) - zev(i) - zei(i)
      enddo
!
!...  extraction pour diagnostics
!
   IF_SERIES: if (series_isstep(' ')) then
      call serxst2(f(ti)    , 'ti',  trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(f(t2)    , 't2',  trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(v(ctp )  , 'bp',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(ctt)   , 'be',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(topthw), 'w3',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(topthi), 'w4',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(iv)    , 'iv',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(p1       , 'nr',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(tcc)   , 'tcc', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(ecc)   , 'ecc', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(eccl)  , 'eccl',trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(eccm)  , 'eccm',trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(ecch)  , 'ecch',trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(ev)    , 'ev',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(ei)    , 'ei',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(ap)    , 'ap',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(fdss)  , 'fs',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(flusolis), 'fu',trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(fsd)   , 'fsd', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(fsf)   , 'fsf', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(fsv)   , 'fsv', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(fsi)   , 'fsi', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(parr)  , 'parr',trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(clb)   , 'clb', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(clt)   , 'clt', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(cstt)  , 'cst', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(csb)   , 'csb', trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(f(cosas) , 'co',  trnch, n,  1, 0.0, 1.0, -1)
      call serxst2(v(cang)  , 'cx',  trnch, n,  1, 0.0, 1.0, -1)

!PV for uv band fluxes
! v(fatb), v(fadb), v(fafb), v(fctb), v(fcdb), v(fcfb)
      do iuv=1, 6
         write(niuv, '(i1)') iuv
         call serxst2(v(fatb+(iuv-1)*n), 'fat'//niuv, trnch, n, 1, 0., 1., -1)
         call serxst2(v(fadb+(iuv-1)*n), 'fad'//niuv, trnch, n, 1, 0., 1., -1)
         call serxst2(v(fafb+(iuv-1)*n), 'faf'//niuv, trnch, n, 1, 0., 1., -1)
         call serxst2(v(fctb+(iuv-1)*n), 'fct'//niuv, trnch, n, 1, 0., 1., -1)
         call serxst2(v(fcdb+(iuv-1)*n), 'fcd'//niuv, trnch, n, 1, 0., 1., -1)
         call serxst2(v(fcfb+(iuv-1)*n), 'fcf'//niuv, trnch, n, 1, 0., 1., -1)
      enddo
      call serxst2(f(o3s), 'OZO', trnch, n, nk, 0.0, 1.0, -1)
   endif IF_SERIES
!
!        tendances de la radiation
         ztrad = zti + zt2
!
!
      return
      end

