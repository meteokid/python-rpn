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
subroutine water1(bus, bussiz, ptsurf, ptsurfsiz, lcl_indx, trnch, kount, &
     n, m, nk )
   use sfclayer_mod, only: sl_prelim,sl_sfclayer,SL_OK
   use cpl_itf     , only: cpl_update
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object Calculate:   - surface roughness length (Z0) over open water
   !                       (not covered by ice) using Charnock's relation with a
   !                       BETA1 parameter (optional sea state dependency);
   !                     - surface fluxes of momentum, heat and moisture.
   !@Arguments
   !             - Input/Output -
   ! BUS         Bus for the WATER surface scheme
   !             - Input -
   ! BUSSIZ      dimension of bus
   ! PTSURF      surface pointers
   ! PTSURFSIZ   dimension of ptsurf
   ! TRNCH       row number
   ! KOUNT       timestep number
   ! N           horizontal dimension (row length)
   ! M           horizontal dimensions of fields
   !             (not used for the moment)
   ! NK          vertical dimension

   integer bussiz, kount, trnch, N, M, NK
   real, target :: bus(bussiz)
   integer ptsurfsiz
   integer ptsurf(ptsurfsiz), lcl_indx(2,n)

   !@Author J. Mailhot, S. Belair and B. Bilodeau (Dec 1998)
   !@Notes
   !          Z0 = BETA1*USTAR**2/GRAV (in metres) with minimum value
   !          Z0MIN and a maximum value Z0MAX
   !*@/
   include "thermoconsts.inc"
   include "sfcinput.cdk"

   include "dintern.inc"
   include "fintern.inc"

   integer surflen
#define x(fptr,fj,fk) ptsurf(vd%fptr%i)+(fk-1)*surflen+fj-1

   real, parameter :: z0max = 5.e-3

   real, parameter :: am = 0.11        !Roughness adjustment coefficient (Beljaars qjrms 1994)
   real, parameter :: beta1 = 0.018    !Charnock's coefficient
   real, parameter :: k_visc = 1.5e-5  !Kinematic viscosity estimate at ~20oc
   real, parameter :: zfrvmin = 0.01   !Minimum friction velocity for inverse relations
   real, parameter :: rho_w = 1025.    !Density of seawater (Rudnick and Martin; Dyn. of Atm. and Oceans, 2002)
   real, parameter :: diff_w = 1.45e-7 !Thermal moelcular diffusivity of seawater (National Physical Laboratory tables)
   real, parameter :: cond_w = 0.58    !Thermal conducitivity of seawater (National Physical Laboratory tables)
   real, parameter :: cv_w = 3950.     !Volumetric heat capacity of seawater (National Physical Laboratory tables)
   real, parameter :: nu = 0.3         !Warm layer profile shape parameter (1 = linear)
   real, parameter :: base_depth = 3.  !Depth (m) of the "base" temperature without diurnal variability
   real, parameter :: skin_depth_min = 0.0002 !Minimum depth of the cold skin (Tu and Tsuang; GRL 2005)
   real, parameter :: ref_depth = 10.  !Reference depth (m) for the Artale et al. (JGR 2002) forumulation of Saunders' constant
   real, parameter :: emis = 0.984     !Sea surface emissivity (Konda et al.; J. of Oceanography 1994)
   character(len = *), parameter :: saunders = 'artale' !Calculation strategy for Saunders' constant ('fairall','artale')
   real, parameter :: dsst_epsilon = 1.e-5 !Convergence threshold for warm layer increment
   real, parameter :: delh_tresh = 1.e-10  !Treshold to compute ctu in ocean coupled mode
   real, parameter :: zt_rho = 1.5     !Height used to compute air density in ocean coupled mode
   integer, parameter :: max_iter = 10 !Maximum number of iterations for convergence of warm layer increment
   integer, parameter ::INDX_SFC = INDX_WATER
   logical, parameter :: WATER_TDIAGLIM = .false.

   real,pointer,dimension(:) ::  alvis_wat, cmu, ctu, fc_wat, mlac
   real,pointer,dimension(:) ::  fv_wat
   real,pointer,dimension(:) ::  hst_wat, hu, ilmo_wat
   real,pointer,dimension(:) ::  ps, qs, th, ts, tt, uu, vv
   real,pointer,dimension(:) ::  z0h, z0m, zdsst
   real,pointer,dimension(:) ::  zalfaq, zalfat, zdlat, zfcor
   real,pointer,dimension(:) ::  zftemp, zfvap, zqdiag, ztdiag
   real,pointer,dimension(:) ::  ztsurf, ztsrad, zudiag, zvdiag
   real,pointer,dimension(:) ::  zfrv, zzusl, zztsl
   real,pointer,dimension(:) ::  zflusolis,zfdsi,zskin_depth,zskin_inc

   real, dimension(n) :: z0m_adjust,tva,rhoa,vmod,vdir,vmodd
   real, dimension(n) :: alpha_w,rho_a,frv_w,frv_a,visc_w,skin_solar,q_bal,skin_q,sst
   real, dimension(n) :: base_bflux,mo_len,warm_increment,stab_func,lambda,ud,vdi,gamma
   real, dimension(n) :: this_inc,prev_inc,denom
   real, dimension(n) :: my_ta,my_qa

   integer I, J, K
   real qsat_o_salty, delh, delq
   logical            :: cplupd

   !** ------------------------------------------------------------------

   SURFLEN = M

   alvis_wat(1:n) => bus( x(alvis,1,indx_sfc) : )
   cmu      (1:n) => bus( x(bm,1,1)           : )
   ctu      (1:n) => bus( x(bt,1,indx_sfc)    : )
   mlac     (1:n) => bus( x(ml,1,1)           : )
   fc_wat   (1:n) => bus( x(fc,1,indx_sfc)    : )
   fv_wat   (1:n) => bus( x(fv,1,indx_sfc)    : )
   hst_wat  (1:n) => bus( x(hst,1,indx_sfc)   : )
   hu       (1:n) => bus( x(humoins,1,nk)     : )
   ilmo_wat (1:n) => bus( x(ilmo,1,indx_sfc)  : )
   ps       (1:n) => bus( x(pmoins,1,1)       : )
   qs       (1:n) => bus( x(qsurf,1,indx_sfc) : )
   th       (1:n) => bus( x(thetaa,1,1)       : )
   ts       (1:n) => bus( x(twater,1,1)       : )
   tt       (1:n) => bus( x(tmoins,1,nk)      : )
   uu       (1:n) => bus( x(umoins,1,nk)      : )
   vv       (1:n) => bus( x(vmoins,1,nk)      : )
   z0h      (1:n) => bus( x(z0t,1,indx_sfc)   : )
   z0m      (1:n) => bus( x(z0,1,indx_sfc)    : )
   zalfaq   (1:n) => bus( x(alfaq,1,1)        : )
   zalfat   (1:n) => bus( x(alfat,1,1)        : )
   zdlat    (1:n) => bus( x(dlat,1,1)         : )
   zdsst    (1:n) => bus( x(dsst,1,1)         : )
   zfcor    (1:n) => bus( x(fcor,1,1)         : )
   zfdsi    (1:n) => bus( x(fdsi,1,1)         : )
   zflusolis(1:n) => bus( x(flusolis,1,1)     : )
   zftemp   (1:n) => bus( x(ftemp,1,indx_sfc) : )
   zfvap    (1:n) => bus( x(fvap,1,indx_sfc)  : )
   zskin_depth(1:n) => bus( x(skin_depth,1,1) : )
   zskin_inc(1:n) => bus( x(skin_inc,1,1)     : )
   ztsurf   (1:n) => bus( x(tsurf,1,1)        : )
   ztsrad   (1:n) => bus( x(tsrad,1,1)        : )
   zudiag   (1:n) => bus( x(udiag,1,1)        : )
   zvdiag   (1:n) => bus( x(vdiag,1,1)        : )
   ztdiag   (1:n) => bus( x(tdiag,1,1)        : )
   zqdiag   (1:n) => bus( x(qdiag,1,1)        : )
   zfrv     (1:n) => bus( x(frv,1,indx_sfc)   : )
   zzusl    (1:n) => bus( x(zusl,1,1)         : )
   zztsl    (1:n) => bus( x(ztsl,1,1)         : )

   !------------------------------------------------------------------------


   ! 0.     Precompute local SST from "base" SST (ts) and diurnal components
   ! -----------------------------------------------------------------------
   sst = ts + zdsst + zskin_inc !add dirunal warm-layer and cool skin increments


   ! 1.     Saturated specific humidity at the water surface
   ! -------------------------------------------------------

   ! Uses FOQSA instead of FOQST to take into account saturation
   ! with respect to sea water (liquid between 0 and -1.8 C)
   qsat_o_salty = 1.0
   if (salty_qsat) qsat_o_salty = 0.98
   do I=1,N
      !        QS(I) = FOQSA(TS(I),PS(I)) * (1.-MLAC(I))*qsat_o_salty
      QS(I) = FOQSA(SST(I),PS(I))
      if(MLAC(I) .le. 0.001) QS(I) = qsat_o_salty*QS(I)
   end do


   ! 2.     Calculate roughness lengths based on generalized Charnock's relation
   ! ---------------------------------------------------------------------------

   if (kount.gt.0 .or. any('frv'==phyinread_list_s(1:phyinread_n))) then

      ! Precompute the Beljaars (QJRMS 1995, pp255-270) adjustment to the Charnock 
      ! relationship to increase the roughness length at low wind speeds (low ZFRV).
      z0m_beljaars: if (z0mtype == 'BELJAARS') then
         Z0M_ADJUST = AM*K_VISC/max(ZFRV,ZFRVMIN)
      else
         Z0M_ADJUST = 0.
      endif z0m_beljaars

      ! Use Charnock's relation to estimate roughness length
      do I=1,N
         Z0M(I) = max( min( BETA1*ZFRV(I)**2/GRAV + Z0M_ADJUST(I),Z0MAX ) , Z0MIN )
      enddo

   endif

   if (Z0TRDPS300) then
      do I=1,N
         ZFRV(I)=max(ZFRVMIN,  ZFRV(I))
         Z0H(I) = min(2.e-05/ZFRV(I), 1.e-04)
      end do
   else
      do I=1,N
         Z0H(I) = Z0M(I)
      end do
   endif


   ! Note:  For |lat| >= Z0TLAT(2)  Charnock's (or Deacu) relation is used
   !        For |lat| <= Z0TLAT(1)  Z0HCON is used.
   !        For Z0TLAT(1) < |lat| < Z0TLAT(2)
   !        we do a linear interpolation between Charnock(or Deacu) and Z0HCON.

   do I=1,N

      if (abs(ZDLAT(I)) .ge. Z0TLAT(2)) then
         Z0H(I) = Z0H(I)
      else if (abs(ZDLAT(I)) .le. Z0TLAT(1)) then
         Z0H(I) = Z0HCON
      else
         Z0H(I)=( ((abs(ZDLAT(I))-Z0TLAT(1))/(Z0TLAT(2)-Z0TLAT(1))) &
              *(Z0H(I)-Z0HCON) ) + Z0HCON
      endif

   end do

   ! 3.     Calculate the surface transfer coefficient and fluxes
   ! ------------------------------------------------------------

   i = sl_prelim(tt,hu,uu,vv,ps,zzusl,rho_air=rho_a,spd_air=vmod,dir_air=vdir,min_wind_speed=VAMIN)
   if (i /= SL_OK) then
      print*, 'Aborting in water() because of error returned by sl_prelim()'
      stop
   endif
   i = sl_sfclayer(th,hu,vmod,vdir,zzusl,zztsl,sst,qs,z0m,z0h,zdlat,zfcor, &
        hghtm_diag=zu,hghtt_diag=zt,coefm=cmu,coeft=ctu,flux_t=zftemp, &
        flux_q=zfvap,ilmo=ilmo_wat,ue=zfrv,h=hst_wat,t_diag=ztdiag,    &
        q_diag=zqdiag,u_diag=zudiag,v_diag=zvdiag,tdiaglim=WATER_TDIAGLIM)
   if (i /= SL_OK) then
      print*, 'Aborting in water() because of error returned by sl_sfclayer()'
      stop
   endif

   ! Also compute the air density at a chosen height (for now, the chosen height is
   ! at the screen level, i.e. at zt = 1.5m)

   i = sl_sfclayer(th,hu,vmod,vdir,zzusl,zztsl,sst,qs,z0m,z0h,zdlat,zfcor, &
        hghtt_diag=1.5,t_diag=my_ta,q_diag=my_qa,tdiaglim=WATER_TDIAGLIM)
   if (i /= SL_OK) then
      print*, 'Aborting in water() because of error returned by sl_sfclayer()'
      stop
   endif



   ! 4.     Finalize the fluxes
   ! --------------------------

!VDIR NODEP
   do I=1,N

      ZTSURF   (I) = SST (I)
      ZTSRAD   (I) = SST (I)

      ZALFAT   (I) = - CTU(I) * ( SST(I)-TH(I) )
      ZALFAQ   (I) = - CTU(I) * ( QS(I)-HU(I) )
      if (.not.IMPFLX) CTU (I) = 0.
      RHOA(i) = PS(I)/(RGASD * my_ta(I)*(1.+DELTA*my_qa(I)))
      !         RHOA(i) = PS(I)/(RGASD * ZTDIAG(I)*(1.+DELTA*ZQDIAG(I)))
      FC_WAT(I)    = -CPD *RHOA(i)*ZALFAT(I)
      FV_WAT(I)    = -CHLC*RHOA(i)*ZALFAQ(I)

      if (IMPFLX) then
         ZALFAT   (I) = - CTU(I) *  SST(I)
         ZALFAQ   (I) = - CTU(I) *  QS(I)
      endif

   end do

   if (cplocn) then
      ! Update with fluxes and diagnostic variables from ocean model
      cplupd=.false.
      call cpl_update (vmod(1:n), 'UVO', lcl_indx, n, u=UU, v=VV, cplu=cplupd)
      if (cplupd) then
         call cpl_update (FC_WAT(1:n), 'SHO' , lcl_indx, n)
         call cpl_update (FV_WAT(1:n), 'LHO' , lcl_indx, n)
         call cpl_update (ZUDIAG(1:n), 'ZUO' , lcl_indx, n)
         call cpl_update (ZVDIAG(1:n), 'ZVO' , lcl_indx, n)
         call cpl_update (ZTDIAG(1:n), 'ZTO' , lcl_indx, n)
         call cpl_update (ZQDIAG(1:n), 'ZQO' , lcl_indx, n)
         call cpl_update (ZTSURF(1:n),   'TMW' , lcl_indx, n)
         call cpl_update (ZTSRAD(1:n),   'T4O' , lcl_indx, n)
         call cpl_update (QS(1:n),       'QSO' , lcl_indx, n)
         call cpl_update (ILMO_WAT(1:n), 'ILO' , lcl_indx, n)
         call cpl_update (Z0M   (1:n),   'ZMO' , lcl_indx, n)
         call cpl_update (Z0H   (1:n),   'ZHO' , lcl_indx, n)
         ! Derives other variables from cpl_update output
         ! assuming FC_WAT and FV_WAT were computed with
         ! RHOA at the same level
         if (zt_rho == zt) then
            my_ta(1:n)=ztdiag(1:n)
            my_qa(1:n)=zqdiag(1:n)
         else
            print*, 'Aborting in water() because of inconsistent density level'
            stop
         endif
         do I=1,N
            RHOA(I) = PS(I)/(RGASD * my_ta(I)*(1.+DELTA*my_qa(I)))
            ZALFAT(I)= FC_WAT(I)/(-CPD *RHOA(I))
            ZALFAQ(I)= FV_WAT(I)/(-CHLC*RHOA(I))
            ZFTEMP(I) = -ZALFAT(I)
            ZFVAP(I)  = -ZALFAQ(I)
            SST(I)    = ZTSURF(I)
            if (IMPFLX) then
               ! CTU consistent with ocean model fluxes (as possible)
               ! Uncertainties in CTU from interpolation/agregation
               ! will be compensated by ZALFAT and ZALFAQ
               delh=SST(I)-TH(I) ; delq=QS(I)-HU(I)
               CTU(I) = 0.5*( -ZALFAT(I)/sign(max(abs(delh),delh_tresh),delh) &
                    -ZALFAQ(I)/sign(max(abs(delq),delh_tresh),delq) )
               CTU(I) = max(0.,CTU(I)) ! CTU<0 should never occur except under vicinity
               ! of null fluxes agregation/interpolation anyway
               ! e.g. sign(ZALFAT) /= sign(delh) or sign(ZALFAQ) /= sign(delq)
               ! ZALFAT and ZALFAQ consistent with FC_WAT and FV_WAT
               ZALFAT(I) = ZALFAT(I) - CTU(I)*TH(I)
               ZALFAQ(I) = ZALFAQ(I) - CTU(I)*HU(I)
            else
               CTU(I) = 0. !Redundant but less dangerous
            endif
         end do
         ! Diagnostic ustar based on ocean model stress (tau=rho*ustar**2)
         call cpl_update (ZFRV  (1:n), 'FRO' , lcl_indx, n, rho=RHOA, vmod=VMOD, cmu=CMU)
         ! Updated consistent CMU needed by implicit scheme using the following relation
         ! ==> CM=ustar/vmod (ustar**2=tau/rho=CM*CM*vmod**2)
      endif
   endif


   ! 5.     Prognostic evolution of SST (based on Zeng and Beljaars; GRL 2005)
   ! -------------------------------------------------------------------------

   DIURNAL_SST: if ( diusst == 'FAIRALL') then

      if (cplocn) STOP 'cplocn=true, diusst NOT TAKEN CARE OF YET'

      ! Preliminary calculations
      i = sl_prelim(tt,hu,uu,vv,ps,zzusl,rho_air=rho_a,spd_air=vmod,dir_air=vdir,min_wind_speed=VAMIN)
      i = sl_sfclayer(th,hu,vmod,vdir,zzusl,zztsl,sst,qs,z0m,z0h,zdlat,zfcor,hghtm_diag=10., &
           ue=frv_a,u_diag=ud,v_diag=vdi,tdiaglim=WATER_TDIAGLIM)
      vmodd = sqrt(ud**2 + vdi**2)
      alpha_w = 8.75e-6 * (sst-tcdk+9.) !expansion coefficient estimate from Hayes et al. (JGR, 1991)
      visc_w = 1.8e-6 - 4.e-8*(sst-tcdk) !kinematic viscosity (from IITC 2011 recommendations at 10oC)
      frv_w = frv_a * sqrt(rho_a/rho_w) !friction velocity of water
      q_bal = -(fv_wat + fc_wat + (emis*stefan*sst**4 - zfdsi)) !surface energy balance (without shortwave)

      ! Compute cool skin properties on request
      COOL_SKIN: if ( diusst_coolskin ) then

         ! Estimate the depth and cooling effect of the cold skin layer (typically 1mm-1cm and 0.3K)
         ! Solar radiation absorbed in the skin layer
         where (zskin_depth < skin_depth_min)
            skin_solar = 0.1 !estimate proposed by Fairall et al. (JGR 1996) section 2.4
         elsewhere
            skin_solar = 0.067 + 11.*zskin_depth - (6.6e-5/zskin_depth) * (1.-exp(-zskin_depth/8.e-4))
         endwhere
         ! Updated skin layer depth (cold skin only exists when net heat is being extracted from the surface)
         skin_q = q_bal + (1.-alvis_wat)*zflusolis*skin_solar
         ! Compute Saunders' constant of proportionality (lambda)
         if (saunders == 'fairall') then !follow Fariall et al. (GRL; 1996)
            where (skin_q <= 0.)
               lambda = 6.*(1. + ((-16.*grav*alpha_w*visc_w**3)/(frv_w**4*diff_w**2*rho_w*cv_w) * (skin_q))**0.75)**(-1./3.)
            elsewhere
               lambda = 6.
            endwhere
         elseif (saunders == 'artale') then !follow Artale et al. (JGR; 2002)
            where (vmodd > 10.)
               gamma = 6.
            elsewhere (vmodd > 7.5)
               gamma = 1.6*vmodd - 10.
            elsewhere
               gamma = 0.2*vmodd + 0.5
            endwhere
            lambda = 86400.*frv_w*cond_w/(gamma*rho_w*cv_w*ref_depth*visc_w)
         else
            print*, 'No Saunders proportionality constants defined for '//trim(saunders)//' in subroutine water()'
            stop
         endif
         zskin_depth = lambda * visc_w / (sqrt(rho_a/rho_w)*frv_a)
         zskin_depth = max(min(zskin_depth,0.01),skin_depth_min) !limit skin depth to physical values
         ! Compute cool skin temperature increment (difference SST - T(zskin_depth))
         zskin_inc = zskin_depth / cond_w * (skin_q)
         if (saunders == 'fairall') zskin_inc = min(zskin_inc,0.)

      endif COOL_SKIN

      ! Compute warm layer properties on request
      WARM_LAYER: if ( diusst_warmlayer ) then

         ! Estimate the evolving temperature departure in the warm layer implicitly (secant iterations)
         this_inc = zdsst ; prev_inc = zdsst+10.*dsst_epsilon ; i = 0
         do while (maxval(abs(this_inc-prev_inc)) > dsst_epsilon .and. i < max_iter)
            i = i+1
            where (abs(this_inc-prev_inc) > dsst_epsilon)
               denom = warm_func(this_inc)-warm_func(prev_inc)
               where (abs(denom) > 1.e-8)
                  ! Use local slope to estimate new increment value
                  warm_increment = this_inc - warm_func(this_inc)*((this_inc-prev_inc)/denom)
                  prev_inc = this_inc
                  this_inc = warm_increment
               elsewhere
                  ! Attempt a reset of the increment difference if the function is locally flat
                  warm_increment = this_inc
                  prev_inc = this_inc+10.*dsst_epsilon
               endwhere
            endwhere
         enddo
         zdsst = warm_increment

      endif WARM_LAYER

   endif DIURNAL_SST

   ! FILL THE ARRAYS TO BE AGGREGATED LATER IN S/R AGREGE
   call FILLAGG ( BUS, BUSSIZ, PTSURF, PTSURFSIZ, INDX_WATER, SURFLEN )

   return

contains

   ! Warm layer function for secant iterations
   function warm_func(inc) result(fval)
      ! Argument declarations
      real, dimension(:), intent(in) :: inc             !warm layer increment
      real, dimension(size(inc)) :: fval                !warm layer function result
      ! Local variables
      real, dimension(size(inc)) :: base_bflux,mo_len,stab_func
      ! Compute buoyancy flux at base depth
      where (inc > 0.)
         base_bflux = sqrt(nu*grav*alpha_w/(5.*base_depth)) * rho_w*cv_w*frv_w**2*sqrt(inc)
      elsewhere
         base_bflux = grav*alpha_w*(q_bal+0.64*zflusolis)
      endwhere
      base_bflux = sign(max(abs(base_bflux),1.e-4),base_bflux)
      mo_len = rho_w*cv_w*frv_w**3/(karman*base_bflux) !Monin-Obukhov length in the warm layer
      mo_len = sign(max(abs(mo_len),1.e-4),mo_len)
      ! Compute stability function at base depth
      where (mo_len > 0.)
         stab_func = 1. + 5.*(base_depth/mo_len)
      elsewhere
         stab_func = (1. - 16.*(base_depth/mo_len))**(-0.5)
      endwhere
      ! Compute warm layer function value
      fval = inc - delt*((q_bal + 0.64*zflusolis) / (base_depth*rho_w*cv_w*nu/(nu+1)) - &
           (nu+1)*karman*frv_w/(base_depth*stab_func) * inc) - zdsst
   end function warm_func

end subroutine water1
