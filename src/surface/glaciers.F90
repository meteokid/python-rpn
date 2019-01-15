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
subroutine glaciers1(BUS, BUSSIZ, PTSURF, PTSURFSIZ, TRNCH, KOUNT, N, M, NK)
   use sfclayer_mod, only: sl_prelim,sl_sfclayer,SL_OK
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object Calculate the surface temperature (and specific humidity) and
   !          the surface fluxes of heat, moisture, and momentum over
   !          continental glaciers and ice sheets.
   !@Arguments
   !               - Input/Output -
   ! BUS           bus of surface variables
   !               - Input -
   ! BUSSIZ        size of the surface bus
   ! PTSURF        surface pointers
   ! PTSURFSIZ     dimension of ptsurf
   ! KOUNT         number of timestep
   ! DELT          timestep
   ! N             running length
   ! M             horizontal dimension
   ! NK            vertical dimension
   ! ICEMELT       switch to control snow melting

   integer BUSSIZ, KOUNT, TRNCH, N, M, NK
   integer PTSURFSIZ
   integer PTSURF(PTSURFSIZ)
   real,target :: bus(bussiz)

   !@Author J. Mailhot (April 1999)
   !@Notes
   !          One-dimensional thermodynamic model to treat glaciers and ice sheets:
   !          - includes snow cover on the top of ice, heat conduction through snow and ice,
   !            thermal inertia of snow and ice layers, and penetrating solar radiation
   !          - includes a parameterization of albedo, conductivity and heat capacity
   !
   ! Note:     - this subroutine expects snow depth in metre 
   !*@/
   
   include "thermoconsts.inc"

   include "dintern.inc"
   include "fintern.inc"

   integer SURFLEN
#define x(fptr,fj,fk) ptsurf(vd%fptr%i)+(fk-1)*surflen+fj-1

   integer, parameter :: INDX_SFC = INDX_GLACIER

   real, save :: CON1,CON2,CON3,CON4,CON5,CON6
   real, save :: CON7,CON8,CON9,CON10,CON11,CON12
   real, save :: FI0,CONDFI,TMELICE,TMELSNO
   real, save :: ALBDI,ALBMI,ALBDS,ALBMS,COEFEXT,EMISICE,EMISNOW
   real, save :: ROICE,ROSNOW(2)
   real, save :: HCAPI,VHFICE,VHFSNO
   real, save :: HCAPS,KSDS,KSMS
   real, save :: Z0GLA

   real,dimension(n) :: a,    b,     c,     c2,   ct,   dqsat,emist,rhoa
   real,dimension(n) :: scr1, scr2,  scr3,  scr4, scr5, scr6, scr7, scr8
   real,dimension(n) :: scr9, scr10, scr11, t2, vmod, vdir, zsnodp_m

   real,pointer,dimension(:) :: al, albsfc, cmu, ctu, fc_glac
   real,pointer,dimension(:) :: fsol, fv_glac
   real,pointer,dimension(:) :: hice, hst_glac, hu, ilmo_glac
   real,pointer,dimension(:) :: ps, qsice, th, snorate, tdeep, ts, tt, uu, vv
   real,pointer,dimension(:) :: z0h, z0m
   real,pointer,dimension(:) :: zalfaq, zalfat, zdlat, zfcor, zfdsi
   real,pointer,dimension(:) :: zftemp, zfvap, zqdiag, zsnodp, ztdiag
   real,pointer,dimension(:) :: ztsurf, ztsrad, zudiag, zvdiag
   real,pointer,dimension(:) :: zfrv, zzusl, zztsl

   logical, parameter :: GLACIER_TDIAGLIM=.false.

   integer I, J
   real DAY, DEEPFRAC

   data  CON1     , CON2   , CON3  , CON4    / &
         2.845E-6 , 2.7E-4 , 233.0 , 0.2   /
   data  CON5     , CON6   , CON7  , CON8  / &
         92.88    , 7.364  , 3.2   , 14.24 /
   data  CON9     , CON10 , CON11 , CON12 / &
         19.39    , 0.1   , 0.44  , 0.075 /
   data  FI0   ,  CONDFI   / &
         0.17  ,  2.034    /
   data  TMELICE , TMELSNO  /  273.05 , 273.15  /
   data  ALBDI   ,  ALBMI ,  ALBDS  ,  ALBMS / &
         0.57    ,  0.50  ,  0.83   ,  0.77  /
   data  COEFEXT / &
         1.5     /
   data  EMISICE , EMISNOW / 0.99 , 0.99 /
   data  ROICE  / &
         913.0  /
   data  ROSNOW  / 330.0 , 450.0 /
   data  HCAPI    , VHFICE   , VHFSNO   / &
         2.062E+3 , 2.679E+8 , 1.097E+8 /
   data  HCAPS   , KSDS  , KSMS   / &
         2.04E+3 , 0.325 , 0.665 /
   data  Z0GLA  / 3.0E-4 /
   
   !------------------------------------------------------------------------
   SURFLEN = M
   DAY = 86400.
   !# Fraction of diurnal temperature cycle
   !# (determines depth of deep in-glacier temperature)
   DEEPFRAC = exp(-1.0)

   albsfc   (1:n) => bus( x(alvis,1,indx_sfc) : )
   cmu      (1:n) => bus( x(bm,1,1)           : )
   ctu      (1:n) => bus( x(bt,1,indx_sfc)    : )
   fc_glac  (1:n) => bus( x(fc,1,indx_sfc)    : )
   if (radslope)  then
      fsol  (1:n) => bus( x(fluslop,1,1)      : )
   else
      fsol  (1:n) => bus( x(flusolis,1,1)     : )
   endif
   fv_glac  (1:n) => bus( x(fv,1,indx_sfc)    : )
   hst_glac (1:n) => bus( x(hst,1,indx_sfc)   : )
   hu       (1:n) => bus( x(humoins,1,nk)     : )
   ilmo_glac(1:n) => bus( x(ilmo,1,indx_sfc)  : )
   ps       (1:n) => bus( x(pmoins,1,1)       : )
   qsice    (1:n) => bus( x(qsurf,1,indx_sfc) : )
   snorate  (1:n) => bus( x(snowrate,1,1)     : )
   tdeep    (1:n) => bus( x(tglacier,1,2)     : )
   th       (1:n) => bus( x(thetaa,1,1)       : )
   ts       (1:n) => bus( x(tglacier,1,1)     : )
   tt       (1:n) => bus( x(tmoins,1,nk)      : )
   uu       (1:n) => bus( x(umoins,1,nk)      : )
   vv       (1:n) => bus( x(vmoins,1,nk)      : )
   z0h      (1:n) => bus( x(z0t,1,indx_sfc)   : )
   z0m      (1:n) => bus( x(z0,1,indx_sfc)    : )
   zalfaq   (1:n) => bus( x(alfaq,1,1)        : )
   zalfat   (1:n) => bus( x(alfat,1,1)        : )
   zdlat    (1:n) => bus( x(dlat,1,1)         : )
   zfcor    (1:n) => bus( x(fcor,1,1)         : )
   zfdsi    (1:n) => bus( x(fdsi,1,1)         : )
   zfrv     (1:n) => bus( x(frv,1,indx_sfc)   : )
   zftemp   (1:n) => bus( x(ftemp,1,indx_sfc) : )
   zfvap    (1:n) => bus( x(fvap,1,indx_sfc)  : )
   zqdiag   (1:n) => bus( x(qdiag,1,1)        : )
   zsnodp   (1:n) => bus( x(snodp,1,indx_sfc) : )
   ztsrad   (1:n) => bus( x(tsrad,1,1)        : )
   ztsurf   (1:n) => bus( x(tsurf,1,1)        : )
   zudiag   (1:n) => bus( x(udiag,1,1)        : )
   zvdiag   (1:n) => bus( x(vdiag,1,1)        : )
   ztdiag   (1:n) => bus( x(tdiag,1,1)        : )
   zzusl    (1:n) => bus( x(zusl,1,1)         : )
   zztsl    (1:n) => bus( x(ztsl,1,1)         : )


!*     1.     Preliminaries
!      --------------------

      i = sl_prelim(tt,hu,uu,vv,ps,zzusl,spd_air=vmod,dir_air=vdir,rho_air=rhoa, &
           min_wind_speed=sqrt(vamin))
      if (i /= SL_OK) then
         print*, 'Aborting in glaciers() because of error returned by sl_prelim()'
         stop
      endif  

      do I=1,N

!                           - in-glacier temperature

        T2(I) = TDEEP(I)

!                           Thermal roughness length for
!                           the ice surface


        Z0H(I) = max(   0.2 * Z0M(I)  ,  Z0GLA   )
        Z0H(I) = min(    Z0H(I)  ,  0.2   )


!                           Saturated specific humidity above
!                           ice surface

        QSICE(I) = FOQST( TS(I), PS(I) )
        DQSAT(I) = FODQS ( QSICE(I), TS(I) )

!       Damp snow depth with h0*tanh(ff/h0)

      if (LIMSNODP) then
        ZSNODP_M(I) = SNOH0*tanh(ZSNODP(I)/SNOH0)
      else
       ZSNODP_M(I)=ZSNODP(I)
      endif

      end do


!       2.     Calculate the drag and heat coefficients
!       -----------------------------------------------

      i = sl_sfclayer(th,hu,vmod,vdir,zzusl,zztsl,ts,qsice,z0m,z0h,zdlat,zfcor, &
           coefm=cmu,coeft=ctu,flux_t=zftemp,flux_q=zfvap,ilmo=ilmo_glac,ue=zfrv, &
           h=hst_glac)
      if (i /= SL_OK) then
         print*, 'Aborting in glaciers() because of error returned by sl_sfclayer()'
         stop
      endif 

!       3.     Parameterizations (albedo, emissivity, conductivity,...)
!       ---------------------------------------------------------------




!                           surface albedo (function of surface
!                           type and temperature)


      do I=1,N

        if( TS(I) .lt. TMELSNO ) then
          SCR2(I) = ALBDS
        else
          SCR2(I) = ALBMS
        endif

        if( TS(I) .lt. TMELICE ) then
          SCR3(I) = ALBDI
        else
          SCR3(I) = ALBMI
        endif

        if( ZSNODP(I) .gt. CON10 ) then
          ALBSFC(I) = SCR2(I)
        else
          ALBSFC(I) = min( SCR2(I) , SCR3(I)+ZSNODP(I)* &
                          (SCR2(I)-SCR3(I))/CON10 )
        endif

      end do

!                               conductivity and capacity (function of surface temperature)
      do I=1,N
!                                            snow layer
        if( TS(I) .lt. TMELSNO ) then
          SCR2(I) = ROSNOW(1)*HCAPS
          SCR4(I) = KSDS
        else
          SCR2(I) = ROSNOW(2)*HCAPS
          SCR4(I) = KSMS
        endif
!                                            ice layer
        SCR3(I) = ROICE*HCAPI
        SCR5(I) = CONDFI
!                                            depths D1 and D2
        SCR7(I) = -sqrt((SCR4(I)/SCR2(I))*DAY/PI)*ALOG(DEEPFRAC)
        SCR8(I) = -sqrt((SCR5(I)/SCR3(I))*DAY/PI)*ALOG(DEEPFRAC)
!                                            equivalent ice depth and
!                                            limit if Tp is in snow layer
        if( ZSNODP_M(I) .lt. SCR7(I) ) then
          SCR6(I) = SCR8(I)*( 1. -ZSNODP_M(I)/SCR7(I) )
          SCR10(I) = ZSNODP_M(I)
        else
          SCR6(I) = 0.0
          SCR10(I) = SCR7(I)
        endif
!                                            compute K/H
        SCR11(I) = SCR4(I)/( SCR10(I) + SCR6(I)*SCR4(I)/SCR5(I) )

      end do

      do I=1,N
!                               heat capacity of combined snow/ice layers H*C
        SCR5(I) = 0.5*( SCR10(I)*SCR2(I)*( 1.+SCR11(I)*SCR6(I)/SCR5(I) ) &
                + SCR3(I)*SCR11(I)*SCR6(I)**2/SCR5(I) )
        CT(I) = 1./SCR5(I)
        C2(I) = CT(I)*SCR11(I)

      end do

!                               emissivity and penetration of solar radiation
!                               (function of surface type snow/ice)
      do I=1,N

        if( ZSNODP(I) .gt. CON10 ) then
          EMIST(I) = EMISNOW
          SCR1(I) = 1.0
        else
          EMIST(I) = EMISICE
          SCR1(I) = 1.0 - FI0
          SCR1(I) = min( 1.0 , SCR1(I)+ZSNODP(I)* &
                        (1.0-SCR1(I))/CON10 )
        endif

        SCR2(I) = 1.-exp(-COEFEXT*SCR6(I))

      end do

!                                              terms A, B, and C for the
!                                              calculation of TS at time 'T+DT'
      do I=1,N

        A(I) = CT(I) * ( 4. * EMIST(I) * STEFAN * TS(I)**3 &
           +  RHOA(I) * CTU(I) * (DQSAT(I) * (CHLC+CHLF) + CPD) ) &
           + C2(I)

        B(I) = CT(I) * ( 3. * EMIST(I) * STEFAN  * TS(I)**3 &
           + RHOA(I) * CTU(I) * DQSAT(I) * (CHLC+CHLF) )

        C(I) = C2(I) * T2(I) + CT(I) * &
           (  RHOA(I)*CTU(I) * (CPD*TH(I) - (CHLC+CHLF)*(QSICE(I)-HU(I))) &
           +  FSOL(I)*(1.-ALBSFC(I)) * (SCR1(I)+(1.-SCR1(I))*SCR2(I)) &
           + EMIST(I)*ZFDSI(I) )

      end do


!       4.     Calculate the surface temperature from
!              force-restore equation at the snow or ice surface
!       -----------------------------------------------------------------

!                                       Energy balance for surface temperature
!  d TS / dt = -2*HA/(C*H)-2*K(TS-TP)/(C*H*H)
!                                       after linearization
!  d TS / dt = - A * TS+ + B * TS- + C
!                                       and deep temperature
!  d TP / dt = - ( TP+ -  TS+ ) / DAY
!                                       can be solved analytically
        do I=1,N
           TS(I) = TS(I) + ( TS(I)*(B(I)-A(I)) + C(I) ) &
                 * ( 1.-exp(-DELT*A(I)) ) / A(I)
           TDEEP(I) = (TDEEP(I)+TS(I)*DELT/DAY)/(1.0+DELT/DAY)
        end do


      do I=1,N
!                                              prevent surface and deep temperatures
!                                              from exceeding melting temperature
        if( ZSNODP(I) .gt. 0.0 ) then
          SCR4(I) = TMELSNO
        else
          SCR4(I) = TMELICE
        endif

        TS(I) = min ( TS(I) , SCR4(I) )
        TDEEP(I) = min ( TDEEP(I) , SCR4(I) )

!                               saturated specific humidity at
!                               surface

        QSICE(I) = FOQST( TS(I), PS(I) )

      end do

      ! Estimate diagnostic-level quantities
      i = sl_sfclayer(th,hu,vmod,vdir,zzusl,zztsl,ts,qsice,z0m,z0h,zdlat,zfcor, &
           hghtm_diag=zu,hghtt_diag=zt,t_diag=ztdiag,q_diag=zqdiag,u_diag=zudiag, &
           v_diag=zvdiag,tdiaglim=GLACIER_TDIAGLIM)
      if (i /= SL_OK) then
         print*, 'Aborting in glaciers() because of error returned by sl_sfclayer()'
         stop
      endif

!       5.     Melting and growth
!       -------------------------
!                                              melting of snow at the surface
!                                              accumulation of snow fall
      if(ICEMELT) then

        do I=1,N

!                               recompute surface albedo with new TS

          if( TS(I) .lt. TMELSNO ) then
            SCR2(I) = ALBDS
          else
            SCR2(I) = ALBMS
          endif

          if( TS(I) .lt. TMELICE ) then
            SCR3(I) = ALBDI
          else
            SCR3(I) = ALBMI
          endif

          if( ZSNODP(I) .gt. CON10 ) then
            ALBSFC(I) = SCR2(I)
          else
            ALBSFC(I) = min( SCR2(I) , SCR3(I)+ZSNODP(I)* &
                           (SCR2(I)-SCR3(I))/CON10 )
          endif

!                               compute K/H * (TB - TS)
          SCR11(I) = SCR11(I)*( TDEEP(I) - TS(I) )

!                               compute HA = H + LE + FWnet + FLnet
          SCR5(I) = RHOA(I)*CTU(I)* &
                    ( CPD*(TS(I)-TH(I))+(CHLC+CHLF)*(QSICE(I)-HU(I)) ) &
                   - FSOL(I)*(1.-ALBSFC(I))*SCR1(I) &
                   + EMIST(I)*( STEFAN*TS(I)**4 - ZFDSI(I) )

          SCR3(I) = SCR5(I) - SCR11(I)

!                               criteria for melting at the surface
          if( TS(I).ge.SCR4(I) .and. &
              SCR5(I).le.0.0 .and. SCR3(I).lt.0.0 ) then

!                               melt available snow...
            ZSNODP(I) = max( 0.0, ZSNODP(I) + DELT*SCR3(I)/VHFSNO )

          endif
!                               add snow fall (units changed from
!                               water equivalent to snow equivalent)
          if( TS(I).lt.TMELICE ) then
            ZSNODP(I) = ZSNODP(I) + DELT*SNORATE(I)*(1000./ROSNOW(1))
          endif

        end do

      endif


!       6.     The fluxes
!       -----------------

!VDIR NODEP
      do I=1,N

        ZTSURF    (I) = TS (I)
        ZTSRAD    (I) = TS (I)

        ZALFAT    (I) = - CTU(I) * ( TS   (I)-TH(I) )
        ZALFAQ    (I) = - CTU(I) * ( QSICE(I)-HU(I) )
        if (.not.IMPFLX) CTU (I) = 0.
        RHOA      (I) = PS(I)/(RGASD * ZTDIAG(I)*(1.+DELTA*ZQDIAG(I)))
        FC_GLAC(I)    = -CPD *RHOA(I)*ZALFAT(I)
        FV_GLAC(I)    = -(CHLC+CHLF)*RHOA(I)*ZALFAQ(I)

        if (IMPFLX) then
          ZALFAT    (I) = - CTU(I) *  TS(I)
          ZALFAQ    (I) = - CTU(I) *  QSICE(I)
        endif
!***

      end do

!     FILL THE ARRAYS TO BE AGGREGATED LATER IN S/R AGREGE
      call FILLAGG ( BUS, BUSSIZ, PTSURF, PTSURFSIZ, INDX_GLACIER, &
                     SURFLEN)

   return
end subroutine glaciers1
