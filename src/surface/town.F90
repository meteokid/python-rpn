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
subroutine town(bus, bussiz, ptsurf, ptsurfsiz, dt, trnch, kount, n, m, nk)
   use sfclayer_mod,   only : sl_prelim,sl_sfclayer
   use modd_town,      only : nni, xtown,                            &
        xq_town,                               &
        xu_canyon,                             &
        xrn_roof,xh_roof,xle_roof,xles_roof,   &
        xgflux_roof,xrunoff_roof,              &
        xrn_road,xh_road,xle_road,xles_road,   &
        xgflux_road,xrunoff_road,              &
        xrn_wall,xh_wall,xle_wall,xgflux_wall, &
        xrnsnow_roof,xhsnow_roof,xlesnow_roof, &
        xgsnow_roof,xmelt_roof,                &
        xrnsnow_road,xhsnow_road,xlesnow_road, &
        xgsnow_road,xmelt_road,                &
        xrn,xh,xle,xgflux,xevap,xrunoff,       &
        xch,xri,xustar
   use modd_teb,       only : xzs, xbld, xbld_height, xz0_town,      &
        xz0_roof,xz0_road,                     &
        xwall_o_hor, xcan_hw_ratio,            &
        xsvf_road,xsvf_wall,                   &
        xalb_roof, xalb_road, xalb_wall,       &
        xemis_roof, xemis_road, xemis_wall,    &
        xhc_roof, xtc_roof, xd_roof,           &
        xhc_road, xtc_road, xd_road,           &
        xhc_wall, xtc_wall, xd_wall,           &
        nroof_layer, nroad_layer, nwall_layer, &
        xh_traffic, xle_traffic,               &
        xh_industry, xle_industry,             &
        xt_roof, xt_road, xt_wall,             &
        xws_roof, xws_road,                    &
        xt_canyon, xq_canyon,                  &
        xti_road, xti_bld,                     &
        xwsnow_roof,  xtsnow_roof, xrsnow_roof,&
        xasnow_roof, xesnow_roof, xtssnow_roof,&
        xwsnow_road,  xtsnow_road, xrsnow_road,&
        xasnow_road, xesnow_road, xtssnow_road
   use modd_csts
   use modi_coupling_teb2
   use modi_sunpos
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object Choose the surface scheme for towns 
   !@Arguments
   !               - Input/Output -
   ! bus           bus of surface variables
   !               - input -
   ! bussiz        size of the surface bus
   ! ptsurf        surface pointers
   ! ptsurfsiz     dimension of ptsurf
   ! kount         number of timestep
   ! trnch         row number
   ! dt            timestep
   ! n             running length
   ! m             horizontal dimension
   ! nk            vertical dimension
   
   integer bussiz, kount, trnch
   real dt
   real, target :: bus(bussiz)
   integer ptsurfsiz
   integer ptsurf(ptsurfsiz)

   !@Author Aude Lemonsu (April 2004)
   !*@/

   include "tebcst.cdk"
   include "town_ptr.cdk"

   integer surflen
#define x(fptr,fj,fk) ptsurf(vd%fptr%i)+(fk-1)*surflen+fj-1

   real, external :: juliand

   integer, parameter :: indx_sfc = indx_urb
   real,    parameter :: xundef   = 999.

   integer :: n, m, nk, i, j, k
   real    :: julien

   integer,dimension (90) :: alloc_status
   integer                :: kyear       ! current year
   integer                :: kmonth      ! current month
   integer                :: kday        ! current day 
   real                   :: ptime       ! current time
   real                   :: ptstep      ! timestep   

   real, pointer, dimension(:)     :: ptsun       ! solar time
   real, pointer, dimension(:)     :: pzenith     ! solar zenithal angle
   real, pointer, dimension(:)     :: pazim       ! solar azimuthal angle (rad from n, clock)
   real,          dimension(1)     :: psw_bands   ! middle wavelength of each band
   real, pointer, dimension(:,:)   :: pdir_sw     ! direct ingoing solar radiation
   real,          dimension(n,1)   :: psca_sw     ! diffuse ingoing solar radiation
   real, pointer, dimension(:)     :: plw         ! ingoing longwave radiation
   real, pointer, dimension(:)     :: pta         ! air temperature at forcing level
   real, pointer, dimension(:)     :: pqa         ! air specific humidity at forcing level
   real,          dimension(n)     :: prhoa       ! air density at forcing level
   real, pointer, dimension(:)     :: pu          ! zonal wind component       
   real, pointer, dimension(:)     :: pv          ! meridional wind component       
   real, pointer, dimension(:)     :: pps         ! surface pressure 
   real, pointer, dimension(:)     :: ppa         ! air pressure at forcing level
   real,          dimension(n)     :: psnow       ! snow rate
   real,          dimension(n)     :: prain       ! rain rate
   real, pointer, dimension(:)     :: pzref       ! height of forcing level for t and q
   real, pointer, dimension(:)     :: puref       ! height of forcing level for the wind
   real, pointer, dimension(:)     :: plat        ! latitude
   real,          dimension(n)     :: psfth       ! flux of heat 
   real,          dimension(n)     :: psftq       ! flux of water vapor    
   real,          dimension(n)     :: psfu        ! zonal momentum flux         
   real,          dimension(n)     :: psfv        ! meridional momentum flux       
   real,          dimension(n)     :: pemis       ! emissivity
   real,          dimension(n)     :: ptrad       ! radiative temperature
   real,          dimension(n,1)   :: pdir_alb    ! direct albedo for each band
   real,          dimension(n,1)   :: psca_alb    ! diffuse albedo for each band
   real,          dimension(n)     :: ztvi        ! virtual temperature
   real,          dimension(n)     :: zvdir       ! direction of the wind
   real,          dimension(n)     :: zvmod       ! module of the wind
   real,          dimension(n)     :: ribn
   real,          dimension(n)     :: lzz0
   real,          dimension(n)     :: lzz0t
   real,          dimension(n)     :: fm
   real,          dimension(n)     :: fh
   real,          dimension(n)     :: dfm
   real,          dimension(n)     :: dfh
   real,          dimension(n)     :: geop
   real,          dimension(n)     :: lat,lon     ! latitude and longitude
   !     variables pour le calcul des angles solaires
   real, target, dimension(n) ::  zday, zheure, zmin, &
        xtsun, xzenith, xazimsol
!---------------------------------------------------------------------------

   !# in  offline mode the t-step 0 is (correctly) not performed
   if (fluvert.eq.'SURFACE'.and.kount.eq.0) return

   surflen = m

#include "town_ptr_as.cdk"

   nroof_layer = roof_layer
   nroad_layer = road_layer
   nwall_layer = wall_layer

   
   !     Allocation
   !     ----------
   !     6. Diagnostic variables :
   allocate( xq_town      (n)             , stat=alloc_status(40) )
   allocate( xles_roof    (n)             , stat=alloc_status(45) )
   allocate( xrunoff_roof (n)             , stat=alloc_status(47) )
   allocate( xles_road    (n)             , stat=alloc_status(51) )
   allocate( xrunoff_road (n)             , stat=alloc_status(53) )
   allocate( xrnsnow_roof (n)             , stat=alloc_status(58) )
   allocate( xhsnow_roof  (n)             , stat=alloc_status(59) )
   allocate( xlesnow_roof (n)             , stat=alloc_status(60) )
   allocate( xgsnow_roof  (n)             , stat=alloc_status(61) )
   allocate( xmelt_roof   (n)             , stat=alloc_status(62) )
   allocate( xrnsnow_road (n)             , stat=alloc_status(63) )
   allocate( xhsnow_road  (n)             , stat=alloc_status(64) )
   allocate( xlesnow_road (n)             , stat=alloc_status(65) )
   allocate( xgsnow_road  (n)             , stat=alloc_status(66) )
   allocate( xmelt_road   (n)             , stat=alloc_status(67) )
   allocate( xevap        (n)             , stat=alloc_status(72) )
   allocate( xrunoff      (n)             , stat=alloc_status(73) )
   allocate( xch          (n)             , stat=alloc_status(74) )
   allocate( xri          (n)             , stat=alloc_status(75) )
   allocate( xustar       (n)             , stat=alloc_status(76) )
   allocate( xz0_roof     (n)             , stat=alloc_status(77) )
   allocate( xz0_road     (n)             , stat=alloc_status(78) )

   !------------------------------------------------------------------------

   !     Initialisation
   !     --------------
   !     6. Diagnostic variables : 
   xq_town         = xundef
   xles_roof       = xundef
   xrunoff_roof    = xundef
   xles_road       = xundef
   xrunoff_road    = xundef
   xrnsnow_roof    = xundef
   xhsnow_roof     = xundef
   xlesnow_roof    = xundef
   xgsnow_roof     = xundef
   xmelt_roof      = xundef
   xrnsnow_road    = xundef
   xhsnow_road     = xundef
   xlesnow_road    = xundef
   xgsnow_road     = xundef
   xmelt_road      = xundef
   xevap           = xundef
   xrunoff         = xundef
   xch             = xundef
   xri             = xundef
   xustar          = xundef
   !-------------------------------------------------------------------------

   call ini_csts

   !     Time
   !     ----
   julien       = juliand(dt,kount,date)
   ptime        = date(5)*3600. + date(6)/100. + dt*(kount)
   kday         = date(3) + int(ptime/86400.)
   kmonth       = date(2)
   kyear        = date(4)
   ptime        = amod(ptime,3600*24.)
   ptstep       = dt
   psw_bands    = 0.

   !-------------------------------------------------------------------------

!     Calcul de l'angle zenithal
!     --------------------------
      do i=1,n
        lat(i) = zdlat(i)*180./XPI
        lon(i) = zdlon(i)*180./XPI
      end do
      zday(:)   = julien
      zheure(:) = int(ptime/3600.)*1.
      zmin(:)   = (ptime/3600.-int(ptime/3600.))*60.

      call sunpos(kyear,kmonth,kday,ptime,lon,lat,xtsun,xzenith,xazimsol)

!---------------------------------------------------------------------------

      ptsun  (1:n)     => xtsun   (1:n)
      pzenith(1:n)     => xzenith (1:n)
      pazim  (1:n)     => xazimsol(1:n)
      pta    (1:n)     => bus(x(tmoins       ,1,nk) :)
      pqa    (1:n)     => bus(x(humoins      ,1,nk) :)
      pu     (1:n)     => bus(x(umoins       ,1,nk) :)
      pv     (1:n)     => bus(x(vmoins       ,1,nk) :)
      ppa    (1:n)     => bus(x(pmoins       ,1,1 ) :)
      pps    (1:n)     => bus(x(pmoins       ,1,1 ) :)
      pdir_sw(1:n,1:1) => bus(x(fdss         ,1,1 ) :)
      plw    (1:n)     => bus(x(fdsi         ,1,1 ) :)
      pzref  (1:n)     => bus(x(ztsl         ,1,1 ) :)
      puref  (1:n)     => bus(x(zusl         ,1,1 ) :)
      plat   (1:n)     => bus(x(dlat         ,1,1 ) :)

      do i=1,n
        psnow         (i) = zsnowrate(i) *1000.
        prain         (i) = zrainrate(i) *1000.
        psca_sw       (i,1) = 0.
        ztvi          (i) = pta(i)*(1+((XRV/XRD)-1)*pqa(i))
        prhoa         (i) = pps(i)/XRD/ztvi(i)
      enddo
!       General variables 
!       -----------------
      xzs    (1:n)     => bus(x(gztherm      ,1,nk) :)
      xtown  (1:n)     => bus(x(urban        ,1,1 ) :)
      geop = xzs * 9.81  !lowest level height to geopotential

!     Urban parameters
!     ----------------
!     1. Geometric parameters :
      xbld          (1:n) => bus(x(bld         ,1,1) : )
      xbld_height   (1:n) => bus(x(bld_height  ,1,1) : )
      xz0_town      (1:n) => bus(x(z0_town     ,1,1) : )
      xz0_roof      (1:n) => bus(x(z0_roof     ,1,1) : )
      xz0_road      (1:n) => bus(x(z0_road     ,1,1) : )
      xwall_o_hor   (1:n) => bus(x(wall_o_hor  ,1,1) : )
      xcan_hw_ratio (1:n) => bus(x(can_hw_ratio,1,1) : )
      xsvf_road     (1:n) => bus(x(svf_road    ,1,1) : )
      xsvf_wall     (1:n) => bus(x(svf_wall    ,1,1) : )
!     2. Radiative properties :
      xalb_roof     (1:n) => bus(x(alb_roof    ,1,1) : )
      xalb_road     (1:n) => bus(x(alb_road    ,1,1) : )
      xalb_wall     (1:n) => bus(x(alb_wall    ,1,1) : )
      xemis_roof    (1:n) => bus(x(emis_roof   ,1,1) : )
      xemis_road    (1:n) => bus(x(emis_road   ,1,1) : )
      xemis_wall    (1:n) => bus(x(emis_wall   ,1,1) : )
!     4. Anthropogenic fluxes :
      xh_traffic    (1:n) => bus(x(h_traffic   ,1,1) : )
      xle_traffic   (1:n) => bus(x(le_traffic  ,1,1) : )
      xh_industry   (1:n) => bus(x(h_industry  ,1,1) : )
      xle_industry  (1:n) => bus(x(le_industry ,1,1) : )
!     4. Pronostic variables :
      xws_roof      (1:n) => bus(x(ws_roof     ,1,1) : )
      xws_road      (1:n) => bus(x(ws_road     ,1,1) : )
      xt_canyon     (1:n) => bus(x(t_canyon    ,1,1) : )
      xq_canyon     (1:n) => bus(x(q_canyon    ,1,1) : )
      xti_bld       (1:n) => bus(x(ti_bld      ,1,1) : )
      xti_road      (1:n) => bus(x(ti_road     ,1,1) : )
      xhc_roof (1:n,1:nroof_layer) => bus(x(hc_roof ,1,1) : )
      xtc_roof (1:n,1:nroof_layer) => bus(x(tc_roof ,1,1) : )
      xd_roof  (1:n,1:nroof_layer) => bus(x(d_roof  ,1,1) : )
      xt_roof  (1:n,1:nroof_layer) => bus(x(t_roof  ,1,1) : )
      xhc_road (1:n,1:nroad_layer) => bus(x(hc_road ,1,1) : )
      xtc_road (1:n,1:nroad_layer) => bus(x(tc_road ,1,1) : )
      xd_road  (1:n,1:nroad_layer) => bus(x(d_road  ,1,1) : )
      xt_road  (1:n,1:nroad_layer) => bus(x(t_road  ,1,1) : )
      xhc_wall (1:n,1:nwall_layer) => bus(x(hc_wall ,1,1) : )
      xtc_wall (1:n,1:nwall_layer) => bus(x(tc_wall ,1,1) : )
      xd_wall  (1:n,1:nwall_layer) => bus(x(d_wall  ,1,1) : )
      xt_wall  (1:n,1:nwall_layer) => bus(x(t_wall  ,1,1) : )
!     5. Snow variables :
      xwsnow_roof (1:n) => bus(x(sroof_wsnow,1,1) : )
      xtsnow_roof (1:n) => bus(x(sroof_t    ,1,1) : )
      xrsnow_roof (1:n) => bus(x(sroof_rho  ,1,1) : )
      xasnow_roof (1:n) => bus(x(sroof_alb  ,1,1) : )
      xesnow_roof (1:n) => bus(x(sroof_emis ,1,1) : )
      xtssnow_roof(1:n) => bus(x(sroof_ts   ,1,1) : )
      xwsnow_road (1:n) => bus(x(sroad_wsnow,1,1) : )
      xtsnow_road (1:n) => bus(x(sroad_t    ,1,1) : )
      xrsnow_road (1:n) => bus(x(sroad_rho  ,1,1) : )
      xasnow_road (1:n) => bus(x(sroad_alb  ,1,1) : )
      xesnow_road (1:n) => bus(x(sroad_emis ,1,1) : )
      xtssnow_road(1:n) => bus(x(sroad_ts   ,1,1) : )
!     6. Diagnostic variables :
      xu_canyon   (1:n) => bus(x( u_canyon,1,1)  : )
      xrn         (1:n) => bus(x( rn_town ,1,1)  : )
      xh          (1:n) => bus(x( h_town  ,1,1)  : )
      xle         (1:n) => bus(x( le_town ,1,1)  : )
      xgflux      (1:n) => bus(x( g_town  ,1,1)  : )
      xrn_roof    (1:n) => bus(x( rn_roof ,1,1)  : )
      xh_roof     (1:n) => bus(x( h_roof  ,1,1)  : )
      xle_roof    (1:n) => bus(x( le_roof ,1,1)  : )
      xgflux_roof (1:n) => bus(x( g_roof  ,1,1)  : )
      xrn_road    (1:n) => bus(x( rn_road ,1,1)  : )
      xh_road     (1:n) => bus(x( h_road  ,1,1)  : )
      xle_road    (1:n) => bus(x( le_road ,1,1)  : )
      xgflux_road (1:n) => bus(x( g_road  ,1,1)  : )
      xrn_wall    (1:n) => bus(x( rn_wall ,1,1)  : )
      xh_wall     (1:n) => bus(x( h_wall  ,1,1)  : )
      xle_wall    (1:n) => bus(x( le_wall ,1,1)  : )
      xgflux_wall (1:n) => bus(x( g_wall  ,1,1)  : )


       ! coherence between solar zenithal angle and radiation
       !
       where (sum(pdir_sw+psca_sw,2)>0.)
         pzenith = min (pzenith,xpi/2.-0.01)
       elsewhere
         pzenith = max (pzenith,xpi/2.)
       end where

      do i=1,n
        ztsun  (i) = ptsun  (i)
        zzenith(i) = pzenith(i)
        zazim  (i) = pazim  (i)
      enddo
!-----------------------------------------------------------------------------

      call    coupling_teb2 (ptstep, kyear, kmonth, kday, ptime,            &
              ptsun, pzenith, pazim, pzref, puref, xzs, pu, pv, pqa, pta,   &
              prhoa, prain, psnow, plw, pdir_sw, psca_sw, psw_bands, pps,   &
              ppa, psftq, psfth, psfu, psfv,                                &
              ptrad, pdir_alb, psca_alb, pemis, plat                        )

!-----------------------------------------------------------------------------

      do i=1,n
!       Variables a agreger
!       -------------------
        zz0(i) = xz0_town   (i)
!   evaluation of the thermal roughness lenght for the diagnostic (first guess)
!   will come from the bus later when z0 is read in geophy (not this version)

        zz0t(i) =  MAX( xz0_town(i)                      *          &
                7.4 * exp( - 1.29 *(  xz0_town(i)        *          &
                0.4 * (pu(i)**2 + pv(i) **2 )**0.5                  &
                /log((puref(i)+xbld_height(i)/3.)/xz0_town(i))      &
                /1.461e-5)**0.25), 1.e-20 )
        ztsurf (i) = ptrad      (i)
        ztsrad (i) = ptrad      (i)
        zqsurf (i) = xq_town    (i)
        zalvis (i) = pdir_alb   (i,1)
        zsnodp (i) = 0. 
        zfc    (i) = psfth      (i)
        zfv    (i) = psftq      (i) * xlvtt
      end do

!-----------------------------------------------------------------------------
!   Diagnostics (tdiag, qdiag at z=zh/2; udiag, vdiag at z=zu above roof level)
!-----------------------------------------------------------------------------
      i = sl_prelim(bus(x(thetaa,1,1):x(thetaa,1,1)+n-1),bus(x(humoins,1,nk):x(humoins,1,nk)+n-1), &
           zumoins,zvmoins,pps,bus(x(zusl,1,1):x(zusl,1,1)+n-1),min_wind_speed=1e-4,spd_air=zvmod,dir_air=zvdir)
      i = sl_sfclayer(bus(x(thetaa,1,1):x(thetaa,1,1)+n-1),bus(x(humoins,1,nk):x(humoins,1,nk)+n-1),&
           zvmod,zvdir,bus(x(zusl,1,1):x(zusl,1,1)+n-1),bus(x(ztsl,1,1):x(ztsl,1,1)+n-1), &
           bus(x(tsurf,1,1):x(tsurf,1,1)+n-1),bus(x(qsurf,1,indx_urb):x(qsurf,1,indx_urb)+n-1), &
           bus(x(z0,1,indx_urb):x(z0,1,indx_urb)+n-1),bus(x(z0t,1,indx_urb):x(z0t,1,indx_urb)+n-1), &
           bus(x(dlat,1,1):x(dlat,1,1)+n-1),bus(x(fcor,1,1):x(fcor,1,1)+n-1),optz0=0,hghtm_diag=zu,hghtt_diag=zt, &
           ilmo=bus(x(ilmo,1,indx_urb):x(ilmo,1,indx_urb)+n-1),h=bus(x(hst,1,indx_urb):x(hst,1,indx_urb)+n-1), &
           ue=bus(x(frv,1,indx_urb):x(frv,1,indx_urb)+n-1),flux_t=bus(x(ftemp,1,indx_urb):x(ftemp,1,indx_urb)+n-1), &
           flux_q=bus(x(fvap,1,indx_urb):x(fvap,1,indx_urb)+n-1),coefm=bus(x(bm,1,1):x(bm,1,1)+n-1), &
           coeft=bus(x(bt,1,1):x(bt,1,1)+n-1))

      do i=1,n
         ztdiag(i) = zt_canyon(i)
         zqdiag(i) = zq_canyon(i)

         zalfat(i) = -psfth(i)/(xcpd*prhoa(i))
         zalfaq(i) = -psftq(i)

        if (.not.impflx) zbt(i) = 0.
        if (impflx) then
         zalfat(i) = - zbt(i) * ztsurf(i) 
         zalfaq(i) = - zbt(i) * zqsurf(i) 
        endif
      end do


      call fillagg ( bus, bussiz, ptsurf, ptsurfsiz, indx_urb, surflen )

!     6. diagnostic variables
      deallocate( xq_town           , stat=alloc_status(40) )
      deallocate( xles_roof         , stat=alloc_status(45) )
      deallocate( xrunoff_roof      , stat=alloc_status(47) )
      deallocate( xles_road         , stat=alloc_status(51) )
      deallocate( xrunoff_road      , stat=alloc_status(53) )
      deallocate( xrnsnow_roof      , stat=alloc_status(58) )
      deallocate( xhsnow_roof       , stat=alloc_status(59) )
      deallocate( xlesnow_roof      , stat=alloc_status(60) )
      deallocate( xgsnow_roof       , stat=alloc_status(61) )
      deallocate( xmelt_roof        , stat=alloc_status(62) )
      deallocate( xrnsnow_road      , stat=alloc_status(63) )
      deallocate( xhsnow_road       , stat=alloc_status(64) )
      deallocate( xlesnow_road      , stat=alloc_status(65) )
      deallocate( xgsnow_road       , stat=alloc_status(66) )
      deallocate( xmelt_road        , stat=alloc_status(67) )
      deallocate( xevap             , stat=alloc_status(72) )
      deallocate( xrunoff           , stat=alloc_status(73) )
      deallocate( xch               , stat=alloc_status(74) )
      deallocate( xri               , stat=alloc_status(75) )
      deallocate( xustar            , stat=alloc_status(76) )
      deallocate( xz0_roof          , stat=alloc_status(77) )
      deallocate( xz0_road          , stat=alloc_status(78) )

      !--------------------------------------------------------------------
   return
end subroutine town
