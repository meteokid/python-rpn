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

module sfcbus_mod
   implicit none 
   public
   save

#define SFCVAR(MYVAR,MYNAME) type(PHYVAR_T) :: MYVAR = PHYVAR_T(-1,0,0,0,0,.false.,MYNAME)

   type :: sfcptr
      sequence
      real, pointer :: ptr(:,:)
   end type sfcptr

   type :: PHYVAR_T
      sequence
      integer :: i, agg, mul, niveaux, mosaik
      logical :: doagg_L
      character(len=32) :: n
   end type PHYVAR_T

   type :: PHYVARLIST_T
      sequence

      SFCVAR(umoins , 'PW_UU:M')
      SFCVAR(uplus  , 'PW_UU:P')
      SFCVAR(vmoins , 'PW_VV:M')
      SFCVAR(vplus  , 'PW_VV:P')
      SFCVAR(tmoins , 'PW_TT:M')
      SFCVAR(tplus  , 'PW_TT:P')
      SFCVAR(sigm   , 'PW_PM:P')
      SFCVAR(humoins, 'TR/HU:M')
      SFCVAR(huplus , 'TR/HU:P')

      SFCVAR(acoef, 'acoef')
      SFCVAR(alb_road, 'alb_road')
      SFCVAR(alb_roaden, 'alb_roaden')
      SFCVAR(alb_roof, 'alb_roof')
      SFCVAR(alb_roofen, 'alb_roofen')
      SFCVAR(alb_wall, 'alb_wall')
      SFCVAR(alb_wallen, 'alb_wallen')
      SFCVAR(alen, 'alen')
      SFCVAR(alfaq, 'alfaq')
      SFCVAR(alfat, 'alfat')
      SFCVAR(alveg, 'alveg')
      SFCVAR(alvis, 'alvis')
      SFCVAR(azim, 'azim')
      SFCVAR(bcoef, 'bcoef')
      SFCVAR(bld, 'bld')
      SFCVAR(blden, 'blden')
      SFCVAR(bld_height, 'bld_height')
      SFCVAR(bld_heighten, 'bld_heighten')
      SFCVAR(bm, 'bm')
      SFCVAR(bt, 'bt')
      SFCVAR(c1sat, 'c1sat')
      SFCVAR(c2ref, 'c2ref')
      SFCVAR(c3ref, 'c3ref')
      SFCVAR(can_hw_ratio, 'can_hw_ratio')
      SFCVAR(cgsat, 'cgsat')
      SFCVAR(clay, 'clay')
      SFCVAR(clayen, 'clayen')
      SFCVAR(cosz, 'cosz')
      SFCVAR(cveg, 'cveg')
      SFCVAR(d_road, 'd_road')
      SFCVAR(d_roaden, 'd_roaden')
      SFCVAR(d_roof, 'd_roof')
      SFCVAR(d_roofen, 'd_roofen')
      SFCVAR(d_wall, 'd_wall')
      SFCVAR(d_wallen, 'd_wallen')
      SFCVAR(dhdx, 'dhdx')
      SFCVAR(dhdxdy, 'dhdxdy')
      SFCVAR(dhdxdyen, 'dhdxdyen')
      SFCVAR(dhdxen, 'dhdxen')
      SFCVAR(dhdy, 'dhdy')
      SFCVAR(dhdyen, 'dhdyen')
      SFCVAR(dlat, 'dlat')
      SFCVAR(dlon, 'dlon')
      SFCVAR(drain, 'drain')
      SFCVAR(dsst, 'dsst')
      SFCVAR(dtdiag, 'dtdiag')
      SFCVAR(eflux, 'eflux')
      SFCVAR(emis_road, 'emis_road')
      SFCVAR(emis_roaden, 'emis_roaden')
      SFCVAR(emis_roof, 'emis_roof')
      SFCVAR(emis_roofen, 'emis_roofen')
      SFCVAR(emis_wall, 'emis_wall')
      SFCVAR(emis_wallen, 'emis_wallen')
      SFCVAR(en, 'en')
      SFCVAR(epstfn, 'epstfn')
      SFCVAR(fc, 'fc')
      SFCVAR(fcor, 'fcor')
      SFCVAR(fdsi, 'fdsi')
      SFCVAR(fdss, 'fdss')
      SFCVAR(fl, 'fl')
      SFCVAR(flusolis, 'flusolis')
      SFCVAR(fluslop, 'fluslop')
      SFCVAR(fq, 'fq')
      SFCVAR(frv, 'frv')
      SFCVAR(ftemp, 'ftemp')
      SFCVAR(fv, 'fv')
      SFCVAR(fvap, 'fvap')
      SFCVAR(g_road, 'g_road')
      SFCVAR(g_roof, 'g_roof')
      SFCVAR(g_town, 'g_town')
      SFCVAR(g_wall, 'g_wall')
      SFCVAR(gamveg, 'gamveg')
      SFCVAR(gc, 'gc')
      SFCVAR(glacen, 'glacen')
      SFCVAR(glacier, 'glacier')
      SFCVAR(glsea, 'glsea')
      SFCVAR(glsea0, 'glsea0')
      SFCVAR(glseaen, 'glseaen')
      SFCVAR(gztherm, 'gztherm')
      SFCVAR(h, 'h')
      SFCVAR(h_industry, 'h_industry')
      SFCVAR(h_industryen, 'h_industryen')
      SFCVAR(h_road, 'h_road')
      SFCVAR(h_roof, 'h_roof')
      SFCVAR(h_town, 'h_town')
      SFCVAR(h_traffic, 'h_traffic')
      SFCVAR(h_trafficen, 'h_trafficen')
      SFCVAR(h_wall, 'h_wall')
      SFCVAR(hc_road, 'hc_road')
      SFCVAR(hc_roaden, 'hc_roaden')
      SFCVAR(hc_roof, 'hc_roof')
      SFCVAR(hc_roofen, 'hc_roofen')
      SFCVAR(hc_wall, 'hc_wall')
      SFCVAR(hc_wallen, 'hc_wallen')
      SFCVAR(hst, 'hst')
      SFCVAR(husurf, 'husurf')
      SFCVAR(hv, 'hv')
      SFCVAR(icedp, 'icedp')
      SFCVAR(icedpen, 'icedpen')
      SFCVAR(iceline, 'iceline')
      SFCVAR(icelinen, 'icelinen')
      SFCVAR(ilmo, 'ilmo')
      SFCVAR(isoil, 'isoil')
      SFCVAR(isoilen, 'isoilen')
      SFCVAR(kcl, 'kcl')
      SFCVAR(km, 'km')
      SFCVAR(kt, 'kt')
      SFCVAR(lai, 'lai')
      SFCVAR(le_industry, 'le_industry')
      SFCVAR(le_industryen, 'le_industryen')
      SFCVAR(le_road, 'le_road')
      SFCVAR(le_roof, 'le_roof')
      SFCVAR(le_town, 'le_town')
      SFCVAR(le_traffic, 'le_traffic')
      SFCVAR(le_trafficen, 'le_trafficen')
      SFCVAR(le_wall, 'le_wall')
      SFCVAR(leg, 'leg')
      SFCVAR(ler, 'ler')
      SFCVAR(les, 'les')
      SFCVAR(letr, 'letr')
      SFCVAR(lev, 'lev')
      SFCVAR(lhtg, 'lhtg')
      SFCVAR(lhtgen, 'lhtgen')
      SFCVAR(melts, 'melts')
      SFCVAR(meltsr, 'meltsr')
      SFCVAR(mf, 'mf')
      SFCVAR(mg, 'mg')
      SFCVAR(mgen, 'mgen')
      SFCVAR(ml, 'ml')
      SFCVAR(mt, 'mt')
      SFCVAR(mtdir, 'mtdir')
      SFCVAR(nat, 'nat')
      SFCVAR(overfl, 'overfl')
      SFCVAR(pav, 'pav')
      SFCVAR(paven, 'paven')
      SFCVAR(pcoef, 'pcoef')
      SFCVAR(pmoins, 'pmoins')
      SFCVAR(psn, 'psn')
      SFCVAR(psng, 'psng')
      SFCVAR(psnv, 'psnv')
      SFCVAR(q_canyon, 'q_canyon')
      SFCVAR(q_canyonen, 'q_canyonen')
      SFCVAR(qdiag, 'qdiag')
      SFCVAR(qsurf, 'qsurf')
      SFCVAR(rainrate, 'rainrate')
      SFCVAR(resa, 'resa')
      SFCVAR(rgl, 'rgl')
      SFCVAR(rib, 'rib')
      SFCVAR(rn_road, 'rn_road')
      SFCVAR(rn_roof, 'rn_roof')
      SFCVAR(rn_town, 'rn_town')
      SFCVAR(rn_wall, 'rn_wall')
      SFCVAR(rnet_s, 'rnet_s')
      SFCVAR(rootdp, 'rootdp')
      SFCVAR(rst, 'rst')
      SFCVAR(rt, 'rt')
      SFCVAR(sand, 'sand')
      SFCVAR(sanden, 'sanden')
      SFCVAR(scl, 'scl')
      SFCVAR(sfcwgt, 'sfcwgt')
      SFCVAR(skin_depth, 'skin_depth')
      SFCVAR(skin_inc, 'skin_inc')
      SFCVAR(slope, 'slope')
      SFCVAR(snoagen, 'snoagen')
      SFCVAR(snoal, 'snoal')
      SFCVAR(snoalen, 'snoalen')
      SFCVAR(snoden, 'snoden')
      SFCVAR(snodp, 'snodp')
      SFCVAR(snodpen, 'snodpen')
      SFCVAR(snoma, 'snoma')
      SFCVAR(snoro, 'snoro')
      SFCVAR(snoroen, 'snoroen')
      SFCVAR(snowrate, 'snowrate')
      SFCVAR(sroad_alb, 'sroad_alb')
      SFCVAR(sroad_alben, 'sroad_alben')
      SFCVAR(sroad_emis, 'sroad_emis')
      SFCVAR(sroad_emisen, 'sroad_emisen')
      SFCVAR(sroad_rho, 'sroad_rho')
      SFCVAR(sroad_rhoen, 'sroad_rhoen')
      SFCVAR(sroad_t, 'sroad_t')
      SFCVAR(sroad_ten, 'sroad_ten')
      SFCVAR(sroad_ts, 'sroad_ts')
      SFCVAR(sroad_tsen, 'sroad_tsen')
      SFCVAR(sroad_wsnow, 'sroad_wsnow')
      SFCVAR(sroad_wsnowen, 'sroad_wsnowen')
      SFCVAR(sroof_alb, 'sroof_alb')
      SFCVAR(sroof_alben, 'sroof_alben')
      SFCVAR(sroof_emis, 'sroof_emis')
      SFCVAR(sroof_emisen, 'sroof_emisen')
      SFCVAR(sroof_rho, 'sroof_rho')
      SFCVAR(sroof_rhoen, 'sroof_rhoen')
      SFCVAR(sroof_t, 'sroof_t')
      SFCVAR(sroof_ten, 'sroof_ten')
      SFCVAR(sroof_ts, 'sroof_ts')
      SFCVAR(sroof_tsen, 'sroof_tsen')
      SFCVAR(sroof_wsnow, 'sroof_wsnow')
      SFCVAR(sroof_wsnowen, 'sroof_wsnowen')
      SFCVAR(stomr, 'stomr')
      SFCVAR(svf_road, 'svf_road')
      SFCVAR(svf_wall, 'svf_wall')
      SFCVAR(t_canyon, 't_canyon')
      SFCVAR(t_canyonen, 't_canyonen')
      SFCVAR(t_road, 't_road')
      SFCVAR(t_roaden, 't_roaden')
      SFCVAR(t_roof, 't_roof')
      SFCVAR(t_roofen, 't_roofen')
      SFCVAR(t_wall, 't_wall')
      SFCVAR(t_wallen, 't_wallen')
      SFCVAR(tc_road, 'tc_road')
      SFCVAR(tc_roaden, 'tc_roaden')
      SFCVAR(tc_roof, 'tc_roof')
      SFCVAR(tc_roofen, 'tc_roofen')
      SFCVAR(tc_wall, 'tc_wall')
      SFCVAR(tc_wallen, 'tc_wallen')
      SFCVAR(tdiag, 'tdiag')
      SFCVAR(tglacen, 'tglacen')
      SFCVAR(tglacier, 'tglacier')
      SFCVAR(thetaa, 'thetaa')
      SFCVAR(ti_bld, 'ti_bld')
      SFCVAR(ti_blden, 'ti_blden')
      SFCVAR(ti_road, 'ti_road')
      SFCVAR(ti_roaden, 'ti_roaden')
      SFCVAR(tmice, 'tmice')
      SFCVAR(tmicen, 'tmicen')
      SFCVAR(tnolim, 'tnolim')
      SFCVAR(tsoil, 'tsoil')
      SFCVAR(tsoilen, 'tsoilen')
      SFCVAR(tsrad, 'tsrad')
      SFCVAR(tss, 'tss')
      SFCVAR(tsun, 'tsun')
      SFCVAR(tsurf, 'tsurf')
      SFCVAR(tve, 'tve')
      SFCVAR(twater, 'twater')
      SFCVAR(twateren, 'twateren')
      SFCVAR(u_canyon, 'u_canyon')
      SFCVAR(udiag, 'udiag')
      SFCVAR(urban, 'urban')
      SFCVAR(vdiag, 'vdiag')
      SFCVAR(vegf, 'vegf')
      SFCVAR(vegfen, 'vegfen')
      SFCVAR(vegfrac, 'vegfrac')
      SFCVAR(wall_o_hor, 'wall_o_hor')
      SFCVAR(wall_o_horen, 'wall_o_horen')
      SFCVAR(wfc, 'wfc')
      SFCVAR(wflux, 'wflux')
      SFCVAR(ws_road, 'ws_road')
      SFCVAR(ws_roaden, 'ws_roaden')
      SFCVAR(ws_roof, 'ws_roof')
      SFCVAR(ws_roofen, 'ws_roofen')
      SFCVAR(wsat, 'wsat')
      SFCVAR(wsnow, 'wsnow')
      SFCVAR(wsnowen, 'wsnowen')
      SFCVAR(wsoil, 'wsoil')
      SFCVAR(wsoilen, 'wsoilen')
      SFCVAR(wveg, 'wveg')
      SFCVAR(wvegen, 'wvegen')
      SFCVAR(wwilt, 'wwilt')
      SFCVAR(xcent, 'xcent')
      SFCVAR(z0, 'z0')
      SFCVAR(z0_road, 'z0_road')
      SFCVAR(z0_roaden, 'z0_roaden')
      SFCVAR(z0_roof, 'z0_roof')
      SFCVAR(z0_roofen, 'z0_roofen')
      SFCVAR(z0_town, 'z0_town')
      SFCVAR(z0_townen, 'z0_townen')
      SFCVAR(z0en, 'z0en')
      SFCVAR(z0t, 'z0t')
      SFCVAR(za, 'za')
      SFCVAR(ze, 'ze')
      SFCVAR(zenith, 'zenith')
      SFCVAR(ztsl, 'ztsl')
      SFCVAR(zusl, 'zusl')

   end type PHYVARLIST_T

   integer, parameter :: INDX_SOIL    =  1
   integer, parameter :: INDX_GLACIER =  2
   integer, parameter :: INDX_WATER   =  3
   integer, parameter :: INDX_ICE     =  4
   integer, parameter :: INDX_AGREGE  =  5
   integer, parameter :: INDX_URB     =  6
   integer, parameter :: INDX_MAX     =  6

   type(PHYVARLIST_T), target  :: vd
   type(PHYVAR_T), allocatable :: vl(:)
   type(sfcptr), allocatable :: busptr(:)
   integer, allocatable :: statut(:,:)

   integer :: surfesptot = 0
   integer :: nvarsurf = 0  !# Number of surface bus var
   integer :: nsurf    = 0  !# Number of surface "types"
   integer :: tsrad_i=0, z0_i=0, z0t_i=0 !#TODO: remove, replace by vd%tsrad...

   integer :: drain=0
   integer :: drainaf=0
   integer :: insmavg=0
   integer :: isoil=0
   integer :: leg=0
   integer :: legaf=0
   integer :: ler=0
   integer :: leraf=0
   integer :: les=0
   integer :: lesaf=0
   integer :: letr=0
   integer :: letraf=0
   integer :: lev=0
   integer :: levaf=0
   integer :: overfl=0
   integer :: overflaf=0
   integer :: rootdp=0
   integer :: wflux=0
   integer :: wfluxaf=0
   integer :: wsoil=0

contains


   function sfcbus_init() result(F_istat)
      use phy_itf, only: phymeta
      use phygetmetaplus_mod, only: phymetaplus, phygetmetaplus
      use sfc_options, only: schmurb
      implicit none
      integer :: F_istat

#include <msg.h>
#include <rmnlib_basics.hf>
#include <clib_interface_mu.hf>

      integer :: i, istat, mulmax, idxmax
      type(PHYVAR_T) :: vl0(1)
      type(phymeta) :: mymeta
      type(phymetaplus) :: mymetaplus

      F_istat = RMN_ERR

      if (nsurf == 0) then
         idxmax = max(INDX_SOIL, INDX_GLACIER, INDX_WATER, INDX_ICE, INDX_AGREGE)
         if (schmurb /= 'NIL') idxmax = max(idxmax, INDX_URB)
         nsurf = idxmax - 1
      endif
      
      nvarsurf = size(transfer(vd, vl0))
      allocate(vl(nvarsurf))
      allocate(busptr(nvarsurf))
      vl = transfer(vd, vl)
      mulmax = 0
      do i = 1,nvarsurf
         vl(i)%i = i
         istat = clib_toupper(vl(i)%n)
         nullify(busptr(i)%ptr)
         istat = phygetmetaplus(mymetaplus, vl(i)%n, F_npath='V', F_bpath='DPVE', F_quiet=.true., F_shortmatch=.false.)
         if (istat >= 0) then
            mymeta = mymetaplus%meta
            !#TODO: put an upper bound on ptr so -C would bite!
            busptr(i)%ptr => mymetaplus%ptr(mymetaplus%index:,:)
            vl(i)%doagg_L = (mymeta%bus(1:1) /= 'E')
            vl(i)%mul = mymeta%fmul
            vl(i)%niveaux = mymeta%nk
            vl(i)%mosaik = mymeta%mosaic + 1
            mulmax = max(mulmax, vl(i)%mul)
         endif
         if (vl(i)%n == 'TSRAD') tsrad_i = i
         if (vl(i)%n == 'Z0')    z0_i = i
         if (vl(i)%n == 'Z0T')   z0t_i = i

      enddo
      vd = transfer(vl, vd)
 
      allocate(statut(nvarsurf, mulmax))
      statut = 0

      F_istat = RMN_OK
      return
   end function sfcbus_init

end module sfcbus_mod
