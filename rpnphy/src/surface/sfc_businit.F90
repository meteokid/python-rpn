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
subroutine sfc_businit(moyhr,n,nk)
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object Establishes requirements in terms of variables in the 4 main buses 
   !        (busent, busdyn, busper and busvol) for the TEB (Urban scheme).
   !@Arguments
   integer, intent(in) ::  moyhr,n,nk !# horiz and vert dimensions
   !@Author M. Desgagne (Oct 1995)

   !@Revision
   ! 001      L. Spacek  (Aug 2010) - Complete rewrite
   ! 002      B. Dugas   (Oct 2010) - A few small corrections
   ! 003      L. Spacek  (Sep 2011) - Eliminate obsolete convection options
   !*@/

   integer :: ier, nmos
   integer :: alb_road, alb_roaden, alb_roof, alb_roofen, alb_wall, &
        alb_wallen, azim, bld, blden, bld_height, bld_heighten, can_hw_ratio, &
        d_road, d_roaden, d_roof, d_roofen, d_wall, d_wallen, emis_road, &
        emis_roaden, emis_roof, emis_roofen, emis_wall, emis_wallen, g_road, &
        g_roof, g_town, g_wall, h_industry, h_industryen, h_road, h_roof, &
        h_town, h_traffic, h_trafficen, h_wall, hc_road, hc_roaden, hc_roof, &
        hc_roofen, hc_wall, hc_wallen, le_industry, le_industryen, le_road, &
        le_roof, le_town, le_traffic, le_trafficen, le_wall, nat, naten, pav, &
        paven, q_canyon, q_canyonen, rn_road, rn_roof, rn_town, rn_wall, &
        sroad_alb, sroad_alben, sroad_emis, sroad_emisen, sroad_rho, &
        sroad_rhoen, sroad_scheme, sroad_t, sroad_ten, sroad_ts, sroad_tsen, &
        sroad_wsnow, sroad_wsnowen, sroof_alb, sroof_alben, sroof_emis, &
        sroof_emisen, sroof_rho, sroof_rhoen, sroof_scheme, sroof_t, &
        sroof_ten, sroof_ts, sroof_tsen, sroof_wsnow, sroof_wsnowen, &
        svf_road, svf_wall, t_canyon, t_canyonen, t_road, t_roaden, t_roof, &
        t_roofen, t_wall, t_wallen, tc_road, tc_roaden, tc_roof, tc_roofen, &
        tc_wall, tc_wallen, ti_bld, ti_blden, ti_road, ti_roaden, tsun, &
        u_canyon, wall_o_hor, wall_o_horen, ws_road, ws_roaden, ws_roof, &
        ws_roofen, z0_road, z0_roaden, z0_roof, z0_roofen, z0_town, &
        z0_townen, zenith
   integer :: acoef, alveg, bcoef, c1sat, c2ref, c3ref, clay, clayen, cveg, &
        eflux, gamveg, husurf, hv, iceline, icelinen, isoilen, lai, melts,  &
        meltsr, pcoef, psn, psng, psnv, resa, rgl, rnet_s, rst, sand, sanden, &
        snoagen, snoalen, snoma, snoro, snoroen, stomr, tsoil, tsoilen, vegf, &
        vegfen, vegfrac, wfc, wsat, wsnow, wsnowen, wsoilen, wveg, wvegen, wwilt
   integer :: alen, cgsat, dsst, dtdiag, glacen, glacier, glsea0, glseaen, &
        icedp, icedpen, mgen, sfcwgt, skin_depth, skin_inc, snoal, snoden, &
        snodp, snodpen, tglacen, tglacier, tmice, tmicen, tnolim, &
        twater, twateren, urban, z0en
   character(len=2) :: nm, nagg, nrow
   !---------------------------------------------------------------------
   write(nagg,'(i2)') nsurf+1

   !# nm is the number of mosaic 'layers'
   nmos = 0
   write(nm,'(i2)') nmos !#TODO: delete

   !# nl is the number of levels in sea ice
   write(nrow,'(i2)') nl

   !#TODO: check if schmsol conditional
   call gesdict(n, nk, alen,         'VN=alen         ;ON=1A  ;VD=visible albedo (E)                             ;VS=row                    ;VB=e1;IN=AL  ;MIN=0')
   call gesdict(n, nk, cgsat,        'VN=cgsat        ;ON=6I  ;VD=thermal coef. at saturation                    ;VS=row                    ;VB=p0')
   call gesdict(n, nk, dsst,         'VN=dsst         ;ON=DSST;VD=warm layer diurnal SST increment               ;VS=row                    ;VB=p0')
   call gesdict(n, nk, dtdiag,       'VN=dtdiag       ;ON=DLIM;VD=DeltaT at screen level of tdiaglim             ;VS=row                    ;VB=p0')
   call gesdict(n, nk, glacen,       'VN=glacen       ;ON=1F  ;VD=continental ice cover (E)                      ;VS=row                    ;VB=e1;IN=GA  ;MIN=0')
   call gesdict(n, nk, glacier,      'VN=glacier      ;ON=2F  ;VD=continental ice fraction                       ;VS=row                    ;VB=p0        ;MIN=0')
   call gesdict(n, nk, glsea0,       'VN=glsea0       ;ON=GY  ;VD=sea ice fraction (unmodified)                  ;VS=row                    ;VB=p0        ;MIN=0')
   call gesdict(n, nk, glseaen,      'VN=glseaen      ;ON=4A  ;VD=oceanic ice cover (E)                          ;VS=row                    ;VB=e1;IN=LG  ;MIN=0')
   call gesdict(n, nk, icedp,        'VN=icedp        ;ON=I8  ;VD=sea ice thickness                              ;VS=row                    ;VB=p0        ;MIN=0')
   call gesdict(n, nk, icedpen,      'VN=icedpen      ;ON=2I  ;VD=sea ice thickness (E)                          ;VS=row                    ;VB=e1;IN=I8  ;MIN=0')
   call gesdict(n, nk, iceline,      'VN=iceline      ;ON=ICEL;VD=ice line                                       ;VS=row                    ;VB=p0')
   call gesdict(n, nk, mgen,         'VN=mgen         ;ON=6A  ;VD=land-sea mask (E)                              ;VS=row                    ;VB=e1;IN=MG  ;MIN=0')
   call gesdict(n, nk, sfcwgt,       'VN=sfcwgt       ;ON=WT  ;VD=wgt of sfc type over a tile                    ;VS=row*'//nagg//'         ;VB=v0        ;MIN=0')
   call gesdict(n, nk, skin_depth,   'VN=skin_depth   ;ON=SDEP;VD=sea surface cold skin depth                    ;VS=row                    ;VB=p0')
   call gesdict(n, nk, skin_inc,     'VN=skin_inc     ;ON=SINC;VD=sea surface cold skin SST increment            ;VS=row                    ;VB=p0')

   call gesdict(n, nk, snoden,       'VN=snoden       ;ON=DN  ;VD=snow density in kg/m3                          ;VS=row@'//nm//'           ;VB=p0        ;MIN=0')
   call gesdict(n, nk, snodp,        'VN=snodp        ;ON=SD  ;VD=snow depth                                     ;VS=row*'//nagg//'         ;VB=p0        ;MIN=0')
   call gesdict(n, nk, snodpen,      'VN=snodpen      ;ON=7A  ;VD=snow depth (E)                                 ;VS=row*'//nagg//'         ;VB=e1;IN=SD  ;MIN=0')
   call gesdict(n, nk, tglacen,      'VN=tglacen      ;ON=2W  ;VD=sfc and dp glacier tp.(E)                      ;VS=row*2                  ;VB=e1;IN=I9  ;')
   call gesdict(n, nk, tglacier,     'VN=tglacier     ;ON=I9  ;VD=glaciers temperature                           ;VS=row*2                  ;VB=p0')
   call gesdict(n, nk, tmice,        'VN=tmice        ;ON=I7  ;VD=sea ice temperature                            ;VS=row*'//nrow//'         ;VB=p0')
   call gesdict(n, nk, tmicen,       'VN=tmicen       ;ON=2S  ;VD=sea ice temperature (E)                        ;VS=row*'//nrow//'         ;VB=e1;IN=I7  ;')
   call gesdict(n, nk, tnolim,       'VN=tnolim       ;ON=TNOL;VD=screen level temp without max on gradient      ;VS=row                    ;VB=p0')
   call gesdict(n, nk, twater,       'VN=twater       ;ON=TM  ;VD=sea surface temperature                        ;VS=row                    ;VB=p0')
   call gesdict(n, nk, twateren,     'VN=twateren     ;ON=8A  ;VD=SST temperature (E)                            ;VS=row                    ;VB=e1;IN=TM  ;')
   call gesdict(n, nk, urban,        'VN=urban        ;ON=UR  ;VD=urban mask                                     ;VS=row                    ;VB=p0')

   call gesdict(n, nk, z0en,         'VN=z0en         ;ON=2B  ;VD=roughness length (E)                           ;VS=row                    ;VB=e1;IN=ZP  ;')

   if (moyhr > 0) &
        call gesdict (n, nk, insmavg,   'VN=insmavg      ;ON=IMAV;VD=integrated soil moist avg over last moyhr hrs  ;VS=row                    ;VB=p0        ;MIN=0')
   if (nclass == 26) &
        call gesdict(n, nk, vegfen,     'VN=vegfen       ;ON=2U  ;VD=vegetation fractions (E)                       ;VS=row*26                 ;VB=e1;IN=VF  ;MIN=0')
   if (icelac) &
        call gesdict(n, nk, icelinen,   'VN=icelinen     ;ON=ICLE;VD=ice line (E)                                   ;VS=row                    ;VB=e1;IN=ICEL;')

   IF_ISBA: if (schmsol == 'ISBA') then
      call gesdict(n, nk, acoef,        'VN=acoef        ;ON=1I  ;VD=a coef. in wgeq                                ;VS=row                    ;VB=p0')
      call gesdict(n, nk, alveg,        'VN=alveg        ;ON=AX  ;VD=visible canopy albedo                          ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, bcoef,        'VN=bcoef        ;ON=1G  ;VD=slope of retention curve                       ;VS=row                    ;VB=p0')
      call gesdict(n, nk, c1sat,        'VN=c1sat        ;ON=3I  ;VD=c1 coef. at saturation                         ;VS=row                    ;VB=p0')
      call gesdict(n, nk, c2ref,        'VN=c2ref        ;ON=4I  ;VD=reference value of c2                          ;VS=row                    ;VB=p0')
      call gesdict(n, nk, c3ref,        'VN=c3ref        ;ON=5I  ;VD=drainage coef. to deeper soil                  ;VS=row                    ;VB=p0')
      call gesdict(n, nk, clay,         'VN=clay         ;ON=J2  ;VD=percentage of clay in soil                     ;VS=row*3                  ;VB=p0        ;MIN=0')
      call gesdict(n, nk, clayen,       'VN=clayen       ;ON=2H  ;VD=perc. of clay in soil (E)                      ;VS=row*3                  ;VB=e1;IN=J2  ;MIN=0')
      call gesdict(n, nk, cveg,         'VN=cveg         ;ON=CV  ;VD=thermal coefficient for canopy                 ;VS=row                    ;VB=p0')
      call gesdict(n, nk, drain,        'VN=drain        ;ON=DR  ;VD=water drainage at bottom of soil layer         ;VS=row                    ;VB=p0')
      call gesdict(n, nk, drainaf,      'VN=drainaf      ;ON=O1  ;VD=accum. of base drainage                        ;VS=row                    ;VB=p0')
      call gesdict(n, nk, eflux,        'VN=eflux        ;ON=4F  ;VD=specific hum. flux (=-alfaq)                   ;VS=row                    ;VB=v0')
      call gesdict(n, nk, gamveg,       'VN=gamveg       ;ON=GG  ;VD=stomatal resistance parameter                  ;VS=row                    ;VB=p0')
      call gesdict(n, nk, husurf,       'VN=husurf       ;ON=FH  ;VD=spec. humid. of the surface                    ;VS=row                    ;VB=v0        ;MIN=0')
      call gesdict(n, nk, hv,           'VN=hv           ;ON=HV  ;VD=relative humidity of veg. canopy               ;VS=row                    ;VB=v0        ;MIN=0')
      call gesdict(n, nk, isoil,        'VN=isoil        ;ON=I2  ;VD=soil volumetric ice contents                   ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, isoilen,      'VN=isoilen      ;ON=2Q  ;VD=frozen water in soil (E)                       ;VS=row                    ;VB=e1;IN=I2  ;MIN=0')
      call gesdict(n, nk, lai,          'VN=lai          ;ON=J4  ;VD=leaf area index                                ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, leg,          'VN=leg          ;ON=L2  ;VD=latent heat flux over bare grnd                ;VS=row                    ;VB=v0')
      call gesdict(n, nk, legaf,        'VN=legaf        ;ON=O5  ;VD=accum. of bare ground LE flux                  ;VS=row                    ;VB=p0')
      call gesdict(n, nk, ler,          'VN=ler          ;ON=LR  ;VD=latent heat flux from leaves                   ;VS=row                    ;VB=v0')
      call gesdict(n, nk, leraf,        'VN=leraf        ;ON=O6  ;VD=accum. of direct veg LE flux                   ;VS=row                    ;VB=p0')
      call gesdict(n, nk, les,          'VN=les          ;ON=LS  ;VD=latent heat flux over snow                     ;VS=row                    ;VB=v0')
      call gesdict(n, nk, lesaf,        'VN=lesaf        ;ON=O7  ;VD=accum. of sublimation from snow                ;VS=row                    ;VB=p0')
      call gesdict(n, nk, letr,         'VN=letr         ;ON=LT  ;VD=latent heat of evapotransp.                    ;VS=row                    ;VB=v0')
      call gesdict(n, nk, letraf,       'VN=letraf       ;ON=O8  ;VD=accum. of veg. transpiration                   ;VS=row                    ;VB=p0')
      call gesdict(n, nk, lev,          'VN=lev          ;ON=LV  ;VD=latent heat flux over vegetation               ;VS=row                    ;VB=v0')
      call gesdict(n, nk, levaf,        'VN=levaf        ;ON=O9  ;VD=accum. of evaporation from veg.                ;VS=row                    ;VB=p0')
      call gesdict(n, nk, melts,        'VN=melts        ;ON=MLTS;VD=accum. snow melting (kg/m2)                    ;VS=row                    ;VB=p0')
      call gesdict(n, nk, meltsr,       'VN=meltsr       ;ON=MLTR;VD=accum. snow melting due to rain (kg/m2)        ;VS=row                    ;VB=p0')
      call gesdict(n, nk, overfl,       'VN=overfl       ;ON=RO  ;VD=overland runoff                                ;VS=row                    ;VB=v0')
      call gesdict(n, nk, overflaf,     'VN=overflaf     ;ON=N0  ;VD=accum. of surface runoff                       ;VS=row                    ;VB=p0')
      call gesdict(n, nk, pcoef,        'VN=pcoef        ;ON=7I  ;VD=p coef. in wgeq                                ;VS=row                    ;VB=p0')
      call gesdict(n, nk, psn,          'VN=psn          ;ON=5P  ;VD=fraction of the grid covered by snow           ;VS=row                    ;VB=v0        ;MIN=0')
      call gesdict(n, nk, psng,         'VN=psng         ;ON=3P  ;VD=fraction of bare ground covered by snow        ;VS=row                    ;VB=v0        ;MIN=0')
      call gesdict(n, nk, psnv,         'VN=psnv         ;ON=4P  ;VD=fraction of vegetation covered by snow         ;VS=row                    ;VB=v0        ;MIN=0')
      call gesdict(n, nk, resa,         'VN=resa         ;ON=RD  ;VD=aerodynamic resistance                         ;VS=row                    ;VB=p0')
      call gesdict(n, nk, rgl,          'VN=rgl          ;ON=RG  ;VD=parameter stomatal resistance                  ;VS=row                    ;VB=p0')
      call gesdict(n, nk, rnet_s,       'VN=rnet_s       ;ON=NR  ;VD=net radiation (soil only)                      ;VS=row                    ;VB=v0')
      call gesdict(n, nk, rootdp,       'VN=rootdp       ;ON=D2  ;VD=rooting soil depth                             ;VS=row                    ;VB=p0')
      call gesdict(n, nk, rst,          'VN=rst          ;ON=R1  ;VD=stomatal resistance                            ;VS=row                    ;VB=v0')
      call gesdict(n, nk, sand,         'VN=sand         ;ON=J1  ;VD=percentage of sand in soil                     ;VS=row*3                  ;VB=p0        ;MIN=0')
      call gesdict(n, nk, sanden,       'VN=sanden       ;ON=2G  ;VD=perc. of sand in soil (E)                      ;VS=row*3                  ;VB=e1;IN=J1  ;MIN=0')
      if (snoalb_anl) then
         call gesdict(n, nk, snoalen,      'VN=snoalen      ;ON=5H  ;VD=snow albedo (E)                                ;VS=row                    ;VB=e1;IN=I6  ;MIN=0')
      else
         call gesdict(n, nk, snoagen,      'VN=snoagen      ;ON=3H  ;VD=age of snow (E)                                ;VS=row                    ;VB=e1;IN=XA  ;MIN=0')
      endif
      call gesdict(n, nk, snoal,        'VN=snoal        ;ON=I6  ;VD=albedo of snow                                 ;VS=row@'//nm//'           ;VB=p0        ;MIN=0')
      call gesdict(n, nk, snoma,        'VN=snoma        ;ON=I5  ;VD=snow mass                                      ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, snoro,        'VN=snoro        ;ON=7S  ;VD=relative snow density                          ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, snoroen,      'VN=snoroen      ;ON=4H  ;VD= rel. density of snow (E)                      ;VS=row                    ;VB=e1;IN=DN  ;MIN=0')
      call gesdict(n, nk, stomr,        'VN=stomr        ;ON=RS  ;VD=minimum stomatal resistance                    ;VS=row                    ;VB=p0')
      call gesdict(n, nk, tsoil,        'VN=tsoil        ;ON=I0  ;VD=surface and soil temperatures                  ;VS=row*2                  ;VB=p0')
      call gesdict(n, nk, tsoilen,      'VN=tsoilen      ;ON=9A  ;VD=surface temperature (E)                        ;VS=row*2                  ;VB=e1;IN=I0  ;')
      call gesdict(n, nk, vegf,         'VN=vegf         ;ON=2V  ;VD=vegetation fractions                           ;VS=row*26                 ;VB=p0        ;MIN=0')
      call gesdict(n, nk, vegfrac,      'VN=vegfrac      ;ON=K1  ;VD=vegetation fraction                            ;VS=row*5                  ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wfc,          'VN=wfc          ;ON=J5  ;VD=vol. water content at field cap.               ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wflux,        'VN=wflux        ;ON=M8  ;VD=water flux from surface to atm.                ;VS=row                    ;VB=v0')
      call gesdict(n, nk, wfluxaf,      'VN=wfluxaf      ;ON=N7  ;VD=acc. of soil surface upward water flux         ;VS=row                    ;VB=p0')
      call gesdict(n, nk, wsat,         'VN=wsat         ;ON=J6  ;VD=vol. water content at saturation               ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wsnow,        'VN=wsnow        ;ON=I4  ;VD=water in the snow pack                         ;VS=row@'//nm//'           ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wsnowen,      'VN=wsnowen      ;ON=2R  ;VD=water in the snow pack (E)                     ;VS=row                    ;VB=e1;IN=I4  ;MIN=0')
      call gesdict(n, nk, wsoil,        'VN=wsoil        ;ON=I1  ;VD=soil volumetric water contents                 ;VS=row*2                  ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wsoilen,      'VN=wsoilen      ;ON=5A  ;VD=soil vol. water content (E)                    ;VS=row*2                  ;VB=e1;IN=I1  ;MIN=0')
      call gesdict(n, nk, wveg,         'VN=wveg         ;ON=I3  ;VD=water retained on the vegetation               ;VS=row                    ;VB=p0        ;MIN=0')
      call gesdict(n, nk, wvegen,       'VN=wvegen       ;ON=2L  ;VD=water retained on the vegetation (E)           ;VS=row                    ;VB=e1;IN=I3  ;MIN=0')
      call gesdict(n, nk, wwilt,        'VN=wwilt        ;ON=J7  ;VD=vol. water cont. at wilting pt.                ;VS=row                    ;VB=p0        ;MIN=0')
   endif IF_ISBA

   if (schmurb /= 'TEB') return

   call gesdict(n, nk, alb_road,     'VN=alb_road     ;ON=ALRD;VD=road albedo                                    ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, alb_roaden,   'VN=alb_roaden   ;ON=TB9 ;VD=road albedo (E)                                ;VS=row   ;VB=e1; IN=ALRD;MIN=0')
   call gesdict(n, nk, alb_roof,     'VN=alb_roof     ;ON=ALRF;VD=roof albedo                                    ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, alb_roofen,   'VN=alb_roofen   ;ON=TB10;VD=roof albedo (E)                                ;VS=row   ;VB=e1; IN=ALRF;MIN=0')
   call gesdict(n, nk, alb_wall,     'VN=alb_wall     ;ON=ALWL;VD=wall albedo                                    ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, alb_wallen,   'VN=alb_wallen   ;ON=TB11;VD=wall albedo (E)                                ;VS=row   ;VB=e1; IN=ALWL;MIN=0')
   call gesdict(n, nk, azim,         'VN=azim         ;ON=AZIM;VD=solar azimuthal angle                          ;VS=row   ;VB=p0')
   call gesdict(n, nk, bld,          'VN=bld          ;ON=BLDF;VD=building fraction                              ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, blden,        'VN=blden        ;ON=TB2 ;VD=building fraction (E)                          ;VS=row   ;VB=e1; IN=BLDF;MIN=0')
   call gesdict(n, nk, bld_height,   'VN=bld_height   ;ON=BLDH;VD=building height                                ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, bld_heighten, 'VN=bld_heighten ;ON=TB3 ;VD=building height (E)                            ;VS=row   ;VB=e1; IN=BLDH;MIN=0')
   call gesdict(n, nk, can_hw_ratio, 'VN=can_hw_ratio ;ON=ASPC;VD=aspect ratio of the street                     ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, d_road,       'VN=d_road       ;ON=DPRD;VD=depth of the road layers                       ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, d_roaden,     'VN=d_roaden     ;ON=TB12;VD=depth of the road layers (E)                   ;VS=row*3 ;VB=e1; IN=DPRD;')
   call gesdict(n, nk, d_roof,       'VN=d_roof       ;ON=DPRF;VD=depth of the roof layers                       ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, d_roofen,     'VN=d_roofen     ;ON=TB13;VD=depth of the roof layers (E)                   ;VS=row*3 ;VB=e1; IN=DPRF;')
   call gesdict(n, nk, d_wall,       'VN=d_wall       ;ON=DPWL;VD=depth of the wall layers                       ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, d_wallen,     'VN=d_wallen     ;ON=TB14;VD=depth of the wall layers (E)                   ;VS=row*3 ;VB=e1; IN=DPWL;')
   call gesdict(n, nk, emis_road,    'VN=emis_road    ;ON=EMRD;VD=road emissivity                                ;VS=row   ;VB=p0')
   call gesdict(n, nk, emis_roaden,  'VN=emis_roaden  ;ON=TB15;VD=road emissivity (E)                            ;VS=row   ;VB=e1; IN=EMRD;')
   call gesdict(n, nk, emis_roof,    'VN=emis_roof    ;ON=EMRF;VD=roof emissivity                                ;VS=row   ;VB=p0')
   call gesdict(n, nk, emis_roofen,  'VN=emis_roofen  ;ON=TB16;VD=roof emissivity (E)                            ;VS=row   ;VB=e1; IN=EMRF;')
   call gesdict(n, nk, emis_wall,    'VN=emis_wall    ;ON=EMWL;VD=wall emissivity                                ;VS=row   ;VB=p0')
   call gesdict(n, nk, emis_wallen,  'VN=emis_wallen  ;ON=TB17;VD=wall emissivity (E)                            ;VS=row   ;VB=e1; IN=EMWL;')
   call gesdict(n, nk, g_road,       'VN=g_road       ;ON=QGRD;VD=storage heat flux for road                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, g_roof,       'VN=g_roof       ;ON=QGRF;VD=storage heat flux for roof                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, g_town,       'VN=g_town       ;ON=QGTW;VD=storage heat flux for town                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, g_wall,       'VN=g_wall       ;ON=QGWL;VD=storage heat flux for wall                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_industry,   'VN=h_industry   ;ON=QHIN;VD=sensible heat flux from industry               ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_industryen, 'VN=h_industryen ;ON=TB18;VD=sensible heat flux from industry (E)           ;VS=row   ;VB=e1; IN=QHIN;')
   call gesdict(n, nk, h_road,       'VN=h_road       ;ON=QHRD;VD=sensible heat flux over road                   ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_roof,       'VN=h_roof       ;ON=QHRF;VD=sensible heat flux over roof                   ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_town,       'VN=h_town       ;ON=QHTW;VD=sensible heat flux over town                   ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_traffic,    'VN=h_traffic    ;ON=QHTR;VD=sensible heat flux from traffic                ;VS=row   ;VB=p0')
   call gesdict(n, nk, h_trafficen,  'VN=h_trafficen  ;ON=TB28;VD=sensible heat flux from traffic (E)            ;VS=row   ;VB=e1; IN=QHTR;')
   call gesdict(n, nk, h_wall,       'VN=h_wall       ;ON=QHWL;VD=sensible heat flux over wall                   ;VS=row   ;VB=p0')
   call gesdict(n, nk, hc_road,      'VN=hc_road      ;ON=HCRD;VD=road heat capacities                           ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, hc_roaden,    'VN=hc_roaden    ;ON=TB19;VD=road heat capacities (E)                       ;VS=row*3 ;VB=e1; IN=HCRD;')
   call gesdict(n, nk, hc_roof,      'VN=hc_roof      ;ON=HCRF;VD=roof heat capacities                           ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, hc_roofen,    'VN=hc_roofen    ;ON=TB20;VD=roof heat capacities (E)                       ;VS=row*3 ;VB=e1; IN=HCRF;')
   call gesdict(n, nk, hc_wall,      'VN=hc_wall      ;ON=HCWL;VD=wall heat capacities                           ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, hc_wallen,    'VN=hc_wallen    ;ON=TB21;VD=wall heat capacities (E)                       ;VS=row*3 ;VB=e1; IN=HCWL;')
   call gesdict(n, nk, le_industry,  'VN=le_industry  ;ON=QEIN;VD=latent heat flux from industry                 ;VS=row   ;VB=p0')
   call gesdict(n, nk, le_industryen,'VN=le_industryen;ON=TB27;VD=latent heat flux from industry (E)             ;VS=row   ;VB=e1; IN=QEIN;')
   call gesdict(n, nk, le_road,      'VN=le_road      ;ON=QERD;VD=latent heat flux over road                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, le_roof,      'VN=le_roof      ;ON=QERF;VD=latent heat flux over roof                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, le_town,      'VN=le_town      ;ON=QETW;VD=latent heat flux over town                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, le_traffic,   'VN=le_traffic   ;ON=QETR;VD=latent heat flux from traffic                  ;VS=row   ;VB=p0')
   call gesdict(n, nk, le_trafficen, 'VN=le_trafficen ;ON=TB26;VD=latent heat flux from traffic (E)              ;VS=row   ;VB=e1; IN=QETR;')
   call gesdict(n, nk, le_wall,      'VN=le_wall      ;ON=QEWL;VD=latent heat flux over wall                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, nat,          'VN=nat          ;ON=NATF;VD=natural surface fraction in urban area         ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, naten,        'VN=naten        ;ON=TB1 ;VD=natural surface fraction in urban area (E)     ;VS=row   ;VB=e1; IN=NATF;MIN=0')
   call gesdict(n, nk, pav,          'VN=pav          ;ON=PAVF;VD=impervious fraction (road)                     ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, paven,        'VN=paven        ;ON=TB4 ;VD=impervious fraction (road) (E)                 ;VS=row   ;VB=e1; IN=PAVF;MIN=0')
   call gesdict(n, nk, q_canyon,     'VN=q_canyon     ;ON=QCAN;VD=specific humi inside the canyon                ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, q_canyonen,   'VN=q_canyonen   ;ON=2XEN; VD=canyon air humi (E)                           ;VS=row   ;VB=e1 ;IN=QCAN;MIN=0')
   call gesdict(n, nk, rn_road,      'VN=rn_road      ;ON=RNRD;VD=Net radiation over road                        ;VS=row   ;VB=p0')
   call gesdict(n, nk, rn_roof,      'VN=rn_roof      ;ON=RNRF;VD=Net radiation over roof                        ;VS=row   ;VB=p0')
   call gesdict(n, nk, rn_town,      'VN=rn_town      ;ON=RNTW;VD=Net radiation over town                        ;VS=row   ;VB=p0')
   call gesdict(n, nk, rn_wall,      'VN=rn_wall      ;ON=RNWL;VD=Net radiation over wall                        ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroad_alb,    'VN=sroad_alb    ;ON=SARD;VD=snow albedo for roads                          ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroad_alben,  'VN=sroad_alben  ;ON=9ZEN; VD=snow albedo for roads (E)                     ;VS=row   ;VB=e1 ;IN=SARD;MIN=0')
   call gesdict(n, nk, sroad_emis,   'VN=sroad_emis   ;ON=SERD;VD=snow emissivity for roads                      ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroad_emisen, 'VN=sroad_emisen ;ON=1MEN; VD=snow emmissivity for roads (E)                ;VS=row   ;VB=e1 ;IN=SERD;MIN=0')
   call gesdict(n, nk, sroad_rho,    'VN=sroad_rho    ;ON=SDRD;VD=snow density for roads                         ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroad_rhoen,  'VN=sroad_rhoen  ;ON=8ZEN; VD=snow density for roads (E)                    ;VS=row   ;VB=e1 ;IN=SDRD;MIN=0')
   call gesdict(n, nk, sroad_scheme, 'VN=sroad_scheme ;ON=SCRD;VD=snow scheme for roads                          ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroad_t,      'VN=sroad_t      ;ON=STRD;VD=snow temperature for roads                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroad_ten,    'VN=sroad_ten    ;ON=7ZEN; VD=snow temp for roads (E)                       ;VS=row   ;VB=e1 ;IN=STRD;')
   call gesdict(n, nk, sroad_ts,     'VN=sroad_ts     ;ON=SSRD;VD=snow surf temperature for roads                ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroad_tsen,   'VN=sroad_tsen   ;ON=2MEN; VD=Snow surface temp for roads (E)               ;VS=row   ;VB=e1 ;IN=SSRD;')
   call gesdict(n, nk, sroad_wsnow,  'VN=sroad_wsnow  ;ON=SWRD;VD=water and snow content for roads               ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroad_wsnowen,'VN=sroad_wsnowen;ON=6ZEN; VD=water/snow content for roads (E)              ;VS=row   ;VB=e1 ;IN=SWRD;')
   call gesdict(n, nk, sroof_alb,    'VN=sroof_alb    ;ON=SARF;VD=snow albedo for roofs                          ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroof_alben,  'VN=sroof_alben  ;ON=1ZEN; VD=snow albedo for roofs (E)                     ;VS=row   ;VB=e1 ;IN=SARF;MIN=0')
   call gesdict(n, nk, sroof_emis,   'VN=sroof_emis   ;ON=SERF;VD=snow emissivity for roofs                      ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroof_emisen, 'VN=sroof_emisen ;ON=2ZEN; VD=snow emmissivity for roofs (E)                ;VS=row   ;VB=e1 ;IN=SERF;MIN=0')
   call gesdict(n, nk, sroof_rho,    'VN=sroof_rho    ;ON=SDRF;VD=snow density for roofs                         ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, sroof_rhoen,  'VN=sroof_rhoen  ;ON=9YEN; VD=snow density for roofs (E)                    ;VS=row   ;VB=e1 ;IN=SDRF;MIN=0')
   call gesdict(n, nk, sroof_scheme, 'VN=sroof_scheme ;ON=SCRF;VD=snow scheme for roofs                          ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroof_t,      'VN=sroof_t      ;ON=STRF;VD=snow temperature for roofs                     ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroof_ten,    'VN=sroof_ten    ;ON=8YEN; VD=snow temp for roofs (E)                       ;VS=row   ;VB=e1 ;IN=STRF;')
   call gesdict(n, nk, sroof_ts,     'VN=sroof_ts     ;ON=SSRF;VD=snow surf temperature for roofs                ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroof_tsen,   'VN=sroof_tsen   ;ON=3ZEN; VD=Snow surface temp for roofs (E)               ;VS=row   ;VB=e1 ;IN=SSRF;')
   call gesdict(n, nk, sroof_wsnow,  'VN=sroof_wsnow  ;ON=SWRF;VD=water and snow content for roofs               ;VS=row   ;VB=p0')
   call gesdict(n, nk, sroof_wsnowen,'VN=sroof_wsnowen;ON=7YEN; VD=water/snow content for roofs (E)              ;VS=row   ;VB=e1 ;IN=SWRF;')
   call gesdict(n, nk, svf_road,     'VN=svf_road     ;ON=SVRD;VD=road sky-view factor                           ;VS=row   ;VB=p0')
   call gesdict(n, nk, svf_wall,     'VN=svf_wall     ;ON=SVWL;VD=wall sky-view factor                           ;VS=row   ;VB=p0')
   call gesdict(n, nk, t_canyon,     'VN=t_canyon     ;ON=TCAN;VD=air temperature inside the canyon              ;VS=row   ;VB=p0')
   call gesdict(n, nk, t_canyonen,   'VN=t_canyonen   ;ON=1XEN; VD=canyon air temp (E)                           ;VS=row   ;VB=e1 ;IN=TCAN;')
   call gesdict(n, nk, t_road,       'VN=t_road       ;ON=TLRD;VD=temperatures of road layers                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, t_roaden,     'VN=t_roaden     ;ON=4XEN; VD=road temperatures (E)                         ;VS=row*3 ;VB=e1 ;IN=TLRD;')
   call gesdict(n, nk, t_roof,       'VN=t_roof       ;ON=TLRF;VD=temperatures of roof layers                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, t_roofen,     'VN=t_roofen     ;ON=3XEN; VD=roof temperatures (E)                         ;VS=row*3 ;VB=e1 ;IN=TLRF;')
   call gesdict(n, nk, t_wall,       'VN=t_wall       ;ON=TLWL;VD=temperatures of wall layers                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, t_wallen,     'VN=t_wallen     ;ON=5XEN; VD=wall temperatures (E)                         ;VS=row*3 ;VB=e1 ;IN=TLWL;')
   call gesdict(n, nk, tc_road,      'VN=tc_road      ;ON=TCRD;VD=road thermal condcutivities                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, tc_roaden,    'VN=tc_roaden    ;ON=TB22;VD=road thermal condcutivities (E)                ;VS=row*3 ;VB=e1; IN=TCRD;') 
   call gesdict(n, nk, tc_roof,      'VN=tc_roof      ;ON=TCRF;VD=roof thermal condcutivities                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, tc_roofen,    'VN=tc_roofen    ;ON=TB23;VD=roof thermal condcutivities (E)                ;VS=row*3 ;VB=e1; IN=TCRF;')
   call gesdict(n, nk, tc_wall,      'VN=tc_wall      ;ON=TCWL;VD=wall thermal condcutivities                    ;VS=row*3 ;VB=p0')
   call gesdict(n, nk, tc_wallen,    'VN=tc_wallen    ;ON=TB24;VD=wall thermal condcutivities (E)                ;VS=row*3 ;VB=e1; IN=TCWL;')
   call gesdict(n, nk, ti_bld,       'VN=ti_bld       ;ON=TBLD;VD=internal building temperature                  ;VS=row   ;VB=p0')
   call gesdict(n, nk, ti_blden,     'VN=ti_blden     ;ON=6QEN; VD=bld internal temp (E)                         ;VS=row    ;VB=e1 ;IN=TBLD;')
   call gesdict(n, nk, ti_road,      'VN=ti_road      ;ON=TIRD;VD=internal road temperature                      ;VS=row   ;VB=p0')
   call gesdict(n, nk, ti_roaden,    'VN=ti_roaden    ;ON=5QEN; VD=road internal temp (E)                        ;VS=row   ;VB=e1 ;IN=TIRD;')
   call gesdict(n, nk, tsun,         'VN=tsun         ;ON=TSUN;VD=solar time (s)                                 ;VS=row   ;VB=p0')
   call gesdict(n, nk, u_canyon,     'VN=u_canyon     ;ON=UCAN;VD=wind in canyon                                 ;VS=row   ;VB=p0')
   call gesdict(n, nk, wall_o_hor,   'VN=wall_o_hor   ;ON=WHOR;VD=ratio vertical per horizontal surf             ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, wall_o_horen, 'VN=wall_o_horen ;ON=TB5 ;VD=ratio vertical per horizontal surf (E)         ;VS=row   ;VB=e1; IN=WHOR;MIN=0')
   call gesdict(n, nk, ws_road,      'VN=ws_road      ;ON=WSRD;VD=water content of road reservoir                ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, ws_roaden,    'VN=ws_roaden    ;ON=4QEN; VD=road water reservoir (E)                      ;VS=row   ;VB=e1 ;IN=WSRD;MIN=0')
   call gesdict(n, nk, ws_roof,      'VN=ws_roof      ;ON=WSRF;VD=water content of roof reservoir                ;VS=row   ;VB=p0         ;MIN=0')
   call gesdict(n, nk, ws_roofen,    'VN=ws_roofen    ;ON=3QEN; VD=roof water reservoir (E)                      ;VS=row   ;VB=e1 ;IN=WSRF;MIN=0')
   call gesdict(n, nk, z0_road,      'VN=z0_road      ;ON=Z0RD;VD=aerodyn roughness length for road              ;VS=row   ;VB=p0')
   call gesdict(n, nk, z0_roaden,    'VN=z0_roaden    ;ON=TB8 ;VD=aerodyn roughness length for road (E)          ;VS=row   ;VB=e1; IN=Z0RD;')
   call gesdict(n, nk, z0_roof,      'VN=z0_roof      ;ON=Z0RF;VD=aerodyn roughness length for roof              ;VS=row   ;VB=p0')
   call gesdict(n, nk, z0_roofen,    'VN=z0_roofen    ;ON=TB7 ;VD=aerodyn roughness length for roof (E)          ;VS=row   ;VB=e1; IN=Z0RF;')
   call gesdict(n, nk, z0_town,      'VN=z0_town      ;ON=Z0TW;VD=aerodyn roughness length for town              ;VS=row   ;VB=p0')
   call gesdict(n, nk, z0_townen,    'VN=z0_townen    ;ON=TB6 ;VD=aerodyn roughness length for town (E)          ;VS=row   ;VB=e1; IN=Z0TW;')
   call gesdict(n, nk, zenith,       'VN=zenith       ;ON=ZENI;VD=solar zenith angle                             ;VS=row   ;VB=p0') !#TODO: del, not used

   !---------------------------------------------------------------------
   return
end subroutine sfc_businit
