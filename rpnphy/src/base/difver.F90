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
subroutine difver7 (db, dsiz, f, fsiz, v, vsiz, seloc, tau, &
     kount, trnch, n, nk, stack)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object to perform the implicit vertical diffusion
   !@Arguments
! 002      A. Zadra (Oct 2015) - add land-water mask (MG) to input
!                                list of baktotq4
   !          - Input/Output -
   ! db       dynamic bus
   ! f        field for permanent physics variables
   ! v        volatile bus
   ! dsiz     dimension of db
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! g        physics work space
   ! espg     dimension of g
   !          - output -
   ! tu       u  tendency
   ! tv       v  tendency
   ! tt       t  tendency
   ! tq       q  tendency
   ! tl       l tendency
   ! t        temperature
   ! uu       x component of the wind
   ! vv       y component of the wind
   ! q        specific humidity
   ! ql       liquid water
   ! ps       surface pressure
   ! tm       temperature at time t-dt
   !          - input -
   ! sg       sigma levels
   ! seloc    staggered sigma levels
   ! tau      timestep * factdt * facdifv
   !          see common block "options"
   ! kount    timestep number
   ! trnch    row number
   ! n        horizontal dimension
   ! nk       vertical dimension
   ! stack    task number

   integer dsiz,fsiz,vsiz,kount,trnch,n,nk,stack,ierror
   real, target :: db(dsiz),f(fsiz),v(vsiz)
   real, target :: seloc(n,nk)
   real tau

   !@Author J. Cote (Oct 1984)
   !*@/

   external difuvdfj
   external difuvdf, atmflux1
   external baktotq3, ficemxp

   integer j,k,type
   real dq,rhortvsg,mrhocmu
   real tplusnk,qplusnk,uplusnk,vplusnk
   real maximum, minimum, rsg

   include "thermoconsts.inc"
   include "surface.cdk"

   real kmsg(n,nk), ktsg(n,nk), bmsg(n), btsg(n), rgam0(n,nk)


   real a(n),c(n,nk),d(n,nk),r(n,nk),r1(n,nk),r2(n,nk),zero(n,nk)
   real aq(n), bq(n), se(nk), sig(nk)
   real uflux(n,nk+1), vflux(n,nk+1), tflux(n,nk+1), qflux(n,nk+1)
   real gam0(n,nk+1), fsloflx(n), qclocal(n,nk  ), ficelocal(n,nk  )

   !     Pointeurs pour champs deja definis dans les bus
   real, pointer, dimension(:)   :: ps
   real, pointer, dimension(:)   :: zalfat, zalfaq, zbm, zbt_ag, &
        zepstfn, zfc_ag, zfdsi, zfdss, &
        zfl, zfnsi, zfq, zfv_ag, &
        zmg, zqsurf_ag, ztsurf, ztsrad, &
        zustress, zvstress
   real, pointer, dimension(:,:) :: tu, tv, tw, tt, tq, tl, uu, vv, w, &
        t, q, ql, sg, tm, &
        sigef, sigex
   real, pointer, dimension(:,:) :: zgq, zgql, zgte, zkm, zkt, zqtbl, ztve
   
   !# fonction-formule
   integer jk
   jk(j,k) = (k-1)*n + j - 1

   !---------------------------------------------------------------------

   if (any(fluvert == (/'SURFACE', 'NIL    '/)))  return

!     Equivalences avec champs  dans les bus
      ps       (1:n) => f( pmoins:)
      zalfaq   (1:n) => v( alfaq:)
      zalfat   (1:n) => v( alfat:)
      zbm      (1:n) => v( bm:)
      zbt_ag   (1:n) => v( bt+(indx_agrege-1)*n:)
      zepstfn  (1:n) => f( epstfn:)
      zfc_ag   (1:n) => v( fc+(indx_agrege-1)*n:)
      zfdsi    (1:n) => f( fdsi:)
      zfdss    (1:n) => f( fdss:)
      zfl      (1:n) => v( fl:)
      zfnsi    (1:n) => v( fnsi:)
      zfq      (1:n) => f( fq:)
      zfv_ag   (1:n) => v( fv+(indx_agrege-1)*n:)
      zmg      (1:n) => f( mg:)
      zqsurf_ag(1:n) => f( qsurf+(indx_agrege-1)*n:)
      ztsurf   (1:n) => f( tsurf:)
      ztsrad   (1:n) => f( tsrad:)
      zustress (1:n) => v( ustress:)
      zvstress (1:n) => v( vstress:)

       q   (1:n,1:nk) => db( huplus:)
      ql   (1:n,1:nk) => db( qcplus:)
      sg   (1:n,1:nk) => db( sigm:)
       t   (1:n,1:nk) => db( tplus:)
      tm   (1:n,1:nk) => db( tmoins:)
      tu   (1:n,1:nk) =>  v( udifv:)
      tv   (1:n,1:nk) =>  v( vdifv:)
      tt   (1:n,1:nk) =>  v( tdifv:)
      tq   (1:n,1:nk) =>  v( qdifv:)
      uu   (1:n,1:nk) => db( uplus:)
      vv   (1:n,1:nk) => db( vplus:)
       w   (1:n,1:nk) => db( wplus:)
      zgq  (1:n,1:nk) =>  v( gq:)
      zgql (1:n,1:nk) =>  v( gql:)
      zgte (1:n,1:nk) =>  v( gte:)
      zkm  (1:n,1:nk) =>  v( km:)
      zkt  (1:n,1:nk) =>  v( kt:)
      zqtbl(1:n,1:nk) =>  f( qtbl:)
      ztve (1:n,1:nk) =>  v( tve:)

      if(diffuw) tw(1:n,1:nk) => v( wdifv:)
      if (ldifv > 0) then
         tl(1:n,1:nk) => v( ldifv:)
      else
         nullify(tl)
      endif
      if(db(sigt)>0) then
         sigef(1:n,1:nk) => db( sigt:)
         sigex(1:n,1:nk) => db( sigt:)
      else
         sigef(1:,1:)    => seloc(1:n,1:nk)
         sigex(1:n,1:nk) => db( sigm:)
      endif      

!     normalization factors for vertical diffusion in sigma coordinates

      rsg = (grav/rgasd)
      do k=1,nk
!vdir nodep
         do j=1,n
            gam0(j,k) = rsg*seloc(j,k)/ztve(j,k)
            kmsg(j,k) = zkm(j,k)*gam0(j,k)**2
            ktsg(j,k) = zkt(j,k)*gam0(j,k)**2
         end do
      end do
      call vsrec(rgam0,gam0,n*(nk))
      do k=1,nk
!vdir nodep
         do j=1,n
            zgte(j,k) = zgte(j,k)*rgam0(j,k)
            zgq(j,k)  = zgq(j,k)*rgam0(j,k)
            zgql(j,k) = zgql(j,k)*rgam0(j,k)
         end do
      end do

!     Slow start

      do j=1,n
         fsloflx(j) = 1.
      end do

      if (nsloflux.gt.0) then
         do j=1,n
!           Over the continent, we perform a slow start for
!           fluxes "fc" and "fv" until timestep "nsloflux",
!           because of imbalances between analyses of temperature
!           at the ground and just above the surface.
            if (zmg(j)>0.5) then
!              Max is used to avoid division by zero
               fsloflx(j) = (float(kount-1))/max(float(nsloflux),1.)
               if (kount.eq.0) fsloflx(j) = 0.0
            endif
         end do
      endif

!vdir nodep
      do j=1,n
         aq(j)=-rsg/(ztsurf(j)*(1. + delta * zqsurf_ag(j)))
         bmsg(j)   = zbm(j) * aq(j)
         btsg(j)   = zbt_ag(j) * aq(j) * fsloflx(j)
         zalfat(j) = zalfat(j)  * aq(j) * fsloflx(j)
         zalfaq(j) = zalfaq(j)  * aq(j) * fsloflx(j)
      end do

      do k=1,nk+1
      do j=1,n
        gam0(j,k) = 0.0
      end do
      end do

      do k=1,nk
         se (k) = seloc(1,k)
         sig(k) = sg (1,k)
      end do

! Diffusion verticale implicite

      zero = 0.

      type=1

! diffuse u

      call difuvdfj(tu,uu,kmsg,zero,zero,zero,bmsg,sg,seloc,tau, &
                    type,1.,c,d,r,r1,n,n,n,nk)

      call atmflux1(uflux,uu,tu,kmsg,gam0,zero(1,nk), &
                   bmsg,ps,t,q,tau,sg,v(at2e), &
                   seloc,c,d,0,n,nk,trnch)

! diffuse v
!
      call difuvdfj(tv,vv,kmsg,zero,zero,zero,bmsg,sg,seloc,tau, &
                    type,1.,c,d,r,r1,n,n,n,nk)

      call atmflux1(vflux,vv,tv,kmsg,gam0,zero(1,nk), &
                   bmsg,ps,t,q,tau,sg,v(at2e), &
                   seloc,c,d,1,n,nk,trnch)

! diffuse w
!
      if(db(sigt)>0) then
         if (tlift.eq.1) then
            type = 6
         else
            type = 5
         endif
      endif
      if (diffuw) then
      call difuvdfj(tw,w,kmsg,zero,zero,zero,zero,sg,sigef,tau, &
                    type,1.,c,d,r,r1,n,n,n,nk)
      endif

! diffuse moisture

      if(evap) then
         do j=1,n
            aq(j) = zalfaq(j)
            bq(j) = btsg(j)
         end do

      else
!
!     La cle 'evap' est valide pour parametrages clef et simplifies
!     mettre termes de surface a zero
         do j=1,n
           aq(j) = 0.0
           bq(j) = 0.0
         end do

      endif

      if (fluvert == 'MOISTKE') then
!                                       diffuse conservative variable qw
!       qtbl contains the implicit cloud water from previous time step.
        do k=1,nk
        do j=1,n
           qclocal(j,k) = max( 0.0 , zqtbl(j,k))
           q(j,k) = q(j,k) + max( 0.0 , qclocal(j,k) )
        end do
        end do

      endif

      call difuvdfj(tq,q,ktsg,v(gq),zero,aq,bq,sg,sigef,tau, &
                    type,1.,c,d,r,r1,n,n,n,nk)

      call atmflux1(qflux,q,tq,ktsg,gam0, &
                   aq,bq,ps,t,q,tau,sg,v(at2e), &
                   sigef,c,d,2,n,nk,trnch)

! diffuse temperature

      if (chauf) then
         do j=1,n
            aq(j) = zalfat(j)
            bq(j) = btsg(j)
         end do

      else

!     La cle 'chauf' est valide pour parametrages clef et simplifies
!     mettre termes de surface a zero
         do j=1,n
           aq(j) = 0.0
           bq(j) = 0.0
         end do

      endif

      if (fluvert == 'MOISTKE') then
!    iffuse conservative variable thetal

         call ficemxp(ficelocal,r1,r2,tm,n,n,nk)
         do k=1,nk
         do j=1,n
!    copy current t in r2 for later use in baktotq
            r2(j,k) = t(j,k)
            t(j,k) = t(j,k) &
               - ((chlc+ficelocal(j,k)*chlf)/cpd) &
                  *max( 0.0 , qclocal(j,k) )
            t(j,k) = t(j,k) * sigex(j,k)**(-cappa)
         end do
         end do
      endif

      call difuvdfj(tt,t,ktsg,v(gte),zero,aq,bq,sg,sigef,tau, &
                    type,1.,c,d,r,r1,n,n,n,nk)

      if (fluvert == 'MOISTKE') then
!     back to non-conservative variables T and Q
!     and their tendencies

         call baktotq4 (t, q, qclocal, r2, sg, db(sigw), ps, tm, ficelocal, &
                       tt, tq, tl, v(tve), f(qtbl), &
                       f(fnn), f(fbl), f(zn), f(zd), f(mg), &
                       v(at2t),v(at2m),v(at2e),tau, n, n, nk)

      endif

!                           counter-gradient term = -g/cp
!                           because temperature is used in
!                           subroutine difuvdf (instead of
!                           potential temperature).

      do k=1,nk+1
      do j=1,n
        gam0(j,k) = -grav/cpd
      end do
      end do

      call atmflux1(tflux,t,tt,ktsg,gam0, &
                   aq,bq, &
                   ps,t,q,tau,sg,v(at2e),sigef,c,d,3,n,nk,trnch)

! calcul final du bilan de surface

!vdir nodep
      do j = 1, n
         rhortvsg = ps(j)/grav
         mrhocmu  = ps(j)/grav*bmsg(j)
         tplusnk  = t (j,nk)+tau*tt(j,nk)
         qplusnk  = q (j,nk)+tau*tq(j,nk)
         uplusnk  = uu(j,nk)+tau*tu(j,nk)
         vplusnk  = vv(j,nk)+tau*tv(j,nk)

!        Recalculer les flux partout

!        ustress et vstress sont calcules apres diffusion car
!        on utilise toujours une formulation implicite pour
!        la condition aux limites de surface pour les vents.
!        Par contre, la formulation est explicite pour
!        la temperature et l'humidite.
!        A noter que, puisque la formulation est explicite,
!        on agrege les flux fc et fv dans le sous-programme
!        AGREGE; on pourrait aussi les calculer ici en tout
!        temps, mais on ne le fait que pendant le "depart lent".
!        Si on utilisait une formulation implicite pour
!        fc et fv, il faudrait que ces derniers soient
!        toujours calcules ici.

         if (nsloflux.gt.0.and.kount.le.nsloflux) then
         zfc_ag(j) =  cpd * rhortvsg * (zalfat(j)+btsg(j)*tplusnk)
         zfv_ag(j) = chlc * rhortvsg * (zalfaq(j)+btsg(j)*qplusnk)
         endif

         zustress(j) = -mrhocmu*uplusnk
         zvstress(j) = -mrhocmu*vplusnk
         zfq(j)      = -mrhocmu*sqrt(uplusnk**2 + vplusnk**2)

         if (.not.chauf)  zfc_ag(j) = 0.0
         if (.not.evap )  zfv_ag(j) = 0.0

         a(j)     = zfdsi(j)*zepstfn(j)/stefan
         zfnsi(j) = a(j)-zepstfn(j)*ztsrad(j)**4
         zfl(j)   = zfnsi(j) + zfdss(j) - zfv_ag(j) - zfc_ag(j)
      enddo

!     Diagnostics

      call serxst2(tu, 'tu', trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(tv, 'tv', trnch, n, nk, 0.0, 1.0, -1)

      call serxst2(tt, 'tf', trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(tq, 'qf', trnch, n, nk, 0.0, 1.0, -1)
      if (associated(tl)) then
         call serxst2(tl, 'lf', trnch, n, nk, 0.0, 1.0, -1)
      endif

      call serxst2(f(fq), 'fq', trnch, n, 1, 0., 1., -1)

      call serxst2(v(fc+(indx_soil   -1)*n), 'f4', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fc+(indx_glacier-1)*n), 'f5', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fc+(indx_water  -1)*n), 'f6', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fc+(indx_ice    -1)*n), 'f7', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fc+(indx_agrege -1)*n), 'fc', trnch, n, 1, 0., 1., -1)

      call serxst2(v(fv+(indx_soil   -1)*n), 'h4', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fv+(indx_glacier-1)*n), 'h5', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fv+(indx_water  -1)*n), 'h6', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fv+(indx_ice    -1)*n), 'h7', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fv+(indx_agrege -1)*n), 'fv', trnch, n, 1, 0., 1., -1)

      call serxst2(a,       'fi', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fnsi), 'si', trnch, n, 1, 0., 1., -1)
      call serxst2(v(fl),   'fl', trnch, n, 1, 0., 1., -1)

!     diagnostics pour le modele ctm
      if (any(fluvert == (/'MOISTKE', 'CLEF   '/)))  then
         call ctmdiag(db, f, v, dsiz, fsiz, vsiz, n, nk)
         call serxst2(v(ue), 'ue', trnch, n, 1, 0.0, 1.0, -1)
      endif

      call serxst2(v(km), 'km', trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(v(kt), 'kt', trnch, n, nk, 0.0, 1.0, -1)

      return
      end
