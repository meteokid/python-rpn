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

      subroutine newrad5 ( d, sized, f, sizef, v, vsiz, &
                           liqwcin, icewcin, liqwp, icewp, nuage, &
                           tau, kount, &
                           trnch , n , m , nk , istak, itask, &
                           nkp, nkrd, nkprd, inrd)
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>

      integer sized,sizef,kount,trnch, vsiz, &
              n,m,nk,istak,itask,nkp,nkrd,nkprd
      integer inrd(nkprd)
      real, target ::  d(sized), f(sizef), v(vsiz)
      real, target ::  nuage(n*nk)
      real liqwcin(m,nk), icewcin(m,nk)
      real liqwp(m,nk), icewp(m,nk)
      real tau

!Author
!          L. Garand and J. Mailhot RPN  (June 1989)
!
!Revision
! 001      see version 5.5.0 for previous history
!
!Object
!          to execute a more advanced scheme in finding the infrared
!          and solar radiation and calculation of clouds
!
!Arguments
!
!          - Input/Output -
! f        field of permanent physics variables
! sizef    dimension of f
!
!          - Input -
! t        temperature
! q        specific humidity
! ps       surface pressure
! s        sigma levels
! tau      timestep
! satuco   .TRUE. if water/ice phase for saturation
!          .FALSE. if water phase only for saturation
! kount    number of timesteps
! kntrad   frequency of call for infra-red radiation
! TRNCH    index of the vertical plane (NI*NK) for which
!          calculations are to be done.
! n        horizontal dimension
! m        1st dimension of T and Q
! nk       number of layers
! istak    stack number to use
! itask    task number
! nkp      number of layers including ground
! nkrd     number of reduced layers
! nkprd    number of reduced layers including ground
! inrd     list of reduced layers
! reduc    .true. to use inrd and compute on reduced layers
!          .false. to use full layers (inrd not used)
! liqwcin  in-cloud liquid water content
! liqwcin  in-cloud ice    water content
! nuage    cloud fraction
!
!Notes
!          newrad produces:
!          Infra-red rate (ti) of cooling
!          Visible rate (t2) of heating
!          Visible flux to ground (fdss)
!          Infra-red flux to ground (fdsi)
!          Infra-red flux to the top of the atmosphere (ei)
!          Visible flux to the top of the atmosphere (ev)
!          Planetary albedo (ap=ev/incident solar flux)

#include "surface.cdk"
#include "clefcon.cdk"
#include "ozopnt.cdk"
#include "radiation.cdk"
#include "nocld.cdk"

      include "thermoconsts.inc"

      real juliand
      external juliand
!
! Seuil du cos de l'angle solaire a partir duquel on considere
! que le soleil est leve
      real, parameter :: seuil=1.E-5

      real fbas,hzp,julien,press,r0r

! Dimensions des champs necessaires
! pour l'option de reduction des niveaux
      integer i, kk, rednk, rednkp, redm
!

      real v1,v2
      real hz0,hz
      integer j,k,ja,nnk,is, it
      integer ierget
      logical opnua
      integer nncl,iopt
      integer nnkp,nnkp2, ik
      real alb, delp
      real ozpak

!
      real, target, dimension(n*nk)      :: del(n*nk), nef(n*nk)
      real, target, dimension(n*nkp)     :: ss
      real, target, dimension(n*nkp)     :: p0
      real, target, dimension(m*nkrd)    :: t_red, q_red, fn_red
      real, target, dimension(n*nkrd)    :: s_red, p0_red
      real, target, dimension(n*nkprd)   :: ss_red, del_red
!
      logical,      dimension(nk)        :: toit
      real,         dimension(m,nkrd)    :: iwprd, lwprd
      real        , dimension(n*nkp*nkp) :: p1,p2,p14
      real        , dimension(n*nkp)     :: p3,p4,p5,p6,p7,p8,p9,p10,q6,q9,&
                                            q10,q11,q12,q13,q14,q15
      real        , dimension(n)         :: p11,p12,p16,p17,q19,q20,q21,q22,&
                                            q23,q24,q25,q26,q27,q28,q29,q30,&
                                            q31,q32,q33,q34,q35
      real        , dimension(n*nkp*2)   :: p13
      real        , dimension(n*nkp*3)   :: p15
      real        , dimension(n*nkp*6)   :: q1,q2
      real        , dimension(n*nkp*3)   :: q3
      real        , dimension(n*nkp*2)   :: q4,q5
      real        , dimension(n*8)       :: q7,q8
!
      real        , dimension(n)         :: sdz,ipb,pbl,ozotoit,dummy1(n),&
                                            dummy2(n),dummy3(n),dummy4(n)
      real        , dimension(n,nk)      :: asy,opd,dzz,ssa
      real        , dimension(n,nk,5)    :: aer
!
      real, pointer, dimension(:)        :: trd, qrd, srd, p0rd, &
                                            delrd, fnrd, ssrd, ps
!
!     Pointers to busdyn
      real, pointer, dimension(:)   :: t1d, q1d, s1d
      real, pointer, dimension(:,:) :: t, q, s
!
!     Pointers to busper
      real, pointer, dimension(:)   :: zalvis_ag, zcosas, zcosz, zei, zev, &
                                       zev0, zfdsi, zfdss, zfdss0,  &
                                       zflusolis, znt, zvozo
      real, pointer, dimension(:,:) :: zt2, zt20, zti
!     Pointers to busvol
      real, pointer, dimension(:)   :: zap, zcang, ziv
      real, pointer, dimension(:,:) :: ztrad
!
!***********************************************************************
!
#include "solcons.cdk"
!
!     Pointers to busdyn
      t(1:m,1:nk) => d( tmoins:)
      q(1:m,1:nk) => d( humoins:)
      ps          => f( pmoins:)
      s(1:n,1:nk) => d( sigw:)
      t1d         => d( tmoins:)
      q1d         => d( humoins:)
      s1d         => d( sigw:)
!     Pointers to busper
      zalvis_ag(1:n)       => f( alvis+(indx_agrege-1)*n:)
      zcosas   (1:n)       => f( cosas:)
      zcosz    (1:n)       => f( cosz:)
      zei      (1:n)       => f( ei:)
      zev      (1:n)       => f( ev:)
      zev0     (1:n)       => f( ev0:)
      zfdsi    (1:n)       => f( fdsi:)
      zfdss    (1:n)       => f( fdss:)
      zfdss0   (1:n)       => f( fdss0:)
      zflusolis(1:n)       => f( flusolis:)
      znt      (1:n)       => f( nt:)
      zvozo    (1:n)       => f( vozo:)
      zt2      (1:n,1:nkp) => f( t2:)
      zt20     (1:n,1:nkp) => f( t20:)
      zti      (1:n,1:nkp) => f( ti:)
!     Pointers to busvol
      zap      (1:n)       => v( ap:)
      zcang    (1:n)       => v( cang:)
      ziv      (1:n)       => v( iv:)
      ztrad    (1:n,1:nkp) => v( trad:)
!
      call raddel(del,ss,s,n,nk,nkp)

      is = istak
      it = itask
!
      ja = n*(nk-1)
      nkp=nk+1
! nkp est nb de niveaux de flux
      nnkp  = n*nkp
      nnk   = n*nk
      nnkp2 = n*nkp*nkp
!
! date(5)=the hour of the day at the start of the run.
! date(6)=hundreds of a second of the day at the start of the run.
!
      hz0 = date(5) + float(date(6))/360000.0
      hz = amod ( hz0+(float(kount)*tau)/3600. , 24. )
!
!
      nncl=n*npcl
!
! Compute optical parameters for vis and ir code
! includes effective ir cloud amount
! and for vis: aerosols, optical depth, asymetry factor,
! and single scattering albedo
!
! Hauteur de couche limite temporairement mise a 1500 metres
! en attendant qu'elle soit passee a newrad
!
      do 7007 i=1,n
      pbl(i)=1500.
 7007 continue
!
       call cldoptx5 (liqwcin,icewcin,liqwp,icewp,nuage,t,s,ps, &
                      f(dlat),f(mg),f(ml),m,n,nk, &
                      pbl,ipb,dzz,sdz, nef,opd,asy, &
                      f(topthw),f(topthi), &
                      v(ctp),v(ctt), &
                      ssa,aer,satuco,cw_rad,ioptix)
!
! Boucle sur le pas de radiation kntrad
!
      IF (KOUNT == 0 .OR. MOD(KOUNT-1,KNTRAD) == 0) THEN
!
      call radfac3 (p0, ozotoit, s, nkp, nk, npcl, &
                    f(dlat), ps, n, n, nkp, &
                    p2, p3, p4, p5, p6, p7, p8, p10, &
                    p11,nlacl, goz(fozon), goz(clat), &
                    goz(pref))
!
      if( reduc ) then
        if(TS_FLXIR)then
           write(*,"(/,'!!',/,a,/,'!!')")'You cannot use TS_FLXIR and REDUC'
           call qqexit(1)
        endif

        do kk=1,nkrd
          k = inrd(kk)

          do i=1,n
            s_red((kk-1)*n+i) = s(i,k)
            t_red((kk-1)*n+i) = t(i,k)
            q_red((kk-1)*n+i) = q(i,k)
            p0_red((kk-1)*n+i)= p0((k-1)*n+i)
          enddo
        enddo
        srd   => s_red(1:)
        trd   => t_red(1:)
        qrd   => q_red(1:)
        p0rd  => p0_red(1:)
        rednk = nkrd
        rednkp= nkprd
        redm  = n
        call raddel(del_red,ss_red,srd,n,nkrd,nkprd)
        call rdmax(fn_red,nef,p1,inrd,n,nk,nkrd)
        fnrd    => fn_red(1:)
        delrd   => del_red(1:)
        ssrd    => ss_red(1:)

      else

        srd    => d(sigw:)
        trd    => d(tmoins:)
        qrd    => d(humoins:)
        p0rd   => p0(1:)
        fnrd   => nef(1:)
        delrd  => del(1:)
        ssrd   => ss(1:)

        rednk  = nk
        rednkp = nkp
        redm   = m

      endif
!
      call pntg123
!
      iopt=0
      opnua=.true.
!

      call radir8 ( f(ti) , p7 , p5 , fnrd , trd , qrd , srd , &
              f(tsrad),ps,rednkp,rednk,p0rd, &
              rednkp,n,n,redm,ntt,mx,mxx,no3,ncx,nco2, &
              g(g1),g(g2),g(g3),g(th2o),g(tro3), &
              g(yg3), g(bcn),g(dbcn),g(bo3), &
              g(dbo3),g(to3),g(uu),g(tt), &
              p1, p2, iopt, opnua, &
              p3 , p4 , p5 , p6 , p7 , ssrd, &
              p9 , delrd, p11 , p12, &
              f(nhaut), f(nmoy), f(nbas), &
              p13, p14, p15, p16, p17, &
              ozpak, ozpak, &
              s,ss,del,nk,nkp)
!
      if( fomic ) then
         call fomichev( f(ti), t,p0,s,ps, m,n,n,nk )
      endif
!
! Flux descendant a la surface
! ...non corrige pour l'emissivite de la surface (s/r fcrest)
!
      do 501 j=1,n
         zfdsi(j) = p7((nkp-1)*n+j)
         zei  (j) = p5(j)      ! Flux ir au sommet de l'atmosphere (w/m2)
         znt  (j) = p12(j)     ! Nuages totaux
  501 continue
!
! Fin du calcul de radiation infrarouge
!
! Albedo utilise dans p10
! Albedo limite entre 6% et 80%
      do 1212 j=1,n
         p10(j)=amin1(zalvis_ag(j),0.80)
         p10(j)=amax1(p10(j),0.06)
 1212 continue
!
!
! Calcul de la variation de la constante solaire
!
      julien = juliand(tau,kount,date)
      alf    = julien/365.*2*pi
      r0r    = solcons(alf)
!
! Parametres d'entree pour le solaire
!
      call setvis2(delrd, p2, p3, p4, p6, &
                   p0rd,srd,trd,ps,p0rd,f(dlat),f(dlon),hz, &
                   julien,date,n,rednk,redm,satuco)
!
       do i=1,n
           zcosz(i) = p6(i)
       end do
!
      if( reduc ) then
        call rdmoy(p7  ,f(lwc) ,q20,inrd,n,nk,nkrd)
        call rdmoy(p8  ,f(iwc) ,q20,inrd,n,nk,nkrd)
        call rdmoy(lwprd,liqwp ,q20,inrd,n,nk,nkrd)
        call rdmoy(iwprd,icewp ,q20,inrd,n,nk,nkrd)
        call rdmoy(fnrd,nuage  ,q20,inrd,n,nk,nkrd)
!
        call cldoptx5 (p7,p8,lwprd,iwprd,fnrd,trd,srd,ps, &
                       f(dlat),f(mg),f(ml),n,n,nkrd, &
                       pbl,ipb,dzz,sdz,nef,opd,asy, &
                       f(topthw),f(topthi), &
                       v(ctp),v(ctt), &
                       ssa,aer,satuco,cw_rad,ioptix)
      else
        fnrd => nuage(1:)
      endif
!
! Calcul du cosinus de l'angle solaire a kount+kntrad-1
      hzp=amod(hz0+ (float(kount+kntrad-1)*tau)/3600., 24.)
      call suncos1(f(cosas),dummy1, dummy2, dummy3, dummy4,n, &
                   f(dlat),f(dlon),hzp,julien,date,.false.)
!
! Initialisation de fdss,t2,ev.
! f(cosas) contiendra la valeur moyenne des cosinus
! entre 2 appels a sun7
!vdir nodep
      do 487 j=1,n
         zfdss(j) = 0.0
         zev  (j) = 0.0
        zcosas(j) = (p6(j)+zcosas(j))*.5
487  continue
!
      do 488 k=1,nk
         zt2(:,k) = 0.0
488   continue
!
! Attention! les calculs sont faits pour un temps intermediaire
! entre kount et kount+kntrad
!

      call sun7( p8, p9, f(t20), f(vozo), ozotoit, &
                 delrd, p2, p3, &
                 p4, ps, trd, qrd, srd, &
                 p0rd, fnrd, aer, f(cosas), p10, &
                 n, rednk, rednkp, n, redm, &
                 q1 , q2 , q3 , q4 , q5 , &
                 q6 , q7 , q8 , q9 , q10, &
                 q11, q12, q13, q14, q15, &
                 ssa, asy, opd, q19, q20, &
                 q21, q22, q23, q24, q25, &
                 q26, q27, q28, q29, q30, &
                 q31, q32, q33, q34, q35, &
                 reduc, &
                 ss, ssrd, del, s, r0r, &
                 nk, nkp, radfix,radfltr)

! ap   : albedo planetaire.
! ev   : flux montant au sommet.
! fdss : flux descendant a la surface.
! On corrige le flux solaire au sol pour l'albedo (s/p fcrest)
!
!vdir nodep
      do 490 j=1,n
         zev0  (j) = p9(j)
         zfdss0(j) = amax1(0.0, p8((nkp-1)*n+j))
         zfdss0(j) = (1.-p10(j)) * zfdss0(j)
 490  continue
!
!
! Moduler les flux et les taux par le cosinus de l'angle solaire.
! v1 = Rapport des cosinus : angle actuel sur angle moyen.
! ap   (albedo planetaire) nul si flux solaire incident < 1 w/m2
!vdir nodep
      do 500 j=1,n
         v1 = p6(j)/zcosas(j)
         if(zcosas(j).gt.seuil.and.p6(j).gt.seuil) then
            zfdss(j)  = zfdss0(j) * v1
            zev  (j)  = zev0  (j) * v1
         endif
         zflusolis(j) =zfdss(j)/(1.-p10(j))
         fbas = p8(j) * v1
         if (fbas.gt.1.) then
            zap(j) = zev(j)/fbas
         else
            zap(j) = 0.
         endif
  500   continue
!
        do 5000 k=1,nk
!vdir nodep
          do 5000 j=1,n
             v1 = p6(j)/zcosas(j)
             if(zcosas(j).gt.seuil.and.p6(j).gt.seuil) then
                zt2(j,k) = zt20(j,k) * v1
             endif
 5000   continue
!
!**********************************************************
! Cas ou mod(kount-1,kntrad) non zero
      else
! Ajustement du solaire aux pas non multiples de kntrad
! par modulation avec cosinus de l'angle solaire
! Calcul du jour julien
      julien = juliand(tau,kount,date)
      call suncos1(p1,dummy1, dummy2, dummy3, dummy4,n, &
                   f(dlat),f(dlon),hz,julien,date, .false.)
!
! Moduler par le cosinus de l'angle solaire. mettre a zero les
! valeurs appropriees de fdss, ev et t2.
! v1 - Rapport des cosinus de l'angle present et de l'angle moyen.
!      albedo limite entre 6% et 80%
      do 5010 j=1,n
          v1 = p1(j)/zcosas(j)
          if(zcosas(j).gt.seuil.and.p1(j).gt.seuil) then
             zfdss (j) = zfdss0(j) * v1
             zev   (j) = zev0  (j) * v1
          else
             zfdss (j) = 0.0
             zev   (j) = 0.0
          endif
          alb = amin1(zalvis_ag(j),0.80)
          alb = amax1(alb,0.06)
          zflusolis(j) =zfdss(j)/(1.-alb)
 5010 continue
!
      do 503 k=1,nk
!vdir nodep
         do 503 j=1,n
            v1 = p1(j)/zcosas(j)
            if(zcosas(j).gt.seuil.and.p1(j).gt.seuil) then
               zt2(j,k) = zt20(j,k) * v1
            else
               zt2(j,k) = 0.0
            endif
  503 continue
!
! Seulement si radfix est vrai...
      if(radfix) then
         do k=1,nk
            toit(k) = .false.
            if (k.eq.1) toit(k) = .true.
         end do
         call sun_radfix1(s,p1,f(cosas),ps,f(t2),toit,n,nk)
      endif

! Fin de boucle sur radiation visible et infrarouge
      endif

! Extraction des series temporelles et
! es diagnostics zonaux des tendances

      call serxst2(f(ti), 'ti', trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(f(t2), 't2', trnch, n, nk, 0.0, 1.0, -1)

! Calcul du jour julien
      julien = juliand(tau,kount,date)
      call suncos1(p1, dummy1, dummy2, dummy3, dummy4,n,f(dlat), &
                   f(dlon),hz,julien,date,.false.)
      alf = julien/365.*2.*pi
      r0r = solcons(alf)

      do j=1,n
         zcang(j) = p1(j)
      end do

!vdir nodep
      do 508 j=1,n
         ziv(j)=consol*r0r*p1(j)*zvozo(j)
         if (ziv(j).gt.1.) then
            zap(j) = zev(j)/ziv(j)
         else
            zap(j) = 0.
         endif
508   continue
!
      do 509 j=1,n
  509    p1(j)=ziv(j)-zev(j)-zei(j)
!
! Extraction pour diagnostics
      call serxst2(v(ctp )  , 'bp', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(ctt)   , 'be', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(topthw), 'w3', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(topthi), 'w4', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(iv)    , 'iv', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(p1       , 'nr', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(nt)    , 'nt', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(ev)    , 'ev', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(ei)    , 'ei', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(ap)    , 'ap', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(fdss)  , 'fs', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(f(flusolis), 'fu', trnch, n, 1, 0.0, 1.0, -1)

      do j=1,n
         delp=ps(j)*0.5*(s(j,2)-s(j,1))
         p2(j)=zti(j,1)*delp
         p3(j)=zt2(j,1)*delp
      end do

       do k=2,nk-1
          do j=1,n
             delp=ps(j)*0.5*(s(j,k+1)-s(j,k-1))
             p2(j)=p2(j)+zti(j,k)*delp
             p3(j)=p3(j)+zt2(j,k)*delp
          end do
       end do

       do j=1,n
          delp=ps(j)*(1.-0.5*(s(j,nk)+s(j,nk-1)))
          p2(j)=p2(j)+zti(j,nk)*delp
          p3(j)=p3(j)+zt2(j,nk)*delp
       end do

       call serxst2(p2, 't3', trnch, n, 1, 0.0, 1.0, -1)
       call serxst2(p3, 't4', trnch, n, 1, 0.0, 1.0, -1)

! Tendances de la radiation
       do k = 1,nk
          ztrad(:,k) =zti(:,k) + zt2(:,k)
       end do

      RETURN
      END
