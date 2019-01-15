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
subroutine turbul(d, dsiz, f, fsiz, v, vsiz, qe, se, &
     kount, trnch, n, m, nk, it)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object interface for turbulent kinetic energy calculations
   !@Arguments
   !          - Input/Output -
   ! d        dynamic             bus
   ! f        permanent variables bus
   ! v        volatile (output)   bus
   ! tke      turbulent energy
   !          - Input -
   ! dsiz     dimension of d
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! qe       specific humidity on 'e' levels
   ! se       sigma level for turbulent energy
   ! kount    index of timestep
   ! trnch    number of the slice
   ! n        horizontal dimension
   ! m        1st dimension of t, q, u, v
   ! nk       vertical dimension
   ! it       task number in multi-tasking

   integer :: dsiz, fsiz, vsiz
   integer :: it,kount,trnch,n,m,nk
   real, target :: d(dsiz), f(fsiz), v(vsiz)
   real :: qe(m,nk)
   real :: se(m,nk)

   !@Author J. Mailhot and B. Bilodeau (Jan 1999)
   !*@/

   include "thermoconsts.inc"
   include "clefcon.cdk"
   include "surface.cdk"
   include "tables.cdk"
   include "phyinput.cdk"

   integer, external :: neark

   real, save :: tfilt = 0.1

   integer :: ierget, i, k, nnk, stat, ksl(n)
   real :: cf1, cf2, eturbtau, zntem, uet, ilmot, &
        fhz, fim, fit, hst_local, &
        work2(n), b(n,nk*4), xb(n), xh(n)
   
   real, pointer, dimension(:)   :: zalfat, zbt_ag, zfrv_ag, zftemp_ag, &
        zfvap_ag, zh, zhst_ag, zilmo_ag, &
        zkcl, zscl, zz0_ag, zztsl, zwstar
   real, pointer, dimension(:,:) :: tke, zenmoins, zkm, zkt, ztmoins, &
        ztve, zze, zzn

   real, dimension(n,nk) :: c,x,x1,wk2d,enold,tmom,qmom,te,qce

   include "dintern.inc"
   include "fintern.inc"

   nnk = n*nk
   zalfat   (1:n) => v( alfat:)
   zbt_ag   (1:n) => v( bt+(indx_agrege-1)*n:)
   zfrv_ag  (1:n) => f( frv+(indx_agrege-1)*n:)
   zftemp_ag(1:n) => f( ftemp+(indx_agrege-1)*n:)
   zfvap_ag (1:n) => f( fvap+(indx_agrege-1)*n:)
   zh       (1:n) => f( h:)
   zhst_ag  (1:n) => f( hst+(indx_agrege-1)*n:)
   zilmo_ag (1:n) => f( ilmo+(indx_agrege-1)*n:)
   zkcl     (1:n) => v( kcl:)
   zscl     (1:n) => f( scl:)
   zwstar   (1:n) => v( wstar:)
   zz0_ag   (1:n) => f( z0+(indx_agrege-1)*n:)
   zztsl    (1:n) => v( ztsl:)

   tke     (1:n,1:nk+1) => f( en:)
   zkm     (1:n,1:nk+1) => v( km:)
   zkt     (1:n,1:nk+1) => v( kt:)
   ztmoins (1:n,1:nk+1) => d( tmoins:)
   ztve    (1:n,1:nk+1) => v( tve:)
   zze     (1:n,1:nk+1) => v( ze:)
   zzn     (1:n,1:nk+1) => f( zn:)

   eturbtau=delt
   if (advectke) then
      zenmoins(1:n,1:nk+1) => d( enmoins:)
      if (kount.gt.1) then
         eturbtau=factdt*delt
      endif
   endif

   !  filtre temporel

   cf1=1.0-tfilt
   cf2=tfilt

   !  initialiser e avec ea,z et h
   
   if (kount == 0) then

      INIT_TKE: if (any('en'==phyinread_list_s(1:phyinread_n))) then 
         tke = max(tke,0.)
      else
         do k=1,nk
!vdir nodep
            do i=1,n
               tke(i,k)= max( etrmin, blconst_cu*zfrv_ag(i)**2 * &
                    exp(-(zze(i,k)- zze(i,nk))/zh(i)) )
            end do
         end do
      endif INIT_TKE

      if (advectke) then
         do k=1,nk
!vdir nodep
            do i=1,n
               zenmoins(i,k) = tke(i,k)
            end do
         end do
      endif

   endif

   if (kount.gt.0) then
      call serxst2(f(zn), 'LM', trnch, n, nk+1, 0.0, 1.0 , -1)
      call serxst2(tke  , 'EN', trnch, n, nk+1, 0.0, 1.0 , -1)
   endif

   do k=1,nk
!vdir nodep
      do i=1,n
         if (advectke) then
            enold(i,k) = zenmoins(i,k)
            tke(i,k)   = max(tke(i,k),etrmin)
         else
            enold(i,k) = tke(i,k)
         endif
      end do
   end do

   
   ! convective velocity scale w*
   ! (passed to MOISTKE3 through XH)
   stat = neark(d(sigt),f(pmoins),1000.,n,nk,ksl)
   do i=1,n
      xb(i)=1.0+delta*qe(i,ksl(i))
      xh(i)=(grav/(xb(i)*ztve(i,ksl(i)))) * ( xb(i)*zftemp_ag(i) &
           + delta*ztve(i,ksl(i))*zfvap_ag(i) )
      xh(i)=max(0.0,xh(i))
      work2(i)=zh(i)*xh(i)
   end do
   call vspown1 (xh,work2,1./3.,n)

   do i=1,n
      zwstar(i) = xh(i)
   enddo
   
   if (fluvert == 'MOISTKE') then

      call moistke6( tke,enold,f(zn),f(zd),v(rif),v(rig),v(shear2),v(kt),f(qtbl), &
           f(fbl),f(fnn),v(gte),v(gq),v(gql),f(h), &
           d(umoins), d(vmoins), d(tmoins), &
           v(tve), d(humoins), qe, f(pmoins), d(sigm),se,d(sigw), &
           v(at2t),v(at2m),v(at2e), &
           wk2d, c, b, x, 4*nk, eturbtau, kount, &
           v(ze), f(z0+(indx_agrege-1)*n), v(gzmom), &
           v(kcl),zfrv_ag,f(turbreg), &
           x1,xb,xh,trnch,n,m,nk,it)

   else

      ! Interpolate input profiles to correct set of levels
      call tothermo(qmom,d(humoins),v(at2m),v(at2m),n,nk+1,nk,.false.)
      call tothermo(tmom,d(tmoins),v(at2m),v(at2m),n,nk+1,nk,.false.)
      call tothermo(d(tmoins),te,v(at2e),v(at2e),n,nk+1,nk,.true.)
      call tothermo(d(qcmoins),qce,v(at2e),v(at2e),n,nk+1,nk,.true.)

      call eturbl9( tke,enold,f(zn),f(zd),v(rif),f(turbreg),v(rig),v(shear2), &
           v(gte),f(ilmo+(indx_agrege-1)*n),f(fbl),v(gql),f(lwc),d(umoins), &
           d(vmoins),tmom,te,v(tve),qmom,qce,qe,f(h),f(pmoins),f(tsurf), &
           d(sigm),se,eturbtau,kount,v(gq),f(ftot),v(kt),v(ze),v(gzmom), &
           v(kcl),std_p_prof,zfrv_ag,xh,trnch,n,nk, &
           f(z0+(indx_agrege-1)*n),it)

      ! implicit diffusion scheme: the diffusion interprets the coefficients of the
      ! surface boundary fluxe condition as those in the alfat+bt*ta expression.
      ! Since the diffusion is made on potential temperature, there is a correction
      ! term that must be included in the non-homogeneous part of the expression -
      ! the alpha coefficient. The correction is of the form za*g/cp. It is
      ! relevant only for the clef option (as opposed to the moistke option) since
      ! in this case, although the diffusion is made on potential temperature,
      ! the diffusion calculation takes an ordinary temperature as the argument.

      if (impflx) then
         do i=1,n
            zalfat(i) = zalfat(i) + zbt_ag(i)*zztsl(i)*grav/cpd
         end do
      endif

   endif

   !    diagnose variables for turbulent wind (gusts and standard deviations)
   
   if (diag_twind) then
      call twind( v(wge), v(wgmax), v(wgmin), v(sdtsws), v(sdtswd), &
           v(tve), enold, d(umoins), d(vmoins), f(udiag), f(vdiag), &
           se, v(ze), f(h), zfrv_ag, &
           zwstar, n, nk)

      call serxst2(v(wge)   , 'WGE' , trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(wgmax) , 'WGX' , trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(wgmin) , 'WGN' , trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(sdtsws), 'SDWS', trnch, n, 1, 0.0, 1.0, -1)
      call serxst2(v(sdtswd), 'SDWD', trnch, n, 1, 0.0, 1.0, -1)
   endif
   
   if (kount.eq.0) then
      call serxst2(f(zn), 'LM', trnch, n, nk+1, 0.0, 1.0, -1)
      call serxst2(tke  , 'EN', trnch, n, nk+1, 0.0, 1.0, -1)
   endif
   
   
   !  ------------------------------------------------------
   !    Hauteur de la couche limite stable ou instable
   !  ------------------------------------------------------
   !
   !     Ici hst est la hauteur calculee dans la routine flxsurf.
   !
   !     kcl contient le k que l'on a diagnostique dans rigrad
   !     et passe a eturbl; il pointe vers le premier niveau
   !     de la couche limite.
   !
   !     scl est utilise comme champ de travail dans la boucle 100;
   !     il est mis a jour dans la boucle 200, et donne la hauteur
   !     de la couche limite en sigma.
   
   
!vdir nodep
   do i=1,n
      if (zilmo_ag(i).gt.0.0) then
         !# Cas stable
         zscl(i) = zhst_ag(i)
      else
         !# Cas instable: max(cas neutre, diagnostic)
         !# Z contient les hauteurs des niveaux intermediaires
         zscl(i) = max( zhst_ag(i) , zze(i,nint(zkcl(i))))
      endif

      !# Si H est en train de chuter, on fait une relaxation pour
      !# ralentir la chute.
      if (zh(i) .gt. zscl(i)) then
         zh(i) = zscl(i) + (zh(i)-zscl(i))*exp(-delt/5400.)
      else
         zh(i) = zscl(i)
      endif
   enddo

   !# On calcule scl avec une approximation hydrostatique
   do i=1,n
      zscl(i)=-grav*zh(i)/(rgasd*ztmoins(i,nk))
   enddo
   call vsexp(zscl,zscl,n)

   call serxst2(f(h)  , 'F2', trnch, n, 1, 0.0, 1.0, -1)
   call serxst2(f(scl), 'SE', trnch, n, 1, 0.0, 1.0, -1)

   do k=1,nk-1
!vdir nodep
      do i=1,n
         !# IBM conv. ; pas d'avantage a precalculer sqrt ci-dessous
         zkm(i,k) = blconst_ck*zzn(i,k)*sqrt(tke(i,k))
         zkt(i,k) = zkm(i,k)*zkt(i,k)
      end do
   end do

   do i=1,n
      if (fluvert /= 'MOISTKE') tke(i,nk) = tke(i,nk-1)
      tke(i,nk+1) = tke(i,nk)
      uet = zfrv_ag(i)
      ilmot = zilmo_ag(i)
      if (ilmot > 0.) then
         !# hst_local is used to avoid division by zero
         hst_local = max( zhst_ag(i), zztsl(i)+1.)
         fhz = 1-zztsl(i)/hst_local
         fim = 0.5*(1+sqrt(1+4*as*zztsl(i)*ilmot*beta/fhz))
         fit = beta*fim
      else
         fim=(1-ci*zztsl(i)*ilmot)**(-.16666666)
         fit=beta*fim**2
         fhz=1
      endif
      zzn(i,nk)   = karman*(zztsl(i)+zz0_ag(i))/fim
      zkm(i,nk)   = uet*zzn(i,nk)*fhz
      zkt(i,nk)   = zkm(i,nk)*fim/fit
      zzn(i,nk+1) = karman*zz0_ag(i)
      zkm(i,nk+1) = uet*zzn(i,nk+1)
      zkt(i,nk+1) = zkm(i,nk+1)/beta
   enddo

   return
end subroutine turbul
