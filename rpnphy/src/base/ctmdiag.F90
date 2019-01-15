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
      subroutine ctmdiag (d, f, v, dsiz, fsiz, vsiz, ni, nk)
!
      use phybus
      use phy_options, only: ilmodiag
      implicit none
#include <arch_specific.hf>
!
      integer dsiz, fsiz, vsiz, ni, nk
      real, target :: d(dsiz), f(fsiz), v(vsiz)
!
!Author
!          B. Bilodeau (March 2000)
!
!Revisions
! 001      see version 5.5.0 for previous history
! 002      A. Zadra, Nov 2014: two corrections to the 
!             calculation of rib (vertical staggering
!             and moisture term in thetavs)
!
!Object
!          to calculate special diagnostics for the CTM (CHRONOS)
!
!Arguments
!
!          - Input/Output -
! d        dynamic             bus
! f        permanent variables bus
! v        volatile (output)   bus
!
!          - input -
! dsiz     dimension of d
! fsiz     dimension of f
! vsiz     dimension of v
!
!          - input -
! ni       number of elements processed in the horizontal
! nk       vertical dimension
!
!Notes
!
!Implicites
!
#include "surface.cdk"
include "thermoconsts.inc"
!
      real dthetav, rho, thetas, thetavs, zp
!
!*
!
      real fcv
      real, pointer, dimension(:) :: zctue, zfc_ag, zfq, zfv_ag, zilmo_ag, &
                                     zmol, zphit0, zpsmoins,zqsurf_ag, zrib, &
                                     zthetaa, ztsurf, zue, zz0t_ag, zza, zztsl
      real, pointer, dimension(:) :: zhumoins, zkt, ztmoins, zumoins, zvmoins
!
!
      integer i, j, k
!
      zctue    (1:ni) => v( ctue:)
      zfc_ag   (1:ni) => v( fc+(indx_agrege-1)*ni:)
      zfq      (1:ni) => f( fq:)
      zfv_ag   (1:ni) => v( fv+(indx_agrege-1)*ni:)
      zilmo_ag (1:ni) => f( ilmo+(indx_agrege-1)*ni:)
      zmol     (1:ni) => v( mol:)
      zphit0   (1:ni) => v( phit0:)
      zpsmoins (1:ni) => f( pmoins:)
      zrib     (1:ni) => v( rib:)
      zqsurf_ag(1:ni) => f( qsurf+(indx_agrege-1)*ni:)
      ztsurf   (1:ni) => f( tsurf:)
      zthetaa  (1:ni) => v( thetaa:)
      zue      (1:ni) => v( ue:)
      zz0t_ag  (1:ni) => f( z0t+(indx_agrege-1)*ni:)
      zza      (1:ni) => v( za:)
      zztsl    (1:ni) => v( ztsl:)
!
      zhumoins (1:ni) => d( humoins+(nk-1)*ni:)
      zkt      (1:ni) => v( kt+nk*ni:)
      ztmoins  (1:ni) => d( tmoins +(nk-1)*ni:)
      zumoins  (1:ni) => d( umoins +(nk-1)*ni:)
      zvmoins  (1:ni) => d( vmoins +(nk-1)*ni:)
!
!VDIR NODEP
      do i=1,ni
!
!        densite
         rho     = zpsmoins(i)/(rgasd*ztmoins(i) * (1.0+delta*zhumoins(i)))
!
!        coefficient de transfert (non necessaire pour CTM)
!        note : en mode "agregation", la formule suivante donne
!        des valeurs negatives pour CTUE

!        v(ctue+i-1) = v(fc+jk(i,indx_agrege)) /
!    +                 (cpd*rho*(ztsurf(i)-v(thetaa+i-1)))
         zctue(i) = 0.0
!
!        vitesse de frottement (fq est calcule dans difver5)
         zue(i) = sqrt(zfq(i)/rho)
!
!        temperature potentielle a la surface
         thetas  = ztsurf(i)
!
!        temperature potentielle virtuelle a la surface
         thetavs = thetas * (1.+delta*zqsurf_ag(i))
!
         dthetav = zthetaa(i) * (1.+delta*zhumoins(i)) - thetavs
!
!        _____________
!        (w' thetav')s
         fcv = (1.+ delta*zqsurf_ag(i)) * zfc_ag(i)/(cpd *rho)   + &
                    delta*thetas    * zfv_ag(i)/(chlc*rho)
!
!
!        fcv doit etre non nul pour eviter les divisions par zero
         fcv = sign( max(abs(fcv),1.e-15) , fcv )
!
!        longueur de monin-obukhov
         zmol(i) = -zue(i)**3 * thetavs / (karman*grav*fcv)
         if (ilmodiag) &
            zilmo_ag(i) = 1. / &
               sign (max ( abs(zmol(i)), 0.1) , - fcv)
!
!        bornes pour MOL
         zmol(i) = min(1000.,max(-1000.,zmol(i)))
!        if (v(mol+i-1).gt.0.0) v(mol+i-1)=min(v(mol+i-1), 1000.0)
!        if (v(mol+i-1).lt.0.0) v(mol+i-1)=max(v(mol+i-1),-1000.0)
!
!        nombre de richardson "bulk"
         zp = zza(i)*zza(i)/zztsl(i)
         zrib(i) = grav * dthetav * zp / &
            ( (thetavs+0.5*dthetav) * max(vamin,(zumoins(i)**2+zvmoins(i)**2)) )
!
!        valeurs limites pour RIB
         if      (zrib(i).ge.0.0) then
            zrib(i) = max(min(zrib(i),  5.0), n0rib)
         else if (zrib(i).lt.0.0) then
            zrib(i) = min(max(zrib(i),-10.0),-n0rib)
         endif
!
!        coherence entre RIB et MOL
         if (zrib(i)*zmol(i).lt.0.0) zmol(i) = sign(1000.,zrib(i))
!
!        fonctions de stabilite
         if (zmol(i).lt.0.0) then
            zphit0(i)=1.0/sqrt(1.0-20.0*zrib(i))
         else
            zphit0(i)=1.0 + 5.0*zrib(i)
         end if
!
!        coefficients de diffusion pour la temperature : niveau du bas
         zkt(i)=karman*zz0t_ag(i)*zue(i) /  zphit0(i)
!
      end do
!
      return
      end
