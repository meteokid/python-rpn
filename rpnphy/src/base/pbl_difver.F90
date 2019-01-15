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
subroutine pbl_difver1(db, dsiz, f, fsiz, v, vsiz, &
     seloc, tau, kount, trnch, n, nk, stack)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object perform the implicit vertical diffusion
   !@Arguments
   !          - Input/Output -
   ! DB       dynamic bus
   ! F        field for permanent physics variables
   ! V        volatile bus
   ! DSIZ     dimension of DB
   ! FSIZ     dimension of F
   ! VSIZ     dimension of V
   ! G        physics work space
   ! ESPG     dimension of G
   !
   !          - Output -
   ! TU       U  tendency
   ! TV       V  tendency
   ! TT       T  tendency
   ! TQ       Q  tendency
   ! TL       L tendency
   ! T        temperature
   ! UU       x component of the wind
   ! VV       y component of the wind
   ! Q        specific humidity
   ! QL       liquid water
   ! PS       surface pressure
   ! TM       temperature at time t-dt
   !
   !          - Input -
   ! SG       sigma levels
   ! SELOC    staggered sigma levels
   ! TAU      timestep * factdt * facdifv
   !          see common block "options"
   ! KOUNT    timestep number
   ! TRNCH    row number
   ! N        horizontal dimension
   ! NK       vertical dimension
   ! STACK    task number

   integer dsiz,fsiz,vsiz,kount,trnch,n,nk,stack,ierror
   real db(dsiz),f(fsiz),v(vsiz)
   real seloc(n*nk)
   real tau

   !@Author J. Cote (Oct 1984)
   !
   !@Revision
   ! 001      B. Bilodeau (Spring 1994) - New physics interface.
   !          Change name from POSTIMP (SEF model) to DIFVER.
   ! 002      R. Sarrazin, J. Mailhot and B. Bilodeau (Jan 1996) -
   !          Bug fixes for time-series extraction of "KM"
   !          Change name from DIFVER to DIFVER2.
   ! 003      M. Desgagne (Oct 1995) - Unified physics interface.
   !          Change name from DIFVER2 to DIFVER3.
   ! 004      B. Bilodeau (Sept 96) - Install the new memory
   !          management system (STK).
   ! 005      B. Bilodeau (Nov 96) - Replace common block pntclp by
   !                                 common block turbus
   ! 006      C. Girard (Feb 1996) - New option ISHLCVT, diffusion
   !          of temperature and cloud water.
   ! 007      G. Pellerin (mars 97) - New calling sequence and change
   !          name to difver4.
   ! 008      Y. Delage and B. Bilodeau (Jul 97)
   !          Move FQ calculation from flxsurf to difver
   ! 009      M. Desgagne (Spring 97) - Diffuse W
   ! 010      M. Roch     (Nov 1997)  - Introduce sponge modulation factor
   ! 011      S. Belair   (June 1998) - Turulent fluxes as output
   !                                    (subroutine ATMFLUX)
   ! 012      J. Mailhot  (Oct 1998) - New SURFACE interface and
   !                                   change name to DIFVER5
   ! 013      B. Bilodeau (Nov 1998) - Merge phyexe and param4
   ! 014      B. Bilodeau (Dec 1999) - NSLOFLUX
   ! 015      B. Bilodeau (Nov 2000) - New comdeck phybus.cdk
   ! 016      B. Bilodeau (Aug 2001) - Add call to CTMDIAG and
   !                                   change name to difver6
   ! 017      J. Mailhot  (May 2000) - Changes to add MOISTKE option (fluvert=MOISTKE)
   ! 018      J. Mailhot  (Jun 2000) - Changes to add mixed-phase mode in
   !                                   MOISTKE option (fluvert=MOISTKE)
   ! 019      J. Mailhot  (Oct 2000) - New calling sequence and
   !                                   change name to DIFVER6
   ! 020      B. Dugas    (Nov 2001) - Save the USTRESS and VSTRESS vector
   !                                   components as well as their module FQ
   ! 021      B. Bilodeau (Mar 2002) - HU tendency=0 if wet=.false.
   ! 022      B. Bilodeau (Mar 2002) - Eliminate unit conversion for
   !                                   KM, KT, BM and BT
   ! 023      J. Mailhot  (Avr 2002) - New calling sequence and
   !                                   change name to BAKTOTQ1
   ! 024      J. Mailhot  (Feb 2003) - MOISTKE option based on implicit clouds only
   !                                   Change call to baktotq2
   !
   ! 025      A-M. Leduc  (Jan 2003) - ISHLCVT becomes ISHLCVT(1)
   ! 026      A. Plante   (Jun 2003) - IBM conversion
   !             - divisions replaced by reciprocals (call to vsrec from massvp4 library)
   ! 027      F. Lemay (Spring 2003) - use average of BT for implicit boundary condition
   ! 028      L. Spacek (Aug 2004)   - cloud clean-up fn changes to fbl
   ! 029      J. Mailhot (Oct 2004) - Multiply BT by FSLOFLX for slow 
   !                                  start when impflx=.true.
   ! 030      C. Pelletier (Mar 2005) - Eliminate KM(NK) = KM(NK-1)
   ! 031      Y. Delage (Apr 2005)   - add return for fluver = SURFACE
   ! 032      L. Spacek/J. Mailhot (Dec 2007)   - add "vertical staggering" option
   !*@/
   
   include "thermoconsts.inc"
   include "surface.cdk"
   include "clefcon.cdk"

   real DQ,rhortvsg,mrhocmu
   real tplusnk,qplusnk,uplusnk,vplusnk,lscp
   integer J,K,type
   logical staggered
   real MAXIMUM, MINIMUM, RSG

   real, dimension(n) :: bmsg,btsg,fsloflx,a,aq,bq
   real, dimension(n,nk) :: presm,presw,prese
   real, dimension(n,pbl_nk), target :: pbl_s,pbl_se
   real, dimension(n,pbl_nk) :: pbl_presm,pbl_prese,kmsg,ktsg,rgam0, &
        zero,pbl_udifv,pbl_vdifv,pbl_wdifv,pbl_tdifv,pbl_qdifv,pbl_tl, &
        pbl_uplus,pbl_vplus,pbl_tplus,pbl_huplus,pbl_qcplus, &
        c,d,r,r1,r2,pbl_omegap,qclocal,ficelocal,pbl_tve,pbl_tvm
   real, dimension(n,pbl_nk+1) :: gam0,uflux,vflux,tflux,qflux

   !     fonction-formule
   integer jk
   jk(j,k) = (k-1)*n + j - 1

   !---------------------------------------------------------------------
   
   if (any(fluvert == (/'SURFACE', 'NIL    '/)))  return

   zero = 0.

   !     COMPUTE PBL COORDINATE AND PRESSURES
   call pbl_coord(pbl_s,pbl_se,db(sigm),seloc,PBL_KTOP,PBL_ZSPLIT,n,nk,PBL_NK)
   do k=1,nk
      do j=1,n
         presm(j,k) = db(sigm+jk(j,k)) * f(pmoins+j-1)
         presw(j,k) = db(sigw+jk(j,k)) * f(pmoins+j-1)
         prese(j,k) = seloc(1+jk(j,k)) * f(pmoins+j-1)
      enddo
   enddo
   do k=1,PBL_NK
      do j=1,n
         pbl_presm(j,k) = pbl_s(j,k) * f(pmoins+j-1)
         pbl_prese(j,k) = pbl_se(j,k) * f(pmoins+j-1)            
      enddo
   enddo

   !     ADAPT LOW RESOLUTION TENDENCIES TO HIGH-RESOLUTION GRID
   call ccpl_increment(pbl_uplus,f(pbl_umoins),db(uplus),db(umoins),pbl_presm,presm,n,pbl_nk,nk,'UU','linear')
   call ccpl_increment(pbl_vplus,f(pbl_vmoins),db(vplus),db(vmoins),pbl_presm,presm,n,pbl_nk,nk,'VV','linear')
   call ccpl_increment(pbl_tplus,f(pbl_tmoins),db(tplus),db(tmoins),pbl_presm,presw,n,pbl_nk,nk,'TT','linear')
   call ccpl_increment(pbl_huplus,f(pbl_humoins),db(huplus),db(humoins),pbl_presm,presw,n,pbl_nk,nk,'HU','linear')
   call ccpl_increment(pbl_qcplus,f(pbl_qcmoins),db(qcplus),db(qcmoins),pbl_presm,presw,n,pbl_nk,nk,'QC','linear')
   where (pbl_huplus < 0.)
      pbl_huplus = 0.
   endwhere

   !     INTERPOLATE TO HIGH VERTICAL RESOLUTION FOR THE PBL
   ! Energy level fields
   call vte_intvertx3(pbl_omegap,db(wplus),presw,pbl_presm,n,nk,pbl_nk,'WW','linear')
   ! Diagnostic quantities
   call mfotvt(pbl_tvm,pbl_tplus,pbl_huplus,n,pbl_nk,n)
   call vte_intvertx3(pbl_tve,pbl_tvm,pbl_presm,pbl_prese,n,pbl_nk,pbl_nk,'TT','linear')
   !
   !     STRUCTURE OF MODEL COORDINATE
   staggered = .false.
   if (db(sigt) > 0.) staggered = .true.
   !
   !     normalization factors for vertical diffusion in sigma coordinates
   !
   RSG = (GRAV/RGASD)
   do K=1,pbl_nk
!VDIR NODEP
      do J=1,N
         GAM0(J,K) = RSG*pbl_se(j,k)/pbl_tve(j,k)
         KMSG(J,K) = v(pbl_km+jk(j,k))*GAM0(J,K)**2
         KTSG(J,K) = v(pbl_kt+jk(j,k))*GAM0(J,K)**2
      end do
   end do
   call VSREC(RGAM0,GAM0,N*(pbl_nk))
   do K=1,pbl_nk
!VDIR NODEP
      do J=1,N
         v(pbl_gte+jk(j,k)) = v(pbl_gte+jk(j,k))*RGAM0(J,K)
         v(pbl_gq+jk(j,k)) = v(pbl_gq+jk(j,k))*RGAM0(J,K)
         v(pbl_gql+jk(j,k)) = v(pbl_gql+jk(j,k))*RGAM0(J,K)
      end do
   end do

   !     "SLOW START"

   do J=1,N
      FSLOFLX(J) = 1.
   end do
   !
   if (NSLOFLUX.gt.0) then
      do J=1,N
         !           OVER THE CONTINENT, WE PERFORM A SLOW START FOR 
         !           FLUXES "FC" AND "FV" UNTIL TIMESTEP "NSLOFLUX",
         !           BECAUSE OF IMBALANCES BETWEEN ANALYSES OF TEMPERATURE
         !           AT THE GROUND AND JUST ABOVE THE SURFACE.
         if (F(MG+J-1).gt.0.5) then
            !              MAX IS USED TO AVOID DIVISION BY ZERO
            FSLOFLX(J) = (FLOAT(KOUNT-1))/max(FLOAT(NSLOFLUX),1.)
            if (KOUNT.eq.0) FSLOFLX(J) = 0.0
         endif
      end do
   endif

!VDIR NODEP
   do J=1,N
      AQ(J)=-RSG/(F(TSURF+J-1)*(1. + &
           DELTA * F(QSURF+ (indx_agrege-1)*N + J-1)))
      BMSG(J)      = V(BM   +J-1)*AQ(J)
      BTSG(J)      = V(BT   + (indx_agrege-1)*N +J-1)*AQ(J)*FSLOFLX(J)
      V(ALFAT+J-1) = V(ALFAT+J-1)*AQ(J)*FSLOFLX(J)
      V(ALFAQ+J-1) = V(ALFAQ+J-1)*AQ(J)*FSLOFLX(J)
   end do

   gam0 = 0.

   ! DIFFUSION VERTICALE IMPLICITE

   type=1
   if(staggered)type=2

   ! DIFFUSE U

   call DIFUVDFJ(pbl_udifv,pbl_uplus,KMSG,ZERO,ZERO,ZERO,BMSG,pbl_s,pbl_se,TAU, &
        type,1.,C,D,R,R1,N,N,N,pbl_nk)

   call pbl_atmflux(uflux,pbl_uplus,pbl_udifv,KMSG,GAM0,ZERO(:,pbl_nk), &
        BMSG,f(pmoins),pbl_tplus,pbl_huplus,TAU,pbl_s    , &
        pbl_se,C,D,0,N,pbl_nk,TRNCH)

   ! DIFFUSE V

   call DIFUVDFJ(pbl_vdifv,pbl_vplus,KMSG,ZERO,ZERO,ZERO,BMSG,pbl_s,pbl_se,TAU, &
        type,1.,C,D,R,R1,N,N,N,pbl_nk)

   call pbl_atmflux(vflux,pbl_vplus,pbl_vdifv,KMSG,GAM0,ZERO(:,pbl_nk), &
        BMSG,f(pmoins),pbl_tplus,pbl_huplus,TAU,pbl_s    , &
        pbl_se,C,D,1,N,pbl_nk,TRNCH)

   ! DIFFUSE W (OMEGAP)

   pbl_wdifv = 0.
   if (diffuw) then
      call DIFUVDFJ(pbl_wdifv,pbl_omegap,KMSG,ZERO,ZERO,ZERO,ZERO,pbl_s,pbl_se,TAU, &
           type,1.,C,D,R,R1,N,N,N,pbl_nk)
   endif

   ! DIFFUSE MOISTURE

   if(EVAP) then
      do J=1,N
         AQ(J) = V(ALFAQ+J-1)
         BQ(J) = BTSG(J)
      end do

   else

      !     LA CLE 'EVAP' EST VALIDE POUR PARAMETRAGES CLEF ET SIMPLIFIES
      !     METTRE TERMES DE SURFACE A ZERO
      do J=1,N
         AQ(J) = 0.0
         BQ(J) = 0.0
      end do

   endif

   if (fluvert == 'MOISTKE') then
      ! diffuse conservative variable qw
      ! QTBL contains the implicit cloud water from previous time step.
      do K=1,pbl_nk
         do J=1,N
            QCLOCAL(J,K) = max( 0.0 , f(pbl_qtbl+jk(j,k)) )
         end do
      end do
      pbl_huplus = pbl_huplus + qclocal
   endif
   
   call DIFUVDFJ(pbl_qdifv,pbl_huplus,KTSG,v(pbl_gq),ZERO,AQ,BQ,pbl_s,pbl_se,TAU, &
        type,1.,C,D,R,R1,N,N,N,pbl_nk)
   
   call pbl_atmflux(qflux,pbl_huplus,pbl_qdifv,KTSG,GAM0           , &
        AQ,BQ,f(pmoins),pbl_tplus,pbl_huplus,TAU,pbl_s, &
        pbl_se,C,D,2,N,pbl_nk,TRNCH)
   
   ! DIFFUSE L'EAU LIQUIDE OPTIONNELLEMENT
     
   pbl_tl = 0.
   if(PBL_SHAL == 'SHALODQC') then
      
      do J=1,N
         BQ(J) = 0.
      end do
      
      call DIFUVDFJ(pbl_tl,pbl_qcplus,KTSG,v(pbl_gql),ZERO,ZERO,BQ,pbl_s,pbl_se,TAU, &
           type,1.,C,D,R,R1,N,N,N,pbl_nk)
      
   endif
   
   ! DIFFUSE TEMPERATURE
   
   if (CHAUF) then
      do J=1,N
         AQ(J) = V(ALFAT+J-1)
         BQ(J) = BTSG(J)
      end do
      
   else
      
      !     LA CLE 'CHAUF' EST VALIDE POUR PARAMETRAGES CLEF ET SIMPLIFIES
      !     METTRE TERMES DE SURFACE A ZERO
      do J=1,N
         AQ(J) = 0.0
         BQ(J) = 0.0
      end do
      !
   endif
   
   if (fluvert == 'MOISTKE') then
      ! diffuse conservative variable thetal
      call FICEMXP(ficelocal,R1,R2,f(pbl_tmoins),N,N,pbl_nk)
      do K=1,pbl_nk
         do J=1,N
            ! copy current T in R2 for later use in BAKTOTQ
            R2(J,K) = pbl_tplus(j,k)
            pbl_tplus(j,k) = pbl_tplus(j,k) &
                 - ((CHLC+FICELOCAL(J,K)*CHLF)/CPD) &
                 *max( 0.0 , QCLOCAL(J,K) )
            pbl_tplus(j,k) = pbl_tplus(j,k) * pbl_s(j,k)**(-CAPPA)
         end do
      end do
   endif
   
   call DIFUVDFJ(pbl_tdifv,pbl_tplus,KTSG,v(pbl_gte),ZERO,AQ,BQ,pbl_s,pbl_se,TAU, &
        type,1.,C,D,R,R1,N,N,N,pbl_nk)

   if (fluvert == 'MOISTKE') then
      ! back to non-conservative variables T and Q
      ! and their tendencies

      ! FIXME for moistke
      write(0,*) 'ERROR - baktotq4() not implemented in difver7() ...'
      stop
      call baktotq4(db(tplus), db(huplus), qclocal, r2, db(sigm), db(sigw), f(pmoins), db(tmoins), ficelocal, &
           v(tdifv), v(qdifv), v(ldifv), v(tve), f(qtbl), &
           f(fnn), f(fbl), f(zn), f(zd), f(mg), &
           v(at2t),v(at2m),v(at2e),tau, n, n, nk)

   endif

   !                           Counter-gradient term = -g/cp
   !                           because temperature is used in
   !                           subroutine DIFUVDF (instead of
   !                           potential temperature).

   gam0 = -GRAV/CPD

   call pbl_atmflux(tflux,pbl_tplus,pbl_tdifv,KTSG,GAM0,AQ,BQ, &
        F(PMOINS),pbl_tplus,pbl_huplus,TAU,pbl_s, &
        pbl_se,C,D,3,N,pbl_nk,TRNCH)

   ! DIFFUSE AVEC CONDENSATION OPTIONNELLEMENT

   if (PBL_SHAL == 'SHALODQC') then
      !
      !        soit avec condensation seulement
      !
      !        lscp permet de convertir les variables du type
      !        f(pmoins)*dq/dt en flux d'energie (W/m2)
      lscp = chlc/cpd

      do k = 1, pbl_nk
         do j = 1, N
            DQ = min(pbl_tl(j,k),-pbl_huplus(j,k)/TAU)+pbl_qcplus(j,k)/TAU
            pbl_qdifv(j,k) = pbl_qdifv(j,k) + DQ
            pbl_tl(j,k) = max( pbl_tl(j,k) , - pbl_huplus(j,k)/TAU )
            pbl_tdifv(j,k) = pbl_tdifv(j,k) - lscp*DQ
         end do
      end do
      !    
   endif

   !     INTERPOLATE OUTPUTS BACK TO PHYSICS VERTICAL GRID
   call vte_intvertx3(v(udifv),pbl_udifv,pbl_presm,presm,n,pbl_nk,nk,'UU','linear')
   call vte_intvertx3(v(vdifv),pbl_vdifv,pbl_presm,presm,n,pbl_nk,nk,'VV','linear')
   call vte_intvertx3(v(tdifv),pbl_tdifv,pbl_presm,presw,n,pbl_nk,nk,'TT','linear')
   call vte_intvertx3(v(qdifv),pbl_qdifv,pbl_presm,presw,n,pbl_nk,nk,'HU','linear')
   call vte_intvertx3(v(ldifv),pbl_tl,pbl_presm,presw,n,pbl_nk,nk,'QC','linear')
   if (diffuw) call vte_intvertx3(v(wdifv),pbl_wdifv,pbl_presm,presw,n,pbl_nk,nk,'WW','linear')
   do k=1,PBL_KTOP-1
      do j=1,n
         v(udifv+jk(j,k)) = 0.
         v(vdifv+jk(j,k)) = 0.
         v(wdifv+jk(j,k)) = 0.
         v(tdifv+jk(j,k)) = 0.
         v(qdifv+jk(j,k)) = 0.
         v(ldifv+jk(j,k)) = 0.
      enddo
   enddo

   !     LOW RESOLUTION TENDENCIES AND TIME-FLIP DONE IN PHY_EXE()

   !     APPLY HIGH RESOLUTION TENDENCIES AND TIME-FLIP
   if (kount > 0) then
      pbl_uplus = pbl_uplus + tau * pbl_udifv
      pbl_vplus = pbl_vplus + tau * pbl_vdifv
      pbl_tplus = pbl_tplus + tau * pbl_tdifv
      pbl_huplus = pbl_huplus + tau * pbl_qdifv
      pbl_qcplus = pbl_qcplus + tau * pbl_tl
      call ccpl_timeflip(f(pbl_umoins),pbl_uplus,n,pbl_nk)
      call ccpl_timeflip(f(pbl_vmoins),pbl_vplus,n,pbl_nk)
      call ccpl_timeflip(f(pbl_tmoins),pbl_tplus,n,pbl_nk)
      call ccpl_timeflip(f(pbl_humoins),pbl_huplus,n,pbl_nk)
      call ccpl_timeflip(f(pbl_qcmoins),pbl_qcplus,n,pbl_nk)
   endif

   ! CALCUL FINAL DU BILAN DE SURFACE

!VDIR NODEP
   do j = 1, n
      rhortvsg = f(pmoins+j-1)/grav
      mrhocmu  = f(pmoins+j-1)/grav*bmsg(j)
      tplusnk  = db(tplus+jk(j,nk))+tau*v(tdifv+jk(j,nk))
      qplusnk  = db(huplus+jk(j,nk))+tau*v(qdifv+jk(j,nk))
      uplusnk  = db(uplus+jk(j,nk))+tau*v(udifv+jk(j,nk))
      vplusnk  = db(vplus+jk(j,nk))+tau*v(vdifv+jk(j,nk))
      !
      !        RECALCULER LES FLUX PARTOUT
      !
      !        USTRESS et VSTRESS sont calcules apres diffusion car
      !        on utilise toujours une formulation implicite pour
      !        la condition aux limites de surface pour les vents.
      !        Par contre, la formulation est explicite pour
      !        la temperature et l'humidite.
      !        A noter que, puisque la formulation est explicite,
      !        on agrege les flux FC et FV dans le sous-programme 
      !        AGREGE; on pourrait aussi les calculer ici en tout
      !        temps, mais on ne le fait que pendant le "depart lent".
      !        Si on utilisait une formulation implicite pour 
      !        FC et FV, il faudrait que ces derniers soient 
      !        toujours calcules ici.

      if (NSLOFLUX.gt.0.and.KOUNT.le.NSLOFLUX) then
         v(fc+(indx_agrege-1)*N+j-1) =  CPD * rhortvsg * &
              (v(alfat+j-1)+btsg(j)*tplusnk)
         v(fv+(indx_agrege-1)*N+j-1) = CHLC * rhortvsg * &
              (v(alfaq+j-1)+btsg(j)*qplusnk)
      endif

      v(ustress+j-1) = -mrhocmu*uplusnk
      v(vstress+j-1) = -mrhocmu*vplusnk
      f(fq+j-1)      = -mrhocmu*sqrt(uplusnk**2 + vplusnk**2)

      if (.not.CHAUF)  v(FC+(indx_agrege-1)*N+j-1) = 0.0
      if (.not.EVAP )  v(FV+(indx_agrege-1)*N+j-1) = 0.0

      A(j) = f(FDSI+j-1)*f(EPSTFN+j-1)/STEFAN
      V(FNSI+J-1) = A(j)-f(EPSTFN+j-1)*f(TSRAD+j-1)**4
      V(FL  +J-1) = V(FNSI                +j-1)    + &
           f(FDSS                +j-1)    - &
           V(FV+(indx_agrege-1)*n+j-1)    - &
           V(FC+(indx_agrege-1)*n+j-1)
   enddo

   !     DIAGNOSTICS
   
   call serxst2(v(udifv), 'TU', TRNCH, N, nk, 0.0, 1.0, -1)
   call serxst2(v(vdifv), 'TV', TRNCH, N, nk, 0.0, 1.0, -1)

   call serxst2(v(tdifv), 'TF', TRNCH, N, nk, 0.0, 1.0, -1)
   call serxst2(v(qdifv), 'QF', TRNCH, N, nk, 0.0, 1.0, -1)
   call serxst2(v(ldifv), 'LF', TRNCH, N, nk, 0.0, 1.0, -1)

   call serxst2(F(FQ),    'FQ', TRNCH, N,  1, 0.0, 1.0, -1)

   call serxst2(V(FC+(indx_soil -1)*N),   'F4', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FC+(indx_glacier-1)*N), 'F5', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FC+(indx_water -1)*N),  'F6', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FC+(indx_ice -1)*N),    'F7', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FC+(indx_agrege -1)*N), 'FC', TRNCH, N, 1, 0., 1., -1)

   call serxst2(V(FV+(indx_soil -1)*N),   'H4', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FV+(indx_glacier-1)*N), 'H5', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FV+(indx_water -1)*N),  'H6', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FV+(indx_ice -1)*N),    'H7', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FV+(indx_agrege -1)*N), 'FV', TRNCH, N, 1, 0., 1., -1)

   call serxst2(A,       'FI', TRNCH, N, 1, 0., 1., -1)
   call serxst2(V(FNSI), 'SI', TRNCH, N, 1, 0.,1.,-1)
   call serxst2(V(FL),   'FL', TRNCH, N, 1, 0., 1., -1)

   ! DIAGNOSTICS POUR LE MODELE CTM
   if (any(fluvert == (/'MOISTKE', 'CLEF   '/))) then
      call CTMDIAG(DB,F,V,DSIZ,FSIZ,VSIZ,N,NK)
      call serxst2(v(ue), 'UE', TRNCH, N, 1, 0.0, 1.0, -1)
   endif

   call serxst2(V(KM), 'KM', TRNCH, N, nk, 0.0, 1.0, -1)
   call serxst2(V(KT), 'KT', TRNCH, N, nk, 0.0, 1.0, -1)

   return
end subroutine pbl_difver1
