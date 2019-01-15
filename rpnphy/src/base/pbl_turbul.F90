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
subroutine pbl_turbul(D, DSIZ, F, FSIZ, V, VSIZ, QE, SE, KOUNT, TRNCH, &
     N, M, NK, IT)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object interface for turbulent kinetic energy calculations
   !@Arguments
   !          - Input/Output -
   ! D        dynamic             bus
   ! F        permanent variables bus
   ! V        volatile (output)   bus
   ! TKE      turbulent energy
   !          - Input -
   ! DSIZ     dimension of D
   ! FSIZ     dimension of F
   ! VSIZ     dimension of V
   ! ESPWORK  dimension of WORK
   ! QE       specific humidity on 'E' levels
   ! SE       sigma level for turbulent energy
   ! KOUNT    index of timestep
   ! TRNCH    number of the slice
   ! N        horizontal dimension
   ! M        1st dimension of T, Q, U, V
   ! NK       vertical dimension
   ! IT       task number in multi-tasking

   integer DSIZ, FSIZ, VSIZ
   integer IT,KOUNT,TRNCH,N,M,NK
   real D(DSIZ), F(FSIZ), V(VSIZ)
   real QE(M,NK)
   real SE(M,NK)

   !@Author J. Mailhot and B. Bilodeau (Jan 1999)
   !@Revision
   ! 001      B. Bilodeau (Nov 2000) - New comdeck phybus.cdk
   ! 002      J. Mailhot  (May 2000) - Add MOISTKE option (fluvert=MOISTKE)
   ! 003      A-M.Leduc   (Dec 2002) - Add argument qc and remove
   !                                   ISHLCVT to call eturbl4--->eturbl5
   ! 004      J. Mailhot  (Feb 2003) - Restore ADVECTKE option
   ! 005      A. Plante   (May 2003) - IBM conversion
   !             - calls to exponen4 (to calculate power function '**')
   !             - calls to vexp routine (from massvp4 library)
   ! 006      B. Bilodeau (Aug 2003) - exponen4 replaced by vspown1
   ! 007      F. Lemay (Spring 2003) - Add implicit boundary condition 
   !                                   option for vert. diff.
   ! 008      Y. Delage (Aug 2003) - Fill extra levels near sfc for
   !                                 TKE, ZN, KM and KT
   ! 009      B. Bilodeau (Jun 2004) - Correct ue2 bugs
   ! 010      S.Belair    (Mar 2003)- Add F(ZD) in call...>eturbl6
   ! 011      L. Spacek (Aug 2004) - cloud clean-up fn, ccn
   !                                 change to fbl, ftot respectively
   ! 012      Y. Delage (Sept 2004) - Change UE2 by FRV
   ! 013      M. Roch and B. Bilodeau (Jan 2006) - Prevent division by zero
   !                                               with hst_local
   ! 014      J. Mailhot (Apr 2009) - Add wind gust diagnostic
   ! 015      A-M. Leduc (Mar 2010) - Move the calculation of XH before the if moistke condition
   !                                  to be available for clef-conres as well.
   !                                  (XH Needed for Bechtold and TWIND)
!*@/
   include "thermoconsts.inc"
   include "surface.cdk"
   include "clefcon.cdk"
   include "tables.cdk"

   real, save :: TFILT = 0.1

   real, dimension(n) :: xb,xh,work2
   real, dimension(n,nk) :: c,x,x1,wk2d
   real, dimension(n,4*nk) :: b

   real, dimension(n) :: pbl_kcl
   real, dimension(PBL_NK) :: pbl_std_p_prof
   real, dimension(n), target :: pbl_zmnk
   real, dimension(n,nk) :: presw,prese,tke
   real, dimension(n,nk+1) :: presm
   real, dimension(n,PBL_NK) :: pbl_s,pbl_se,pbl_prese,pbl_enold,  &
        pbl_lwc,pbl_ftot,pbl_ze,pbl_te,pbl_qe,pbl_qce, &
        pbl_tve,pbl_fbl,pbl_zd,pbl_rif,pbl_rig,pbl_shear2
   real, dimension(n,PBL_NK+1) :: pbl_presm,pbl_zm
   real, dimension(n,4*PBL_NK) :: pbl_b

   real CF1,CF2, ETURBTAU, ZNTEM, uet,ilmot,fhz,fim,fit, hst_local
   integer I,K, IERGET

   ! fonction-formule
   integer ik
   include "dintern.inc"
   include "fintern.inc"
   ik(i,k) = (k-1)*n + i -1
  

  ! Check for required lift of last thermodynamic level
  if (tlift .ne. 1) then
     print*, 'STOP in pbl_turbul(): Schm_Tlift=1 must be set for PBL coupling'
     stop
  endif

  eturbtau=delt
  if (advectke) then
     if (kount.gt.1) then
        eturbtau=factdt*delt
     endif
  endif

  !  FILTRE TEMPOREL 

  CF1=1.0-TFILT
  CF2=TFILT
  !
  !  INITIALISER E AVEC EA,Z ET H
  !
  if (KOUNT.eq.0) then
     do K=1,NK
        !VDIR NODEP
        do I=1,N
           TKE(I,K)= max( ETRMIN, BLCONST_CU*F(FRV+(indx_agrege-1)*N+I-1)**2 * &
                exp(-(V(ZE+ik(I,K))- V(ZE+ik(I,NK)))/F(H+I-1)) )
        end do
     end do
  endif
  !
  !
  if (KOUNT.gt.0) then
     call serxst2(F(ZN), 'LM', TRNCH, N, nk, 0.0, 1.0, -1)
     call serxst2(TKE  , 'EN', TRNCH, N, nk, 0.0, 1.0, -1)
  endif

  !  COMPUTE COORDINATE VALUES AND PRESSURES FOR PBL SCHEME
  call pbl_coord(pbl_s,pbl_se,d(sigm),se,PBL_KTOP,PBL_ZSPLIT,n,nk,PBL_NK)
  do k=1,nk
     do i=1,n
        presm(i,k) = d(sigm+ik(i,k)) * f(pmoins+i-1)
        presw(i,k) = d(sigw+ik(i,k)) * f(pmoins+i-1)
        prese(i,k) = se(i,k) * f(pmoins+i-1)
     enddo
  enddo
  do i=1,n !need a "momentum level" at the surface for mixing length calculation (mixlen3)
     presm(i,nk+1) = f(pmoins+i-1)
  enddo
  do k=1,PBL_NK
     do i=1,n
        pbl_presm(i,k) = pbl_s(i,k) * f(pmoins+i-1)
        pbl_prese(i,k) = pbl_se(i,k) * f(pmoins+i-1)
     enddo
  enddo
  do i=1,n
     pbl_presm(i,PBL_NK+1) = f(pmoins+i-1)
  enddo

  !     ADAPT LOW RESOLUTION PROFILE TO HIGH-RESOLUTION GRID
  if (kount == 0) then
     call vte_intvertx3(f(pbl_umoins),d(umoins),presm,pbl_presm,n,nk,pbl_nk,'UU','cubic')
     call vte_intvertx3(f(pbl_vmoins),d(vmoins),presm,pbl_presm,n,nk,pbl_nk,'VV','cubic')
     call vte_intvertx3(f(pbl_tmoins),d(tmoins),presw,pbl_presm,n,nk,pbl_nk,'TT','cubic')
     call vte_intvertx3(f(pbl_humoins),d(humoins),presw,pbl_presm,n,nk,pbl_nk,'HU','linear')
     call vte_intvertx3(f(pbl_qcmoins),d(qcmoins),presw,pbl_presm,n,nk,pbl_nk,'QC','linear')
     call vte_intvertx3(f(pbl_tke),tke,prese,pbl_prese,n,nk,pbl_nk,'EN','cubic')
     do k=1,nk
        do i=1,n
           f(pblm_umoins+ik(i,k)) = d(umoins+ik(i,k))
           f(pblm_vmoins+ik(i,k)) = d(vmoins+ik(i,k))
           f(pblm_tmoins+ik(i,k)) = d(tmoins+ik(i,k))
           f(pblm_humoins+ik(i,k)) = d(humoins+ik(i,k))
           f(pblm_qcmoins+ik(i,k)) = d(qcmoins+ik(i,k))
        enddo
     enddo
  else
     do k=1,nk
        do i=1,n
           f(pblm_humoins+ik(i,k)) = max(0.,f(pblm_humoins+ik(i,k)))
        enddo
     enddo
     call ccpl_increment(f(pbl_umoins),f(pbl_umoins),d(umoins),f(pblm_umoins),pbl_presm,presm,n,pbl_nk,nk,'UU','cubic')
     call ccpl_increment(f(pbl_vmoins),f(pbl_vmoins),d(vmoins),f(pblm_vmoins),pbl_presm,presm,n,pbl_nk,nk,'VV','cubic')
     call ccpl_increment(f(pbl_tmoins),f(pbl_tmoins),d(tmoins),f(pblm_tmoins),pbl_presm,presw,n,pbl_nk,nk,'TT','cubic')
     call ccpl_increment(f(pbl_humoins),f(pbl_humoins),d(humoins),f(pblm_humoins),pbl_presm,presw,n,pbl_nk,nk,'HU','linear')
     call ccpl_increment(f(pbl_qcmoins),f(pbl_qcmoins),d(qcmoins),f(pblm_qcmoins),pbl_presm,presw,n,pbl_nk,nk,'QC','linear')
     do k=1,pbl_nk
        do i=1,n
           f(pbl_humoins+ik(i,k)) = max(0.,f(pbl_humoins+ik(i,k)))
        enddo
     enddo
  endif

  !  INTERPOLATE TO HIGH VERTICAL RESOLUTION FOR THE PBL
  ! Momentum level fields
  call vte_intvertx3(pbl_zm,v(gzmom),presm,pbl_presm,n,nk+1,pbl_nk+1,'Z','linear')
  call vte_intvertx3(pbl_lwc,f(lwc),presw,pbl_presm,n,nk,pbl_nk,'LWC','linear')
  call vte_intvertx3(pbl_ftot,f(ftot),presw,pbl_presm,n,nk,pbl_nk,'NT','linear')
  call vte_intvertx3(pbl_std_p_prof,std_p_prof,presm,pbl_presm,1,nk,pbl_nk,'PX','linear')
  ! Energy level fields
  call vte_intvertx3(pbl_ze,v(ze),prese,pbl_prese,n,nk,pbl_nk,'Z','linear')
  call vte_intvertx3(pbl_te,f(pbl_tmoins),pbl_presm,pbl_prese,n,pbl_nk,pbl_nk,'TT','cubic')
  call vte_intvertx3(pbl_qe,f(pbl_humoins),pbl_presm,pbl_prese,n,pbl_nk,pbl_nk,'HU','linear')
  call vte_intvertx3(pbl_qce,f(pbl_qcmoins),pbl_presm,pbl_prese,n,pbl_nk,pbl_nk,'QC','linear')
  ! Diagnostic quantities
  do i=1,n
     do k=1,pbl_nk
        pbl_enold(i,k) = f(pbl_tke+ik(i,k))
     enddo
  enddo
  call mfotvt(pbl_tve,pbl_te,pbl_qe,n,pbl_nk,n)

  !     Convective velocity scale w* (passed to MOISTKE3 through XH)
  do I=1,N
     XB(I)=1.0+DELTA*QE(I,NK)
     XH(I)=(GRAV/(XB(I)*V(TVE+ik(I,NK))))* &
          ( XB(I)*F(FTEMP+(indx_agrege-1)*N+I-1) &
          + DELTA*V(TVE+ik(I,NK))*F(FVAP+(indx_agrege-1)*N+I-1) )
     XH(I)=max(0.0,XH(I))
     WORK2(I)=F(H+I-1)*XH(I)
  end do
  call VSPOWN1 (XH,WORK2,1./3.,N)
  
  do i=1,n
     v(wstar+i-1) = xh(i)
  enddo
  

  if (fluvert=='MOISTKE') then

     print*, 'STOP in pbl_turbul(): moistke not implemented for PBL_COUPLED=.true.'
     stop

  else

     call eturbl9( f(pbl_tke),pbl_enold,f(pbl_zn),pbl_zd,pbl_rif,f(pbl_turbreg), &
          pbl_rig,pbl_shear2,v(pbl_gte),f(ilmo+(indx_agrege-1)*n),pbl_fbl, &
          v(pbl_gql),pbl_lwc,f(pbl_umoins),f(pbl_vmoins),f(pbl_tmoins),pbl_te, &
          pbl_tve,f(pbl_humoins),pbl_qce,pbl_qe,f(h),f(pmoins), &
          f(tsurf),pbl_s,pbl_se,eturbtau,kount,v(pbl_gq),pbl_ftot, &
          v(pbl_kt),pbl_ze,pbl_zm,pbl_kcl,pbl_std_p_prof,F(frv+(indx_agrege-1)*n),xh, &
          trnch,n,pbl_nk,f(z0+(indx_agrege-1)*n),it)

     !     Implicit diffusion scheme: the diffusion interprets the coefficients of the
     !     surface boundary fluxe condition as those in the ALFAT+BT*TA expression.
     !     Since the diffusion is made on potential temperature, there is a correction
     !     term that must be included in the non-homogeneous part of the expression - 
     !     the alpha coefficient. The correction is of the form za*g/cp. It is 
     !     relevant only for the CLEF option (as opposed to the MOISTKE option) since
     !     in this case, although the diffusion is made on potential temperature, 
     !     the diffusion calculation takes an ordinary temperature as the argument.
     !
     if (IMPFLX) then
        do I=1,N
           V(ALFAT+I-1) = V(ALFAT+I-1) + V(BT+(indx_agrege-1)*N+I-1)*V(ztsl+I-1)*GRAV/CPD
        end do
     endif
     !
  endif
  !
  !    Diagnose variables for turbulent wind (gusts and standard deviations)
  !
  if (Diag_twind) then
     print*, 'STOP in pbl_turbul(): Diag_twind not implemented for PBL_COUPLED=.true.'
     stop
  endif
  !
  !
  !     FILTRE VERTICAL SUR EN
  !
  call pbl_sfltr ( f(pbl_tke), f(pbl_tke), 0.1, N, pbl_nk-1 )
  !
  !
  !  ------------------------------------------------------
  !    HAUTEUR DE LA COUCHE LIMITE STABLE OU INSTABLE
  !  ------------------------------------------------------
  !
  !     Ici HST est la hauteur calculee dans la routine FLXSURF1.
  !
  !     KCL contient le K que l'on a diagnostique dans RIGRAD
  !     et passe a ETURBL3; il pointe vers le premier niveau
  !     de la couche limite.
  !
  !     SCL est utilise comme champ de travail dans la boucle 100;
  !     il est mis a jour dans la boucle 200, et donne la hauteur
  !     de la couche limite en sigma.
  !
  !
  !VDIR NODEP
  do I=0,N-1

     if(f(ilmo+(indx_agrege-1)*n+i).gt.0.0) then
        !         Cas stable
        f(scl+i) = f(hst+(indx_agrege-1)*n+i)
     else
        !         Cas instable: max(cas neutre, diagnostic)
        !         Z contient les hauteurs des niveaux intermediaires
        f(scl+i) = max( f(hst+(indx_agrege-1)*n+i) ,  &
             v(ze +ik(i+1,nint(v(kcl+i)))))
     endif

     !       Si H est en train de chuter, on fait une relaxation pour
     !       ralentir la chute.

     if(f(h+i) .gt. f(scl+i)) then
        f(h+i) = f(scl+i) + (f(h+i)-f(scl+i))*exp(-delt/5400.)
     else
        f(h+i) = f(scl+i)
     endif

  enddo

  !     On calcule SCL avec une approximation hydrostatique
  do i=0,n-1
     f(scl+i)=-grav*f(h+i)/(rgasd*d(tmoins+ik(i+1,nk)))
  enddo
  call vsexp(f(scl),f(scl),n)

  call serxst2(f(H)  , 'F2', TRNCH, N, 1, 0.0, 1.0, -1)
  call serxst2(F(SCL), 'SE', TRNCH, N, 1, 0.0, 1.0, -1)


  do K=1,pbl_nk-1
     !VDIR NODEP
     do I=1,N
        !                                                 KM
        !           IBM CONV. ; PAS D'AVANTAGE A PRECALCULER SQRT CI-DESSOUS
        V(pbl_km+ik(I,K))=BLCONST_CK*F(pbl_zn+ik(i,k))*sqrt(f(pbl_tke+ik(i,k)))
        !
        !                                                 KT
        v(pbl_kt+ik(i,k)) = v(pbl_km+ik(I,K))*v(pbl_kt+ik(i,k))
        !
     end do
  end do
  !
  !     CALCULATE SURFACE PARAMETERS
  do i=1,n
     uet=f(frv+(indx_agrege-1)*n+i-1)
     ilmot=f(ilmo+ik(i,indx_agrege))
     if(ilmot.gt.0.) then
        !         hst_local is used to avoid division by zero         
        hst_local = max( f(hst+ik(i,indx_agrege) ), (v(ztsl+i-1)+1.) )
        fhz=1-v(ztsl+i-1)/hst_local
        fim=0.5*(1+sqrt(1+4*as*v(ztsl+i-1)*ilmot*beta/fhz))
        fit=beta*fim
     else
        fim=(1-ci*v(ztsl+i-1)*ilmot)**(-.16666666)
        fit=beta*fim**2
        fhz=1
     endif
  enddo
  !
  !     FILL IN LOWER-LEVEL PBL LEVELS
  do i=1,n
     if (fluvert /= 'MOISTKE') &
          f(pbl_tke+ik(i,pbl_nk)) = f(pbl_tke+ik(i,pbl_nk-1))
     f(pbl_tke+ik(i,pbl_nk+1)) = f(pbl_tke+ik(i,pbl_nk))
     f(pbl_zn+ik(i,pbl_nk)) = karman*(v(ztsl+i-1)+f(z0+ik(i,indx_agrege)))/fim
     v(pbl_km+ik(I,pbl_nk)) = uet*f(pbl_zn+ik(i,pbl_nk))*fhz
     v(pbl_kt+ik(I,pbl_nk)) = v(pbl_km+ik(i,pbl_nk))*fim/fit
     f(pbl_zn+ik(i,pbl_nk+1)) = karman*f(z0+ik(i,indx_agrege))
     v(pbl_km+ik(I,pbl_nk+1)) = uet*f(pbl_zn+ik(i,pbl_nk+1))
     v(pbl_kt+ik(I,pbl_nk+1)) = v(pbl_km+ik(I,pbl_nk+1))/beta
  enddo

  !     INTERPOLATE OUTPUTS BACK TO LOW-RESOLUTION VERTICAL GRID
  call vte_intvertx3(f(en),f(pbl_tke),pbl_prese,prese,n,pbl_nk,nk,'EN','linear')
  call vte_intvertx3(f(zn),f(pbl_zn),pbl_prese,prese,n,pbl_nk,nk,'ZN','linear')
  call vte_intvertx3(f(zd),pbl_zd,pbl_prese,prese,n,pbl_nk,nk,'ZD','linear')
  call vte_intvertx3(f(fbl),pbl_fbl,pbl_prese,prese,n,pbl_nk,nk,'FBL','linear')
  call vte_intvertx3(v(km),v(pbl_km),pbl_prese,prese,n,pbl_nk,nk,'KM','linear')
  call vte_intvertx3(v(kt),v(pbl_kt),pbl_prese,prese,n,pbl_nk,nk,'KT','linear')
  call vte_intvertx3(v(gte),v(pbl_gte),pbl_prese,prese,n,pbl_nk,nk,'GAMA','linear')
  call vte_intvertx3(v(gql),v(pbl_gql),pbl_prese,prese,n,pbl_nk,nk,'GAMAL','linear')
  call vte_intvertx3(v(gq),v(pbl_gq),pbl_prese,prese,n,pbl_nk,nk,'GAMAQ','linear')
  call vte_intvertx3(v(rif),pbl_rif,pbl_prese,prese,n,pbl_nk,nk,'RI','linear')
  call vte_intvertx3(v(rig),pbl_rig,pbl_prese,prese,n,pbl_nk,nk,'RI','linear')
  call vte_intvertx3(v(shear2),pbl_shear2,pbl_prese,prese,n,pbl_nk,nk,'UU','linear')

  !     RETRIEVE PBL TOP INDEX
  do i=1,n
     v(kcl+i-1) = nint(pbl_kcl(i))/(2*PBL_ZSPLIT) + 1
  enddo

  !     TIME SERIES OUTPUTS OF PHYSICS-LEVELED FIELDS
  if (KOUNT.eq.0) then
     call serxst2(F(ZN), 'LM', TRNCH, N, nk, 0.0, 1.0, -1)
     call serxst2(TKE  , 'EN', TRNCH, N, nk, 0.0, 1.0, -1)
  endif

  !     FILL IN LOWER-LEVEL LOW-RESOLUTION MODEL LEVELS
  do i=1,n
     if (fluvert /= 'MOISTKE') &
          f(en+ik(i,nk))=f(en+ik(i,nk-1))
     f(en+ik(i,nk+1))=f(en+ik(i,nk))
     F(ZN+ik(i,nk))=karman*(v(ztsl+i-1)+f(z0+ik(i,indx_agrege)))/fim
     V(KM+ik(I,NK))=uet*F(ZN+ik(i,nk))*fhz
     V(KT+ik(I,NK))=V(KM+ik(I,NK))*fim/fit
     F(ZN+ik(i,nk+1))=karman*f(z0+ik(i,indx_agrege))
     V(KM+ik(I,NK+1))=uet*F(ZN+ik(i,nk+1))
     V(KT+ik(I,NK+1))=V(KM+ik(I,NK+1))/beta
  enddo

  return
end subroutine pbl_turbul
