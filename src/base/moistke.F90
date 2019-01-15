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

      subroutine MOISTKE6(EN,ENOLD,ZN,ZD,RIF,RIG,SHR2,KT,QC,FRAC,FNN, &
                          GAMA,GAMAQ,GAMAL,H, &
                          U,V,T,TVE,Q,QE,PS,S,SE,SW, &
                          AT2T,AT2M,AT2E, &
                          ZE,C,B,X,NKB,TAU,KOUNT, &
                          Z,Z0,GZMOM,KCL,FRV,TURBREG, &
                          X1,XB,XH,TRNCH,N,M,NK,IT)
      use phy_options
      implicit none
#include <arch_specific.hf>
      integer N,M,NK,i
      integer NKB,KOUNT
      integer IT,TRNCH
      real EN(N,NK),ENOLD(N,NK),ZN(N,NK),ZD(N,NK),KT(N,NK)
      real RIF(N,NK),RIG(N,NK),SHR2(N,NK),TURBREG(N,NK)
      real QC(N,NK),QE(N,NK),FRAC(N,NK),FNN(N,NK)
      real GAMA(N,NK),GAMAQ(N,NK),GAMAL(N,NK),H(N)
      real U(M,NK),V(M,NK)
      real T(N,NK),TVE(N,NK),Q(N,NK),PS(N)
      real S(N,NK),SE(N,NK),SW(N,NK),AT2T(n,NK),AT2M(n,NK),AT2E(n,NK)
      real ZE(N,NK),C(N,NK),B(N,NKB),X(N,NK)
      real TAU
      real Z(N,NK),Z0(N),GZMOM(N,NK)
      real KCL(N),FRV(N)
      real X1(N,NK)
      real XB(N),XH(N)
!
!Author
!          J. Mailhot (Nov 2000)
!
!Revision
! 001      J. Mailhot (Jun 2002) Add cloud ice fraction
!                      Change calling sequence and rename MOISTKE1
! 002      J. Mailhot (Feb 2003) Add boundary layer cloud content
!                      Change calling sequence and rename MOISTKE2
! 003      A. Plante  (May 2003) IBM conversion
!                        - calls to exponen4 (to calculate power function '**')
!                        - divisions replaced by reciprocals (call to vsrec from massvp4 library)
! 004      B. Bilodeau (Aug 2003) exponen4 replaced by vspown1
!                                 call to mixlen2
! 005      Y. Delage (Sep 2004) Replace UE2 by FRV and rename subroutine. Introduce log-linear
!                   stability function in mixing length for near-neutral cases.  Perform
!                    optimisation in calcualtion of KT
! 006     A-M. Leduc (June 2007) Add z0 argument, moistke3-->moistke4.
!                                 Z0 was missing in calculation of ZN.
! 007      L. Spacek (Dec 2007) - add "vertical staggering" option
!                                 correction FITI=BETA*FITI, limit ZN < 5000
!
!Object
!          Calculate the turbulence variables (TKE, mixing length,...)
!          for a partly cloudy boundary layer, in the framework of a
!          unified turbulence-cloudiness formulation.
!          Uses moist conservative variables (thetal and qw), diagnostic
!          relations for the mixing and dissipation lengths, and a predictive
!          equation for moist TKE.
!
!
!Arguments
!
!          - Input/Output -
! EN       turbulent energy
! ZN       mixing length of the turbulence
! ZD       dissipation length of the turbulence
!
!          - Output -
! RIF      flux Richardson number
! RIG      gradient Richardson number
! SHR2     square of wind shear
!
!          - Input -
! ENOLD    turbulent energy (at time -)
! QC       boundary layer cloud water content
! FRAC     cloud fraction (computed in BAKTOTQ2)
!          - Output -
! FRAC     constant C1 in second-order moment closure (used by CLSGS)
!
!          - Input -
! FNN      flux enhancement factor (computed in BAKTOTQ2)
! GAMA     countergradient term in the transport coefficient of theta
! GAMAQ    countergradient term in the transport coefficient of q
! GAMAL    countergradient term in the transport coefficient of ql
! H        height of the the boundary layer
!
!          - Input -
! U        east-west component of wind
! V        north-south component of wind
! T        temperature
! TVE      virtual temperature on 'E' levels
! Q        specific humidity
! QE       specific humidity on 'E' levels
!
!          - Input -
! PS       surface pressure
! S        sigma level
! SE       sigma level on 'E' levels
! SW       sigma level on working levels
! AT2T     coefficients for interpolation of T,Q to thermo levels
! AT2M     coefficients for interpolation of T,Q to momentum levels
! AT2E     coefficients for interpolation of T,Q to energy levels
! TAU      timestep
! KOUNT    index of timestep
! KT       ratio of KT on KM (real KT calculated in DIFVRAD)
! Z        height of sigma level
! Z0       roughness length
! GZMOM    height of sigma momentum levels
!
!          - Input/Output -
! KCL      index of 1st level in boundary layer
!
!          - Input -
! FRV      friction velocity
! ZE       work space (N,NK)
! C        work space (N,NK)
! B        work space (N,NKB)
! X        work space (N,NK)
! X1       work space (N,NK)
! XB       work space (N)
! XH       work space (N)
! NKB      second dimension of work field B
! TRNCH    number of the slice
! N        horizontal dimension
! M        1st dimension of T, Q, U, V
! NK       vertical dimension
! IT       number of the task in muli-tasking (1,2,...) =>ZONXST
!
!Notes
!          Refer to J.Mailhot and R.Benoit JAS 39 (1982)Pg2249-2266
!          and Master thesis of J.Mailhot.
!          Mixing length formulation based on Bougeault and Lacarrere .....
!          Subgrid-scale cloudiness scheme appropriate for TKE scheme
!          based on studies by Bechtold et al:
!          - Bechtold and Siebesma 1998, JAS 55, 888-895
!          - Cuijpers and Bechtold 1995, JAS 52, 2486-2490
!          - Bechtold et al. 1995, JAS 52, 455-463
!

include "thermoconsts.inc"

#include "clefcon.cdk"
#include "surface.cdk"
#include "machcon.cdk"
#include "tables.cdk"
#include "phyinput.cdk"

      external DIFUVDFJ
      external  BLCLOUD3, TKEALG

      real EXP_TAU
      integer IERGET,STAT,NEARK
      external NEARK
!
!
!********************** AUTOMATIC ARRAYS
!
      real ZNOLD(N,NK)
!
!
!*
!
!
      real FIMS,PETIT,BETAI,dif_tau
      integer J,K
!
      integer type
!
!------------------------------------------------------------------------
!
      real ICAB,C1,LMDA
      save ICAB,C1,LMDA
      data ICAB, C1, LMDA  / 0.4, 0.32, 200. /
!***********************************************************************
!     AUTOMATIC ARRAYS
!***********************************************************************
!
      integer, dimension(N) :: SLK
      real, dimension(N,NK) :: FIMI
      real*8, dimension(N,NK) :: FIMIR
      real, dimension(N,NK) :: FITI
      real*8, dimension(N,NK) :: FITIR
      real*8, dimension(N,NK) :: FITSR
      real, dimension(N,NK) :: WORK
      real, dimension(N,NK) :: TE
      real, dimension(N,NK) :: QCE
!
      type=4
      PETIT=1.E-6
!
!      0.     Keep the mixing lenght zn from the previous time step
!      ------------------------------------------------------------
!
       ZNOLD(:,:)  = ZN(:,:)
!
!
!
!      1.     Preliminaries
!      --------------------
!
!
      if(KOUNT.eq.0) then
        do K=1,NK
        do J=1,N
          ZN(J,K)=min(KARMAN*(Z(J,K)+Z0(J)),LMDA)
          ZD(J,K)=ZN(J,K)
        end do
        end do
        if (.not.any('qtbl' == phyinread_list_s(1:phyinread_n))) &
             QC(:,:) = 0.0
        if (.not.any('fnn' == phyinread_list_s(1:phyinread_n))) &
             FNN(:,:) = 0.0
        if (.not.any('fbl' == phyinread_list_s(1:phyinread_n))) &
             FRAC(:,:) = 0.0
      endif
!
!
!      2.     Boundary layer cloud properties
!      --------------------------------------
!
!
      call BLCLOUD3 (U, V, T, TVE, Q, QC, FNN, &
                     S, SW,PS, SHR2, RIG, X, &
                     AT2M,AT2E, &
                     N, M, NK)
!
!
!                                GAMA terms set to zero
!                                (when formulation uses conservative variables)
      do K=1,NK
      do J=1,N
         GAMA(J,K)=0.0
         GAMAQ(J,K)=0.0
         GAMAL(J,K)=0.0
      end do
      end do
!
!
      do K=1,NK-1
      do J=1,N
!                                top of the unstable PBL (from top down)
        if( X(J,K).gt.0. ) KCL(J) = K
      end do
      end do
!
!
      call serxst2(RIG, 'RI', TRNCH, N, nk, 0., 1., -1)
!
      call serxst2(RIG, 'RM', TRNCH, N, nk, 0., 1., -1)
!
      do K=1,NK
      do J=1,N
         WORK(J,K)=1-CI*min(RIG(J,K),0.)
      enddo
      enddo
      call VSPOWN1 (FIMI,WORK,-1/6.,N*NK)
      call VSPOWN1 (FITI,WORK,-1/3.,N*NK)
      FITI=BETA*FITI
      FITIR=FITI
      FIMIR=FIMI
      call VREC(FITIR,FITIR,N*NK)
      call VREC(FIMIR,FIMIR,N*NK)
      BETAI=1/BETA
      do K=1,NK
      do J=1,N
         FIMS=min(1+AS*max(RIG(J,K),0.),1/max(PETIT,1-ASX*RIG(J,K)))
         ZN(J,K)=min(KARMAN*(Z(J,K)+Z0(J)),LMDA)
         if( RIG(J,K).ge.0.0 ) then
           ZN(J,K)=ZN(J,K)/FIMS
         else
           ZN(J,K)=ZN(J,K)*FIMIR(J,K)
           ZN(J,K)=min(ZN(J,K),5000.)
         endif
!
!                                KT contains the ratio KT/KM (=FIM/FIT)
!
         if(RIG(J,K).ge.0.0) then
           KT(J,K)=BETAI
         else
            KT(J,K)=FIMI(J,K)*FITIR(J,K)
         endif
      end do
      end do
!
!                                From gradient to flux form of buoyancy flux
!                                and flux Richardson number (for time series output)
      do K=1,NK
      do J=1,N
         X(J,K)=KT(J,K)*X(J,K)
         RIF(J,K)=KT(J,K)*RIG(J,K)
!                                Computes constant C1
         FRAC(J,K)=2.0*BLCONST_CK*KT(J,K)*ICAB
      end do
      end do
!
!     EXPAND NEUTRAL REGIME
      stat = neark(se,ps,3000.,n,nk,slk) !determine "surface layer" vertical index
      if (kount == 0) then
         INIT_TURB: if (.not.any('turbreg'==phyinread_list_s(1:phyinread_n))) then
            do k=1,nk
               do j=1,n
                  if (k <= slk(j)) then
                     if (RIF(j,k) > PBL_RICRIT(1)) then
                        TURBREG(j,k) = LAMINAR
                     else
                        TURBREG(j,k) = TURBULENT
                     endif
                  else
                     TURBREG(j,k) = TURBULENT
                  endif
               enddo
            enddo
         endif INIT_TURB
      endif
      do k=1,nk             !do not apply to the lowest level (-2)
         if (std_p_prof(k) < 60000.) cycle
         do j=1,n
            ABOVE_SFCLAYER: if (k <= slk(j)) then
               if (RIF(j,k) < PBL_RICRIT(1)) then
                  TURBREG(j,k) = TURBULENT
               elseif (RIF(j,k) > PBL_RICRIT(2)) then
                  TURBREG(j,k) = LAMINAR
               endif
               ! Neutral regime: set buoyant suppression to mechanical generation (cute: FIXME)
               if (RIF(j,k) > PBL_RICRIT(1) .and. RIF(j,k) < 1. .and. nint(TURBREG(j,k)) == LAMINAR) X(J,K) = SHR2(J,K)
               if (RIF(j,k) > 1. .and. RIF(j,k) < PBL_RICRIT(2) .and. nint(TURBREG(j,k)) == TURBULENT) X(J,K) = SHR2(J,K)
            endif ABOVE_SFCLAYER
         enddo
      enddo

      call serxst2(RIF, 'RF', TRNCH, N, nk, 0.0, 1.0, -1)

!      3.     Mixing and dissipation length scales
!      -------------------------------------------
!
!
!                                Compute the mixing and dissipation lengths
!                                according to Bougeault and Lacarrere (1989)
!                                and Belair et al (1999)
!
      call VSPOWN1 (X1,SE,-CAPPA,N*NK)
!                                Virtual potential temperature (THV)
!
      call TOTHERMO(T, TE,  AT2T,AT2M,N,NK+1,NK,.true.)
      call TOTHERMO(QC,QCE, AT2T,AT2M,N,NK+1,NK,.true.)

      X1(:,:)=TE(:,:)*(1.0+DELTA*QE(:,:)-QCE(:,:))*X1(:,:)

      if (longmel == 'BOUJO') then
         if (TMP_BOUJO_HEIGHT_CORR) then
            call mixlen3(zn,x1,enold,z,h,s,ps,n,nk)
         else
            call MIXLEN3( ZN, X1, ENOLD, GZMOM(1,2), H, S, PS, N, NK)
         endif
      endif

!     No time filtering at kount=0 since this step is used for initialization only
      if(KOUNT == 0)then
         if (any('zn'==phyinread_list_s(1:phyinread_n))) zn = znold
      else     
         ZN_TIME_RELAXATION: if (PBL_ZNTAU > 0.) then
            EXP_TAU = exp(-TAU/PBL_ZNTAU)
            ZN = ZN + (ZNOLD-ZN)*EXP_TAU
         endif ZN_TIME_RELAXATION
      end if

      if (longmel == 'BLAC62') then
        ZE(:,:) = max(ZN(:,:),1.E-6)
      else if(longmel == 'BOUJO') then
        ZE(:,:) = ZN(:,:) * ( 1. - min( RIF(:,:) , 0.4) ) &
                  / ( 1. - 2.*min( RIF(:,:) , 0.4) )
        ZE(:,:) = max ( ZE(:,:) , 1.E-6 )
      end if

      if (PBL_DISS == 'LIM50') ze = min(ze,50.)

      ZD(:,:) = ZE(:,:)



      call serxst2(ZN, 'L1', TRNCH, N, nk, 0.0, 1.0, -1)
      call serxst2(ZD, 'L2', TRNCH, N, nk, 0.0, 1.0, -1)

      call serxst2(ZD, 'LE', TRNCH, N, nk, 0.0, 1.0, -1)


!      4.     Turbulent kinetic energy
!      -------------------------------

      if(KOUNT.eq.0)then

        do K=1,NK
        do J=1,N
           X(J,K)=0.0
        end do
        end do

        call serxst2(X, 'EM', TRNCH, N, nk, 0.0, 1.0, -1)
        call serxst2(X, 'EB', TRNCH, N, nk, 0.0, 1.0, -1)
        call serxst2(X, 'ED', TRNCH, N, nk, 0.0, 1.0, -1)
        call serxst2(X, 'ET', TRNCH, N, nk, 0.0, 1.0, -1)
        call serxst2(X, 'ER', TRNCH, N, nk, 0.0, 1.0, -1)

      else

!                                Solve the algebraic part of the TKE equation
!                                --------------------------------------------
!
!                                Put dissipation length in ZE (work array)
      do K=1,NK
      do J=1,N
         ZE(J,K) = ZD(J,K)
      end do
      end do

         ! The original was incorrect since the vectors are not conformal
         ! This has no impact on the integration but made -g -C option complain
         !B = SHR2
         B(1:n,1:nk) = SHR2(1:n,1:nk)
         call TKEALG(C,EN,ZN,ZE,B,X,TAU,N,NK)

!                                Mechanical production term
         call serxst2(B, 'EM', TRNCH, N, NKB, 0.0, 1.0, -1)
!                                Thermal production term
         call serxst2(X, 'EB', TRNCH, N, NK, 0.0, 1.0, -1)
!                                Viscous dissipation term
         call serxst2(ZE, 'ED', TRNCH, N, NK, 0.0, 1.0, -1)

!                                Solve the diffusion part of the TKE equation
!                                --------------------------------------------
!                                (uses scheme i of Kalnay-Kanamitsu 1988 with
!                                 double timestep, implicit time scheme and time
!                                 filter with coefficient of 0.5)

         do K=1,NK
         do J=1,N
!                                X contains (E*-EN)/TAU
            X(J,K)=(C(J,K)-EN(J,K))/TAU
!                                ZE contains E*
            ZE(J,K)=C(J,K)
!                                C contains K(EN) with normalization factor
            C(J,K) = ( (GRAV/RGASD)*SE(J,K)/TVE(J,K) )**2
            C(J,K)=BLCONST_CK*CLEFAE*ZN(J,K)*sqrt(ENOLD(J,K))*C(J,K)
!                                countergradient and inhomogeneous terms set to zero
            X1(J,K)=0.0
         end do
         end do
!
         if( type.eq.4 ) then
!                                surface boundary condition
           do J=1,N
             XB(J)=BLCONST_CU*FRV(J)**2 + BLCONST_CW*XH(J)**2
             ZE(J,NK)=XB(J)
           end do
!
         endif
!
         dif_tau = TAU
         if(PBL_TKEDIFF2DT) dif_tau = 2.*TAU
         call DIFUVDFJ (EN,ZE,C,X1,X1,XB,XH,S,SE,dif_tau,type,1., &
                        B(1,1),B(1,NK+1),B(1,2*NK+1),B(1,3*NK+1), &
                        N,N,N,NK)
!
         ! New TKE
         en = max(ETRMIN,ze+tau*en)

      endif
!
!
      return
      end
