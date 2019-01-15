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
subroutine prep_cw_rad2 (f, fsiz, d, dsiz, v, vsiz, &
     tm,qm,ps,sigma,cloud, &
     liqwcin,icewcin,liqwpin,icewpin, &
     trav2d,seloc, &
     kount, trnch, task, ni, m, nk)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>

   integer fsiz, dsiz, vsiz, ni, m, nk, nkp
   integer kount, trnch, task
   real, target ::  f(fsiz), d(dsiz), v(vsiz)
   real tm(m,nk), qm(m,nk), ps(ni),sigma(ni,nk)
   real liqwcin(ni,nk), icewcin(ni,nk)
   real liqwpin(ni,nk), icewpin(ni,nk)
   real cloud(ni,nk), trav2d(ni,nk), seloc(ni,nk)
   real press


   !@Author L. Spacek (Oct 2004)

   !@Object  Prepare liquid/ice water contents and cloudiness for the radiation
   !@Arguments
   !     - input -
   !     dsiz     dimension of d
   !     fsiz     dimension of f
   !     vsiz     dimension of v
   !     tm       temperature
   !     qm       specific humidity
   !     ps       surface pressure
   !     sigma    sigma levels
   !     kount    index of timestep
   !     trnch    number of the slice
   !     task     task number
   !     n        horizontal dimension
   !     m        1st dimension of tm and qm
   !     nk       number of layers
   !     - output -
   !     liqwcin  in-cloud liquid water content
   !     icewcin  in-cloud ice    water content
   !     liqwpin  in-cloud liquid water path (g/m^2)
   !     icewpin  in-cloud ice    water path (g/m^2)
   !     cloud    cloudiness passed to radiation
   !*@/
   include "thermoconsts.inc"
   include "phyinput.cdk"
   include "surface.cdk"
   include "nocld.cdk"

   integer, parameter :: ioptpart = 2
   real, dimension(ni,nk) :: c3d, frac, lwcth, tcel, vtcel, vliqwcin

   integer i, j, k
   real dp,lwcm1,iwcm1,zz,rec_grav
   logical strcld,nostrlwc

   real, pointer, dimension(:,:) :: zfbl, zfdc, zftot, ziwc, zlwc, zqcplus, &
   	 zqiplus, zsnow, zqi_cat1, zqi_cat2, zqi_cat3, zqi_cat4

   zfbl   (1:ni,1:nk) => f(fbl:)
   zfdc   (1:ni,1:nk) => f(fdc:)
   zftot  (1:ni,1:nk) => f(ftot:)
   ziwc   (1:ni,1:nk) => f(iwc:)
   zlwc   (1:ni,1:nk) => f(lwc:)
   zqcplus(1:ni,1:nk) => d(qcplus:)

   rec_grav = 1./grav
   nkp      = nk+1
   nostrlwc = (climat.or.stratos)

   !     extracted from inichamp1

   IF_KOUNT0: if (kount == 0) then
      if (inilwc) then
         if (.not. any(trim(stcond) == (/'CONDS','NIL  '/))) then

            !     initialiser le champ d'eau nuageuse ainsi que la
            !     fraction nuageuse pour l'appel a la radiation a kount=0
            !     seulement.
            !     ces valeurs seront remplacees par celles calculees dans
            !     les modules de condensation.

            call cldwin(f(ftot),f(lwc),d(tmoins),d(humoins),f(pmoins), &
                 trav2d,d(sigw),ni,nk,satuco)
         endif
      endif
      !     Diagnostic initialization of QC has been suppressed (because
      !     QC has been read as an input).  ftot still comes from cldwin
      !     if inilwc==.true., but lwc is replaced by the input value here.
      IF_MP: if (stcond(1:2)=='MP')then
         if ( any(dyninread_list_s == 'QC').or.&
              any(phyinread_list_s(1:phyinread_n) == 'tr/mpqc:p') )then
            do k=1,nk
               do i=1,ni
                  zlwc(i,k) = zqcplus(i,k)
               enddo
            enddo
         endif

         if ( (stcond(1:6)=='MP_MY2').and. &
              (any(dyninread_list_s == 'MPQI') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/mpqi:p')) .and. &
              (any(dyninread_list_s == 'MPQS') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/mpqs:p')) )then
            zqiplus(1:ni,1:nk) => d(qiplus:)
            zsnow(1:ni,1:nk) => d(qnplus:)
            ziwc = zqiplus + zsnow

         elseif ( (stcond=='MP_P3') .and. &
              (any(dyninread_list_s == 'I1QT') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/i1qt:p')) )then
            zqi_cat1(1:ni,1:nk) => d(i1qtplus:)
            ziwc = zqi_cat1
            if ( (mp_p3_ncat >= 2).and. &
                 (any(dyninread_list_s == 'I2QT') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/i2qt:p')) )then
               zqi_cat2(1:ni,1:nk) => d(i2qtplus:)
               ziwc = ziwc + zqi_cat2
            endif
            if ( (mp_p3_ncat >= 3).and. &
                 (any(dyninread_list_s == 'I3QT') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/i3qt:p')) )then
               zqi_cat3(1:ni,1:nk) => d(i3qtplus:)
               ziwc = ziwc + zqi_cat3
            endif
            if ( (mp_p3_ncat >= 4).and. &
                 (any(dyninread_list_s == 'I4QT') .or. any(phyinread_list_s(1:phyinread_n) == 'tr/i4qt:p')) )then
               zqi_cat4(1:ni,1:nk) => d(i4qtplus:)
               ziwc = ziwc + zqi_cat4
            endif
         endif
         do k=1,nk
            do i=1,ni
               if (zlwc(i,k)+ziwc(i,k) .gt. 1.e-6)then
                  zftot(i,k)=1
               else
                  zftot(i,k)=0
               endif
            enddo
         enddo
      else  !# IF_MP
         if ( (any(dyninread_list_s == 'QC').or.&
              any(phyinread_list_s(1:phyinread_n) == 'tr/qc:p')).and. &
              .not.any('lwc'==phyinread_list_s(1:phyinread_n)) )then
            !#TODO: should it read :
            !# .not.(any(dyninread_list_s == 'QD').or.any('lwc'==phyinread_list_s(1:phyinread_n))) )then
            do k=1,nk
               do i=1,ni
                  zlwc(i,k) = zqcplus(i,k)
               enddo
            enddo
         endif
      endif IF_MP
   endif IF_KOUNT0

   !     For maximum of lwc (when using newrad) or Liquid water content when
   !     istcond=1 Always execute this part of the code to allow calculation
   !     of NT in calcNT

   call liqwc(lwcth,sigma,tm,ps,ni,nk,m,satuco)

   !     extracted from newrad3

   IF_CONDS: if (stcond=='CONDS') then

      !     Correct stratospheric clouds (bd, mars 1995)
      !     --------------------------------------------------
      strcld = .not.nostrlwc
      call nuages2 ( f(nhaut) , f(nmoy) , f(nbas) , &
           c3d, v(basc), qm, tm, ps, f(scl), &
           f(ilmo+(indx_agrege-1)*ni), sigma, &
           trnch, ni, m, nk, task, satuco, strcld)

      do k=1,nk
         do i=1,ni
            if (zfbl(i,k).gt.0.0) c3d(i,k) = 0.
            zfbl (i,k) = min(1.,c3d(i,k)+zfbl(i,k))
            zftot(i,k) = zfbl(i,k)
         enddo
      enddo
      do k=1,nk
         do i=1,ni
            if (sigma(i,k).lt.0.050) then
               zlwc(i,k) = 0.
            else
               zlwc(i,k) = 0.4*lwcth(i,k)
            endif
         enddo
      enddo
   endif IF_CONDS

   IF_NEWSUND: if (stcond == 'NEWSUND') then

      do k = 1 , nk-1
         do i = 1, ni
            if ( zlwc(i,k) .ge. 0.1e-8 ) then
               cloud(i,k)   = zftot(i,k)
            elseif(zfdc(i,k) .gt. 0.09) then
               cloud(i,k)   = zfdc(i,k)
               zlwc(i,k) = 10.0e-5 * zfdc(i,k)
            else
               cloud(i,k) = 0.0
               zlwc(i,k)  = 0.0
            endif
         enddo
      enddo

      do i=1,ni
         cloud(i,nk)  = 0.0
         zlwc (i,nk)  = 0.0
      end do

   else !IF_NEWSUND

      do k=1,nk
         do i=1,ni
            cloud(i,k) = zftot(i,k)
         enddo
      enddo

   endif IF_NEWSUND

   !...  "no stratospheric lwc" mode when CLIMAT or STRATOS = true
   !...  no clouds above TOPC or where HU < MINQ (see nocld.cdk for topc and minq)

   if (nostrlwc) then
      do k=1,nk
         do i=1,ni
            press = sigma(i,k)*ps(i)
            if (topc.gt.press .or. minq.ge.qm(i,k) ) then
               cloud(i,k) = 0.0
               zlwc (i,k) = 0.0
               ziwc (i,k) = 0.0
            endif
         enddo
      enddo
   endif

   !     ************************************************************
   !     one branch for radia /= CCCMARAD and a simplified branch for radia=CCCMARAD
   !     -----------------------------------------------------------
   !
   !      If(radia /= 'CCCMARAD') Then
   !
   !    Always execute this part of the code to allow calculation of NT in calcNT


   DO_K: do k=1,nk
      do i=1,ni
         liqwcin(i,k) = max(zlwc(i,k),0.)
         if     (cw_rad.le.1) then
            icewcin(i,k)  = 0.0
         else
            icewcin(i,k) = max(ziwc(i,k),0.)
         endif

         if ( any(trim(stcond) == (/'NEWSUND','CONSUN '/)) ) then

            !     The following line is an artificial source of clouds
            !     when using the "CONDS" condensation option (harmful
            !     in the stratosphere)

            if ((liqwcin(i,k)+icewcin(i,k)) .gt. 1.e-6) then
               cloud(i,k) = max(cloud(i,k) ,0.01)
            else
               cloud(i,k) = 0.0
            endif
         endif
         
         if (cloud(i,k) .lt. 0.01) then
            liqwcin(i,k) = 0.
            icewcin(i,k) = 0.
         endif

         !     Min,Max of cloud

         cloud(i,k) = min(cloud(i,k),1.)
         cloud(i,k) = max(cloud(i,k),0.)

         if(cw_rad.gt.0) then

            !     Normalize water contents to get in-cloud values

            zz=max(cloud(i,k),0.05)
            lwcm1=liqwcin(i,k)/zz
            iwcm1=icewcin(i,k)/zz

            !     Consider diabatic lifting limit when Sundquist scheme only

            if ( .not. stcond(1:2) == 'MP')  then
               liqwcin(i,k)=min(lwcm1,lwcth(i,k))
               icewcin(i,k)=min(iwcm1,lwcth(i,k))
            else
               liqwcin(i,k)=lwcm1
               icewcin(i,k)=iwcm1
            endif
         endif

         if     (cw_rad.lt.2) then
            !       calculation of argument for call vsexp
            tcel(i,k)=tm(i,k)-TCDK
            vtcel(i,k)=-.003102*tcel(i,k)*tcel(i,k)
         endif

      end do
   end do DO_K

   !     liquid/solid water partition when not provided by
   !     microphysics scheme ( i.e. cw_rad.lt.2 )
   !     as in cldoptx4 of phy4.2 - after Rockel et al, Beitr. Atmos. Phys, 1991,
   !     p.10 (depends on T only [frac = .0059+.9941*Exp(-.003102 * tcel*tcel)]

   if ( cw_rad .lt. 2 ) then
      call VSEXP (frac,vtcel,nk*ni)
      do k=1,nk
         do i=1,ni
            if (tcel(i,k) .ge. 0.) then
               frac(i,k) = 1.0
            else
               frac(i,k) = .0059+.9941*frac(i,k)
            endif
            if (frac(i,k) .lt. 0.01) frac(i,k) = 0.

            icewcin(i,k) = (1.-frac(i,k))*liqwcin(i,k)
            liqwcin(i,k) = frac(i,k)*liqwcin(i,k)
         enddo
      enddo
   endif

   !     calculate in-cloud liquid and ice water paths in each layer
   !     note: the calculation of the thickness of the layers done here is not
   !     coherent with what is done elsewhere for newrad (radir and sun) or
   !     cccmarad this code was extracted from cldoptx4 for phy4.4 dp(nk) is wrong

   do i=1,ni
      dp=0.5*(sigma(i,1)+sigma(i,2))
      dp=max(dp*ps(i),0.)
      icewpin(i,1) = icewcin(i,1)*dp*rec_grav*1000.
      liqwpin(i,1) = liqwcin(i,1)*dp*rec_grav*1000.

      dp=0.5*(1.-sigma(i,nk))
      dp=max(dp*ps(i),0.)
      icewpin(i,nk) = icewcin(i,nk)*dp*rec_grav*1000.
      liqwpin(i,nk) = liqwcin(i,nk)*dp*rec_grav*1000.
   end do

   do k=2,nk-1
      do i=1,ni
         dp=0.5*(sigma(i,k+1)-sigma(i,k-1))
         dp=max(dp*ps(i),0.)
         icewpin(i,k) = icewcin(i,k)*dp*rec_grav*1000.
         liqwpin(i,k) = liqwcin(i,k)*dp*rec_grav*1000.
      end do
   end do


   !... cccmarad simplified branch
   !      Else

   IF_CCCMARAD: if (radia(1:8) == 'CCCMARAD') then

      !    Begin - Calculation of NT - reproduction of NT obtained with newrad
      !    code (see cldoptx4)

      call calcnt(liqwpin,icewpin,cloud,f(nt),ni,nk,nkp)
      call serxst2(f(nt), 'nt', trnch, ni, 1, 0.0, 1.0, -1)

      !     End - Calculation of NT - reproduction of NT obtained with newrad code
      !     (see cldoptx4)
      !     impose coherent thresholds to cloud fraction and content

      do k=1,nk
         do i=1,ni
            cloud  (i,k) = min(cloud(i,k),1.)
            cloud  (i,k) = max(cloud(i,k),0.)
            liqwcin(i,k) = max(zlwc (i,k),0.)
            icewcin(i,k) = max(ziwc (i,k),0.)

            !            If ((liqwcin(i,k)+icewcin(i,k)) .Le. 1.e-6) Then
            !               cloud(i,k) = 0.0
            !            Endif

            if ((liqwcin(i,k)+icewcin(i,k)) .gt. 1.e-6) then
               cloud(i,k) = max(cloud(i,k) ,0.01)
            else
               cloud(i,k) = 0.0
            endif


            if (cloud(i,k) .lt. 0.01) then
               liqwcin(i,k) = 0.
               icewcin(i,k) = 0.
               cloud(i,k) = 0.0
            endif

            !     Normalize water contents to get in-cloud values

            if(cw_rad.gt.0) then
               !               zz=Max(cloud(i,k),0.01)
               zz=max(cloud(i,k),0.05)
               liqwcin(i,k)=liqwcin(i,k)/zz
               icewcin(i,k)=icewcin(i,k)/zz
            endif
         end do
      enddo


      !    calculate liquid/solid water partition when not provided by
      !    microphysics scheme ( i.e. cw_rad.lt.2 )
      !    ioptpart=1 : as for newrad - after Rockel et al, Beitr. Atmos. Phys, 1991,
      !    p.10 (depends on T only) [frac = .0059+.9941*Exp(-.003102 * tcel*tcel)]
      !    ioptpart=2 : after Boudala et al. (2004), QJRMS, 130, pp. 2919-2931.
      !    (depends on T and twc) [frac=twc^(0.141)*exp(0.037*(tcel))]

      IF_CWRAD_2: if (cw_rad < 2) then
         if (ioptpart == 1) then
            do k=1,nk
               do i=1,ni
                  tcel(i,k)=tm(i,k)-TCDK
                  vtcel(i,k)=-.003102*tcel(i,k)*tcel(i,k)
               enddo
            enddo
         elseif (ioptpart .eq. 2) then
            do k=1,nk
               do i=1,ni
                  tcel(i,k)=tm(i,k)-TCDK
                  vtcel(i,k)=.037*tcel(i,k)
               enddo
            enddo
            call VSPOWN1(vliqwcin, liqwcin, 0.141, nk * ni)
         endif
         call VSEXP (frac,vtcel,nk*ni)

         if (ioptpart == 1) then
            do k=1,nk
               do i=1,ni
                  frac(i,k) = .0059+.9941*frac(i,k)
               enddo
            enddo
         elseif (ioptpart .eq. 2) then
            do k=1,nk
               do i=1,ni
                  !# frac(i,k) = vliqwcin(i,k)*frac(i,k)
                  frac(i,k) = vliqwcin(i,k)*exp(0.037*tcel(i,k))
                  !# frac(i,k) = liqwcin(i,k)**(0.141)*exp(0.037*tcel(i,k))
               enddo
            enddo
         endif
         do k=1,nk
            do I=1,ni
               if (tcel(i,k) .ge. 0.) then
                  frac(i,k) = 1.0
               elseif (tcel(i,k) .lt. -38.) then
                  frac(i,k) = 0.0
               endif
               if (frac(i,k) .lt. 0.01) frac(i,k) = 0.

               icewcin(i,k) = (1.-frac(i,k))*liqwcin(i,k)
               liqwcin(i,k) = frac(i,k)*liqwcin(i,k)
            enddo
         enddo
      endif IF_CWRAD_2

      !    calculate in-cloud liquid and ice water paths in each layer
      !    note: the calculation of the thickness of the layers done here is
      !    not coherent with what is done elsewhere for newrad (radir and sun)
      !    or cccmarad this code was extracted from cldoptx4 for phy4.4
      !    dp(nk) is wrong

      do i=1,ni
         dp=0.5*(sigma(i,1)+sigma(i,2))
         dp=dp*ps(i)
         icewpin(i,1) = icewcin(i,1)*dp*rec_grav*1000.
         liqwpin(i,1) = liqwcin(i,1)*dp*rec_grav*1000.
         dp=0.5*(1.-sigma(i,nk))
         dp=dp*ps(i)
         icewpin(i,nk) = icewcin(i,nk)*dp*rec_grav*1000.
         liqwpin(i,nk) = liqwcin(i,nk)*dp*rec_grav*1000.
      end do

      do k=2,nk-1
         do i=1,ni
            dp=0.5*(sigma(i,k+1)-sigma(i,k-1))
            dp=dp*ps(i)
            icewpin(i,k) = icewcin(i,k)*dp*rec_grav*1000.
            liqwpin(i,k) = liqwcin(i,k)*dp*rec_grav*1000.
         end do
      end do

      !     to impose coherence between cloud fraction and thresholds on liqwpin
      !     and icewpin in calculations of cloud optical properties (cldoppro)
      !     in cccmarad

      do k=1,nk
         do i=1,ni
            if(liqwpin(i,k).le.0.001.and.icewpin(i,k).le.0.001) &
                 cloud(i,k)=0.0
         end do
      end do

   endif IF_CCCMARAD

   !     to simulate a clear sky radiative transfer, de-comment following lines
   !        do k=1,nk
   !        do i=1,ni
   !              liqwcin(i,k)   = 0.0
   !              icewcin(i,k)   = 0.0
   !              liqwpin(i,k)   = 0.0
   !              icewpin(i,k)   = 0.0
   !              cloud(i,k)     = 0.0
   !        end do
   !        end do

   return
end subroutine prep_cw_rad2
