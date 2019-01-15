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

subroutine cnv_main(d, dsiz, f, fsiz, v, vsiz, t0, q0, qc0, ilab, bett, ccfcp,&
     zcte,  zste,  zcqe,  zsqe, zcqce, zsqce, zcqre, zsqre, &
     dt,    ni,    n,     nk, kount, trnch, icpu)
   use phybus
   use phy_options
   use cnv_options
   implicit none
#include <arch_specific.hf>
   !@objective Interface to convection/condensation
   !@Arguments
   !          - Input -
   ! dsiz     dimension of dbus
   ! fsiz     dimension of fbus
   ! vsiz     dimension of vbus
   ! dt       timestep (parameter) (sec.)
   ! ni       horizontal running length
   ! nk       vertical dimension (nk-1)
   ! kount    timestep number
   !
   !          - Input/Output -
   ! d        dynamics input field
   ! f        historic variables for the physics
   ! v        physics tendencies and other output fields from the physics
   !
   !          - Output -
   ! ilab     flag array: an indication of convective activity from Kuo schemes
   ! bett     estimated averaged cloud fraction growth rate for kuostd
   ! ccfcp    cloud fractional coverage area for kfc and bechtold
   ! zcte     convective temperature tendency
   ! zste     stratiform temperature tendency
   ! zcqe     convective humidity tendency
   ! zsqe     stratiform humidity tendency
   ! zcqce    convective total condensate tendency
   ! zsqce    stratiform total condensate tendency
   ! zcqre    convective rain mixing ratio tendency
   ! zsqre    stratiform rain mixing ratio tendency

   integer fsiz,vsiz,dsiz,ni,n,nk,kount,trnch,icpu
   integer,dimension(ni,nk) :: ilab
   real,   dimension(ni)    :: bett
   real,   dimension(ni,nk) :: t0,q0,qc0,zcte, zste, zcqe, zsqe
   real,   dimension(ni,nk) :: zcqce, zsqce, zcqre, zsqre, ccfcp
   real,   target           :: f(fsiz), v(vsiz), d(dsiz)
   real dt
   !@Author L.Spacek, November 2011
   !@Revisions
   ! 001   -PV/JM-nov2014- fix communication between deep convection and MY_DM
   ! 002   -PV-dec2014 -
   ! 003   -JY-PV-Jul2015- Separate deep and shallow convection scheme in unified bechtold scheme

   include "thermoconsts.inc"
   include "surface.cdk"
   include "comphy.cdk"

   logical,save :: dbgkuo = .false.

   real    :: prestop, cdt1, rcdt1
   integer :: i,k,niter, ier

   logical                  :: lkfbe
   integer,dimension(ni)    :: nca
   real,   dimension(ni)    :: psb,raincv
   real,   dimension(ni,nk) :: zfm,scr3,avert,dotp, tdmask2d
   real, dimension(ni,nk+1) :: bvert
   !
   !*Set Dimensions for convection call
   !
   !INTEGER KENS = 0  ! number of additional ensemble members for convection
   ! (apart from base scheme) [ 1 - 3 ]
   ! NOTA: for base scheme (1 deep + 1 shallow draft)
   !       just put KENS = 0. This will be more
   !       efficient numerically but results will be
   !       less smooth
   !       RECOMMENDATION: Use KENS = 3 for smooth
   !           climate and NWP prediction runs
   integer, parameter :: kchm = 1 ! maximum number of chemical tracers

   ! --------------------------------------------------------------------

   integer:: kidia = 1        ! start index for  horizontal computations
   integer:: kbdia = 1        ! start index for vertical computations
   integer:: ktdia = 1        ! end index for vertical computations
   ! defined as klev + 1 - ktdia
   ! --------------------------------------------------------------------
   !*Define convective in and output arrays
   !
   integer, dimension(ni)         :: kcount
   real,    dimension(ni,nk)      :: dummy1, dummy2, geop
   real,    dimension(ni,nk)      :: ppres,pudr, pddr, phsflx, purv
   real,    dimension(ni,nk,kchm) :: pch1, pch1ten


   ! --------------------------------------------------------------------
   ! Switches for convection call
   !
   !integer:: kice = 0           ! take ice phase into account or not (0)
   !logical:: lrefresh = .true.  ! refresh tendencies at every call
   ! of the scheme
   !logical:: ldown    = .true.  ! switch for convective downdrafts
   !logical:: luvconv  = .false. ! account for transport of hor. wind
   ! attention: should be set to false as not yet sufficiently validated
   !logical:: lch1conv = .false. ! account for transport of tracers
   ! in the convection code
   !logical:: lsettadj = .false. ! set adjustment times
   ! (otherwise it is computed automatically)
   !real   :: xtadjd=3600., xtadjs=10800. ! only used if lsettadj = .true.
   ! ---------------------------------------------------------------------

   ! Pointer to buses

   !#TODO: ? replace by fn to get a bus pointer + indx + dims, bus var pointer + dims
   real, pointer, dimension(:) :: psm, psp, ztdmask, zfcpflg, zkkfc, zrckfc, &
        ztlc, ztsc
   real, pointer, dimension(:,:) :: ncp, nip, qcm, qcp, qip, qrp, qqm, qqp, &
        sigma, ttm, ttp, uu, vv, ww, zfdc, zgztherm, zhufcp, zhushal, &
        zprcten, zpriten, zqckfc, ztfcp, ztshal, ztusc, ztvsc, zufcp, zvfcp, &
        i1qtp, i1ntp
   real :: ncp_prescribed

#define ASSIGN_PTR1D(ptr,bus,idx,n1)    nullify(ptr); if (idx > 0) ptr(1:n1) => bus(idx:idx+n1-1)
#define ASSIGN_PTR2D(ptr,bus,idx,n1,n2) nullify(ptr); if (idx > 0) ptr(1:n1,1:n2) => bus(idx:idx+(n1-1)+(n2-1)*n1)

!!$   nullify(psm); if (pmoins > 0) psm(1:ni) => f(pmoins:)
   ASSIGN_PTR1D(psm,f,pmoins,ni)
   !#TODO: convert other assignment using the macro
   nullify(psp); if (pmoins > 0) psp(1:ni) => f(pmoins:)
   nullify(ztdmask); if (tdmask > 0) ztdmask(1:ni) => f(tdmask:)
   nullify(zfcpflg); if (fcpflg > 0) zfcpflg(1:ni) => f(fcpflg:)
   nullify(zkkfc); if (kkfc > 0) zkkfc(1:ni) => v(kkfc:)
   nullify(zrckfc); if (rckfc > 0) zrckfc(1:ni) => f(rckfc:)
   nullify(ztlc); if (tlc > 0) ztlc(1:ni) => f(tlc:)
   nullify(ztsc); if (tsc > 0) ztsc(1:ni) => f(tsc:)

!!$   nullify(ncp); if (ncplus > 0) ncp(1:ni,1:nk) => d(ncplus:)
   ASSIGN_PTR2D(ncp,d,ncplus,ni,nk)
   !#TODO: convert other assignment using the macro
   nullify(nip); if (niplus > 0) nip(1:ni,1:nk) => d(niplus:)
   nullify(qcm); if (qcmoins > 0) qcm(1:ni,1:nk) => d(qcmoins:)
   nullify(qcp); if (qcplus > 0) qcp(1:ni,1:nk) => d(qcplus:)
   nullify(qip); if (qiplus > 0) qip(1:ni,1:nk) => d(qiplus:)
   nullify(qrp); if (qrplus > 0) qrp(1:ni,1:nk) => d(qrplus:)
   nullify(qqm); if (humoins > 0) qqm(1:ni,1:nk) => d(humoins:)
   nullify(qqp); if (huplus > 0) qqp(1:ni,1:nk) => d(huplus:)
   nullify(sigma); if (sigw > 0) sigma(1:ni,1:nk+1) => d(sigw:)
   nullify(ttm); if (tmoins > 0) ttm(1:ni,1:nk) => d(tmoins:)
   nullify(ttp); if (tplus > 0) ttp(1:ni,1:nk) => d(tplus:)
   nullify(uu); if (uplus > 0) uu(1:ni,1:nk) => d(uplus:)
   nullify(vv); if (vplus > 0) vv(1:ni,1:nk) => d(vplus:)
   nullify(ww); if (wplus > 0) ww(1:ni,1:nk) => d(wplus:)
   nullify(zfdc); if (fdc > 0) zfdc(1:ni,1:nk) => f(fdc:)
   nullify(zgztherm); if (gztherm > 0) zgztherm(1:ni,1:nk) => v(gztherm:)
   nullify(zhufcp); if (hufcp > 0) zhufcp(1:ni,1:nk) => f(hufcp:)
   nullify(zhushal); if (hushal > 0) zhushal(1:ni,1:nk) => v(hushal:)
   nullify(zprcten); if (prcten > 0) zprcten(1:ni,1:nk) => f(prcten:)
   nullify(zpriten); if (priten > 0) zpriten(1:ni,1:nk) => f(priten:)
   nullify(zqckfc); if (qckfc > 0) zqckfc(1:ni,1:nk) => f(qckfc:)
   nullify(ztfcp); if (tfcp > 0) ztfcp(1:ni,1:nk) => f(tfcp:)
   nullify(ztshal); if (tshal > 0) ztshal(1:ni,1:nk) => v(tshal:)
   nullify(ztusc); if (tusc > 0) ztusc(1:ni,1:nk) => v(tusc:)
   nullify(ztvsc); if (tvsc > 0) ztvsc(1:ni,1:nk) => v(tvsc:)
   nullify(zufcp); if (ufcp > 0) zufcp(1:ni,1:nk) => f(ufcp:)
   nullify(zvfcp); if (vfcp > 0) zvfcp(1:ni,1:nk) => f(vfcp:)
   nullify(i1qtp); if (i1qtplus > 0) i1qtp(1:ni,1:nk) => d(i1qtplus:)
   nullify(i1ntp); if (i1ntplus > 0) i1ntp(1:ni,1:nk) => d(i1ntplus:)

   cdt1  = dt
   rcdt1 = 1./cdt1
   tdmask2d(:,:) = spread(ztdmask, dim=2, ncopies=nk)
   tdmask2d(:,:) = tdmask2d(:,:) * delt
   geop  = zgztherm*grav
   lkfbe = (convec == 'KFC' .or. convec == 'BECHTOLD')

   if (my_ccntype == 1) then
      ncp_prescribed = 0.8e+8  !maritime aerosols
   else
      ncp_prescribed = 2.0e+8  !continential aerosols
   endif

   ier  = 0
   ilab = 0
   zcte = 0.
   zcqe = 0.
   zcqce= 0.
   zcqre= 0.
   zste = 0.
   zsqe = 0.
   zsqce = 0.
   zsqre = 0.
   qc0   = 0.
   ccfcp = 0.
   bett  = 0.
   pch1  = 0.
   pch1ten = 0.

   t0 = ttp
   q0 = qqp
   where(qqp < 0.) qqp = 0.

   if (.not. any(trim(stcond) == (/'CONDS','NIL  '/))) then
      qc0 = qcp
      where(qcp < 0.) qcp = 0.
      where(qcm < 0.) qcm = 0.
   endif

   if (convec == 'SEC') then            ! ajustement convectif sec
      call secajus(zcte, ttp, sigma, psp, niter, 0.1, cdt1, ni, nk)
      ttp = ttp + tdmask2d*zcte
   endif

   if (convec == 'OLDKUO') then
      zfm(:,:) = qcp(:,:)
      call omega_from_w(dotp,ww,ttp,qqp,psp,sigma,ni,nk)
      call kuo4(zcte,zcqe,f(tlc),f(tsc), &
           ilab,f(fdc),f(fbl),dotp,zfm, &
           ttp,ttm,qqp,qqm, &
           geop,psp,psm,v(kcl), &
           sigma, cdt1, ni, ni, nk, &
           dbgkuo, satuco, ccctim, cmelt)
      if (.not. any(trim(stcond) == (/'CONDS','NIL  '/))) &
           zcqce = (zfm -qcp)*rcdt1
   endif

   if (convec == 'KFC') &
        call inifcp(psb,psp,psm,raincv,f(rckfc), &
        f(fcpflg),nca,scr3,ww, &
        avert,bvert,sigma, &
        prestop,ni,nk,dt)
   !PV - pourrais eliminer sous-routine inifcp; remplacer avert,psb pour intrants de kfc

   if (convec == 'KFC') then
      call kfcp7(ni,nk,f(fcpflg),v(kkfc),psb,ttp,qqp, &
           uu,vv,ww, &
           f(tfcp),f(hufcp),f(ufcp),f(vfcp), &
           f(prcten),f(priten),f(qrkfc), &
           avert,f(dxdy),f(rckfc),geop, &
           f(capekfc),f(areaup),ccfcp,f(dmfkfc), &
           f(peffkfc),f(umfkfc),f(zbasekfc), &
           f(ztopkfc),f(wumaxkfc), &
           f(qldi),f(qsdi), &
           f(rliq_int),f(rice_int), &
           f(kfcrf),f(kfcsf), &
           kount,f(dlat),f(mg),f(ml),critmask, delt)
      ztlc = zrckfc
      !PV - sum of tendency for liq and solid
      zqckfc = zprcten + zpriten


   else if (convec == 'BECHTOLD') then
      kcount(:) = int(zfcpflg(:))
      phsflx(:,:) = 0.
      do k = 1,nk
         do i = 1,ni
            ppres(i,k) = sigma(i,k)*psp(i)
         enddo
      enddo
      dummy1 = 1.e-9
      dummy2 = 1.e-9

      call bkf_deep(ni, nk,  kidia, ni, kbdia, ktdia,           &
           dt, bkf_lrefresh, bkf_ldown, bkf_kice,               &
           bkf_lsettadj, bkf_xtadjd, bkf_kens,                  &
           ppres, zgztherm,f(dxdy), phsflx,                     &
           ttp, qqp, dummy1, dummy2, uu, vv, ww,                &
           kcount, f(tfcp), f(hufcp), f(prcten), f(priten),     &
           ztlc,ztsc,                                           &
           f(umfkfc), f(dmfkfc), f(kfcrf),f(kfcsf), f(capekfc), f(ztopkfc), f(zbasekfc), &
           purv, f(qldi), f(qsdi),ccfcp,                        &
           f(areaup), f(wumaxkfc),                              &
           kfcmom, f(ufcp), f(vfcp),                            &
           bkf_lch1conv, bkf_kch, pch1, pch1ten,                &
           pudr, pddr,zkkfc)


      zqckfc = zprcten + zpriten
      !PV remettre compteur a zero au pas de temps zero
      if (kount == 0) kcount(:) = 0

      zfcpflg(:) = real(kcount(:))
      !     where(kcount > 0) zkkfc = 1.

   else if (convec == 'KUOSTD') then

      call omega_from_w(dotp,ww,ttp,qqp,psp,sigma,ni,nk)
      call lsctrol(ilab, dotp, sigma, ni, nk)
      call kuostd(zcte,zcqe,ilab,f(fdc),bett, &
           ttp,ttm,qqp,qqm,geop,psp,psm, &
           sigma, cdt1, ni, nk)

   endif
   if (conv_shal == 'BECHTOLD') Then
      do k = 1,nk
         do i = 1,ni
            ppres(i,k) = sigma(i,k)*psp(i)
         enddo
      enddo
      dummy1 = 1.e-9
      dummy2 = 1.e-9
      call bkf_shallow(ni, nk,  kidia, ni, kbdia, ktdia,        &
           dt, bkf_kice,                                        &
           bkf_lsettadj,  bkf_xtadjs,                           &
           ppres, zgztherm,                                     &
           ttp, qqp, dummy1, dummy2, uu, vv, ww,                &
           bkf_lshalm,f(fsc),v(qlsc),v(qssc),                   &
           v(tshal), v(hushal), v(tusc), v(tvsc),               &
           bkf_lch1conv, bkf_kch, pch1, pch1ten,                &
           v(kshal))
      call apply_tendencies1(d,dsiz,v,vsiz,f,fsiz,tplus,tshal,ni,nk)
      call apply_tendencies1(d,dsiz,v,vsiz,f,fsiz,huplus,hushal,ni,nk)
      If (bkf_lshalm) Then
         call apply_tendencies1(d,dsiz,v,vsiz,f,fsiz,uplus,tusc,ni,nk)
         call apply_tendencies1(d,dsiz,v,vsiz,f,fsiz,vplus,tvsc,ni,nk)
      Endif
   endif


   if (lkfbe) then
      zcte = ztfcp
      zcqe = zhufcp
      zfdc = ccfcp
      zcqce = zqckfc
   endif
   !# application des tendances convectives
   ttp = ttp + tdmask2d*zcte
   qqp = qqp + tdmask2d*zcqe
   if (kfcmom) then
      uu = uu + tdmask2d*zufcp
      vv = vv + tdmask2d*zvfcp
   endif


   !PV-JM: update ice and liquid independently if possible (deep == kfc .or. bkf and stcond == MY)

   IF_KFBE_MP: if (lkfbe .and. stcond(1:3) == 'MP_') then

      IF_MP_MY2: if (stcond(1:6) == 'MP_MY2') then !note: this includes 'MP_MY2' and 'MP_MY2_OLD'

       !cloud droplets (number):
         do k = 1,nk
            do i = 1,ni
               if (zprcten(i,k) > 0.) then
                  if (qcp(i,k) > 1.e-9 .and. ncp(i,k) > 1.e-3) then
                     !assume mean size of the detrained droplets is the same size as those in the pre-existing clouds
                     ncp(i,k) = ncp(i,k) + (ncp(i,k)/qcp(i,k))*tdmask2d(i,k)*zprcten(i,k)
                     ncp(i,k) = min(ncp(i,k), 1.e+9)
                  else
                     !initialize cloud number mixing ratios based on specified aerosol type
                     ncp(i,k) = ncp_prescribed  !maritime aerosols
                  endif
               endif
            enddo
         enddo

        !cloud droplets (mass)
         qcp(:,:) = qcp(:,:) + tdmask2d(:,:)*zprcten(:,:)

        !ice crystals (number):
         do k = 1,nk
            do i = 1,ni
               if (zpriten(i,k) > 0.) then
                  if (qip(i,k) > 1.e-9 .and. nip(i,k) > 1.e-3) then
                     !assume mean size of the detrained ice is the same size as those in the pre-existing "anvil"
                     nip(i,k) = nip(i,k) + (nip(i,k)/qip(i,k))*tdmask2d(i,k)*zpriten(i,k)
                     nip(i,k) = min(nip(i,k), 1.e+7)
                  else
                     !initialize ice number mixing ratio based on Cooper (1986)
                     nip(i,k) = 5.*exp(0.304*(TRPL-max(233.,ttp(i,k))))
                     nip(i,k) = min(nip(i,k), 1.e+7)
                  endif
               endif
            enddo
         enddo

        !ice crystals (mass):
         qip(:,:) = qip(:,:) + tdmask2d(:,:)*zpriten(:,:)

      elseif (stcond(1:5) == 'MP_P3') then

        !cloud droplets (number):
        ! note:  In the current version of P3, cloud droplet number is prescribed (there is no 'ncp').
        !        For future 2-moment cloud configuration, 'ncp' should be updated here (as above for MP_MY2)

        !cloud droplets (mass):
         qcp(:,:) = qcp(:,:) + tdmask2d(:,:)*zprcten(:,:)

        !ice crystals (number):
        ! note:  The initialization of ice number is the same for P3 as for MY2, but the two schemes have
        !         different variables.
         do k = 1,nk
            do i = 1,ni
               if (zpriten(i,k) > 0.) then
                  if (i1qtp(i,k) > 1.e-9 .and. i1ntp(i,k) > 1.e-3) then
                     !assume mean size of the detrained ice is the same size as in the pre-existing "anvil"
                     i1ntp(i,k) = i1ntp(i,k) + (i1ntp(i,k)/i1qtp(i,k))*tdmask2d(i,k)*zpriten(i,k)
                     i1ntp(i,k) = min(i1ntp(i,k), 1.e+7)
                  else
                     !initialize ice number mixing ratio based on Cooper (1986)
                     i1ntp(i,k) = 5.*exp(0.304*(TRPL-max(233.,ttp(i,k))))
                     i1ntp(i,k) = min(i1ntp(i,k), 1.e+7)
                  endif
               endif
            enddo
         enddo

        !ice crystals (mass):
         i1qtp(:,:) = i1qtp(:,:) + tdmask2d(:,:)*zpriten(:,:)

      endif IF_MP_MY2

   else ! not using combination kfc-bkf and MY

      qcp(:,:) = qcp(:,:)+ tdmask2d(:,:)*zcqce(:,:)

   endif IF_KFBE_MP

   !PV: zcqre should stay zero with current deep convection options
   !  If(lkfbe) qrp = qrp + tdmask2d*zcqre


   return
end subroutine cnv_main
