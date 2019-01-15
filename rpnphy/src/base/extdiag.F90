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
subroutine extdiag2(d, f, v, dsiz, fsiz, vsiz, kount, trnch, icpu, ni, nk)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object calculate averages and accumulators of tendencies and diagnostics
   !@Arguments
   !          - Input/Output -
   ! f        permanent bus
   !          - input -
   ! d        dynamic bus
   ! v        volatile (output) bus
   !          - input -
   ! dsiz     dimension of d
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! trnch    slice number
   ! icpu     task number
   ! n        horizontal running length
   ! nk       vertical dimension

   integer :: dsiz, fsiz, vsiz, icpu, kount, trnch, ni, nk, istat
   real, target :: d(dsiz), f(fsiz), v(vsiz)

   !@Author B. Bilodeau Feb 2003 - from serdyn5 and phyexe1
   !*@/
   include "thermoconsts.inc"
   include "buses.cdk"
   logical, external :: series_isstep, series_isvar

   character(len=4) :: oname
   integer :: i, k, ierget, s1, ord, kam, ivar, busidx

   real, dimension(ni)     :: work1d
   real, dimension(ni, nk) :: p, work2d1, work2d2

   real, pointer, dimension(:)    :: zpmoins, busptr
   real, pointer, dimension(:, :) :: zhuplus, zsigw, ztplus, &
        zuplus, zvplus, ztcond, zze

   if (.not.series_isstep(' ')) return

   s1 = ni*(nk-1)-1

   ! Extract time series and zonal diagnostics on nk levels
   call sersetm('KA', trnch, nk)

   call serxst2(v(tdew),     'TW', trnch, ni, 1, 0., 1., -1) !TDK
   call serxst2(f(flusolaf), 'AF', trnch, ni, 1, 0., 1., -1) !N4

   if (moyhr > 0) then
      call serxst2(f(uvsmax), 'UX', trnch, ni, 1, 0., 1., -1) !UVMX
      call serxst2(f(uvsavg), 'WA', trnch, ni, 1, 0., 1., -1) !UVAV
      call serxst2(f(hrsmax), 'HX', trnch, ni, 1, 0., 1., -1) !HRMX
      call serxst2(f(hrsmin), 'HN', trnch, ni, 1, 0., 1., -1) !HRMN
      call serxst2(f(husavg), 'QV', trnch, ni, 1, 0., 1., -1) !QSAV
      call serxst2(f(ttmin+S1), 'T5', trnch, ni, 1, 0., 1., -1) !T5.. +S1!
      call serxst2(f(ttmax+S1), 'T9', trnch, ni, 1, 0., 1., -1) !T9.. +S1!
   endif

   ! Clouds
   call serxst2(f(ftot), 'NU', trnch, ni, nk, 0., 1., -1) !FN
   call serxst2(f(fbl),  'NJ', trnch, ni, nk, 0., 1., -1) !NC
   call serxst2(f(qtbl), 'QB', trnch, ni, nk, 0., 1., -1) !QTBL

   if (.not.any(convec == (/'NIL   ', 'SEC   ', 'MANABE'/))) then
      ! Nuages convectifs
      call serxst2(F(FDC), 'NC', trnch, ni, nk, 0., 1., -1) !CK
   endif

   ! Effect of precipitation evaporation
   if (series_isvar('EP')) then
      ztcond (1:ni, 1:nk) => v( tcond :)
      do k=1, nk-1
         do i=1, ni
            work2d1(i, k)=min(0., ztcond(i, k))
         enddo
      enddo
      call serxst2(work2d1, 'EP', trnch, ni, nk, 0.0, 1., -1)
   endif

   call serxst2(d(uplus), 'UUWE', trnch, ni, nk, 0.0, 1.0, -1) !UP
   call serxst2(d(vplus), 'VVSN', trnch, ni, nk, 0.0, 1.0, -1) !VP
   call serxst2(d(wplus), 'WW',   trnch, ni, nk, 0.0, 1.0, -1) !WP
   call serxst2(d(tplus), 'TT',   trnch, ni, nk, 0.0, 1.0, -1) !2T

   ! Potential temperature
   if (series_isvar('TH')) then
      zpmoins(1:ni) => f(pmoins:)
      zsigw(1:ni, 1:nk) => d(sigw:)
      ztplus(1:ni, 1:nk) => d(tplus:)
      do k=1, nk
!vdir nodep
         do i=1, ni
            p(i, k) = zsigw(i, k)*zpmoins(i)
            work2d1(i, k) = 1.e-5*p(i, k)
         end do
      end do
      call vspown1(work2d1, work2d1, -cappa, ni*nk)
      do k=1, nk
         do i=1, ni
            work2d1(i, k) = ztplus(i, k)*work2d1(i, k)
         end do
      end do
      call serxst2(work2d1, 'TH', trnch, ni, nk, 0.0, 1.0, -1)
      call serxst2(P,       'PX', TRNCH, NI, nk, 0.0, 1.0, -1)
   endif

   !# Dew point temperature
   if (series_isvar('TD')) then
      ! Find dew point depression first (es). saturation calculated with
      ! respect to water only (since td may be compared to observed tephigram).
      ztplus(1:ni, 1:nk) => d(tplus:)
      call mhuaes3(work2d1, d(huplus), d(tplus), p, .false., ni, nk, ni)
      do k=1, nk
!vdir nodep
         do i=1, ni
            work2d1(i, k) = min( &
                 ztplus(i, k), &
                 ztplus(i, k) - work2d1(i, k) &
                 )
         end do
      end do
      call serxst2(work2d1, 'TD', trnch, ni, nk, 0.0, 1.0, -1)
   endif

   !# Moisture: Eliminate negative values
   if (series_isvar('HR')) then
      zhuplus(1:ni, 1:nk) => d(huplus:)
!vdir nodep
      do k=1, nk
         do i=1, ni
            work2d1(i, k) = max(0.0, zhuplus(i, k))
         end do
      end do
      call serxst2(work2d1, 'HU', trnch, ni, nk, 0.0, 1.0, -1)
   endif

   !# Relative humidity
   if (series_isvar('HR')) then
      zpmoins(1:ni) => f(pmoins:)
      zsigw(1:ni, 1:nk) => d(sigw:)
      do k=1, nk
         do i=1, ni
            work2d2(i, k) = zsigw(i, k)*zpmoins(i)
         end do
      end do
      call mfohr4 (work2d1, d(huplus), d(tplus), &
           work2d2, ni, nk, ni, satuco)
      call serxst2(work2d1, 'HR', trnch, ni, nk, 0.0, 1.0, -1)
   endif

   if (.not. any(trim(stcond) == (/'CONDS', 'NIL  '/))) then
      !       Total cloud water
      call serxst2(d(qcplus), 'QC', trnch, ni, nk, 0.0, 1.0, -1) !CQ
   endif

   call serxst2(f(pmoins), 'P0', trnch, ni, 1, 0.0, 1.0, -1) !P8

   ! Modulus of the surface wind
   if (series_isvar('VE')) then
      zuplus(1:ni, 1:nk) => d( uplus:)
      zvplus(1:ni, 1:nk) => d( vplus:)
      do i = 1, ni
         work1d(i) = zuplus(i, nk)*zuplus(i, nk) + &
              zvplus(i, nk)*zvplus(i, nk)
      end do
      call vssqrt(work1d, work1d, ni)
      call serxst2(work1d, 'VE', trnch, ni, 1, 0.0, 1.0, -1)
   endif

   ! Geopotential height (in dam)
   if (series_isvar('GZ')) then
      zze(1:ni, 1:nk) => v(ze:)
      do k=1, nk
         do i=1, ni
            work2d1(i, k) = 0.1 * zze(i, k)
         end do
      enddo
      call serxst2(work2d1, 'GZ', trnch, ni, nk, 0., 1., -1)
   endif


   !# Extract Series for all bus var
!!$   print *,'Extract Series for all bus var', series_nvars,dyntop,pertop,voltop
   ord = -9
   do ivar=1,dyntop
      oname  = dynnm(ivar, BUSNM_ON)
      if (series_isvar(oname)) then
         kam    = dynpar(ivar, BUSPAR_NK)
         busidx = dynpar(ivar, BUSPAR_I0)
         busptr => d(busidx:)
!!$         print *,'Extract Series D ',  trim(oname), trnch ,kam
         call serxst2(busptr, oname, trnch, ni, kam, 0., 1., ord)
      endif
   enddo
   do ivar=1,pertop
      oname  = pernm(ivar, BUSNM_ON)
      if (series_isvar(oname)) then
         kam    = perpar(ivar, BUSPAR_NK)
         busidx = perpar(ivar, BUSPAR_I0)
         busptr => f(busidx:)
!!$         print *,'Extract Series P ',  trim(oname), trnch ,kam
         call serxst2(busptr, oname, trnch, ni, kam, 0., 1., ord)
      endif
   enddo
   do ivar=1,voltop
      oname  = volnm(ivar, BUSNM_ON)
      if (series_isvar(oname)) then
         kam    = volpar(ivar, BUSPAR_NK)
         busidx = volpar(ivar, BUSPAR_I0)
         busptr => v(busidx:)
!!$         print *,'Extract Series V ',  trim(oname), trnch ,kam
         call serxst2(busptr, oname, trnch, ni, kam, 0., 1., ord)
      endif
   enddo

   return
end subroutine extdiag2
