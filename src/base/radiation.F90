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
subroutine radiation2(d,dsiz,f,fsiz,v,vsiz,e,esiz,seloc,&
     ni,nk,kount,trnch,icpu)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object Interface to radiation
   !@Arguments
   !          - Input -
   ! dsiz     dimension of dbus
   ! fsiz     dimension of fbus
   ! vsiz     dimension of vbus
   ! ni       horizontal running length
   ! nk       vertical dimension
   ! kount    timestep number
   ! trnch    slice number
   ! icpu     cpu number executing slice "trnch"
   !          - Input/Output -
   ! dbus     dynamics input field
   ! fbus     historic variables for the physics
   ! vbus     physics tendencies and other output fields from the physics

   integer                  :: fsiz,vsiz,dsiz,esiz,ni,nk,kount,trnch,icpu
   real, dimension(dsiz)    :: d
   real, dimension(fsiz)    :: f
   real, dimension(vsiz)    :: v
   real, dimension(esiz)    :: e
   real, dimension(ni,nk)   :: seloc

   !@Author L.Spacek, November 2011
   !*@/

   real, external :: juliand

   real :: hz0, hz, hz3i, julien, heurser
   real, dimension(ni,nk) :: cldfrac,liqwcin,icewcin,liqwp,icewp,trav2d

   if (radia /= 'NIL') then

      !     call init_radiation(e, esiz, f, fsiz, kount, trnch, ni, nk)

      call prep_cw_rad2(f, fsiz, d, dsiz, v, vsiz, &
           d(tmoins), d(humoins),f(pmoins), d(sigw), &
           cldfrac, liqwcin, icewcin, liqwp, icewp, &
           trav2d,seloc, &
           kount, trnch, icpu, ni, ni, nk-1)

      call diagno_cw_rad(f, fsiz, d,dsiz, v, vsiz, &
           liqwcin, icewcin, liqwp, icewp, &
           cldfrac, heurser, &
           kount, trnch, icpu, ni, nk)

      if (radia == 'CCCMARAD') then
         call cccmarad(d, dsiz, f, fsiz, v, vsiz, &
              d(tmoins), d(humoins), &
              f(pmoins), d(sigw), delt, kount, icpu, &
              trnch, ni, ni, nk-1, nk, &
              liqwcin, icewcin, liqwp, icewp, cldfrac)

      else if (radia == 'CCCMARAD2') then
         call ccc2_cccmarad(d, dsiz, f, fsiz, v, vsiz, &
              d(tmoins), d(humoins), &
              f(pmoins), d(sigw), delt, kount, icpu, &
              trnch, ni, ni, nk-1, nk, &
              liqwcin, icewcin, liqwp, icewp, cldfrac)

      else if (radia == 'NEWRAD') then
         call newrad5(d, dsiz, f, fsiz, v, vsiz, &
              liqwcin, icewcin, liqwp, icewp, cldfrac, &
              delt, kount, &
              trnch, ni, ni, nk-1, icpu, icpu, &
              nk,radnivl(1)-1, radnivl(1), radnivl(2))
      endif
   endif

   !     date(5)=the hour of the day at the start of the run.
   !     date(6)=hundreds of a second of the day at the start of the run.

   hz0 = date(5) + float(date(6))/360000.0
   hz = amod ( hz0+(float(kount)*delt)/3600. , 24. )
   julien = juliand(delt,kount,date)

   call radslop(f, fsiz, v, vsiz, ni, hz, julien, date, trnch, kount)

   return
end subroutine radiation2
