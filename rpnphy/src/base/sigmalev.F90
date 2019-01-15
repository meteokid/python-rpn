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
!** s/p sigmalev
      subroutine sigmalev (se, s, st, vbus, vsiz, n, nk)
      use phybus
      implicit none
#include <arch_specific.hf>
      integer n, nk, ka, vsiz, trnch
      real se(n,nk), s(n,nk), st(n,nk)
      real vbus(vsiz)
!
      integer k, i
!
!Author
!          L. Spacek (Dec 2007)
!
!Revision
! 001      L. Spacek (Sep 2008) - add coefficients for extrapolation in gwd5
!
!Object
!          The subroutine determines whether the model is staggered.
!          Based on that information defines the pointer sigw,
!          calculates sigma coordinates of energy levels and
!          linear interpolation coefficients  for temperature/humidity
!          interpolation to momentum, thermo and energy levels.
!
!Arguments
!
!          - Output -
! se       sigma levels for ('E')
!
!          - Input -
! s        sigma momentum levels
! st       sigma thermo levels
! v        volatile bus
! n        1st horizontal dimension
! nk       vertical dimension
!
!*
!
!
      if (st(1,1)<0) then       ! model is unstaggered
!
         do k=1,nk-2
            do i=1,n
              se(i,k)=0.5*(s(i,k)+s(i,k+1))
            enddo
         enddo
!
         do k=nk-1,nk
            do i=1,n
              se(i,k)=1.
            enddo
         enddo
         call sweights(vbus(at2e),s,st,n,nk,nk-1,.false.)
         call sweights(vbus(at2t),s,st,n,nk,nk-1,.false.)
         call mweights(vbus(at2m),s,st,n,nk,nk-1,.false.)
         sigw=sigm
      else
!
         do k=1,nk-2            ! model is staggered
            do i=1,n
              se(i,k)=st(i,k)
            enddo
         enddo
!
         do k=nk-1,nk
            do i=1,n
              se(i,k)=1.
            enddo
         enddo
         call sweights(vbus(at2e),s,st,n,nk,nk-1,.true.)
         call tweights(vbus(at2t),s,st,n,nk,nk-1,.true.)
         call mweights(vbus(at2m),s,st,n,nk,nk-1,.true.)
         call tweights(vbus(au2t),s,st,n,nk,nk,.false.)
         sigw=sigt
      endif
!      do i=0,n*nk-1
!         vbus(au2t+i)=.5
!      enddo
!
      return
      end
