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
!**   S/P DIAGNO_CW_RAD
      Subroutine diagno_cw_rad (f, fsiz, d,dsiz, v, vsiz, &
           liqwcin, icewcin, liqwp, icewp, cloud, heurser, &
           kount, trnch, task, ni, nk)

      use phybus
      implicit none
#include <arch_specific.hf>

      Integer fsiz, dsiz, vsiz, ni, nk
      Integer kount, trnch, task
      Real heurser
      Real, target ::  f(fsiz), d(dsiz), v(vsiz)
      Real liqwcin(ni,nk), icewcin(ni,nk)
      Real liqwp(ni,nk-1), icewp(ni,nk-1)
      Real cloud(ni,nk)

!     Author
!     L. Spacek (Apr 2005)
!
!     Revisions
!     001      see version 5.5.0 for previous history
!
!     Object
!     Calculate diagnostic for the radiation package
!
!     Arguments
!
!     - input -
!     dsiz     Dimension of d
!     fsiz     Dimension of f
!     vsiz     Dimension of v
!     liqwcin  in-cloud liquid water content
!     icewcin  in-cloud ice    water content
!     liqwp    in-cloud liquid water path
!     icewp    in-cloud ice    water path
!     cloud    cloudiness passed to radiation
!     kount    index of timestep
!     trnch    number of the slice
!     task     task number
!     n        horizontal Dimension
!     nk       number of layers
!
!     - output -
!     tlwp     total integrated liquid water path
!     tiwp     total integrated ice    water path
!     tlwpin   total integrated in-cloud liquid water path
!     tiwpin   total integrated in-cloud ice    water path
!     lwcrad   liquid water content passed to radiation
!     iwcrad   ice    water content passed to radiation
!     cldrad  cloudiness passed to radiation

      Integer i, j, k, ik, it

      real, pointer, dimension(:)   :: ztlwp, ztiwp, ztlwpin, ztiwpin
      real, pointer, dimension(:,:) :: zlwcrad, ziwcrad, zcldrad

      ztlwp  (1:ni)      => f( tlwp:)
      ztiwp  (1:ni)      => f( tiwp:)
      ztlwpin(1:ni)      => f( tlwpin:)
      ztiwpin(1:ni)      => f( tiwpin:)
      zlwcrad(1:ni,1:nk) => v( lwcrad:)
      ziwcrad(1:ni,1:nk) => v( iwcrad:)
      zcldrad(1:ni,1:nk) => v( cldrad:)

      Do i=1,ni
         ztlwp(i)      = 0.0
         ztiwp(i)      = 0.0
         ztlwpin(i)    = 0.0
         ztiwpin(i)    = 0.0
         zlwcrad(i,nk) = 0.0
         ziwcrad(i,nk) = 0.0
         zcldrad(i,nk) = 0.0
      Enddo

      Do k=1,nk-1
         Do i=1,ni
            ztlwp(i)   = ztlwp(i)   + liqwp(i,k)*cloud(i,k)
            ztiwp(i)   = ztiwp(i)   + icewp(i,k)*cloud(i,k)
            ztlwpin(i) = ztlwpin(i) + liqwp(i,k)
            ztiwpin(i) = ztiwpin(i) + icewp(i,k)
         Enddo
      Enddo

!     conversion d'unites : tlwp et tiwp en kg/m2

      Do i=1,ni
         ztlwp(i)   = ztlwp(i) * 0.001
         ztiwp(i)   = ztiwp(i) * 0.001
         ztlwpin(i) = ztlwpin(i) * 0.001
         ztiwpin(i) = ztiwpin(i) * 0.001
      Enddo

      Do k=1,nk-1
         Do i=1,ni
            zlwcrad(i,k) = liqwcin(i,k)*cloud(i,k)
            ziwcrad(i,k) = icewcin(i,k)*cloud(i,k)
            zcldrad(i,k) = cloud(i,k)
         Enddo
      Enddo

!     extraction pour diagnostics
      Call serxst2(f(tlwp), 'icr', trnch, ni, 1, 0.0, 1.0, -1)
      Call serxst2(f(tiwp), 'iir', trnch, ni, 1, 0.0, 1.0, -1)
      Call serxst2(f(tlwpin), 'w1', trnch, ni, 1, 0.0, 1.0, -1)
      Call serxst2(f(tiwpin), 'w2', trnch, ni, 1, 0.0, 1.0, -1)

      Call serxst2(v(iwcrad), 'iwcr', trnch, ni, nk, 0.0, 1.0, -1)
      Call serxst2(v(lwcrad), 'lwcr', trnch, ni, nk, 0.0, 1.0, -1)
      Call serxst2(v(cldrad), 'cldr', trnch, ni, nk, 0.0, 1.0, -1)

      End Subroutine diagno_cw_rad
