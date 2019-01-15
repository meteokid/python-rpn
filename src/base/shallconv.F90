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
!** s/p shallconv3
      subroutine shallconv3 (d, dsiz, f, fsiz, v, vsiz, kount,trnch, &
                             cdt1, ni, nk              )
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>

      integer ni, nk, kount, trnch
      integer dsiz,fsiz,vsiz
      real, target :: d(dsiz), f(fsiz), v(vsiz)
      REAL cdt1
!
!Author
!          A-M Leduc. (March 2002)
!
!Revisions
! 001      see version 5.5.0 for previous history
!
!Object
!          This is the interface routine for shallow convection schemes
!
!Arguments
!
!          - Input -
! d        "dynamic" bus (dynamics input fields)
!
!          - input/output -
! f        "permanent" bus (field of permanent physics variables)
! v        "volatile" bus
!
!          - input -
! dsiz     dimension of d
! fsiz     dimension of f
! vsiz     dimension of v
! trnch    row number
! kount    timestep number
! cdt1     timestep * factdt
!          see common block "options"
! ni       horizontal running length
! nk       vertical dimension
!
!*
  Include "thermoconsts.inc"

      integer :: i,k,ik
      integer, dimension(ni,nk) :: zilab
      real, dimension(ni) :: zdbdt
      real, dimension(ni,nk) :: geop
      real, pointer, dimension(:) :: ztscs, ztlcs
      real, pointer, dimension(:,:) :: zgztherm
!
!*
      if(conv_shal.eq.'NIL'.or.conv_shal.eq.'BECHTOLD')return
!
      zgztherm(1:ni,1:nk) => v( gztherm:)
      geop = zgztherm*grav
      zilab = 0.

      if(conv_shal.eq.'KTRSNT')then
!
!       Kuo transient shallow convection - it does generate precipitation
!                                             ----
!
        call ktrsnt3( v(tshal), v(hushal), zilab, f(fsc), v(qlsc), &
                     v(qssc), zdbdt,f(tlcs),f(tscs),v(qcz), &
                     d(tplus), d(tmoins), d(huplus), d(humoins), &
                     geop, v(qdifv), f(pmoins), f(pmoins), &
                     d(sigw), cdt1, v(kshal),ni, nk-1  )
!
!       Transformation en hauteur d'eau (m) - diviser par densite eau
!
        ztscs(1:ni) => f(tscs:)
        ztlcs(1:ni) => f(tlcs:)
        do i=1,ni
           ztscs(i) = ztscs(i) * 1.e-03
           ztlcs(i) = ztlcs(i) * 1.e-03
        end do
!
      else if(conv_shal.eq.'KTRSNT_MG') then
!
!       Kuo transient shallow convection used by the meso-global model (mg)
!
        call ktrsnt_mg   ( v(tshal), v(hushal), zilab, f(fsc), v(qlsc), &
                         v(qssc), zdbdt, &
                         d(tplus), d(tmoins), d(huplus), d(humoins), &
                         geop, v(qdifv), f(pmoins), f(pmoins), &
                         d(sigw), cdt1, v(kshal),ni, nk-1  )
!
      endif
!
      call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,tplus,tshal,ni,nk)
      call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,huplus,hushal,ni,nk)
!
      return
      end
