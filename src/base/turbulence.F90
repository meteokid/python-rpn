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

subroutine turbulence1 ( d,   f,   v, dsiz, fsiz, vsiz  , &
     ficebl, seloc, cdt1, &
     kount, trnch, icpu, ni, nk )
   use phy_options
   implicit none
#include <arch_specific.hf>
   !@Object
   !@Arguments
   !          - Input -
   ! e        entry    input field
   ! d        dynamics input field
   !          - Input/Output -
   ! f        historic variables for the physics
   !          - Output -
   ! v        physics tendencies and other output fields from the physics
   !          - Input -
   ! esiz     dimension of e
   ! dsiz     dimension of d
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! dt       timestep (sec.)
   ! trnch    slice number
   ! kount    timestep number
   ! icpu     cpu number executing slice "trnch"
   ! n        horizontal running length
   ! nk       vertical dimension

   integer dsiz,fsiz,vsiz,trnch,kount,icpu,ni,nk
   real d(dsiz), f(fsiz), v(vsiz)
   real qcdifv(ni,nk), ficebl(ni,nk), seloc(ni,nk), cdt1

   !@Author L. Spacek (Nov 2011)

   if (.not.any(fluvert == (/'MOISTKE', 'CLEF   '/))) return

   if (pbl_coupled) then
      call boundary_layer2 ( d,   f,   v, dsiz, fsiz, vsiz  , &
           ficebl, seloc, cdt1, &
           kount, trnch, icpu, ni, nk )
   else
      call boundary_layer_modlevs2 ( d,   f,   v, dsiz, fsiz, vsiz  , &
           ficebl, seloc, cdt1, &
           kount, trnch, icpu, ni, nk )
   endif

   return
end subroutine turbulence1
