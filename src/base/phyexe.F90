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
subroutine phyexe(e, d, f, v, esiz, dsiz, fsiz, vsiz, trnch, kount, ni, nk)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@object this is the main interface subroutine for the cmc/rpn unified physics
   !@arguments
   !          - input -
   ! e        entry    input field
   ! d        dynamics input field
   !          - input/output -
   ! f        historic variables for the physics
   !          - output -
   ! v        physics tendencies and other output fields from the physics
   !          - input -
   ! esiz     dimension of e
   ! dsiz     dimension of d
   ! fsiz     dimension of f
   ! vsiz     dimension of v
   ! trnch    slice number
   ! kount    timestep number
   ! n        horizontal running length
   ! nk       vertical dimension

   integer :: esiz,dsiz,fsiz,vsiz,trnch,kount,ni,nk
   real    :: e(esiz), d(dsiz), f(fsiz), v(vsiz)

   !@author L. Spacek (oct 2011) 
   !@notes
   !          phy_exe is called by all the models that use the cmc/rpn
   !          common physics library. it returns tendencies to the
   !          dynamics.
   !*@/
   include "tables.cdk"
   include "physteps.cdk"

   integer :: icpu
   real    :: dt, cdt1, rcdt1

   real, dimension(ni,nk)   :: uplus0,vplus0,wplus0,tplus0,huplus0,qcplus0
   real, dimension(ni,nk)   :: seloc,ficebl
   !----------------------------------------------------------------
   icpu = 1
   dt   = delt

   call inichamp4(kount, trnch, ni, nk)

   call phystepinit1(uplus0, vplus0, wplus0, tplus0, huplus0, qcplus0, v, d, f,&
        seloc, dt, cdt1, rcdt1, vsiz, dsiz, fsiz, kount, trnch, icpu, ni, nk)

   call radiation2(d, dsiz, f, fsiz, v, vsiz, e, esiz, seloc, ni, nk, kount, &
        trnch, icpu)

   call sfc_main(seloc, trnch, kount, dt, ni, ni, nk, icpu)

   call metox2(d, v, f, dsiz, vsiz, fsiz, ni, nk)

   call gwd8(d, f, v, e, dsiz, fsiz, vsiz, esiz, std_p_prof, cdt1, kount, &
        trnch, ni, ni, nk-1, icpu )

   if (radia /= 'NIL') &
        call apply_tendencies1(d, dsiz, v, vsiz, f, fsiz, tplus, trad, ni, nk-1)

   call turbulence1(d, f, v, dsiz, fsiz, vsiz, ficebl, seloc, cdt1, kount, &
        trnch, icpu, ni, nk )

   call shallconv3(d, dsiz, f, fsiz, v, vsiz, kount, trnch, cdt1, ni, nk)

   Call precipitation(d, dsiz, f, fsiz, v, vsiz, dt, ni, nk, kount, trnch, icpu)

   call prep_cw2(f, fsiz, d, dsiz, v, vsiz, ficebl, kount, trnch, icpu, ni, nk)

   call tendency4(uplus0, vplus0, wplus0, tplus0, huplus0, qcplus0, v, d, &
        rcdt1, vsiz, dsiz, kount, ni, nk)

   call ens_ptp1(d, v, f, dsiz, fsiz, vsiz, ni, nk, kount)

   call calcdiag(d, f, v, dsiz, fsiz, vsiz, dt, trnch, kount, ni, nk)
   call sfc_calcdiag2(f, v, fsiz, vsiz, moyhr, acchr, dt, trnch, kount, step_driver, ni, nk)

   call chm_exe(e, d, f, v, esiz, dsiz, fsiz, vsiz, dt, trnch, kount, &
        icpu, ni, nk)

   call diagnosurf3(ni, ni, nk, trnch, icpu, kount)
   call extdiag2(d, f, v, dsiz,fsiz, vsiz, kount, trnch, icpu, ni, nk)

   !----------------------------------------------------------------
   return
end subroutine phyexe
