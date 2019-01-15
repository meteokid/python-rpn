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

subroutine sfcexch_options2()
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object initialization of the surface parameters at the beginning
   !        of each execution of the model
   !@Author  L. Spacek (Spring 2013)
   !@Revisions
   !*

#include <msg.h>
#include <rmnlib_basics.hf>
#include <WhiteBoard.hf>
   include "isbapar.cdk"
   include "tebcst.cdk"
   
   integer :: ier, nv, options, iverb
   !---------------------------------------------------------------------

   options = WB_REWRITE_NONE+WB_IS_LOCAL
   iverb = wb_verbosity(WB_MSG_INFO)
   ier = WB_OK

   ier = min(wb_get('phy/date',date,nv),ier)
   ier = min(wb_get('phy/climat',climat),ier)
   ier = min(wb_get('phy/delt',delt),ier)
   ier = min(wb_get('phy/fluvert',fluvert),ier)
   ier = min(wb_get('phy/radslope',radslope),ier)
   ier = min(wb_get('phy/radia',radia),ier)

   ier = min(wb_put('sfc/as',as,options),ier)
   ier = min(wb_put('sfc/beta',beta,options),ier)
   ier = min(wb_put('sfc/ci',ci,options),ier)
   ier = min(wb_put('sfc/critlac',critlac,options),ier)
   ier = min(wb_put('sfc/critmask',critmask,options),ier)
   ier = min(wb_put('sfc/critsnow',critsnow,options),ier)
   ier = min(wb_put('sfc/drylaps',drylaps,options),ier)
   ier = min(wb_put('sfc/impflx',impflx,options),ier)
   ier = min(wb_put('sfc/indx_soil',indx_soil,options),ier)
   ier = min(wb_put('sfc/indx_glacier',indx_glacier,options),ier)
   ier = min(wb_put('sfc/indx_water',indx_water,options),ier)
   ier = min(wb_put('sfc/indx_ice',indx_ice,options),ier)
   ier = min(wb_put('sfc/indx_urb',indx_urb,options),ier)
   ier = min(wb_put('sfc/indx_agrege',indx_agrege,options),ier)
   ier = min(wb_put('sfc/leadfrac',leadfrac,options),ier)
   ier = min(wb_put('sfc/n0rib',n0rib,options),ier)
   ier = min(wb_put('sfc/tdiaglim',tdiaglim,options),ier)
   ier = min(wb_put('sfc/vamin',vamin,options),ier)
   ier = min(wb_put('sfc/veg_rs_mult',veg_rs_mult,options),ier)
   ier = min(wb_put('sfc/z0dir',z0dir,options),ier)
   ier = min(wb_put('sfc/zt',zt,options),ier)
   ier = min(wb_put('sfc/zta',zta,options),ier)
   ier = min(wb_put('sfc/zu',zu,options),ier)
   ier = min(wb_put('sfc/zua',zua,options),ier)

   iverb = wb_verbosity(iverb)
   
   if (.not.RMN_IS_OK(ier)) then
      call msg(MSG_ERROR,'(sfc_exch_options) probleme in wb_put/get')
      call qqexit(1)
   endif
   !----------------------------------------------------------------------
   return
end subroutine sfcexch_options2
