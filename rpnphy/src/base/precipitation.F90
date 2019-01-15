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

subroutine precipitation(dbus, dsiz, fbus, fsiz, vbus, vsiz,&
                         dt, ni, nk, kount, trnch, icpu)
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
   !@Object Interface to convection/condensation
   !@Arguments
   !
   !          - Input -
   ! dsiz     dimension of dbus
   ! fsiz     dimension of fbus
   ! vsiz     dimension of vbus
   ! dt       timestep (sec.)
   ! ni       horizontal running length
   ! nk       vertical dimension
   ! kount    timestep number
   ! trnch    slice number
   ! icpu     cpu number executing slice "trnch"
   !
   !          - Input/Output -
   ! dbus     dynamics input field
   ! fbus     historic variables for the physics
   ! vbus     physics tendencies and other output fields from the physics

   integer      :: fsiz,vsiz,dsiz,ni,nk,kount,trnch,icpu
   real         :: dt
   real, target :: dbus(dsiz), fbus(fsiz), vbus(vsiz)

   !@Author L.Spacek, November 2011
   
   integer,dimension(ni,nk) :: ilab
   real,   dimension(ni)    :: beta
   real,   dimension(ni,nk) :: t0,q0,qc0,zcte, zste, zcqe, zsqe
   real,   dimension(ni,nk) :: zcqce, zsqce, zcqre, zsqre, ccfcp
   
   !#TODO: move into phyexe

   if (convec == 'NIL' .and. stcond == 'NIL' .and. conv_shal /= 'BECHTOLD') return

   call cnv_main( dbus, dsiz, fbus, fsiz, vbus, vsiz, &
        t0,q0,qc0, ilab,beta,ccfcp,&
        zcte, zste, zcqe, zsqe, &
        zcqce, zsqce, zcqre, zsqre, &
        dt, ni, ni, nk-1, &
        kount, trnch, icpu )

   call condensation ( dbus, dsiz, fbus, fsiz, vbus, vsiz, &
        t0,q0,qc0, ilab,beta,ccfcp,&
        zcte, zste, zcqe, zsqe, &
        zcqce, zsqce, zcqre, zsqre, &
        dt, ni, ni, nk-1, &
        kount, trnch, icpu )

   return
end subroutine precipitation
