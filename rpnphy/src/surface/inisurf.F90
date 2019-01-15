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
subroutine inisurf4(kount, ni, nk, trnch)
   use sfc_options
   use sfcbus_mod
   implicit none
#include <arch_specific.hf>
   !@Object Transfer and initialize geophysical fields for the
   !        surface schemes
   !@Arguments
   !       - Input/Ouput -
   ! f        field for permanent physics variables
   ! fsiz     dimension of f
   ! e        field for entry variables
   ! esiz     dimension of e
   ! ni       horizontal dimension

   integer ni, nk, kount, trnch

   !@Author Stephane Belair (February 1999)
   !@NOTE: This subroutine expects snow depth in cm.
   !       The snow depth is converted in metre (in this s/r)
   !       when the 'entry variables' are transfered to the
   !       permanent variables.
   !*@/

   include "thermoconsts.inc"
   include "isbapar.cdk"
   include "sfcinput.cdk"

   real, parameter :: z0ice = 0.001
   real, parameter :: z0sea = 0.001

   real, save :: almin  = 0.50
   real, save :: tauf   = 0.24
   real, save :: tauday = 24.

   real    :: prcor
   integer :: i, k

   real, pointer, dimension(:) :: zalen, zdhdx, zdhdxdy, zdhdxdyen, zdhdxen, zdhdy, zdhdyen, zepstfn, zglacen, zglacier, zglsea, zglsea0, zglseaen, zicedp, zicedpen, ziceline, zicelinen, zlhtg, zlhtgen, zmg, zmgen, zml, zresa, zsnoal, zsnoalen, zsnoagen, zsnoden, zsnoma, zsnoro, zsnoroen, ztsrad, ztwater, ztwateren, zwveg, zwvegen, zwsnow, zwsnowen, zz0en
   real, pointer, dimension(:,:) :: zalvis, zclay, zclayen, zisoil, zisoilen, zsand, zsanden, zsnodp, zsnodpen, ztglacen, ztglacier, ztmice, ztmicen, ztmoins, ztsoil, ztsoilen, zvegf, zvegfen, zwsoil, zwsoilen, zz0, zz0t

#define MKPTR1D(NAME1,NAME2) nullify(NAME1); if (vd%NAME2%i > 0 .and. associated(busptr(vd%NAME2%i)%ptr)) NAME1(1:ni) => busptr(vd%NAME2%i)%ptr(:,trnch)
#define MKPTR2D(NAME1,NAME2) nullify(NAME1); if (vd%NAME2%i > 0 .and. associated(busptr(vd%NAME2%i)%ptr)) NAME1(1:ni,1:vd%NAME2%mul*vd%NAME2%niveaux) => busptr(vd%NAME2%i)%ptr(:,trnch)

   !-------------------------------------------------------------

   !  Nothing to process if no fields were read
   if (phyinread_n == 0) return

   MKPTR1D(zalen,alen)
   MKPTR1D(zdhdx,dhdx)
   MKPTR1D(zdhdxdy,dhdxdy)
   MKPTR1D(zdhdxdyen,dhdxdyen)
   MKPTR1D(zdhdxen,dhdxen)
   MKPTR1D(zdhdy,dhdy)
   MKPTR1D(zdhdyen,dhdyen)
   MKPTR1D(zepstfn,epstfn)
   MKPTR1D(zglacen,glacen)
   MKPTR1D(zglacier,glacier)
   MKPTR1D(zglsea,glsea)
   MKPTR1D(zglsea0,glsea0)
   MKPTR1D(zglseaen,glseaen)
   MKPTR1D(zicedp,icedp)
   MKPTR1D(zicedpen,icedpen)
   MKPTR1D(ziceline,iceline)
   MKPTR1D(zicelinen,icelinen)
   MKPTR1D(zlhtg,lhtg)
   MKPTR1D(zlhtgen,lhtgen)
   MKPTR1D(zmg,mg)
   MKPTR1D(zmgen,mgen)
   MKPTR1D(zml,ml)
   MKPTR1D(zresa,resa)
   MKPTR1D(zsnoal,snoal)
   MKPTR1D(zsnoalen,snoalen)
   MKPTR1D(zsnoagen,snoagen)
   MKPTR1D(zsnoden,snoden)
   MKPTR1D(zsnoma,snoma)
   MKPTR1D(zsnoro,snoro)
   MKPTR1D(zsnoroen,snoroen)
   MKPTR1D(ztsrad,tsrad)
   MKPTR1D(ztwater,twater)
   MKPTR1D(ztwateren,twateren)
   MKPTR1D(zwveg,wveg)
   MKPTR1D(zwvegen,wvegen)
   MKPTR1D(zwsnow,wsnow)
   MKPTR1D(zwsnowen,wsnowen)
   MKPTR1D(zz0en,z0en)

   MKPTR2D(zalvis,alvis)
   MKPTR2D(zclay,clay)
   MKPTR2D(zclayen,clayen)
   MKPTR2D(zisoil,isoil)
   MKPTR2D(zisoilen,isoilen)
   MKPTR2D(zsand,sand)
   MKPTR2D(zsanden,sanden)
   MKPTR2D(zsnodp,snodp)
   MKPTR2D(zsnodpen,snodpen)
   MKPTR2D(ztglacen,tglacen)
   MKPTR2D(ztglacier,tglacier)
   MKPTR2D(ztmice,tmice)
   MKPTR2D(ztmicen,tmicen)
   MKPTR2D(ztmoins,tmoins)
   MKPTR2D(ztsoil,tsoil)
   MKPTR2D(ztsoilen,tsoilen)
   MKPTR2D(zvegf,vegf)
   MKPTR2D(zvegfen,vegfen)
   MKPTR2D(zwsoil,wsoil)
   MKPTR2D(zwsoilen,wsoilen)
   MKPTR2D(zz0,z0)
   MKPTR2D(zz0t,z0t)

   if (any('vegfen' == phyinread_list_s(1:phyinread_n))) then
      do i=1,ni
         do k=1,nclass
            zvegf(i,k) = zvegfen(i,k)
         end do
      end do
   endif

   ! Several treatments on geophysical fields valid for both isba and fcrest
   ! the water temperature (tm) is decreased for points where the 
   ! filtering of mountains lead to an icrease of the water level
   ! (old subroutine modtmtp of gem's dynamic library)


   if ( drylaps ) then
      prcor = grav/cpd
   else
      prcor = grav*stlo
   end if

   ! Other consistency tests ...

   if (any('snodpen' == phyinread_list_s(1:phyinread_n))) then
!VDIR NODEP
      do k=1,nsurf
         do i=1,ni
            zsnodpen(i,k) = max( 0., zsnodpen(i,k))
         end do
      end do
   endif

   if (any('tglacen' == phyinread_list_s(1:phyinread_n))) then
!VDIR NODEP
      do i=1,ni
         ztglacen(i,1) = min( trpl, ztglacen(i,1))
         ztglacen(i,2) = min( trpl, ztglacen(i,2))
      end do
   endif

   ! From the "entry" to the "permanent" bus
   !
   !========================================================================
   !          for variables common to all surface schemes
   !========================================================================
   !
   !
!VDIR NODEP
   DO_I: do i=1,ni
      if (any('alen' == phyinread_list_s(1:phyinread_n))) then
         zalvis(i,indx_soil   ) = zalen(i)
         zalvis(i,indx_glacier) = zalen(i)
         zalvis(i,indx_water  ) = zalen(i)
         zalvis(i,indx_ice    ) = zalen(i)
         zalvis(i,indx_agrege ) = zalen(i)
         if (schmurb.ne.'NIL') then
            zalvis(i,indx_urb ) = zalen(i)
         endif
      endif

      if (any('glacen' == phyinread_list_s(1:phyinread_n))) then
         zglacier(i) = zglacen(i)
      endif
      if (any('mgen' == phyinread_list_s(1:phyinread_n))) then
         zmg(i) = zmgen(i)
      endif

      !       --- snodp deja en metres
      if (any('snodpen' == phyinread_list_s(1:phyinread_n))) then
         zsnodp(i,indx_soil   ) = zsnodpen(i,1)
         zsnodp(i,indx_glacier) = zsnodpen(i,2)
         zsnodp(i,indx_water  ) = 0.0
         zsnodp(i,indx_ice    ) = zsnodpen(i,4)
      endif

      if (any('twateren' == phyinread_list_s(1:phyinread_n))) then
         ztwater(i) = ztwateren(i)
      endif
      if (any('tsoilen' == phyinread_list_s(1:phyinread_n))) then
         ztsrad(i) = ztsoilen(i,1)
      endif
      if (any('z0en' == phyinread_list_s(1:phyinread_n))) then
         zz0 (i,indx_soil   ) = zz0en(i)
         zz0 (i,indx_glacier) = zz0en(i)
         zz0 (i,indx_water  ) = z0sea
         zz0 (i,indx_ice    ) = z0ice
         zz0 (i,indx_agrege ) = zz0en(i)
         zz0t(i,indx_soil   ) = zz0en(i)
         zz0t(i,indx_glacier) = zz0en(i)
         zz0t(i,indx_water  ) = z0sea
         zz0t(i,indx_ice    ) = z0ice
         zz0t(i,indx_agrege ) = zz0en(i)
      endif
      if (any('lhtgen' == phyinread_list_s(1:phyinread_n))) then
         zlhtg(i) = zlhtgen(i)
      endif
      if (any('icedpen' == phyinread_list_s(1:phyinread_n))) then
         zicedp(i) = zicedpen(i)
      endif
      if (any('tglacen' == phyinread_list_s(1:phyinread_n))) then
         ztglacier(i,1) = ztglacen(i,1)
         ztglacier(i,2) = ztglacen(i,2)
      endif
      if (any('glseaen' == phyinread_list_s(1:phyinread_n))) then
         zglsea (i) = zglseaen(i)
         zglsea0(i) = zglseaen(i)
      endif
      !       Mask for the lakes
      if (any('vegfen' == phyinread_list_s(1:phyinread_n))) then
         zml(i) = zvegfen(i,3)
      endif
      if(icelac) then
         if (any('icelinen' == phyinread_list_s(1:phyinread_n))) then
            ziceline(i) = zicelinen(i)
         endif
      else
         if (kount == 0) ziceline(i) = 1.
      endif

      !       transvidage des variables necessaires au blocage orographique
      if (any('dhdxen' == phyinread_list_s(1:phyinread_n))) then
         zdhdx(i) = zdhdxen(i)
      endif
      if (any('dhdyen' == phyinread_list_s(1:phyinread_n))) then
         zdhdy(i) = zdhdyen(i)
      endif
      if (any('dhdxdyen' == phyinread_list_s(1:phyinread_n))) then
         zdhdxdy(i) = zdhdxdyen(i)
      endif
   end do DO_I

   if (any('tmicen' == phyinread_list_s(1:phyinread_n))) then
      do k=1,nl
         do i=1,ni
            ztmice(i,k) = ztmicen(i,k)
            ztmice(i,k) = min(tcdk, ztmice(i,k))
         end do
      end do
   endif

   !========================================================================
   !                             for lakes only
   !========================================================================

   if (any('vegfen' == phyinread_list_s(1:phyinread_n))) then
      if (schmsol == 'ISBA') then
!VDIR NODEP
         do k=1,nclass
            do i=1,ni
               zvegf(i,k) = zvegfen(i,k)
            end do
         end do
      endif
   endif

   call lacs4(climat, ni, trnch)

   !========================================================================
   !     Special cases

   if (any('icedpen' == phyinread_list_s(1:phyinread_n))) then
!VDIR NODEP
      do i=1,ni
         !           no snow allowed in the absence of marine ice
         if (zicedp(i).lt.himin) then
            zsnodp(i,indx_ice) = 0.0
         endif
      end do
   endif

   if (kount == 0) then
!VDIR NODEP
      do i=1,ni
         !           For force-restore scheme only : if no radiation scheme
         !           is used, then surface IR emissivity is set to 0.
         if (radia == 'NIL') then
            zepstfn(i) = 0.
         else
            zepstfn(i) = stefan
         end if
      end do
   endif

   !=========================================================================
   !                                      FOR ISBA ... FOR ISBA ... FOR ISBA
   !=========================================================================

   IF_ISBA: if (schmsol == 'ISBA') then

!VDIR NODEP
      do i=1,ni

         if (any('tsoilen' == phyinread_list_s(1:phyinread_n))) then
            ztsoil(i,1) = ztsoilen(i,1)
            ztsoil(i,2) = ztsoilen(i,2)
         endif
         if (any('wsoilen' == phyinread_list_s(1:phyinread_n))) then
            zwsoil(i,1) = zwsoilen(i,1)
            zwsoil(i,2) = zwsoilen(i,2)
         endif
         if (any('wvegen' == phyinread_list_s(1:phyinread_n))) then
            zwveg(i) = zwvegen(i)
         endif
         if (any('isoilen' == phyinread_list_s(1:phyinread_n))) then
            zisoil(i,1) = zisoilen(i,1)
         endif
         if (any('wsnowen' == phyinread_list_s(1:phyinread_n))) then
            zwsnow(i) = zwsnowen(i)
         endif
         if (kount == 0) then
            zresa(i) = 50.
         endif
      end do

      ! Special operations for the snow variables
      !
      ! Careful here about the units:
      ! "snoro" is the relative density of snow, 
      !         i.e., rho_ice / rho_water (no units)
      ! "snoma" is the snow water equivalent in mm (i.e., kg / m2)
      ! "snoal" is the snow albedo determined from the snow age
      !
      ! Note that "snoag" is in hours ... (tauday also)

!VDIR NODEP
      do i=1,ni
         if (any('snoroen' == phyinread_list_s(1:phyinread_n))) then
            zsnoro(i) = max(100.,zsnoroen(i)) / rauw
         endif
         if (any('snoroen' == phyinread_list_s(1:phyinread_n)) .or. &
              any('snodpen' == phyinread_list_s(1:phyinread_n))) then
            zsnoma(i) = rauw * zsnoro(i) * zsnodp(i,indx_soil)
         endif
      end do

      ! For the albedo, there are two possibilities:
      !
      ! 1) if switch "snoalb_anl" is true, then the "i6"
      !    record in the starting standard file (snoalen) contains the snow albedo
      !
      ! 2) if switch "snoalb_anl" is false, then we use the snow age (snoagen) 
      !    to derive the snow albedo
      !
      IF_SNO_ALB: if (snoalb_anl) then

         if (any('snoalen' == phyinread_list_s(1:phyinread_n))) then
            do i=1,ni
               zsnoal(i)  =  zsnoalen(i)
            end do
         endif

      else

         ! snow albedo is determined from the snow age according to two different
         ! expressions depending if the snow pack is melting or not

         if (any('snoagen' == phyinread_list_s(1:phyinread_n)) .or. &
              any('snoalen' == phyinread_list_s(1:phyinread_n))) then
!VDIR NODEP
            do i=1,ni
               if (ztmoins(i,nk).lt.trpl) then
                  zsnoal(i)  = ansmax - todry*zsnoagen(i)/tauday
               else
                  zsnoal(i)  = (ansmax-almin) * &
                       exp( -tauf*zsnoagen(i)/tauday ) &
                       + almin
               end if
               zsnoal(i)  = max( zsnoal(i) , almin )
               zsnoal(i)  = min( zsnoal(i) , ansmax )
            end do
         endif

      end if IF_SNO_ALB

      !  Initialize the parameters that depend on vegetation

      if (any('vegfen' == phyinread_list_s(1:phyinread_n))) then
         call inicover2(0, ni, trnch)
      endif

      ! Sand and clay fractions of the soil are taken as simple averages 
      ! of the first 3 layers

!VDIR NODEP
      do i=1,ni
         if (any('sanden' == phyinread_list_s(1:phyinread_n))) then
            zsand(i,1) = (  zsanden(i,1) &
                 + zsanden(i,2) &
                 + zsanden(i,3) ) / 3.
         endif
         if (any('clayen' == phyinread_list_s(1:phyinread_n))) then
            zclay(i,1) = (  zclayen(i,1) &
                 + zclayen(i,2) &
                 + zclayen(i,3) ) / 3.
         endif
      end do

      ! Make sure the entry fields are coherent ...

      call coherence3(ni, trnch)

      ! Initialize the soil characteristics using the soil texture

      if (any('clayen' == phyinread_list_s(1:phyinread_n)) .or. &
           any('sanden' == phyinread_list_s(1:phyinread_n))) then
         call inisoili2(ni, trnch)
      endif

   end if IF_ISBA

   !========================================================================
   !                             for TEB only
   !========================================================================

   ! Note that TEB variables do not support reading for kount>0:  phyincread_list_s
   !  would need to be processed within initown() to implement this support.
   if (kount == 0 .and. schmurb == 'TEB') &
        call initown2(ni, nk, trnch)

   return
end subroutine inisurf4
