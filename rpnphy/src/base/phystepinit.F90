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

subroutine phystepinit1(uplus0,vplus0,wplus0,tplus0,huplus0,qcplus0, &
                        vbus,dbus,fbus,seloc,dt,cdt1,rcdt1,          &
                        vsiz,dsiz,fsiz,kount,trnch,icpu,ni,nk)
   use phy_options
   use phy_itf
   use phybus
   use phygetmetaplus_mod, only: phymetaplus, phygetmetaplus
   implicit none
#include <arch_specific.hf>
  integer                :: vsiz,dsiz,fsiz,kount,trnch,icpu,ni,nk
  real, dimension(ni,nk) :: uplus0,vplus0,wplus0,tplus0,huplus0,qcplus0
  real, dimension(ni,nk) :: seloc
  real, target           :: vbus(vsiz), dbus(dsiz), fbus(fsiz)
  real                   :: dt,cdt1,rcdt1
  !
  !Author
  !          L. Spacek (Oct 2011)
  !
  !Revision
  !
  !Object
  !          Setup for physics timestep
  !
  !Arguments
  !
  !          - Input -
  ! dsiz     dimension of dbus
  ! vsiz     dimension of vbus
  ! dt       timestep (parameter) (sec.)
  ! trnch    slice number
  ! icpu     cpu number executing slice "trnch"
  ! ni       horizontal running length
  ! nk       vertical dimension
  !
  !          - Output -
  ! uplus0   initial value of dbus(uplus)
  ! vplus0   initial value of dbus(vplus)
  ! tplus0   initial value of dbus(tplus)
  ! huplus0  initial value of dbus(huplus)
  ! qcplus0  initial value of dbus(qcplus)
  ! cdt1     timestep (sec.)
  ! rcdt1    1/cdt1
  !
  !          - input/output -
  ! dbus     dynamics input field
  ! vbus     physics tendencies and other output fields from the physics

#include <rmnlib_basics.hf>
  include "surface.cdk"
  include "phyinput.cdk"
  include "thermoconsts.inc"
  
  logical,parameter:: SHORTMATCH_L = .true.
  integer,parameter:: MYMAX = 1024

  character(len=32) :: prefix_S,basename_S,time_S,ext_S
!!$  character(len=32) :: varlist_S(MYMAX),prefix_S,basename_S,time_S,ext_S
  integer                :: i,j,k,ierget,ivar,nvars,idxm,idxp,nik,istat
  real                   :: sc
  real, dimension(ni,nk) :: work
  real, dimension(ni,nk) :: gzmoins, qe

  type(phymetaplus) :: meta_m, meta_p
  type(phymeta), pointer :: metalist(:)

  real, pointer, dimension(:)   :: zdlat, zfcor, zpmoins, ztdiag, zthetaa, &
                                   zqdiag, zza, zztsl, zzusl, zme, zp0, &
                                   zudiag, zvdiag
  real, pointer, dimension(:,:) :: zgzmom, zgz_moins, zhumoins, zhuplus, &
         zqadv, zqcmoins, zqcplus, zsigm, zsigt, ztadv, ztmoins, ztplus, &
         zuadv, zumoins, zuplus, zvadv, zvmoins, zvplus, zwplus, zze, &
         zgztherm
  real, pointer, dimension(:,:) :: tmp1,tmp2
  real, pointer, dimension(:) :: tmp1b,tmp2b

  zdlat    (1:ni)      => fbus(dlat:)
  zfcor    (1:ni)      => vbus(fcor:)
  zpmoins  (1:ni)      => fbus(pmoins:)
  zqdiag   (1:ni)      => fbus(qdiag:)
  ztdiag   (1:ni)      => fbus(tdiag:)
  zudiag   (1:ni)      => fbus(udiag:)
  zvdiag   (1:ni)      => fbus(vdiag:)
  zthetaa  (1:ni)      => vbus(thetaa:)
  zza      (1:ni)      => vbus(za:)
  zztsl    (1:ni)      => vbus(ztsl:)
  zzusl    (1:ni)      => vbus(zusl:)
  zme      (1:ni)      => dbus(me_moins:)
  zp0      (1:ni)      => dbus(p0_moins:)
  zgzmom   (1:ni,1:nk) => vbus(gzmom:)
  zgztherm (1:ni,1:nk) => vbus(gztherm:)
  zgz_moins(1:ni,1:nk) => dbus(gz_moins:)
  zhumoins (1:ni,1:nk) => dbus(humoins:)
  zhuplus  (1:ni,1:nk) => dbus(huplus:)
  zqadv    (1:ni,1:nk) => vbus(qadv:)
  zqcmoins (1:ni,1:nk) => dbus(qcmoins:)
  zqcplus  (1:ni,1:nk) => dbus(qcplus:)
  zsigm    (1:ni,1:nk) => dbus(sigm:)
  zsigt    (1:ni,1:nk) => dbus(sigt:)
  ztadv    (1:ni,1:nk) => vbus(tadv:)
  ztmoins  (1:ni,1:nk) => dbus(tmoins:)
  ztplus   (1:ni,1:nk) => dbus(tplus:)
  zuadv    (1:ni,1:nk) => vbus(uadv:)
  zumoins  (1:ni,1:nk) => dbus(umoins:)
  zuplus   (1:ni,1:nk) => dbus(uplus:)
  zvadv    (1:ni,1:nk) => vbus(vadv:)
  zvmoins  (1:ni,1:nk) => dbus(vmoins:)
  zvplus   (1:ni,1:nk) => dbus(vplus:)
  zze      (1:ni,1:nk) => vbus(ze:)

!  call statvps (dbus,kount,trnch,'sinit',ni,nk,'D')
!  call statvps (fbus,kount,trnch,'sinit',ni,nk,'P')
!  call statvps (vbus,kount,trnch,'sinit',ni,nk,'V')

  cdt1  = dt
  rcdt1 = 1./cdt1

  do k=1,nk-1
  do i=1,ni
     zsigm    (i,k) = zsigm    (i,k)/zp0(i)
     zsigt    (i,k) = zsigt    (i,k)/zp0(i)
  end do
  end do
  zsigm    (:,nk) = 1.
  zsigt    (:,nk) = 1.

  zpmoins(:) = zp0(:)

  vbus=0.0

  call sigmalev(seloc,dbus(sigm),dbus(sigt),vbus,vsiz,ni,nk)

  where(zhuplus <0.) zhuplus  = 0.
  where(zhumoins<0.) zhumoins = 0.

  do k=1,nk
  do i=1,ni
     huplus0(i,k) = zhuplus(i,k)
     qcplus0(i,k) = zqcplus(i,k)
     uplus0 (i,k) = zuplus (i,k)
     vplus0 (i,k) = zvplus (i,k)
     tplus0 (i,k) = ztplus (i,k)
  enddo
  enddo
  if(diffuw)then
     zwplus  (1:ni,1:nk) => dbus(wplus:)
     do k=1,nk
     do i=1,ni
        wplus0(i,k)  = zwplus(i,k)
     enddo
     enddo
  endif
  !
  !***********************************************************************
  !     calcul des tendances de la dynamique                             *
  !     ----------------------------------------                         *
  !***********************************************************************
  !
  call sersetm('KA', icpu, nk)

  do k = 1,nk
  do i = 1,ni
     ztadv(i,k) = (ztplus (i,k) - ztmoins (i,k)) * rcdt1
     zqadv(i,k) = (zhuplus(i,k) - zhumoins(i,k)) * rcdt1
     zuadv(i,k) = (zuplus (i,k) - zumoins (i,k)) * rcdt1
     zvadv(i,k) = (zvplus (i,k) - zvmoins (i,k)) * rcdt1
  end do
  end do

  call serxst2(vbus(tadv), 'XT', trnch, ni, nk, 0., 1., -1)
  call serxst2(vbus(qadv), 'XQ', trnch, ni, nk, 0., 1., -1)

  if (.not. any(trim(stcond) == (/'CONDS','NIL  '/))) then
     do k = 1,nk
     do i = 1,ni
        work(i,k) = (zqcplus(i,k) - zqcmoins(i,k)) * rcdt1
     enddo
     enddo
     call serxst2(work, 'XL', trnch, ni, nk, 0., 1., -1)
  endif

  call sersetm('KA', icpu, nk-1)

  ! Precompute heights for the paramterizations
  do k=1,nk-1
     zgzmom(:,k) = (zgz_moins(:,k)-zme)/grav
  enddo
  zgzmom(:,nk) = 0.
  call tothermo(zgzmom,zgztherm,vbus(au2t),vbus(au2t),ni,nk,nk,.true.)
  zze = zgztherm
  zze(:,nk-1:nk) = 0.
  zza = zgzmom(:,nk-1)
  zzusl = zza
  zztsl = zgztherm(:,nk-1)

         !     z0 directionnel
         if (z0dir) &
!        calcul de z0 avec z1,z2,z3,z4 et umoins,vmoins
         call calcz0(fbus(mg),fbus(z0),fbus(z1),fbus(z2),fbus(z3),fbus(z4), &
                     dbus(umoins+(nk-2)*ni), &
                     dbus(vmoins+(nk-2)*ni), ni)
!
!       calcul du facteur de coriolis (fcor), de la hauteur du
!       dernier niveau actif (za) et de la temperature potentielle
!       a ce niveau (thetaa)
        call tothermo(dbus(tmoins), vbus(tve), vbus(at2t),vbus(at2m),&
                                                  ni,nk,nk-1,.true.)
        call tothermo(dbus(humoins),qe,        vbus(at2t),vbus(at2m),&
                                                  ni,nk,nk-1,.true.)

!       Initialize diagnostic level values in the profile
        if (any('pw_tt:p' == phyinread_list_s(1:phyinread_n))) then
           ztdiag = ztplus(:,nk)
        elseif (kount == 0) then
           ztplus(:,nk) = ztplus(:,nk-1)
           ztdiag = ztplus(:,nk)
        endif
        if (any('tr/hu:p' == phyinread_list_s(1:phyinread_n))) then
           zqdiag = zhuplus(:,nk)
        elseif (kount  ==  0) then
           zhuplus(:,nk) = zhuplus(:,nk-1)
           zqdiag = zhuplus(:,nk)
        endif
        if (any('pw_uu:p' == phyinread_list_s(1:phyinread_n))) then
           zudiag = zuplus(:,nk)
        elseif (kount  ==  0) then
           zuplus(:,nk) = zuplus(:,nk-1)
           zudiag = zuplus(:,nk)
        endif
        if (any('pw_vv:p' == phyinread_list_s(1:phyinread_n))) then
           zvdiag = zvplus(:,nk)
        elseif (kount == 0) then
           zvplus(:,nk) = zvplus(:,nk-1)
           zvdiag = zvplus(:,nk)
        endif
        ztmoins(:,nk) = ztdiag
        zhumoins(:,nk) = zqdiag
        zumoins(:,nk) = zudiag
        zvmoins(:,nk) = zvdiag

        nullify(metalist)
        nvars = phy_getmeta(metalist, 'tr/', F_npath='V', F_bpath='D', &
             F_maxmeta=MYMAX, F_shortmatch=SHORTMATCH_L)
        do ivar = 1,nvars
           call gmmx_name_parts(metalist(ivar)%vname, prefix_S, basename_S, &
                time_S, ext_S)
           if (all(time_S /= (/':M',':m'/)) .and. &
                .not.any(metalist(ivar)%vname == (/'tr/hu:m', 'tr/hu:p'/)) &
               ) then
              istat = phygetmetaplus(meta_m, &
                   trim(prefix_S)//trim(basename_S)//':M', F_npath='V', &
                   F_bpath='D', F_quiet=.true., F_shortmatch=.false.)
              istat = min(phygetmetaplus(meta_p, &
                   trim(prefix_S)//trim(basename_S)//':P', F_npath='V', &
                   F_bpath='D', F_quiet=.true., F_shortmatch=.false.),istat)
              if (RMN_IS_OK(istat)) then
                 !#TODO: use  meta_m%ptr instead of dbus (bitpattern change)
!!$                 tmp1b(1:ni*nk) => meta_m%ptr(meta_m%index:meta_m%index+ni*nk-1,trnch)
!!$                 tmp2b(1:ni*nk) => meta_p%ptr(meta_p%index:meta_p%index+ni*nk-1,trnch)
!!$                 tmp1b = tmp2b
                 tmp1(1:ni,1:nk) => dbus(meta_m%index:)
                 tmp2(1:ni,1:nk) => dbus(meta_p%index:)
                 tmp1(1:ni,nk) = tmp2(1:ni,nk)
              endif
           endif
        enddo
        deallocate(metalist,stat=istat)

        call mfotvt(vbus(tve),vbus(tve),qe,ni,nk-1,ni)

        do i=1,ni
          sc = zsigt(i,nk-1)**(-cappa)
          zthetaa(i) = sc*ztmoins(i,nk-1)
          zfcor  (i)= 1.45441e-4*sin(zdlat(i))

        end do
        if(zua.gt.0..and.zta.gt.0.) then
           do i=1,ni
             zztsl(i) = zta
             zzusl(i) = zua
           enddo
        endif
        
        if (any(pcptype == (/'NIL   ', 'BOURGE'/))) then
           call surf_precip(dbus(tmoins+(nk-2)*ni), &
                fbus(tlc), fbus(tls), fbus(tsc), fbus(tss), &
                vbus(rainrate), vbus(snowrate), ni)
        elseif (pcptype == 'BOURGE3D') then
           call surf_precip3(dbus(tmoins+(nk-2)*ni),fbus(tlc),fbus(tls), &
                fbus(tlcs), fbus(tsc), fbus(tss), fbus(tscs), &
                vbus(rainrate), vbus(snowrate),&
                fbus(fneige+(nk-1)*ni),fbus(fip+(nk-1)*ni),ni)
        endif

   !-------------------------------------------------------------
   return
end subroutine phystepinit1
