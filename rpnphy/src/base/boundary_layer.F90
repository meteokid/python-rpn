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
!** S/P boundary_layer
subroutine boundary_layer2 ( d,   f,   v, dsiz, fsiz, vsiz  , &
     ficebl, seloc, cdt1, &
     kount, trnch, icpu, ni, nk )
   use phy_options
   use phybus
   implicit none
#include <arch_specific.hf>
  
  integer dsiz,fsiz,vsiz,trnch,kount,icpu,ni,nk
  real, target :: d(dsiz), f(fsiz), v(vsiz)
  real ficebl(ni,nk), seloc(ni,nk), cdt1
  
  !Author
  !          M. Desgagne summer 2011
  !
  !Revisions
  ! 001      M. Desgagne (Sept. 2011) - initial version
  !
  !Object
  !
  !Arguments
  !
  !          - Input -
  ! E        entry    input field
  ! D        dynamics input field
  !
  !          - Input/Output -
  ! F        historic variables for the physics
  !
  !          - Output -
  ! V        physics tendencies and other output fields from the physics
  !
  !          - Input -
  ! ESIZ     dimension of e
  ! DSIZ     dimension of d
  ! FSIZ     dimension of f
  ! VSIZ     dimension of v
  ! DT       timestep (sec.)
  ! TRNCH    slice number
  ! KOUNT    timestep number
  ! ICPU     cpu number executing slice "trnch"
  ! N        horizontal running length
  ! NK       vertical dimension
  !

  integer i,k
  real wk1(ni,nk), wk2(ni,nk), qe(ni,nk), rcdt1
! Pointers to busdyn
      real, pointer, dimension(:,:) :: zqmoins, zqplus, zqcplus, ztplus, &
                                       zuplus, zvplus,  zwplus
! Pointers to busper
      real, pointer, dimension(:)   :: zqdiag, ztdiag, zudiag, zvdiag
! Pointers to busvol
      real, pointer, dimension(:,:) :: zldifv, zqdifv, ztdifv, &
                                       zudifv, zvdifv, zwdifv
!! Pointers to buses

! Pointers to busdyn
      zqplus (1:ni,1:nk) => d( huplus:)
      ztplus (1:ni,1:nk) => d( tplus:)
      zuplus (1:ni,1:nk) => d( uplus:)
      zvplus (1:ni,1:nk) => d( vplus:)
      zwplus (1:ni,1:nk) => d( wplus:)
      zqmoins(1:ni,1:nk) => d( humoins:)
      if (fluvert == 'MOISTKE') &
           zqcplus(1:ni,1:nk) => d( qcplus:)
! Pointers to busper
      zqdiag (1:ni)      => f( qdiag:)
      ztdiag (1:ni)      => f( tdiag:)
      zudiag (1:ni)      => f( udiag:)
      zvdiag (1:ni)      => f( vdiag:)
! Pointers to busvol
      zqdifv (1:ni,1:nk) => v( qdifv:)
      ztdifv (1:ni,1:nk) => v( tdifv:)
      zudifv (1:ni,1:nk) => v( udifv:)
      zvdifv (1:ni,1:nk) => v( vdifv:)
      if (diffuw) &
           zwdifv (1:ni,1:nk) => v( wdifv:)
      if (fluvert == 'MOISTKE') &
           zldifv (1:ni,1:nk) => v( ldifv:)
  
  !----------------------------------------------------------------
  
  rcdt1 = 1./cdt1
  
  !***********************************************************************
  !        energie cinetique turbulente, operateurs de diffusion         *
  !        et hauteur de la couche limite stable ou instable             *
  !***********************************************************************
  
  if (any(fluvert == (/'MOISTKE', 'CLEF   '/)))  then

     call TOTHERMO(D(HUMOINS), qe, V(AT2T), V(AT2M), NI, NK, NK-1, .true.)
     do i=1,ni
        qe(i,nk-1)  = zqmoins(i,nk)
     end do

     ! Remove this once coupled moistke is implemented
     if (fluvert == 'MOISTKE') then
        print *, 'STOP in boundary_layer(): moistke not implemented for PBL_COUPLED=.true.'
        stop
     endif

     call pbl_turbul(d, dsiz, f, fsiz, v, vsiz, qe, seloc, &
          kount, trnch, ni, ni, nk-1, icpu)

  endif
  
  !***********************************************************************
  !     diffusion verticale                                              *
  !***********************************************************************
  call pbl_difver1(d, dsiz, f, fsiz, v, vsiz, seloc, &
       cdt1, kount, trnch, ni, nk-1, icpu)

  !# application des tendances de la diffusion
  if (any(fluvert == (/'PHYSIMP', 'MOISTKE', 'CLEF   '/)))  then
     call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,huplus,qdifv,ni,nk-1)
     call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,tplus, tdifv,ni,nk-1)
     call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,uplus, udifv,ni,nk-1)
     call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,vplus, vdifv,ni,nk-1)
     if (fluvert == 'MOISTKE') &
          call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,qcplus,ldifv,ni,nk-1)
     if (diffuw) &
          call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,wplus, wdifv,ni,nk-1)
  endif

  ! Time-flip low resolution boundary layer state
  if (kount > 0) then
     call ccpl_timeflip(f(pblm_umoins),d(uplus),ni,nk-1)
     call ccpl_timeflip(f(pblm_vmoins),d(vplus),ni,nk-1)
     call ccpl_timeflip(f(pblm_tmoins),d(tplus),ni,nk-1)
     call ccpl_timeflip(f(pblm_humoins),d(huplus),ni,nk-1)
     call ccpl_timeflip(f(pblm_qcmoins),d(qcplus),ni,nk-1)
  endif

  if (fluvert == 'MOISTKE') then
     !# BL ice fraction for later use (in cloud water section)
     call ficemxp (ficebl, wk1, wk2, d(tplus), ni, ni, nk-1)
  endif

  return
end subroutine boundary_layer2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Model-gridded PBL scheme calls used if PBL_COUPLED=.false. ... to be removed asap (ron)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine boundary_layer_modlevs2 ( d,   f,   v, dsiz, fsiz, vsiz  , &
                                  ficebl, seloc, cdt1, &
                                  kount, trnch, icpu, ni, nk )
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>
!
      integer dsiz,fsiz,vsiz,trnch,kount,icpu,ni,nk
      real, target :: d(dsiz), f(fsiz), v(vsiz)
      real ficebl(ni,nk), seloc(ni,nk), cdt1
!
!Author
!          M. Desgagne summer 2011
!
!Revisions
! 001      M. Desgagne (Sept. 2011) - initial version
!
!Object
!
!Arguments
!
!          - Input -
! E        entry    input field
! D        dynamics input field
!
!          - Input/Output -
! F        historic variables for the physics
!
!          - Output -
! V        physics tendencies and other output fields from the physics
!
!          - Input -
! ESIZ     dimension of e
! DSIZ     dimension of d
! FSIZ     dimension of f
! VSIZ     dimension of v
! DT       timestep (sec.)
! TRNCH    slice number
! KOUNT    timestep number
! ICPU     cpu number executing slice "trnch"
! N        horizontal running length
! NK       vertical dimension
!
      integer i,k
      real wk1(ni,nk), wk2(ni,nk), qe(ni,nk), rcdt1
! Pointers to busdyn
      real, pointer, dimension(:,:) :: zqmoins, zqplus, zqcplus, ztplus, &
                                       zuplus, zvplus,  zwplus
! Pointers to busper
      real, pointer, dimension(:)   :: zqdiag, ztdiag, zudiag, zvdiag
! Pointers to busvol
      real, pointer, dimension(:,:) :: zldifv, zqdifv, ztdifv, &
                                       zudifv, zvdifv, zwdifv
!! Pointers to buses
!include "phybuses.inc"
!
! Pointers to busdyn
      zqplus (1:ni,1:nk) => d( huplus:)
      ztplus (1:ni,1:nk) => d( tplus:)
      zuplus (1:ni,1:nk) => d( uplus:)
      zvplus (1:ni,1:nk) => d( vplus:)
      zwplus (1:ni,1:nk) => d( wplus:)
      zqmoins(1:ni,1:nk) => d( humoins:)
      if (fluvert=='MOISTKE') &
           zqcplus(1:ni,1:nk) => d( qcplus:)
! Pointers to busper
      zqdiag (1:ni)      => f( qdiag:)
      ztdiag (1:ni)      => f( tdiag:)
      zudiag (1:ni)      => f( udiag:)
      zvdiag (1:ni)      => f( vdiag:)
! Pointers to busvol
      zqdifv (1:ni,1:nk) => v( qdifv:)
      ztdifv (1:ni,1:nk) => v( tdifv:)
      zudifv (1:ni,1:nk) => v( udifv:)
      zvdifv (1:ni,1:nk) => v( vdifv:)
      if (diffuw) &
           zwdifv (1:ni,1:nk) => v( wdifv:)
      if (fluvert == 'MOISTKE') &
           zldifv (1:ni,1:nk) => v( ldifv:)
!
!
!----------------------------------------------------------------
!
      rcdt1 = 1./cdt1
!
!***********************************************************************
!
!        energie cinetique turbulente, operateurs de diffusion         *
!        -----------------------------------------------------         *
!                                                                      *
!        et hauteur de la couche limite stable ou instable             *
!        -------------------------------------------------             *
!                                                                      *
!***********************************************************************
!
!
      if (any(fluvert == (/'MOISTKE', 'CLEF   '/)))  then

         call tothermo(d(humoins),qe,v(at2e),v(at2e),ni,nk,nk-1,.true.)

         call turbul ( d, dsiz, f, fsiz, v, vsiz, qe, seloc, &
                       kount, trnch, ni, ni, nk-1, icpu )
!
      endif
!
!***********************************************************************
!     diffusion verticale                                              *
!     -------------------                                              *
!***********************************************************************
!
!

      if (fluvert == 'NIL' .or. &
         (fluvert == 'PHYSIMP' .and..not.drag)) then
         !# volbus already set to zero in phystepinit
      else

         !# calcul des tendances de la diffusion au niveau diagnostique
         !# (dont les series temporelles sont extraites dans difver6)

         do i=1,ni
            zqdifv (i,nk) = (zqdiag(i) - zqplus(i,nk))*rcdt1
            ztdifv (i,nk) = (ztdiag(i) - ztplus(i,nk))*rcdt1
            zudifv (i,nk) = (zudiag(i) - zuplus(i,nk))*rcdt1
            zvdifv (i,nk) = (zvdiag(i) - zvplus(i,nk))*rcdt1
            zqplus(i,nk) =   zqdiag(i)
            ztplus (i,nk) =  ztdiag(i)
            zuplus (i,nk) =  zudiag(i)
            zvplus (i,nk) =  zvdiag(i)
         end do

      endif

      call difver7 ( d, dsiz, f, fsiz, v, vsiz, seloc, &
                     cdt1, kount, trnch, ni, nk-1, icpu )

      !     application des tendances de la diffusion
      !     -----------------------------------------

      if (any(fluvert == (/'PHYSIMP', 'MOISTKE', 'CLEF   '/)))  then
         call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,huplus,qdifv,ni,nk-1)
         call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,tplus, tdifv,ni,nk-1)
         call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,uplus, udifv,ni,nk-1)
         call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,vplus, vdifv,ni,nk-1)
         if (fluvert == 'MOISTKE') &
              call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,qcplus,ldifv,ni,nk-1)
         if (diffuw)&
              call apply_tendencies1 (d,dsiz,v,vsiz,f,fsiz,wplus, wdifv,ni,nk-1)
      endif

      if (fluvert == 'MOISTKE') then
         !# BL ice fraction for later use (in cloud water section)
         call ficemxp (ficebl, wk1, wk2, d(tplus), ni, ni, nk-1)
      endif

    end subroutine boundary_layer_modlevs2
