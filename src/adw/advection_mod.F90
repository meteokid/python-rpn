module advection_mod
   ! Semi-Lagrangian advection
   !
   ! Author
   !     
   !
   ! Revision
   !

     use adv_interpolation_mod, only: adv_interpol_cubic , adv_cliptraj3
     use adv_trajectories_mod , only: traject_trapeze



  implicit none


   private

#include "adv_dims.cdk"
#include "adv_pos.cdk"
#include "adv_grid.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "glb_ld.cdk"
#include "lam.cdk"
#include "gmm.hf"
#include "grd.cdk"
#include "inuvl.cdk"
#include "schm.cdk"
#include "lctl.cdk"
#include "step.cdk"
#include "orh.cdk"
#include "ver.cdk"
#include "msg.h"
#include "vt0.cdk"
#include "vt1.cdk"
#include "vt2.cdk"
#include "rhsc.cdk"
   public :: advection_semilag

contains
subroutine advection_semilag (F_fnitraj)

      integer  F_fnitraj  ! total number of iterations for trajectories
 	   logical ::  doAdwStat_L  

      real, dimension(l_ni,l_nj,l_nk)  :: pxm   , pym  , pzm
 
      real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk) :: ud,vd,wd
      real, dimension(l_ni,l_nj,l_nk)  :: ua,va,wa,wat
 
      integer :: i0,j0,in,jn,i0u,inu,j0v,jnv,jext, k0, k0t
      integer :: err  

   

    doAdwStat_L = .false.
    if (Step_gstat.gt.0) then
         doAdwStat_L = (mod(Lctl_step,Step_gstat) == 0)
    end if
    doAdwStat_L = doAdwStat_L .and. (Orh_icn == Schm_itcn)
 

      err = gmm_get (gmmk_orhsu_s, orhsu)
      err = gmm_get (gmmk_orhsv_s, orhsv)
      err = gmm_get (gmmk_orhst_s, orhst)
      err = gmm_get (gmmk_orhsc_s, orhsc)
      err = gmm_get (gmmk_orhsf_s, orhsf)
      err = gmm_get (gmmk_orhsw_s, orhsw)
      err = gmm_get (gmmk_orhsx_s, orhsx)
      err = gmm_get (gmmk_orhsq_s, orhsq)

      err = gmm_get (gmmk_rhsu_s, rhsu)
      err = gmm_get (gmmk_rhsv_s, rhsv)
      err = gmm_get (gmmk_rhst_s, rhst)
      err = gmm_get (gmmk_rhsc_s, rhsc)
      err = gmm_get (gmmk_rhsf_s, rhsf)
      err = gmm_get (gmmk_rhsw_s, rhsw)
      err = gmm_get (gmmk_rhsx_s, rhsx)
      err = gmm_get (gmmk_rhsq_s, rhsq)
            
      
      !  TODO : wat=zdt0 ???

      call adv_get_ij0n (i0,in,j0,jn)
      i0u= i0
      inu= in
      j0v= j0
      jnv= jn

      jext = 2
      if (Grd_yinyang_L) jext = 0

      if (l_west)  i0u= i0 + jext
      if (l_east)  inu= in - jext
      if (l_south) j0v= j0 + jext
      if (l_north) jnv= jn - jext

      k0 = Lam_gbpil_t+1
      k0t= k0
      if(Lam_gbpil_t.gt.0) k0t=k0-1

      call adv_prepare_winds ( ud, vd, wd, ua, va, wa, wat , &
                               l_minx,l_maxx,l_miny,l_maxy , &
                               l_nk, l_ni, l_nj ) 



      call traject_trapeze (F_fnitraj, pxm , pym , pzm  ,&
                           ud ,vd ,wd ,ua,va ,wa ,wat ,i0 ,in,j0 ,jn ,k0, k0t, doAdwStat_L  )

!
         
 !RHS interpolation : momentum levels                     
      call adv_cliptraj3 ( pxmu, pymu, i0u, inu, j0, jn, k0, 'INTERP '//trim('m'))


     call adv_interpol_cubic ('RHSU_S', rhsu, orhsu, pxmu, pymu, pzmu,  &
                                 i0u, inu, j0, jn, k0, 'm',0,0)

     call adv_cliptraj3 ( pxmv, pymv, i0, in, j0v, jnv,  k0,  'INTERP '//trim('m'))

      call adv_interpol_cubic ('RHSV_S', rhsv, orhsv, pxmv, pymv, pzmv, &
                                  i0, in, j0v, jnv, k0, 'm',0,0)

      call adv_cliptraj3 ( pxm, pym, i0, in, j0, jn, k0, 'INTERP '//trim('m'))
    
     call adv_interpol_cubic ('RHSC_S', rhsc ,orhsc , pxm, pym, pzm, &
                                 i0, in, j0, jn, k0, 'm',0,0)


! RHS Interpolation: thermo levels
      call adv_cliptraj3 ( pxt, pyt, i0, in, j0, jn,k0, 'INTERP '//trim('t'))
    
       call adv_interpol_cubic ('RHST_S', rhst, orhst, pxt, pyt, pzt,   &
                                 i0, in, j0, jn, k0t, 't',0,0)

       call adv_interpol_cubic ('RHSX_S', rhsx, orhsx, pxt, pyt, pzt,  &
                                 i0, in, j0, jn, k0t, 't',0,0)
       
       if (.not.Schm_hydro_L) then
          call adv_interpol_cubic ('RHSF_S', rhsf, orhsf, pxt, pyt, pzt,  &
                                   i0, in, j0, jn, k0t, 't',0,0)
       endif
       
       if(.not.Schm_hydro_L) then	
         call adv_interpol_cubic ('RHSW_S', rhsw ,orhsw , pxt, pyt, pzt,  &
                                 i0, in, j0, jn, k0t, 't',0,0)
       else
	 call adv_interpol_cubic ('RHSQ_S', rhsq ,orhsq , pxt, pyt, pzt,  &
                                  i0, in, j0, jn, k0t, 't',0,0)
       endif
    
  end subroutine advection_semilag


   subroutine adv_prepare_winds (F_ud, F_vd, F_wd, F_ua, F_va, F_wa, F_wat, &
		                F_minx, F_maxx, F_miny, F_maxy,F_nk ,F_lni, F_lnj )

 implicit none

 ! Process winds in preparation for advection
      integer, intent(in) :: F_minx,F_maxx,F_miny,F_maxy,F_nk,F_lni,F_lnj
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk) :: F_ud, F_vd, F_wd    ! model de-stag winds
      real, dimension(F_lni,F_lnj,F_nk) :: F_ua,F_va,F_wa,F_wat
      real, dimension(:,:,:), allocatable :: uh,vh,wm,wh
      real ::  beta, err
      integer :: i,j,k
 !-----------------------------------------------------------------

   allocate ( uh(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              vh(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              wm(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              wh(l_minx:l_maxx,l_miny:l_maxy,l_nk) )
   
   err = gmm_get(gmmk_ut0_s ,  ut0)
   err = gmm_get(gmmk_vt0_s ,  vt0)
   err = gmm_get(gmmk_zdt0_s, zdt0)
   err = gmm_get(gmmk_ut1_s ,  ut1)
   err = gmm_get(gmmk_vt1_s ,  vt1)
   err = gmm_get(gmmk_zdt1_s, zdt1)

if(Schm_step_settls_L) then
      err = gmm_get(gmmk_ut2_s ,  ut2)
      err = gmm_get(gmmk_vt2_s ,  vt2)
      err = gmm_get(gmmk_zdt2_s, zdt2)
endif


if(.NOT.Schm_step_settls_L) then
      uh = ut0
      vh = vt0

      call adv_destag_wind (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)

      call adv_interp_thermo2mom (wm,zdt0,l_minx,l_maxx,l_miny,l_maxy,l_nk)

! Destag arrival winds
      F_ua = uh(1:l_ni,1:l_nj,1:l_nk)
      F_va = vh(1:l_ni,1:l_nj,1:l_nk)
      F_wa = wm(1:l_ni,1:l_nj,1:l_nk)
      F_wat= zdt0(1:l_ni,1:l_nj,1:l_nk)

      uh = ut1
      vh = vt1
      wh = zdt1
elseif(Schm_step_settls_L) then

      !Set V_a = V(r,t1)
      !-----------------
      uh = ut1 ; vh = vt1
      call adv_destag_wind (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      call adv_interp_thermo2mom (wm,zdt1,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      F_ua =  uh(1:l_ni,1:l_nj,1:l_nk)
      F_va =  vh(1:l_ni,1:l_nj,1:l_nk)
      F_wa =  wm(1:l_ni,1:l_nj,1:l_nk)
      F_wat=zdt1(1:l_ni,1:l_nj,1:l_nk)
       
      !Set V_d = 2*V(r,t1)-V(r,t2)
      !---------------------------
      uh = 2.*ut1-ut2 ; vh = 2.*vt1-vt2 ; wh = 2.*zdt1-zdt2

      !SETTLS limiter according to Diamantakis
      beta=1.9
         !do k=12,20 ! was sufficient on test case
      do k=1,l_nk
         do j=1,l_nj
            do i=1,l_ni
               if(abs(zdt1(i,j,k)-zdt2(i,j,k)).gt.beta*0.5*(abs(zdt1(i,j,k))+abs(zdt2(i,j,k)))) then
                  wh(i,j,k)=zdt1(i,j,k)
               endif
            enddo
         enddo
      enddo

  endif
      call adv_destag_wind (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)

      call adv_interp_thermo2mom (wm,wh,l_minx,l_maxx,l_miny,l_maxy,l_nk)

! Destag departure winds
      F_ud=uh
      F_vd=vh
      F_wd=wm
  
     deallocate(uh,vh,wm,wh)


   end subroutine adv_prepare_winds

   subroutine adv_destag_wind (F_uth, F_vth, minx,maxx,miny,maxy,nk)
      ! Unstagger wind components (Interpolate to mass points)

      implicit none
#include <arch_specific.hf>
      !@objective unstagger wind components (Interpolate to geopotential grid)
      !@arguments

      integer minx,maxx,miny,maxy,nk
      real,intent(inout) ::  F_uth(minx:maxx,miny:maxy,nk), F_vth(minx:maxx,miny:maxy,nk)

      !@revisions
      ! v4_40 - Qaddouri/Lee      - Yin-Yang, to exchange unstaggered winds
      !*@/


      integer :: i,j,k,i0,in,j0,jn,i0u,inu,j0v,jnv,jext
      real, dimension(:,:,:), allocatable :: uu,vv
      real*8  :: inv_rayt_8
      real*8, parameter ::  alpha1=-1.d0/16.d0 , alpha2=9.d0/16.d0
      !---------------------------------------------------------------------

      inv_rayt_8 = 1.D0 / Dcst_rayt_8

      call rpn_comm_xch_halo(F_uth,l_minx,l_maxx,l_miny,l_maxy,&
           l_niu,l_nj,l_nk,G_halox,G_haloy,G_periodx,G_periody,G_niu,0)
      call rpn_comm_xch_halo(F_vth,l_minx,l_maxx,l_miny,l_maxy,&
           l_ni,l_njv,l_nk,G_halox,G_haloy,G_periodx,G_periody,G_ni,0)

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj

      
         jext=2
         if (l_west)  i0u = 1    + jext
         if (l_east)  inu = l_ni - jext
         if (l_south) j0v = 1    + jext
         if (l_north) jnv = l_nj - jext
    

       !- Interpolate advection winds to geopotential grid

      allocate (uu(l_minx:l_maxx,l_miny:l_maxy,l_nk),vv(l_minx:l_maxx,l_miny:l_maxy,l_nk))

!$omp parallel private(i,j,k)
!$omp do
     DO_K: do k=1,l_nk
         do j = j0, jn
            do i = i0u, inu
               uu(i,j,k) = ( F_uth(i-2,j,k) + F_uth(i+1,j,k) )*alpha1 &
                          + ( F_uth(i  ,j,k) + F_uth(i-1,j,k) )*alpha2                         
            enddo
         enddo

         do j = j0v, jnv
            do i = i0, in
               vv(i,j,k) = inuvl_wyvy3_8(j,1)*F_vth(i,j-2,k) + inuvl_wyvy3_8(j,2)*F_vth(i,j-1,k) &
                         + inuvl_wyvy3_8(j,3)*F_vth(i,j  ,k) + inuvl_wyvy3_8(j,4)*F_vth(i,j+1,k)
            enddo
         enddo
    enddo DO_K
!$omp enddo

!$omp do
         do k = 1,l_nk
            do j = j0,jn
               do i = i0u,inu
                  F_uth(i,j,k) = inv_rayt_8 * uu(i,j,k)
               enddo
            enddo
            do j = j0v,jnv
               do i = i0,in
                  F_vth(i,j,k) = inv_rayt_8 * vv(i,j,k)
               enddo
            enddo
        enddo
!$omp enddo
      
!$omp end parallel

   deallocate (uu,vv)
   
  end subroutine adv_destag_wind


   subroutine adv_interp_thermo2mom (F_fld_m, F_fld_t, minx, maxx, miny, maxy, nk)
      ! interpolate from thermodynamic to momentum levels

 implicit none

      integer, intent(in) :: minx,maxx,miny,maxy,nk
      real, dimension(minx:maxx,miny:maxy,nk),intent(in) :: F_fld_t
      real, dimension(minx:maxx,miny:maxy,nk), intent(out) :: F_fld_m

      integer :: i,j,k,km2, i0,j0,in,jn
      real*8  :: xx, x1, x2, x3, x4, w1, w2, w3, w4
      real*8  :: zd_z_8(l_nk+1)

#define lag3(xx, x1, x2, x3, x4)  ((((xx) - (x2)) * ((xx) - (x3)) * ((xx) - (x4)))/( ((x1) - (x2)) * ((x1) - (x3)) * ((x1) - (x4))))

      zd_z_8(1) = Cstv_Ztop_8
      zd_z_8(2:l_nk+1) = Ver_z_8%t(1:l_nk)

!$omp parallel private(i0,in,j0,jn,xx,x1,x2,x3,x4,&
!$omp                  i,j,k,w1,w2,w3,w4)
      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      if (G_lam .and. .not. Grd_yinyang_L) then
         if (l_west)  i0 = 3
         if (l_east)  in = l_ni - 1
         if (l_south) j0 = 3
         if (l_north) jn = l_nj - 1
      endif

!$omp do
      do k=2,l_nk-1
         xx = Ver_z_8%m(k)
         x1 = zd_z_8(k-1)
         x2 = zd_z_8(k)
         x3 = zd_z_8(k+1)
         x4 = zd_z_8(k+2)
         w1 = lag3(xx, x1, x2, x3, x4)
         w2 = lag3(xx, x2, x1, x3, x4)
         w3 = lag3(xx, x3, x1, x2, x4)
         w4 = lag3(xx, x4, x1, x2, x3)

         ! zdt=0 is not present in vector but this allow to use this
         ! boundary condition anyway.
         km2=max(1,k-2)

         if(k.eq.2) then
            w1=0.d0
         end if

         do j = j0, jn
            do i = i0, in
               F_fld_m(i,j,k)= &
                    w1*F_fld_t(i,j,km2) + w2*F_fld_t(i,j,k-1)  + &
                    w3*F_fld_t(i,j,k  ) + w4*F_fld_t(i,j,k+1)

            enddo
         enddo
      enddo
!$omp enddo

      !- Note zdot at top = 0
      k = 1
      w2 = (zd_z_8(k)-Ver_z_8%m(k)) / (zd_z_8(k)-zd_z_8(k+1))

!$omp do
      do j = j0, jn
         do i = i0, in
            F_fld_m(i,j,1) = w2 * F_fld_t(i,j,1)
         enddo
      enddo
!$omp enddo

      !- Note  zdot at surface = 0
      k = l_nk
      w1 = (Ver_z_8%m(k)-zd_z_8(k+1)) / (zd_z_8(k)-zd_z_8(k+1))

!$omp do
      do j = j0, jn
         do i = i0, in
            F_fld_m(i,j,l_nk) = w1 * F_fld_t(i,j,l_nk-1)
         enddo
      enddo
!$omp enddo

!$omp end parallel
   end subroutine adv_interp_thermo2mom


 subroutine adv_get_ij0n(i0,in,j0,jn)
   implicit none
#include <arch_specific.hf>
   !@objective Return advection computational i,j domain
   !@arguments
   integer :: i0,j0,in,jn     !O, scope of advection operations
   !@author  Stephane Chamberland, 2009-12
   !@revisions
   !@ v4_40 - Qaddouri/Lee    - Yin-Yang, expand calculation by 1 point around

   !*@/
   !---------------------------------------------------------------------
   integer :: jext

   i0 = 1
   in = l_ni
   j0 = 1
   jn = l_nj

      jext=1
      if (Grd_yinyang_L) jext=2
      if (l_west)  i0 =        pil_w - jext
      if (l_east)  in = l_ni - pil_e + jext
      if (l_south) j0 =        pil_s - jext
      if (l_north) jn = l_nj - pil_n + jext


   !---------------------------------------------------------------------
   return
end subroutine adv_get_ij0n


end module advection_mod
