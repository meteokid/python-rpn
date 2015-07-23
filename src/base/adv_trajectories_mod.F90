module adv_trajectories_mod
   ! Find departure points
   !
   ! Author
   !    
   !
   ! Revision
   !


   use  adv_interpolation_mod , only : interp_cubic_winds
   implicit none

   private

#include <gmm.hf>
#include "adv_nml.cdk"
#include "adv_dims.cdk"
#include "adv_pos.cdk"
#include "adv_grid.cdk"
#include "glb_ld.cdk"
#include "adv_interp.cdk"
#include "inuvl.cdk"
#include "cstv.cdk"
#include "vth.cdk"
   public :: traject_trapeze

contains
   subroutine traject_trapeze (F_nb_iter, pxm , pym , pzm , &
                                 ud, vd, wd, ua, va, wa, wat, i0 ,in,j0 ,jn ,k0, k0t ,doAdwStat_L  )


      integer  F_nb_iter     , err                                            ! total number of iterations for traj
      logical :: doAdwStat_L 
      integer, intent(in) :: k0 , k0t                                                                   ! scope of the operation k0 to l_nk
      integer, intent(in) :: i0, j0 , in, jn 						 !0 , scope of advection operations
  !    real, dimension(l_ni,l_nj,l_nk) :: pxm  , pym  , pzm                                 ! upstream positions valid at t1
      real, dimension(l_ni,l_nj,l_nk), intent(out) :: pxm  , pym  , pzm

       real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk), intent(in) :: ud   , vd  , wd  ! real destag winds
      real, dimension(l_ni,l_nj,l_nk), intent(in) :: ua,   va,   wa                            ! Arival winds
      real, dimension(l_ni,l_nj,l_nk) :: wdm
      real, dimension(l_ni,l_nj,l_nk),intent(in) ::  wat


        !------------------------------------------------------------------------
      if (.not.associated(pxt)) allocate (pxt(l_ni,l_nj,l_nk), &
                                       pyt(l_ni,l_nj,l_nk), &
                                       pzt(l_ni,l_nj,l_nk), &
                                       pxmu(l_ni,l_nj,l_nk), &
                                       pymu(l_ni,l_nj,l_nk), &
                                       pzmu(l_ni,l_nj,l_nk), &
                                       pxmv(l_ni,l_nj,l_nk), &
                                       pymv(l_ni,l_nj,l_nk), &
                                      pzmv(l_ni,l_nj,l_nk)  )


nullify (xth, yth, zth)

     err = gmm_get(gmmk_xth_s , xth)
     err = gmm_get(gmmk_yth_s , yth)
     err = gmm_get(gmmk_zth_s , zth)

      call traject_moment_pos (F_nb_iter, pxm , pym , pzm , ud, vd, wd, ua, va, wa, wdm , &
                              xth , yth, zth, i0,j0,in,jn,max(k0t-2,1))


      call interp_hori_moment_pos (pxmu, pymu, pzmu, pxmv, pymv, pzmv, pxm, pym, pzm, k0, i0,in,j0,jn)



      call interp_vert_thermo_pos (pxt,pyt,pzt,pxm,pym,pzm,wat,wdm,k0t,i0,in,j0,jn, .true.)

if ( doAdwStat_L ) then 
         call  adv_cfl_lam3 (pxm, pym, pzm, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'m')
         call  adv_cfl_lam3 (pxt, pyt, pzt, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'t')                                        
endif

end subroutine traject_trapeze


   subroutine traject_moment_pos(F_nb_iter    ,        &
                                    F_px  ,F_py  ,F_pz  , &
                                    F_u   ,F_v   ,F_w   , &
                                    F_ua  ,F_va  ,F_wa  , F_wdm, &
                                    F_xth  , F_yth  , F_zth , &
                                    i0 ,j0 ,in ,jn , k0)
      ! Calculate upstream positions at t1 using angular displacement

      integer, intent(in) :: F_nb_iter                                                                     ! total number of iterations for traj
       integer, intent(in) :: k0                                                                            ! scope of the operation k0 to l_nk
      real, dimension(l_ni,l_nj,l_nk), intent(out) :: F_px  , F_py  , F_pz                                 ! upstream positions valid at t1
      real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk), intent(in) :: F_u   , F_v   ! real destag winds, may be on super grid
     real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk), intent(in),target :: F_w
      real, dimension(l_ni,l_nj,l_nk), intent(in) :: F_ua,   F_va,   F_wa                            ! Arival winds
      real, dimension(l_ni,l_nj,l_nk), intent(inout) :: F_wdm
      real, dimension(l_ni,l_nj,l_nk),intent(inout):: F_xth  , F_yth  , F_zth

      logical,parameter :: CLIP_TRAJ = .true.
      logical,parameter :: DO_W      = .false.
      logical,parameter :: DO_UV     = .true.

      integer :: i, j, k, iter  
      integer :: i0,in,j0,jn
      real    :: dth
      real, dimension(l_ni,l_nj,l_nk) :: u_d,v_d,w_d

      real,   dimension(:,:,:), pointer :: dummy3d
      real :: ztop_bound, zbot_bound
      real*8 :: inv_cy_8

      !---------------------------------------------------------------------

      dummy3d => F_w ! TODO : c'est laid ...

      dth  = 0.5 * cstv_dt_8

      ztop_bound=adv_verZ_8%m(0)
      zbot_bound=adv_verZ_8%m(l_nk+1)

     
  

      do iter = 1, F_nb_iter

         call interp_cubic_winds (u_d,v_d, F_u,F_v, F_xth,F_yth,F_zth, &
                                  DO_UV, i0,in,j0,jn,k0)

!omp parallel private (inv_cy_8)
!omp do
         do k=k0,l_nk
            do j=j0,jn
               inv_cy_8 = 1.d0 / adv_cy_8(j)
               do i=i0,in
                  F_xth(i,j,k) = adv_xx_8(i) - (u_d(i,j,k) / cos(F_yth(i,j,k)) + F_ua(i,j,k) * inv_cy_8) * dth
                  F_yth(i,j,k) = adv_yy_8(j) - (v_d(i,j,k) + F_va(i,j,k)) * dth
                 end do
            end do
         enddo
!omp enddo
!omp end parallel

 
         !- 3D interpol of zeta dot and new upstream pos along zeta

	  call interp_cubic_winds (w_d,v_d, F_w,dummy3d, F_xth,F_yth,F_zth, &
                                  DO_W, i0,in,j0,jn,k0)
!$omp parallel
!$omp do
         do k = max(1,k0),l_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) = adv_verZ_8%m(k) - (w_d(i,j,k)+F_wa(i,j,k))*dth
                  F_zth(i,j,k) = min(zbot_bound,max(f_zth(i,j,k),ztop_bound))
               enddo
            enddo
         enddo

!$omp enddo
!$omp end parallel
      end do

      F_px = F_xth
      F_py = F_yth
      F_pz = F_zth
      F_wdm = w_d


   end subroutine traject_moment_pos


   subroutine interp_hori_moment_pos ( F_xmu, F_ymu, F_zmu, F_xmv, F_ymv, F_zmv, &
                                      F_xm, F_ym, F_zm,F_k0,i0,in,j0,jn)
!
   implicit none
#include <arch_specific.hf>
!
   integer :: F_k0,i0,in,j0,jn
   real, dimension(l_ni,l_nj,l_nk),intent(out) :: F_xmu,F_ymu,F_zmu
   real, dimension(l_ni,l_nj,l_nk),intent(out) :: F_xmv,F_ymv,F_zmv
   real, dimension(l_ni,l_nj,l_nk),intent(in) :: F_xm,F_ym,F_zm
!
!authors
!     A. Plante & C. Girard
!
!revision
!
!object
!
!arguments
!______________________________________________________________________
!              |                                                 |     |
! NAME         | DESCRIPTION                                     | I/O |
!--------------|-------------------------------------------------|-----|
!              |                                                 |     |
! F_xt         | upwind longitudes for themodynamic level        |  o  |
! F_yt         | upwind latitudes for themodynamic level         |  o  |
! F_zt         | upwind height for themodynamic level            |  o  |
! F_xm         | upwind longitudes for momentum level            |  i  |
! F_ym         | upwind latitudes for momentum level             |  i  |
! F_zm         | upwind height for momentum level                |  i  |
!______________|_________________________________________________|_____|
!                      |
!----------------------------------------------------------------------
!

!***********************************************************************
      integer i,j,k,i0u,inu,j0v,jnv
      real*8 aa, bb, cc, dd

     real, pointer, dimension(:,:,:) :: pxh,pyh,pzh
     logical,save :: done = .false.
!
      nullify (pxh,pyh,pzh)
!***********************************************************************
      allocate(pxh(-1:l_ni+2,-1:l_nj+2,l_nk))
      allocate(pyh(-1:l_ni+2,-1:l_nj+2,l_nk))
      allocate(pzh(-1:l_ni+2,-1:l_nj+2,l_nk))

      do k=F_k0,l_nk
      do j=1, l_nj
      do i=1, l_ni
         pxh(i,j,k)=F_xm(i,j,k)
         pyh(i,j,k)=F_ym(i,j,k)
        pzh(i,j,k)=F_zm(i,j,k)
      enddo
      enddo
      enddo

 
      call rpn_comm_xch_halo(pxh,-1,l_ni+2,-1,l_nj+2,l_ni,l_nj,l_nk,2,2,.false.,.false.,l_ni,0)
      call rpn_comm_xch_halo(pyh,-1,l_ni+2,-1,l_nj+2,l_ni,l_nj,l_nk,2,2,.false.,.false.,l_ni,0)
      call rpn_comm_xch_halo(pzh,-1,l_ni+2,-1,l_nj+2,l_ni,l_nj,l_nk,2,2,.false.,.false.,l_ni,0)

      if(.not.done) then
	 F_xmu=F_xm;F_ymu=F_ym;F_zmu=F_zm;F_xmv=F_xm;F_ymv=F_ym;F_zmv=F_zm
         done = .true.
      endif

      i0u=i0
      inu=in
      j0v=j0
      jnv=jn
      if(l_west) i0u=i0+1
      if(l_east) inu=in-2
      if(l_south) j0v=j0+1
      if(l_north) jnv=jn-2

      aa=-0.0625d0
      bb=+0.5625d0
      cc=adv_dlx_8(l_ni/2)*0.5d0




      do k=F_k0,l_nk
         do j=j0,jn
         do i=i0u,inu
            F_xmu(i,j,k) =  aa*(pxh(i-1,j,k)+pxh(i+2,j,k)) &
                          + bb*(pxh(i  ,j,k)+pxh(i+1,j,k)) - cc
             F_ymu(i,j,k) =  aa*(pyh(i-1,j,k)+pyh(i+2,j,k)) &
                          + bb*(pyh(i  ,j,k)+pyh(i+1,j,k))
            F_zmu(i,j,k) =  aa*(pzh(i-1,j,k)+pzh(i+2,j,k)) &
                          + bb*(pzh(i  ,j,k)+pzh(i+1,j,k))
         end do
         end do
         do j=j0v,jnv

         do i=i0,in

            F_xmv(i,j,k) =  aa*(pxh(i,j-1,k)+pxh(i,j+2,k)) &
                          + bb*(pxh(i,j  ,k)+pxh(i,j+1,k))

            F_ymv(i,j,k) =  aa*(pyh(i,j-1,k)+pyh(i,j+2,k)) &
                          + bb*(pyh(i,j  ,k)+pyh(i,j+1,k)) - cc

            F_zmv(i,j,k) =  aa*(pzh(i,j-1,k)+pzh(i,j+2,k)) &
                          + bb*(pzh(i,j  ,k)+pzh(i,j+1,k))
         enddo
         enddo

      enddo

    deallocate(pxh,pyh,pzh)

   end subroutine interp_hori_moment_pos


  subroutine interp_vert_thermo_pos ( F_xt, F_yt, F_zt, F_xm, F_ym, F_zm, F_wat, F_wdm, F_k0,i0,in,j0,jn,F_cubic_xy_L)
!
   implicit none

#include "constants.h"
#include <arch_specific.hf>
!
   integer :: F_k0
   integer i0,in,j0,jn,k00
   real, dimension(l_ni,l_nj,l_nk) :: F_xt,F_yt,F_zt
   real, dimension(l_ni,l_nj,l_nk) :: F_xm,F_ym,F_zm
   real, dimension(l_ni,l_nj,l_nk) :: F_wat,F_wdm
   logical :: F_cubic_xy_L
!
!authors
!     A. Plante & C. Girard based on adv_meanpos from sylvie gravel
!
!revision
!     A. Plante - nov 2011 - add vertial scope for top piloting
!     C. Girard, S. Gaudreault - Feb 2014 - Simplification for angular displacement
!
!object
!     see id section
!
!arguments
!______________________________________________________________________
!              |                                                 |     |
! NAME         | DESCRIPTION                                     | I/O |
!--------------|-------------------------------------------------|-----|
!              |                                                 |     |
! F_xt         | upwind longitudes for themodynamic level        |  o  |
! F_yt         | upwind latitudes for themodynamic level         |  o  |
! F_zt         | upwind height for themodynamic level            |  o  |
! F_xm         | upwind longitudes for momentum level            |  i  |
! F_ym         | upwind latitudes for momentum level             |  i  |
! F_zm         | upwind height for momentum level                |  i  |
!______________|_________________________________________________|_____|
!                      |
!                      |
!    No top nesting    |       With top nesting
!                      |
!                      | NOTES: F_k0 = Lam_gbpil_t+1
!                      | For this example Lam_gbpil_t=3 -> F_k0=4
!                      |
!  ======== Model Top  |   ==========
!  -------- %m 1       |   ---------- uptream pos. not available (F_k0-3)
!  Linear   %t 1       |   Not needed
!  -------- %m 2       |   ----------
!  Cubic    %t 2       |   Not needed
!  -------- %m 3       |   ----------
!  Cubic    %t 3       |   Cubic      %t F_k0-1
!  -------- %m 4       |   ---------- %m F_k0
!                      |
!     ...              |       ...
!                      |
!  Cubic    %t N-2     |   Cubic
!  -------- %m N-1     |   ----------
!  Linear   %t N-1     |   Linear
!  -------- %m N       |   ----------
!  Constant %t N       |   Constant
!  ======== Model Sfc  |   ===========
!                      |
!----------------------------------------------------------------------
!


!***********************************************************************
   integer vnik, vnikm, i,j,k
!
   real*8  r2pi_8, two, half, alpha
   real*8, dimension(2:l_nk-2) :: w1, w2, w3, w4
   real*8, dimension(i0:in,l_nk) :: wdt
   real*8 :: lag3, hh, x, x1, x2, x3, x4
   real :: ztop_bound, zbot_bound
   parameter (two = 2.0, half=0.5)
   !***********************************************************************
   !
   lag3( x, x1, x2, x3, x4 ) = &
        ( ( x  - x2 ) * ( x  - x3 ) * ( x  - x4 ) )/ &
        ( ( x1 - x2 ) * ( x1 - x3 ) * ( x1 - x4 ) )
   !
   !***********************************************************************
   ! Note : extra computation are done in the pilot zone if
   !        (Lam_gbpil_t != 0) for coding simplicity
   !***********************************************************************
   !

   ztop_bound=adv_verZ_8%m(0)
   zbot_bound=adv_verZ_8%m(l_nk+1)
   !

   vnik = (in-i0+1)*l_nk
   vnikm= (in-i0+1)*(l_nk-1)
   !
   r2pi_8 = two * CONST_PI_8
   !
   ! Prepare parameters for cubic intepolation
   do k=2,l_nk-2
      hh = adv_verZ_8%t(k)
      x1 = adv_verZ_8%m(k-1)
      x2 = adv_verZ_8%m(k  )
      x3 = adv_verZ_8%m(k+1)
      x4 = adv_verZ_8%m(k+2)
      w1(k) = lag3( hh, x1, x2, x3, x4 )
      w2(k) = lag3( hh, x2, x1, x3, x4 )
      w3(k) = lag3( hh, x3, x1, x2, x4 )
      w4(k) = lag3( hh, x4, x1, x2, x3 )
   enddo

   k00=max(F_k0-1,1)

!***********************************************************************
!     call tmg_start ( 33, 'adv_meanpos' )
!$omp parallel private(i,j,k,wdt)
!$omp do
   do j=j0,jn
!
!     Fill non computed upstream positions with zero to avoid math exceptions
!     in the case of top piloting
      do k=1,k00-1
         do i=i0,in
            F_xt(i,j,k)=0.0
            F_yt(i,j,k)=0.0
            F_zt(i,j,k)=0.0
         end do
      enddo

!***********************************************************************
! For last thermodynamic level, positions in the horizontal are those  *
! of the momentum levels; no displacement allowed in the vertical      *
! at bottum. At top vertical displacement is obtian from linear inter. *
! and is bound to first thermo level.                                  *
!***********************************************************************
      do i=i0,in
         F_xt(i,j,l_nk) = F_xm(i,j,l_nk)
         F_yt(i,j,l_nk) = F_ym(i,j,l_nk)
         F_zt(i,j,l_nk) = zbot_bound
      enddo

!
!***********************************************************************
      do k=k00,l_nk-1
         if(F_cubic_xy_L.and.k.ge.2.and.k.le.l_nk-2)then
            ! Cubic
            do i=i0,in
               F_xt(i,j,k) = w1(k)*F_xm(i,j,k-1)+ &
                             w2(k)*F_xm(i,j,k  )+ &
                             w3(k)*F_xm(i,j,k+1)+ &
                             w4(k)*F_xm(i,j,k+2)
               F_yt(i,j,k) = w1(k)*F_ym(i,j,k-1)+ &
                             w2(k)*F_ym(i,j,k  )+ &
                             w3(k)*F_ym(i,j,k+1)+ &
                             w4(k)*F_ym(i,j,k+2)
            enddo
         else
            ! Linear
            do i=i0,in
               F_xt(i,j,k) = (F_xm(i,j,k)+F_xm(i,j,k+1))*half
               F_yt(i,j,k) = (F_ym(i,j,k)+F_ym(i,j,k+1))*half
            enddo
         endif
      enddo


!        working with displacements
         do k=k00,l_nk-1
            do i=i0,in
                   if(k.ge.2.and.k.le.l_nk-2)then
                  !Cubic
                  wdt(i,k)= &
                       w1(k)*F_wdm(i,j,k-1)+ &
                       w2(k)*F_wdm(i,j,k  )+ &
                       w3(k)*F_wdm(i,j,k+1)+ &
                       w4(k)*F_wdm(i,j,k+2)
               else
                  !Linear
                  wdt(i,k) = (F_wdm(i,j,k)+F_wdm(i,j,k+1))*half

               endif
               F_zt(i,j,k)=adv_verZ_8%t(k)-(wdt(i,k)+F_wat(i,j,k))*cstv_dt_8*0.5d0
               F_zt(i,j,k)=max(F_zt(i,j,k),ztop_bound)
               F_zt(i,j,k)=min(F_zt(i,j,k),zbot_bound)

            end do
         end do


   enddo

!$omp enddo
!$omp end parallel
!
   end subroutine interp_vert_thermo_pos


  end module adv_trajectories_mod
