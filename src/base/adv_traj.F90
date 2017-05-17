!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                       Environnement Canada
! This library is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, version 2.1 of the License. This library is
! distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
! PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
! You should have received a copy of the GNU Lesser General Public License
! along with this library; if not, write to the Free Software Foundation, Inc.,
! 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!---------------------------------- LICENCE END --------------------------------- 
 
      subroutine adv_traj ( F_nb_iter, pxm , pym , pzm, &
                               F_u, F_v, F_w, F_ua, F_va, F_wa, F_wat,&
                               F_xth ,F_yth, F_zth, &
                               i0,in,j0 ,jn,i0u,inu,j0v,jnv,&
                               k0, k0m,k0t, F_aminx, F_amaxx,&
                               F_aminy, F_amaxy, F_ni,F_nj,F_nk )
      use step_options
      use adv_pos
      use adv_options
      use grid_options
      use gem_options
      implicit none
#include <arch_specific.hf>
      integer  F_nb_iter                                                                  ! total number of iterations for traj
      integer, intent(in) :: k0 , k0m,  k0t                                               ! scope of the operation F_k0 to F_nk
      integer, intent(in) :: i0, j0 , in, jn 		            	                        !0 , scope of advection operations
      integer, intent(in) :: i0u, inu, j0v, jnv	
      integer, intent(in) :: F_ni,F_nj,F_nk                                               ! dims of position fields
      integer, intent(in) :: F_aminx,F_amaxx,F_aminy, F_amaxy                             ! wind fields array bounds
      real, dimension(F_ni,F_nj,F_nk), intent(out) :: pxm  , pym  , pzm                   ! upstream positions valid at t1   
      real, dimension(F_ni,F_nj,F_nk) :: F_xth  , F_yth  , F_zth                          ! upwind longitudes at time t1
      real, dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk), intent(in), target :: F_u, F_v, F_w      ! destag winds
      real, dimension(F_ni,F_nj,F_nk), intent(in) :: F_ua,   F_va,   F_wa                      ! Arival winds
      real, dimension(F_ni,F_nj,F_nk) :: wdm
      real, dimension(F_ni,F_nj,F_nk),intent(in) :: F_wat

!@Objectives computes trajectories using trapezoidal rule
!@ Author RPN-A Model Infrastructure Group (base on adx_pos_angular_m, adx_pos_angular_t, adx_pos_muv) june 2015 

#include "adv_grid.cdk"
#include "adv_gmm.cdk"
#include "adv_interp.cdk"
#include "ver.cdk"
#include "tracers.cdk"
#include "cstv.cdk"
#include "glb_ld.cdk"

      integer :: i , j , k, i0u_e, inu_e, j0v_e, jnv_e
      integer ,  dimension(:), allocatable :: ii
      real, dimension(F_ni, F_nj, F_nk) :: ud , vd ,wd
      integer ::  iter,  num , nind
      integer :: BCS_BASE ,n,cnt,nc, sum_cnt,totaln,err  ! BCS points for Yin-Yang, normal LAM
      real :: ztop_bound, zbot_bound
      real*8 :: inv_cy_8
      real, dimension(1,1,1), target :: no_conserv, no_slice, no_flux
      real,   dimension(:,:,:), pointer :: dummy3d 
      real,   parameter :: DTH_1     = 1.  , EPS_8 = 1.D-5
      real :: xpos,ypos,minposx,maxposx,minposy,maxposy
 !     
!---------------------------------------------------------------------
!     
      num=F_ni*F_nj*F_nk
      nind=(in-i0+1)*(jn-j0+1)*(F_nk-k0m+1)

      ztop_bound=Ver_z_8%m(0)
      zbot_bound=Ver_z_8%m(F_nk+1)

!Horizontal  Clipping  to processor  boundaries
       BCS_BASE= 4
      if (Grd_yinyang_L) BCS_BASE = 3
      minposx = adv_xx_8(adv_lminx+1) + EPS_8
      if (l_west)  minposx = adv_xx_8(1+BCS_BASE) + EPS_8
      maxposx = adv_xx_8(adv_lmaxx-1) - EPS_8
      if (l_east)  maxposx = adv_xx_8(F_ni-BCS_BASE) - EPS_8
      minposy = adv_yy_8(adv_lminy+1) + EPS_8
      if (l_south) minposy = adv_yy_8(1+BCS_BASE) + EPS_8
      maxposy = adv_yy_8(adv_lmaxy-1) - EPS_8
      if (l_north) maxposy = adv_yy_8(F_nj-BCS_BASE) - EPS_8
     
      !  This is required to fill the array for the copy marked with 'filled copy here' below.  If
      !  this initialization is not done, some array elements are not defined before the copy is made.
      wd = 0.


      if(Schm_cub_traj_L) allocate( ii(nind*4) )    

! Calculate upstream positions at t1 using angular displacement

      do  iter = 1, F_nb_iter

    ! 1. INTERPOLATION OF WINDS

! Interpolate with tricubic method

     if(Schm_cub_traj_L) then
        
            call adv_get_indices(ii, F_xth, F_yth, F_zth , num, nind, i0, in, j0, jn, k0m, F_nk, 'm')           
            	
			   call adv_tricub_lag3d(ud, no_conserv, no_conserv, no_conserv, no_conserv, F_u, no_slice, 0, & 
                                  no_flux, no_flux, no_flux, no_flux, 0, &
                                  F_xth, F_yth, F_zth, &
                                  no_slice, no_slice, no_slice, & 
                                  no_slice, no_slice, no_slice, &
                                  num, nind, ii, k0m ,F_nk , &
                                  .false. , .false. , 'm')   
			   call adv_tricub_lag3d(vd, no_conserv, no_conserv, no_conserv, no_conserv, F_v, no_slice, 0, & 
                                  no_flux, no_flux, no_flux, no_flux, 0, &
                                  F_xth, F_yth, F_zth, &
                                  no_slice, no_slice, no_slice, & 
                                  no_slice, no_slice, no_slice, &
                                  num, nind, ii, k0m ,F_nk , &
                                  .false. , .false. , 'm')         
     else
     call adv_trilin(ud,vd,F_u,F_v,F_xth,F_yth,F_zth, &
                          DTH_1, .true., i0, in, j0, jn, &
                          F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0m,F_nk)
     endif

!-  CALCULATION OF DEPARTURE POSITIONS  WITH THE TRAPEZOIDAL/MIDPOINT RULES


 cnt=0 
           if(Schm_trapeze_L) then

!$omp parallel private (inv_cy_8,i,j,k,nc)
 nc = 0
!$omp do
         do k=k0m,F_nk
            do j=j0,jn
               inv_cy_8 = 1.d0 / adv_cy_8(j)
               do i=i0,in
                  xpos = adv_xx_8(i) - Cstv_dtD_8*  ud(i,j,k)/cos(F_yth(i,j,k)) &
                                             - Cstv_dtA_8*F_ua(i,j,k)*inv_cy_8
                  ypos = adv_yy_8(j) - Cstv_dtD_8*  vd(i,j,k) &
                                             - Cstv_dtA_8*F_va(i,j,k)
		  F_xth(i,j,k) = min(max(xpos,minposx),maxposx)
                  F_yth(i,j,k) = min(max(ypos,minposy),maxposy)
                ! Traj trunc  stat
                  nc=nc+min(1,max(0,ceiling(abs(F_xth(i,j,k)-xpos)+abs(F_yth(i,j,k)-ypos))))
               end do
            end do
         enddo
!$omp enddo
!$omp atomic
 cnt=cnt+nc
!$omp end parallel 
          else
!$omp parallel private(k,i,j)
!$omp do
         do k=k0m,F_nk
            do j=j0,jn
               do i=i0,in
                  F_xth(i,j,k) = adv_xx_8(i) - 0.5d0*Cstv_dt_8* ud(i,j,k)/cos(F_yth(i,j,k)) 
                  F_yth(i,j,k) = adv_yy_8(j) - 0.5d0*Cstv_dt_8* vd(i,j,k)
                  F_xth(i,j,k) = min(max(F_xth(i,j,k),minposx),maxposx)
                  F_yth(i,j,k) = min(max(F_yth(i,j,k),minposy),maxposy)
               end do
            end do
         enddo
!$omp enddo
!$omp end parallel
           endif
!- 3D interpol of zeta dot and new upstream pos along zeta
      
     if(Schm_cub_traj_L) then         
               call adv_get_indices(ii, F_xth, F_yth, F_zth, num, nind, i0, in, j0, jn, k0m, F_nk, 'm') 
               call adv_tricub_lag3d(wd, no_conserv, no_conserv, no_conserv, no_conserv, F_w, no_slice, 0, &
                                  no_flux, no_flux, no_flux, no_flux, 0, &
                                  F_xth,F_yth,F_zth, &
                                  no_slice, no_slice, no_slice, &
                                  no_slice, no_slice, no_slice, &
                                  num, nind, ii, k0m ,F_nk , &
                                  .false. , .false. , 'm')
     else
            dummy3d => F_w
            call adv_trilin(wd,vd,F_w,dummy3d, F_xth, F_yth,F_zth, &
                           DTH_1, .false., i0, in, j0, jn, &
                           F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,k0m,F_nk)
     endif

!$omp parallel private(k,i,j) shared (wd)
      if(Schm_trapeze_L) then

!$omp do
         do k = max(1,k0m),F_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) = Ver_z_8%m(k) - Cstv_dtzD_8*  wd(i,j,k) &
                                              - Cstv_dtzA_8*F_wa(i,j,k)
                  F_zth(i,j,k) = min(zbot_bound,max(F_zth(i,j,k),ztop_bound))
               enddo
            enddo
         enddo

!$omp enddo
      else
!$omp do
         do k = max(1,k0m),F_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) =Ver_z_8%m(k)  - Cstv_dt_8* wd(i,j,k)
                  F_zth(i,j,k) = min(zbot_bound,max(F_zth(i,j,k),ztop_bound))
                  F_zth(i,j,k) = 0.5D0*(F_zth(i,j,k) + Ver_z_8%m(k))
               enddo
            enddo
         enddo
!$omp enddo
      endif
!$omp end parallel

! Print trajectory clipping stat
  call adv_print_cliptrj_s (cnt,F_ni,F_nj,F_nk,k0,'m')

 enddo

 ! Departure point
   if (Schm_trapeze_L) then
      ! nothing to do ...
      pxm = F_xth
      pym = F_yth
      pzm = F_zth
      wdm = wd   !filled copy here
   else

   cnt=0
!$omp parallel private (k,j,i,nc)
   nc=0
!$omp do
      do k=k0m,F_nk
         do j=j0,jn
            do i=i0,in
               xpos = adv_xx_8(i) - Cstv_dt_8* ud(i,j,k)/cos(F_yth(i,j,k)) 
               ypos = adv_yy_8(j) - Cstv_dt_8* vd(i,j,k) 
               pzm(i,j,k) = F_zth(i,j,k) - Ver_z_8%m(k)
               pzm(i,j,k) = Ver_z_8%m(k) + 2.0 * pzm(i,j,k)
	       pxm(i,j,k) = min(max(xpos,minposx),maxposx)
               pym(i,j,k) = min(max(ypos,minposy),maxposy)
               pzm(i,j,k) = min(zbot_bound,max(pzm(i,j,k),ztop_bound))
             
              ! Horizontal positions trunc stat
               nc=nc+min(1,max(0,ceiling(abs(pxm(i,j,k)-xpos)+abs(pym(i,j,k)-ypos))))
            end do
         end do
      enddo
!$omp enddo
!$omp atomic
 cnt=cnt+nc
!$omp end parallel

! Print  Trajectory clipping stat 
  call adv_print_cliptrj_s (cnt,F_ni,F_nj,F_nk,k0,'m')

 endif

if(Schm_cub_traj_L) deallocate ( ii )

!
! Interpolate Momentun positions on Thermo Level and U V  grid

      call adv_int_horiz_m (pxmu, pymu, pzmu, pxmv, pymv, pzmv, pxm, pym, pzm, &
                            F_ni,F_nj,F_nk,k0, i0, in, j0, jn)
      if (Tr_slice_L) then  
      call adv_int_horiz_m_slice (pxmu_s, pymu_s, pzmu_s, pxmv_s, pymv_s, pzmv_s, pxm, pym, pzm, &
                            F_ni,F_nj,F_nk,k0, i0, in, j0, jn, i0u_e, inu_e, j0v_e, jnv_e)
      endif
      call adv_int_vert_t (pxt,pyt,pzt,pxtn,pytn,pztn,pxm,pym,pzm,F_wat,wdm, &
                           F_ni,F_nj,F_nk,k0t,i0,in,j0,jn, .true.)
     
      if (Tr_slice_L) then
       call adv_cliptraj (pxmu_s,pymu_s,F_ni,F_nj,F_nk,i0u_e,inu_e,j0,jn,k0,'INTERP '//trim('m'))
       call adv_cliptraj (pxmv_s,pymv_s,F_ni,F_nj,F_nk,i0,in,j0v_e,jnv_e,k0,'INTERP '//trim('m'))
      endif
    
!     
!---------------------------------------------------------------------
!     
      return
      end subroutine adv_traj

    


 
