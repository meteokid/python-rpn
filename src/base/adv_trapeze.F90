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
 
  subroutine adv_trapeze (F_nb_iter, pxm , pym , pzm , &
                           F_u, F_v, F_w, F_ua, F_va, F_wa, F_wat, &
                           F_xth ,F_yth, F_zth, &
                           i0,in,j0 ,jn,i0u,inu,j0v,jnv,k0,k0m,k0t , &
                           F_minx, F_maxx, F_miny, F_maxy , F_ni , F_nj , F_nk )

implicit none
#include <arch_specific.hf>
 
      integer  F_nb_iter                                                                        ! total number of iterations for traj
      integer, intent(in) :: k0 , k0m,  k0t                                                       ! scope of the operation F_k0 to F_nk
      integer, intent(in) :: i0, j0 , in, jn 		            	                        !0 , scope of advection operations
      integer, intent(in) :: i0u, inu, j0v, jnv	
      integer, intent(in) :: F_ni,F_nj,F_nk                                                    ! dims of position fields
      integer, intent(in) :: F_minx,F_maxx,F_miny, F_maxy                                      ! wind fields array bounds
      real, dimension(F_ni,F_nj,F_nk), intent(out) :: pxm  , pym  , pzm                        ! upstream positions valid at t1   
      real, dimension(F_ni,F_nj,F_nk) :: F_xth  , F_yth  , F_zth                ! upwind longitudes at time t1
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk), intent(in) :: F_u   , F_v  , F_w      ! destag winds
      real, dimension(F_ni,F_nj,F_nk), intent(in) :: F_ua,   F_va,   F_wa                      ! Arival winds
      real, dimension(F_ni,F_nj,F_nk) :: wdm
      real, dimension(F_ni,F_nj,F_nk),intent(in) :: F_wat

#include "msg.h"   
#include "adv_grid.cdk"
#include "adv_gmm.cdk"
#include "adv_nml.cdk"
#include "adv_pos.cdk"
#include "cstv.cdk"

     integer :: i , j , k
     real, dimension(F_ni, F_nj, F_nk) :: ud , vd ,wd
     integer :: dt , err, iter, jext , num
     real :: ztop_bound, zbot_bound
     real*8 :: inv_cy_8
     real, dimension(1,1,1), target :: no_conserv
   
 !------------------------------------------------------------------------
           
     num=F_ni*F_nj*F_nk  
     
     ztop_bound=adv_verZ_8%m(0)
     zbot_bound=adv_verZ_8%m(F_nk+1)

! Calculate upstream positions at t1 using angular displacement

     
 dt  = 0.5 * cstv_dt_8
      
      do iter = 1, F_nb_iter

    ! 1. INTERPOLATION OF WINDS

! Clipping trajectories
 
      call adv_cliptraj (F_xth, F_yth, F_ni, F_nj,F_nk,i0, in, j0, jn, k0m,'')

! Interpolate with tricubic method
      if (adw_catmullrom_L) then
          call adv_tricub_catmullrom(ud, F_u, F_xth, F_yth, F_zth, num, .false. , i0, in, j0, jn, k0m, F_nk, 'm')
          call adv_tricub_catmullrom(vd, F_v, F_xth, F_yth, F_zth, num, .false. , i0, in, j0, jn, k0m, F_nk, 'm')
      else 
          call adv_tricub_lag3d(ud, no_conserv, no_conserv, no_conserv, no_conserv, F_u, F_xth, F_yth, F_zth, &
                               num, .false. , .false. ,i0 ,in ,j0 ,jn ,k0m ,F_nk , 'm')   

          call adv_tricub_lag3d(vd, no_conserv, no_conserv, no_conserv, no_conserv, F_v, F_xth, F_yth, F_zth, &
                               num, .false. , .false. ,i0 ,in ,j0 ,jn ,k0m,F_nk , 'm')
     end if


!-  CALCULATION OF DEPARTURE POSITIONS  WITH THE TRAPEZOIDALE RULE 

!omp parallel private (inv_cy_8)
!omp do
         do k=k0m,F_nk
            do j=j0,jn
               inv_cy_8 = 1.d0 / adv_cy_8(j)
               do i=i0,in
                  F_xth(i,j,k) = adv_xx_8(i) - (ud(i,j,k) / cos(F_yth(i,j,k)) + F_ua(i,j,k) * inv_cy_8) * dt
                  F_yth(i,j,k) = adv_yy_8(j) - (vd(i,j,k) + F_va(i,j,k)) * dt
               end do
            end do
         enddo
!omp enddo
!omp end parallel

 
!- 3D interpol of zeta dot and new upstream pos along zeta
      
         call adv_cliptraj (F_xth,F_yth, F_ni, F_nj,F_nk,i0,in,j0,jn,max(k0t-2,1),'')
 
          if (adw_catmullrom_L) then
          call adv_tricub_catmullrom(wd, F_w, F_xth,F_yth,F_zth,num,.false., i0,in,j0,jn,k0m,F_nk,'m')
         else 
          call adv_tricub_lag3d(wd, no_conserv, no_conserv, no_conserv, no_conserv, F_w, F_xth,F_yth,F_zth, &
                               num, .false.,.false.,i0,in,j0,jn,k0m,F_nk, 'm')
         end if

!$omp parallel
!$omp do
         do k = max(1,k0m),F_nk
            do j = j0,jn
               do i = i0,in
                  F_zth(i,j,k) = adv_verZ_8%m(k) - (wd(i,j,k)+F_wa(i,j,k))*dt
                  F_zth(i,j,k) = min(zbot_bound,max(F_zth(i,j,k),ztop_bound))
               enddo
            enddo
         enddo

!$omp enddo
!$omp end parallel
        end do

      pxm = F_xth
      pym = F_yth
      pzm = F_zth
      wdm = wd


      call adv_int_horiz_m (pxmu, pymu, pzmu, pxmv, pymv, pzmv, pxm, pym, pzm, &
                               F_ni,F_nj,F_nk,k0, i0, in, j0, jn)

      call adv_int_vert_t (pxt,pyt,pzt,pxm,pym,pzm,F_wat,wdm, &
                             F_ni,F_nj,F_nk,k0t,i0,in,j0,jn, .true.)

! Clipping trajectories 
    
      call adv_cliptraj (pxm,pym,F_ni,F_nj,F_nk,i0,in,j0,jn,k0,'INTERP '//trim('m'))
      call adv_cliptraj (pxmu,pymu,F_ni,F_nj,F_nk,i0u,inu,j0,jn,k0,'INTERP '//trim('m'))
      call adv_cliptraj (pxmv,pymv,F_ni,F_nj,F_nk,i0,in,j0v,jnv,k0,'INTERP '//trim('m'))
      call adv_cliptraj (pxt,pyt,F_ni,F_nj,F_nk,i0,in,j0,jn,k0,'INTERP '//trim('t'))
    

 
 end subroutine adv_trapeze



   
     
   




 