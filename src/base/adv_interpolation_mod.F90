module adv_interpolation_mod
   ! Find departure points
   !
   ! Author
   !    
   !
   ! Revision
   !
   implicit none

   private

#include "msg.h"
#include "gmm.hf"
#include "adv_dims.cdk"
#include "adv_nml.cdk"
#include "adv_grid.cdk"
#include "orh.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "vt_tracers.cdk"

   public :: interp_cubic_winds, adv_interpol_cubic , adv_cliptraj3

contains

  subroutine interp_cubic_winds(F_wrkx1,F_wrky1,F_u1,F_u2,F_xth,F_yth,F_zth, &
                              F_has_u2_L, F_i0,F_in,F_j0,F_jn,F_k0)
   implicit none
#include <arch_specific.hf>
   !@objective
   !@arguments

   logical, intent(in) :: F_has_u2_L                                                    ! .T. if F_u2 needs to be treated
   integer, intent(in) :: F_i0,F_in,F_j0,F_jn,F_k0                                      ! operator scope
   real, dimension(l_ni,l_nj,l_nk), intent(inout) :: F_xth,F_yth                        ! x,y positions
   real, dimension(l_ni,l_nj,l_nk), intent(in) :: F_zth                                ! z positions
   real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk), intent(in) :: F_u1,F_u2 ! field to interpol
   real, dimension(l_ni,l_nj,l_nk), intent(out) :: F_wrkx1,F_wrky1                      ! F_dt * result of interp
 
   !
   !@revisions
   !  2014-09,  Monique Tanguay    : GEM4 Mass-Conservation
   !**/
   !---------------------------------------------------------------------

   integer :: num
   real, dimension(1,1,1), target :: no_conserv

   call msg(MSG_DEBUG,'adv_int_winds_lam')


    call adv_cliptraj3 (F_xth,F_yth,F_i0,F_in,F_j0,F_jn,F_k0,'')

    num=l_ni*l_nj*l_nk
   
      if (adw_catmullrom_L) then

         call adx_tricub_catmullrom(F_wrkx1, F_u1, F_xth,F_yth,F_zth,num, &
                                    .false., F_i0,F_in,F_j0,F_jn,F_k0,l_nk, 'm')

         if(F_has_u2_L) then
            call adx_tricub_catmullrom(F_wrky1, F_u2, F_xth,F_yth,F_zth,num, &
                                       .false., F_i0,F_in,F_j0,F_jn,F_k0,l_nk, 'm')
         end if
      else

 
         call adv_tricub_lag3d(F_wrkx1, no_conserv, no_conserv, no_conserv, no_conserv, F_u1, F_xth,F_yth,F_zth,num, &
                                .false.,.false.,F_i0,F_in,F_j0,F_jn,F_k0,l_nk, 'm')
   

   if(F_has_u2_L) then
            call adv_tricub_lag3d(F_wrky1, no_conserv, no_conserv, no_conserv, no_conserv, F_u2, F_xth,F_yth,F_zth,num, &
                                   .false.,.false.,F_i0,F_in,F_j0,F_jn,F_k0,l_nk, 'm')
         end if
      end if


    call msg(MSG_DEBUG,'adv_int_winds_lam [end]')

   return

  end subroutine interp_cubic_winds



  subroutine adv_interpol_cubic (F_name,fld_out ,fld_in, F_capx, F_capy, F_capz, &
                                 i0, in, j0, jn, k0, lev_S, mono_kind,mass_kind)
   implicit none


#include <arch_specific.hf>
!
   !@objective Interpolation of rhs
!
   !@arguments
   character(len=*), intent(in) :: F_name
   logical :: F_is_mom_L     !I, momentum level if .true. (thermo if not)
   logical :: F_doAdwStat_L  !I, compute stats if .true.
   integer :: k0           !I, vertical scope k0 to l_nk
   real, intent(in)::  F_capx(*), F_capy(*), F_capz(*) !I, upstream positions at t1

!
   !@author alain patoine
   !@revisions
   ! v2_31 - Desgagne & Tanguay  - removed stkmemw, introduce tracers
   ! v2_31                       - tracers not monotone in anal mode
   ! v3_00 - Desgagne & Lee      - Lam configuration
   ! v3_02 - Tanguay             - Restore tracers monotone in anal mode
   ! v3_02 - Lee V.              - revert adv_exch_1 for GLB only,
   ! v3_02                         added adv_ckbd_lam,adv_cfl_lam for LAM only
   ! v3_03 - Tanguay M.          - stop if adv_exch_1 is activated when anal mode
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v3_11 - Gravel S.           - introduce key adv_mono_L
   ! v3_20 - Gravel & Valin & Tanguay - Lagrange 3D
   ! v3_20 - Tanguay M.          - Improve alarm when points outside advection grid
   ! v3_20 - Dugas B.            - correct calculation for LAM when Glb_pil gt 7
   ! v3_21 - Desgagne M.         - if  Lagrange 3D, call adv_main_3_intlag
   ! v4_04 - Tanguay M.          - Staggered version TL/AD
   ! v4_05 - Lepine M.           - VMM replacement with GMM
   ! v1_10 - Plante A.           - Thermo upstream positions
   ! v4_40 - Tanguay M.          - Revision TL/AD
   ! v4_XX - Tanguay M.          - GEM4 Mass-Conservation
!**/

   logical :: mono_L, conserv_L
   character(len=1) :: lev_S
   integer :: n,i0,j0,in,jn
   integer, intent(in) :: mono_kind      !I, Kind of Shape preservation
   integer, intent(in) :: mass_kind      !I, Kind of  Mass conservation

   logical, parameter :: EXTEND_L = .true.
   integer ::  i, j, k, nbpts
  ! real    :: fld_adw(l_minx:l_maxx,l_miny:l_maxy,l_nk)
   real, dimension(l_ni,l_nj,l_nk) :: wrkc, w_mono_c, w_lin_c, w_min_c, w_max_c
   real, dimension(l_minx:l_maxx, l_miny:l_maxy ,l_nk), intent(in)  :: Fld_in
   real, dimension(l_minx:l_maxx, l_miny:l_maxy ,l_nk), intent(out) :: Fld_out
   
   type(gmm_metadata) :: mymeta
   real, dimension(1,1,1), target :: no_conserv
   integer :: err

     call msg(MSG_DEBUG,'adv_interp')

       mono_L = .false.

      if (F_name(1:3) == 'TR/') then
         mono_L = adw_mono_L
      endif

       conserv_L = mono_kind/=0.or.mass_kind/=0

!          call rpn_comm_xch_halox( fld_in, l_minx, l_maxx,l_miny, l_maxy , &
!				 l_ni, l_nj, l_nk, adv_halox, adv_haloy   , &
!				 G_periodx, G_periody                     , &
!				 fld_adw, l_minx,l_maxx,l_miny,l_maxy, l_ni, 0)
 
 
     nbpts = l_ni*l_nj*l_nk

    if (conserv_L) then

      nullify(fld_cub,fld_mono,fld_lin,fld_min,fld_max)

      err = gmm_get(gmmk_cub_s ,fld_cub ,mymeta)
      err = gmm_get(gmmk_mono_s,fld_mono,mymeta)
      err = gmm_get(gmmk_lin_s ,fld_lin ,mymeta)
      err = gmm_get(gmmk_min_s ,fld_min ,mymeta)
      err = gmm_get(gmmk_max_s ,fld_max ,mymeta)

   else

      fld_cub  => no_conserv
      fld_mono => no_conserv
      fld_lin  => no_conserv
      fld_min  => no_conserv
      fld_max  => no_conserv

   endif
     if (adw_catmullrom_L) then
      call adx_tricub_catmullrom (wrkc, fld_in, F_capx, F_capy, F_capz, nbpts, &
                                  mono_L, i0,in,j0,jn,k0, l_nk, lev_S)
   else

      call adv_tricub_lag3d (wrkc, w_mono_c, w_lin_c, w_min_c, w_max_c, fld_in, F_capx, F_capy, F_capz, nbpts, &
                              mono_L, conserv_L, i0,in,j0,jn,k0, l_nk, lev_S)
   end if
 
!$omp parallel
 if (.NOT. conserv_L) then
!$omp do
   do k = k0, l_nk
      Fld_out(i0:in,j0:jn,k) = wrkc(i0:in,j0:jn,k)
   enddo
!$omp enddo
 else
!$omp do
   do k = k0, l_nk
      Fld_cub (i0:in,j0:jn,k) = wrkc    (i0:in,j0:jn,k)
      Fld_mono(i0:in,j0:jn,k) = w_mono_c(i0:in,j0:jn,k)
      Fld_lin (i0:in,j0:jn,k) = w_lin_c (i0:in,j0:jn,k)
      Fld_min (i0:in,j0:jn,k) = w_min_c (i0:in,j0:jn,k)
      Fld_max (i0:in,j0:jn,k) = w_max_c (i0:in,j0:jn,k)
   enddo
!$omp enddo
 endif
!$omp end parallel

if (conserv_L) call adv_tracers_mono_mass (F_name, fld_out, fld_cub,fld_mono, fld_lin, fld_min, fld_max , fld_in , &
                                            l_minx, l_maxx , l_miny, l_maxy ,l_nk   ,&
                                            i0, in ,j0 ,jn ,k0 , mono_kind, mass_kind )


    call msg(MSG_DEBUG,'adv_interp [end]')


   end subroutine adv_interpol_cubic



 subroutine adv_cliptraj3 (F_x,  F_y, i0, in, j0, jn,k0,mesg)
   implicit none

!#include <arch_specific.hf>
#include "stop_mpi.h"


   !@objective Clip SL hor. trajectories to either fit inside the
   !                    physical domain of the processor or to the
   !                    actual maximum allowed COURANT number (LAM)
   !@arguments
   character(len=*) :: mesg
    integer,intent(in) :: i0,in,j0,jn,k0  !I, scope of the operator
   real, dimension(l_ni,l_nj,l_nk)  ::  F_x, F_y              !I/O, upstream pos

!@author Michel Desgagne, Spring 2008
   !@revisions
   ! v3_31 - Desgagne M.  - Initial version
   ! v4_40 - Qaddouri/Lee - Yin-Yang trajectory clipping

   real*8,  parameter :: EPS_8 = 1.D-5
!  integer, parameter :: BCS_BASE = 4
   integer :: BCS_BASE ! BCS points for Yin-Yang, normal LAM

   character(len=MSG_MAXLEN) :: msg_S
   integer :: n, i,j,k, cnt, sum_cnt, err, totaln
   real :: minposx,maxposx,minposy,maxposy, posxmin,posxmax,posymin,posymax
   !---------------------------------------------------------------------

   BCS_BASE= 4
   if (Grd_yinyang_L) BCS_BASE = 3
   minposx = adv_xx_8(l_minx+1) + EPS_8
   if (l_west)  minposx = adv_xx_8(1+BCS_BASE) + EPS_8
   maxposx = adv_xx_8(l_maxx-1) - EPS_8
   if (l_east)  maxposx = adv_xx_8(l_ni-BCS_BASE) - EPS_8
   minposy = adv_yy_8(l_miny+1) + EPS_8
   if (l_south) minposy = adv_yy_8(1+BCS_BASE) + EPS_8
   maxposy = adv_yy_8(l_maxy-1) - EPS_8
   if (l_north) maxposy = adv_yy_8(l_nj-BCS_BASE) - EPS_8

   cnt=0

   !- Clipping to processor boundary
   do k=k0,l_nk
      do j=j0,jn
         do i=i0,in
            if ( (F_x(i,j,k)<minposx).or.(F_x(i,j,k)>maxposx).or. &
                 (F_y(i,j,k)<minposy).or.(F_y(i,j,k)>maxposy) ) then
               cnt=cnt+1
            endif
            F_x(i,j,k) = min(max(F_x(i,j,k),minposx),maxposx)
            F_y(i,j,k) = min(max(F_y(i,j,k),minposy),maxposy)
         enddo
      enddo
   enddo

   n = max(1,Grd_maxcfl)
   totaln = (l_ni*n*2 + (l_nj-2*n)*n*2) * (l_nk-k0+1)

   call rpn_comm_Allreduce(cnt,sum_cnt,1,"MPI_INTEGER", "MPI_SUM","grid",err)

  
   if (trim(mesg).ne."" .and. sum_cnt>0) then
      write(msg_S,'(a,i5,a,f6.2,2x,a)')  &
           ' ADW trajtrunc: npts=',sum_cnt, &
           ', %=',real(sum_cnt)/real(totaln)*100., &
           mesg
      call msg(MSG_INFO,msg_S)
   endif
      !---------------------------------------------------------------------
   return
end subroutine adv_cliptraj3

!========================================================================
!== stubs ===============================================================
!========================================================================

subroutine adv_cliptraj()
   call stop_mpi(STOP_ERROR,'adv_cliptraj','called a stub')
   return
end subroutine
subroutine adv_cliptraj2()
   call stop_mpi(STOP_ERROR,'adv_cliptraj2','called a stub')
   return
end subroutine


  end module adv_interpolation_mod
