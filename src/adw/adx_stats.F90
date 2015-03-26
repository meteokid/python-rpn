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

subroutine adx_stats( &
     F_l_S, &
     F_su,F_sv,F_sw, &
     F_px  ,F_py  ,F_pz  , &
     F_xth ,F_yth ,F_zth , &
     F_xcth,F_ycth,F_zcth, &
     F_xct1,F_yct1,F_zct1, &
     minx,maxx,miny,maxy,F_nks, &
     F_ni, F_nj, F_nkm, F_nk, i0,in,j0,jn,k0,kn)
   implicit none
#include <arch_specific.hf>
   !@objective Print stats fo computed pos
   !@arguments
   integer :: F_nb_iter          !I, total number of iterations for traj
   character(len=1) :: F_l_S     !I, m/t for momentum or thermo level
   integer :: F_ni, F_nj         !I, dims of position fields
   integer :: F_nkm,F_nk         !I, nb levels
   integer :: i0,in,j0,jn,k0,kn
   integer :: minx,maxx,miny,maxy,F_nks
   real, dimension(F_ni,F_nj,F_nk) :: &
        F_px  , F_py  , F_pz     !I, upstream positions valid at t1
   real, dimension(F_ni,F_nj,F_nk) :: &
        F_xth , F_yth , F_zth,&  !I, upwind longitudes at central time 
        F_xcth, F_ycth, F_zcth,& !I, upwind cartesian positions at central time
        F_xct1, F_yct1, F_zct1   !I, upstream cartesian positions at t1
   real, dimension(minx:maxx,miny:maxy,F_nks) :: &
        F_su,F_sv,F_sw
   !*@/
   !---------------------------------------------------------------------
!!$   print *,'s',minx,maxx,miny,maxy,1,F_nks
!!$   print *,trim(F_l_S),1,F_ni,1,F_nj,1,F_nk
!!$   print *,trim(F_l_S),i0,in,j0,jn,k0,kn

   call glbstat2 (F_px, 'px', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_py, 'py', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_pz, 'pz', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_xth, 'xth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_yth, 'yth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_zth, 'zth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_xcth, 'xcth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_ycth, 'ycth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_zcth, 'zcth', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)

   kn = F_nkm
   call glbstat2 (F_xct1, 'xct1', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_yct1, 'yct1', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_zct1, 'zct1', 'p_'//trim(F_l_S),  &
        1,F_ni,1,F_nj,1,F_nk, &
        i0,in,j0,jn,k0,kn)

   kn = F_nks
   call glbstat2 (F_su, 'su', 'p_s'//trim(F_l_S),  &
        minx,maxx,miny,maxy,1,F_nks, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_sv, 'sv', 'p_s'//trim(F_l_S),  &
        minx,maxx,miny,maxy,1,F_nks, &
        i0,in,j0,jn,k0,kn)
   call glbstat2 (F_sw, 'sw', 'p_s'//trim(F_l_S),  &
        minx,maxx,miny,maxy,1,F_nks, &
        i0,in,j0,jn,k0,kn)
   !---------------------------------------------------------------------
   return
end subroutine adx_stats
