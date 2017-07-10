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

!**s/r dcmip_Schaer_mountain - Setup for Mountain waves over a Schaer-type mountain on a small planet (DCMIP 2012)

      subroutine dcmip_Schaer_mountain (F_u,F_v,F_zd,F_t,F_q,F_topo,F_s, &
                                        Mminx,Mmaxx,Mminy,Mmaxy,Nk,Shear,Set_topo_L)

      use dcmip_2012_init_1_2_3
      use canonical
      use gem_options
      use geomh

      implicit none

      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u    (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_v    (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd   (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_t    (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_q    (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_s    (Mminx:Mmaxx,Mminy:Mmaxy),    &
           F_topo (Mminx:Mmaxx,Mminy:Mmaxy)

      integer  :: Shear      ! 0 or 1 see below
      logical  :: Set_topo_L ! If TRUE : Set F_topo to initialize  Topo_High and calculate Reference  variables
                             ! If FALSE: Use F_topo initialized as Topo_low  and calculate Associated variables

      !object
      !======================================================================================
      !   Setup for Mountain waves over a Schaer-type mountain on a small planet (DCMIP 2012)
      !======================================================================================

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "ver.cdk"
#include "ptopo.cdk"
#include "cstv.cdk"

      !-------------------------------------------------------------------------------

      integer i,j,k,istat

      real(8) x_a_8,y_a_8,utt_8,vtt_8,s_8(2,2),rlon_8

      real(8)  :: lon, &          ! Longitude (radians)
                  lat, &          ! Latitude (radians)
                  z               ! Height (m)

      real(8)  :: p               ! Pressure  (Pa)

      integer  :: zcoords         ! 0 or 1 see below

      real(8)  :: u, &            ! Zonal wind (m s^-1)
                  v, &            ! Meridional wind (m s^-1)
                  w, &            ! Vertical Velocity (m s^-1)
                  t, &            ! Temperature (K)
                  tv,&            ! Virtual Temperature (K)
                  phis, &         ! Surface Geopotential (m^2 s^-2)
                  ps, &           ! Surface Pressure (Pa)
                  rho, &          ! density (kg m^-3)
                  q               ! Specific Humidity (kg/kg)

      real(8) f_rayleigh_friction,fact_8

      real(8), parameter :: TAU0_8  = 25.  ! Damping time scale (sec)

      !-------------------------------------------------------------------------------

      if (Lun_out.gt.0.and.Cstv_tstr_8.ne.300.0) call handle_error(-1,'DCMIP_SCHAER_MOUNTAIN','SET TSTR AS T AT EQUATOR')

      if (Lun_out.gt.0) write (Lun_out,1000) Shear, Set_topo_L

      zcoords = 0

      !Factors in Rayleigh damped layer (using sin**2)
      !-----------------------------------------------
      if (Set_topo_L) then

         fact_8 = min(1.d0,Cstv_dt_8 / TAU0_8) !May need to be related to Earth's radius

         istat = gmm_get(gmmk_fcu_s,fcu)
         istat = gmm_get(gmmk_fcv_s,fcv)
         istat = gmm_get(gmmk_fcw_s,fcw)

      endif

      !Initial conditions: T,ZD,Q,S,TOPO
      !---------------------------------
      do k = 1,Nk

         do j = 1,l_nj

            lat   = geomh_y_8(j)
            y_a_8 = geomh_y_8(j)

            if (Ptopo_couleur.eq.0) then

               do i = 1,l_ni

                  lon = geomh_x_8(i)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%t(k),Ver_b_8%t(k),Cstv_pref_8, &
                                              Shear,u,v,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  F_t(i,j,k)  = tv
                  F_q(i,j,k)  = q
                  F_s(i,j)    = log(ps/Cstv_pref_8)

                  if (Set_topo_L) F_topo(i,j) = phis

                  F_zd(i,j,k) = w ! It is zero

                  if (Set_topo_L) fcw(i,j,k) = f_rayleigh_friction * fact_8

               end do

            else

               do i = 1,l_ni

                  x_a_8 = geomh_x_8(i) - acos(-1.D0)

                  call smat(s_8,rlon_8,lat,x_a_8,y_a_8)

                  lon = rlon_8 + acos(-1.D0)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%t(k),Ver_b_8%t(k),Cstv_pref_8, &
                                              Shear,utt_8,vtt_8,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  F_t(i,j,k)  = tv
                  F_q(i,j,k)  = q
                  F_s(i,j)    = log(ps/Cstv_pref_8)

                  if (Set_topo_L) F_topo(i,j) = phis

                  F_zd(i,j,k) = w ! It is zero

                  if (Set_topo_L) fcw(i,j,k) = f_rayleigh_friction * fact_8

               end do

            end if

         end do

      end do

      !Initial conditions: U True
      !--------------------------
      do k = 1,Nk

         do j = 1,l_nj

            lat   = geomh_y_8(j)
            y_a_8 = geomh_y_8(j)

            if (Ptopo_couleur.eq.0) then

               do i = 1,l_niu

                  lon = geomh_xu_8(i)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%m(k),Ver_b_8%m(k),Cstv_pref_8, &
                                              Shear,u,v,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  F_u(i,j,k) = u

                  if (Set_topo_L) fcu(i,j,k) = f_rayleigh_friction * fact_8

               end do

            else

               do i = 1,l_niu

                  x_a_8 = geomh_xu_8(i) - acos(-1.D0)

                  call smat(s_8,rlon_8,lat,x_a_8,y_a_8)

                  lon = rlon_8 + acos(-1.D0)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%m(k),Ver_b_8%m(k),Cstv_pref_8, &
                                              Shear,utt_8,vtt_8,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  u = s_8(1,1)*utt_8 + s_8(1,2)*vtt_8

                  F_u(i,j,k) = u

                  if (Set_topo_L) fcu(i,j,k) = f_rayleigh_friction * fact_8

               end do

            end if

         end do

      end do

      !Initial conditions: V True
      !--------------------------
      do k = 1,Nk

         do j = 1,l_njv

            lat   = geomh_yv_8(j)
            y_a_8 = geomh_yv_8(j)

            if (Ptopo_couleur.eq.0) then

               do i = 1,l_ni

                  lon = geomh_x_8(i)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%m(k),Ver_b_8%m(k),Cstv_pref_8, &
                                              Shear,u,v,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  F_v(i,j,k) = v

                  if (Set_topo_L) fcv(i,j,k) = f_rayleigh_friction * fact_8

               end do

            else

               do i = 1,l_ni

                  x_a_8 = geomh_x_8(i) - acos(-1.D0)

                  call smat(s_8,rlon_8,lat,x_a_8,y_a_8)

                  lon = rlon_8 + acos(-1.D0)

                  if (.NOT.Set_topo_L) phis = F_topo(i,j)

                  call test2_schaer_mountain (lon,lat,p,z,zcoords,Ver_a_8%m(k),Ver_b_8%m(k),Cstv_pref_8, &
                                              Shear,utt_8,vtt_8,w,t,tv,phis,ps,rho,q,f_rayleigh_friction,Set_topo_L)

                  v = s_8(2,1)*utt_8 + s_8(2,2)*vtt_8

                  F_v(i,j,k) = v

                  if (Set_topo_L) fcv(i,j,k) = f_rayleigh_friction * fact_8

               end do

            end if

         end do

      end do

      if (Set_topo_L) then

         !Initialize u,v,w REFERENCE for Rayleigh damped layer
         !----------------------------------------------------
         istat = gmm_get(gmmk_uref_s,uref)
         istat = gmm_get(gmmk_vref_s,vref)
         istat = gmm_get(gmmk_wref_s,wref)

         do k=1,Nk
            do j= 1,l_nj
               do i= 1,l_niu
                  uref(i,j,k) = F_u(i,j,k)
               end do
            end do
            do j= 1,l_njv
               do i= 1,l_ni
                  vref(i,j,k) = F_v(i,j,k)
               end do
            end do
            do j= 1,l_nj
               do i= 1,l_ni
                  wref(i,j,k) = 0.0 !ZERO Dz/Dt
               end do
            end do
         end do

      endif

      return

      !---------------------------------------------------------------

 1000 format( &
      /,'USE INITIAL CONDITIONS FOR MOUNTAIN WAVES OVER A SCHAER-TYPE MOUNTAIN ON A SMALL PLANET : (S/R DCMIP_SCHAER_MOUNTAIN)',   &
      /,'=====================================================================================================================',/, &
        ' Shear wind = ',I1                                                                                                    ,   &
        ' Set Topo   = ',L2                                                                                                    ,   &
      /,'=====================================================================================================================',/,/)

      end subroutine dcmip_Schaer_mountain
