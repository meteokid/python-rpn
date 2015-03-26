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

!**s/r get_density - Evaluate Fluid's density and mass 

      subroutine get_density ( F_density, F_mass, F_time, &
                               Minx,Maxx,Miny,Maxy,F_nk,k0) 

      implicit none

      !Arguments
      !---------
      integer,                                    intent(in) :: F_time              !I, Time 0 or Time 1
      integer,                                    intent(in) :: Minx,Maxx,Miny,Maxy !I, Dimension H  
      integer,                                    intent(in) :: k0                  !I, scope of operator
      integer,                                    intent(in) :: F_nk                !I, number of vertical levels

      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),  intent(out):: F_density           !O, Fluid's density 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),  intent(out):: F_mass              !O, Fluid's mass 

      !@author  Monique Tanguay 
      !@revisions
      ! v4_70 - Tanguay,M.        - Initial Version
      ! v4_70 - Qaddouri,A.       - Version for Yin-Yang Grid 

!*@/
#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "ver.cdk"
#include "vt0.cdk"
#include "vt1.cdk"
#include "wil_williamson.cdk"
!-----------------------------------------------------------------------------

      real, pointer, dimension(:,:,:)  :: rho
      real, pointer, dimension(:,:)    :: w2d 
      integer i,j,k,istat
      real, dimension(Minx:Maxx,Miny:Maxy)         :: pr_p0
      real, dimension(Minx:Maxx,Miny:Maxy,1:F_nk+1):: pr_m,pr_t
     
      !Extracted form linoz_o3col (J. de Grandpre) !!! how about cstv.cdk
      !-------------------------------------------
      real*8, parameter :: Nav = 6.022142d23 , g0 = 9.80665d0 , air_molmass = 28.9644d0
      real*8, parameter :: cst = (1e-4/g0)*( Nav / (1.e-3*air_molmass) )* 1.D3 / 2.687D19  ! molec/cm2 -> DU

!-----------------------------------------------------------------------------

      !Recuperate GMM variables at appropriate time   
      !--------------------------------------------
      if (.NOT.Advection_2D_3D_L) then

         nullify (w2d)
         if (F_time.eq.0) istat = gmm_get(gmmk_st0_s,w2d)
         if (F_time.eq.1) istat = gmm_get(gmmk_st1_s,w2d)
      else

         nullify (rho)
         if (F_time.eq.0) istat = gmm_get('TR/HU:'//'M',rho)
         if (F_time.eq.1) istat = gmm_get('TR/HU:'//'P',rho)

      endif

      !Evaluate Pressure based on pw_update_GPW 
      !----------------------------------------
      if (.NOT.Advection_2D_3D_L) then

         call calc_pressure ( pr_m, pr_t, pr_p0, w2d, &
                              l_minx,l_maxx, l_miny,l_maxy, G_nk )
         pr_m(:,:,F_nk+1) = pr_p0(:,:)  

      endif

      !Evaluate Fluid's density and mass
      !---------------------------------
      if (.NOT.Advection_2D_3D_L) then

!$omp parallel do private(i,j) shared(pr_m) 
      do k=k0,F_nk

         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e

               F_density(i,j,k) =       (pr_m(i,j,k+1) - pr_m(i,j,k)) * Ver_idz_8%t(k)

               F_mass   (i,j,k) = cst * (pr_m(i,j,k+1) - pr_m(i,j,k)) * Geomg_area_8(i,j)

         end do
         end do

      end do
!$omp end parallel do

      else

!$omp parallel do private(i,j)
      do k=k0,F_nk

         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e

               F_density(i,j,k) = rho(i,j,k)
               F_mass   (i,j,k) = rho(i,j,k) * Geomg_area_8(i,j) * Ver_dz_8%t(k)

         end do
         end do

      end do
!$omp end parallel do

      endif

      return
      end
