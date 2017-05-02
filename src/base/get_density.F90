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

      subroutine get_density (F_density,F_mass,F_time,Minx,Maxx,Miny,Maxy,F_nk,F_k0)

      use gmm_vt1
      implicit none

      !Arguments
      !---------
      integer,                                    intent(in) :: F_time              !I, Time 0 or Time 1
      integer,                                    intent(in) :: Minx,Maxx,Miny,Maxy !I, Dimension H  
      integer,                                    intent(in) :: F_k0                !I, scope of operator
      integer,                                    intent(in) :: F_nk                !I, number of vertical levels

      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),  intent(out):: F_density           !O, Fluid's density 
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),  intent(out):: F_mass              !O, Fluid's mass 

      !@author  Monique Tanguay 
      !@revisions
      ! v4_70 - Tanguay,M.        - Initial Version
      ! v4_70 - Qaddouri,A.       - Version for Yin-Yang Grid 
      ! v5_00 - Tanguay M.        - Adjust density

!*@/
#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "ver.cdk"
#include "vt0.cdk"
#include "schm.cdk"
#include "wil_williamson.cdk"
#include "dcst.cdk"
#include "tracers.cdk"
!-----------------------------------------------------------------------------

      real, pointer, dimension(:,:)   :: w2d 
      integer i,j,k,istat
      real, dimension(Minx:Maxx,Miny:Maxy)         :: pr_p0
      real, dimension(Minx:Maxx,Miny:Maxy,1:F_nk+1):: pr_m,pr_t
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk)    :: sumq
      real, pointer, dimension(:,:,:)              :: tr
      character*1 timelevel_S
      logical Analytical_Density_L

!-----------------------------------------------------------------------------

      Analytical_Density_L = Williamson_case>0

      !Recuperate GMM variables at appropriate time   
      !--------------------------------------------
      nullify (w2d)
      if (F_time.eq.0) istat = gmm_get(gmmk_st0_s,w2d)
      if (F_time.eq.1) istat = gmm_get(gmmk_st1_s,w2d)

      !Evaluate Pressure based on pw_update_GPW 
      !----------------------------------------
      call calc_pressure ( pr_m, pr_t, pr_p0, w2d, l_minx, l_maxx, l_miny, l_maxy, F_nk ) 

      pr_m(:,:,F_nk+1) = pr_p0(:,:)  

      !Evaluate water tracers if dry mixing ratio
      !------------------------------------------
      sumq = 0.

      if (Schm_dry_mixing_ratio_L) then
           
         if (F_time.eq.1) timelevel_S = 'P'
         if (F_time.eq.0) timelevel_S = 'M'

         call sumhydro (sumq,l_minx,l_maxx,l_miny,l_maxy,l_nk,timelevel_S)

         istat = gmm_get('TR/HU:'//timelevel_S,tr)

!$omp    parallel do shared(sumq,tr)
         do k=1,l_nk
            sumq(1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)= &
            sumq(1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)+ &
            tr  (1+pil_w:l_ni-pil_e,1+pil_s:l_nj-pil_n,k)
         end do
!$omp    end parallel do

      endif

      !Evaluate Fluid's density and mass
      !---------------------------------
      if (.NOT.Analytical_Density_L) then

!$omp    parallel do private(i,j) shared(pr_m,sumq)
         do k=F_k0,F_nk
            do j=1,l_nj
            do i=1,l_ni
               F_density(i,j,k) = +(pr_m(i,j,k+1) - pr_m(i,j,k)) * (1.-sumq(i,j,k)) * Ver_idz_8%t(k) / Dcst_grav_8
            end do
            end do
         end do
!$omp    end parallel do

      else

!$omp    parallel do private(i,j) shared(pr_t,w2d)
         do k=F_k0,F_nk
            do j=1,l_nj
            do i=1,l_ni
               F_density(i,j,k) = +pr_t(i,j,k) * (1.0 + Ver_dbdz_8%t(k) * w2d(i,j)) / Dcst_grav_8
            end do
            end do
         end do
!$omp    end parallel do

      endif

!$omp parallel do private(i,j)
      do k=F_k0,F_nk
         do j=1,l_nj
         do i=1,l_ni
            F_mass(i,j,k) = F_density(i,j,k) * Geomg_area_8(i,j) * Ver_dz_8%t(k)
         end do
         end do
      end do
!$omp end parallel do

      return
      end
