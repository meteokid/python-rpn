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
!
!**s/p mass_tr - Evaluate Mass of Tracer (assuming in Mixing Ratio)

      subroutine mass_tr (F_mass_tracer_8,F_name_S,F_tracer,F_air_mass,Minx,Maxx,Miny,Maxy,F_nk,F_k0)

      use grid_options
      use gem_options
      use tdpack
      implicit none

      !Arguments
      !---------
      real*8           , intent(out):: F_mass_tracer_8                      !O, Mass of Tracer 
      character (len=4), intent(in) :: F_name_S                             !I, Name of Tracer
      integer,           intent(in) :: Minx,Maxx,Miny,Maxy                  !I, Dimension H
      integer,           intent(in) :: F_k0                                 !I, scope of operator
      integer,           intent(in) :: F_nk                                 !I, number of vertical levels
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),   intent(in) :: F_tracer   !I: Current Tracer (Mixing Ratio)  
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),   intent(in) :: F_air_mass !I: Air mass   
 
      !@author  Monique Tanguay
      !@revisions
      ! v4_70 - Tanguay,M.        - Initial Version
      ! v4_70 - Qaddouri,A.       - Version for Yin-Yang Grid
      ! v5_00 - Tanguay M.        - Air mass provided/Efficiency/Scaling/Mixing Ratio 

!*@/
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "lun.cdk"
#include "tracers.cdk"
#include "cstv.cdk"

      !----------------------------------------------------------
      integer i,j,k,err,i0,in,j0,jn
      real*8   c_mass_8, c_area_8, c_level_8(F_nk), gc_mass_8, scale_8 
      character(len= 9) communicate_S
      real*8, parameter :: QUATRO_8 = 4.0
      logical LAM_L
      logical,save :: done_L=.FALSE.
      real*8 ,save :: gc_area_8

      !----------------------------------------------------------
      !Extracted from linoz_o3col (J.de Grandpre)
      !----------------------------------------------------------
      real*8, parameter :: Nav = 6.022142d23 , g0 = 9.80665d0 , air_molmass = 28.9644d0
      real*8, parameter :: cst = (1e-4/g0)*( Nav / (1.e-3*air_molmass) )* 1.D3 / 2.687D19  ! molec/cm2 -> DU !Ozone (De Grandpre)

      !----------------------------------------------------------
      !Extracted from CO2 (S.Polavarapu)
      !----------------------------------------------------------
      real*8, parameter :: cst2= (1e-21/g0)

      LAM_L = .not.Grd_yinyang_L

      i0 = 1+pil_w
      in = l_ni-pil_e
      j0 = 1+pil_s
      jn = l_nj-pil_n

      if (F_name_S=='FLUX') then
         i0 = 1
         in = l_ni
         j0 = 1
         jn = l_nj
      endif

      !-------------
      !Evaluate Area  
      !-------------
      if (.NOT.done_L) then

         if (.NOT.LAM_L) then

            gc_area_8 = QUATRO_8 * pi_8

         else

             c_area_8 = 0.0d0
            gc_area_8 = 0.0d0

            do j=1+pil_s,l_nj-pil_n !Note: Even with F_name_S=FLUX, we divide by CORE area
            do i=1+pil_w,l_ni-pil_e
               c_area_8 = c_area_8 + Geomg_area_8(i,j)
            enddo
            enddo

            call rpn_comm_ALLREDUCE(c_area_8,gc_area_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err )

         endif

         done_L = .TRUE.

      endif

      !-------------------
      !Evaluate Local Mass  
      !-------------------
!$omp parallel do private(i,j) shared(c_level_8)
      do k=F_k0,F_nk
         c_level_8(k) = 0.0d0
         do j=j0,jn
         do i=i0,in
            c_level_8(k) = c_level_8(k) + F_tracer(i,j,k) * F_air_mass(i,j,k) * Geomg_mask_8(i,j) 
         enddo
         enddo
      enddo
!$omp end parallel do

      c_mass_8 = 0.0d0 

      do k=F_k0,F_nk
         c_mass_8 = c_mass_8 + c_level_8(k) 
      enddo

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      !----------------------------------------
      !Evaluate Global Mass using MPI_ALLREDUCE
      !----------------------------------------
      call rpn_comm_ALLREDUCE(c_mass_8,gc_mass_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err )

      if (Tr_scaling==0.or.Tr_scaling==2.or.Schm_testcases_adv_L) then
         gc_mass_8 = gc_mass_8 / gc_area_8
         if (Schm_autobar_L) gc_mass_8 = gc_mass_8 / (Cstv_pref_8-Cstv_ptop_8) * grav_8
      elseif (Tr_scaling==1) then
         gc_mass_8 = gc_mass_8*rayt_8*rayt_8
      endif

      if (Tr_scaling==0.or.Schm_testcases_adv_L) then
         scale_8 = 1.0d0 
      elseif (Tr_scaling==1) then
         scale_8 = cst2
      elseif (Tr_scaling==2) then
         scale_8 = cst
      endif

      gc_mass_8 = gc_mass_8 * scale_8

      F_mass_tracer_8 = gc_mass_8 

      return
      end
