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
!**s/p mass_tr - Evaluate Mass of Tracer. F_tracer is Mixing Ratio if F_mixing_L=T 
!                (otherwise F_tracer is Tracer's Density)

      subroutine mass_tr (F_mass_tracer_8,F_time,F_name_S,F_tracer,F_mixing_L,  &
                          Minx,Maxx,Miny,Maxy,F_nk,k0,F_comment_S,F_reset_RHO_L)
      implicit none

      !Arguments
      !---------
      real*8           , intent(out):: F_mass_tracer_8                    !O, Mass of Tracer 
      integer          , intent(in) :: F_time                             !I, Time 0 or Time 1
      character (len=4), intent(in) :: F_name_S                           !I, Name of Tracer
      logical          , intent(in) :: F_mixing_L                         !I, T if F_tracer is Mixing Ratio
      integer,           intent(in) :: Minx,Maxx,Miny,Maxy                !I, Dimension H
      integer,           intent(in) :: k0                                 !I, scope of operator
      integer,           intent(in) :: F_nk                               !I, number of vertical levels
      character (len=*), intent(in) :: F_comment_S                        !I, Comment (If="", not print is done)  
      logical          , intent(in) :: F_reset_RHO_L                      !I, call get_density if T  
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk),   intent(in) :: F_tracer !I: Current Tracer (Density or Mixing Ratio)  
 
      !@author  Monique Tanguay
      !@revisions
      ! v4_70 - Tanguay,M.        - Initial Version
      ! v4_70 - Qaddouri,A.       - Version for Yin-Yang Grid

!*@/
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "lun.cdk"
#include "dcst.cdk"
#include "ver.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "ptopo.cdk"
#include "tracers.cdk"

      !----------------------------------------------------------
      integer i,j,k,err,i0,in,j0,jn
      real*8   c_mass_8, s_mass_8, c_area_8, s_area_8, & 
              gc_mass_8,gs_mass_8,gp_mass_8,scale_8
      real, dimension(:,:,:), allocatable, save :: density,mass
      character(len=15) type_S
      character(len= 7) time_S
      character(len= 9) communicate_S
      real*8, parameter :: QUATRO_8 = 4.0
      logical,parameter :: SAROJA_L=.TRUE. 
      logical LAM_L
      logical,save :: done_L=.FALSE.
      real*8 ,save :: gc_area_8,gs_area_8 

      !Extracted form linoz_o3col (J. de Grandpre)
      !----------------------------------------------------------
      real*8, parameter :: Nav = 6.022142d23 , g0 = 9.80665d0 , air_molmass = 28.9644d0
      real*8, parameter :: cst = (1e-4/g0)*( Nav / (1.e-3*air_molmass) )* 1.D3 / 2.687D19  ! molec/cm2 -> DU

      real*8, parameter :: cst2= (1e-21/g0) !SAROJA
      !----------------------------------------------------------

      LAM_L = G_lam.and..not.Grd_yinyang_L

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

      if (.not. allocated(density)) allocate(density(Minx:Maxx,Miny:Maxy,F_nk))
      if (.not. allocated(   mass)) allocate(   mass(Minx:Maxx,Miny:Maxy,F_nk))

      if (F_reset_RHO_L) call get_density (density,mass,F_time,Minx,Maxx,Miny,Maxy,F_nk,k0)

      if (.NOT.F_mixing_L.and.F_name_S/='RHO ') density = F_tracer

      !Evaluate Area(s)  
      !----------------
      if (.NOT.done_L) then

         if (.NOT.LAM_L) then

            gc_area_8 = QUATRO_8 * Dcst_pi_8

             s_area_8 = 0.0d0
            gs_area_8 = 0.0d0

            !Evaluate area on a subset of Yin or Yan
            !---------------------------------------
            if (Grd_yinyang_L) then

            do j=1+Tr_pil_sub_s,l_nj-Tr_pil_sub_n
            do i=1+Tr_pil_sub_w,l_ni-Tr_pil_sub_e
               s_area_8 = s_area_8 + Geomg_area_8(i,j)
            enddo
            enddo

            call rpn_comm_ALLREDUCE(s_area_8,gs_area_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err )

            endif

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

      !Evaluate Local Mass  
      !-------------------
      c_mass_8 = 0.0d0
      s_mass_8 = 0.0d0

      if (F_mixing_L) then

         if (Grd_yinyang_L) then

         do k=k0,F_nk
            do j=j0,jn
            do i=i0,in
               c_mass_8 = c_mass_8 + F_tracer(i,j,k) * mass(i,j,k) * Geomg_mask_8(i,j) 
            enddo
            enddo
         enddo

         else 

         do k=k0,F_nk
            do j=j0,jn
            do i=i0,in
               c_mass_8 = c_mass_8 + F_tracer(i,j,k) * mass(i,j,k)
            enddo
            enddo
         enddo

         endif

         !Evaluate mass on a subset of Yin or Yan
         !---------------------------------------
         if (Grd_yinyang_L) then

         do k=k0,F_nk
            do j=1+Tr_pil_sub_s,l_nj-Tr_pil_sub_n
            do i=1+Tr_pil_sub_w,l_ni-Tr_pil_sub_e
               s_mass_8 = s_mass_8 + F_tracer(i,j,k) * mass(i,j,k)
            enddo
            enddo
         enddo

         endif 

      else

         do k=k0,F_nk
            do j=j0,jn
            do i=i0,in
               c_mass_8 = c_mass_8 + density(i,j,k) * Geomg_area_8(i,j) * Ver_dz_8%t(k) * Geomg_mask_8(i,j)
           enddo
           enddo
         enddo

         !Evaluate mass on a subset of Yin or Yan
         !---------------------------------------
         if (Grd_yinyang_L) then

         do k=k0,F_nk
            do j=1+Tr_pil_sub_s,l_nj-Tr_pil_sub_n
            do i=1+Tr_pil_sub_w,l_ni-Tr_pil_sub_e
               s_mass_8 = s_mass_8 + density(i,j,k) * Geomg_area_8(i,j) * Ver_dz_8%t(k)
           enddo
           enddo
         enddo

         endif

      endif

      communicate_S = "GRID"
      if (Grd_yinyang_L) communicate_S = "MULTIGRID"

      gp_mass_8 = 0.0d0
      gs_mass_8 = 0.0d0

      !----------------------------------------
      !Evaluate Global Mass using MPI_ALLREDUCE
      !----------------------------------------
      call rpn_comm_ALLREDUCE(c_mass_8,gc_mass_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM",communicate_S,err )
      if (Grd_yinyang_L) then 
      call rpn_comm_ALLREDUCE(s_mass_8,gs_mass_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID"       ,err )
      call rpn_comm_ALLREDUCE(c_mass_8,gp_mass_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID"       ,err )
      endif

      if ((G_lam.or.F_name_S=='RHO ').or..NOT.SAROJA_L) then
         gc_mass_8 = gc_mass_8 / gc_area_8
      else
         gc_mass_8 = gc_mass_8*Dcst_rayt_8*Dcst_rayt_8
      endif

      if (Grd_yinyang_L) then
          gp_mass_8 = gp_mass_8 / gc_area_8 * 2.0d0
          gs_mass_8 = gs_mass_8 / gs_area_8
      endif

      if (G_lam.or..NOT.SAROJA_L) then
         scale_8 = cst
      else
         scale_8 = cst2
      endif

      if (Tr_2D_3D_L.or.F_name_S=='RHO '.or..NOT.F_mixing_L) scale_8 = 1.0d0

      gc_mass_8 = gc_mass_8 * scale_8
      gp_mass_8 = gp_mass_8 * scale_8
      gs_mass_8 = gs_mass_8 * scale_8

      if (.NOT.F_mixing_L) type_S = "Mass of Density"
      if (     F_mixing_L) type_S = "Mass of Mixing "

      if( F_time==1) time_S = "TIME T1"
      if (F_time==0) time_S = "TIME T0"

      if (Lun_out>0.and.F_comment_S/="") then

          write(Lun_out,1002) 'TRACERS: ',type_S,time_S,' C= ',gc_mass_8,F_name_S,F_comment_S,' COLOR=',Ptopo_couleur

          if (Grd_yinyang_L) then
          write(Lun_out,1002) 'TRACERS: ',type_S,time_S,' P= ',gp_mass_8,F_name_S,F_comment_S,' COLOR=',Ptopo_couleur
          write(Lun_out,1002) 'TRACERS: ',type_S,time_S,' S= ',gs_mass_8,F_name_S,F_comment_S,' COLOR=',Ptopo_couleur
          endif

      endif

      F_mass_tracer_8 = gc_mass_8 

 1002 format(1X,A9,A15,1X,A7,A4,E19.12,1X,A4,1X,A16,A7,I1)

      return
      end
