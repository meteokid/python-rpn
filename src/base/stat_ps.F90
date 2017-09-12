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

!**s/r stat_ps - Evaluate average of wet/dry surface pressure (Based on psadj)
!
      subroutine stat_ps

      use gmm_vt1
      use grid_options
      use gem_options
      use geomh
      use tdpack
      use glb_ld
      use cstv
      use lun
      use tr3d
      use gmm_itf_mod
      use glb_pil
      use tracers
      use ptopo
      implicit none

!author M.Tanguay

!revision
! v4_80 - Tanguay M.        - initial MPI version


      type(gmm_metadata) :: mymeta
      integer err, i, j, istat, k
      real, pointer, dimension(:,:,:)                        :: tr
      real,   dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk)    :: sumq
      real*8, dimension(l_minx:l_maxx,l_miny:l_maxy)         :: pr_p0_8
      real*8, dimension(l_minx:l_maxx,l_miny:l_maxy,1:l_nk+1):: pr_m_8,pr_t_8
      real*8 :: l_avg_8(4),gc_avg_8(2),gp_avg_8(2),gs_avg_8(2),c_area_8,gc_area_8,s_area_8,gs_area_8
      real*8, parameter ::  ZERO_8 = 0.0, QUATRO_8 = 4.0
      character (len=12) type_S
      logical LAM_L
!     _________________________________________________________________

      LAM_L = .not.Grd_yinyang_L

      type_S = "GRID"
      if (Grd_yinyang_L) type_S = "MULTIGRID"

      istat = gmm_get(gmmk_st1_s,st1,mymeta)

      !Obtain pressure levels
      !----------------------
      call calc_pressure_8 (pr_m_8,pr_t_8,pr_p0_8,st1,l_minx,l_maxx,l_miny,l_maxy,l_nk)

      pr_m_8(:,:,l_nk+1) = pr_p0_8(:,:)

      call sumhydro (sumq,l_minx,l_maxx,l_miny,l_maxy,l_nk,'P')

      istat = gmm_get('TR/HU:P',tr)

!$omp parallel do private(k) shared(sumq,tr)
      do k=1,l_nk
         sumq(1:l_ni,1:l_nj,k) = sumq(1:l_ni,1:l_nj,k) + tr(1:l_ni,1:l_nj,k)
      end do
!$omp end parallel do

      l_avg_8 = 0.0d0

      do j=1+pil_s,l_nj-pil_n
      do i=1+pil_w,l_ni-pil_e
         do k=1,l_nk
            l_avg_8(1) = l_avg_8(1) + geomh_area_8(i,j)*geomh_mask_8(i,j)                      *(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
            l_avg_8(2) = l_avg_8(2) + geomh_area_8(i,j)*geomh_mask_8(i,j)*(1.0d0 - sumq(i,j,k))*(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
         enddo
      enddo
      enddo

      if (Grd_yinyang_L) then

         do j=1+Tr_pil_sub_s,l_nj-Tr_pil_sub_n
         do i=1+Tr_pil_sub_w,l_ni-Tr_pil_sub_e
            do k=1,l_nk
               l_avg_8(3) = l_avg_8(3) + geomh_area_8(i,j)*geomh_mask_8(i,j)                      *(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
               l_avg_8(4) = l_avg_8(4) + geomh_area_8(i,j)*geomh_mask_8(i,j)*(1.0d0 - sumq(i,j,k))*(pr_m_8(i,j,k+1) - pr_m_8(i,j,k))
            enddo
         enddo
         enddo

      endif

      call rpn_comm_ALLREDUCE (l_avg_8,gc_avg_8,2,"MPI_DOUBLE_PRECISION","MPI_SUM",type_S,err)
      if (Grd_yinyang_L) then
      call rpn_comm_ALLREDUCE (l_avg_8(3:4),gs_avg_8(1:2),2,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err)
      call rpn_comm_ALLREDUCE (l_avg_8(1:2),gp_avg_8(1:2),2,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err)
      endif

      if (.NOT.LAM_L) then

         gc_area_8 = QUATRO_8 * pi_8

          s_area_8 = 0.0d0
         gs_area_8 = 0.0d0

         if (Grd_yinyang_L) then

         do j=1+Tr_pil_sub_s,l_nj-Tr_pil_sub_n
         do i=1+Tr_pil_sub_w,l_ni-Tr_pil_sub_e
            s_area_8 = s_area_8 + geomh_area_8(i,j)
         enddo
         enddo

         call rpn_comm_ALLREDUCE (s_area_8,gs_area_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err)

         endif

      else

          c_area_8 = 0.0d0
         gc_area_8 = 0.0d0

         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            c_area_8 = c_area_8 + geomh_area_8(i,j)
         enddo
         enddo

         call rpn_comm_ALLREDUCE(c_area_8,gc_area_8,1,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",err )

      endif

      gc_avg_8(1:2) = gc_avg_8(1:2) / gc_area_8

      if (Grd_yinyang_L) then
          gp_avg_8(1:2) = gp_avg_8(1:2) / gc_area_8 * 2.0d0
          gs_avg_8(1:2) = gs_avg_8(1:2) / gs_area_8
      endif

      if (Lun_out>0) then

         write(Lun_out,*)    ''
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE WET',' C= ',gc_avg_8(1),' COLOR=',Ptopo_couleur
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE DRY',' C= ',gc_avg_8(2),' COLOR=',Ptopo_couleur

         if (Grd_yinyang_L) then
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE WET',' P= ',gp_avg_8(1),' COLOR=',Ptopo_couleur
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE DRY',' P= ',gp_avg_8(2),' COLOR=',Ptopo_couleur
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE WET',' S= ',gs_avg_8(1),' COLOR=',Ptopo_couleur
         write(Lun_out,1002) 'AVERAGE SURFACE PRESSURE DRY',' S= ',gs_avg_8(2),' COLOR=',Ptopo_couleur
         endif
         write(Lun_out,*)    ''

      endif

 1002 format(1X,A28,1X,A4,E19.12,1X,A7,1X,I1)

!     _________________________________________________________________
!
      return
      end
