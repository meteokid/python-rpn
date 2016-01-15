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

!**s/r init_bar - prepare data for autobarotropic runs (Williamson cases)
!
      subroutine init_bar ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo,&
                            Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                            F_trprefix_S, F_trsuffix_S, F_datev )
      use vgrid_wb, only: vgrid_wb_get
      use vertical_interpolation, only: vertint2
      use vGrid_Descriptors
      use nest_blending, only: nest_blend
      implicit none
#include <arch_specific.hf>

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_v (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_w (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_t (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_s (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q (Mminx:Mmaxx,Mminy:Mmaxy,2:Nk+1), &
           F_topo(Mminx:Mmaxx,Mminy:Mmaxy)

#include "gmm.hf"
#include "glb_ld.cdk"
#include "dcst.cdk"
#include "cstv.cdk"
#include "tr3d.cdk"
#include "ver.cdk"
#include "wil_williamson.cdk"

      character(len=4) vname
      integer i,j,n,istat,k
      real, dimension (:,:,:), pointer :: trp
      real, dimension (:,:,:), allocatable :: gz_temp,u_temp,v_temp
      real, dimension (:,:)  , allocatable :: topo_abdes
!
!-----------------------------------------------------------------------
!
      call inp_data ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo,&
                      Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                      F_trprefix_S, F_trsuffix_S, F_datev )

      allocate ( gz_temp(l_ni,l_nj ,G_nk),u_temp(l_niu,l_nj,G_nk),&
                  v_temp(l_ni,l_njv,G_nk),topo_abdes(l_ni,l_nj) )

      if (Williamson_case==1) then
          call wil_case1   (gz_temp,l_ni,l_nj,G_nk)
          call wil_uvcase1 (u_temp,l_niu,l_nj,v_temp,l_ni,l_njv,G_nk)
      endif
      if (Williamson_case==2) then
          call wil_case2   (gz_temp,l_ni,l_nj,G_nk)
          call wil_uvcase2 (u_temp,l_niu,l_nj,v_temp,l_ni,l_njv,G_nk)
      endif
      if (Williamson_case==5) then
          call wil_case5   (gz_temp,topo_abdes,l_ni,l_nj,G_nk)
          call wil_uvcase5 (u_temp,l_niu,l_nj,v_temp,l_ni,l_njv,G_nk)
      endif
      if (Williamson_case==6) then
          call wil_case6   (gz_temp,l_ni,l_nj,G_nk)
          call wil_uvcase6 (u_temp,l_niu,l_nj,v_temp,l_ni,l_njv,G_nk)
      endif
      if (Williamson_case==8) then
          call wil_case8   (gz_temp,l_ni,l_nj,G_nk)
          call wil_uvcase8 (u_temp,l_niu,l_nj,v_temp,l_ni,l_njv,G_nk)
      endif

      !Define log(surface pressure)
      !----------------------------
      if (Williamson_case/=1) then

         if (Williamson_case==5) then
            do j=1,l_nj
            do i=1,l_ni
               F_topo(i,j) = topo_abdes(i,j)*Dcst_grav_8
            enddo
            enddo
         elseif (Williamson_case==7) then
            do j=1,l_nj
            do i=1,l_ni
               F_topo(i,j) = 0.0
            !!!F_topo(i,j) = F_topo(i,j)/10. 
            enddo
            enddo
         endif

         if (Williamson_case/=7) then !Initialized in inp_data for CASE7
         do j=1,l_nj
         do i=1,l_ni

            F_s(i,j) =  (Dcst_grav_8*gz_temp(i,j,1)-F_topo(i,j)) &
                       /(Dcst_Rgasd_8*Cstv_Tstr_8) &
                       +Ver_z_8%m(1)-Cstv_Zsrf_8

         enddo
         enddo
         endif

      else

         F_s = 0.

         do n=1,Tr3d_ntr
            nullify (trp)
            vname= trim(Tr3d_name_S(n))
            if (trim(vname) == 'HU') cycle
            istat= gmm_get (&
                  trim(F_trprefix_S)//trim(vname)//trim(F_trsuffix_S),trp)
            do k=1,G_nk
               trp(1:l_ni,1:l_nj,k) = max(gz_temp(1:l_ni,1:l_nj,1), 0.)
            enddo
         end do

      endif

      F_t= Cstv_Tstr_8 ; F_zd= 0. ; F_w= 0.

      if (Williamson_case/=7) then !Initialized in inp_data for CASE7
      do k=1,G_nk
         F_u(1:l_niu,1:l_nj ,k) = u_temp(1:l_niu,1:l_nj ,1)
         F_v(1:l_ni ,1:l_njv,k) = v_temp(1:l_ni ,1:l_njv,1)
      end do
      endif

      deallocate (gz_temp,u_temp,v_temp,topo_abdes)
!
!-----------------------------------------------------------------------
!
      return
      end
