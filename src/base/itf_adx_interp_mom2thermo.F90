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

   subroutine itf_adx_interp_mom2thermo2 (F_fld_t,F_fld_m, &
                                   minx,maxx,miny,maxy,nk)
      implicit none
#include <arch_specific.hf>

      !@objective interpolate from momentum to thermodynamic levels
      !@arguments

      integer minx,maxx,miny,maxy,nk
      real F_fld_t(minx:maxx,miny:maxy,nk+1), &
           F_fld_m(minx:maxx,miny:maxy,nk  )

      !@revisions
      ! v4_40 - Qaddouri/Lee      - Yin-Yang, to use global range
      !*@/

#include "glb_ld.cdk"
#include "grd.cdk"
#undef TYPE_CDK
#include "ver.cdk"
#include "schm.cdk"
#include "cstv.cdk"

      integer :: i,j,k, i0,j0,in,jn
      real*8  :: xx, x1, x2, x3, x4, w1, w2, w3, w4,den
      real*8  :: zd_z_8(l_nk+1)

#define lag3(xx, x1, x2, x3, x4)  (((xx - x2) * (xx - x3) * (xx - x4))/( (x1 - x2) * (x1 - x3) * (x1 - x4)))

      !---------------------------------------------------------------------

      zd_z_8(1)=Cstv_Ztop_8
      zd_z_8(2:l_nk+1)=Ver_a_8%t(1:l_nk)
      zd_z_8(l_nk+1)=Cstv_Zsrf_8
      if(Schm_Tlift.eq.1)then
         zd_z_8(l_nk+1)=Ver_z_8%m(l_nk+1)
         call handle_error(-1,'itf_adx_interp_mom2thermo','review for Tlift')
      endif

!$omp parallel private(i0,in,j0,jn,xx,x1,x2,x3,x4,&
!$omp                  i,j,k,w1,w2,w3,w4,den)
      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      if (G_lam.and..not.Grd_yinyang_L) then
         if (l_west)  i0 = 3
         if (l_east)  in = l_ni - 1
         if (l_south) j0 = 3
         if (l_north) jn = l_nj - 1
      endif

!$omp do
      do k = 3, l_nk-1
         xx = zd_z_8(k)
         x1 = Ver_z_8%m(k-2)
         x2 = Ver_z_8%m(k-1)
         x3 = Ver_z_8%m(k  )
         x4 = Ver_z_8%m(k+1)
         w1 = lag3(xx, x1, x2, x3, x4)
         w2 = lag3(xx, x2, x1, x3, x4)
         w3 = lag3(xx, x3, x1, x2, x4)
         w4 = lag3(xx, x4, x1, x2, x3)
         do j = j0, jn
            do i = i0, in
               F_fld_t(i,j,k) = &
                    w1*F_fld_m(i,j,k-2) + w2*F_fld_m(i,j,k-1) + &
                    w3*F_fld_m(i,j,k  ) + w4*F_fld_m(i,j,k+1)
            enddo
         enddo
      enddo
!$omp enddo

      den = 1.d0/(Ver_z_8%m(1)-Ver_z_8%m(2))
      k = 2      
      w1 = (zd_z_8(k)   -Ver_z_8%m(2)) * den
      w2 = (Ver_z_8%m(1)-zd_z_8(k)   ) * den

!$omp do     
      do j = j0, jn
         do i = i0, in
            F_fld_t(i,j,k) = w1*F_fld_m(i,j,1) + w2*F_fld_m(i,j,2)
         enddo
      enddo
!$omp enddo

      !- Updward Extrapolation
      k = 1      
      w1 = (zd_z_8(k)   -Ver_z_8%m(2)) * den
      w2 = (Ver_z_8%m(1)-zd_z_8(k)   ) * den

!$omp do     
      do j = j0, jn
         do i = i0, in
            F_fld_t(i,j,k) = w1*F_fld_m(i,j,1) + w2*F_fld_m(i,j,2)
         enddo
      enddo
!$omp enddo

      den = 1.d0/(Ver_z_8%m(l_nk-1)-Ver_z_8%m(l_nk))
      k = l_nk
      w1 = (zd_z_8(k)        -Ver_z_8%m(l_nk)) * den
      w2 = (Ver_z_8%m(l_nk-1)-zd_z_8(k)      ) * den

!$omp do     
      do j = j0, jn
         do i = i0, in
            F_fld_t(i,j,k) = w1*F_fld_m(i,j,l_nk-1) + w2*F_fld_m(i,j,l_nk)
         enddo
      enddo
!$omp enddo

      !- Downdward Extrapolation
      k = l_nk+1
      w1 = (zd_z_8(k)        -Ver_z_8%m(l_nk)) * den
      w2 = (Ver_z_8%m(l_nk-1)-zd_z_8(k)      ) * den

!$omp do     
      do j = j0, jn
         do i = i0, in
            F_fld_t(i,j,k) = w1*F_fld_m(i,j,l_nk-1) + w2*F_fld_m(i,j,l_nk)
         enddo
      enddo
!$omp enddo
!$omp end parallel

      !---------------------------------------------------------------------
      return
   end subroutine itf_adx_interp_mom2thermo2
