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



      subroutine set_betav_2(betav_m,betav_t,F_s,F_sl, Minx,Maxx,Miny,Maxy, Nk)

      use grid_options
      use gem_options
      use theo_options
      use tdpack
      implicit none
#include <arch_specific.hf>

      integer  Minx,Maxx,Miny,Maxy, Nk

#include "glb_pil.cdk"
#include "glb_ld.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "dcst.cdk"

      real betav_m(Minx:Maxx,Miny:Maxy,Nk),betav_t(Minx:Maxx,Miny:Maxy,Nk)
      real F_s(Minx:Maxx,Miny:Maxy),F_sl(Minx:Maxx,Miny:Maxy)
!
      real Zblen_max,Zblen_top

      real*8 work1,work2,fact

      integer i,j,k,i0,in,j0,jn

      zblen_max=Cstv_Zsrf_8*(1.-Zblen_hmin/(287.*290.))
      zblen_top=Cstv_ztop_8
      fact=1.d0
      if(Theo_case_S .eq. 'MTN_SCHAR' ) then
         fact=sqrt(2.0*mtn_flo*Cstv_dt_8/Grd_dx/(Dcst_rayt_8*pi_8/180.))
      endif

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      if (l_west ) i0 = 1+pil_w
      if (l_east ) in = l_ni-pil_e
      if (l_south) j0 = 1+pil_s
      if (l_north) jn = l_nj-pil_n

      do k=1,l_nk
         do j=j0,jn
            do i=i0,in
               work1=zblen_max-(Ver_a_8%m(k)+Ver_b_8%m(k)*F_s(i,j)+Ver_c_8%m(k)*F_sl(i,j))
               work2=zblen_max-zblen_top
               work1=min(1.d0,max(0.d0,work1/work2))
               betav_m(i,j,k)=work1*work1*min(1.d0,fact)
               work1=zblen_max-(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(i,j)+Ver_c_8%t(k)*F_sl(i,j))
               work1=min(1.d0,max(0.d0,work1/work2))
               betav_t(i,j,k)=work1*work1*min(1.d0,fact)
            enddo
         enddo
      enddo

      return

      end

subroutine set_betav
   print*,'Called a stub, please update to set_betav_2'
   stop
end subroutine set_betav
