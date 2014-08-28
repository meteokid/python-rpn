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

!**s/r hzd_exp_visco - applies horizontal explicit diffusion
!
      subroutine hzd_exp_visco2(F_f2hzd, F_grd_S, Minx,Maxx,Miny,Maxy, NK)
      implicit none
#include <arch_specific.hf>

      character*(*) F_grd_S
      integer       Minx,Maxx,Miny,Maxy,Nk
      real          F_f2hzd (Minx:Maxx,Miny:Maxy,Nk)
!
!AUTHORs    C. Girard & M. Desgagne
!
!revision
! v3_21 - Desgagne M.    - initial MPI version
! v3_30 - Desgagne M.    - openMP precision in shared variables
! v4_15 - Desgagne M.    - refonte majeure

#include "glb_ld.cdk"
#include "hzd.cdk"

      integer i, j, k, nn, mm
      real wk1(l_minx:l_maxx,l_miny:l_maxy,Nk),rnr,pwr
      real*8 pt25,nu_dif,epsilon,lnr,visco(nk)

      parameter (epsilon = 1.0d-12, pt25=0.25d0)
!     __________________________________________________________________
!
      rnr = log(1.- Hzd_lnR)
      pwr = Hzd_pwr

      if (F_grd_S.eq.'S_THETA') then
         rnr = log(1.- Hzd_lnR_theta)
         pwr = Hzd_pwr_theta
      endif
      if (F_grd_S.eq.'S_TR') then
         rnr = log(1.- Hzd_lnR_tr)
         pwr = Hzd_pwr_tr
      endif

      if ((F_grd_S.eq.'S_THETA').or.(F_grd_S.eq.'S_TR')) then

         lnr    = 1.0d0 - exp(rnr)
         nu_dif = 0.0d0
         if (pwr.gt.0) nu_dif = pt25*lnr**(2.d0/pwr)
         nu_dif = min ( nu_dif, pt25-epsilon )
         visco  = min ( nu_dif, pt25 )
         if (nu_dif.lt.1.0e-10) return

      else

         do k=1,Nk
            rnr    = log(1.- Hzd_visco_8(k))
            lnr    = 1.0d0 - exp(rnr)
            nu_dif = 0.0d0
            if (Hzd_del(k).gt.0) nu_dif = pt25*lnr**(2.d0/Hzd_del(k))
            nu_dif  = min ( nu_dif, pt25-epsilon )
            visco(k)= min ( nu_dif, pt25 )
            if (nu_dif.lt.1.0e-10) return
         end do

      endif

      nn = pwr/2
      
      call rpn_comm_xch_halo ( F_f2hzd, l_minx,l_maxx,l_miny,l_maxy,&
           l_ni,l_nj, Nk, G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      do mm=1,nn

         call hzd_nudeln2(F_f2hzd, wk1, l_minx,l_maxx,l_miny,l_maxy,&
                                                   Nk, visco, mm,nn )

         if (mm.ne.nn) &
              call rpn_comm_xch_halo( wk1, l_minx,l_maxx,l_miny,l_maxy,&
              l_ni,l_nj, Nk, G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      end do
!     __________________________________________________________________
!
      return
      end
