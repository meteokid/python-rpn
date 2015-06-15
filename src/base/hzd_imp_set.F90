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

!**s/r hzd_imp_set - Compute implicit diffusion operator
!
      subroutine hzd_imp_set
      implicit none
#include <arch_specific.hf>

!author    
!     J.P. Toviessi - CMC - Jan 1999
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v2_10 - Qaddouri&Desgagne - higher order diffusion operator
! v2_11 - Desgagne M.       - remove vertical modulation
! v2_31 - Qaddouri A.       - remove stkmemw and correction to yp2
! v3_00 - Qaddouri & Lee    - Lam configuration
! v3_01 - Toviessi J. P.    - add eigenmodes with definite parity
! v3_01 - Lee V.            - add setup for horizontal sponge
! v3_20 - Qaddouri/Toviessi - variable higher order diffusion operator
! v3_20 - Tanguay M.        - 1d higher order diffusion operator
! v3_30 - Tanguay M.        - activate Hzd_type_S='HO_EXP' 
! v4_40 - Lee V.            - allow matrix setup only when Hzd_type_S="HO_IMP"
! v4_70 - Desgagne M.       - major revision

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "hzd.cdk"
#include "dcst.cdk"
#include "fft.cdk"
#include "trp.cdk"
#include "opr.cdk"
#include "cstv.cdk"
#include "lun.cdk"
#include "ptopo.cdk"

      integer i,j,k,dim,dim1,dim2,dpwr1,dpwr2,NSTOR,dimx,dimy
      real*8, dimension(:) , allocatable :: wk1_8,wk2_8
      real*8 c_8
      real*8, parameter :: ZERO_8= 0.d0, ONE_8=1.d0, HALF_8=0.5d0
      real  , parameter :: eps= 1.0e-5
!
!     ---------------------------------------------------------------

      call hzd_imp_transpose ( Ptopo_npex, Ptopo_npey, .false. )

      if (Hzd_lnr_theta.gt.0.) then
         if ( (Hzd_pwr_theta.ne.Hzd_pwr) .or. &
              (abs((Hzd_lnr_theta-Hzd_lnr)/Hzd_lnr).gt.eps)) then
            if (Lun_out.gt.0) write (Lun_out,1001)
            call handle_error(-1,'hzd_set','PROBLEM WITH THETA DIFFUSION')
         endif
      endif

      dimx = 3*G_ni*2
      dimy = 3*G_nj*2

      !     Compute eigenvalues and eigenvector for high-order diffusion.
      !              Eigenvalue problem in East-West direction
      !        -------------------------------------------------------

      allocate ( Hzd_xeval_8 (G_ni*2) )

      if ( .not. Fft_fast_L ) then

         allocate ( Hzd_xevec_8 (G_ni*G_ni*2) )
         call set_poic  ( Hzd_xeval_8, Hzd_xevec_8 , Hzd_xp0_8, &
                          Hzd_xp2_8, G_ni, G_ni )
         allocate ( Hzd_wevec_8(G_ni*G_ni), Hzd_wuevec_8(G_ni*G_ni) )
         do j=1,G_ni
            do i=1,G_ni
               Hzd_wuevec_8((j-1)*G_ni+i)=Hzd_xevec_8((j+Lam_pil_w-1)*G_ni+i+Lam_pil_w)
               Hzd_wevec_8 ((j-1)*G_ni+i)=Opr_xevec_8((j+Lam_pil_w-1)*G_ni+i+Lam_pil_w)
            enddo
         enddo
         
      else

         c_8 = Dcst_pi_8 / dble( G_ni )
         Hzd_xeval_8(1)    =   ZERO_8
         Hzd_xeval_8(G_ni) = - ONE_8 / ( c_8 ** 2. )
         do i = 1, (G_ni-1)/2
            Hzd_xeval_8(2*i+1) = - (sin(dble(i) * c_8) / c_8) **2.
            Hzd_xeval_8(2*i)   =  Hzd_xeval_8(2*i+1)
         end do

      endif
      
      !     initialize operator nord_south for U, V and scalar grids
      !                              for high-order diffusion-solver
      
      dpwr1= Hzd_pwr    / 2
      dpwr2= Hzd_pwr_tr / 2
      dim = (trp_22max-trp_22min+1)
      dim1 = dim * G_nj * dpwr1*dpwr1
      dim2 = dim * G_nj * dpwr2*dpwr2

      allocate ( Hzd_au_8  (dim1),Hzd_cu_8  (dim1),Hzd_deltau_8  (dim1), &
           Hzd_av_8  (dim1),Hzd_cv_8  (dim1),Hzd_deltav_8  (dim1), &
           Hzd_as_8  (dim1),Hzd_cs_8  (dim1),Hzd_deltas_8  (dim1), &
           Hzd_astr_8(dim2),Hzd_cstr_8(dim2),Hzd_deltastr_8(dim2)  )

      i=G_ni/2
      j=G_nj/2
      c_8= min ( G_xg_8(i+1) - G_xg_8(i), G_yg_8(j+1) - G_yg_8(j) )

      Hzd_cdiff = 0. ; Hzd_cdiff_tr = 0.
      if (Hzd_lnR.gt.0) &
      Hzd_cdiff    = (2./c_8)**Hzd_pwr    / (-log(1.- Hzd_lnR   ))
      if (Hzd_lnR_tr.gt.0) &
      Hzd_cdiff_tr = (2./c_8)**Hzd_pwr_tr / (-log(1.- Hzd_lnR_tr))

      if (Lun_out.gt.0) then
         write(Lun_out,1010)  &
              (Dcst_rayt_8**2.)/(Cstv_dt_8*Hzd_cdiff),Hzd_pwr/2,'U,V,W,ZD,THETA'
         write(Lun_out,1010)  &
              (Dcst_rayt_8**2.)/(Cstv_dt_8*Hzd_cdiff_tr),Hzd_pwr_tr/2,'Tracers'
         print*
      endif

      allocate ( wk1_8(3*G_nj), wk2_8(3*G_nj) )
      do j = 1,3*G_nj
         wk1_8 (j) = ZERO_8
         wk2_8 (j) = ZERO_8
      enddo
      do j = 1+Lam_pil_s, G_nj-Lam_pil_n
         wk1_8 (       j) = ZERO_8
         wk1_8 (  G_nj+j) = Opr_opsyp0_8(G_nj+j)*Hzd_h0_8(j)/cos(G_yg_8(j))**2
         wk1_8 (2*G_nj+j) = ZERO_8
         wk2_8 (       j) = ZERO_8
         wk2_8 (  G_nj+j) = (sin(G_yg_8(j+1))-sin(G_yg_8(j)))*Hzd_h0_8(j)/ &
              ( cos ( (G_yg_8(j+1  )+G_yg_8(j))* HALF_8) **2)
         wk2_8 (2*G_nj+j) = ZERO_8
      end do

      call hzd_delpwr (Hzd_av_8,Hzd_cv_8,Hzd_deltav_8,dpwr1, &
           trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
           Hzd_yp0_8,Hzd_yp2_8,wk2_8, &
           Opr_xeval_8, Hzd_cdiff)

      if (Hzd_difva_L) then
         call hzd_delpwr (Hzd_au_8,Hzd_cu_8,Hzd_deltau_8,dpwr1, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Hzd_yp2su_8,wk1_8, &
              Hzd_xeval_8, Hzd_cdiff)

         call hzd_delpwr (Hzd_as_8,Hzd_cs_8,Hzd_deltas_8,dpwr1, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Hzd_yp2su_8,wk1_8, &
              Opr_xeval_8, Hzd_cdiff)

         call hzd_delpwr (Hzd_astr_8,Hzd_cstr_8,Hzd_deltastr_8,dpwr2, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Hzd_yp2su_8,wk1_8, &
              Opr_xeval_8, Hzd_cdiff_tr)
      else
         call hzd_delpwr (Hzd_au_8,Hzd_cu_8,Hzd_deltau_8,dpwr1, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Opr_opsyp2_8,wk1_8, &
              Hzd_xeval_8, Hzd_cdiff)

         call hzd_delpwr (Hzd_as_8,Hzd_cs_8,Hzd_deltas_8,dpwr1, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Opr_opsyp2_8,wk1_8, &
              Opr_xeval_8, Hzd_cdiff)

         call hzd_delpwr (Hzd_astr_8,Hzd_cstr_8,Hzd_deltastr_8,dpwr2, &
              trp_22min,trp_22max,G_nj,trp_22n,trp_22n0, &
              Opr_opsyp0_8,Opr_opsyp2_8,wk1_8, &
              Opr_xeval_8, Hzd_cdiff_tr)
      endif

      deallocate (wk1_8,wk2_8)

 1001 format(/,'Horizontal Diffusion of THETA with HO_IMP only available if (Hzd_pwr_theta=Hzd_pwr).and.(Hzd_lnr_theta=Hzd_lnr)')
 1010 format (X,'Diffusion Coefficient = (',e22.14,' m**2)**',i1,'/sec ',a )
!
!     ---------------------------------------------------------------
      return
      end
