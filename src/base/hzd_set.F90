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

!**s/r hzd_set - Horizontal diffusion setup
!
      subroutine hzd_set
      implicit none
#include <arch_specific.hf>
!
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
#include "hzd.cdk"
#include "lun.cdk"
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,1002)

      Hzd_lnr = min(max(0.,Hzd_lnr),0.9999999)
      Hzd_pwr = Hzd_pwr / 2
      Hzd_pwr = min(max(2,Hzd_pwr*2),8)
      
      Hzd_lnr_theta= min(max(0.,Hzd_lnr_theta),0.9999999)
      Hzd_pwr_theta= Hzd_pwr_theta / 2
      Hzd_pwr_theta= min(max(2,Hzd_pwr_theta*2),8)
      
      if (Hzd_lnr_tr.lt.0.) Hzd_lnr_tr = Hzd_lnr
      if (Hzd_pwr_tr.lt.0 ) Hzd_pwr_tr = Hzd_pwr
      Hzd_lnr_tr = min(max(0.,Hzd_lnr_tr),0.9999999)
      Hzd_pwr_tr = Hzd_pwr_tr / 2
      Hzd_pwr_tr = min(max(2,Hzd_pwr_tr*2),8)

      call hzd_set_base

      if ((Hzd_lnr.le.0.).and.(Hzd_lnr_theta.le.0.)  &
                         .and.(Hzd_lnr_tr   .le.0.)) then
         if (Lun_out.gt.0) write(Lun_out,1003)
         Hzd_type_S = 'NIL'
      endif

      if (G_lam) then
         call hzd_exp_set
      else
         call hzd_imp_set
      endif

 1002 format(/,'INITIALIZATING HIGH ORDER HORIZONTAL DIFFUSION ',  &
               '(S/R HZD_SET)',/,60('='))
 1003 format(/,'NO HORIZONTAL DIFFUSION REQUIRED',/,32('='))
!
!----------------------------------------------------------------------
!
      return
      end
