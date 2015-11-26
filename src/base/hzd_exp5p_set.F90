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

!**s/r hzd_exp5p_set - Horizontal diffusion delN setup for LAMs

      subroutine hzd_exp5p_set
      implicit none
#include <arch_specific.hf>

!author    
!    Abdessamad Qaddouri - summer 2015
!
!revision
! v4_80 - Qaddouri A.      - initial version
! v4_80 - Lee   - optimization
!

#include "glb_ld.cdk"
#include "grd.cdk"
#include "hzd.cdk"
#include "dcst.cdk"
#include "cstv.cdk"
#include "lun.cdk"

      real*8 coef_8,nutop_8,c_8,deg2rad_8
!
!     ---------------------------------------------------------------
!
      deg2rad_8 = acos( -1.0d0 ) / 180.0d0
      c_8= min(Grd_dx,Grd_dy)
      c_8= c_8 * deg2rad_8

      if( Lun_out.gt.0) write(Lun_out,1000)
      !for U,V,W,Zd
      if (Hzd_lnR.gt.0) then
         nutop_8 = 1./4. * Hzd_lnR**(2./Hzd_pwr)
!         if( Lun_out.gt.0) write(Lun_out,1005) 'U,V,W,Zd',Hzd_lnR_theta,&
!                              Hzd_pwr_theta,nutop_8

         Hzd_Niter= max(int(8.d0*nutop_8+0.9999999),1)
         coef_8= nutop_8/max(1.,float(HZD_niter))* &
                                  ((Dcst_rayt_8*c_8)**2)/Cstv_dt_8
         allocate( Hzd_coef_8(G_nk))
         Hzd_coef_8(1:G_nk) = coef_8/(Dcst_rayt_8**2)*Cstv_dt_8
!         if( Lun_out.gt.0)write(Lun_out,*)'Hzd_coef_8=',Hzd_coef_8(1)
         if( Lun_out.gt.0) write(Lun_out,1010) &
           coef_8 ,Hzd_pwr/2,'U,V,W,ZD ',Hzd_Niter
      endif

      !for Theta
      if (Hzd_lnR_theta.gt.0) then
         nutop_8 = 1./4. * Hzd_lnR_theta**(2./Hzd_pwr_theta)
!         if( Lun_out.gt.0) write(Lun_out,1005) 'Theta',Hzd_lnR_theta,&
!                              Hzd_pwr_theta,nutop_8

         Hzd_Niter_theta = max(int(8.d0*nutop_8+0.9999999),1)
         coef_8=nutop_8/max(1.,float(hzd_niter_theta))* &
                                  ((Dcst_rayt_8*c_8)**2)/Cstv_dt_8
         allocate( Hzd_coef_8_theta(G_nk))
         Hzd_coef_8_theta(1:G_nk) = coef_8/(Dcst_rayt_8**2)*Cstv_dt_8
!         if( Lun_out.gt.0)write(Lun_out,*)'Hzd_coef_8_theta=',Hzd_coef_8_theta(1)
         if( Lun_out.gt.0) write(Lun_out,1010) &
             coef_8,Hzd_pwr_theta/2,'Theta ',Hzd_Niter_theta

      endif

      !for Tracers
      if (Hzd_lnR_tr.gt.0) then
         nutop_8 = 1./4. * Hzd_lnR_tr**(2./Hzd_pwr_tr)
!         if( Lun_out.gt.0) write(Lun_out,1005) 'Tracer',Hzd_lnR_tr,&
!                              Hzd_pwr_tr,nutop_8

         Hzd_Niter_tr = max(int(8.d0*nutop_8+0.9999999),1)
         coef_8=nutop_8/max(1.,float(hzd_niter_tr))* &
                                  ((Dcst_rayt_8*c_8)**2)/Cstv_dt_8
         allocate( Hzd_coef_8_tr(G_nk))
         Hzd_coef_8_tr(1:G_nk) = coef_8/(Dcst_rayt_8**2)*Cstv_dt_8
!         if( Lun_out.gt.0)write(Lun_out,*)'Hzd_coef_8_tr=',Hzd_coef_8_tr(1)
         if( Lun_out.gt.0) write(Lun_out,1010) &
             coef_8,Hzd_pwr_tr/2,'Tracer',Hzd_Niter_tr
      endif

1000 format (3X,'For the 5 points diffusion operator:')
1005 format (3X,a,' lnR=',e18.7,' pwr=',i1,' nutop=', e18.7)
1010 format (3X,'Diffusion Coefficient =  (',e15.10,' m**2)**',i1,'/sec ',a,' Niter=',i2 )
!
!     ---------------------------------------------------------------
!
      return
      end
