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

!**s/r out_dyn_casc - model output for cascade

      subroutine out_dyn_casc
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      use out_vref_mod, only: out_vref
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Desgagne  -  summer 2015
!revision
! v4_80 - Desgagne M.       - initial version

#include "gmm.hf"
#include "glb_ld.cdk"
#include "p_geof.cdk"
#include "vt1.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "schm.cdk"
#include "grdc.cdk"
#include "tr3d.cdk"

      character* 512 name
      integer k,istat,indo(G_nk+2)
      integer, dimension(:), pointer  :: ip1m
      real, dimension(:    ), pointer :: hybm,hybt,hybt_w
      real, dimension(:,:  ), pointer :: tr1
      real, dimension(:,:,:), pointer :: tr2
      type(vgrid_descriptor) :: vcoord
!
!------------------------------------------------------------------
!
      istat = gmm_get(gmmk_ut1_s ,ut1 )
      istat = gmm_get(gmmk_vt1_s ,vt1 )
      istat = gmm_get(gmmk_zdt1_s,zdt1)
      istat = gmm_get(gmmk_tt1_s ,tt1 )
      istat = gmm_get(gmmk_st1_s ,st1 )
      istat = gmm_get(gmmk_fis0_s,fis0)
      istat = gmm_get(gmmk_wt1_s ,wt1 )
      if (.not.Schm_hydro_L) istat = gmm_get(gmmk_qt1_s,qt1)

      do k=1,G_nk+2
         indo(k) = k
      end do
      nullify (ip1m,hybm,hybt,hybt_w)
      istat = vgrid_wb_get('ref-m',vcoord,ip1m)
      deallocate(ip1m); nullify(ip1m)
      istat = vgd_get (vcoord,'VCDM - vertical coordinate (m)',hybm)
      istat = vgd_get (vcoord,'VCDT - vertical coordinate (t)',hybt)
      allocate(tr1(l_minx:l_maxx,l_miny:l_maxy))

      Out_reduc_l= .true.

      call out_href3 ( 'Mass_point', Grdc_gid, Grdc_gif, 1,&
                                     Grdc_gjd, Grdc_gjf, 1 )
      call out_vref  ( etiket=Out3_etik_S )

      call out_fstecr3 ( tt1 ,l_minx,l_maxx,l_miny,l_maxy,hybt,'TT1' ,1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. )
      call out_fstecr3 ( wt1 ,l_minx,l_maxx,l_miny,l_maxy,hybt,'WT1' ,1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. )
      call out_fstecr3 ( zdt1,l_minx,l_maxx,l_miny,l_maxy,hybt,'ZDT1',1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. )
      if (.not.Schm_hydro_L) &
      call out_fstecr3 ( qt1 ,l_minx,l_maxx,l_miny,l_maxy,hybm,'QT1' ,1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. ) 
      call out_fstecr3 ( st1 ,l_minx,l_maxx,l_miny,l_maxy,0.  ,'ST1 ',1., &
                         0.,5,-1,1,indo,1,32,.false. )
      call out_fstecr3 ( fis0,l_minx,l_maxx,l_miny,l_maxy,0.  ,'FIS0',1., &
                         0.,5,-1,1,indo,1,32,.false. )
      if ( out3_sfcdiag_L ) then
         tr1 = tt1(1:l_ni,1:l_nj,G_nk)
         call itf_phy_sfcdiag(tr1(l_minx,l_miny),&
              l_minx,l_maxx,l_miny,l_maxy,'PW_TT:P',istat,.false.)
         call out_fstecr3 ( tr1 ,l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2),'TT1' ,1., &
                            0.,4,-1,1,indo,1,32,.false. )
      endif

      do k=1,Grdc_ntr
         nullify (tr2)
         name = 'TR/'//trim(Grdc_trnm_S(k))//':P'
         istat= gmm_get (name,tr2)
         call out_fstecr3 ( tr2 ,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                            Grdc_trnm_S(k),1.,0.,5,-1,G_nk,indo,G_nk,32,.false.)
         if ( out3_sfcdiag_L ) then
            tr1 = tr2(1:l_ni,1:l_nj,G_nk)
            call itf_phy_sfcdiag ( tr1(l_minx,l_miny),l_minx,l_maxx, &
                                   l_miny,l_maxy,name,istat,.true. )
            call out_fstecr3 ( tr1 ,l_minx,l_maxx,l_miny,l_maxy, &
                               hybt(G_nk+2),Grdc_trnm_S(k),1.  , &
                               0.,4,-1,1,indo,1,32,.false. )
         endif
      end do
     
      call out_href3 ( 'U_point', Grdc_gid, Grdc_gif, 1,&
                                  Grdc_gjd, Grdc_gjf, 1 )
      call out_fstecr3 ( ut1 ,l_minx,l_maxx,l_miny,l_maxy,hybm,'URT1' ,1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. ) 
      if ( out3_sfcdiag_L ) then
         tr1 = ut1(1:l_ni,1:l_nj,G_nk)
         call itf_phy_sfcdiag(tr1(l_minx,l_miny),&
              l_minx,l_maxx,l_miny,l_maxy,'PW_UU:P',istat,.false.)
         call out_fstecr3 ( tr1 ,l_minx,l_maxx,l_miny,l_maxy,hybm(G_nk+2), &
                            'URT1' ,1., 0.,4,-1,1,indo,1,32,.false. ) 
      endif
      call out_href3 ( 'V_point', Grdc_gid, Grdc_gif, 1,&
                                  Grdc_gjd, Grdc_gjf, 1 )
      call out_fstecr3 ( vt1 ,l_minx,l_maxx,l_miny,l_maxy,hybm,'VRT1' ,1., &
                         0.,5,-1,G_nk,indo,G_nk,32,.false. )
      if ( out3_sfcdiag_L ) then
         tr1 = vt1(1:l_ni,1:l_nj,G_nk)
         call itf_phy_sfcdiag(tr1(l_minx,l_miny),&
              l_minx,l_maxx,l_miny,l_maxy,'PW_VV:P',istat,.false.)
         call out_fstecr3 ( tr1 ,l_minx,l_maxx,l_miny,l_maxy,hybm(G_nk+2), &
                            'VRT1' ,1., 0.,4,-1,1,indo,1,32,.false. ) 
      endif

      deallocate (hybm,hybt,tr1)
!
!------------------------------------------------------------------
!
      return
      end

