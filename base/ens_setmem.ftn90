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

!**s/r ens_setmem - initialize ensemble prevision system
!
      subroutine ens_setmem (l_minx,l_maxx,G_halox,l_ni,     &
                                 l_miny,l_maxy,G_haloy,l_nj,l_nk,Lun_out)
      implicit none
#include <arch_specific.hf>
!
      integer l_minx,l_maxx,G_halox,l_ni
      integer l_miny,l_maxy,G_haloy,l_nj,l_nk,Lun_out
!
!     author   
!     Lubos Spacek - February 2010
!     
!     revision
! v4_12 - Spacek L.        - Initial version
! v_4.1.3 - N. Gagnon      - Change name of most parameters in the NAMELIST
!

#include "gmm.hf"
#include "var_gmm.cdk"
#include "ens_gmm_dim.cdk"
#include "ens_gmm_var.cdk"
#include "ens_param.cdk"

      integer :: istat
!-------------------------------------------------------------------
!
      if (.not.Ens_conf) return

      if (Lun_out.gt.0) write(Lun_out,6010)

      call gmm_build_meta3D(meta3d_sh2,          &
                            1,l_ni,0,0,l_ni,     &
                            1,l_nj,0,0,l_nj,     &
                            1,l_nk,0,0,l_nk,     &
                            0,GMM_NULL_FLAGS)
      call gmm_build_meta2D(meta2d_anm,                                     &
                            1,Ens_mc3d_dim,0,0,Ens_mc3d_dim,                &
                            -Ens_mc3d_mzt,Ens_mc3d_mzt,0,0,2*Ens_mc3d_mzt+1,&
                            0,GMM_NULL_FLAGS)
      call gmm_build_meta2D(meta2d_znm,                       &
                            1,Ens_dim2_max,0,0,Ens_dim2_max,  &
                            1,Ens_mc2d_ncha,0,0,Ens_mc2d_ncha,&
                            0,GMM_NULL_FLAGS)
      call gmm_build_meta2D(meta2d_dum,                       &
                            1,36,0,0,36,                      &
                            1,MAX2DC+MAX3DC,0,0,MAX2DC+MAX3DC,&
                            0,GMM_NULL_FLAGS)
      gmmk_mcsph1_s= 'MCSPH1'
      gmmk_difut1_s= 'DIFUT1'
      gmmk_difvt1_s= 'DIFVT1'
      gmmk_ugwdt1_s= 'UGWDT1'
      gmmk_vgwdt1_s= 'VGWDT1'
      gmmk_ensdiv_s= 'ENSDIV'
      gmmk_ensvor_s= 'ENSVOR'
      gmmk_anm_s   = 'ANMENS'
      gmmk_znm_s   = 'ZNMENS'
      gmmk_dumdum_s= 'DUMDUM'

      istat = gmm_create(gmmk_mcsph1_s,mcsph1,meta3d_sh2,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'mcsph1'
      istat = gmm_create(gmmk_difut1_s,difut1,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'difut1'
      istat = gmm_create(gmmk_difvt1_s,difvt1,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'difvt1'
      istat = gmm_create(gmmk_ugwdt1_s,ugwdt1,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'ugwdt1'
      istat = gmm_create(gmmk_vgwdt1_s,vgwdt1,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'vgwdt1'
      istat = gmm_create(gmmk_ensdiv_s,ensdiv,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'ensdiv'
      istat = gmm_create(gmmk_ensvor_s,ensvor,meta3d_nk,GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'ensvor'
      istat = gmm_create(gmmk_anm_s   ,anm   ,meta2d_anm,GMM_FLAG_RSTR+GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'anm'
      istat = gmm_create(gmmk_znm_s   ,znm   ,meta2d_znm,GMM_FLAG_RSTR+GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'znm'
      istat = gmm_create(gmmk_dumdum_s,dumdum,meta2d_dum,GMM_FLAG_RSTR+GMM_FLAG_INAN)
      if (GMM_IS_ERROR(istat))write(*,6000)'dum'

 6000 format('ens_set_mem at gmm_create(',A,')')
 6010 format(/,'INITIALIZATION OF MEMORY FOR ENSEMBLES (S/R ENS_SETMEM)' &
             /(55('=')))
!
!-------------------------------------------------------------------
!
      return
      end
