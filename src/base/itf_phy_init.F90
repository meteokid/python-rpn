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

!**s/r itf_phy_init - Initializes physics parameterization package
!
      subroutine itf_phy_init
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_put,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      use phy_itf, only: phy_init
      implicit none
#include <arch_specific.hf>

!authors 
!     Desgagne, McTaggart-Cowan, Chamberland -- Spring 2014
!
!revision
! v4_70 - authors          - initial version

#include <WhiteBoard.hf>
#include "glb_ld.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "out3.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "ver.cdk"
#include "path.cdk"
#include "level.cdk"

      type(vgrid_descriptor) :: vcoord
      integer err,zuip,ztip
      integer, dimension(:), pointer :: ip1m
      real :: zu,zt
!
!     ---------------------------------------------------------------
!
      out3_sfcdiag_L= .false.
      if (.not.Schm_phyms_L) return

      if (Lun_out.gt.0) write(Lun_out,1000)

! We put mandatory variables in the WhiteBoard

      err= 0
      err= min(wb_put('itf_phy/VSTAG'       , .true.    ), err)
      err= min(wb_put('itf_phy/TLIFT'       , Schm_Tlift), err)
      
! Complete physics initialization (see phy_init for interface content)

      err= phy_init ( Path_phy_S, Step_CMCdate0, real(Cstv_dt_8), &
                      'model/Hgrid/lclphy','model/Hgrid/lclcore', &
                      'model/Hgrid/global','model/Hgrid/local'  , &
                                       G_nk+1, Ver_std_p_prof%m )

! Retrieve the heights of the diagnostic levels (thermodynamic
! and momentum) from the physics ( zero means NO diagnostic level)

      err= min(wb_get('phy/zu', zu), err)
      err= min(wb_get('phy/zt', zt), err)

      call gem_error ( err,'itf_phy_init','phy_init or WB_get' )

! Add the diagnostic heights to the vertical coordinate of the model

      err = VGD_OK
      if ((zu.gt.0.) .and. (zt.gt.0.) ) then
         nullify(ip1m)
         Level_kind_diag=4
         err = min(vgrid_wb_get('ref-m',vcoord,ip1m), err)
         call convip(zuip,zu,Level_kind_diag,+2,'',.true.)
         call convip(ztip,zt,Level_kind_diag,+2,'',.true.)
         err = min(vgd_put(vcoord,'DIPM - IP1 of diagnostic level (m)',zuip), err)
         err = min(vgd_put(vcoord,'DIPT - IP1 of diagnostic level (t)',ztip), err)
         out3_sfcdiag_L= .true.
      endif

      call gem_error ( err,'itf_phy_init','setting diagnostic level in vertical descriptor' )

!     ---------------------------------------------------------------
 1000 format(/,'INITIALIZATION OF PHYSICS PACKAGE (S/R itf_phy_init)', &
             /,'====================================================')
 9500 format(/,' PHYSICS NOT SUPPORTED FOR NOW IN THEORETICAL CASE')
!     ---------------------------------------------------------------
!
      return
      end
