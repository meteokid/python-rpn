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
      use phy_itf, only: phy_init, phymeta,phy_getmeta
      implicit none
#include <arch_specific.hf>

!authors 
!     Desgagne, McTaggart-Cowan, Chamberland -- Spring 2014
!
!revision
! v4_70 - authors          - initial version

#include <WhiteBoard.hf>
#include <clib_interface_mu.hf>
#include "glb_ld.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "out3.cdk"
#include "outp.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "ver.cdk"
#include "path.cdk"
#include "level.cdk"

      type(vgrid_descriptor) :: vcoord
      integer err,zuip,ztip
      integer, dimension(:), pointer :: ip1m
      real :: zu,zt

      integer,parameter :: NBUS = 3
      character(len=9) :: BUS_LIST_S(NBUS) = &
                  (/'Entry    ', 'Permanent', 'Volatile '/)

!For sorting the output
      integer istat
      character(len=32) :: varname_S,outname_S,bus0_S
      integer pnerror,i,k,j,ibus,multxmosaic
      type(phymeta) :: pmeta
!
!     ---------------------------------------------------------------
!
      Out3_sfcdiag_L= .false.
      if (.not.Schm_phyms_L) return

      if (Lun_out.gt.0) write(Lun_out,1000)

! We put mandatory variables in the WhiteBoard

      err= 0
      err= min(wb_put('itf_phy/VSTAG'       , .true.       ), err)
      err= min(wb_put('itf_phy/TLIFT'       , Schm_Tlift   ), err)
      err= min(wb_put('itf_phy/DYNOUT'      , Out3_accavg_L), err)
      
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

! Print table of variables requested for output

      if (Lun_out.gt.0) write(Lun_out,1001)
      multxmosaic = 0

      DO_BUS: do ibus = 1,NBUS
         bus0_S = BUS_LIST_S(ibus)
         if (Lun_out.gt.0)  then
            write(Lun_out,1006)
            write(Lun_out,1002) bus0_S
            write(Lun_out,1006)
            write(Lun_out,1003)
         endif
         do k=1, Outp_sets
            do j=1,Outp_var_max(k)
               istat = phy_getmeta(pmeta,Outp_varnm_S(j,k), &
                       F_npath='VO',F_bpath=bus0_S(1:1),F_quiet=.true.)
               if (istat <= 0) then
                  cycle
               endif
               varname_S = pmeta%vname
               outname_S = pmeta%oname
               istat = clib_toupper(varname_S)
               istat = clib_toupper(outname_S)
               Outp_var_S(j,k) = outname_S(1:4)
               multxmosaic = max(multxmosaic,pmeta%fmul*(pmeta%mosaic+1))
               if (Lun_out.gt.0) write(Lun_out,1007) &
                    outname_S(1:4),varname_S(1:16),Outp_nbit(j,k), &
                    Outp_filtpass(j,k),Outp_filtcoef(j,k), &
                    Level_typ_S(Outp_lev(k))
            enddo
         enddo
      enddo DO_BUS
!     maximum size of slices with one given field that is multiple+mosaic
      Outp_multxmosaic = multxmosaic+10
      if (Lun_out.gt.0)  write(Lun_out,1006)

! Add the diagnostic heights to the vertical coordinate of the model

      err = VGD_OK
      if ((zu.gt.0.) .and. (zt.gt.0.) ) then
         nullify(ip1m)
         Level_kind_diag=4
         err = min ( vgrid_wb_get('ref-m',vcoord,ip1m), err)
         deallocate(ip1m)
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
 1001 format(/'+',35('-'),'+',17('-'),'+',5('-'),'+'/'| PHYSICS VARIABLES REQUESTED FOR OUTPUT              |',5x,'|')
 1002 format('|',5X,a9,' Bus ',40x, '|')
 1003 format('|',1x,'OUTPUT',1x,'|',2x,'PHYSIC NAME ',2x,'|',2x,' BITS  |','FILTPASS|FILTCOEF| LEV |')
 1006 format('+',8('-'),'+',16('-'),'+',9('-'),'+',8('-'),'+',8('-'),'+',5('-'))
 1007 format('|',2x,a4,2x,'|',a16,'|',i5,'    |',i8,'|',f8.3,'|',a4,' |')
!     ---------------------------------------------------------------
!
      return
      end
