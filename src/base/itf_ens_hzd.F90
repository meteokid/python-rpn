!COMP_ARCH=intel13sp1u2 ; -suppress=-C
!COMP_ARCH=intel-2016.1.156; -suppress=-C

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


      subroutine itf_ens_hzd ( F_ut1, F_vt1, F_tt1, Minx,Maxx,Miny,Maxy, Nk )
      use phy_itf, only: phy_get
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy, Nk
      real    F_ut1(Minx:Maxx,Miny:Maxy,Nk), &
              F_vt1(Minx:Maxx,Miny:Maxy,Nk), &
              F_tt1(Minx:Maxx,Miny:Maxy,Nk)

!author  Lubos Spacek - February 2010
!     
!revision
! v4_12 - Spacek L.            - initial version 
! v4_20 - Spacek L./Gagnon N.  - Modify the logic of the main switches (Ens_conf, 
!                                Ens_skeb_conf and Ens_ptp_conf) to not impact 
!                                non-ensemble users.

#include <rmnlib_basics.hf>
#include "gmm.hf"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "dcst.cdk"
#include "cstv.cdk"
#include "ens_param.cdk"
#include "ens_gmm_var.cdk"
#include "grd.cdk"

      character*4, save :: mode= "SAVE"
      integer i,j,k,istat
      real dummy
      real, dimension(:,:,:), pointer :: ptr3d
!     _________________________________________________________________

      if (.not.Ens_conf) return

! Case I: ens_skeb_conf = .true.
!
      if (ens_skeb_conf) then

         call stop_mpi(-1,'itf_ens_hzd','itf_ens_hzd needs updating')

         istat = gmm_get(gmmk_difut1_s,difut1)
         istat = gmm_get(gmmk_difvt1_s,difvt1)
         istat = gmm_get(gmmk_diout1_s,diout1)
         istat = gmm_get(gmmk_diovt1_s,diovt1)
         istat = gmm_get(gmmk_ugwdt1_s,ugwdt1)
         istat = gmm_get(gmmk_vgwdt1_s,vgwdt1)

         if (mode == "SAVE") then
            difut1 = F_ut1
            difvt1 = F_vt1
            goto 999
         endif

         if (mode == "CENS") then
            if(Ens_skeb_gwd)then
               ptr3d => ugwdt1(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,:)
               istat = phy_get(ptr3d,'phytd_ugwd',F_npath='V',F_bpath='V',F_end=(/-1,-1,l_nk/))
               if (.not.RMN_IS_OK(istat))write(*,6000)'ugwdt1-s'
               ptr3d => vgwdt1(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,:)
               istat = phy_get(ptr3d,'phytd_vgwd',F_npath='V',F_bpath='V',F_end=(/-1,-1,l_nk/))
               if (.not.RMN_IS_OK(istat))write(*,6000)'vgwdt1-s'
!!$               call itf_phy_uvgridscal (ugwdt1,vgwdt1,l_minx,l_maxx,l_miny,l_maxy,l_nk,.false.)
            endif

            difut1 = F_ut1 - difut1
            difvt1 = F_vt1 - difvt1

            diout1 = difut1           
            diovt1 = difvt1           


!NG         Multiplication of gravity wave drag tendencies by the time step to convert to wind units as in GEM4.6
            ugwdt1 = ugwdt1 * Cstv_dt_8
            vgwdt1 = vgwdt1 * Cstv_dt_8

            call ens_filter (ugwdt1,vgwdt1,difut1,difvt1, F_ut1,F_vt1,F_tt1,&
                             l_minx,l_maxx,l_miny,l_maxy, Nk)
         endif
!
! Case II: ens_skeb_conf = .false. however, in the same time
!          ens_ptp_conf  = .true.
!
      elseif (ens_ptp_conf .and. .not.ens_skeb_conf )then

         if (mode == "SAVE") then
            goto 999
         elseif (mode == "CENS") then
            call ens_filter (dummy,dummy,dummy,dummy, F_ut1,F_vt1,F_tt1, &
                             l_minx,l_maxx,l_miny,l_maxy, Nk)
         endif

      endif
!
! Case III: default (no action)
!
 999  if (mode == "SAVE") then
         mode = "CENS"
      else
         mode = "SAVE"
      endif
!     _________________________________________________________________
!
 6000 format('itf_ens_hzd at gmm_get(',A,')')

      return
      end
