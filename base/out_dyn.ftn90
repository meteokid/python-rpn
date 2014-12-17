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

!**s/r out_dyn - perform dynamic output
!
      subroutine out_dyn ( F_reg_out, F_casc_L )
      use out_vref_mod, only: out_vref
      implicit none
#include <arch_specific.hf>

      logical F_reg_out, F_casc_L

!author
!     V. Lee    - rpn - July  2004 (from dynout2 v3_12)
!
!revision
! v3_20 - Lee V.            - Initial MPI version (from dynout2 v3_11)

! v3_30 - Milbrandt J.      - Added extra hydrometeor variables (snow and hail)
!                             for Milbrandt-Yau scheme for water-loading
! v3_30 - McTaggart-Cowan R.- Allow for user-defined domain tag extensions
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
!                             QR to QL(rain),QG to QJ(graupel),
!                             QC to QB(cloud;Kong-Yau and Milbrandt-Yau only)
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_03 - Lee V.            - modification of Out_etik_S in out_sgrid only
! v4_04 - Tanguay M.        - Staggered version TL/AD
! v4_05 - Lepine/Lee        - VMM replacement with GMM
! v4_40 - Lee V.            - change in argument call for this routine, prgen
!                             in order to select the "bot" levels
! v4_40 - Tanguay M.        - Revision TL/AD
!
!object
!     Subroutine to control the production of
!     the output of the dynamic variables

#include "gmm.hf"
#include "glb_ld.cdk"
#include "out3.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "init.cdk"
#include "vt1.cdk"
#include "grid.cdk"
#include "level.cdk"
#include "outd.cdk"
#include "tr3d.cdk"
#include "grdc.cdk"
#include "lctl.cdk"
#include "cstv.cdk"
#include "out.cdk"
#include "pw.cdk"
#include "out_listes.cdk"
#include "ver.cdk"

      character*15 datev,pdate
      logical periodx_L,ontimec,flag_clos
      integer i,j,k,istat,jj,kk,levset,gridset
      real wlnpi_m(l_minx:l_maxx,l_miny:l_maxy,G_nk+1), &
           wlnpi_t(l_minx:l_maxx,l_miny:l_maxy,G_nk+1)
      real*8 dayfrac,sec_in_day
      parameter ( sec_in_day=86400.0d0 )
!
!----------------------------------------------------------------------
!
      call timing_start ( 80, 'OUT_DYN')
!
!########## REGULAR OUTPUT #######################################
!
      if (F_reg_out) then

         if (outd_sorties(0,Lctl_step).lt.1) then
            if (Lun_out.gt.0) write(Lun_out,7002) Lctl_step
            goto 887
         endif

         if (Lun_out.gt.0) write(Lun_out,7001) Lctl_step,trim(Out_laststep_S)

         call diag_output

         out3_type_S= 'REGDYN'

         istat= gmm_get(gmmk_pw_pm_plus_s ,pw_pm_plus)
         istat= gmm_get(gmmk_pw_pt_plus_s ,pw_pt_plus)
         istat= gmm_get(gmmk_pw_p0_plus_s ,pw_p0_plus)
         istat= gmm_get(gmmk_st1_s        ,st1       )

         do k=1,G_nk
            wlnpi_m(1:l_ni,1:l_nj,k)= log(pw_pm_plus(1:l_ni,1:l_nj,k))
            wlnpi_t(1:l_ni,1:l_nj,k)= log(pw_pt_plus(1:l_ni,1:l_nj,k))
         enddo
         wlnpi_m(1:l_ni,1:l_nj,G_nk+1)= log(pw_p0_plus(1:l_ni,1:l_nj))
         wlnpi_t(1:l_ni,1:l_nj,G_nk+1)= log(pw_p0_plus(1:l_ni,1:l_nj))

         call out_padbuf (wlnpi_m,l_minx,l_maxx,l_miny,l_maxy,G_nk+1)
         call out_padbuf (wlnpi_t,l_minx,l_maxx,l_miny,l_maxy,G_nk+1)

         Out_ip3 = 0
         if (Out3_ip3.eq.-1) Out_ip3 = Lctl_step
         if (Out3_ip3.gt.0 ) Out_ip3 = Out3_ip3

         do jj=1, outd_sorties(0,Lctl_step)

            kk       = outd_sorties(jj,Lctl_step)
            gridset  = Outd_grid(kk)
            levset   = Outd_lev(kk)
            periodx_L= .not.G_lam .and. ((Grid_x1(gridset)-Grid_x0(gridset)+1).eq.G_ni)

            call out_sgrid2 ( Grid_x0 (gridset),Grid_x1 (gridset), &
                              Grid_y0 (gridset),Grid_y1 (gridset), &
                              Grid_ig1(gridset),Grid_ig2(gridset), &
                              periodx_L, Grid_stride(gridset)    , &
                              Grid_etikext_s(gridset) )

            Out_prefix_S(1:1) = 'd'
            Out_prefix_S(2:2) = Level_typ_S(levset)
         
            call out_sfile2 ( Out3_closestep,Lctl_step )
            
            call out_href2  ( 'Mass_point' )

            if (Level_typ_S(levset).eq.'M') then
               call out_vref (etiket=Out_etik_S)
            elseif (Level_typ_S(levset).eq.'P') then
               call out_vref (Level_allpres(1:Level_npres),etiket=Out_etik_S)
            endif

            !     output of 3-D tracers
            call out_tracer (wlnpi_t,l_minx,l_maxx,l_miny,l_maxy,G_nk,levset,kk)

            !     output of temperature, humidity and mass fields,omega

            call out_thm (wlnpi_m,wlnpi_t,st1,l_minx,l_maxx,l_miny,l_maxy,G_nk,levset,kk)

            !     output of winds
            call out_uv (wlnpi_m,l_minx,l_maxx,l_miny,l_maxy,G_nk,levset,kk)

            !     output of divergence and vorticity
            call out_dq (wlnpi_m,l_minx,l_maxx,l_miny,l_maxy,G_nk,levset,kk)

            !     output of gmm
            call out_gmm2 (levset,kk)

            flag_clos= .true.
            if (jj .lt. outd_sorties(0,Lctl_step)) then
              flag_clos= .not.( (gridset.eq.Outd_grid(outd_sorties(jj+1,Lctl_step))).and. &
              (Level_typ_S(levset).eq.Level_typ_S(Outd_lev(outd_sorties(jj+1,Lctl_step)))))
            endif

            if (flag_clos) call out_cfile2

         end do

      endif
!
!#################################################################
!
 887  continue
!
!########## SPECIAL OUTPUT FOR CASCADE ###########################
!
      if ((F_casc_L) .and. (Grdc_ndt.gt.0)) then

         ontimec = .false.
         if ( Lctl_step.ge.Grdc_start.and.Lctl_step.le.Grdc_end) &
              ontimec = (mod(Lctl_step+Grdc_start,Grdc_ndt).eq.0)

         if ( Init_mode_L .and. (Lctl_step.ge.Init_halfspan) ) &
              ontimec = .false.

         if ( ontimec ) then

            out3_type_S= 'CASDYN'

            if (Lun_out.gt.0) write(Lun_out,8001) Lctl_step,trim(Out_laststep_S)
            call out_sgrid2 (Grdc_gid,Grdc_gif,Grdc_gjd,Grdc_gjf, &
                                                0,0,.false.,1,'')
            call datf2p (pdate,Out3_date)
            dayfrac = dble(Lctl_step) * Cstv_dt_8 / sec_in_day
            call incdatsd (datev,pdate,dayfrac)

            call out_dyn_3df_2 (datev)

         endif

         ontimec = .false.

      endif

      call timing_stop ( 80 )

 7001 format(/,' OUT_DYN- WRITING DYNAMIC OUTPUT FOR STEP (',I8,') in directory: ',a)
 7002 format(/,' OUT_DYN- NO DYNAMIC OUTPUT FOR STEP (',I8,')')
 8001 format(/,' OUT_DYN- WRITING CASCADE OUTPUT FOR STEP (',I8,') in directory: ',a)
!
!--------------------------------------------------------------------
!
      return
      end

