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

      subroutine out_dyn ( F_reg_out, F_casc_L )
      use out_vref_mod, only: out_vref
      implicit none
#include <arch_specific.hf>

      logical F_reg_out, F_casc_L

!author
!     V. Lee    - rpn - July  2004 (from dynout2 v3_12)
!
!revision
! v4_80 - Desgagne M.       - major re-factorization of output

#include "out3.cdk"
#include "out.cdk"
#include "lun.cdk"
#include "init.cdk"
#include "grid.cdk"
#include "level.cdk"
#include "outd.cdk"
#include "grdc.cdk"
#include "lctl.cdk"
#include "out_listes.cdk"
#include "step.cdk"
#include <rmnlib_basics.hf>

      character*15 prefix
      logical ontimec,flag_clos
      integer k,kk,jj,levset,gridset,istat
!
!----------------------------------------------------------------------
!
      call timing_start2 ( 80, 'OUT_DYN', 1)
      if (.not.Lun_debug_L) istat= fstopc('MSGLVL','SYSTEM',.false.)

      Out_type_S   = 'REGDYN'
!
!########## REGULAR OUTPUT #######################################
!
      if (F_reg_out) then

         if (outd_sorties(0,Lctl_step).lt.1) then
            if (Lun_out.gt.0) write(Lun_out,7002) Lctl_step
            goto 887
         endif

         if (Lun_out.gt.0) write(Lun_out,7001) &
                           Lctl_step,trim(Out_laststep_S)

         do jj=1, outd_sorties(0,Lctl_step)

            kk       = outd_sorties(jj,Lctl_step)
            gridset  = Outd_grid(kk)
            levset   = Outd_lev(kk)

            Out_prefix_S(1:1) = 'd'
            Out_prefix_S(2:2) = Level_typ_S(levset)
            call up2low (Out_prefix_S ,prefix)
            Out_reduc_l       = Grid_reduc(gridset)

            call out_open_file (trim(prefix))

            call out_href3 ( 'Mass_point'                , &
                  Grid_x0 (gridset), Grid_x1 (gridset), 1, &
                  Grid_y0 (gridset), Grid_y1 (gridset), 1 )

            if (Level_typ_S(levset).eq.'M') then
               call out_vref (etiket=Out_etik_S)
            elseif (Level_typ_S(levset).eq.'P') then
               call out_vref (Level_allpres(1:Level_npres),etiket=Out_etik_S)
            endif

            call out_tracer (levset, kk)

            call out_thm    (levset, kk)

            call out_uv     (levset, kk)

            call out_dq     (levset, kk)

            call out_gmm2   (levset, kk)

            flag_clos= .true.
            if (jj .lt. outd_sorties(0,Lctl_step)) then
              flag_clos= .not.( (gridset.eq.Outd_grid(outd_sorties(jj+1,Lctl_step))).and. &
              (Level_typ_S(levset).eq.Level_typ_S(Outd_lev(outd_sorties(jj+1,Lctl_step)))))
            endif

            if (flag_clos) call out_cfile3

         end do

      endif

!#################################################################
!
 887  continue
!
!########## SPECIAL OUTPUT FOR CASCADE ###########################

      if ((F_casc_L) .and. (Grdc_ndt.gt.0)) then

         ontimec = .false.
         if ( Lctl_step.ge.Grdc_start.and.Lctl_step.le.Grdc_end) &
              ontimec = (mod(Lctl_step+Grdc_start,Grdc_ndt).eq.0)

         if ( Init_mode_L .and. (Step_kount.ge.Init_halfspan) ) &
              ontimec = .false.

         if ( ontimec ) then

            call out_open_file ('casc')

            call out_dyn_casc

            call out_cfile3

         endif

         ontimec = .false.

      endif

      istat = fstopc('MSGLVL','INFORM',.false.)
      call timing_stop ( 80 )

 7001 format(/,' OUT_DYN- WRITING DYNAMIC OUTPUT FOR STEP (',I8,') in directory: ',a)
 7002 format(/,' OUT_DYN- NO DYNAMIC OUTPUT FOR STEP (',I8,')')
 8001 format(/,' OUT_DYN- WRITING CASCADE OUTPUT FOR STEP (',I8,') in directory: ',a)
!
!--------------------------------------------------------------------
!
      return
      end

