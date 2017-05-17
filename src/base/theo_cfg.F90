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

!**s/r theo_cfg - reads parameters from namelist theo_cfgs
!
      subroutine theo_cfg
      use step_options
      use gem_options
      use grid_options
      implicit none
#include <arch_specific.hf>
#include "cstv.cdk"
#include "out3.cdk"
#include "theonml.cdk"
#include "cst_lis.cdk"
#include "path.cdk"

      integer cte_ok,istat
      logical set_dcst_8
      external set_dcst_8
      integer  fnom,gem_nml,mtn_cfg,bubble_cfg,adv_nml
      external fnom,gem_nml,mtn_cfg,bubble_cfg,adv_nml

      integer k, unf, status, err, nrec
!
!     ---------------------------------------------------------------
      cte_ok = 0
      if ( .not. set_dcst_8 ( Dcst_cpd_8,liste_S,cnbre, 0,1 ) ) &
           cte_ok=-1

      status = -1

      if ( Ptopo_npey .gt.1) then
           if (Lun_out.gt.0) write (Lun_out, 9240)
           goto 9999
      endif

      err = gem_nml ('')
      err = adv_nml (Path_nml_S)

      Theo_case_S    = 'xxx'
      Step_runstrt_S = '19980101.000000'
      Out3_etik_S    = 'THEOC'
      Out3_ip3       = -1
      Lctl_debug_L   = .false.

      unf = 0
      if (fnom (unf, trim(Path_nml_S), 'SEQ+OLD' , nrec) .ne. 0) goto 9110

      rewind(unf)
      read (unf, nml=theo_cfgs, end = 9000, err=9000)

      if (  Theo_case_S .eq. 'MTN_SCHAR' &
           .or. Theo_case_S .eq. 'MTN_SCHAR2' &
           .or. Theo_case_S .eq. 'MTN_PINTY' &
           .or. Theo_case_S .eq. 'MTN_PINTY2' &
           .or. Theo_case_S .eq. 'MTN_PINTYNL' &
           .or. Theo_case_S .eq. 'NOFLOW' ) then
         print *,'Theo_case_S=',Theo_case_S
         err = mtn_cfg (unf)
         print *,'after mtn_cfg err=',err
      else if (  Theo_case_S .eq. 'BUBBLE' ) then
         print *,'Theo_case_S=',Theo_case_S
         err = bubble_cfg (unf)
         print *,'after bubble_cfg err=',err
      else
         if (Lun_out.gt.0) then
            write (Lun_out, 9200) Theo_case_S
            write (Lun_out, 8000)
         endif
         err = -1
      endif
      call fclos (unf)
      if (err.lt.0) goto 9999


      if (Lun_out.gt.0) write (Lun_out, 7050) Theo_case_S
      status=1
      return 

 9110 if (Lun_out.gt.0) then 
         write (Lun_out, 9050)
         write (Lun_out, 8000)
      endif
      goto 9999

 9000 if (Lun_out.gt.0) then
         call fclos (unf)
         write (Lun_out, 9100)
         write (Lun_out, 8000)
      endif

 9999 continue
      call handle_error(-1,'theo_cfg','') !TODO: fix me
!
!     ---------------------------------------------------------------
 7050 format (/' THEORETICAL CASE IS: ',a/)
 8000 format (/,'========= ABORT IN S/R theo_cfg.f ============='/)
 9050 format (/,' FILE: model_settings NOT AVAILABLE'/)
 9100 format (/,' NAMELIST theo_cfgs ABSENT or INVALID FROM FILE: model_settings'/)
 9200 format (/,' Unsupported theoretical case: ',a/)
 9240 format (/,' For theoretical cases, number of PEs in y must be 1 '/)

      return
      end

