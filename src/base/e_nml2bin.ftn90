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

!**s/r e_nml2bin - writes parameters to labfl
!
      subroutine e_nml2bin
      implicit none
#include <arch_specific.hf>

      integer l_ni, l_nj
      parameter (l_ni=1, l_nj=1)

!author michel desgagne - Jan 2000
!
!revision
! v2_20 - Pellerin P.       - modified for physics 3.66
! v2_30 - A. Methot         - introduction of a new stretch grid design
! v2_30                       with upper limits on grid point spacing
! v2_30 - Corbeil L.        - removed the part for pres_surf and top
! v2_31 - Desgagne M.       - added ptopo.cdk and tracers
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_11 - Tanguay M.        - Introduce Grd_gauss_L
! v3_12 - Winger K.         - transfer Anal_cond
! v3_21 - Lemonsu A.        - add P_pbl_schurb_s
! v3_22 - Lee V.            - removed Trvs tracers
! v3_30 - Lee/Desgagne      - minimized parameters passed in labfl.bin
! v4_03 - Spacek/Desgagne   - ISST
! v4_04 - Plante A.         - Remove offline mode
! v4_14 - Dugas B.          - add leap year control option

#include "tr3d.cdk"
#include "e_anal.cdk"
#include "ptopo.cdk"
#include "path.cdk"
#include "modconst.cdk"
#include "step.cdk"
#include <clib_interface_mu.hf>
#include "modipc.cdk"

      integer, external :: fnom
      integer labfl,err
      character*1024 rootfn
!
!     ---------------------------------------------------------------
!
      err = 0

      if (Ptopo_myproc.eq.0) then
         Mod_runstrt_S  = Step_runstrt_S
         horzint_L      = (Anal_hav(1).ne.11)
         labfl=0
         rootfn = trim(Path_output_S)//'/GEMDM_input/'
         err    = clib_mkdir (trim(rootfn))
         err    = fnom(labfl,trim(rootfn)//'/Modntr_ipc.nml','SEQ',0)
         if (err.ge.0) then
            write (labfl  ,nml=modipc_cfgs)
            close (labfl)
         endif
      endif

      call handle_error(err,'e_nm2bin','Trying to open file labfl.bin')
!
!     ---------------------------------------------------------------
!
      return
      end
!
