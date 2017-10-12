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

!**s/r gem_nml

      integer function gem_nml (F_namelistf_S)
      use gem_options
      use grid_options
      use grdc_options
      use lun
      implicit none
#include <arch_specific.hf>
#include <clib_interface_mu.hf>

      character(len=*) F_namelistf_S


      integer, external :: fnom, canonical_nml
      character(len=64) :: dumc_S
      logical dum_L
      integer err,unf
!
!-------------------------------------------------------------------
!
      gem_nml = -1

      if ((F_namelistf_S == 'print').or.(F_namelistf_S == 'PRINT')) then
         gem_nml = 0
         if ( Lun_out >= 0) then
            write (Lun_out,nml=gem_cfgs_p)
            if (Grdc_ndt > 0) write (lun_out,nml=grdc_p)
            err= canonical_nml ('print', Lun_out, dum_L, dum_L)
         endif
         return
      endif

      if (F_namelistf_S /= '') then

         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', 0) /= 0) goto 9110
         rewind(unf)
         read (unf, nml=gem_cfgs, end = 9120, err=9120)
         rewind(unf)
         read (unf, nml=grdc,     end = 1000, err=9130)
 1000    call fclos (unf)

      endif

      call low2up (Lctl_rxstat_S ,dumc_S)
      Lctl_rxstat_S = dumc_S
      err = clib_toupper(Schm_phycpl_S)

      err= canonical_nml (F_namelistf_S, Lun_out, Schm_canonical_dcmip_L,&
                                             Schm_canonical_williamson_L )

      if (err == 1) gem_nml= 1

      goto 9999

 9110 if (Lun_out > 0) then
         write (Lun_out, 9050) trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9120 call fclos (unf)
      if (Lun_out >= 0) then
         write (Lun_out, 9150) 'gem_cfgs',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9130 call fclos (unf)
      if (Lun_out >= 0) then
         write (Lun_out, 9150) 'grdc',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif

 8000 format (/,'========= ABORT IN S/R gem_nml.f ============='/)
 9050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 9150 format (/,' NAMELIST ',A,' INVALID IN FILE: ',A/)
!
!-------------------------------------------------------------------
!
 9999 return
      end
