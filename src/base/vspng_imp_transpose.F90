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

!**s/r vspng_imp_transpose - Establish layout for the 2 level-transpose 
!                            used in the vertical sponge

      integer function vspng_imp_transpose2 &
                        ( F_npex, F_npey, F_checkparti_L )
      implicit none
#include <arch_specific.hf>

      logical F_checkparti_L
      integer F_npex, F_npey

#include "ldnh.cdk"
#include "lun.cdk"
#include "trp.cdk"

      logical, external :: decomp3
      integer,parameter :: lowest = 2
      integer minx, maxx, n, npartiel, n0, ierr
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)
      vspng_imp_transpose2= 0

!                  Vspng_n_spng top layers NOT distributed
!                  ldnh_maxy distributed on F_npex PEs
!                  G_ni NOT distributed

      if (Lun_out.gt.0) write(Lun_out,1002) ' Transpose 1===>2 for VSPNG (no halo):', &
                ' ldnh_maxy distributed on F_npex PEs', ldnh_maxy,F_npex

      if (.not. decomp3 (ldnh_maxy, minx, maxx, n, npartiel, 0, n0, &
                  .true. , .true., F_npex, lowest, F_checkparti_L, 0 )) &
      vspng_imp_transpose2= -1

      trp_12emin = 1
      trp_12emax = maxx
      trp_12en   = n
      trp_12en0  = n0

!                  Vspng_n_spng top layers NOT distributed
!                  ldnh_maxx distributed on F_npey PEs
!                  G_nj NOT distributed
!
      if (Lun_out.gt.0) write(Lun_out,1002) ' Transpose 2===>2 for VSPNG (no halo):', &
                  ' ldnh_maxx distributed on F_npey', ldnh_maxx,F_npey

      if (.not. decomp3 (ldnh_maxx, minx, maxx, n, npartiel, 0, n0, &
                  .false. , .true., F_npey, lowest, F_checkparti_L, 0 )) &
      vspng_imp_transpose2= -1

      trp_22emin = 1
      trp_22emax = maxx
      trp_22en   = n
      trp_22en0  = n0

      if (vspng_imp_transpose2.lt.0) then
         if  (Lun_out.gt.0) &
         write(lun_out,*) 'VSPNG_IMP_TRANSPOSE: ILLEGAL DOMAIN PARTITIONING'
      endif

 1000 format (/' VSPNG_IMP_TRANSPOSE: checking VSPNG dimension partitionning:')
 1002 format (a/a45,i6,' /',i5)
!     ---------------------------------------------------------------
!
      return
      end

