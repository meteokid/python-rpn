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

!**s/r set_transpose - Establish layout for the 2 level-transpose used
!                      in the solver and the horizontal diffusion
!
      subroutine hzd_imp_transpose ( F_npex, F_npey, F_checkparti_L )
      implicit none
#include <arch_specific.hf>
!
      logical F_checkparti_L
      integer F_npex, F_npey
!author
!     M. Desgagne - fall 2013
!
!revision
! v4_70 - Desgagne M.       - initial version
!

#include "glb_ld.cdk"
#include "lun.cdk"
#include "trp.cdk"

      logical, external :: decomp3
      integer,parameter :: lowest = 2
      integer minx, maxx, n, npartiel, n0, n1, ierr
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)
      ierr=0

! Transpose 1===>2:G_nk distributed on F_npex PEs (no halo)
!                  G_nj distributed on F_npey PEs (original layout)
!                  G_ni NOT distributed
!         initial layout  : (l_minx:l_maxx,    l_miny:l_maxy    ,G_nk)
!         transpose layout: (l_miny:l_maxy,trp_12dmin:trp_12dmax,G_ni)

      if (Lun_out.gt.0) write(Lun_out,1002) ' Transpose 1===>2 for HZD (no halo):', &
                  ' G_nk   distributed on F_npex PEs', G_nk,F_npex

      if (.not. decomp3 (G_nk, minx, maxx, n, npartiel, 0, n0, &
                  .true. , .true., F_npex, lowest, F_checkparti_L, 0 )) ierr=-1

      trp_12dmin = minx ! most likely = 1 since no halo
      trp_12dmax = maxx
      trp_12dn   = n
      trp_12dn0  = n0
      
! Transpose 1===>2:G_nk distributed on F_npex PEs (no halo)
!                  G_nj distributed on F_npey PEs (original layout)
!                  G_ni NOT distributed
!         initial layout  : (l_minx:l_maxx,     l_miny:l_maxy     ,G_nk)
!         transpose layout: (l_miny:l_maxy,trp_p12dmin:trp_p12dmax,G_ni)

      if (Lun_out.gt.0) write(Lun_out,1002) ' Transpose 1===>2 for HZD (no halo):', &
                  ' G_nk distributed on F_npex PEs', G_nk,F_npex

      if (.not. decomp3 (G_nk, minx, maxx, n, npartiel, 0, n0, &
                  .true. , .true., F_npex, lowest, F_checkparti_L, 0 )) ierr=-1

      trp_p12dmin = minx ! most likely = 1 since no halo
      trp_p12dmax = maxx
      trp_p12dn   = n
      trp_p12dn0  = n0

! Transpose 2===>2:G_nk distributed on F_npex PEs (no halo)
!                  G_nj NOT distributed
!                  G_ni distributed on F_npey PEs (no Halo)
!  initial layout  : (    l_miny:l_maxy    ,trp_12smin:trp_12smax,G_ni)
!  transpose layout: (trp_12smin:trp_12smax,trp_22min :trp_22max ,G_nj)

      if (trp_22n.lt.0) then

         if (Lun_out.gt.0) write(Lun_out,1002) ' Transpose 2===>2 for SOLVER (no halo):', &
                     ' G_ni distributed on F_npey PEs', G_ni,F_npey

         if (.not. decomp3 (G_ni, minx, maxx, n, npartiel, 0, n0, &
                     .false., .true., F_npey, lowest, F_checkparti_L, 0 )) ierr=-1

         trp_22min = minx ! most likely = 1 since no halo
         trp_22max = maxx
         trp_22n   = n
         trp_22n0  = n0
      endif
      
      if  (Lun_out.gt.0) then
         if (ierr.lt.0)  then
            write(lun_out,*) 'HZD_IMP_TRANSPOSE: ILLEGAL DOMAIN PARTITIONING'
         else
            write(lun_out,*) 'HZD_IMP_TRANSPOSE: PARTITIONING is OK'
         endif
      endif

      if (.not.F_checkparti_L) &
      call handle_error(ierr,'HZD_IMP_TRANSPOSE','ILLEGAL DOMAIN PARTITIONING -- ABORTING')

 1000 format (/' HZD_IMP_TRANSPOSE: checking HZD dimension partitionning:')
 1002 format (a/a45,i6,' /',i5)
!     ---------------------------------------------------------------
!
      return
      end

