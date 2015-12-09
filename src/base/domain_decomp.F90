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

!**s/r domain_decomp

      integer function domain_decomp3 ( F_npex, F_npey, F_checkparti_L )
      implicit none
#include <arch_specific.hf>

      logical F_checkparti_L
      integer F_npex, F_npey

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "grd.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include "lam.cdk"

      logical, external :: decomp3
      integer err,ierr
!
!-------------------------------------------------------------------
!
!     Establishing data topology

      l_west  = (0            .eq. Ptopo_mycol)
      l_east  = (Ptopo_npex-1 .eq. Ptopo_mycol)
      l_south = (0            .eq. Ptopo_myrow)
      l_north = (Ptopo_npey-1 .eq. Ptopo_myrow)
      north   = 0
      south   = 0
      east    = 0
      west    = 0
      if (l_north) north = 1
      if (l_south) south = 1
      if (l_east ) east  = 1
      if (l_west ) west  = 1
      pil_w     = 0
      pil_n     = 0
      pil_e     = 0
      pil_s     = 0
      Lam_pil_w = 0
      Lam_pil_n = 0
      Lam_pil_e = 0
      Lam_pil_s = 0
      G_periodx = .true.
      G_periody = .false.

      if (G_lam) then
         if (l_west ) pil_w = Glb_pil_w
         if (l_north) pil_n = Glb_pil_n
         if (l_east ) pil_e = Glb_pil_e
         if (l_south) pil_s = Glb_pil_s
         G_periodx = .false.
         Lam_pil_w = Glb_pil_w
         Lam_pil_n = Glb_pil_n
         Lam_pil_e = Glb_pil_e
         Lam_pil_s = Glb_pil_s
      endif

      domain_decomp3= -1
      if (Lun_out.gt.0) write (Lun_out,1000) G_ni,F_npex,G_nj,F_npey

      if (decomp3 (G_ni,l_minx,l_maxx,l_ni,G_lnimax,G_halox,l_i0,.true. ,.true.,&
                   F_npex, (Grd_extension+1), F_checkparti_L, 0 ) .and.         &
          decomp3 (G_nj,l_miny,l_maxy,l_nj,G_lnjmax,G_haloy,l_j0,.false.,.true.,&
                   F_npey, (Grd_extension+1), F_checkparti_L, 0 ))              &
      domain_decomp3= 0

      if (domain_decomp3.lt.0)  then
         if  (Lun_out.gt.0) &
         write(lun_out,*) 'DECOMP: ILLEGAL DOMAIN PARTITIONING'
         return
      endif

      l_nk = G_nk
      l_njv= l_nj
      l_niu= l_ni
      if (l_north) l_njv= l_nj - 1
      if ((l_east).and.(G_lam)) l_niu = l_ni - 1

      if (.not.F_checkparti_L) then
         call glbpos   
      endif

 1000 format (/' DOMAIN_DECOMP: checking partitionning of G_ni and G_nj'/&
               2(i6,' in ',i6,' subdomains',5x)/)
!
!-------------------------------------------------------------------
!
      return
      end


