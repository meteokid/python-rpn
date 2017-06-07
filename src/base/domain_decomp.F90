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
      use gem_options
      use grid_options
      implicit none
#include <arch_specific.hf>

      logical F_checkparti_L
      integer F_npex, F_npey

#include "glb_ld.cdk"
#include "lun.cdk"

      logical, external :: decomp3
      integer err, g_err, status

!-------------------------------------------------------------------
!
      err=-1
      if (Lun_out.gt.0) write (Lun_out,1000) G_ni,F_npex,G_nj,F_npey
      
      if (decomp3 (G_ni,l_minx,l_maxx,l_ni,G_lnimax,G_halox,l_i0,.true. ,.true.,&
                   F_npex, (Grd_extension+1), F_checkparti_L, 0 ) .and.         &
          decomp3 (G_nj,l_miny,l_maxy,l_nj,G_lnjmax,G_haloy,l_j0,.false.,.true.,&
                   F_npey, (Grd_extension+1), F_checkparti_L, 0 ))              &
          err=0

      call rpn_comm_Allreduce (err,g_err,1,"MPI_INTEGER","MPI_MIN","GRID",status)
      domain_decomp3= g_err

      if (domain_decomp3.lt.0)  then
         if  (Lun_out.gt.0) &
         write(lun_out,*) 'DECOMP: ILLEGAL DOMAIN PARTITIONING'
         return
      endif

      l_nk = G_nk
      l_njv= l_nj
      l_niu= l_ni
      if (l_north) l_njv= l_nj - 1
      if (l_east ) l_niu= l_ni - 1

      if (.not.F_checkparti_L) call glbpos   

 1000 format (/' DOMAIN_DECOMP: checking partitionning of G_ni and G_nj'/&
               2(i6,' in ',i6,' subdomains',5x)/)
!
!-------------------------------------------------------------------
!
      return
      end


