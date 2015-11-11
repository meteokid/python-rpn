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

!**s/r out_fstecr2

      subroutine out_fstecr2 ( fa,lminx,lmaxx,lminy,lmaxy,rf,nomvar,&
                               mul,add, kind,nkfa,ind_o,nk_o       ,&
                               nbit,F_empty_stk_L)
      implicit none
#include <arch_specific.hf>

      character* (*) nomvar
      logical F_empty_stk_L
      integer lminx,lmaxx,lminy,lmaxy,nkfa,nbit,nk_o,kind
      integer ind_o(nk_o)
      real fa (lminx:lmaxx,lminy:lmaxy,nkfa), rf(nkfa), mul,add

!author
!    Michel Desgagne - Fall 2012
!
!revision
! v4_50 - Desgagne M. - Initial version
! v4_80 - Desgagne M. - major re-factorization of output

#include "glb_ld.cdk"
#include "out.cdk"
#include "out3.cdk"
      include "out_meta.cdk"

Interface
subroutine out_stkecr2 ( fa,lminx,lmaxx,lminy,lmaxy,meta,nplans, &
                         g_id,g_if,g_jd,g_jf )
      include "out_meta.cdk"
      integer lminx,lmaxx,lminy,lmaxy,nplans
      integer g_id,g_if,g_jd,g_jf
      real fa(lminx:lmaxx,lminy:lmaxy,nplans)
      type (meta_fstecr), dimension(:), pointer :: meta
End Subroutine out_stkecr2
End Interface

      character*8 dumc
      logical, save :: done= .false.
      integer modeip1,i,j,k
      integer max_stack
      integer, save :: istk = 0
      real, save, dimension (:,:,:), pointer :: f2c => null()
      type (meta_fstecr), save, dimension(:), pointer :: meta => null()
!
!----------------------------------------------------------------------
!
      max_stack= Out3_npes
      if (.not.associated(meta)) allocate (meta(max_stack))
      if (.not.associated(f2c )) &
         allocate (f2c(l_minx:l_maxx,l_miny:l_maxy,max_stack))
      if (.not.done) then
         out_stk_full= 0 ; out_stk_cnt= 0 ; done= .true.
      endif

      if (F_empty_stk_L) then
         if ( istk .gt. 0) then
            call out_stkecr2 ( f2c,l_minx,l_maxx,l_miny,l_maxy ,&
                               meta,istk, Out_gridi0,Out_gridin,&
                                          Out_gridj0,Out_gridjn )
            out_stk_cnt= out_stk_cnt + 1
            out_stk_part(out_stk_cnt) = istk
         endif
         istk= 0
         deallocate (meta, f2c) ; nullify (meta, f2c)
         return
      endif

      modeip1= 1
      if (kind.eq.2) modeip1= 3 !old ip1 style for pressure lvls output

      do k= 1, nk_o
         istk = istk + 1
         do j= 1, l_nj
         do i= 1, l_ni
            f2c(i,j,istk)= fa(i,j,ind_o(k))*mul + add
         end do
         end do
         call convip ( meta(istk)%ip1, rf(ind_o(k)), kind,&
                       modeip1,dumc,.false. )
         meta(istk)%nv   = nomvar
         meta(istk)%ip2  = Out_ip2
         meta(istk)%ig1  = Out_ig1
         meta(istk)%ig2  = Out_ig2
         meta(istk)%ig3  = Out_ig3
         meta(istk)%ni   = Out_gridin - Out_gridi0 + 1
         meta(istk)%nj   = Out_gridjn - Out_gridj0 + 1
         meta(istk)%nbits= nbit
         meta(istk)%dtyp = 134
         if (istk.eq.max_stack) then
            call out_stkecr2 ( f2c,l_minx,l_maxx,l_miny,l_maxy ,&
                               meta,istk, Out_gridi0,Out_gridin,&
                                          Out_gridj0,Out_gridjn )
            istk=0
            out_stk_full= out_stk_full + 1
         endif
      end do
!
!--------------------------------------------------------------------
!
      return
      end
!**s/r out_fstecr3

      subroutine out_fstecr3 ( fa,lminx,lmaxx,lminy,lmaxy,rf,nomvar,&
                               mul,add,ip2,kind,nkfa,ind_o,nk_o    ,&
                               nbit,F_empty_stk_L)
      implicit none
#include <arch_specific.hf>

      character* (*) nomvar
      logical F_empty_stk_L
      integer lminx,lmaxx,lminy,lmaxy,nkfa,nbit,nk_o,kind,ip2
      integer ind_o(nk_o)
      real fa (lminx:lmaxx,lminy:lmaxy,nkfa), rf(nkfa), mul,add

!author
!    Michel Desgagne - Fall 2012
!
!revision
! v4_50 - Desgagne M. - Initial version
! v4_80 - Desgagne M. - major re-factorization of output

#include "glb_ld.cdk"
#include "out.cdk"
#include "out3.cdk"
      include "out_meta.cdk"

Interface
subroutine out_stkecr2 ( fa,lminx,lmaxx,lminy,lmaxy,meta,nplans, &
                         g_id,g_if,g_jd,g_jf )
      include "out_meta.cdk"
      integer lminx,lmaxx,lminy,lmaxy,nplans
      integer g_id,g_if,g_jd,g_jf
      real fa(lminx:lmaxx,lminy:lmaxy,nplans)
      type (meta_fstecr), dimension(:), pointer :: meta
End Subroutine out_stkecr2
End Interface

      character*8 dumc
      logical, save :: done= .false.
      integer modeip1,i,j,k
      integer max_stack
      integer, save :: istk = 0
      real, save, dimension (:,:,:), pointer :: f2c => null()
      type (meta_fstecr), save, dimension(:), pointer :: meta => null()
!
!----------------------------------------------------------------------
!
      max_stack= Out3_npes
      if (.not.associated(meta)) allocate (meta(max_stack))
      if (.not.associated(f2c )) &
         allocate (f2c(l_minx:l_maxx,l_miny:l_maxy,max_stack))
      if (.not.done) then
         out_stk_full= 0 ; out_stk_cnt= 0 ; done= .true.
      endif

      if (F_empty_stk_L) then
         if ( istk .gt. 0) then
            call out_stkecr2 ( f2c,l_minx,l_maxx,l_miny,l_maxy ,&
                               meta,istk, Out_gridi0,Out_gridin,&
                                          Out_gridj0,Out_gridjn )
            out_stk_cnt= out_stk_cnt + 1
            out_stk_part(out_stk_cnt) = istk
         endif
         istk= 0
         deallocate (meta, f2c) ; nullify (meta, f2c)
         return
      endif

      modeip1= 1
      if (kind.eq.2) modeip1= 3 !old ip1 style for pressure lvls output

      do k= 1, nk_o
         istk = istk + 1
         do j= 1, l_nj
         do i= 1, l_ni
            f2c(i,j,istk)= fa(i,j,ind_o(k))*mul + add
         end do
         end do
         call convip ( meta(istk)%ip1, rf(ind_o(k)), kind,&
                       modeip1,dumc,.false. )
         meta(istk)%nv   = nomvar
         meta(istk)%ip2  = ip2
         meta(istk)%ig1  = Out_ig1
         meta(istk)%ig2  = Out_ig2
         meta(istk)%ig3  = Out_ig3
         meta(istk)%ni   = Out_gridin - Out_gridi0 + 1
         meta(istk)%nj   = Out_gridjn - Out_gridj0 + 1
         meta(istk)%nbits= nbit
         meta(istk)%dtyp = 134
         if (istk.eq.max_stack) then
            call out_stkecr2 ( f2c,l_minx,l_maxx,l_miny,l_maxy ,&
                               meta,istk, Out_gridi0,Out_gridin,&
                                          Out_gridj0,Out_gridjn )
            istk=0
            out_stk_full= out_stk_full + 1
         endif
      end do
!
!--------------------------------------------------------------------
!
      return
      end
