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

!**s/r out_stkecr

      subroutine out_stkecr2 ( fa,lminx,lmaxx,lminy,lmaxy, &
                               meta,nplans, g_id,g_if,g_jd,g_jf )
      use iso_c_binding
      implicit none
#include <arch_specific.hf>

      include "out_meta.cdk"

      integer lminx,lmaxx,lminy,lmaxy,nplans
      integer g_id,g_if,g_jd,g_jf
      real fa(lminx:lmaxx,lminy:lmaxy,nplans)
      type (meta_fstecr), dimension(:), pointer :: meta

!author
!    Michel Desgagne - Fall 2012
!revision
! v4_50 - Desgagne M. - Initial version
! v4_80 - Desgagne M. - switch to RPN_COMM_shuf_ezcoll

#include "glb_ld.cdk"
#include "grd.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "ptopo.cdk"
#include <rmnlib_basics.hf>
      include "rpn_comm.inc"

      integer, external :: RPN_COMM_shuf_ezcoll
      logical wrapit_L
      integer  nz, err, ni, nis, njs, k, kk
      integer, dimension (:)    , allocatable :: zlist
      real   , dimension (:,:  ), pointer     :: guwrap
      real   , dimension (:,:,:), pointer     :: wk, wk_glb
!
!----------------------------------------------------------------------
!
      nis= g_if - g_id + 1
      njs= g_jf - g_jd + 1
      if ( (nis .le. 1) .or. (njs .le. 1) ) return

      nz    = (nplans + Out3_npes -1) / Out3_npes
      allocate ( wk(nis,njs,nz+1), zlist(nz) , wk_glb(G_ni,G_nj,nz+1) )
      zlist= -1

      if (out_type_S .eq. 'REGDYN') then
         call timing_start2 ( 82, 'OUT_DUCOL', 80)
      else
         call timing_start2 ( 91, 'OUT_PUCOL', 48)
      endif

      err= RPN_COMM_shuf_ezcoll (Out3_comm_setno, Out3_comm_id, wk_glb,&
                                 nz, fa, nplans, zlist)

      if (out_type_S .eq. 'REGDYN') then
         call timing_stop (82)
         call timing_start2 ( 84, 'OUT_DUECR', 80)
      else
         call timing_stop (91)
         call timing_start2 ( 92, 'OUT_PUECR', 48)
      endif

      if (Out3_iome .ge.0) then

         wk(1:nis,1:njs,:) = wk_glb(g_id:g_if,g_jd:g_jf,:)

         wrapit_L = ( (Grd_typ_S(1:2) == 'GU') .and. (nis.eq.G_ni) )
         if (wrapit_L) allocate ( guwrap(G_ni+1,njs) )

         do k= nz, 1, -1

            if (zlist(k).gt.0) then
               kk= zlist(k)

               if ( (Grd_yinyang_L) .and. (.not.Out_reduc_l) ) then

                  call out_mergeyy (wk(1,1,k), nis*njs)
                  if (Ptopo_couleur.eq.0) &
                  err = fstecr (wk(1,1,k),wk,-meta(kk)%nbits,Out_unf,Out_dateo ,&
                                Out_deet,Out_npas,nis,2*njs,1,meta(kk)%ip1     ,&
                                Out_ip2,Out_ip3,Out_typvar_S,meta(kk)%nv       ,&
                                Out_etik_S,'U',meta(kk)%ig1,meta(kk)%ig2       ,&
                                meta(kk)%ig3,Out_ig4,meta(kk)%dtyp,Out_rewrit_L)
               else

                  if (wrapit_L) then
                     guwrap(1:G_ni,1:njs) = wk(1:G_ni,1:njs,k)
                     guwrap(G_ni+1,:) = guwrap(1,:) ; ni= G_ni+1
                  else
                     guwrap => wk(1:nis,1:njs,k)    ; ni= nis
                  endif

                  err = fstecr(guwrap,guwrap,-meta(kk)%nbits,Out_unf,Out_dateo,&
                               Out_deet,out_npas,ni,njs,1,meta(kk)%ip1        ,&
                               Out_ip2,Out_ip3,Out_typvar_S,meta(kk)%nv       ,&
                               Out_etik_S,'Z',meta(kk)%ig1,meta(kk)%ig2       ,&
                               meta(kk)%ig3,Out_ig4,meta(kk)%dtyp,Out_rewrit_L)
               endif

            endif

         end do

         if (wrapit_L) deallocate (guwrap)

      endif

      deallocate (wk,wk_glb,zlist)

      if (out_type_S .eq. 'REGDYN') then
         call timing_stop (84)
      else
         call timing_stop (92)
      endif
!
!--------------------------------------------------------------------
!
      return
      end
