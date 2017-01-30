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
      use out_collector, only: block_collect_fullp, Bloc_me
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

      logical wrapit_L, iope_L
      integer  nz, nz2, err, ni, nis, njs, k, kk
      integer, dimension (:)    , pointer     :: zlist
      real   , dimension (:,:  ), pointer     :: guwrap
      real   , dimension (:,:,:), pointer     :: wk, wk_glb
!
!----------------------------------------------------------------------
!
      nis= g_if - g_id + 1
      njs= g_jf - g_jd + 1
      if ( (nis .lt. 1) .or. (njs .lt. 1) ) return

      if (out_type_S .eq. 'REGDYN') then
         call timing_start2 ( 81, 'OUT_DUCOL', 80)
      else
         call timing_start2 ( 91, 'OUT_PUCOL', 48)
      endif

      if (Out3_ezcoll_L) then
         iope_L= (Out3_iome .ge. 0)
         nz    = (nplans + Out3_npes -1) / Out3_npes
         if (Out3_iome .ge.0) then
            allocate (wk_glb(G_ni,G_nj,nz),zlist(nz))
         else
            allocate (wk_glb(1,1,1),zlist(1))
         endif
         zlist= -1
         err= RPN_COMM_shuf_ezcoll ( Out3_comm_setno, Out3_comm_id, &
                                     wk_glb, nz, fa, nplans, zlist )
      else
         iope_L= (Bloc_me == 0)
         nullify (wk_glb, zlist, wk)
         call block_collect_fullp ( fa, l_minx,l_maxx,l_miny,l_maxy, &
                                    nplans, wk_glb, nz, zlist )
      endif
     
      if ( (iope_L) .and. (nz>0) ) then
         if ((Grd_yinyang_L) .and. (Ptopo_couleur.eq.0)) then
            allocate (wk(nis,njs*2,nz))
         else
            allocate (wk(nis,njs,nz))
         endif
         wk(1:nis,1:njs,1:nz) = wk_glb(g_id:g_if,g_jd:g_jf,1:nz)
!!$         do k= 1, nz
!!$            if (zlist(k).gt.0) then
!!$               call low_pass_dwt2d_r4 (wk(1,1,k),nis,njs,nis,njs)
!!$!                call low_pass_quant_dwt2d_r4 (wk(1,1,k),nis,njs,nis,njs)
!!$            endif
!!$         end do
         deallocate (wk_glb)
      endif

      if (out_type_S .eq. 'REGDYN') then
         call timing_stop (81)
         call timing_start2 ( 82, 'OUT_DUECR', 80)
      else
         call timing_stop (91)
         call timing_start2 ( 92, 'OUT_PUECR', 48)
      endif

      if (iope_L) then

         wrapit_L = ( (Grd_typ_S(1:2) == 'GU') .and. (nis.eq.G_ni) )
         if (wrapit_L) allocate ( guwrap(G_ni+1,njs) )

         do k= nz, 1, -1

            if (zlist(k).gt.0) then
               kk= zlist(k)

               if ( (Grd_yinyang_L) .and. (.not.Out_reduc_l) ) then

                  call out_mergeyy (wk(1,1,k), nis*njs)

                  if (Ptopo_couleur.eq.0) &

                  err = fstecr ( wk(1,1,k),wk,-meta(kk)%nbits,Out_unf, &
                              Out_dateo,Out_deet,Out_npas,nis,2*njs,1, &
                              meta(kk)%ip1,meta(kk)%ip2,meta(kk)%ip3 , &
                              Out_typvar_S,meta(kk)%nv,Out_etik_S,'U', &
                              meta(kk)%ig1,meta(kk)%ig2,meta(kk)%ig3 , &
                              Out_ig4,meta(kk)%dtyp,Out_rewrit_L )
               else

                  if (wrapit_L) then
                     guwrap(1:G_ni,1:njs) = wk(1:G_ni,1:njs,k)
                     guwrap(G_ni+1,:) = guwrap(1,:) ; ni= G_ni+1
                  else
                     guwrap => wk(1:nis,1:njs,k)    ; ni= nis
                  endif

                  err = fstecr ( guwrap,guwrap,-meta(kk)%nbits,Out_unf,&
                              Out_dateo,Out_deet,out_npas,ni,njs,1    ,&
                              meta(kk)%ip1,meta(kk)%ip2,meta(kk)%ip3  ,&
                              Out_typvar_S,meta(kk)%nv,Out_etik_S,'Z' ,&
                              meta(kk)%ig1,meta(kk)%ig2,meta(kk)%ig3  ,&
                              Out_ig4,meta(kk)%dtyp,Out_rewrit_L )

               endif

            endif

         end do

         if (wrapit_L) deallocate (guwrap)
         if (associated(wk)) deallocate (wk)
         if (associated(zlist)) deallocate (zlist)

      endif

      if (out_type_S .eq. 'REGDYN') then
         call timing_stop (82)
      else
         call timing_stop (92)
      endif
!
!--------------------------------------------------------------------
!
      return
      end
