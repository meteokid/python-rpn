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

!**s/r out_uv - output winds
!
      subroutine out_uv (levset, set)
      use vertical_interpolation, only: vertint2
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      implicit none
#include <arch_specific.hf>

      integer levset,set

!author
!     V. Lee    - rpn - July  2004 (from dynout2 v3_12)
!
!revision
! v4_80 - Desgagne M.       - major re-factorization of output

#include "gmm.hf"
#include "glb_ld.cdk"
#include "dcst.cdk"
#include "out3.cdk"
#include "geomg.cdk"
#include "out.cdk"
#include "outp.cdk"
#include "pw.cdk"
#include "vt1.cdk"
#include "level.cdk"
#include "outd.cdk"
#include "lctl.cdk"
#include "grd.cdk"
#include "schm.cdk"

      type(vgrid_descriptor) :: vcoord

      integer i,j,k,ii,istat,psum, kind,nko,nk_src,l_ninj
      integer i0,in,j0,jn,i0v,inv,j0v,jnv,pnuu,pnvv,pnuv
      integer, dimension(:), allocatable::indo
      integer, dimension(:), pointer :: ip1m
      logical, save :: done_L= .false.
      integer, save :: lastdt = -1

      real uv(l_minx:l_maxx,l_miny:l_maxy,G_nk+1)
      real, dimension(:    ), allocatable::prprlvl,rf
      real, dimension(:    ), pointer :: hybm
      save hybm
      real, dimension(:,:  ), pointer :: udiag,vdiag
      real, dimension(:,:,:), allocatable:: uv_pres,uu_pres,vv_pres,cible
      real, dimension(:,:,:), pointer, save :: uu,vv
      real*8 c1_8(l_nj)
      logical :: write_diag_lev,near_sfc_L
!
!-------------------------------------------------------------------
!
      pnuu=0 ; pnvv=0 ; pnuv=0

      do ii=1,Outd_var_max(set)
         if (Outd_var_S(ii,set).eq.'UU')then
            pnuu=ii
         endif
         if (Outd_var_S(ii,set).eq.'VV')then
            pnvv=ii
         endif
         if (Outd_var_S(ii,set).eq.'UV')then
            pnuv=ii
         endif
      enddo

      psum=pnuu+pnuv+pnvv
      if (psum.eq.0)return

      If (.not.done_L) then
         done_L=.true.
         lastdt=Lctl_step-1
         allocate ( uu(l_minx:l_maxx,l_miny:l_maxy,G_nk+1) )
         allocate ( vv(l_minx:l_maxx,l_miny:l_maxy,G_nk+1) )
      endif

      if (lastdt .ne. Lctl_step) then

         nullify(udiag,vdiag)
         istat = gmm_get(gmmk_ut1_s,ut1)
         istat = gmm_get(gmmk_vt1_s,vt1)
         istat = gmm_get(gmmk_diag_uu_s,udiag)
         istat = gmm_get(gmmk_diag_vv_s,vdiag)

         call uv_acg2g (uu ,ut1 ,1,0,l_minx,l_maxx,l_miny,l_maxy,G_nk ,i0 ,in ,j0 ,jn )
         call uv_acg2g (vv ,vt1 ,2,0,l_minx,l_maxx,l_miny,l_maxy,G_nk ,i0v,inv,j0v,jnv)

         if (G_lam) then
!         Borders need to be filled for LAM configuration
            do k=1,G_nk
               do i=1,i0-1
                  do j=1,l_nj
                     uu(i,j,k)=uu(i0,j,k)
                  enddo
               enddo
               do i=in+1,l_ni
                  do j=1,l_nj
                     uu(i,j,k)=uu(in,j,k)
                  enddo
               enddo
               do j=1,j0-1
                  do i=1,l_ni
                     uu(i,j,k)=uu(i,j0,k)
                  enddo
               enddo
               do j=jn+1,l_nj
                  do i=1,l_ni
                     uu(i,j,k)=uu(i,jn,k)
                  enddo
               enddo
               do i=1,i0v-1
                  do j=1,l_nj
                     vv(i,j,k)=vv(i0v,j,k)
                  enddo
               enddo
               do i=inv+1,l_ni
                  do j=1,l_nj
                     vv(i,j,k)=vv(inv,j,k)
                  enddo
               enddo
               do j=1,j0v-1
                  do i=1,l_ni
                     vv(i,j,k)=vv(i,j0v,k)
                  enddo
               enddo
               do j=jnv+1,l_nj
                  do i=1,l_ni
                     vv(i,j,k)=vv(i,jnv,k)
                  enddo
               enddo
            enddo
         endif

         uu(:,:,G_nk+1) = udiag
         vv(:,:,G_nk+1) = vdiag

      endif

      lastdt = Lctl_step

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      nk_src = G_nk
      if (Out3_sfcdiag_L) nk_src = G_nk+1

      if (Level_typ_S(levset) .eq. 'M') then  ! Output on model levels

         kind=Level_kind_ip1
!       Setup the indexing for output
         allocate (indo(G_nk+1))
         call out_slev2 ( Level(1,levset), Level_max(levset), &
                          Level_momentum,indo,nko,near_sfc_L)
         write_diag_lev= near_sfc_L .and. out3_sfcdiag_L

!        Retreieve vertical coordinate description
         if ( .not. associated(hybm) ) then
            nullify(ip1m,hybm)
            istat = vgrid_wb_get('ref-m',vcoord,ip1m)
            deallocate(ip1m); nullify(ip1m)
            if (vgd_get(vcoord,'VCDM - vertical coordinate (m)',hybm) /= VGD_OK) istat = VGD_ERROR
         endif

         if (pnuu.ne.0) then
            call out_fstecr3(uu,l_minx,l_maxx,l_miny,l_maxy,hybm   ,&
              'UU  ',Outd_convmult(pnuu,set),Outd_convadd(pnuu,set),&
              kind,-1,G_nk, indo, nko,Outd_nbit(pnuu,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(uu(l_minx,l_miny,G_nk+1), &
                            l_minx,l_maxx,l_miny,l_maxy, hybm(G_nk+2),&
                'UU  ',Outd_convmult(pnuu,set),Outd_convadd(pnuu,set),&
                Level_kind_diag,-1,1,1,1,Outd_nbit(pnuu,set),.false. )
            endif
         endif

         if (pnvv.ne.0) then
            call out_fstecr3(vv,l_minx,l_maxx,l_miny,l_maxy,hybm   ,&
              'VV  ',Outd_convmult(pnvv,set),Outd_convadd(pnvv,set),&
              kind,-1,G_nk, indo, nko,Outd_nbit(pnvv,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(vv(l_minx,l_miny,G_nk+1) ,&
                l_minx,l_maxx,l_miny,l_maxy, hybm(G_nk+2),&
                'VV  ',Outd_convmult(pnvv,set),Outd_convadd(pnvv,set),&
                Level_kind_diag,-1,1,1,1, Outd_nbit(pnvv,set),.false. )
            endif
         endif

         if (pnuv.ne.0) then
            do k = 1, nk_src
               do j = j0, jn
                  do i = i0, in
                     uv(i,j,k) = sqrt(uu(i,j,k)*uu(i,j,k)+ &
                                      vv(i,j,k)*vv(i,j,k))
                  enddo
               enddo
            enddo
            call out_fstecr3(uv,l_minx,l_maxx,l_miny,l_maxy,hybm      ,&
                 'UV  ',Outd_convmult(pnuv,set),Outd_convadd(pnuv,set),&
                 kind,-1,G_nk, indo, nko, Outd_nbit(pnuv,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(uv(l_minx,l_miny,G_nk+1)     ,&
                    l_minx,l_maxx,l_miny,l_maxy, hybm(G_nk+2),&
                    'UV  ',Outd_convmult(pnuv,set),Outd_convadd(pnuv,set),&
                    Level_kind_diag,-1,1,1,1, Outd_nbit(pnuv,set),.false. )
            endif
         endif
         deallocate(indo)

      else   ! Output on pressure levels

         istat= gmm_get(gmmk_pw_log_pm_s, pw_log_pm)

!       Set kind to 2 for pressure output
         kind=2
!       Setup the indexing for output
         nko=Level_max(levset)
         allocate ( indo(nko), rf(nko) , prprlvl(nko), &
                    cible(l_minx:l_maxx,l_miny:l_maxy,nko) )
         do i = 1, nko
            indo(i)=i
            rf(i)= Level(i,levset)
            prprlvl(i) = rf(i) * 100.0
            cible(:,:,i) = log(prprlvl(i))
         enddo
         allocate(uu_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(vv_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(uv_pres(l_minx:l_maxx,l_miny:l_maxy,nko))

!        Vertical interpolation

         call vertint2 ( uu_pres, cible, nko, uu, pw_log_pm, nk_src,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                         inttype='linear' )
         call vertint2 ( vv_pres, cible, nko, vv, pw_log_pm, nk_src,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                         inttype='linear' )

         if(pnuv.ne.0) then
!        Compute UV
            do k =  1, nko
               do j = j0, jn
                  do i = i0, in
                     uv_pres(i,j,k) = sqrt(uu_pres(i,j,k)*uu_pres(i,j,k)+ &
                                          vv_pres(i,j,k)*vv_pres(i,j,k))
                  enddo
               enddo
            enddo
            if (Outd_filtpass(pnuv,set).gt.0) &
               call filter2( uv_pres,Outd_filtpass(pnuv,set),&
                             Outd_filtcoef(pnuv,set), &
                             l_minx,l_maxx,l_miny,l_maxy,nko)
            call out_fstecr3(uv_pres,l_minx,l_maxx,l_miny,l_maxy,rf   ,&
                 'UV  ',Outd_convmult(pnuv,set),Outd_convadd(pnuv,set),&
                 kind,-1,nko, indo, nko, Outd_nbit(pnuv,set),.false. )
         endif

         if (pnuu.ne.0) then
            if (Outd_filtpass(pnuu,set).gt.0) &
                call filter2( uu_pres,Outd_filtpass(pnuu,set),&
                              Outd_filtcoef(pnuu,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko)
            call out_fstecr3(uu_pres,l_minx,l_maxx,l_miny,l_maxy,rf   ,&
                 'UU  ',Outd_convmult(pnuu,set),Outd_convadd(pnuu,set),&
                 kind,-1,nko, indo, nko, Outd_nbit(pnuu,set),.false. )
         endif

         if (pnvv.ne.0) then
            if (Outd_filtpass(pnvv,set).gt.0) &
                 call filter2( vv_pres,Outd_filtpass(pnvv,set),&
                               Outd_filtcoef(pnvv,set), &
                               l_minx,l_maxx,l_miny,l_maxy,nko)
            call out_fstecr3(vv_pres,l_minx,l_maxx,l_miny,l_maxy,rf   ,&
                 'VV  ',Outd_convmult(pnvv,set),Outd_convadd(pnvv,set),&
                 kind,-1,nko, indo, nko, Outd_nbit(pnvv,set),.false. )
         endif

         deallocate(indo,rf,prprlvl,uu_pres,vv_pres,uv_pres,cible)

      endif
!
!-------------------------------------------------------------------
!
      return
      end subroutine out_uv
