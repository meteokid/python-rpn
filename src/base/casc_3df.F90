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

!**s/r casc_3df  - Reads 3DF pilot files
!
      subroutine casc_3df3 ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo, &
                             Mminx,Mmaxx,Mminy,Mmaxy, Nk               , &
                             F_trprefix_S, F_trsuffix_S, F_datev )
      use nest_blending, only: nest_blend
      implicit none
#include <arch_specific.hf>

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u (Mminx:Mmaxx,Mminy:Mmaxy,  Nk), F_v(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_w (Mminx:Mmaxx,Mminy:Mmaxy,  Nk), F_t(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd(Mminx:Mmaxx,Mminy:Mmaxy,  Nk), F_s(Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q (Mminx:Mmaxx,Mminy:Mmaxy,2:Nk+1), F_topo(Mminx:Mmaxx,Mminy:Mmaxy)

!author
!     M. Desgagne  April 2006 (MC2 casc_3df_dynp)
!
!revision
! v3_30 - Lee V.         - initial version for GEMDM
! v3_30 - McTaggart-Cowan R. - implement variable orography
! v4_03 - Lee/Desgagne   - ISST
! v4_05 - Desgagne M.    - Add P_pbl_icelac_L, Lam_cascsfc_L 
!                          and Lam_blendoro_L options
! v4_05 - Lee V.         - Ind_u,Ind_v vectors are filled incorrectly
! v4_06 - McTaggart-Cowan R. - Nesajr correction
! v4_06 - Lee V.         - Predat is called outside of this routine

#include "gmm.hf"
#include "glb_ld.cdk"
#include "bcsgrds.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "ifd.cdk"
#include "lam.cdk"
#include "ptopo.cdk"
#include "schm.cdk"
#include "tr3d.cdk"
#include "vt1.cdk"
#include "lun.cdk"
#include "lctl.cdk"
#include "path.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "nest.cdk"
#include "vtopo.cdk"
#include "p_geof.cdk"
#include "filename.cdk"
#include "hgc.cdk"

!        presstype
!            0 => p =    A                         (B=0),  prs-anal
!            1 => p =    A+B*   ps       , F_p0=ps (A=0),  sig-anal
!            2 => p =    A+B*   ps       , F_p0=ps      ,  etasef-anal
!            3 => p =    A+B*   ps       , F_p0=ps      ,  eta-anal
!            4 => p =    A+B*   ps       , F_p0=ps      ,  hyb-anal
!        ?   5 => p =    A+B*   ps       , F_p0=ps      ,  ecm-anal
!            6 => p =exp(A+B*ln(ps/pref)), F_p0=ps      ,  stg-anal

Interface
subroutine blk_dist_georef (F_xpaq    ,F_ypaq    ,F_xpau    ,F_ypav, &
                        F_ana_am_8,F_ana_bm_8,F_ana_at_8,F_ana_bt_8, &
                        F_nia, F_nja, F_nka_t, F_nka_m, F_diag_lvl , &
                        F_ntra, F_vtype, F_datev_S, F_unf, F_err)
      character* (*) F_datev_S
      integer F_nia,F_nja,F_nka_t,F_nka_m,F_diag_lvl,F_ntra,F_vtype,F_unf,F_err
      real*8, dimension (:), pointer :: F_xpaq,F_ypaq,F_xpau,F_ypav, &
                         F_ana_am_8,F_ana_bm_8,F_ana_at_8,F_ana_bt_8
End Subroutine blk_dist_georef
End Interface

      type(gmm_metadata) :: mymeta
      character(len=4), dimension (:), pointer :: trname_a
      character*1024 fn

      logical initial_data
      integer i,j,k,jj,jjj,kk,nia,nja,nk0,nka_t,nka_m,nkgz,ntra, &
              ni1,nj1,nk1,n,err,presstype,nbits,unf,diag_lvl   , &
              ofi,ofj,l_in,l_jn,cnt,pid,gid,nij,ijk,istat,nid1,njd1
      integer i0,in,j0,jn,ni2,nj2
      integer idd,jdo,mult,shp,bigk,offbb,offbo,offg,ng,nga
      integer, dimension (:  ), pointer :: idx,idy,nks
      real topo_temp(l_ni,l_nj), xi,xf,yi,yf
      real*8, dimension (:  ), pointer ::  &
              xpaq=>null(),ypaq=>null(),xpau=>null(),ypav=>null()            , &
              xpuu=>null(),ypvv=>null(),cxa =>null(),cxb =>null(),cxc=>null(), &
              cxd =>null(),cya =>null(),cyb =>null(),cyc =>null(),cyd=>null()

      real, dimension (:,:), pointer :: &
              uun =>null(), vvn =>null(), zdn=>null(), ttn=>null(), &
              ssqn=>null(), meqn=>null(), qqn=>null(), wwn=>null(), &
              meqr=>null(), topo_temp1=>null(), ss1=>null()

      real, dimension (:,:,:  ), pointer :: &
              uu1=>null(), vv1=>null(), zd1=>null(), ttt1=>null(), &
              qq1=>null(), ww1=>null(), trn=>null(), trp =>null()
      real, dimension (:,:,:,:), pointer :: tr1=>null()

      real ssq_temp (l_minx:l_maxx,l_miny:l_maxy),&
           topo_nest(l_minx:l_maxx,l_miny:l_maxy), step_current
      real, dimension(l_minx:l_maxx,l_miny:l_maxy,G_nk) :: qh,tt
      real*8 xpxext(0:G_ni+1), ypxext(0:G_nj+1), lnpis_8, diffd
!
!-----------------------------------------------------------------------
!
      if (Lun_out.gt.0) write(lun_out,9000) trim(F_datev)

      nid1 = l_ni+1-(east *1)
      njd1 = l_nj+1-(north*1)
      ng   = l_ni * l_nj
      ofi  = l_i0 - 1
      ofj  = l_j0 - 1
!
! Positional parameters on extended global mass point grid
!
      do i=1,G_ni
         xpxext(i) = G_xg_8(i)
      end do
      xpxext(      0) = xpxext(    1) - (xpxext(    2)-xpxext(      1))
      xpxext(G_ni+1) = xpxext(G_ni) + (xpxext(G_ni)-xpxext(G_ni-1))

      do i=1,G_nj
         ypxext(i) = G_yg_8(i)
      end do
      ypxext(      0) = ypxext(    1) - (ypxext(    2)-ypxext(      1))
      ypxext(G_nj+1) = ypxext(G_nj) + (ypxext(G_nj)-ypxext(G_nj-1))
!
! Positional parameters of model target u and v grids (xpuu and ypvv).
!
      nullify (xpuu,ypvv)

      allocate (xpuu(l_ni),ypvv(l_nj))
      do i=1,l_ni
         xpuu(i) = 0.5d0 * (xpxext(ofi+i+1)+xpxext(ofi+i))
      end do
      do j=1,l_nj
         ypvv(j) = 0.5d0 * (ypxext(ofj+j+1)+ypxext(ofj+j))
      end do

      initial_data = trim(F_datev).eq.trim(Step_runstrt_S)
!
! Read all needed files and construct the source domain for
! the horizontal interpolation
!
      unf= 76

      call timing_start ( 71, '3DF_input')

      call blk_dist_georef ( xpaq    ,ypaq    ,xpau    ,ypav,&
                         ana_am_8,ana_bm_8,ana_at_8,ana_bt_8,&
                         nia, nja, nka_t, nka_m, diag_lvl   ,&
                         ntra, presstype, F_datev, unf, err )
      nga = nia*nja
      nkgz= 1
      if (presstype.eq.0) nkgz= nka_t

      nullify (trname_a,idx,idy,nks)
      nullify (cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd)
      nullify (uun,vvn,zdn,ttn,ssqn,meqn,qqn,wwn)
      nullify (meqr,topo_temp1,ss1)
      nullify (uu1,vv1,zd1,ttt1,qq1,ww1,trn,trp,tr1)

      allocate ( uun(nga,nka_m+diag_lvl), vvn(nga,nka_m+diag_lvl)     ,&
                 ttn(nga,nka_t+diag_lvl), trn(nga,nka_t+diag_lvl,ntra),&
                 qqn(nga,nka_m), wwn(nga,nka_t), zdn(nga,nka_t)       ,&
                 ssqn(nga,1), topo_temp1(nid1,njd1), meqn(nga,nkgz)   ,&
                 trname_a(ntra) )

      call blk_dist_data ( uun,vvn,zdn,ttn,ssqn,meqn,qqn,wwn,trn, &
                           trname_a, F_datev, nga, nka_t, nka_m , &
                           diag_lvl, nkgz, ntra, unf, err )

      call handle_error (err,'casc_3df_dynp','casc_3df_dynp')

      call timing_stop  ( 71 )
      call timing_start ( 72, '3DF_interp')

      if ( initial_data .and. (Step_kount.eq.0) ) then
         allocate ( meqr(l_ni,l_nj))
         allocate (idx(l_ni), idy(l_nj), &
                   cxa(l_ni), cxb(l_ni), cxc(l_ni) ,cxd(l_ni), &
                   cya(l_nj), cyb(l_nj), cyc(l_nj), cyd(l_nj) )
         if (.not. Ana_horzint_L) Lam_hint_S = 'NEAREST'
         call grid_to_grid_coef (xpxext(l_i0),l_ni, &
                                 xpaq,nia,idx,cxa,cxb,cxc,cxd,Lam_hint_S)
         call grid_to_grid_coef (ypxext(l_j0),l_nj, &
                                 ypaq,nja,idy,cya,cyb,cyc,cyd,Lam_hint_S)

         call hinterpo ( meqr,l_ni,l_nj, meqn,nia,nja,    1,idx,idy, &
                         cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)

         call get_topo (topo_temp,l_ni,l_nj)

         call adjust_topo2( F_topo, topo_temp, meqr             , &
                           ((presstype.ne.0).and.Lam_blendoro_L), &
                            l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj )

         deallocate (idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,meqr)
      endif

      if (Vtopo_L) then
         call difdatsd (diffd,Step_runstrt_S,F_datev)
         step_current = diffd*86400.d0/dble(step_dt)
         call var_topo2 (topo_nest,step_current,l_minx,l_maxx,l_miny,l_maxy)
      else
         topo_nest = F_topo
      endif

      call rpn_comm_xch_halo ( topo_nest,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,&
                               1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

      ! topo is extended
      do j=1,l_nj+1-(north*1)
      do i=1,l_ni+1-(east*1)
         topo_temp1(i,j) = topo_nest(i,j)
      enddo
      enddo

      allocate ( uu1 (l_ni,l_nj, G_nk  ), vv1 (l_ni,l_nj, G_nk), &
                 ttt1(l_ni,l_nj, G_nk  ), zd1 (l_ni,l_nj, G_nk), &
                 qq1 (l_ni,l_nj, G_nk  ), ww1 (l_ni,l_nj, G_nk), &
                 tr1 (l_ni,l_nj, G_nk,Tr3d_ntr), ss1 (l_ni,l_nj) )

      call casc_hvi (uu1,vv1,ttt1,zd1,ss1 ,qq1,ww1,tr1,topo_temp1,&
                     uun,vvn,ttn ,zdn,ssqn,qqn,wwn,trn,meqn      ,&
                     xpxext(l_i0),ypxext(l_j0),xpuu,ypvv         ,&
                     xpaq,ypaq,xpau,ypav,trname_a                ,&
                     l_ni,l_nj,G_nk,nid1,njd1,nkgz,nia,nja       ,&
                     nka_t, nka_m, diag_lvl, presstype,ntra)

      F_u (1:l_niu,1:l_nj,1:G_nk) = uu1(1:l_niu,1:l_nj,1:G_nk)
      F_v (1:l_ni,1:l_njv,1:G_nk) = vv1(1:l_ni,1:l_njv,1:G_nk)
      F_s (1:l_ni,1:l_nj        ) = ss1

      if (ana_zd_L) F_zd(1:l_ni,1:l_nj, 1:G_nk  ) = zd1
      if (ana_w_L ) F_w (1:l_ni,1:l_nj, 1:G_nk  ) = ww1
      if (ana_q_L ) F_q (1:l_ni,1:l_nj, 2:G_nk+1) = qq1

      do n=1,Tr3d_ntr
         nullify (trp)
         istat= gmm_get (trim(F_trprefix_S)//trim(Tr3d_name_S(n))//trim(F_trsuffix_S),trp)
         trp(1:l_ni,1:l_nj,1:G_nk) = tr1(1:l_ni,1:l_nj,1:G_nk,n)
      end do

      if (ana_vt_l) then
         F_t (1:l_ni,1:l_nj,1:G_nk) = ttt1
      else
         tt (1:l_ni,1:l_nj,1:G_nk) = ttt1
         nullify(trp)
         istat= gmm_get (trim(F_trprefix_S)//'HU'//trim(F_trsuffix_S),trp)
         call sumhydro (qh,l_minx,l_maxx,l_miny,l_maxy,G_nk,'P')
         if (F_trprefix_S(1:5) == 'NEST/') qh= 0. ! Must fix sumhydro or tt2virt
         call mfottvh2 ( tt,F_t,trp,qh,l_minx,l_maxx,l_miny,l_maxy, &
                         G_nk, 1,l_ni,1,l_nj, .true. )
      endif

      deallocate (xpuu,ypvv,xpaq,ypaq,xpau,ypav)
      deallocate (ana_am_8,ana_bm_8,ana_at_8,ana_bt_8)
      deallocate (uun,vvn,zdn,ttn,qqn,wwn,ssqn,meqn)
      deallocate (trn,topo_temp1,trname_a)
      deallocate (uu1,vv1,ttt1,zd1,ss1,qq1,ww1,tr1)
      if (associated(nks  )) deallocate(nks  )

      call timing_stop ( 72 )

 100  format (' ',65('*'))
 101  format (' (CASC_3DF_DYNP) JUST READ INIT DATA FOR DATE: ',a15,1x,i3)
 203  format (/' PROBLEM WITH FILE: ',a,', PROC#:',i4,' --ABORT--'/)
 204  format (/' NO DATA IN CASC_3DF_DYNP --ABORT--'/)
 205  format (/' Unrecognizable tag found: ',a,'?'/)
 9000 format(/,' TREATING INPUT DATA VALID AT: ',a,&
             /,' ===============================================')
!
!-----------------------------------------------------------------------
!
      return
      end

