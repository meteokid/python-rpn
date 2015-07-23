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

!** casc_hvi - Horizontal & vertical interpolation of lo-res input data
!              from 3df files to the hi-res destination grid
!
      subroutine casc_hvi (uu1,vv1,tt1,zd1,ssq1,qq1,ww1,tr1,topo1     ,&
                           uun,vvn,ttn,zdn,ssqn,qqn,wwn,trn,meqn      ,&
                           xpq1,ypq1,xpu1,ypv1,xpaq1,ypaq1,xpau1,ypav1,&
                           trname_a,F_ni,F_nj,lnk,F_nid1,F_njd1,F_nkgz,&
                           nia,nja,nka_t,nka_m,diag_lvl,presstype,ntra)
      implicit none
#include <arch_specific.hf>
!
      integer :: F_ni,F_nj,lnk,F_nid1,F_njd1,F_nkgz, &
                 nia,nja,nka_t,nka_m,diag_lvl,presstype,ntra
      character(len=*) :: trname_a(ntra)
      real*8 xpq1(F_nid1), ypq1(F_njd1), xpu1(F_ni), ypv1(F_nj)
      real*8 xpaq1(nia),ypaq1(nja),xpau1(nia),ypav1(nja)
      real uun(nia,nja,nka_m+diag_lvl  ), vvn(nia,nja,nka_m+diag_lvl)     ,&
           ttn(nia,nja,nka_t+diag_lvl  ), trn(nia,nja,nka_t+diag_lvl,ntra),&
           zdn(nia,nja,nka_t), qqn(nia,nja,nka_m), wwn(nia,nja,nka_t)     ,&
           meqn(nia,nja,F_nkgz), ssqn(nia,nja), topo1(F_nid1,F_njd1)
      real uu1(F_ni,F_nj,lnk)  , vv1(F_ni,F_nj,lnk), &
           tt1(F_ni,F_nj,lnk)  , zd1(F_ni,F_nj,lnk), &
           qq1(F_ni,F_nj,lnk)  , ww1(F_ni,F_nj,lnk), &
          ssq1(F_ni,F_nj)      , tr1(F_ni,F_nj,lnk,* )

!author    M. Desgagne  2001
!
!revision
! v3_30 - Lee V.       - initial version for GEMLAM
! v4_03 - Lee V.       - Adapt to using new pressure functions
! v4_05 - Plante A.    - top nesting
! v4_05 - Lee V.       - force vertical interpolation all the time unless ACID
! v4_14 - Desgagne M.  - Major revision
!
!presstype:
! 0 - pressure
! 1 - sigma
! 2 - etasef
! 3 - eta (rcoef=1.0)
! 4 - hybrid
! 5 - ecmwf
! 6 - staggered hybrid
!
! ? - ecmwf - not available

#include "glb_ld.cdk"
#include "bcsgrds.cdk"
#include "lam.cdk"
#include "geomg.cdk"
#include "schm.cdk"
#include "pres.cdk"
#include "tr3d.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "ptopo.cdk"
#include "lun.cdk"
#include "grd.cdk"

      integer,external ::  get_px,samevert

      logical Vertint_L,adjust_sfcpres,flag_sfcpres
      integer i,j,k,n,nga,err,jj,mustvinterp,i0,in,j0,jn
      integer nid1,njd1,ngd,ngd1,nidu,njdv,nka
      integer, dimension (:      ), allocatable :: idx,idu,idy
      real   , dimension (:      ), allocatable :: rna
      real   , dimension (:,:    ), allocatable :: ssqr,ssur,ssvr,ssq0,ssu0,ssv0
      real   , dimension (:,:    ), allocatable :: ssq0x,ssqx
      real   , dimension (:,:,:  ), allocatable :: dstlev,srclev,uur,vvr,ttr,zdr, &
                                                   qqr,wwr,meqx,ttx,vtx,hutx,pres
      real   , dimension (:,:,:,:), allocatable :: trr
      real*8 , dimension (:      ), allocatable :: cxa,cxb,cxc,cxd,cua,cub,cuc,cud, &
                                                   cya,cyb,cyc,cyd
      real*8 , dimension (:      ), allocatable :: ana_aw_8,ana_bw_8, dst_aw_8,dst_bw_8
!
!-----------------------------------------------------------------------
!
      nid1 = F_nid1
      njd1 = F_njd1

      nga  = nia  * nja
      ngd  = F_ni * F_nj
      ngd1 = nid1 * njd1

      if (ngd.le.0) return

      allocate ( idx(nid1), idy(njd1), idu(max(F_ni,F_nj)) )
      allocate ( cxa(nid1),cxb(nid1),cxc(nid1),cxd(nid1),  &
                 cua(max(nid1,njd1)),cub(max(nid1,njd1)),  &
                 cuc(max(nid1,njd1)),cud(max(nid1,njd1)),  &
                 cya(njd1),cyb(njd1),cyc(njd1),cyd(njd1),  &
                 ana_aw_8(nka_t),ana_bw_8(nka_t)        ,  &
                 dst_aw_8(G_nk),dst_bw_8(G_nk)             )

      allocate ( uur(F_ni,F_nj,nka_m+diag_lvl  ), vvr(F_ni,F_nj,nka_m+diag_lvl)     ,&
                 ttr(F_ni,F_nj,nka_t+diag_lvl  ), trr(F_ni,F_nj,nka_t+diag_lvl,ntra),&
                 qqr(F_ni,F_nj,nka_m), wwr(F_ni,F_nj,nka_t), zdr(F_ni,F_nj,nka_t)   ,&
                 ssur(F_ni,F_nj), ssvr(F_ni,F_nj), ssqr(F_ni,F_nj)                  ,&
                 ssq0(F_ni,F_nj), ssu0(F_ni,F_nj), ssv0(F_ni,F_nj) )

      allocate ( ttx (nid1,njd1,nka_t+diag_lvl), meqx(nid1,njd1,nka_t),&
                 hutx(nid1,njd1,nka_t+diag_lvl), ssqx(nid1,njd1), ssq0x(nid1,njd1) )

! Perform horizontal interpolations for ttx, meqx and ssqx on extended (nid1,njd1) grid

      call grid_to_grid_coef (xpq1,nid1,xpaq1,nia,idx,cxa,cxb,cxc,cxd, Lam_hint_S)
      call grid_to_grid_coef (ypq1,njd1,ypaq1,nja,idy,cya,cyb,cyc,cyd, Lam_hint_S)

      call hinterpo ( ttx,nid1,njd1, ttn,nia,nja,nka_t+diag_lvl, &
                      idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S )
      ttr(1:F_ni,1:F_nj,:) = ttx(1:F_ni,1:F_nj,:)

      if (presstype.eq.0) then !pressure levels
          call hinterpo ( meqx,nid1,njd1, meqn,nia,nja,nka_t, &
                          idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
          ssqx=0.0
      else
          call hinterpo ( meqx,nid1,njd1, meqn,nia,nja,   1, &
                          idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
          call hinterpo ( ssqx,nid1,njd1, ssqn,nia,nja,   1, &
                          idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
          ssqr(1:F_ni,1:F_nj) = ssqx (1:F_ni,1:F_nj)
      endif

! Perform horizontal interpolations for all other variables on 
! regular (F_ni,F_nj) grid

      if (ana_q_L)                                      &
      call hinterpo ( qqr,F_ni,F_nj, qqn,nia,nja,nka_m, &
                      idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
      if (ana_zd_L)                                     &
      call hinterpo ( zdr,F_ni,F_nj, zdn,nia,nja,nka_t, &
                      idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
      if (ana_w_L)                                      &
      call hinterpo ( wwr,F_ni,F_nj, wwn,nia,nja,nka_t, &
                      idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
      do k=1,ntra
         if (trname_a(k)(1:4).eq.'HU  ')                                          &
         call hinterpo (hutx        ,nid1,njd1,trn(1,1,1,k),nia,nja, &
                        nka_t+diag_lvl,idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
         if (trname_a(k).ne.'!@@NOT@@')                                           &
         call hinterpo (trr(1,1,1,k),F_ni,F_nj,trn(1,1,1,k),nia,nja, &
                        nka_t+diag_lvl,idx,idy,cxa,cxb,cxc,cxd,cya,cyb,cyc,cyd,Lam_hint_S)
      end do

! Horizontal interpolation ===> U point (xpu1,ypq1)
! unn=>uur (xpau1,ypaq1) ===> (xpu1,ypq1)
      call grid_to_grid_coef (xpu1,F_ni,xpau1,nia,idu,cua,cub,cuc,cud,Lam_hint_S)
      call hinterpo (uur,F_ni,F_nj,uun,nia,nja,nka_m+diag_lvl, &
                     idu,idy,cua,cub,cuc,cud,cya,cyb,cyc,cyd,Lam_hint_S)

! ssqn=>ssur (xpaq1,ypaq1) ===> (xpu1,ypq1)
      call grid_to_grid_coef (xpu1,F_ni,xpaq1,nia,idu,cua,cub,cuc,cud,Lam_hint_S)
      call hinterpo (ssur,F_ni,F_nj,ssqn,nia,nja,1, &
                     idu,idy,cua,cub,cuc,cud,cya,cyb,cyc,cyd,Lam_hint_S)

! Horizontal interpolation ===> V point (xpq1,ypv1)
! vvn=>vvr (xpaq1,ypav1) ===> (xpq1,ypv1)
      call grid_to_grid_coef (ypv1,F_nj,ypav1,nja,idu,cua,cub,cuc,cud,Lam_hint_S)
      call hinterpo (vvr,F_ni,F_nj,vvn,nia,nja,nka_m+diag_lvl, &
                     idx,idu,cxa,cxb,cxc,cxd,cua,cub,cuc,cud,Lam_hint_S)

      call grid_to_grid_coef (ypv1,F_nj,ypaq1,nja,idu,cua,cub,cuc,cud,Lam_hint_S)
! ssqn=>ssvr (xpaq1,ypaq1) ===> (xpq1,ypv1)
      call hinterpo (ssvr,F_ni,F_nj,ssqn,nia,nja,1, &
                     idx,idu,cxa,cxb,cxc,cxd,cua,cub,cuc,cud,Lam_hint_S)

      deallocate (idx,idy,idu,cxa,cxb,cxc,cxd,cua,cub,cuc,cud, &
                                              cya,cyb,cyc,cyd)

! Prepare destination ssq1,ssu0,ssv0 surface conditions for vertical interpolation

      Vertint_L = .true.

!     Obtain pressure S
      if (presstype.eq.0) then ! Analysis is on pressure coordinates
          allocate (rna(nka_t))
          do i=1,nka_t
             rna(i)=ana_am_8(i)
          enddo
          call gz2p0(ssq0x,meqx,topo1,rna,ngd1,nka_t)
          deallocate (rna)
          ssq0x(1:nid1,1:njd1) =ssq0x(1:nid1,1:njd1)-log(Cstv_pref_8)
      else if ( (presstype.eq.Ver_code .and. .not.Ana_horzint_L) .or. &
                (Lam_hint_S.eq.'NEAREST') ) then
!         NOTE: test ".or. (Lam_hint_S.eq.'NEAREST')" is needed for the acid test
          Vertint_L   = .false.
          i0=1;in=l_ni;j0=1;jn=l_nj
          if (Grd_yinyang_L) then
              i0=1+pil_w
              in=l_ni-pil_e
              j0=1+pil_s
              jn=l_nj-pil_n
          endif
          mustvinterp = samevert(ana_am_8,ana_bm_8,nka_m+1,ana_at_8,ana_bt_8,&
                        nka_t+1,meqx,topo1,nid1,njd1,i0,in,j0,jn)
          if ( mustvinterp .gt. 0 ) Vertint_L = .true.
      endif
      
      if ( abs(ana_at_8(nka_t+1)-ana_at_8(nka_t))/ana_at_8(nka_t+1) .lt. 1.e-6 ) then
         ! driver without physics
         ana_aw_8(1:nka_t)= ana_at_8(1:nka_t)
         ana_bw_8(1:nka_t)= ana_bt_8(1:nka_t)
         nka = nka_t
      else
         ! driver with physics
         ana_aw_8(1:nka_t-1)= ana_at_8(1:nka_t-1)
         ana_bw_8(1:nka_t-1)= ana_bt_8(1:nka_t-1)
         ana_aw_8(  nka_t  )= ana_at_8(  nka_t+1)
         ana_bw_8(  nka_t  )= ana_bt_8(  nka_t+1)
         nka = nka_t + 1
      endif

      if (Lun_out.gt.0) write(Lun_out,*) 'Vertical interpolation=',Vertint_L

      if (Vertint_L) then
         adjust_sfcpres= (nka .eq. (nka_t+diag_lvl))
         if ( (presstype.ne.0).and.(adjust_sfcpres) ) then

            allocate (pres(nid1,njd1,nka),vtx(nid1,njd1,nka))
            err = get_px(pres,ssqx,ngd1,ana_at_8,ana_bt_8,nka,presstype,.false.)
            if (presstype.eq.6) then
               flag_sfcpres= .false. ! do it only if necessary
               do j=1,njd1
               do i=1,nid1
                  if (abs(pres(i,j,nka)-Cstv_pref_8*exp(ssqx(i,j))) / &
                          pres(i,j,nka) .gt. 1.e-6) flag_sfcpres= .true.
               enddo
               enddo
               if (flag_sfcpres) pres(:,:,nka) = Cstv_pref_8*exp(ssqx(:,:))
            endif

            if (ana_vt_l) then
               vtx(:,:,1:nka) = ttx(:,:,1:nka)
            else
               do k=1,nka
                  call mfotvt (vtx(1,1,k),ttx(1,1,k),hutx(1,1,k),ngd1,1,ngd1)
               enddo
            endif

            call adj_ss2topo(ssq0x, topo1,pres,meqx,ttx,ngd1,nka)
            ssq0x(1:nid1,1:njd1) = log(ssq0x(1:nid1,1:njd1)/Cstv_pref_8)

            deallocate (vtx,pres)

         else

            if (presstype.eq.6) ssq0x(1:nid1,1:njd1)= ssqx(1:nid1,1:njd1)

         endif

! Compute ssq1,ssu0 and ssv0 and a first set of srclev,dstlev

         ssq1(1:F_ni,1:F_nj) = ssq0x(1:F_ni,1:F_nj)

         nidu=F_ni
         njdv=F_nj
         if (l_east) nidu=F_ni-1
         if (l_north)njdv=F_nj-1
         do j=1,F_nj
         do i=1,nidu
            ssu0(i,j)= (ssq0x(i,j)+ssq0x(i+1,j  ))*.5
         enddo
         enddo
         if (l_east) then
            do j=1,F_nj
               ssu0(F_ni,j)= ssq0x(F_ni,j)
            enddo 
         endif
         do j=1,njdv
         do i=1,F_ni
            ssv0(i,j)= (ssq0x(i,j)+ssq0x(i  ,j+1))*.5
         enddo
         enddo
         if (l_north) then
            do i=1,F_ni
               ssv0(i,F_nj)= ssq0x(i,F_nj)
             enddo 
         endif

         allocate (dstlev(F_ni,F_nj,G_nk),srclev(F_ni,F_nj,nka_t+1))

      else

         ssq1(1:F_ni,1:F_nj) = ssqx (1:F_ni,1:F_nj)

      endif

      deallocate (ttx, meqx, ssqx, ssq0x, hutx)

! VERTICAL INTERPOLATION (in log(pressure))

      if  ( (ana_zd_L) .or. (ana_w_L) ) then
         if (Schm_phyms_L) then
            dst_aw_8(1:G_nk-1)= Ver_a_8%t(1:G_nk-1)
            dst_bw_8(1:G_nk-1)= Ver_b_8%t(1:G_nk-1)
            dst_aw_8(  G_nk  )= Ver_a_8%t(  G_nk+1)
            dst_bw_8(  G_nk  )= Ver_b_8%t(  G_nk+1)
         else
            dst_aw_8(1:G_nk)= Ver_a_8%t(1:G_nk)
            dst_bw_8(1:G_nk)= Ver_b_8%t(1:G_nk)            
         endif

         err = get_px(srclev,ssqr,ngd,ana_aw_8,ana_bw_8, nka_t,presstype,.true.)
         err = get_px(dstlev,ssq1,ngd,dst_aw_8,dst_bw_8, G_nk ,Ver_code ,.true.)

! Interpolate ZD,W 
         if (ana_zd_L) then
            if (Vertint_L) then
               call vertint (zd1,dstlev,G_nk, zdr,srclev,nka_t,&
                             1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)
            else
               zd1(:,:,1:G_nk) = zdr(:,:,1:G_nk)
            endif
         endif

! Interpolate W 
         if (ana_w_L) then
            if (Vertint_L) then
               call vertint (ww1,dstlev,G_nk, wwr,srclev,nka_t,&
                             1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)
            else
               ww1(:,:,1:G_nk) = wwr(:,:,1:G_nk)
            endif
         endif
      endif

! Interpolate TT

      if (Vertint_L) then

         err = get_px(srclev,ssqr,ngd,ana_at_8,ana_bt_8,nka_t+diag_lvl,presstype,.true.)
         err = get_px(dstlev,ssq1,ngd,Ver_a_8%t,Ver_b_8%t,G_nk,Ver_code,.true.)

         call vertint ( tt1,dstlev,G_nk, ttr,srclev,nka_t+diag_lvl, &
                        1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.true.)
      else 
         tt1(:,:,1:G_nk) = ttr(:,:,1:G_nk)
      endif

! Interpolate Tracers

     do n=1,Tr3d_ntr
         jj=-1
         do k=1,ntra
            if (Tr3d_name_S(n).eq.trname_a(k)(1:4)) jj=k
         end do
         if (jj.gt.0) then
            if (Vertint_L) then
               call vertint ( tr1(1,1,1,n),dstlev,G_nk, trr(1,1,1,jj),srclev,nka_t+diag_lvl,&
                              1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)
               if (Schm_bitpattern_L) then
                  tr1(:,:,1:G_nk,n) = tr1(:,:,1:G_nk,n)
               else
                  tr1(:,:,1:G_nk,n) = max(tr1(:,:,1:G_nk,n),0.0)
               endif
            else
               if (Schm_bitpattern_L) then
                  tr1(:,:,1:G_nk,n) = trr(:,:,1:G_nk,jj)
               else
                  tr1(:,:,1:G_nk,n) = max(trr(:,:,1:G_nk,jj),0.0)
               endif
            endif
         else
            tr1(:,:,:,n)= 0.
         endif
      end do

! Interpolate Q

      if(ana_q_L) then
         if (Vertint_L) then
            err = get_px(dstlev,ssq1,ngd,Ver_a_8%m(2),Ver_b_8%m(2),G_nk  ,Ver_code ,.true.)
            err = get_px(srclev,ssqr,ngd,ana_am_8 (2),ana_bm_8 (2),nka_m ,presstype,.true.)
            call vertint ( qq1,dstlev,G_nk, qqr,srclev,nka_m,&
                           1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)
         else
            qq1(:,:,1:G_nk) = qqr(:,:,1:G_nk)
         endif
      endif

! Interpolate UT1 and VT1

      if (Vertint_L) then
         err = get_px(srclev,ssur,ngd,ana_am_8,ana_bm_8,nka_m+diag_lvl,presstype,.true.)
         err = get_px(dstlev,ssu0,ngd,Ver_a_8%m,Ver_b_8%m,G_nk,Ver_code,.true.)

         call vertint ( uu1,dstlev,G_nk, uur,srclev,nka_m+diag_lvl,&
                        1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)

         err = get_px(srclev,ssvr,ngd,ana_am_8,ana_bm_8,nka_m+diag_lvl,presstype,.true.)
         err = get_px(dstlev,ssv0,ngd,Ver_a_8%m,Ver_b_8%m,G_nk,Ver_code,.true.)

         call vertint ( vv1,dstlev,G_nk, vvr,srclev,nka_m+diag_lvl,&
                        1,F_ni,1,F_nj, 1,F_ni,1,F_nj,'cubic',.false.)

         deallocate (dstlev,srclev)

      else
         uu1(:,:,1:G_nk) = uur(:,:,1:G_nk)
         vv1(:,:,1:G_nk) = vvr(:,:,1:G_nk) 
      endif

      deallocate (ssur,ssvr,ssqr,ssq0,ssu0,ssv0)
      deallocate (uur,vvr,zdr,ttr,qqr,wwr,trr)
!
!-----------------------------------------------------------------------
!
      return
      end

