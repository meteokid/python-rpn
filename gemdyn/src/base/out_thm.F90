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

!**s/r out_thm - output  temperature, humidity and mass fields

      subroutine out_thm ( levset, set )
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
#include "geomn.cdk"
#include "schm.cdk"
#include "vt1.cdk"
#include "p_geof.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "grd.cdk"
#include "level.cdk"
#include "outd.cdk"
#include "ptopo.cdk"
#include "lctl.cdk"
#include "pw.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "vinterpo.cdk"
#include "outp.cdk"

      type :: stg_i
         integer :: t,m,p
      end type stg_i

      type(vgrid_descriptor) :: vcoord

      logical,save :: done_L= .false.
      logical      :: write_diag_lev,near_sfc_L
      logical      :: satues_L= .false.

      integer,save :: lastdt= -1
      integer i,j,k,ii,l_ninj,nko,nko_t,nko_m,istat,nk_under,nk_src,&
              n,kind
      integer pngz,pnvt,pntt,pnes,pntd, pnhr,pnpx,pntw,pnwe,pnww,&
              pnzz,pnth,pnpn,pnp0,psum,pnpt,pnla,pnlo,pnme,pnmx
      integer, dimension(:), allocatable :: indo
      integer, dimension(:), pointer     :: ip1m

      real  , parameter :: theta_p0 = 100000.
      real*8, parameter :: ZERO_8   = 0.0

      real w1(l_minx:l_maxx,l_miny:l_maxy), w2(l_minx:l_maxx,l_miny:l_maxy),&
         ptop(l_minx:l_maxx,l_miny:l_maxy), p0(l_minx:l_maxx,l_miny:l_maxy),&
         deg2rad,zd2etad

      real, dimension(:,:  ), pointer    :: tdiag,qdiag
      real, dimension(:,:,:), pointer    :: hut1, wlnph_m, wlnph_ta
      real ,dimension(:,:,:), allocatable:: px_pres,hu_pres,td_pres    ,&
                                            tt_pres,vt_pres,w5,w6,cible,&
                                            gzm,gzt,ttx,htx,ffwe       ,&
                                            px_ta,px_m,th,t8,omega
      real, dimension(:,:  ), allocatable:: wlao
      real ,dimension(:    ), allocatable:: prprlvl,rf
      real, dimension(:    ), pointer    :: hybm,hybt,hybt_w

      real tt (l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
           hu (l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
           vt (l_minx:l_maxx,l_miny:l_maxy,G_nk+1)

      save gzm,gzt,htx,ttx,wlao,hybm,hybt,hybt_w
!
!-------------------------------------------------------------------
!
      l_ninj= (l_maxx-l_minx+1)*(l_maxy-l_miny+1)

      pnpn=0 ; pnp0=0 ; pnpt=0 ; pnla=0 ; pnlo=0 ; pnme=0 ; pnmx=0
      pngz=0 ; pnvt=0 ; pntt=0 ; pnes=0 ; pntd=0 ; pnhr=0 ; pnpx=0
      pntw=0 ; pnwe=0 ; pnww=0 ; pnzz=0 ; pnth=0

      do ii=1,Outd_var_max(set)
         if (Outd_var_S(ii,set).eq.'PN') pnpn=ii
         if (Outd_var_S(ii,set).eq.'P0') pnp0=ii
         if (Outd_var_S(ii,set).eq.'PT') pnpt=ii
         if (Outd_var_S(ii,set).eq.'LA') pnla=ii
         if (Outd_var_S(ii,set).eq.'LO') pnlo=ii
         if (Outd_var_S(ii,set).eq.'ME') pnme=ii
         if (Outd_var_S(ii,set).eq.'MX') pnmx=ii
         if (Outd_var_S(ii,set).eq.'GZ') pngz=ii
         if (Outd_var_S(ii,set).eq.'VT') pnvt=ii
         if (Outd_var_S(ii,set).eq.'TT') pntt=ii
         if (Outd_var_S(ii,set).eq.'ES') pnes=ii
         if (Outd_var_S(ii,set).eq.'TD') pntd=ii
         if (Outd_var_S(ii,set).eq.'HR') pnhr=ii
         if (Outd_var_S(ii,set).eq.'PX') pnpx=ii
         if (Outd_var_S(ii,set).eq.'TW') pntw=ii
         if (Outd_var_S(ii,set).eq.'WE') pnwe=ii
         if (Outd_var_S(ii,set).eq.'WW') pnww=ii
         if (Outd_var_S(ii,set).eq.'ZZ') pnzz=ii
         if (Outd_var_S(ii,set).eq.'TH') pnth=ii
      enddo

      if (pnpt.ne.0.and.Grd_rcoef(2).ne.1.0) pnpt=0

      psum=pnpn+pnp0+pnpt+pnla+pnlo+pnme+pnmx
      psum=psum +  &
           pngz+pnvt+pntt+pnes+pntd+pnhr+pnpx+ &
           pntw+pnwe+pnww+pnzz+pnth

      if (psum.eq.0) return

      if (pnww.ne.0) allocate ( omega(l_minx:l_maxx,l_miny:l_maxy,G_nk  ) ) 
      if (pnth.ne.0) allocate ( th   (l_minx:l_maxx,l_miny:l_maxy,G_nk+1) )

!     Obtain humidity HUT1 and other GMM variables
      nullify (hut1,wlnph_m,wlnph_ta,tdiag,qdiag)
      istat= gmm_get('TR/'//'HU'//':P',hut1    )
      istat= gmm_get(gmmk_tt1_s      , tt1     )
      istat= gmm_get(gmmk_wt1_s      , wt1     )
      istat= gmm_get(gmmk_fis0_s     , fis0    )
      istat= gmm_get(gmmk_pw_log_pm_s, wlnph_m )
      istat= gmm_get(gmmk_pw_log_pt_s, wlnph_ta)
      istat= gmm_get(gmmk_st1_s      , st1     )
      istat= gmm_get(gmmk_diag_tt_s  , tdiag   )
      istat= gmm_get(gmmk_diag_hu_s  , qdiag   )
      call out_padbuf (wlnph_m ,l_minx,l_maxx,l_miny,l_maxy,G_nk+1)
      call out_padbuf (wlnph_ta,l_minx,l_maxx,l_miny,l_maxy,G_nk+1)

!     Determine number of output levels
      nk_src= l_nk
      if (Out3_sfcdiag_L) nk_src= l_nk+1

!     Obtain HU from HUT1 and physics diag level
      hu(:,:,1:l_nk) = hut1(:,:,1:l_nk)
      hu(:,:,l_nk+1) = qdiag
      call out_padbuf(hu,l_minx,l_maxx,l_miny,l_maxy,l_nk+1)
!
!     Compute TT (in tt)
!         
      call tt2virt2 (tt,.false.,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      tt(:,:,l_nk+1)= tdiag
      call out_padbuf(tt,l_minx,l_maxx,l_miny,l_maxy,l_nk+1)

!     Obtain Virtual temperature from TT1 and physics diag level
!     On diag level there is no hydrometeor.

      vt(:,:,1:l_nk)= tt1(:,:,1:l_nk  )
      vt(:,:,l_nk+1)= tt1(:,:,l_nk)
      if (Out3_sfcdiag_L) then
         call mfotvt (vt(l_minx,l_miny,nk_src),tt(l_minx,l_miny,nk_src),&
                      hu(l_minx,l_miny,nk_src),l_ninj,1,l_ninj)
      endif 
      call out_padbuf(vt,l_minx,l_maxx,l_miny,l_maxy,l_nk+1)

!     Store PTOP (PT)
      ptop (:,:) = Cstv_ptop_8
      l_ninj=(l_maxx-l_minx+1)*(l_maxy-l_miny+1)

!     Compute and store P0
      w1(:,:)= st1(:,:)
      call vsexp (w2,w1,l_ninj)
      do j=l_miny,l_maxy
      do i=l_minx,l_maxx
         p0(i,j) = w2(i,j)*Cstv_pref_8
      end do
      end do

      if(pnwe.ne.0)then
         !
         ! Compute WE (Normalized velocity in eta) maily used by EER Lagrangian Dispertion Model
         !
         ! ZETA = ZETAs + ln(hyb)
         ! 
         ! Taking the total time derivative
         !  .      .
         ! ZETA = hyb/hyb
         !  .     .
         ! hyb = ZETA*hyb
         ! 
         ! Normalizing by domain height
         !       .
         ! WE = ZETA*hyb/( hyb(s) - hyb(t) )     
         !
         istat = gmm_get(gmmk_zdt1_s,zdt1)
         ! Note: put WE=0 at first thermo level to close the domain
         !       Do not write we for thermo level nk+3/4 since zdt is at surface
         !       I user wants data at nk_3/4 us 0.5*ffwe(nk-1) (liniar interpolation)
         allocate(ffwe(l_minx:l_maxx,l_miny:l_maxy,G_nk))
         ffwe(:,:,G_nk)= 0.
         do k=1,G_nk-1
            zd2etad=Ver_hyb%t(k)/(1.-Cstv_ptop_8/Cstv_pref_8)
            do j=l_miny,l_maxy
            do i=l_minx,l_maxx
               ffwe(i,j,k)=zdt1(i,j,k)*zd2etad
            end do
            end do
         end do
      endif
!_________________________________________________________________
!
!     2.0    Output 2D variables 
!_________________________________________________________________
!     output 2D fields on 0mb (pressure)
      kind=2
 
      if (pnme.ne.0)then
            call out_fstecr3(fis0,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'ME  ',Outd_convmult(pnme,set),Outd_convadd(pnme,set),&
              kind,-1,1,1, 1, Outd_nbit(pnme,set),.false. )
         endif
      if (pnmx.ne.0)then
            call out_fstecr3(fis0,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'MX  ',Outd_convmult(pnmx,set),Outd_convadd(pnmx,set),&
              kind,-1,1,1, 1, Outd_nbit(pnmx,set),.false. )
         endif
      if (pnpt.ne.0) &
          call out_fstecr3(ptop,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'PT  ',Outd_convmult(pnpt,set),Outd_convadd(pnpt,set),&
              kind,-1,1, 1, 1, Outd_nbit(pnpt,set),.false. )
      if (pnla.ne.0) &
          call out_fstecr3(Geomn_latrx,1,l_ni,1,l_nj,0.0, &
              'LA  ',Outd_convmult(pnla,set),Outd_convadd(pnla,set),&
              kind,-1,1, 1, 1, Outd_nbit(pnla,set),.false. )
      if (pnlo.ne.0) &
          call out_fstecr3(Geomn_lonrx,1,l_ni,1,l_nj,0.0, &
              'LO  ',Outd_convmult(pnlo,set),Outd_convadd(pnlo,set),&
              kind,-1,1, 1, 1, Outd_nbit(pnlo,set),.false. )
!_______________________________________________________________________
!
!     3.0    Precomputations for output over pressure levels or PN or 
!            GZ on thermo levels
!
!        The underground extrapolation can use precalculated
!        temperatures over fictitious underground geopotential levels.
!        The levels in meters are stored in Out3_lieb_levels(Out3_lieb_nk).
!        Out3_lieb_levels is a user's given parameters.
!_______________________________________________________________________
!
      nk_under = Out3_lieb_nk

      If (.not.done_L) then
          lastdt= Lctl_step-1
          done_L=.true.
          allocate ( ttx (l_minx:l_maxx,l_miny:l_maxy,nk_under),&
                     htx (l_minx:l_maxx,l_miny:l_maxy,nk_under),&
                     gzt (l_minx:l_maxx,l_miny:l_maxy,G_nk+1  ),&
                     gzm (l_minx:l_maxx,l_miny:l_maxy,G_nk+1  ),&
                     wlao(l_minx:l_maxx,l_miny:l_maxy))
!         Store WLAO (latitude in rad)
          deg2rad= acos( -1.0)/180.
          do j=1,l_nj
          do i=1,l_ni
             wlao (i,j) = Geomn_latrx(i,j) * deg2rad
          end do
          end do
      endif

!     Compute GZ on thermo levels in gzt
!     Compute ttx and htx (underground)

      if ( lastdt .ne. Lctl_step ) then

         istat = gmm_get(gmmk_qt1_s,qt1)
         call diag_fi (gzm, st1, tt1, qt1, fis0, &
                       l_minx,l_maxx,l_miny,l_maxy,G_nk, 1, l_ni, 1, l_nj)         

         gzt(:,:,l_nk+1)= gzm(:,:,l_nk+1)

         call vertint2 ( gzt, wlnph_ta,G_nk, gzm, wlnph_m,G_nk+1  ,&
                         l_minx,l_maxx,l_miny,l_maxy,1,l_ni,1,l_nj )
         call out_liebman (ttx, htx, vt, gzt, fis0, wlao, &
                           l_minx,l_maxx,l_miny,l_maxy,Out3_lieb_nk,nk_src)
         
      endif

      lastdt = Lctl_step

!     Calculate PN
      if (pnpn.ne.0) then
         call vslog (w2, p0, l_ninj)
         call pnm2  (w1, vt(l_minx,l_miny,nk_src),fis0,w2,wlao, &
                ttx,htx,nk_under,l_minx,l_maxx,l_miny,l_maxy,1)
         if (Outd_filtpass(pnpn,set).gt.0) &
             call filter2( w1,Outd_filtpass(pnpn,set),Outd_filtcoef(pnpn,set),&
                           l_minx,l_maxx,l_miny,l_maxy,1)
         call out_fstecr3( w1,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'PN  ',Outd_convmult(pnpn,set),Outd_convadd(pnpn,set), &
              kind,-1,1, 1, 1, Outd_nbit(pnpn,set),.false. )
      endif
      
!     Calculate P0      
      if (pnp0.ne.0) then
         do j=l_miny,l_maxy
         do i=l_minx,l_maxx
            w1(i,j) = p0(i,j)
         enddo
         enddo
         if (Outd_filtpass(pnp0,set).gt.0)&
         call filter2( w1,Outd_filtpass(pnp0,set),Outd_filtcoef(pnp0,set), &
                       l_minx,l_maxx,l_miny,l_maxy,1)
         call out_fstecr3 (w1,l_minx,l_maxx,l_miny,l_maxy,0.0,&
              'P0  ',Outd_convmult(pnp0,set),Outd_convadd(pnp0,set), &
              kind,-1,1, 1, 1, Outd_nbit(pnp0,set),.false.) 
      endif

      if (pnww.ne.0) then
         call calomeg_w (omega,st1,wt1,tt1,l_minx,l_maxx,l_miny,l_maxy,G_nk)
      endif

      if (pnth.ne.0) then
         do k= 1, nk_src
            do j= 1,l_nj
            do i= 1,l_ni
               th(i,j,k)= tt(i,j,k)*(theta_p0/exp(wlnph_ta(i,j,k)))**Dcst_cappa_8
            enddo
            enddo
         enddo
      endif

      if (Level_typ_S(levset) .eq. 'M') then  ! Output on model levels

!       Setup the indexing for output 
         kind= Level_kind_ip1
         allocate ( indo(G_nk+1) )
         call out_slev2 ( Level(1,levset), Level_max(levset), &
                          Level_momentum,indo,nko,near_sfc_L)
         write_diag_lev= near_sfc_L .and. out3_sfcdiag_L

!        Retrieve vertical coordinate description
         if ( .not. associated (hybm) ) then
            nullify(ip1m,hybm,hybt,hybt_w)
            istat = vgrid_wb_get('ref-m',vcoord,ip1m)
            deallocate(ip1m); nullify(ip1m)
            if (vgd_get(vcoord,'VCDM - vertical coordinate (m)',hybm) /= VGD_OK) istat = VGD_ERROR
            if (vgd_get(vcoord,'VCDT - vertical coordinate (t)',hybt) /= VGD_OK) istat = VGD_ERROR
            allocate(hybt_w(G_nk))
            ! For vertical motion quantities, we place level NK at the surface
            hybt_w(1:G_nk)= hybt(1:G_nk)
            if (.not. Schm_lift_ltl_L) hybt_w(G_nk)= 1.
         endif
          
         if (pngz.ne.0)then
            call out_fstecr3(gzm,l_minx,l_maxx,l_miny,l_maxy,hybm, &
               'GZ  ',Outd_convmult(pngz,set),Outd_convadd(pngz,set),&
               kind,-1,G_nk+1,indo,nko,Outd_nbit(pngz,set),.false. )
            call out_fstecr3(gzt,l_minx,l_maxx,l_miny,l_maxy,hybt, &
               'GZ  ',Outd_convmult(pngz,set),Outd_convadd(pngz,set),&
               kind,-1,G_nk+1,indo,nko,Outd_nbit(pngz,set),.false. )
            if (near_sfc_L) then
               call out_fstecr3(gzt(l_minx,l_miny,G_nk+1)    , &
                    l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+1), &
               'GZ  ',Outd_convmult(pngz,set),Outd_convadd(pngz,set),&
               kind,-1,1,1,1,Outd_nbit(pngz,set),.false.)
            endif
         endif

         if (pnvt.ne.0)then
            call out_fstecr3(vt,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'VT  ',Outd_convmult(pnvt,set),Outd_convadd(pnvt,set),&
                 kind,-1,G_nk+1,indo,nko,Outd_nbit(pnvt,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(vt(l_minx,l_miny,G_nk+1),&
                               l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'VT  ',Outd_convmult(pnvt,set),Outd_convadd(pnvt,set),&
                    Level_kind_diag,-1,1,1,1,Outd_nbit(pnvt,set),.false. )
            endif
         endif
         if (pnth.ne.0) then
               call out_fstecr3(th,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'TH  ',Outd_convmult(pnth,set),Outd_convadd(pnth,set),&
                    kind,-1,G_nk+1,indo,nko,Outd_nbit(pnth,set),.false. )
               if (write_diag_lev) then
                  call out_fstecr3(th(l_minx,l_miny,G_nk+1),&
                                  l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                       'TH  ',Outd_convmult(pnth,set),Outd_convadd(pnth,set),&
                       Level_kind_diag,-1,1,1,1,Outd_nbit(pnth,set),.false. )
               endif
         endif

         if (pntt.ne.0)then
            call out_fstecr3(tt,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'TT  ' ,Outd_convmult(pntt,set),Outd_convadd(pntt,set), &
                 kind,-1, G_nk+1,indo,nko,Outd_nbit(pntt,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(tt(l_minx,l_miny,G_nk+1),&
                               l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                 'TT  ',Outd_convmult(pntt,set),Outd_convadd(pntt,set),&
                 Level_kind_diag,-1,1,1,1,Outd_nbit(pntt,set),.false. )
            endif
         endif

         if (pnes.ne.0.or.pnpx.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) then

            allocate ( px_ta(l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
                       px_m (l_minx:l_maxx,l_miny:l_maxy,G_nk+1) )

!            Calculate PX (in px), thermo levels.
!            And output all the levels!
             call vsexp(px_ta(l_minx,l_miny,1),wlnph_ta(l_minx,l_miny,1),(l_ninj*G_nk))
             px_ta(:,:,G_nk+1) = p0

!            Calculate PX (in px), momentum levels.
             call vsexp(px_m(l_minx,l_miny,1),wlnph_m(l_minx,l_miny,1),l_ninj*G_nk)
             px_m(:,:,G_nk+1) = p0

         endif

         if (pnpx.ne.0)then
             call out_fstecr3(px_m,l_minx,l_maxx,l_miny,l_maxy,hybm, &
                  'PX  ',Outd_convmult(pnpx,set),Outd_convadd(pnpx,set), &
                  kind,-1,G_nk+1,indo,nko,Outd_nbit(pnpx,set),.false. )
             call out_fstecr3(px_ta,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                  'PX  ',Outd_convmult(pnpx,set),Outd_convadd(pnpx,set),&
                  kind,-1,G_nk+1,indo,nko,Outd_nbit(pnpx,set),.false. )
             if (near_sfc_L) then
                call out_fstecr3(px_ta(l_minx,l_miny,G_nk+1), &
                                l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+1), &
                     'PX  ',Outd_convmult(pnpx,set),Outd_convadd(pnpx,set),&
                     kind,-1,1,1,1,Outd_nbit(pnpx,set),.false. )
             endif
         endif

         if (pnes.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) &
               allocate (t8 (l_minx:l_maxx,l_miny:l_maxy,G_nk+1) )                     

         if (pntw.ne.0) then
!        Calculate THETAW TW (t8=TW) (px=PX)
             call mthtaw4 (t8,hu,tt, px_ta,satues_l, &
                           .true.,Dcst_trpl_8,l_ninj,nk_src,l_ninj)
             call out_fstecr3(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                  'TW  ',Outd_convmult(pntw,set),Outd_convadd(pntw,set), &
                  kind,-1,G_nk+1, indo, nko, Outd_nbit(pntw,set),.false. )
             if (write_diag_lev) then
                call out_fstecr3(t8(l_minx,l_miny,G_nk+1),&
                     l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                     'TW  ',Outd_convmult(pntw,set),Outd_convadd(pntw,set),&
                     Level_kind_diag,-1,1,1,1, Outd_nbit(pntw,set),.false. )
             endif
         endif

         if (pnes.ne.0 .or. pntd.ne.0) then
!        Calculate ES (t8=ES) (px=PX)
            call mhuaes3 (t8,hu,tt,px_ta,satues_l, &
                                  l_ninj,nk_src,l_ninj)

            if (Out3_cliph_L) then
               do k= 1,nk_src
                  do j= 1,l_nj
                  do i= 1,l_ni
                    t8(i,j,k) = max ( min( t8(i,j,k), 30. ), 0.)
                  enddo
                  enddo
               enddo
            endif

            if (pnes.ne.0) then
               call out_fstecr3(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'ES  ',Outd_convmult(pnes,set),Outd_convadd(pnes,set),&
                    kind,-1,G_nk+1,indo,nko,Outd_nbit(pnes,set),.false. )
               if (write_diag_lev) then
                  call out_fstecr3(t8(l_minx,l_miny,G_nk+1), &
                                  l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                       'ES  ',Outd_convmult(pnes,set),Outd_convadd(pnes,set),&
                       Level_kind_diag,-1,1,1,1,Outd_nbit(pnes,set),.false. )
               endif
            endif

            if (pntd.ne.0) then
!            Calculate TD (tt=TT,t8=old ES, t8=TD=TT-ES)
               do k= 1,nk_src
                  do j= 1,l_nj
                  do i= 1,l_ni
                     t8(i,j,k) = tt(i,j,k) - t8(i,j,k)
                  enddo
                  enddo
               enddo
               call out_fstecr3(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'TD  ',Outd_convmult(pntd,set),Outd_convadd(pntd,set),&
                    kind,-1,G_nk+1,indo,nko,Outd_nbit(pntd,set),.false. )
               if (write_diag_lev) then
                  call out_fstecr3(t8(l_minx,l_miny,G_nk+1), &
                       l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                       'TD  ',Outd_convmult(pntd,set),Outd_convadd(pntd,set),&
                       Level_kind_diag,-1,1,1,1,Outd_nbit(pntd,set),.false. )
               endif
            endif
         endif

         if (pnhr.ne.0) then
!            Calculate HR (t8=HR,tt=TT,px=PX)
            call mfohr4 (t8,hu,tt,px_ta,l_ninj,nk_src,l_ninj,satues_l)
            if ( Out3_cliph_L ) then
               do k= 1,nk_src
                  do j= 1,l_nj
                  do i= 1,l_ni
                     t8(i,j,k)= max ( min( t8(i,j,k), 1.0 ), 0. )
                  enddo
                  enddo
               enddo
            endif
            call out_fstecr3(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'HR  ',Outd_convmult(pnhr,set),Outd_convadd(pnhr,set),&
                 kind,-1,G_nk+1,indo,nko,Outd_nbit(pnhr,set),.false. )
            if (write_diag_lev) then
               call out_fstecr3(t8(l_minx,l_miny,G_nk+1), &
                    l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'HR  ',Outd_convmult(pnhr,set),Outd_convadd(pnhr,set),&
                    Level_kind_diag,-1,1,1,1,Outd_nbit(pnhr,set),.false. )
            endif
         endif

         if (pnww.ne.0) then
            call out_fstecr3(omega,l_minx,l_maxx,l_miny,l_maxy,hybt_w, &
                 'WW  ',Outd_convmult(pnww,set),Outd_convadd(pnww,set),&
                 kind,-1,G_nk,indo,nko,Outd_nbit(pnww,set),.false. )
         endif

         if (pnwe.ne.0) then
            call out_fstecr3(  ffwe,l_minx,l_maxx,l_miny,l_maxy,hybt_w,&
                 'WE  ',Outd_convmult(pnwe,set),Outd_convadd(pnwe,set),&
                 kind,-1,G_nk,indo,min(nko,G_nk),Outd_nbit(pnwe,set),.false.)
         endif

         if (pnzz.ne.0) then
            call out_fstecr3(wt1,l_minx,l_maxx,l_miny,l_maxy,hybt_w, &
                 'ZZ  ',Outd_convmult(pnzz,set),Outd_convadd(pnzz,set),&
                 kind,-1,G_nk,indo,nko,Outd_nbit(pnzz,set),.false. )
         endif

         deallocate (indo)

         if (pnes.ne.0.or.pnpx.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) &
            deallocate (px_ta,px_m)
         if (pnes.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) &
            deallocate (t8)

      else   ! Output on pressure levels

         nko= Level_max(levset)
         allocate ( hu_pres(l_minx:l_maxx,l_miny:l_maxy,nko), &
                    vt_pres(l_minx:l_maxx,l_miny:l_maxy,nko), &
                    tt_pres(l_minx:l_maxx,l_miny:l_maxy,nko), &
                    td_pres(l_minx:l_maxx,l_miny:l_maxy,nko), &
                    px_pres(l_minx:l_maxx,l_miny:l_maxy,nko), & 
                    w5     (l_minx:l_maxx,l_miny:l_maxy,nko), &
                    w6     (l_minx:l_maxx,l_miny:l_maxy,nko), &
                    cible  (l_minx:l_maxx,l_miny:l_maxy,nko), &
                    indo(nko), rf(nko) , prprlvl(nko) )

         kind=2 !for pressure output

         do i = 1, nko !Setup the indexing for output
            indo     (i)= i
            rf       (i)= Level(i,levset)
            prprlvl  (i)= rf(i) * 100.0
            cible(:,:,i)= log(prprlvl(i))
         enddo

! Compute HU (hu_pres=HU,px_ta=vert.der)

         call vertint2 ( hu_pres,cible,nko, hu,wlnph_ta,nk_src,&
                         l_minx,l_maxx,l_miny,l_maxy          ,&
                         1,l_ni,1,l_nj, inttype='linear' )

         if ( Out3_cliph_L ) then
            do k= 1, nko
               do j= 1,l_nj
                  do i= 1,l_ni
                     hu_pres(i,j,k) = amax1( hu_pres(i,j,k), 0. )
                  enddo
               enddo
            enddo
         endif

! Compute GZ,VT (w5=GZ_pres, vt_pres=VT_pres)

        call prgzvta( w5, vt_pres, prprlvl, nko , &
                      gzt, vt, wlnph_ta, wlao   , &
                      ttx, htx, nk_under,.false., &
                      Out3_linbot, l_minx,l_maxx,l_miny,l_maxy,nk_src)

        call out_padbuf(vt_pres,l_minx,l_maxx,l_miny,l_maxy,nko)

        if (pngz.ne.0) then
           if (Outd_filtpass(pngz,set).gt.0)then
              call filter2( w5,Outd_filtpass(pngz,set),Outd_filtcoef(pngz,set), &
                            l_minx,l_maxx,l_miny,l_maxy,nko)
           endif
           call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
              'GZ  ',Outd_convmult(pngz,set),Outd_convadd(pngz,set), &
              kind,-1,nko,indo,nko,Outd_nbit(pngz,set),.false. )
        endif

        if (pntt.ne.0.or.pntd.ne.0.or.pnhr.ne.0) then

! Compute TT (tt_pres=TT,vt_pres=VT,hu_pres=HU)
           call mfottv2 (tt_pres,vt_pres,hu_pres,l_minx,l_maxx,l_miny,l_maxy, &
                         nko,1,l_ni,1,l_nj,.false.)
        endif

        if ( pnes.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) then
! Compute PX for ES,TD,HR
            do k=1,nko
               do j= 1, l_nj
               do i= 1, l_ni
                  px_pres(i,j,k) = prprlvl(k)
               enddo
               enddo
            enddo
            call out_padbuf(px_pres,l_minx,l_maxx,l_miny,l_maxy,nko)
            call out_padbuf(tt_pres,l_minx,l_maxx,l_miny,l_maxy,nko)
            call out_padbuf(hu_pres,l_minx,l_maxx,l_miny,l_maxy,nko)
        endif

        if (pntw.ne.0) then
! Compute THETAW TW (w5=TW_pres) (px_pres=PX)
            call mfottv2 (w6,vt_pres,hu_pres,l_minx,l_maxx, &
                        l_miny,l_maxy,nko,1,l_ni,1,l_nj,.false.)
            call out_padbuf(w6,l_minx,l_maxx,l_miny,l_maxy,nko)
            call mthtaw4 (w5,hu_pres,w6, &
                           px_pres,satues_l, &
                           .true.,Dcst_trpl_8,l_ninj,nko,l_ninj)
            if (Outd_filtpass(pntw,set).gt.0) &
                call filter2( w5,Outd_filtpass(pntw,set),Outd_filtcoef(pntw,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko )
            call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'TW  ',Outd_convmult(pntw,set),Outd_convadd(pntw,set), &
                kind,-1,nko, indo, nko, Outd_nbit(pntw,set),.false. )
        endif

        if (pnes.ne.0.or.pntd.ne.0) then
! Compute ES (w5=ES_pres,hu_pres=HU,w2=VT,px_pres=PX)
            call mfottv2 (w6,vt_pres,hu_pres,l_minx,l_maxx, &
                        l_miny,l_maxy,nko,1,l_ni,1,l_nj,.false.)
            call out_padbuf(w6,l_minx,l_maxx,l_miny,l_maxy,nko)
            call mhuaes3 (w5, hu_pres,w6, px_pres,satues_l, &
                          l_ninj, nko, l_ninj)
            if ( Out3_cliph_L ) then
               do k=1,nko
                 do j= 1, l_nj
                 do i= 1, l_ni
                    w5(i,j,k) = min( w5(i,j,k), 30.)
                    w5(i,j,k) = max( w5(i,j,k), 0. )
                 enddo
                 enddo
               enddo
            endif

            if (pntd.ne.0) then
! Compute TD (tt_pres=TT,w5=ES, TD=TT-ES)
              do k=1,nko
                 do j= 1, l_nj
                 do i= 1, l_ni
                    td_pres(i,j,k) = tt_pres(i,j,k) - w5(i,j,k)
                 enddo
                 enddo
              enddo
              call filter2( td_pres,Outd_filtpass(pntd,set),Outd_filtcoef(pntd,set), &
                            l_minx,l_maxx,l_miny,l_maxy, nko )
              call out_fstecr3(td_pres,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'TD  ',Outd_convmult(pntd,set),Outd_convadd(pntd,set),&
                kind,-1,nko,indo,nko,Outd_nbit(pntd,set),.false. )
            endif

            if (pnes.ne.0) then
                if (Outd_filtpass(pnes,set).gt.0) &
                    call filter2( w5,Outd_filtpass(pnes,set),Outd_filtcoef(pnes,set), &
                                  l_minx,l_maxx,l_miny,l_maxy,nko )
                call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                   'ES  ',Outd_convmult(pnes,set),Outd_convadd(pnes,set),&
                   kind,-1,nko,indo,nko,Outd_nbit(pnes,set),.false.)
            endif
        endif

        if (pnhr.ne.0) then
! Compute HR (w5=HR_pres:hu_pres=HU,tt_pres=TT,px_pres=PX)
           call mfohr4 (w5,hu_pres,tt_pres,px_pres,l_ninj,nko,l_ninj,satues_l)
           if ( Out3_cliph_L ) then
              do k=1,nko
                 do j= 1, l_nj
                    do i= 1, l_ni
                       w5(i,j,k) = min( w5(i,j,k), 1.0 )
                       w5(i,j,k) = max( w5(i,j,k), 0.  )
                    enddo
                 enddo
              enddo
           endif
           if (Outd_filtpass(pnhr,set).gt.0) &
                call filter2( w5,Outd_filtpass(pnhr,set),Outd_filtcoef(pnhr,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko )
           call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'HR  ',Outd_convmult(pnhr,set),Outd_convadd(pnhr,set), &
                kind,-1,nko, indo, nko, Outd_nbit(pnhr,set),.false. )
        endif
        
        if (pnvt.ne.0) then
            if (Outd_filtpass(pnvt,set).gt.0) &
                call filter2( vt_pres,Outd_filtpass(pnvt,set),Outd_filtcoef(pnvt,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko )
            call out_fstecr3(vt_pres,l_minx,l_maxx,l_miny,l_maxy,rf, &
                 'VT  ',Outd_convmult(pnvt,set),Outd_convadd(pnvt,set), &
                 kind,-1,nko,indo, nko, Outd_nbit(pnvt,set),.false. )
        endif

         if (pnth.ne.0) then
            call vertint2 ( w5,cible,nko, th,wlnph_ta,G_nk+1          ,&
                            l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                           inttype='linear' )
            call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                 'TH  ',Outd_convmult(pnth,set),Outd_convadd(pnth,set), &
                 kind,-1,nko, indo, nko, Outd_nbit(pnth,set),.false. )
         endif

        if (pntt.ne.0) then
            if (Outd_filtpass(pntt,set).gt.0) &
                call filter2( tt_pres,Outd_filtpass(pntt,set),Outd_filtcoef(pntt,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko )
            call out_fstecr3(tt_pres,l_minx,l_maxx,l_miny,l_maxy,rf,  &
                 'TT  ',Outd_convmult(pntt,set),Outd_convadd(pntt,set), &
                 kind,-1,nko, indo, nko, Outd_nbit(pntt,set),.false. )
        endif

        if (pnww.ne.0) then
            call vertint2 ( w5,cible,nko, omega,wlnph_ta,G_nk         ,&
                            l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                            inttype='linear' )
            if (Outd_filtpass(pnww,set).gt.0) &
                call filter2( w5,Outd_filtpass(pnww,set),Outd_filtcoef(pnww,set), &
                              l_minx,l_maxx,l_miny,l_maxy,nko )
             call out_fstecr3(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                  'WW  ',Outd_convmult(pnww,set),Outd_convadd(pnww,set), &
                  kind,-1,nko, indo, nko, Outd_nbit(pnww,set),.false. )
        endif

        deallocate(indo,rf,prprlvl,cible)
        deallocate(w5,w6,px_pres,hu_pres,td_pres,tt_pres,vt_pres)
      endif

      if (pnww.ne.0) deallocate (omega)
      if (pnwe.ne.0) deallocate (ffwe )
      if (pnth.ne.0) deallocate (th   )
!
!-------------------------------------------------------------------
!
      return
      end
