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
!
      subroutine out_thm ( wlnph_m, wlnph_ta, F_st1, Minx,Maxx,Miny,Maxy,&
                                                         nk, levset, set )
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      implicit none
#include <arch_specific.hf>
!
      integer nk,Minx,Maxx,Miny,Maxy,levset,set

      real wlnph_m (Minx:Maxx,Miny:Maxy,nk+1), &
           wlnph_ta(Minx:Maxx,Miny:Maxy,nk+1), &
           F_st1   (Minx:Maxx,Miny:Maxy)
   
!
!author
!     james caveen/andre methot  - rpn june/nov 1995
!
!revision
! v2_00 - Lee V.            - initial MPI version (from blocthm v1_03)
! v2_11 - Desgagne M.       - ptop producubility
! v2_21 - Desgagne M.       - new calling sequence for glbdist + correct
! v2_21                       calling sequence mfohra3
! v2_21 - J. P. Toviessi    - set dieze (#) slab output and rename
! v2_21                       truncate model output names to 4 characters
! v2_30 - Lee V.            - reorganize slab output to be more efficient
! v2_30 - Edouard S.        - adapt for vertical hybrid coordinate
! v2_30                     - change call to p0vt2gz_hyb
! v2_32 - Lee V.            - reduce dynamic allocation size, add HU,ME output
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_01 - Lee V.            - Added output of ThetaW
! v3_01 - Morneau J.        - remove conversion to Celcius for TL or AD output
! v3_02 - Plante A.         - Water loading
! v3_02 - Lee V.            - LA and LO output (not physics), add QC output
! v3_03 - Lee V.            - correct bug for illegal access to all h2o tracers
! v3_03                       if Schm_phyms_L is false.
! v3_11   Tanguay M.        - Add TLM and ADJ increments TT and P0
!                           - Extend TRAJ for conversion for DYNOUT2
! v3_20   Lee V.            - Output in blocks, standard files
! v3_21 - Lee V.            - Output Optimization
! v3_22 - Tanguay M.        - pad fit1 (undefined values when Out3_vt2gz
! is F)
! v3_22 - Lee V.            - reduced args in calling sequence for calzz
! v3_30 - Bilodeau/Tanguay  - Output pair (TT,HU) for the adjoint
! v3_30 - Plante A.         - Correction for THETA (TH) output
! v4_04 - Tanguay M.        - Staggered version TL/AD
! v4_05 - Lee V.            - adaptation to GMM
! v4_40 - Lee V.            - change in argument call for this routine & prgen
! v4_40 - Lee V.            - correction to call to mfottvh2 for calc of VT
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! dostep     I    - array containing indices corresponding to the
!                     timestep sets that requires output at this time step.
! dostep_max I    - size of dostep array
!
! Index vectors for level identifications
! ----------------------------------------
!
! =========== no dynamic variable available at top
! 
! - - - - - -  m1
!
! -----------  t1
!  
!    ...      
!
! - - - - - -  m nk 
! o o o o o o  t nk
! ===========  diag nk+1
!
!----------------------------------------------------------------------
! 
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
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "vinterpo.cdk"

      type :: stg_i
         integer :: t,m,p
      end type stg_i

      real, parameter :: theta_p0 = 100000.

      type(gmm_metadata) :: mymeta
      real deg2rad,zd2etad
      integer i,j,k,ii,l_ninj,nko,nko_t,nko_m,istat,nk_under,nk_src
      integer, dimension(:), allocatable::indo
      real w1(l_minx:l_maxx,l_miny:l_maxy), w2(l_minx:l_maxx,l_miny:l_maxy)
      real w3(l_minx:l_maxx,l_miny:l_maxy,G_nk+1), qh(l_minx:l_maxx,l_miny:l_maxy,G_nk+1)
      real posit((l_maxx-l_minx+1)*(l_maxy-l_miny+1)*(G_nk+1)*6)
      real huv((l_maxx-l_minx+1)*(l_maxy-l_miny+1)*(G_nk+6)*2)
      real ,dimension(:,:,:), allocatable::px_pres,hu_pres,td_pres,w5,w6,cible
      real ,dimension(:,:,:), allocatable::tt_pres,vt_pres
      real ,dimension(:), allocatable::prprlvl,rf
      real ptop(l_minx:l_maxx,l_miny:l_maxy),p0(l_minx:l_maxx,l_miny:l_maxy)
      real px_ta(l_minx:l_maxx,l_miny:l_maxy,G_nk+1), px_m(l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
           th   (l_minx:l_maxx,l_miny:l_maxy,G_nk+1), tt  (l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
           hu   (l_minx:l_maxx,l_miny:l_maxy,G_nk+1), t8  (l_minx:l_maxx,l_miny:l_maxy,G_nk+1),&
           vt   (l_minx:l_maxx,l_miny:l_maxy,G_nk+1),omega(l_minx:l_maxx,l_miny:l_maxy,G_nk)
      real, pointer    , dimension(:,:,:) :: tr1,hut1,qct1
      integer n,kind
      real, dimension(:,:,:), pointer :: gzm,gzt,ttx,htx,ffwe
      real, dimension(:,:), pointer :: wlao

      integer,save :: lastdt = -1
      save gzm,gzt,htx,ttx,wlao

      integer pngz,pnvt,pntt,pnes,pntd, &
              pnhr,pnpx,pntw,pnwe,pnww,pnzz,pnth
      integer pnpn,pnp0,psum,pnpt,pnla,pnlo,pnme,pnmx

      integer nbit(0:Outd_var_max(set)+1),filt(0:Outd_var_max(set)+1)
      real    coef(0:Outd_var_max(set)+1)

      real*8, parameter :: ZERO_8 = 0.0
      real    prmult_pngz, prmult_pnpx, prmult_pnme
      real    pradd_pnvt,  pradd_pntt,  pradd_pntd, work

      type(vgrid_descriptor) :: vcoord
      integer, dimension(:), pointer :: ip1m
      real, dimension(:), pointer :: hybm,hybt,hybt_w
      save hybm,hybt,hybt_w
      logical :: write_diag_lev,near_sfc_L
      logical :: satues_L = .false.
!
!-------------------------------------------------------------------
!
      l_ninj= (l_maxx-l_minx+1)*(l_maxy-l_miny+1)

      prmult_pngz  = 0.1 / Dcst_grav_8
      prmult_pnpx  = 0.01
      prmult_pnme  = 1.0 / Dcst_grav_8

      pradd_pnvt   = -Dcst_tcdk_8
      pradd_pntt   = -Dcst_tcdk_8
      pradd_pntd   = -Dcst_tcdk_8

      pnpn=0
      pnp0=0
      pnpt=0
      pnla=0
      pnlo=0
      pnme=0
      pnmx=0

      pngz=0
      pnvt=0
      pntt=0
      pnes=0
      pntd=0
      pnhr=0
      pnpx=0
      pntw=0
      pnwe=0
      pnww=0
      pnzz=0
      pnth=0

      do ii=0,Outd_var_max(set)
         coef(ii)=0.0
         filt(ii)=0
         nbit(ii)=0
      enddo
      
      do ii=1,Outd_var_max(set)
        if (Outd_var_S(ii,set).eq.'PN') pnpn=ii
        if (Outd_var_S(ii,set).eq.'P0') pnp0=ii
        if (Outd_var_S(ii,set).eq.'PT') pnpt=ii
        if (Outd_var_S(ii,set).eq.'LA') pnla=ii
        if (Outd_var_S(ii,set).eq.'LO') pnlo=ii
        if (Outd_var_S(ii,set).eq.'ME') pnme=ii
        if (Outd_var_S(ii,set).eq.'MX') pnmx=ii
        nbit(ii)=Outd_nbit(ii,set)
        filt(ii)=Outd_filtpass(ii,set)
        coef(ii)=Outd_filtcoef(ii,set)
      enddo

      do ii=1,Outd_var_max(set)
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
        nbit(ii)=Outd_nbit(ii,set)
        filt(ii)=Outd_filtpass(ii,set)
        coef(ii)=Outd_filtcoef(ii,set)
      enddo

      if (pnpt.ne.0.and.Grd_rcoef(2).ne.1.0) pnpt=0

      psum=pnpn+pnp0+pnpt+pnla+pnlo+pnme+pnmx

      psum=psum +  &
           pngz+pnvt+pntt+pnes+pntd+pnhr+pnpx+ &
           pntw+pnwe+pnww+pnzz+pnth

      if (psum.eq.0) return

!_______________________________________________________________________

!     Obtain humidity HUT1 and other GMM variables
      nullify(hut1)
      istat = gmm_get('TR/'//'HU'//':P',hut1,mymeta)
      istat = gmm_get(gmmk_tt1_s,tt1,mymeta)
      istat = gmm_get(gmmk_wt1_s,wt1,mymeta)
      istat = gmm_get(gmmk_fis0_s,fis0,mymeta)
!_______________________________________________________________________
!
!     Obtain HU from HUT1 and physics diag level
      nk_src= l_nk
      if (Out3_sfcdiag_L) nk_src= l_nk+1

      hu(:,:,1:l_nk  )= hut1(:,:,1:l_nk)
      hu(:,:,  l_nk+1)= hut1(:,:,  l_nk)
      if (Out3_sfcdiag_L) &
           call itf_phy_sfcdiag (hu(l_minx,l_miny,nk_src),l_minx,l_maxx,l_miny,l_maxy,&
                                 'TR/HU:P',istat,.false.)
      call out_padbuf(hu,l_minx,l_maxx,l_miny,l_maxy,nk_src)
!
!     Compute TT (in tt)
!         
      call tt2virt2 (tt,.false.,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      tt(:,:,l_nk+1)= tt(:,:,l_nk)
      if (Out3_sfcdiag_L) &
           call itf_phy_sfcdiag (tt(l_minx,l_miny,nk_src),l_minx,l_maxx,l_miny,l_maxy,&
                                 'PW_TT:P',istat,.false.)
      call out_padbuf(tt,l_minx,l_maxx,l_miny,l_maxy,nk_src)

!     Obtain Virtual temperature from TT1 and physics diag level
!     On diag level there is no hydrometeor.

      vt = tt1
      if (Out3_sfcdiag_L) then
         call mfotvt (vt(l_minx,l_miny,nk_src),tt(l_minx,l_miny,nk_src),&
                      hu(l_minx,l_miny,nk_src),l_ninj,1,l_ninj)
      endif
      
      call out_padbuf(vt,l_minx,l_maxx,l_miny,l_maxy,nk_src)

!     Store PTOP (PT)
      ptop (:,:) = Cstv_ptop_8
      l_ninj=(l_maxx-l_minx+1)*(l_maxy-l_miny+1)

!     Compute and store P0
      w1(:,:)=F_st1(:,:)
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
         istat = gmm_get(gmmk_zdt1_s,zdt1,mymeta)
         ! Note: put WE=0 at first thermo level to close the domain
         !       Do not write we for thermo level nk+3/4 since zdt is at surface
         !       I user wants data at nk_3/4 us 0.5*ffwe(nk-1) (liniar interpolation)
         allocate(ffwe(l_minx:l_maxx,l_miny:l_maxy,nk))
         ffwe(:,:,nk)=0.
         do k=1,nk-1
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
            call ecris_fst2(fis0,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'ME  ',prmult_pnme,0.0,kind,1,1, 1, nbit(pnme) )
         endif
      if (pnmx.ne.0)then
            call ecris_fst2(fis0,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'MX  ',1.0,0.0,kind,1,1, 1, nbit(pnmx) )
         endif
      if (pnpt.ne.0) &
          call ecris_fst2(ptop,l_minx,l_maxx,l_miny,l_maxy,0.0, &
              'PT  ',.01,0.0,kind,1, 1, 1, nbit(pnpt) )
      if (pnla.ne.0) &
          call ecris_fst2(Geomn_latrx,1,l_ni,1,l_nj,0.0, &
              'LA  ',1.0,0.0,kind,1, 1, 1, nbit(pnla) )
      if (pnlo.ne.0) &
          call ecris_fst2(Geomn_lonrx,1,l_ni,1,l_nj,0.0, &
              'LO  ',1.0,0.0,kind,1, 1, 1, nbit(pnlo) )
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

      If (lastdt .eq. -1) then
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

         istat = gmm_get(gmmk_qt1_s,qt1,mymeta)
         call diag_fi (gzm, F_st1, tt1, qt1, fis0, &
                       l_minx,l_maxx,l_miny,l_maxy,G_nk, 1, l_ni, 1, l_nj)         

         gzt(:,:,l_nk+1)= gzm(:,:,l_nk+1)

         call vertint ( gzt, wlnph_ta,G_nk, gzm, wlnph_m,G_nk+1  ,&
                        l_minx,l_maxx,l_miny,l_maxy,1,l_ni,1,l_nj,&
                        'cubic', .false. )

         call out_liebman (ttx, htx, vt, gzt, fis0, wlao, &
                           Minx,Maxx,Miny,Maxy,Out3_lieb_nk,nk_src)
         
      endif

      lastdt = Lctl_step

!     Calculate PN
      if (pnpn.ne.0) then
         call vslog (w2, p0, l_ninj)
         call pnm2  (w1, vt(l_minx,l_miny,nk_src),fis0,w2,wlao, &
                ttx,htx,nk_under,l_minx,l_maxx,l_miny,l_maxy,1)
         if (filt(pnpn).gt.0) &
              call filter (w1,filt(pnpn),coef(pnpn),'G', .false., &
                           l_minx,l_maxx,l_miny,l_maxy, 1)
         call ecris_fst2( w1,l_minx,l_maxx,l_miny,l_maxy,0.0, &
                          'PN  ',.01, 0.0, kind, 1, 1, 1, nbit(pnpn) )
      endif
      
!     Calculate P0      
      if (pnp0.ne.0) then
         do j=l_miny,l_maxy
         do i=l_minx,l_maxx
            w1(i,j) = p0(i,j)
         enddo
         enddo
         if (filt(pnp0).gt.0)&
         call filter (w1,filt(pnp0),coef(pnp0),'G', .false.,&
                      l_minx,l_maxx,l_miny,l_maxy, 1)
         call ecris_fst2 (w1,l_minx,l_maxx,l_miny,l_maxy,0.0,&
                         'P0  ',.01, 0.0, kind, 1, 1, 1, nbit(pnp0)) 
      endif

      if (pnww.ne.0) then
         call calomeg_w (omega,F_st1,wt1,tt1,l_minx,l_maxx,l_miny,l_maxy,G_nk)
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
         kind=Level_kind_ip1
         allocate ( indo(G_nk+1) )
         call out_slev2 ( Level(1,levset), Level_max(levset), &
                          Level_momentum,indo,nko,near_sfc_L)
         write_diag_lev= near_sfc_L .and. out3_sfcdiag_L

!        Retreieve vertical coordinate description
         if ( .not. associated (hybm) ) then
            nullify(ip1m,hybm,hybt,hybt_w)
            istat = vgrid_wb_get('ref-m',vcoord,ip1m)
            deallocate(ip1m); nullify(ip1m)
            if (vgd_get(vcoord,'VCDM - vertical coordinate (m)',hybm) /= VGD_OK) istat = VGD_ERROR
            if (vgd_get(vcoord,'VCDT - vertical coordinate (t)',hybt) /= VGD_OK) istat = VGD_ERROR
            allocate(hybt_w(nk))
            ! For vertical motion quantities, we place level NK at the surface
            hybt_w(1:nk-1)=hybt(1:nk-1)
            hybt_w(nk)=1.
         endif
          
         if (pngz.ne.0)then
            call ecris_fst2(gzm,l_minx,l_maxx,l_miny,l_maxy,hybm,'GZ  ',&
                            prmult_pngz,0.0,kind,G_nk+1, indo, nko, nbit(pngz) )
            call ecris_fst2(gzt,l_minx,l_maxx,l_miny,l_maxy,hybt,'GZ  ',&
                            prmult_pngz,0.0,kind,G_nk+1, indo, nko, nbit(pngz) )
            if (near_sfc_L) then
               call ecris_fst2(gzt(l_minx,l_miny,G_nk+1)    , &
                    l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+1), &
                    'GZ  ',prmult_pngz,0.0,kind,1,1,1,nbit(pngz))
            endif
         endif

         if (pnvt.ne.0)then
            call ecris_fst2(vt,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'VT  ',1.0,pradd_pnvt,kind,G_nk+1,indo,nko,nbit(pnvt) )
            if (write_diag_lev) then
               call ecris_fst2(vt(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'VT  ',1.0,pradd_pnvt,Level_kind_diag,1,1,1,nbit(pnvt) )
            endif
         endif
         if (pnth.ne.0) then
               call ecris_fst2(th,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'TH  ',1.0,0.0,kind,G_nk+1,indo,nko,nbit(pnth) )
               if (write_diag_lev) then
                  call ecris_fst2(th(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                       'TH  ',1.0,0.0,Level_kind_diag,1,1,1,nbit(pnth) )
               endif
         endif

         if (pntt.ne.0)then
            call ecris_fst2(tt,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'TT  ' ,1.0,pradd_pntt, kind,G_nk+1,indo,nko, nbit(pntt) )
            if (write_diag_lev) then
               call ecris_fst2(tt(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'TT  ',1.0,pradd_pntt,Level_kind_diag,1,1,1,nbit(pntt) )
            endif
         endif
         if (pnes.ne.0.or.pnpx.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0)then

!            Calculate PX (in px), thermo levels.
!            And output all the levels!
             call vsexp(px_ta(l_minx,l_miny,1),wlnph_ta(Minx,Miny,1),(l_ninj*G_nk))
             px_ta(:,:,G_nk+1) = p0

!            Calculate PX (in px), momentum levels.
             call vsexp(px_m(l_minx,l_miny,1),wlnph_m(l_minx,l_miny,1),l_ninj*G_nk)
             px_m(:,:,G_nk+1) = p0

         endif
                     
         if (pnpx.ne.0)then
             call ecris_fst2(px_m,l_minx,l_maxx,l_miny,l_maxy,hybm, &
              'PX  ',prmult_pnpx,0.0,kind,G_nk+1,indo,nko,nbit(pnpx) )
             call ecris_fst2(px_ta,l_minx,l_maxx,l_miny,l_maxy,hybt, &
              'PX  ',prmult_pnpx,0.0,kind,G_nk+1,indo,nko,nbit(pnpx) )
             if (near_sfc_L) then
                call ecris_fst2(px_ta(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+1), &
                    'PX  ',prmult_pnpx,0.0,kind,1,1,1,nbit(pnpx) )
             endif
         endif

         if (pntw.ne.0) then
!        Calculate THETAW TW (t8=TW) (px=PX)
             call mthtaw4 (t8,hu,tt, px_ta,satues_l, &
                           .true.,Dcst_trpl_8,l_ninj,nk_src,l_ninj)
             call ecris_fst2(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
              'TW  ',1.0,0.0, kind,G_nk+1, indo, nko, nbit(pntw) )
             if (write_diag_lev) then
                call ecris_fst2(t8(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                     'TW  ',1.0,0.0, Level_kind_diag,1,1,1, nbit(pntw) )
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
               call ecris_fst2(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'ES  ',1.0,0.0, kind,G_nk+1, indo, nko, nbit(pnes) )
               if (write_diag_lev) then
                  call ecris_fst2(t8(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'ES  ',1.0,0.0, Level_kind_diag,1,1,1, nbit(pnes) )
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
               call ecris_fst2(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                    'TD  ',1.0,pradd_pntd,kind,G_nk+1,indo,nko,nbit(pntd) )
               if (write_diag_lev) then
                  call ecris_fst2(t8(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                       'TD  ',1.0,pradd_pntd,Level_kind_diag,1,1,1,nbit(pntd) )
               endif
            endif
         endif

         if (pnhr.ne.0) then
!            Calculate HR (t8=HR,tt=TT,px=PX)
            call mfohr4 (t8,hu,tt,px_ta,l_ninj,nk_src,l_ninj,satues_l)
            if ( Out3_cliph_L ) then
               do k= 1,G_nk+1
                  do j= 1,l_nj
                  do i= 1,l_ni
                     t8(i,j,k)= max ( min( t8(i,j,k), 1.0 ), 0. )
                  enddo
                  enddo
               enddo
            endif
            call ecris_fst2(t8,l_minx,l_maxx,l_miny,l_maxy,hybt, &
                 'HR  ',1.0,0.0, kind,G_nk+1, indo, nko, nbit(pnhr) )
            if (write_diag_lev) then
               call ecris_fst2(t8(l_minx,l_miny,G_nk+1),l_minx,l_maxx,l_miny,l_maxy,hybt(G_nk+2), &
                    'HR  ',1.0,0.0, Level_kind_diag,1,1,1, nbit(pnhr) )
            endif
         endif

         if (pnww.ne.0) then
            call ecris_fst2(omega,l_minx,l_maxx,l_miny,l_maxy,hybt_w, &
                 'WW  ',1.0,0.0, kind,G_nk, indo, nko, nbit(pnww) )
         endif

         if (pnwe.ne.0) then
            call ecris_fst2(ffwe,l_minx,l_maxx,l_miny,l_maxy,hybt_w, &
                 'WE  ',1.0,0.0, kind,nk, indo, min(nko,nk), nbit(pnwe) )
         endif

         if (pnzz.ne.0) then
            call ecris_fst2(wt1,l_minx,l_maxx,l_miny,l_maxy,hybt_w, &
                    'ZZ  ',1.0,0.0, kind,G_nk, indo, nko, nbit(pnzz) )
         endif
         deallocate(indo)

      else   ! Output on pressure levels

!        Set kind to 2 for pressure output
         kind=2
!        Setup the indexing for output 
         nko=Level_max(levset)
         allocate ( indo(nko), rf(nko) , prprlvl(nko), &
                    cible(l_minx:l_maxx,l_miny:l_maxy,nko) )

         do i = 1, nko
            indo(i)=i
            rf(i)= Level(i,levset)
            prprlvl(i)   = rf(i) * 100.0
            cible(:,:,i) = log(prprlvl(i))
         enddo

         allocate(hu_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(vt_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(tt_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(td_pres(l_minx:l_maxx,l_miny:l_maxy,nko))
         allocate(px_pres(l_minx:l_maxx,l_miny:l_maxy,nko)) 
         allocate(w5(l_minx:l_maxx,l_miny:l_maxy,nko),w6(l_minx:l_maxx,l_miny:l_maxy,nko))

!       Calculate HU (hu_pres=HU,px_ta=vert.der)

         call vertint ( hu_pres,cible,nko, hu,wlnph_ta,nk_src,&
                  l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                  'linear', .false. )

         if ( Out3_cliph_L ) then
            do k= 1, nko
               do j= 1,l_nj
               do i= 1,l_ni
                  hu_pres(i,j,k) = amax1( hu_pres(i,j,k), 0. )
               enddo
               enddo
            enddo
         endif

!       Calculate GZ,VT (w5=GZ_pres, vt_pres=VT_pres)
        call prgzvta( w5, vt_pres, prprlvl, nko, &
                      gzt, vt, wlnph_ta, wlao  , &
                      ttx, htx, nk_under,Out3_cubzt_L,  &
                      Out3_linbot, l_minx,l_maxx,l_miny,l_maxy,nk_src)

        call out_padbuf(vt_pres,l_minx,l_maxx,l_miny,l_maxy,nko)

        if (pngz.ne.0) then
           if (filt(pngz).gt.0)then
              call filter(w5,filt(pngz),coef(pngz),'G', .false., &
                    l_minx,l_maxx,l_miny,l_maxy, nko)
           endif
           call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
              'GZ  ',prmult_pngz,0.0, kind,nko,indo,nko,nbit(pngz) )
        endif

        if (pntt.ne.0.or.pntd.ne.0.or.pnhr.ne.0) then

!           Calculate TT (tt_pres=TT,vt_pres=VT,hu_pres=HU)
           call mfottv2(tt_pres,vt_pres,hu_pres, l_minx, l_maxx,l_miny,l_maxy, &
                            nko,1,l_ni,1,l_nj,.false.)
        endif

        if ( pnes.ne.0.or.pntw.ne.0.or.pntd.ne.0.or.pnhr.ne.0) then
!           Calculate PX for ES,TD,HR
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
!           Calculate THETAW TW (w5=TW_pres) (px_pres=PX)
            call mfottv2 (w6,vt_pres,hu_pres,l_minx,l_maxx, &
                        l_miny,l_maxy,nko,1,l_ni,1,l_nj,.false.)
            call out_padbuf(w6,l_minx,l_maxx,l_miny,l_maxy,nko)
            call mthtaw4 (w5,hu_pres,w6, &
                           px_pres,satues_l, &
                           .true.,Dcst_trpl_8,l_ninj,nko,l_ninj)
            if (filt(pntw).gt.0) &
                call filter(w5,filt(pntw),coef(pntw),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
            call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'TW  ',1.0,0.0, kind,nko, indo, nko, nbit(pntw) )
        endif
!
        if (pnes.ne.0.or.pntd.ne.0) then
!           Calculate ES (w5=ES_pres,hu_pres=HU,w2=VT,px_pres=PX)
            call mfottv2 (w6,vt_pres,hu_pres,l_minx,l_maxx, &
                        l_miny,l_maxy,nko,1,l_ni,1,l_nj,.false.)
            call out_padbuf(w6,l_minx,l_maxx,l_miny,l_maxy,nko)
            call mhuaes3 (w5, hu_pres,w6, &
                          px_pres,satues_l, &
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
!           Calculate TD (tt_pres=TT,w5=ES, TD=TT-ES)
              do k=1,nko
                 do j= 1, l_nj
                 do i= 1, l_ni
                    td_pres(i,j,k) = tt_pres(i,j,k) - w5(i,j,k)
                 enddo
                 enddo
              enddo
              call filter(td_pres,filt(pntd),coef(pntd),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
              call ecris_fst2(td_pres,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'TD  ',1.0,pradd_pntd, kind,nko,indo,nko,nbit(pntd) )
            endif

            if (pnes.ne.0) then
                if (filt(pnes).gt.0) &
                    call filter(w5,filt(pnes),coef(pnes),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
                call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                   'ES  ',1.0,0.0, kind,nko, indo, nko, nbit(pnes) )
            endif
        endif

        if (pnhr.ne.0) then
!           Calculate HR (w5=HR_pres:hu_pres=HU,tt_pres=TT,px_pres=PX)
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
           if (filt(pnhr).gt.0) &
                call filter(w5,filt(pnhr),coef(pnhr),'G', .false., &
                            l_minx,l_maxx,l_miny,l_maxy, nko)
           call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                'HR  ',1.0,0.0, kind,nko, indo, nko, nbit(pnhr) )
        endif
        
        if (pnvt.ne.0) then
            if (filt(pnvt).gt.0) &
                call filter(vt_pres,filt(pnvt),coef(pnvt),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
            call ecris_fst2(vt_pres,l_minx,l_maxx,l_miny,l_maxy,rf, &
              'VT  ',1.0,pradd_pnvt, kind,nko,indo, nko, nbit(pnvt) )
        endif

         if (pnth.ne.0) then
            call vertint ( w5,cible,nko, th,wlnph_ta,G_nk+1          ,&
                           l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                           'linear', .false. )
            call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
              'TH  ',1.0,      0.0,  kind,nko, indo, nko, nbit(pnth) )
         endif

        if (pntt.ne.0) then
            if (filt(pntt).gt.0) &
                call filter(tt_pres,filt(pntt),coef(pntt),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
            call ecris_fst2(tt_pres,l_minx,l_maxx,l_miny,l_maxy,rf,  &
              'TT  ',1.0,pradd_pntt, kind,nko, indo, nko, nbit(pntt) )
        endif

        if (pnww.ne.0) then
            call vertint ( w5,cible,nko, omega,wlnph_ta,G_nk         ,&
                           l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                           'linear', .false. )
            if (filt(pnww).gt.0) &
                call filter(w5,filt(pnww),coef(pnww),'G', .false., &
                        l_minx,l_maxx,l_miny,l_maxy, nko)
             call ecris_fst2(w5,l_minx,l_maxx,l_miny,l_maxy,rf, &
                 'WW  ',1.0,0.0, kind,nko, indo, nko, nbit(pnww) )
        endif

      deallocate(indo,rf,prprlvl,cible)
      deallocate(w5,w6,px_pres,hu_pres,td_pres,tt_pres,vt_pres)

      endif
!
!-------------------------------------------------------------------
!
      return
      end
