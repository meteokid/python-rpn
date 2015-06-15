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

!**s/r itf_phy_output
!
      subroutine itf_phy_output2 (stepno)
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
      use out_vref_mod, only: out_vref
      use phy_itf, only: phy_get,phymeta,phy_getmeta
      implicit none
#include <arch_specific.hf>

      integer stepno

!AUTHOR     Michel Desgagne                July 2004
!
!REVISION
! v3_20 - Lee V.            -  initial GEMDM version
! v3_21 - Lee V.            -  bugfix for LAM output
! v3_30 - McTaggart-Cowan R.-  allow for user-defined domain tag extensions
! v3_31 - Lee V.            - kind is set to 2 (press) for 2D fields, not -1
! v4_03 - Lee V.            - modification of Out_etik_S in out_sgrid only
! v4_05 - Lepine M.         - VMM replacement with GMM
! v4_06 - Lee V.            - out_sgrid,out_href interface changed
! v4_40 - Lee V.            - add mosaic output, pressure output

#include "gmm.hf"
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "dcst.cdk"
#include "lctl.cdk"
#include "lun.cdk"
#include "out3.cdk"
#include "init.cdk"
#include "out.cdk"
#include "grd.cdk"
#include "grid.cdk"
#include "level.cdk"
#include "outp.cdk"
#include "pw.cdk"
#include "vt1.cdk"
#include "out_listes.cdk"

      type(phymeta) :: pmeta
      type(vgrid_descriptor) :: vcoord
      character*4 ext_S
      character*6 etikadd_S
      integer i,ii,jj,k,kk,levset,nko,nko_pres,cnt,ip3,istat,&
              gridset,ig1,mult,mosaic,bcs_ext,p_li0,p_li1,p_lj0,p_lj1
      integer grille_x0,grille_x1,grille_y0,grille_y1
      integer, dimension (:), allocatable :: indo_pres,indo,irff
      integer, dimension(:), pointer :: ip1m
      logical periodx_L, flag_clos, write_diag_lev
      real mul
      real, dimension (l_ni,l_nj,G_nk+1), target ::  wlnpi_m,wlnpi_t
      real, dimension(:), allocatable::prprlvl,rff
      real, dimension(:), pointer :: hybm,hybt
      real, dimension(:,:,:), allocatable :: buso_pres,cible
      real, dimension(:,:,:), pointer :: lnpres
      real, dimension(:,:,:), allocatable, target :: data3d
      real, dimension(:,:,:), pointer :: ptr3d
!
!----------------------------------------------------------------------
!
      if (outp_sorties(0,stepno).le.0) then
         if (Lun_out.gt.0) write(Lun_out,7002) stepno
         return
      else
         if (Lun_out.gt.0) write(Lun_out,7001) stepno,trim(Out_laststep_S)
      endif

      out3_type_S= 'REGPHY'
!
!     setup of ip3 and modifs to label
!
      ip3=0
      etikadd_S = ' '
      ext_S=""
      if (Out3_ip3.eq.-1) ip3 = stepno
      if (Out3_ip3.gt.0 ) ip3 = Out3_ip3

!     setup of filename extension if needed
      if ( Init_mode_L .and. (stepno.gt.Init_halfspan) ) ext_S= '_dgf'

!     setup domain extent to retrieve physics data
      allocate(data3d(l_ni,l_nj,G_nk+1))
      data3d = 0.

      istat= gmm_get(gmmk_pw_pm_plus_s ,pw_pm_plus)
      istat= gmm_get(gmmk_pw_pt_plus_s ,pw_pt_plus)
      istat= gmm_get(gmmk_pw_p0_plus_s ,pw_p0_plus)

      do k=1,G_nk
         wlnpi_m(1:l_ni,1:l_nj,k)= log(pw_pm_plus(1:l_ni,1:l_nj,k))
         wlnpi_t(1:l_ni,1:l_nj,k)= log(pw_pt_plus(1:l_ni,1:l_nj,k))
      enddo
      wlnpi_m(1:l_ni,1:l_nj,G_nk+1)= log(pw_p0_plus(1:l_ni,1:l_nj))
      wlnpi_t(1:l_ni,1:l_nj,G_nk+1)= log(pw_p0_plus(1:l_ni,1:l_nj))
      
      allocate (rff(Outp_multxmosaic),irff(Outp_multxmosaic))

      Out_ip3 = 0
      if (Out3_ip3.eq.-1) Out_ip3 = Lctl_step
      if (Out3_ip3.gt.0 ) Out_ip3 = Out3_ip3

      p_li0= Grd_lphy_i0 ; p_li1=Grd_lphy_in
      p_lj0= Grd_lphy_j0 ; p_lj1=Grd_lphy_jn

!     Retreieve vertical coordinate description
      nullify(ip1m,hybm,hybt)
      istat = vgrid_wb_get('ref-m',vcoord,ip1m)
      deallocate(ip1m); nullify(ip1m)
      if (vgd_get(vcoord,'VCDM - vertical coordinate (m)',hybm) /= VGD_OK) istat = VGD_ERROR
      if (vgd_get(vcoord,'VCDT - vertical coordinate (t)',hybt) /= VGD_OK) istat = VGD_ERROR

      do jj=1, outp_sorties(0,stepno)

         kk       = outp_sorties(jj,stepno)
         gridset  = Outp_grid(kk)
         levset   = Outp_lev(kk)
         periodx_L= .not.G_lam .and. ((Grid_x1(gridset)-Grid_x0(gridset)+1).eq.G_ni)

         allocate (indo  ( min(Level_max(levset),Level_momentum) ))
         
         call out_slev2 (Level(1,levset), Level_max(levset), &
                        Level_momentum,indo,nko,write_diag_lev)

         nko_pres = Level_max(levset)
         allocate ( indo_pres(nko_pres),buso_pres(l_ni,l_nj,nko_pres),prprlvl(nko_pres),&
                    cible(l_ni,l_nj,nko_pres) )
         buso_pres= 0.
         do i = 1, nko_pres
            indo_pres(i)= i
            prprlvl(i)   = level(i,levset) * 100.0
            cible(:,:,i) = log(prprlvl(i))
         enddo

         ig1 = Grid_ig1(gridset)
         bcs_ext = 0
         if (G_lam) bcs_ext = Grd_bsc_ext1
         grille_x0 = max( 1   +bcs_ext, Grid_x0(gridset) )
         grille_x1 = min( G_ni-bcs_ext, Grid_x1(gridset) )
         grille_y0 = max( 1   +bcs_ext, Grid_y0(gridset) )
         grille_y1 = min( G_nj-bcs_ext, Grid_y1(gridset) )
         if (G_lam .and. &
              ( grille_x0.ne.Grid_x0(gridset).or. &
                grille_x1.ne.Grid_x1(gridset).or. &
                grille_y0.ne.Grid_y0(gridset).or. &
                grille_y1.ne.Grid_y1(gridset) ) ) ig1=Grid_ig1(gridset)+100

         call out_sgrid2 ( grille_x0,grille_x1,grille_y0,grille_y1, &
                           ig1,Grid_ig2(gridset)                  , &
                           periodx_L, Grid_stride(gridset)        , &
                           Grid_etikext_s(gridset))
         Out_prefix_S(1:1) = 'p'
         Out_prefix_S(2:2) = Level_typ_S(levset)

         call out_sfile3 (stepno)

         call out_href2 ( 'Mass_point' )

         if (Level_typ_S(levset).eq.'M') then
! Preparation for Model level output
             call out_vref(ig1=ig1,etiket=Out_etik_S)
         else
! Preparation for Pressure output
             call out_vref(Level(1:Level_max(levset),levset),ig1=ig1,etiket=Out_etik_S)
         endif

         PHYSICS_VARS: do ii=1, Outp_var_max(kk)

            WRITE_FIELD: if (phy_getmeta ( pmeta, Outp_var_S(ii,kk), F_npath='O',&
                 F_bpath='PVE', F_quiet=.true. ) > 0 ) then

               FIELD_SHAPE: if (pmeta%nk .eq. 1) then
!                 2D field
                  mul= 1.0
                  if (trim(Outp_var_S(ii,kk)).eq.'LA') mul= 180./Dcst_pi_8
                  if (trim(Outp_var_S(ii,kk)).eq.'LO') mul= 180./Dcst_pi_8
                  if (trim(Outp_var_S(ii,kk)).eq.'SD') mul= 100.

                  if ( pmeta%fmul.eq.1 .and. pmeta%mosaic.eq.0) then
                     ! 2D none multiple, none mosaic field, 0mb field, kind=2
                     ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,1:1)
                     istat = phy_get ( ptr3d, Outp_var_S(ii,kk),F_npath='O', &
                                       F_bpath='PV')
                     call ecris_fst2 ( data3d, 1,l_ni, 1,l_nj, 0.0,Outp_var_S(ii,kk),&
                                       mul,0., 2, 1,1,1,Outp_nbit(ii,kk) )
                  else
                     ! 2D (multiple) field - on arbitrary levels, kind=3
!     multiple 2D fields and mosaic tiles are stored in
!     slices of NI where the order is as follows:
!     Example is for Mult=3, mosaic = 3
!     1 (multiple 1)
!     2 (multiple 2)
!     3 (multiple 3)
!     1.01 (aggregate of tile 1 for multiple 1
!     2.01 (aggregate of tile 1 for multiple 2
!     3.01 (aggregate of tile 1 for multiple 3
!     1.02 (aggregate of tile 2 for multiple 1
!     2.02 (aggregate of tile 2 for multiple 2
!     3.02 (aggregate of tile 2 for multiple 3
!     1.03 (aggregate of tile 2 for multiple 1
!     2.03 (aggregate of tile 2 for multiple 2
!     3.03 (aggregate of tile 2 for multiple 3
                     do mult=1,pmeta%fmul
                         rff(mult)= mult
                        irff(mult)= mult 
                     enddo
                     cnt= pmeta%fmul
                     if(pmeta%mosaic.gt.0) then
                        !Add to output if Mosaic field found
                        do mosaic=1, pmeta%mosaic
                        do mult  =1, pmeta%fmul
                           cnt= cnt+1
                            rff(cnt)= mult*1.0 + mosaic/100.
                           irff(cnt)= cnt
                        end do
                        end do
                     endif
                     ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,1:cnt)
                     istat = phy_get ( ptr3d, Outp_var_S(ii,kk),F_npath='O', &
                                       F_bpath='PV')
                     call ecris_fst2 ( data3d, 1,l_ni, 1,l_nj, rff,Outp_var_S(ii,kk),&
                                       mul,0., 3, cnt, irff,cnt, Outp_nbit(ii,kk) )
                  endif
               else
                  ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,:)
                  istat = phy_get ( ptr3d, Outp_var_S(ii,kk),F_npath='O', &
                                     F_bpath='PV')

                  if (Level_typ_S(levset).eq.'M') then

                     if (pmeta%stag .eq. 1) then ! thermo
                        call ecris_fst2 (data3d, 1,l_ni, 1,l_nj, hybt  , &
                                         Outp_var_S(ii,kk),1.,0.,Level_kind_ip1, &
                                         G_nk,indo,nko,Outp_nbit(ii,kk) )
                        if (write_diag_lev) then
                           call ecris_fst2 (data3d(1,1,G_nk+1), 1,l_ni, 1,l_nj, hybt(G_nk+2), &
                                Outp_var_S(ii,kk),1.,0.,Level_kind_diag,1,1,1, Outp_nbit(ii,kk) )
                        endif
                     else  ! momentum
                        call ecris_fst2 (data3d, 1,l_ni, 1,l_nj, hybm     , &
                                         Outp_var_S(ii,kk),1.,0.,Level_kind_ip1, &
                                         G_nk,indo,nko,Outp_nbit(ii,kk) )
                        if (write_diag_lev) then
                           call ecris_fst2 (data3d(1,1,G_nk+1), 1,l_ni, 1,l_nj, hybm(G_nk+2), &
                                Outp_var_S(ii,kk),1.,0.,Level_kind_diag,1,1,1, Outp_nbit(ii,kk) )
                        endif
                     endif

                  elseif (Level_typ_S(levset).eq.'P') then

                     lnpres => wlnpi_m
                     if ( pmeta%stag .eq. 1 ) lnpres => wlnpi_t

                     call vertint ( buso_pres, cible, nko_pres, data3d, lnpres, G_nk,&
                                    1,l_ni, 1,l_nj, 1,l_ni, 1,l_nj, 'linear', .false. )

                     call ecris_fst2 ( buso_pres, 1,l_ni, 1,l_nj    , &
                             level(1,levset),Outp_var_S(ii,kk),1.,0., &
                             2,nko_pres,indo_pres,nko_pres,Outp_nbit(ii,kk) )

                  endif
               endif FIELD_SHAPE
            endif WRITE_FIELD
         end do PHYSICS_VARS

         deallocate (indo, buso_pres, indo_pres, prprlvl, cible)
         
         flag_clos= .true.
         if (jj .lt. outp_sorties(0,stepno)) then
            flag_clos= .not.( (gridset.eq.Outp_grid(outp_sorties(jj+1,stepno))).and. &
                 (Level_typ_S(levset).eq.Level_typ_S(Outp_lev(outp_sorties(jj+1,stepno)))))
         endif
         
         if (flag_clos) call out_cfile2
         
      end do
      
      deallocate(rff,irff,data3d)
      deallocate(hybm,hybt); nullify(hybm,hybt)

 7001 format(/,' OUT_PHY- WRITING PHYSICS OUTPUT FOR STEP (',I8,') in directory: ',a)
 8001 format(/,' OUT_PHY- WRITING CASCADE OUTPUT FOR STEP (',I8,') in directory: ',a)
 7002 format(/,' OUT_PHY- NO PHYSICS OUTPUT FOR STEP (',I8,')')
!
!----------------------------------------------------------------------
!
 999  return
      end


