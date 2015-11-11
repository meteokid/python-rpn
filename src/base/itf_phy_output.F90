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
      use phy_itf, only: phy_get,phymeta,phy_getmeta,phy_put
      implicit none
#include <arch_specific.hf>

      integer stepno

!AUTHOR     Michel Desgagne                July 2004
!
!REVISION
! v4_80 - Desgagne M.       - major re-factorization of output

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lctl.cdk"
#include "lun.cdk"
#include "out3.cdk"
#include "out.cdk"
#include "grd.cdk"
#include "grid.cdk"
#include "level.cdk"
#include "outp.cdk"
#include "ptopo.cdk"
#include "step.cdk"
#include "pw.cdk"
#include "out_listes.cdk"
#include <rmnlib_basics.hf>

      type(phymeta) :: pmeta
      type(vgrid_descriptor) :: vcoord
      character*15 prefix
      character*8 dumc
      integer i,ii,jj,k,kk,levset,nko,nko_pres,cnt,istat,&
              gridset,ig1,mult,mosaic, ip2,modeip1,kind ,&
              bcs_ext,p_li0,p_li1,p_lj0,p_lj1
      integer grille_x0,grille_x1,grille_y0,grille_y1
      integer, dimension(:), allocatable :: indo_pres,indo,irff
      integer, dimension(:), pointer     :: ip1m
      logical flag_clos, write_diag_lev
      real mul,avgfact,period
      real, dimension (l_ni,l_nj,G_nk+1), target :: wlnpi_m,wlnpi_t
      real, dimension(:), pointer    :: hybm,hybt
      real, dimension(:), allocatable:: prprlvl,rff
      real, dimension(:,:,:), pointer :: lnpres,ptr3d
      real, dimension(:,:,:), allocatable         :: buso_pres,cible
      real, dimension(:,:,:), allocatable, target :: data3d
!
!----------------------------------------------------------------------
!
      if (outp_sorties(0,stepno).le.0) then
         if (Lun_out.gt.0) write(Lun_out,7002) stepno
         return
      else
         if (Lun_out.gt.0) write(Lun_out,7001) &
                           stepno,trim(Out_laststep_S)
      endif

      istat = fstopc('MSGLVL','SYSTEM',.false.)
      out_type_S   = 'REGPHY'

!     setup domain extent to retrieve physics data
      allocate ( data3d(l_ni,l_nj,G_nk+1), &
                 rff(Outp_multxmosaic), irff(Outp_multxmosaic))
      data3d = 0.
      
      p_li0= Grd_lphy_i0 ; p_li1=Grd_lphy_in
      p_lj0= Grd_lphy_j0 ; p_lj1=Grd_lphy_jn

      istat= gmm_get(gmmk_pw_log_pm_s, pw_log_pm)
      istat= gmm_get(gmmk_pw_log_pt_s, pw_log_pt)
      wlnpi_m(1:l_ni,1:l_nj,:)= pw_log_pm(1:l_ni,1:l_nj,:)
      wlnpi_t(1:l_ni,1:l_nj,:)= pw_log_pt(1:l_ni,1:l_nj,:)

!     Retrieeve vertical coordinate description
      nullify(ip1m,hybm,hybt)
      istat = vgrid_wb_get('ref-m',vcoord,ip1m)
      deallocate(ip1m); nullify(ip1m)
      if (vgd_get(vcoord,'VCDM - vertical coordinate (m)',hybm) /= VGD_OK) istat = VGD_ERROR
      if (vgd_get(vcoord,'VCDT - vertical coordinate (t)',hybt) /= VGD_OK) istat = VGD_ERROR

      do jj=1, outp_sorties(0,stepno)

         kk       = outp_sorties(jj,stepno)
         gridset  = Outp_grid(kk)
         levset   = Outp_lev(kk)
         avgfact  = 1./float(Outp_numstep(kk))
         ip2      = Out_ip2
         if (Outp_avg_L(kk).or.Outp_accum_L(kk)) then
             period = dble(Outp_numstep(kk)) * dble(Step_dt) / 3600.d0
             kind   = 10
             !call convip_plus ( ip2, period, kind, 2,dumc,.false. )
             call convip ( ip2, period, kind, 2,dumc,.false. )
         endif

         allocate ( indo( min(Level_max(levset),Level_momentum ) ) )
         
         call out_slev2 (Level(1,levset), Level_max(levset), &
                         Level_momentum,indo,nko,write_diag_lev)

         if (Level_typ_S(levset).eq.'P') then
            nko_pres = Level_max(levset)
            allocate ( indo_pres(nko_pres),buso_pres(l_ni,l_nj,nko_pres),&
                       prprlvl(nko_pres),cible(l_ni,l_nj,nko_pres) )
            buso_pres= 0.
            do i = 1, nko_pres
               indo_pres(i)= i
               prprlvl(i)   = level(i,levset) * 100.0
               cible(:,:,i) = log(prprlvl(i))
            enddo
         endif

         Out_prefix_S(1:1) = 'p'
         Out_prefix_S(2:2) = Level_typ_S(levset)
         call up2low (Out_prefix_S ,prefix)
         Out_reduc_l       = Grid_reduc(gridset)

         call out_open_file (trim(prefix))

         bcs_ext = 0
         if (G_lam) bcs_ext = Grd_bsc_ext1
         grille_x0 = max( 1   +bcs_ext, Grid_x0(gridset) )
         grille_x1 = min( G_ni-bcs_ext, Grid_x1(gridset) )
         grille_y0 = max( 1   +bcs_ext, Grid_y0(gridset) )
         grille_y1 = min( G_nj-bcs_ext, Grid_y1(gridset) )

         call out_href3 ( 'Mass_point',grille_x0,grille_x1,1,&
                                       grille_y0,grille_y1,1 )

         if (Level_typ_S(levset).eq.'M') then
            call out_vref (etiket=Out_etik_S)
         elseif (Level_typ_S(levset).eq.'P') then
            call out_vref (Level_allpres(1:Level_npres),&
                           etiket=Out_etik_S)
         endif

         PHYSICS_VARS: do ii=1, Outp_var_max(kk)

            WRITE_FIELD: if (phy_getmeta (pmeta, Outp_var_S(ii,kk), &
                             F_npath='O',F_bpath='PVE', F_quiet=.true.)&
                             > 0 ) then

               FIELD_SHAPE: if (pmeta%nk .eq. 1) then
!                 2D field

                  if ( pmeta%fmul.eq.1 .and. pmeta%mosaic.eq.0) then
                     ! 2D none multiple, none mosaic field, 0mb field, kind=2
                     ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,1:1)
                     istat = phy_get ( ptr3d, Outp_var_S(ii,kk), &
                                       F_npath='O', F_bpath='PV')
                     if (Outp_avg_L(kk)) data3d = data3d*avgfact
                     call out_fstecr3 ( data3d, 1,l_ni, 1,l_nj, 0.0, &
                             Outp_var_S(ii,kk),Outp_convmult(ii,kk), &
                             Outp_convadd(ii,kk), ip2,2, 1,1,1     , &
                             Outp_nbit(ii,kk),.false. )
                     if (Outp_avg_L(kk) .or.Outp_accum_L(kk)) then
                         data3d = 0.0
                         ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,1:1)
                         istat = phy_put ( ptr3d, Outp_var_S(ii,kk), &
                                           F_npath='O', F_bpath='PV')
                     endif
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
                     istat = phy_get (ptr3d,Outp_var_S(ii,kk), &
                                      F_npath='O',F_bpath='PV')
                     if (Outp_avg_L(kk)) data3d = data3d*avgfact
                     call out_fstecr3 ( data3d, 1,l_ni, 1,l_nj, rff ,&
                             Outp_var_S(ii,kk),Outp_convmult(ii,kk) ,&
                             Outp_convadd(ii,kk), ip2,3,cnt,irff,cnt,&
                             Outp_nbit(ii,kk),.false. )
                     if (Outp_avg_L(kk).or.Outp_accum_L(kk)) then
                         data3d = 0.0
                         ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,1:cnt)
                         istat = phy_put (ptr3d,Outp_var_S(ii,kk), &
                                          F_npath='O',F_bpath='PV')
                     endif
                  endif
               else
                  ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,:)
                  istat = phy_get (ptr3d,Outp_var_S(ii,kk),F_npath='O', &
                                   F_bpath='PV')
                  if (Outp_avg_L(kk)) data3d = data3d*avgfact

                  if (Level_typ_S(levset).eq.'M') then

                     if (pmeta%stag .eq. 1) then ! thermo
                        call out_fstecr3 (data3d                       ,&
                                 1,l_ni, 1,l_nj, hybt                  ,&
                                 Outp_var_S(ii,kk),Outp_convmult(ii,kk),&
                                 Outp_convadd(ii,kk),ip2,Level_kind_ip1,&
                                 G_nk,indo,nko,Outp_nbit(ii,kk),.false. )
                        if (write_diag_lev) then
                           call out_fstecr3 (data3d(1,1,G_nk+1)        ,&
                                 1,l_ni, 1,l_nj, hybt(G_nk+2)          ,&
                                 Outp_var_S(ii,kk),Outp_convmult(ii,kk),&
                                 Outp_convadd(ii,kk),ip2,Level_kind_diag,&
                                 1,1,1,Outp_nbit(ii,kk),.false. )
                        endif
                     else  ! momentum
                        call out_fstecr3 (data3d                       ,&
                                 1,l_ni, 1,l_nj, hybm                  ,&
                                 Outp_var_S(ii,kk),Outp_convmult(ii,kk),&
                                 Outp_convadd(ii,kk),ip2,Level_kind_ip1,&
                                 G_nk,indo,nko,Outp_nbit(ii,kk),.false. )
                        if (write_diag_lev) then
                           call out_fstecr3 (data3d(1,1,G_nk+1)        ,&
                                 1,l_ni, 1,l_nj, hybm(G_nk+2)          ,&
                                 Outp_var_S(ii,kk),Outp_convmult(ii,kk),&
                                 Outp_convadd(ii,kk),ip2,Level_kind_diag,&
                                 1,1,1,Outp_nbit(ii,kk),.false. )
                        endif
                     endif

                  elseif (Level_typ_S(levset).eq.'P') then

                     lnpres => wlnpi_m
                     if ( pmeta%stag .eq. 1 ) lnpres => wlnpi_t

                     call vertint ( buso_pres, cible, nko_pres, data3d,&
                                    lnpres, G_nk, 1,l_ni, 1,l_nj      ,&
                                    1,l_ni, 1,l_nj, 'linear', .false. )

                     call out_fstecr3 ( buso_pres, 1,l_ni, 1,l_nj      ,&
                           level(1,levset),Outp_var_S(ii,kk)           ,&
                           Outp_convmult(ii,kk),Outp_convadd(ii,kk),ip2,2,&
                           nko_pres,indo_pres,nko_pres,Outp_nbit(ii,kk),&
                           .false. )

                  endif
                  if (Outp_avg_L(kk).or.Outp_accum_L(kk)) then
                      data3d = 0.0
                      ptr3d => data3d(p_li0:p_li1,p_lj0:p_lj1,:)
                      istat = phy_put (ptr3d,Outp_var_S(ii,kk),F_npath='O', &
                                       F_bpath='PV')
                  endif
               endif FIELD_SHAPE
            endif WRITE_FIELD
         end do PHYSICS_VARS

         deallocate (indo)
         if (Level_typ_S(levset).eq.'P') deallocate (buso_pres, indo_pres, prprlvl, cible)
         
         flag_clos= .true.
         if (jj .lt. outp_sorties(0,stepno)) then
            flag_clos= .not.( (gridset.eq.Outp_grid(outp_sorties(jj+1,stepno))).and. &
                 (Level_typ_S(levset).eq.Level_typ_S(Outp_lev(outp_sorties(jj+1,stepno)))))
         endif
         
         if (flag_clos) call out_cfile3
         
      end do
      
      deallocate(rff,irff,data3d)
      deallocate(hybm,hybt); nullify(hybm,hybt)

      istat = fstopc('MSGLVL','INFORM',.false.)

 7001 format(/,' OUT_PHY- WRITING PHYSICS OUTPUT FOR STEP (',I8,') in directory: ',a)
 8001 format(/,' OUT_PHY- WRITING CASCADE OUTPUT FOR STEP (',I8,') in directory: ',a)
 7002 format(/,' OUT_PHY- NO PHYSICS OUTPUT FOR STEP (',I8,')')
!
!----------------------------------------------------------------------
!
 999  return
      end


