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

!**s/r indata - Read and process the input data at 
!               beginning of integration
!
      subroutine indata
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Roch - rpn - apr 1994
!
!revision
! v2_00 - Desgagne M.       - initial MPI version (from indata v1_03)
! v2_10 - Tanguay M.        - introduce partition of preprocessing when 4D-Var 
! v2_20 - Pellerin P.       - read geophysical fields depending on schemes
! v2_20 - Lee V.            - eliminated p_slicgeo, output of geophysical fields
! v2_20 -                     will be from the entry or permanent physics bus
! v2_30 - Desgagne M.       - entry vertical interpolator in gemdm
! v2_31 - Tanguay M.        - adapt for vertical hybrid coordinate 
! v3_02 - Buehner M.        - leave winds as images for 4dvar or SV jobs
! v3_03 - Tanguay M.        - Adjoint Lam configuration 
! v3_11 - Gravel S.         - Adapt for theoretical cases and varying topo
! v3_30 - Desgagne M.       - re-organize code to eliminate v4d controls
! v3_30 - Lee V.            - new LAM I/O interface
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_03 - Tanguay M.        - Williamson's cases
! v4_03 - Lee/Desgagne - ISST
! v4_06 - Lepine M.         - VMM replacement with GMM
! v4_06 - Lee V.            - predat is called after readdyn, casc_3df_dynp
! v4_10 - Tanguay M.        - VMM replacement with GMM for (TL/AD)
! v4_12 - Tanguay M.        - Adapt to revised predat
! v4_13 - Spacek L.         - Delete call to readgeo, add Path_phy_S
! v4_13 - Tanguay M.        - Adjustments GEM413 .not. Schm_hydro_L
! v4_21 - Plante A.         - Call predat4     
! v4_40 - Lee/Qaddouri      - Add exchange of ME for Yin-Yang seam
! v4_80 - Tanguay M.        - Add check_tracers

#include "gmm.hf"
#include "grd.cdk"
#include "acq.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "p_geof.cdk"
#include "vt1.cdk"
#include "lun.cdk"
#include "perturb.cdk"
#include "step.cdk"
#include "tr3d.cdk"
#include "bcsgrds.cdk"

      integer i,j,k,istat,dim
      real, dimension(:,:,:), pointer :: plus,minus
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)

      istat = gmm_get (gmmk_ut1_s ,ut1 )
      istat = gmm_get (gmmk_vt1_s ,vt1 )
      istat = gmm_get (gmmk_wt1_s ,wt1 )
      istat = gmm_get (gmmk_tt1_s ,tt1 )
      istat = gmm_get (gmmk_zdt1_s,zdt1)
      istat = gmm_get (gmmk_xdt1_s,xdt1)
      istat = gmm_get (gmmk_st1_s ,st1 )
      istat = gmm_get (gmmk_fis0_s,fis0)
      istat = gmm_get (gmmk_qt1_s ,qt1 )

      zdt1=0. ; wt1=0. ; qt1= 0.

      if ( Schm_theoc_L ) then
         call theo_3D_2 ( ut1,vt1,wt1,tt1,zdt1,st1,qt1,fis0,&
                                                  'TR/',':P') 
      elseif ( Schm_autobar_L ) then
         call init_bar ( ut1,vt1,wt1,tt1,zdt1,st1,qt1,fis0,&
                               l_minx,l_maxx,l_miny,l_maxy,&
                            G_nk,'TR/',':P',Step_runstrt_S )
      else
         call timing_start2 ( 71, 'INITIAL_input', 2)
         istat= gmm_get (gmmk_topo_low_s , topo_low )
         istat= gmm_get (gmmk_topo_high_s, topo_high)
         call get_topo2 ( topo_high, l_minx,l_maxx,l_miny,l_maxy, &
                          1,l_ni,1,l_nj )
         topo_low(1:l_ni,1:l_nj) = topo_high(1:l_ni,1:l_nj)
         call inp_data ( ut1,vt1,wt1,tt1,zdt1,st1,qt1,fis0,&
                               l_minx,l_maxx,l_miny,l_maxy,&
                            G_nk,'TR/',':P',Step_runstrt_S )
         call timing_stop  ( 71 )
      endif

      if (Schm_adxlegacy_L) then
         call adx_check_tracers
      else
         call adv_check_tracers
      endif

      dim=(l_maxx-l_minx+1)*(l_maxy-l_miny+1)*G_nk
      call bitflip (ut1, vt1, tt1, perturb_nbits, perturb_npts, dim)

      call gemtim4 ( Lun_out, 'AFTER INITIAL INPUT', .false. )

      call set_dync

      if (Grd_yinyang_L) then
         call yyg_xchng (fis0, l_minx,l_maxx,l_miny,l_maxy, &
                         1, .false., 'CUBIC')
         call rpn_comm_xch_halo(fis0, l_minx,l_maxx,l_miny,l_maxy,&
            l_ni,l_nj,1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
         call yyg_xchng_all
      else
         call rpn_comm_xch_halo(fis0, l_minx,l_maxx,l_miny,l_maxy,&
            l_ni,l_nj,1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      endif

      do k=1, Tr3d_ntr
         nullify (plus, minus)
         istat = gmm_get('TR/'//trim(Tr3d_name_S(k))//':M',minus)
         istat = gmm_get('TR/'//trim(Tr3d_name_S(k))//':P',plus )
         minus = plus
      enddo

      call diag_zd_w2 ( zdt1,wt1,xdt1, ut1,vt1,tt1,st1   ,&
                        l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                        .not.Ana_zd_L, .not.Ana_w_L )

      if ( Grd_yinyang_L ) then
         call yyg_blend (Schm_nblendyy.ge.0)
      else
         if ( G_lam ) call nest_init ()
      endif
         
      if (.not.Acql_pwuv) call pw_update_UV
      if (.not.Acql_pwtt) call pw_update_T
      call pw_update_GPW

      call frstgss ()

      call psadj_init

      call glbstat2 ( fis0,'ME',"indata",l_minx,l_maxx,l_miny,l_maxy, &
                      1,1, 1,G_ni,1,G_nj,1,1 )
!
!     ---------------------------------------------------------------
!
 1000 format(/,'TREATING INITIAL CONDITIONS  (S/R INDATA)',/,41('='))
 1002 format(/,' FILE ',A,'_gfilemap.txt IS NOT AVAILABLE --CONTINUE--',/,/)

      return
      end

      subroutine bitflip (u,v,t,nbits,npts,n)
      implicit none
!
      integer n,nbits,npts
      integer u(n),v(n),t(n)
!
!author   Michel Desgagne -- summer 2012
!
!revision
! v4.50 - M. Desgagne       - initial version
!
      integer stride,i
!
! ---------------------------------------------------------------------
!
      if (nbits .lt. 1) return
      stride = min(max(1,npts),n)
      do i=1,n,stride
         u(i) = xor(u(i),nbits)
         v(i) = xor(v(i),nbits)
         t(i) = xor(t(i),nbits)
      end do
!
! ---------------------------------------------------------------------
!
      return
      end
