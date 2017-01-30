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

      subroutine indata
      implicit none
#include <arch_specific.hf>

#include "gmm.hf"
#include "grd.cdk"
#include "schm.cdk"
#include "glb_ld.cdk"
#include "p_geof.cdk"
#include "lctl.cdk"
#include "vt1.cdk"
#include "lun.cdk"
#include "perturb.cdk"
#include "step.cdk"
#include "tr3d.cdk"
#include "inp.cdk"
#include "pw.cdk"

      integer i,j,k,istat,dim,err
      real, dimension(:,:,:), pointer :: plus,minus
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)

      istat = gmm_get (gmmk_pw_uu_plus_s, pw_uu_plus)
      istat = gmm_get (gmmk_pw_vv_plus_s, pw_vv_plus)
      istat = gmm_get (gmmk_pw_tt_plus_s, pw_tt_plus)
      istat = gmm_get (gmmk_ut1_s ,ut1 )
      istat = gmm_get (gmmk_vt1_s ,vt1 )
      istat = gmm_get (gmmk_wt1_s ,wt1 )
      istat = gmm_get (gmmk_tt1_s ,tt1 )
      istat = gmm_get (gmmk_zdt1_s,zdt1)
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
         dim=(l_maxx-l_minx+1)*(l_maxy-l_miny+1)*G_nk
         call inp_data ( pw_uu_plus,pw_vv_plus,wt1,pw_tt_plus,&
                         zdt1,st1,qt1,fis0                   ,&
                         l_minx,l_maxx,l_miny,l_maxy,G_nk    ,&
                         .false. ,'TR/',':P',Step_runstrt_S )
         call bitflip ( pw_uu_plus, pw_vv_plus, pw_tt_plus, &
                        perturb_nbits, perturb_npts, dim )
         call timing_stop  ( 71 )
      endif

      call gemtim4 ( Lun_out, 'AFTER INITIAL INPUT', .false. )

      call set_dync ( .true., err )

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

      call tt2virt2 (tt1, .true., l_minx,l_maxx,l_miny,l_maxy, G_nk)
      call hwnd_stag ( ut1,vt1, pw_uu_plus,pw_vv_plus,&
                       l_minx,l_maxx,l_miny,l_maxy,G_nk,.true. )

      call diag_zd_w2 ( zdt1,wt1, ut1,vt1,tt1,st1   ,&
                        l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                        .not.Inp_zd_L, .not.Inp_w_L )

      if (.not. Grd_yinyang_L) call nest_init ()

      call pw_update_GPW
      call pw_init

      call out_outdir (Step_total)
      call iau_apply2 (0)

      if ( Schm_phyms_L ) call itf_phy_step (0,Lctl_step)

      call frstgss

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

      integer n,nbits,npts
      integer u(n),v(n),t(n)

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
