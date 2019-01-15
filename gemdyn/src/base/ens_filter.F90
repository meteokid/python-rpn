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

!**s/r ens_filter - Fourier filter
!
      subroutine ens_filter (F_ugwdt1,F_vgwdt1,F_difut1,F_difvt1, &
                             F_ut1,F_vt1,F_tt1, Minx,Maxx,Miny,Maxy, Nk)
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy, Nk
      real    F_ugwdt1(Minx:Maxx,Miny:Maxy,Nk), F_vgwdt1(Minx:Maxx,Miny:Maxy,Nk)
      real    F_difut1(Minx:Maxx,Miny:Maxy,Nk), F_difvt1(Minx:Maxx,Miny:Maxy,Nk)
      real    F_ut1(Minx:Maxx,Miny:Maxy,Nk), F_vt1(Minx:Maxx,Miny:Maxy,Nk), F_tt1(Minx:Maxx,Miny:Maxy,Nk)

!author
!     Lubos Spacek - rpn - apr 2005
!
!revision
! v4_12 - Spacek L.         - staggered + gmm version
! v4.1.3 -N. Gagnon         - Change name of some parameters from NAMELIST

#include "gmm.hf"
#include "var_gmm.cdk"

#include "glb_ld.cdk"
#include "geomg.cdk"
#include "dcst.cdk"
#include "grd.cdk"

#include "ens_gmm_var.cdk"
#include "ens_param.cdk"

      integer ix,iy,iz,iz_local,kx,ky,kz, E_nk
      integer np, rank, ierr, keys(8), npartiel
      integer i, j, k, i0, j0, in, jn, nij, inn, trx
      real    pi, err, deltax, cpdi, dummy
      real*8  dpi, aaa
      real  , dimension(:,:,:), allocatable  :: dsp_local
      real  , dimension(:,:,:), allocatable  :: dsp_dif, dsp_gwd
      real  , dimension(l_ni,l_nj)  :: fgem
      type(gmm_metadata) :: meta3d
      integer :: gmmstat
!
!     ---------------------------------------------------------------
!
      E_nk = Nk
!
      dpi=4.d0*atan(1.0d0)
      cpdi=1./real(Dcst_cpd_8)
      aaa= Dcst_rayt_8**2
      deltax=sqrt(((2.0*dpi/real(G_ni))*Dcst_rayt_8)* &
                     ((dpi/real(G_nj))*Dcst_rayt_8))
!
!     Get needed fields in memory
!
      gmmstat = gmm_get(gmmk_mcsph1_s ,mcsph1,meta3d)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'mcsph11'
!
!     Markov chain step and if Ens_skeb_conf=.false. return
!


    call ens_marfield_ptp

   if (Ens_skeb_conf) then
     call ens_marfield_skeb (fgem)

     do k=1,E_nk
     mcsph1(:,:,k)=fgem(:,:)
     enddo

   else 
     return
   endif
  

      allocate( dsp_local(l_minx:l_maxx,l_miny:l_maxy,E_nk))
      allocate( dsp_dif(l_minx:l_maxx,l_miny:l_maxy,E_nk))
      allocate( dsp_gwd(l_minx:l_maxx,l_miny:l_maxy,E_nk))

      call rpn_comm_xch_halo (F_ut1,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_vt1,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_difut1,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_difvt1,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_ugwdt1,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_vgwdt1,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

!
!     Calculate kinetic energy of diffusion tendency
!     ===============================================

      i0 = 1
      in = G_ni
      j0 = 1
      jn = G_nj
      inn= 0
      if (G_lam) then
           inn=1
      endif
!
!     Diffusion backscatter
!
      if(Ens_stat)then
         call glbstat2 (F_ut1,'URT1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_vt1,'VRT1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difut1,'DUT1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difvt1,'DVT1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_ugwdt1,'UGW1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_vgwdt1,'VGW1','AT BEG', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      dsp_dif=0.0
      if(Ens_skeb_dif)then
        F_difut1=F_ut1*F_difut1 ; F_difvt1=F_vt1*F_difvt1
        call ens_uvduv (dsp_dif, F_difut1, F_difvt1, l_minx,l_maxx,l_miny,l_maxy, E_nk )
      endif

      if(Ens_stat)then
         call glbstat2 (F_difut1,'DUT1','DIFF', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difvt1,'DVT1','DIFF', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif
!
!     Gravity wave drag backscatter
!
      dsp_gwd=0.0
      if(Ens_skeb_gwd)then
        F_difut1=F_ut1*F_ugwdt1 ; F_difvt1=F_vt1*F_vgwdt1
        call ens_uvgwd (dsp_gwd, F_difut1, F_difvt1, l_minx,l_maxx,l_miny,l_maxy, E_nk )
      endif
!
      if(Ens_stat)then
         call glbstat2 (F_difut1,'DUT1','GWD', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difvt1,'DVT1','GWD', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      dsp_local=dsp_dif+dsp_gwd

!
!     Apply 2D Gaussian filter
!     ===================================================================
      if(Ens_stat)then
          call glbstat2 (dsp_dif,'DSP','DIF', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
          call glbstat2 (dsp_gwd,'DSP','GWD', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
          call glbstat2 (dsp_local,'DSP','TOT', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      if(Grd_yinyang_L) then
         call ens_filter_ggauss(dble(Ens_skeb_bfc),dble(Ens_skeb_lam),dsp_local)
      else
         call ens_filter_gauss(dble(Ens_skeb_bfc),dble(Ens_skeb_lam),dsp_local)
      endif

      if(Ens_stat)then
         call glbstat2 (dsp_local,'DSP','FLTTOT', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      if(Ens_skeb_alpt/=0.0)then
         F_tt1(1:l_ni,1:l_nj,1:E_nk)=F_tt1(1:l_ni,1:l_nj,1:E_nk)+&
         Ens_skeb_alpt*cpdi* &
         dsp_local(1:l_ni,1:l_nj,:)*mcsph1(1:l_ni,1:l_nj,1:E_nk)
      endif
!
       dsp_local(1:l_ni,1:l_nj,:)=sqrt(dsp_local(1:l_ni,1:l_nj,:))* &
                                 mcsph1(1:l_ni,1:l_nj,1:E_nk)

      if(Ens_stat)then
         call glbstat2 (dsp_local,'DSP','FLTTO2', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      call rpn_comm_xch_halo (dsp_local,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,E_nk, &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

!      call rpn_comm_Barrier("grid", ierr)

!     Compute gradient of filtered field
!     ==================================

      i0 = 1
      in = l_niu
      j0 = 1
      jn = l_njv

      F_difut1(:,:,E_nk)=0.0;F_difvt1(:,:,E_nk)=0.0;

      do k=1,G_nk
         do j= j0, jn
            do i= i0, l_ni
               F_difut1(i,j,k) = (dsp_local(i,j,k)-dsp_local(i-1,j,k))* &
                                  Geomg_invDXv_8(j)
               F_difut1(i,j,k) = Ens_skeb_alph*deltax*F_difut1(i,j,k)
               F_vt1(i,j,k)    = F_vt1(i,j,k)+F_difut1(i,j,k)
            enddo
         enddo

         do j= j0, l_nj !-pil_n
            do i= i0, in
               F_difvt1(i,j,k) = (dsp_local(i,j,k) - dsp_local(i,j-1,k)) * &
                                  Geomg_invDY_8
               F_difvt1(i,j,k)= -Ens_skeb_alph*deltax*F_difvt1(i,j,k)
               F_ut1(i,j,k) = F_ut1(i,j,k)+F_difvt1(i,j,k)
            enddo
         enddo
      enddo

      if(Ens_stat)then
         call glbstat2 (F_ut1,'URT1','AT END', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_vt1,'VRT1','AT END', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difut1,'DUT1','AT END', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (F_difvt1,'DVT1','AT END', &
           l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      if(Ens_skeb_div)then
         call rpn_comm_xch_halo (F_difut1,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
         call rpn_comm_xch_halo (F_difvt1,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv,E_nk, &
                   G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!
      gmmstat = gmm_get(gmmk_ensdiv_s ,ensdiv,meta3d)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ensvor'
      gmmstat = gmm_get(gmmk_ensvor_s ,ensvor,meta3d)
      if (GMM_IS_ERROR(gmmstat))write(*,6000)'ensvor'

! This call to cal_ddqq is wrong because F_difvt1, F_difut1 are inverted
! It has anyways been replaced by calls to cal_div and cal_vor and it
! definitively needs to be checked.


!      call cal_ddqq ( ensdiv, ensvor,dummy,F_difvt1, F_difut1, &
!                      0,dummy,0,dummy,.true.,.true.,.false., &
!                      Minx,Maxx,Miny,Maxy, E_nk )

      call cal_div ( ensdiv, F_difut1, F_difvt1, 0, dummy,&
                     l_minx,l_maxx,l_miny,l_maxy, E_nk )
      call cal_vor ( ensvor, dummy, F_difut1, F_difvt1, 0, dummy, .false., &
                     l_minx,l_maxx,l_miny,l_maxy, E_nk )

     ! call gem_error(-1,'ens_filter','Must check calls to cal_div and cal_vor')


      if(Ens_stat)then
         call glbstat2 (ensdiv,'DVRG','FRCING', &
              l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
         call glbstat2 (ensvor,'VORT','FRCING', &
              l_minx,l_maxx,l_miny,l_maxy,1,E_nk,1,G_ni,1,G_nj,1,E_nk)
      endif

      endif

      deallocate(dsp_dif,dsp_gwd,dsp_local)

 6000 format('ens_filter at gmm_get(',A,')')
!
!     ---------------------------------------------------------------
!
      end subroutine ens_filter
