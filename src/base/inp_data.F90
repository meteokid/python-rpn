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

!**s/r inp_data  - Reads FST input files

      subroutine inp_data ( F_u, F_v, F_w, F_t, F_zd, F_s, F_q, F_topo,&
                            Mminx,Mmaxx,Mminy,Mmaxy, Nk               ,&
                            F_trprefix_S, F_trsuffix_S, F_datev )
      use vgrid_wb, only: vgrid_wb_get
      use vGrid_Descriptors
      use nest_blending, only: nest_blend
      implicit none
#include <arch_specific.hf>

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      integer Mminx,Mmaxx,Mminy,Mmaxy,Nk
      real F_u (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_v (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_w (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_t (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_zd(Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_s (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q (Mminx:Mmaxx,Mminy:Mmaxy,2:Nk+1), &
           F_topo(Mminx:Mmaxx,Mminy:Mmaxy)

#include <WhiteBoard.hf>
#include "gmm.hf"
#include "glb_ld.cdk"
#include "dcst.cdk"
#include "cstv.cdk"
#include "bcsgrds.cdk"
#include "lam.cdk"
#include "schm.cdk"
#include "tr3d.cdk"
#include "inp.cdk"
#include "lun.cdk"
#include "ver.cdk"
#include "step.cdk"
#include "p_geof.cdk"
#include "vtopo.cdk"
#include "ptopo.cdk"
#include <rmnlib_basics.hf>

Interface
      integer function inp_read ( F_var_S, F_hgrid_S, F_dest, &
                                  F_ip1, F_nka )
      implicit none
      character*(*)                     , intent(IN)  :: F_var_S,F_hgrid_S
      integer                           , intent(OUT) :: F_nka
      integer, dimension(:    ), pointer, intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer, intent(OUT) :: F_dest
      End function inp_read
End Interface

      integer, external :: inp_get
      character(len=4) vname
      logical initial_data, blend_oro
      integer fst_handle, i,j,n, nia,nja,nka, istat, err
      integer, dimension (:), pointer :: ip1_list, ip1_dum
      real topo_temp(l_minx:l_maxx,l_miny:l_maxy),step_current,&
           p0(1:l_ni,1:l_nj)
      real, dimension (:,:  ), allocatable :: ssu0,ssv0
      real, dimension (:,:  ), pointer     :: ssq0,pres
      real, dimension (:,:,:), allocatable, target :: dstlev,srclev
      real, dimension (:,:,:), pointer :: ssqr,ssur,ssvr,&
                                          meqr,ttr,hur,trp,ptr3d
      real*8 diffd, pref_a_8
      type(vgrid_descriptor) :: vgd_src, vgd_dst
!
!-----------------------------------------------------------------------
!
      if (Lun_out.gt.0) write(lun_out,9000) trim(F_datev)

      call timing_start2 ( 71, 'FST_input', 2)

      if (.not.Lun_debug_L) istat= fstopc ('MSGLVL','SYSTEM',.false.)

      initial_data= trim(F_datev).eq.trim(Step_runstrt_S)

      call inp_open ( F_datev, vgd_src  )

      nullify (meqr,ip1_list)
      err= inp_read ( 'OROGRAPHY', 'Q', meqr, ip1_list, nka )

      if ( initial_data .and. (Step_kount.eq.0) ) then
         call get_topo2 ( F_topo,l_minx,l_maxx,l_miny,l_maxy,&
                          1,l_ni,1,l_nj )
         blend_oro= Lam_blendoro_L
         if ( .not. associated(meqr) ) then
            blend_oro= .false. ; Vtopo_L= .false.
         endif
         if (G_lam .and. blend_oro) then
            topo_temp(1:l_ni,1:l_nj) = meqr(1:l_ni,1:l_nj,1)
            call nest_blend ( F_topo, topo_temp, l_minx,l_maxx,l_miny,l_maxy,&
                             'M', level=G_nk+1 )
         endif
         call rpn_comm_xch_halo ( F_topo, l_minx,l_maxx,l_miny,l_maxy,&
                l_ni,l_nj,1, G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      endif

      if (Vtopo_L) then
         if ( initial_data .and. (Step_kount.eq.0) ) then
            istat = gmm_get(gmmk_topo_low_s , topo_low )
            istat = gmm_get(gmmk_topo_high_s, topo_high)
            topo_low (1:l_ni,1:l_nj) = meqr  (1:l_ni,1:l_nj,1)
            topo_high(1:l_ni,1:l_nj) = F_topo(1:l_ni,1:l_nj)
         endif
         call difdatsd (diffd,Step_runstrt_S,F_datev)
         step_current = diffd*86400.d0/dble(step_dt)
         call var_topo2 ( topo_temp,step_current,&
                          l_minx,l_maxx,l_miny,l_maxy )
      else
         topo_temp(1:l_ni,1:l_nj) = F_topo(1:l_ni,1:l_nj)
      endif

      nullify (ssqr,ssur,ssvr,ttr,hur)
      err= inp_read ( 'SFCPRES'    , 'Q', ssqr, ip1_list, nka )
      err= inp_read ( 'SFCPRES'    , 'U', ssur, ip1_list, nka )
      err= inp_read ( 'SFCPRES'    , 'V', ssvr, ip1_list, nka )
      err= inp_read ( 'TEMPERATURE', 'Q', ttr , ip1_list, nka )
      err= inp_read ( 'TR/HU'      , 'Q', hur , ip1_list, nka )

      if (Inp_kind /= 105) &
      call mfottv2 ( ttr, ttr, hur, l_minx,l_maxx,l_miny,l_maxy,&
                     nka, 1,l_ni,1,l_nj, .true. )

      allocate ( srclev(l_minx:l_maxx,l_miny:l_maxy,nka) ,&
                 dstlev(l_minx:l_maxx,l_miny:l_maxy,G_nk),&
                 ssq0  (l_minx:l_maxx,l_miny:l_maxy)     ,&
                 ssu0  (l_minx:l_maxx,l_miny:l_maxy)     ,&
                 ssv0  (l_minx:l_maxx,l_miny:l_maxy) )

      if (associated(ssqr).and.associated(meqr)) then
         pres  => ssqr(1:l_ni,1:l_nj,1)
         ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels (vgd_src, ip1_list, ptr3d, pres)
         srclev(1:l_ni,1:l_nj,nka)= ssqr(1:l_ni,1:l_nj,1)
         call adj_ss2topo2 ( ssq0, topo_temp, srclev,meqr,ttr, &
                             l_minx,l_maxx,l_miny,l_maxy, nka, &
                             1,l_ni,1,l_nj )
         deallocate (meqr) ; nullify (meqr)
      else
         if (lun_out.gt.0) &
         write(lun_out,'(" NO surface pressure adjustment")')
         ssq0(1:l_ni,1:l_nj)= ssqr(1:l_ni,1:l_nj,1)
      endif

      call rpn_comm_xch_halo ( ssq0, l_minx,l_maxx,l_miny,l_maxy,&
           l_ni,l_nj,1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

      do j=1,l_nj
      do i=1,l_ni-east
         ssu0(i,j)= (ssq0(i,j)+ssq0(i+1,j  ))*.5d0
      enddo
      enddo
      if (l_east) ssu0(l_ni,1:l_nj)= ssq0(l_ni,1:l_nj)

      do j=1,l_nj-north
      do i=1,l_ni
         ssv0(i,j)= (ssq0(i,j)+ssq0(i  ,j+1))*.5d0
      enddo
      enddo
      if (l_north) ssv0(1:l_ni,l_nj)= ssq0(1:l_ni,l_nj)

      F_s(1:l_ni,1:l_nj) = log(ssq0(1:l_ni,1:l_nj)/Cstv_pref_8)

      nullify (ip1_dum)
      istat= vgrid_wb_get ('ref-m', vgd_dst, ip1_dum )
      deallocate (ip1_dum); nullify (ip1_dum)

      do n= 1, nka-1
         srclev(1:l_ni,1:l_nj,n) = log(srclev(1:l_ni,1:l_nj,n))
      end do
      srclev(1:l_ni,1:l_nj,nka) = log(ssqr(1:l_ni,1:l_nj,1))

      pres  => ssq0  (1:l_ni,1:l_nj)
      ptr3d => dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels ( vgd_dst, Ver_ip1%t(1:G_nk), ptr3d, &
                          pres, in_log=.true. )
      call vertint ( F_t,dstlev,G_nk, ttr,srclev,nka, &
                     l_minx,l_maxx,l_miny,l_maxy    , &
                     1,l_ni, 1,l_nj, 'cubic', .true. )
      nullify (trp)
      istat= gmm_get (trim(F_trprefix_S)//'HU'//trim(F_trsuffix_S),trp)
      call vertint ( trp,dstlev,G_nk, hur,srclev,nka, &
                     l_minx,l_maxx,l_miny,l_maxy    , &
                     1,l_ni, 1,l_nj, 'cubic', .true. )
      deallocate (ttr,hur,srclev,dstlev) ; nullify (ttr,hur)
      if (associated(ip1_list)) deallocate (ip1_list)

      NTR_Tr3d_ntr= 0
      do n=1,Tr3d_ntr
         nullify (trp)
         vname= trim(Tr3d_name_S(n))
         if (trim(vname) == 'HU') cycle
         istat= gmm_get (&
               trim(F_trprefix_S)//trim(vname)//trim(F_trsuffix_S),trp)
         err= inp_get ( 'TR/'//trim(vname),'Q', Ver_ip1%t,&
                        vgd_src, vgd_dst, ssqr, ssq0, trp,&
                        l_minx,l_maxx,l_miny,l_maxy,G_nk )
         if (err == 0) then
            NTR_Tr3d_ntr= NTR_Tr3d_ntr + 1
            NTR_Tr3d_name_S(NTR_Tr3d_ntr) = trim(vname)
         endif
      end do

      ana_w_L = (inp_get ('WT1',  'Q', Ver_ip1%t     ,&
                       vgd_src,vgd_dst,ssqr,ssq0,F_w ,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk) == 0)

      ana_zd_L= (inp_get ('ZDT1', 'Q', Ver_ip1%t     ,&
                       vgd_src,vgd_dst,ssqr,ssq0,F_zd,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk) == 0)

      if (.not.Schm_hydro_L) &
      err= inp_get ( &
             'QT1', 'Q', Ver_ip1%m(2:G_nk+1)        ,&
                      vgd_src,vgd_dst,ssqr,ssq0,F_q ,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk )

      if (Inp_kind == 105) then

         err= inp_get ( 'UT1', 'U', Ver_ip1%m          ,&
                         vgd_src,vgd_dst,ssur,ssu0,F_u ,&
                         l_minx,l_maxx,l_miny,l_maxy,G_nk )

         err= inp_get ( 'VT1', 'V', Ver_ip1%m          ,&
                         vgd_src,vgd_dst,ssvr,ssv0,F_v ,&
                         l_minx,l_maxx,l_miny,l_maxy,G_nk )
      else

         call inp_hwnd ( F_u,F_v, vgd_src,vgd_dst, ssur,ssvr, &
                         ssu0,ssv0, l_minx,l_maxx,l_miny,l_maxy,G_nk )

      endif

      deallocate (ssqr,ssur,ssvr,ssq0,ssu0,ssv0)

      call inp_close ()

      istat = fstopc ('MSGLVL','INFORM',.false.)
      call timing_stop  ( 71 )

 9000 format(/,' TREATING INPUT DATA VALID AT: ',a,&
             /,' ===============================================')
!
!-----------------------------------------------------------------------
!
      return
      end

