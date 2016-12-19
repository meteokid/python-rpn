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
                            F_stag_L,F_trprefix_S,F_trsuffix_S,F_datev )
      use vgrid_wb, only: vgrid_wb_get
      use vGrid_Descriptors
      use vertical_interpolation, only: vertint2
      use nest_blending, only: nest_blend
      use inp_base, only: inp_get, inp_read
      implicit none
#include <arch_specific.hf>

      character* (*) F_trprefix_S, F_trsuffix_S, F_datev
      logical F_stag_L
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
#include "lam.cdk"
#include "schm.cdk"
#include "tr3d.cdk"
#include "inp.cdk"
#include "lun.cdk"
#include "ver.cdk"
#include "step.cdk"
#include "p_geof.cdk"
#include "vtopo.cdk"
#include "grd.cdk"
#include <rmnlib_basics.hf>

      character(len=4) vname
      logical urt1_l, ut1_l, sfcTT_L
      integer i,j,k,n, nka, istat, err, kind,kt,kh
      integer nka_gz, nka_tt, nka_hu, ip1_w(G_nk)
      integer, dimension (:), pointer :: ip1_list, HU_ip1_list, ip1_dum
      real topo_temp(l_minx:l_maxx,l_miny:l_maxy),step_current,&
           p0(1:l_ni,1:l_nj), lev
      real, dimension (:    ), allocatable :: rna
      real, dimension (:,:  ), allocatable :: ssu0,ssv0
      real, dimension (:,:  ), pointer     :: ssq0,pres
      real, dimension (:,:,:), allocatable, target :: dstlev,srclev
      real, dimension (:,:,:), pointer :: ssqr,ssur,ssvr,meqr,&
                                          tv,ttr,hur,gzr,trp,ptr3d
      real*8 diffd, pref_a_8
      type(vgrid_descriptor) :: vgd_src, vgd_dst
!
!-----------------------------------------------------------------------
!
      if (Lun_out.gt.0) write(lun_out,9000) trim(F_datev)

      if (.not.Lun_debug_L) istat= fstopc ('MSGLVL','SYSTEM',.false.)

      call inp_open ( F_datev, vgd_src )

      nullify (meqr, ip1_list, HU_ip1_list)
      err= inp_read ( 'OROGRAPHY', 'Q', meqr, ip1_list, nka )

      if ( trim(F_datev) .eq. trim(Step_runstrt_S) ) then
         if ( associated(meqr) ) then
            istat= gmm_get(gmmk_topo_low_s , topo_low )
            topo_low(1:l_ni,1:l_nj)= meqr(1:l_ni,1:l_nj,1)
         else
            Vtopo_L= .false.
         endif
      endif

      call difdatsd (diffd,Step_runstrt_S,F_datev)
      step_current = diffd*86400.d0 / Step_dt + Step_initial
      call var_topo2 ( F_topo, step_current, &
                       l_minx,l_maxx,l_miny,l_maxy )

      if ( associated(meqr) .and. G_lam .and. .not. Grd_yinyang_L) then
      if ( Lam_blendoro_L ) then
         topo_temp(1:l_ni,1:l_nj)= meqr(1:l_ni,1:l_nj,1)
         call nest_blend ( F_topo, topo_temp, l_minx,l_maxx, &
              l_miny,l_maxy, 'M', level=G_nk+1 ) 
      endif
      endif

      nullify (ssqr,ssur,ssvr,ttr,hur)
      err= inp_read ( 'SFCPRES'    , 'Q', ssqr,    ip1_list, nka    )
      err= inp_read ( 'SFCPRES'    , 'U', ssur,    ip1_list, nka    )
      err= inp_read ( 'SFCPRES'    , 'V', ssvr,    ip1_list, nka    )
      err= inp_read ( 'TEMPERATURE', 'Q', ttr ,    ip1_list, nka_tt )
      err= inp_read ( 'TR/HU'      , 'Q', hur , HU_ip1_list, nka_hu )

      if (nka_tt.lt.1) &
      call gem_error (-1,'inp_data','Missing field: TT - temperature TT')

      nka= nka_tt
      allocate (tv(l_minx:l_maxx,l_miny:l_maxy,nka))
      tv=ttr

      if (Inp_kind /= 105) then
         do kt=1, nka_tt
         do kh=1, nka_hu
            if (ip1_list(kt) == HU_ip1_list(kh)) then
               call mfottv2 ( tv(l_minx,l_miny,kt),tv(l_minx,l_miny,kt),&
                             hur(l_minx,l_miny,kh),l_minx,l_maxx       ,&
                             l_miny,l_maxy,1, 1,l_ni,1,l_nj, .true. )
               goto 67
            endif
         end do
 67      end do
      endif

      call convip (ip1_list(nka), lev, i,-1, vname, .false.)
      sfcTT_L = (i == 4) .or. ( (i /= 2) .and. (abs(lev-1.) <= 1.e-5) )

      allocate ( srclev(l_minx:l_maxx,l_miny:l_maxy,nka) ,&
                 dstlev(l_minx:l_maxx,l_miny:l_maxy,G_nk),&
                 ssq0  (l_minx:l_maxx,l_miny:l_maxy)     ,&
                 ssu0  (l_minx:l_maxx,l_miny:l_maxy)     ,&
                 ssv0  (l_minx:l_maxx,l_miny:l_maxy) )

      if (.not.associated(ssqr)) then
         ! check for input on pressure vertical coordinate
         if (Inp_kind == 2) then
            nullify(gzr)
            err= inp_read ( 'GEOPOTENTIAL' , 'Q', gzr, ip1_list, nka_gz )
            if (nka_gz == nka) then
               allocate (rna(nka))
               do k=1,nka
                  call convip(ip1_list(k),rna(k),kind,-1,' ',.false.)
               enddo
               call gz2p02 ( ssq0, gzr, F_topo, rna     ,&
                             l_minx,l_maxx,l_miny,l_maxy,&
                             nka,1,l_ni,1,l_nj )
               do k=1,nka
                  srclev(1:l_ni,1:l_nj,k)= rna(k)*100.
               enddo
               deallocate (rna,gzr) ; nullify (gzr)
               allocate (ssqr(l_minx:l_maxx,l_miny:l_maxy,1))
               allocate (ssur(l_minx:l_maxx,l_miny:l_maxy,1))
               allocate (ssvr(l_minx:l_maxx,l_miny:l_maxy,1))
               ssqr(1:l_ni,1:l_nj,1) = srclev(1:l_ni,1:l_nj,nka)
               ssur(1:l_ni,1:l_nj,1) = srclev(1:l_ni,1:l_nj,nka)
               ssvr(1:l_ni,1:l_nj,1) = srclev(1:l_ni,1:l_nj,nka)
            endif
         endif
      else
         pres  => ssqr(1:l_ni,1:l_nj,1)
         ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels (vgd_src, ip1_list(1:nka), ptr3d, pres)
         if ( associated(meqr) .and. sfcTT_L ) then
            if (lun_out.gt.0) &
            write(lun_out,'(" PERFORMING surface pressure adjustment")')
            srclev(1:l_ni,1:l_nj,nka)= ssqr(1:l_ni,1:l_nj,1)
            call adj_ss2topo2 ( ssq0, F_topo, srclev, meqr, tv  , &
                                l_minx,l_maxx,l_miny,l_maxy, nka, &
                                1,l_ni,1,l_nj )
            deallocate (meqr) ; nullify (meqr)
         else
            if (lun_out.gt.0) &
            write(lun_out,'(" NO surface pressure adjustment")')
            ssq0(1:l_ni,1:l_nj)= ssqr(1:l_ni,1:l_nj,1)
         endif
      endif

      if (.not.associated(ssqr)) &
      call gem_error ( -1, 'inp_data', &
          'Missing input data: surface pressure')

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

      pres  => ssqr(1:l_ni,1:l_nj,1)
      ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
      istat= vgd_levels ( vgd_src, ip1_list(1:nka), ptr3d, &
                          pres, in_log=.true. )

      pres  => ssq0  (1:l_ni,1:l_nj)
      ptr3d => dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels ( vgd_dst, Ver_ip1%t(1:G_nk), ptr3d, &
                          pres, in_log=.true. )

      if (F_stag_L) then
         call vertint2 ( F_t,dstlev,G_nk, tv,srclev,nka, &
                         l_minx,l_maxx,l_miny,l_maxy   , &
                         1,l_ni, 1,l_nj, varname='TT' )
      else
         call vertint2 ( F_t,dstlev,G_nk,ttr,srclev,nka, &
                         l_minx,l_maxx,l_miny,l_maxy   , &
                         1,l_ni, 1,l_nj, varname='TT' )
      endif

      nullify (trp)
      istat= gmm_get (trim(F_trprefix_S)//'HU'//trim(F_trsuffix_S),trp)
      if ( nka_hu > 1 ) then
      err= inp_read ( 'TR/HU'      , 'Q', hur , HU_ip1_list, nka_hu )
         pres  => ssqr(1:l_ni,1:l_nj,1)
         ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels ( vgd_src, HU_ip1_list(1:nka_hu), ptr3d, &
                             pres, in_log=.true. )
         call vertint2 ( trp,dstlev,G_nk, hur,srclev,nka_hu         ,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni, 1,l_nj,&
                         inttype=Inp_vertintype_tracers_S )
      else
         trp = 0.
      endif

      deallocate (tv,ttr,hur,srclev,dstlev) ; nullify (ttr,hur)
      if (associated(ip1_list)) deallocate (ip1_list)

      NTR_Tr3d_ntr= 0
      do n=1,Tr3d_ntr
         nullify (trp)
         vname= trim(Tr3d_name_S(n))
         istat= gmm_get (&
               trim(F_trprefix_S)//trim(vname)//trim(F_trsuffix_S),trp)
         if (trim(vname) /= 'HU') then
            err= inp_get ( 'TR/'//trim(vname),'Q', Ver_ip1%t,&
                           vgd_src, vgd_dst, ssqr, ssq0, trp,&
                           l_minx,l_maxx,l_miny,l_maxy,G_nk ,&
                           F_inttype_S=Inp_vertintype_tracers_S )
            if (err == 0) then
               NTR_Tr3d_ntr= NTR_Tr3d_ntr + 1
               NTR_Tr3d_name_S(NTR_Tr3d_ntr) = trim(vname)
            endif
         endif
         trp= max(trp,Tr3d_vmin(n))
      end do

      ip1_w(1:G_nk)= Ver_ip1%t(1:G_nk)
      if (.not. Schm_lift_ltl_L) ip1_w(G_nk)=Ver_ip1%t(G_nk+1)
      err= inp_get ('WT1',  'Q', ip1_w            ,&
                    vgd_src,vgd_dst,ssqr,ssq0,F_w ,&
                    l_minx,l_maxx,l_miny,l_maxy,G_nk)
      Inp_w_L= ( err == 0 )

      err= inp_get ('ZDT1', 'Q', Ver_ip1%t        ,&
                    vgd_src,vgd_dst,ssqr,ssq0,F_zd,&
                    l_minx,l_maxx,l_miny,l_maxy,G_nk)
      Inp_zd_L= ( err == 0 )

      if (.not.Schm_hydro_L) &
      err= inp_get ( 'QT1', 'Q', Ver_ip1%m(2:G_nk+1),&
                      vgd_src,vgd_dst,ssqr,ssq0,F_q ,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk )

      ut1_L= .false. ; urt1_L= .false.

      if (F_stag_L) then
      err= inp_get ( 'URT1', 'U', Ver_ip1%m       ,&
                     vgd_src,vgd_dst,ssur,ssu0,F_u,&
                     l_minx,l_maxx,l_miny,l_maxy,G_nk )
      if ( err == 0 ) &
      err= inp_get ( 'VRT1', 'V', Ver_ip1%m       ,&
                     vgd_src,vgd_dst,ssvr,ssv0,F_v,&
                     l_minx,l_maxx,l_miny,l_maxy,G_nk )
      urt1_L= ( err == 0 )

      if (.not. urt1_L) then
         err= inp_get ( 'UT1', 'U', Ver_ip1%m        ,&
                        vgd_src,vgd_dst,ssur,ssu0,F_u,&
                        l_minx,l_maxx,l_miny,l_maxy,G_nk )
         if ( err == 0 ) &
         err= inp_get ( 'VT1', 'V', Ver_ip1%m        ,&
                        vgd_src,vgd_dst,ssvr,ssv0,F_v,&
                        l_minx,l_maxx,l_miny,l_maxy,G_nk )
         ut1_L= ( err == 0 )
         ! Remove the .and. part of this test be 2021
         if ( ut1_L .and. Inp_ut1_is_urt1 == -1 ) &
              call image_to_real_winds ( F_u,F_v, l_minx,l_maxx,&
                                         l_miny,l_maxy, G_nk )

      endif
      endif

      if ((.not. urt1_L) .and. (.not. ut1_L)) &
         call inp_hwnd ( F_u,F_v, vgd_src,vgd_dst, F_stag_L,&
                         ssqr,ssur,ssvr, ssq0,ssu0,ssv0    ,&
                         l_minx,l_maxx,l_miny,l_maxy,G_nk )

      deallocate (ssqr,ssur,ssvr,ssq0,ssu0,ssv0)

      call inp_close ()

      istat = fstopc ('MSGLVL','INFORM',.false.)

 9000 format(/,' TREATING INPUT DATA VALID AT: ',a,&
             /,' ===============================================')
!
!-----------------------------------------------------------------------
!
      return
      end

