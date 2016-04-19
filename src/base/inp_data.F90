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
      use vertical_interpolation, only: vertint2
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
#include "grd.cdk"
#include <rmnlib_basics.hf>

Interface
      integer function inp_read ( F_var_S, F_hgrid_S, F_dest, &
                                  F_ip1, F_nka, F_hint_S )
      implicit none
      character*(*)                     ,intent(IN)  :: F_var_S,F_hgrid_S
      character*(*),            optional,intent(IN)  :: F_hint_S
      integer                           ,intent(OUT) :: F_nka
      integer, dimension(:    ), pointer,intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer,intent(OUT) :: F_dest
      end function inp_read

      integer function inp_get ( F_var_S, F_hgrid_S, F_ver_ip1,&
                                 F_vgd_src, F_vgd_dst         ,&
                                 F_sfc_src, F_sfc_dst, F_dest ,&
                                 Minx,Maxx,Miny,Maxy, F_nk    ,&
                                 F_inttype_S )
      use vGrid_Descriptors
      implicit none
      character(len=*)          , intent(IN) :: F_var_S,F_hgrid_S
      character(len=*), optional, intent(IN) :: F_inttype_S
      integer                , intent(IN)  :: Minx,Maxx,Miny,Maxy, F_nk
      integer                , intent(IN)  :: F_ver_ip1(F_nk)
      type(vgrid_descriptor) , intent(IN)  :: F_vgd_src, F_vgd_dst
      real, dimension(Minx:Maxx,Miny:Maxy     ), target, &
                                   intent(IN) :: F_sfc_src, F_sfc_dst
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(OUT):: F_dest
      end function inp_get
End Interface

      character(len=4) vname
      logical initial_data, blend_oro, urt1_l, ut1_l
      integer fst_handle, i,j,k,n, nia,nja,nka, istat, err, kind
      integer nka_gz, nka_tt, nka_hu
      integer, dimension (:), pointer :: ip1_list, ip1_dum
      real topo_temp(l_minx:l_maxx,l_miny:l_maxy),step_current,&
           p0(1:l_ni,1:l_nj)
      real, dimension (:    ), allocatable :: rna
      real, dimension (:,:  ), allocatable :: ssu0,ssv0
      real, dimension (:,:  ), pointer     :: ssq0,pres
      real, dimension (:,:,:), allocatable, target :: dstlev,srclev
      real, dimension (:,:,:), pointer :: ssqr,ssur,ssvr,&
                                          meqr,ttr,hur,gzr,trp,ptr3d
      real*8 diffd, pref_a_8
      type(vgrid_descriptor) :: vgd_src, vgd_dst
!
!-----------------------------------------------------------------------
!
      if (Lun_out.gt.0) write(lun_out,9000) trim(F_datev)

      if (.not.Lun_debug_L) istat= fstopc ('MSGLVL','SYSTEM',.false.)

      initial_data= ( trim(F_datev) .eq. trim(Step_runstrt_S) )

      call inp_open ( F_datev, vgd_src )

      nullify (meqr, ip1_list)
      err= inp_read ( 'OROGRAPHY', 'Q', meqr, ip1_list, nka )

      if ( initial_data ) then
         if ( associated(meqr) ) then
            istat= gmm_get(gmmk_topo_low_s , topo_low )
            topo_low(1:l_ni,1:l_nj)= meqr(1:l_ni,1:l_nj,1)
         else
            Vtopo_L= .false.
         endif
      endif

      call difdatsd (diffd,Step_runstrt_S,F_datev)
      step_current = diffd*86400.d0 / Step_dt
      call var_topo2 ( F_topo, step_current, &
                       l_minx,l_maxx,l_miny,l_maxy )

      if ( associated(meqr) .and. G_lam .and. .not. Grd_yinyang_L ) then
         topo_temp(1:l_ni,1:l_nj)= meqr(1:l_ni,1:l_nj,1)
         call nest_blend ( F_topo, topo_temp, l_minx,l_maxx, &
                           l_miny,l_maxy, 'M', level=G_nk+1 )
      endif

      nullify (ssqr,ssur,ssvr,ttr,hur)
      err= inp_read ( 'SFCPRES'    , 'Q', ssqr, ip1_list, nka    )
      err= inp_read ( 'SFCPRES'    , 'U', ssur, ip1_list, nka    )
      err= inp_read ( 'SFCPRES'    , 'V', ssvr, ip1_list, nka    )
      err= inp_read ( 'TEMPERATURE', 'Q', ttr , ip1_list, nka_tt )
      err= inp_read ( 'TR/HU'      , 'Q', hur , ip1_list, nka_hu )

      if ((nka_tt.lt.1).or.(nka_hu.lt.1).or.(nka_tt.ne.nka_hu)) &
      call gem_error ( -1, 'inp_data', &
      'Missing or inconsistent input data: temperature and/or humidity')

      nka= nka_tt
      if (Inp_kind /= 105) &
      call mfottv2 ( ttr, ttr, hur, l_minx,l_maxx,l_miny,l_maxy,&
                     nka, 1,l_ni,1,l_nj, .true. )

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
         if ( associated(meqr) ) then
            srclev(1:l_ni,1:l_nj,nka)= ssqr(1:l_ni,1:l_nj,1)
            call adj_ss2topo2 ( ssq0, F_topo, srclev, meqr, ttr , &
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

      do n= 1, nka-1
         srclev(1:l_ni,1:l_nj,n) = log(srclev(1:l_ni,1:l_nj,n))
      end do
      srclev(1:l_ni,1:l_nj,nka) = log(ssqr(1:l_ni,1:l_nj,1))

      pres  => ssq0  (1:l_ni,1:l_nj)
      ptr3d => dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels ( vgd_dst, Ver_ip1%t(1:G_nk), ptr3d, &
                          pres, in_log=.true. )
      call vertint2 ( F_t,dstlev,G_nk, ttr,srclev,nka, &
                      l_minx,l_maxx,l_miny,l_maxy    , &
                      1,l_ni, 1,l_nj, varname='TT' )
      nullify (trp)
      istat= gmm_get (trim(F_trprefix_S)//'HU'//trim(F_trsuffix_S),trp)
      call vertint2 ( trp,dstlev,G_nk, hur,srclev,nka, &
                      l_minx,l_maxx,l_miny,l_maxy    , &
                      1,l_ni, 1,l_nj, inttype=Inp_vertintype_tracers_S )
      deallocate (ttr,hur,srclev,dstlev) ; nullify (ttr,hur)
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

      err= inp_get ('WT1',  'Q', Ver_ip1%t        ,&
                    vgd_src,vgd_dst,ssqr,ssq0,F_w ,&
                    l_minx,l_maxx,l_miny,l_maxy,G_nk)
      ana_w_L= ( err == 0 )

      err= inp_get ('ZDT1', 'Q', Ver_ip1%t        ,&
                    vgd_src,vgd_dst,ssqr,ssq0,F_zd,&
                    l_minx,l_maxx,l_miny,l_maxy,G_nk)
      ana_zd_L= ( err == 0 )

      if (.not.Schm_hydro_L) &
      err= inp_get ( 'QT1', 'Q', Ver_ip1%m(2:G_nk+1),&
                      vgd_src,vgd_dst,ssqr,ssq0,F_q ,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk )

      ut1_L= .false.
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

      if ((.not. urt1_L) .and. (.not. ut1_L)) &
      call inp_hwnd ( F_u,F_v, vgd_src,vgd_dst, ssur,ssvr, &
                      ssu0,ssv0, l_minx,l_maxx,l_miny,l_maxy,G_nk )

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

