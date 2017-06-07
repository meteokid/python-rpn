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
!**s/r set_geom - initialize model geometry

      subroutine set_geom
      use hgrid_wb, only: hgrid_wb_put
      use gmm_geof
      use grid_options
      use gem_options
      use tdpack
      implicit none
#include <arch_specific.hf>

#include <WhiteBoard.hf>
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "lun.cdk"
#include "geomn.cdk"
#include "geomg.cdk"
#include "hgc.cdk"
#include "ptopo.cdk"
#include "type.cdk"
#include "cstv.cdk"
#include "ver.cdk"

      integer, external :: ezgdef_fmem,gdll
      character*8 dumc
      integer offi,offj,indx,err,dgid,hgc(4)
      integer i,j,k,dimy,istat,pnip1,ni,nj,offset
      real xfi(0:l_ni+1),yfi(0:l_nj+1),height,heightp1
      real gxfi(G_ni),gyfi(G_nj), dxmax, dymax
      real*8 posx_8(1-G_halox:G_ni+G_halox+1), posy_8(1-G_haloy:G_nj+G_haloy+1)
      real*8 rad2deg_8,deg2rad_8,x0,xl,y0,yl
      real*8 ZERO_8, HALF_8, ONE_8, TWO_8, CLXXX_8
      real*8 scale_factor, scale_factor_v
      real*8 Del_xg , Del_yg 
      parameter( ZERO_8  = 0.0 )
      parameter( HALF_8  = 0.5 )
      parameter( ONE_8   = 1.0 )
      parameter( TWO_8   = 2.0 )
      parameter( CLXXX_8 = 180.0 )
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)

      rad2deg_8 = CLXXX_8/pi_8
      deg2rad_8 = acos( -ONE_8 )/CLXXX_8

      Geomn_minx= west
      Geomn_miny= south
      Geomn_maxx= l_ni + 1 - east 
      Geomn_maxy= l_nj + 1 - north

      allocate (G_xg_8(1-G_halox:G_ni+G_halox+1) , G_yg_8(1-G_haloy:G_nj+G_haloy+1) )
      allocate (Geomn_latrx(l_ni,l_nj), Geomn_lonrx(l_ni,l_nj))
      allocate (Geomn_lonQ(1-G_halox:G_ni+G_halox), Geomn_latQ(1-G_haloy:G_nj+G_haloy),&
                Geomn_lonF(1-G_halox:G_ni+G_halox), Geomn_latF(1-G_haloy:G_nj+G_haloy) )
      allocate (Geomn_latgs(G_nj),Geomn_longs(G_ni+1), &
                Geomn_latgv(G_nj),Geomn_longu(G_ni+1), &
                Geomn_latij(Geomn_minx:Geomn_maxx,Geomn_miny:Geomn_maxy), &
                Geomn_lonij(Geomn_minx:Geomn_maxx,Geomn_miny:Geomn_maxy))

      ni= Grd_ni + 2*G_halox + 1
      nj= Grd_nj + 2*G_haloy + 1
      x0= Grd_x0_8 - dble(G_halox)  *dble(Grd_dx)
      y0= Grd_y0_8 - dble(G_haloy)  *dble(Grd_dy)
      xl= Grd_xl_8 + dble(G_halox+1)*dble(Grd_dx)
      yl= Grd_yl_8 + dble(G_haloy+1)*dble(Grd_dy)

      call set_gemHgrid4 ( posx_8, posy_8, ni, nj, &
                           Grd_dx, Grd_dy, x0,xl,y0,yl, Grd_yinyang_L )

      if (Lun_out.gt.0) then
         write (Lun_out,1005) G_nk,Hyb_rcoef
         do k=1,G_nk
            height  =-16000./alog(10.)*alog(Ver_hyb%m(k))
            if (k.lt.G_nk)&
            heightp1 =-16000./alog(10.)*alog(Ver_hyb%m(k+1))
            if (k.eq.G_nk) heightp1 = 0.
            call convip(pnip1,Ver_hyb%m(k),5,1,dumc,.false.)
            write (Lun_out,1006) k,Ver_hyb%m(k),height, &
                                 height-heightp1,pnip1
         end do
      endif

      call canonical_cases ("SET_GEOM")

      G_xg_8(1-G_halox:G_ni+G_halox+1) = posx_8(1-G_halox:G_ni+G_halox+1)*deg2rad_8
      G_yg_8(1-G_haloy:G_nj+G_haloy+1) = posy_8(1-G_haloy:G_nj+G_haloy+1)*deg2rad_8

      Geomn_lonQ(1-G_halox:G_ni+G_halox) = posx_8(1-G_halox:G_ni+G_halox)
      Geomn_latQ(1-G_haloy:G_nj+G_haloy) = posy_8(1-G_haloy:G_nj+G_haloy)
      
      do i= 1-G_halox, G_ni+G_halox
         Geomn_lonF(i) = (posx_8(i+1) + posx_8(i)) * HALF_8
      end do
      do j= 1-G_haloy, G_nj+G_haloy
         Geomn_latF(j) = (posy_8(j+1) + posy_8(j)) * HALF_8
      end do
      
      do i = 1, G_ni+1
         Geomn_longs(i) =  G_xg_8(i) * rad2deg_8
         Geomn_longu(i) = (G_xg_8(i+1)+G_xg_8(i))*HALF_8*rad2deg_8
      end do

      do i = 1, G_nj
         Geomn_latgs(i) =  G_yg_8(i) * rad2deg_8
         Geomn_latgv(i) = (G_yg_8(i+1)+G_yg_8(i))*HALF_8*rad2deg_8
      end do

      allocate (Geomg_x_8 (l_minx:l_maxx),Geomg_xu_8 (l_minx:l_maxx),&
                Geomg_sx_8(l_minx:l_maxx),Geomg_sy_8 (l_miny:l_maxy),&
                Geomg_cx_8(l_minx:l_maxx),Geomg_cy_8 (l_miny:l_maxy),&
                
                Geomg_y_8      (l_miny:l_maxy),Geomg_yv_8     (l_miny:l_maxy),&
                Geomg_cy2_8    (l_miny:l_maxy),Geomg_cyv_8    (l_miny:l_maxy),&
                Geomg_tyoa_8   (l_miny:l_maxy),Geomg_tyoav_8  (l_miny:l_maxy),&
                Geomg_cyv2_8   (l_miny:l_maxy),Geomg_cyM_8    (l_miny:l_maxy),&
                Geomg_invDYM_8 (l_miny:l_maxy),Geomg_invDYMv_8(l_miny:l_maxy),&
                Geomg_invcy_8  (l_miny:l_maxy),Geomg_invcyv_8 (l_miny:l_maxy),&
                Geomg_invcy2_8 (l_miny:l_maxy),Geomg_invcyv2_8(l_miny:l_maxy),&
                Geomg_invDX_8  (l_miny:l_maxy),Geomg_invDXM_8 (l_miny:l_maxy),&
                Geomg_invDXMu_8(l_miny:l_maxy),Geomg_invDXv_8 (l_miny:l_maxy),&
                Geomg_area_8(l_ni,l_nj),Geomg_mask_8(l_ni,l_nj))
         
      offi = Ptopo_gindx(1,Ptopo_myproc+1)-1
      offj = Ptopo_gindx(3,Ptopo_myproc+1)-1
      
      Del_xg= G_xg_8(2) - G_xg_8(1)
      Del_yg= G_yg_8(2) - G_yg_8(1)
      
      Geomg_hx_8 = Del_xg   
      Geomg_hy_8 = Del_yg
      
      Geomg_invDY_8  = ONE_8 / ( rayt_8 * Del_yg )

      do i=1-G_halox, l_ni+G_halox
         indx = offi + i
         Geomg_x_8 (i) =  G_xg_8(indx)
         Geomg_xu_8(i) = (G_xg_8(indx+1) + G_xg_8(indx)) * HALF_8
         Geomg_sx_8(i) = sin( Geomg_x_8(i) )
         Geomg_cx_8(i) = cos( Geomg_x_8(i) )
      end do
      
      do j=1-G_haloy, l_nj+G_haloy
         indx = offj + j   
         Geomg_y_8  (j) =  G_yg_8(indx)
         Geomg_yv_8 (j) = (G_yg_8(indx+1) + G_yg_8(indx)) * HALF_8
         Geomg_sy_8  (j)= sin( Geomg_y_8 (j) )
         Geomg_cy_8  (j)= cos( Geomg_y_8 (j) )
         Geomg_cy2_8 (j)= cos( Geomg_y_8 (j) )**2
         Geomg_cyv_8 (j)= cos( Geomg_yv_8(j) )
         Geomg_cyv2_8(j)= cos( Geomg_yv_8(j) )**2
         Geomg_cyM_8 (j)= Geomg_cyv_8(j)
      end do

      dimy = l_nj+2*G_haloy
      call vrec ( geomg_invcy2_8  , geomg_cy2_8 , dimy )
      call vrec ( geomg_invcyv2_8 , geomg_cyv2_8, dimy )
      call vrec ( geomg_invcy_8   , geomg_cy_8  , dimy )
      call vrec ( geomg_invcyv_8  , geomg_cyv_8 , dimy )
             
      do j=1-G_haloy, l_nj+G_haloy
         indx = offj + j
         Geomg_invDYMv_8(j)= Geomg_invDY_8
         Geomg_invDYM_8 (j)= Geomg_invDY_8*geomg_invcy_8(j)
         Geomg_tyoa_8   (j)= 0.d0
         Geomg_tyoav_8  (j)= 0.d0
      end do
 
      do j=1-G_haloy, l_nj+G_haloy
         Geomg_tyoa_8(j) =tan(Geomg_y_8 (j))/rayt_8
         Geomg_tyoav_8(j)=tan(Geomg_yv_8(j))/rayt_8
      enddo

      do j=1-G_haloy, l_nj+G_haloy
         scale_factor   = rayt_8 * Geomg_cy_8(j)
         scale_factor_v = rayt_8 * Geomg_cyv_8(j)
         Geomg_invDX_8  (j) = ONE_8/(scale_factor   * Del_xg )
         Geomg_invDXv_8(j)  = ONE_8/(scale_factor_v * Del_xg )
         Geomg_invDXM_8 (j) = Geomg_invDX_8(j)
         Geomg_invDXMu_8(j) = Geomg_invDX_8(j)  
      end do

      do i=1,l_ni
         indx = offi + i
         xfi(i) = posx_8(indx)
      end do
      do i=1,l_nj
         indx = offj + i
         yfi(i) = posy_8(indx)
      end do
      gxfi(1:G_ni) = posx_8(1:G_ni)
      gyfi(1:G_nj) = posy_8(1:G_nj)

      Grd_global_gid = ezgdef_fmem (G_ni , G_nj , 'Z', 'E', Hgc_ig1ro, &
                       Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, gxfi(1) , gyfi(1))
      Grd_local_gid  = ezgdef_fmem (l_ni , l_nj , 'Z', 'E', Hgc_ig1ro, &
                       Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro,  xfi(1) ,  yfi(1))
      Grd_lclcore_gid= ezgdef_fmem (l_ni-pil_w-pil_e, l_nj-pil_s-pil_n,&
                       'Z', 'E', Hgc_ig1ro, Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro,&
                       xfi(1+pil_w), yfi(1+pil_s) )

      offset= 2
      Grd_lphy_i0 = 1 + offset*west  ; Grd_lphy_in = l_ni-offset*east
      Grd_lphy_j0 = 1 + offset*south ; Grd_lphy_jn = l_nj-offset*north
      Grd_lphy_ni = Grd_lphy_in - Grd_lphy_i0 + 1
      Grd_lphy_nj = Grd_lphy_jn - Grd_lphy_j0 + 1

      Grd_lphy_gid  = ezgdef_fmem ( Grd_lphy_ni, Grd_lphy_nj, 'Z', 'E', &
                      Hgc_ig1ro, Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro      , &
                      xfi(Grd_lphy_i0), yfi(Grd_lphy_j0) )

      istat= hgrid_wb_put ('model/Hgrid/global' ,Grd_global_gid  , &
                            F_lni=G_ni,F_lnj=G_nj,F_rewrite_L=.true.)
      istat= hgrid_wb_put ('model/Hgrid/local'  ,Grd_local_gid   , &
                            F_lni=l_ni,F_lnj=l_nj,F_rewrite_L=.true.)
      istat= hgrid_wb_put ('model/Hgrid/lclcore',Grd_lclcore_gid , &
                  F_lni=l_ni-pil_w-pil_e, F_lnj=l_nj-pil_s-pil_n , &
                  F_i0=1+pil_w, F_j0=1+pil_s,F_rewrite_L=.true.)
      istat= hgrid_wb_put ('model/Hgrid/lclphy' ,Grd_lphy_gid    , &
                  F_lni=Grd_lphy_ni, F_lnj=Grd_lphy_nj           , &
                  F_i0=Grd_lphy_i0, F_j0=Grd_lphy_j0,F_rewrite_L=.true.)

      hgc(1)= Hgc_ig1ro
      hgc(2)= Hgc_ig2ro
      hgc(3)= Hgc_ig3ro
      hgc(4)= Hgc_ig4ro

      istat= wb_put('model/Hgrid/hgcrot',hgc,WB_REWRITE_NONE+WB_IS_LOCAL)

      err = gdll (Grd_local_gid, Geomn_latrx, Geomn_lonrx)
      do j=1,l_nj
      do i=1,l_ni
         if (Geomn_lonrx(i,j).ge.180.0) Geomn_lonrx(i,j)=Geomn_lonrx(i,j)-360.0
      enddo
      enddo

      do i=Geomn_minx, Geomn_maxx
         indx = offi + i
         xfi(i) = G_xg_8(indx)*rad2deg_8
      end do
      do i=Geomn_miny, Geomn_maxy
         indx = offj + i
         yfi(i) = G_yg_8(indx)*rad2deg_8
      end do
      ni = Geomn_maxx-Geomn_minx+1
      nj = Geomn_maxy-Geomn_miny+1
      dgid = ezgdef_fmem (ni , nj , 'Z', 'E', Hgc_ig1ro, &
                Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xfi(Geomn_minx) , yfi(Geomn_miny) )
      err = gdll (dgid,Geomn_latij,Geomn_lonij)

 1000 format(/,'INITIALIZATION OF MODEL GEOMETRY (S/R SET_GEOM)', &
             /'===============================================')
 1005 format (/'STAGGERED VERTICAL LAYERING ON',I4,' MOMENTUM HYBRID LEVELS WITH ', &
               'Grd_rcoef= ',2f7.2,':'/ &
               2x,'level',10x,'HYB',8x,'~HEIGHTS',5x,'~DELTA_Z',7x,'IP1')
 1006 format (1x,i4,3x,es15.5,2(6x,f6.0),4x,i10)
!
!     ---------------------------------------------------------------
!
      return
      end
