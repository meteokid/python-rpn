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
!
module geomh
   implicit none
   public
   save
!______________________________________________________________________
!                                                                      |
!  HORIZONTAL GRID COORDINATES and other related parameters            |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! geomh_x_8          | longitude                                       |
! geomh_xu_8         | longitude of u points                           |
! geomh_y_8          | latitude                                        |
! geomh_yv_8         | latitude  of v points                           |
!--------------------|-------------------------------------------------|
! geomh_hx_8         | distance between grid points in x direction     |
! geomh_hy_8         | distance between grid points in y direction     |   |
!----------------------------------------------------------------------
! geomh_sx_8         | sine of longitude                               |
! geomh_sy_8         | sine of latitude                                |
! geomh_cx_8         | cosine of longitude                             |
! geomh_cy_8         | cosine of latitude                              |
! geomh_cy2_8        | cosine of latitude squared                      |
! geomh_cyv_8        | cosine of latitude for v points                 |
! geomh_cyv2_8       | cosine of latitude squared for v points         |
! geomh_cyM_8     ** | SPECIAL for MODEL core only                     |
!--------------------|-------------------------------------------------|
! geomh_invcy_8      | inverse of geomh_cy_8                           |
! geomh_invcyv_8     | inverse of geomh_cyv_8                          |
! geomh_invcy2_8     | inverse of geomh_cy2_8                          |
! geomh_invcyv2_8    | inverse of geomh_cyv2_8                         |
!--------------------|-------------------------------------------------|
! geomh_tyoa_8       | tangent of latitude divided by rayt             |
! geomh_tyoav_8      | tangent of latitude divided by rayt (v points)  |
!--------------------|-------------------------------------------------|
! geomh_invDY_8      | inverse DELTAY = cos(theta)2/dsin(theta)        |
! geomh_invDYM_8  ** | SPECIAL for MODEL core only                     |
! geomh_invDYMv_8 ** | SPECIAL for MODEL core only                     |
! geomh_invDX_8      | inverse DELTAX = 1/d(lambda)                    |
! geomh_invDXM_8  ** | SPECIAL for MODEL core only                     |
! geomh_invDXMu_8 ** | SPECIAL for MODEL core only                     |
! geomh_invDXv_8     | inverse DELTAX for v points (ex: for vorticity) |
!----------------------------------------------------------------------
! geomh_area_8       |                                                 |
! geomh_mask_8       |                                                 |
!----------------------------------------------------------------------
   real*8 :: geomh_invDY_8, geomh_hx_8, geomh_hy_8

   real*8, dimension(:), pointer :: &
          geomh_x_8    ,geomh_xu_8    ,geomh_y_8     ,geomh_yv_8     ,&
          geomh_sx_8   ,geomh_sy_8    ,geomh_cx_8    ,geomh_cy_8     ,&
          geomh_cy2_8  ,geomh_cyv_8   ,geomh_cyv2_8  ,geomh_cyM_8    ,&
          geomh_invcy_8,geomh_invcyv_8,geomh_invcy2_8,geomh_invcyv2_8,&
          geomh_tyoa_8 ,geomh_tyoav_8 ,&
          geomh_invDYM_8,geomh_invDYMv_8,&
          geomh_invDX_8 ,geomh_invDXM_8,geomh_invDXMu_8,geomh_invDXv_8

   real*8, dimension(:,:), pointer :: geomh_area_8, geomh_mask_8

!______________________________________________________________________
!                                                                      |
!  LATITUDES AND LONGITUDES IN DEGREES                                 |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! geomh_latgs        | phi latitudes in degrees for model output       |
! geomh_longs        | phi longitudes in degrees for model output      |
! geomh_latgv        | V   latitudes in degrees for model output       |
! geomh_longu        | U   longitudes in degrees for model output      |
! geomh_latrx        | phi geographical latitudes in degrees           |
! geomh_lonrx        | phi geographical longitudes in degrees          |
!----------------------------------------------------------------------
      integer geomh_minx, geomh_maxx, geomh_miny, geomh_maxy
      real, dimension(:  ), pointer :: geomh_lonQ, geomh_latQ, &
                                       geomh_lonF, geomh_latF
      real, dimension(:  ), pointer :: geomh_latgs, geomh_longs, &
                                       geomh_latgv, geomh_longu
      real, dimension(:,:), pointer :: geomh_latrx, geomh_lonrx, &
                                       geomh_latij, geomh_lonij

   public set_geomh

contains

   subroutine set_geomh()
      ! initialize model horizontal geometry

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
#include "hgc.cdk"
#include "ptopo.cdk"
#include "type.cdk"
#include "cstv.cdk"
#include "dcst.cdk"

      integer, external :: ezgdef_fmem,gdll
      integer offi,offj,indx,err,dgid,hgc(4)
      integer i,j,dimy,istat,ni,nj,offset
      real xfi(0:l_ni+1),yfi(0:l_nj+1)
      real gxfi(G_ni),gyfi(G_nj)
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

      geomh_minx= west
      geomh_miny= south
      geomh_maxx= l_ni + 1 - east
      geomh_maxy= l_nj + 1 - north

      allocate (G_xg_8(1-G_halox:G_ni+G_halox+1) , G_yg_8(1-G_haloy:G_nj+G_haloy+1) )
      allocate (geomh_latrx(l_ni,l_nj), geomh_lonrx(l_ni,l_nj))
      allocate (geomh_lonQ(1-G_halox:G_ni+G_halox), geomh_latQ(1-G_haloy:G_nj+G_haloy),&
                geomh_lonF(1-G_halox:G_ni+G_halox), geomh_latF(1-G_haloy:G_nj+G_haloy) )
      allocate (geomh_latgs(G_nj),geomh_longs(G_ni+1), &
                geomh_latgv(G_nj),geomh_longu(G_ni+1), &
                geomh_latij(geomh_minx:geomh_maxx,geomh_miny:geomh_maxy), &
                geomh_lonij(geomh_minx:geomh_maxx,geomh_miny:geomh_maxy))

      ni= Grd_ni + 2*G_halox + 1
      nj= Grd_nj + 2*G_haloy + 1
      x0= Grd_x0_8 - dble(G_halox)  *dble(Grd_dx)
      y0= Grd_y0_8 - dble(G_haloy)  *dble(Grd_dy)
      xl= Grd_xl_8 + dble(G_halox+1)*dble(Grd_dx)
      yl= Grd_yl_8 + dble(G_haloy+1)*dble(Grd_dy)

      call set_gemHgrid4 ( posx_8, posy_8, ni, nj, &
                           Grd_dx, Grd_dy, x0,xl,y0,yl, Grd_yinyang_L )

      G_xg_8(1-G_halox:G_ni+G_halox+1) = posx_8(1-G_halox:G_ni+G_halox+1)*deg2rad_8
      G_yg_8(1-G_haloy:G_nj+G_haloy+1) = posy_8(1-G_haloy:G_nj+G_haloy+1)*deg2rad_8

      geomh_lonQ(1-G_halox:G_ni+G_halox) = posx_8(1-G_halox:G_ni+G_halox)
      geomh_latQ(1-G_haloy:G_nj+G_haloy) = posy_8(1-G_haloy:G_nj+G_haloy)

      do i= 1-G_halox, G_ni+G_halox
         geomh_lonF(i) = (posx_8(i+1) + posx_8(i)) * HALF_8
      end do
      do j= 1-G_haloy, G_nj+G_haloy
         geomh_latF(j) = (posy_8(j+1) + posy_8(j)) * HALF_8
      end do

      do i = 1, G_ni+1
         geomh_longs(i) =  G_xg_8(i) * rad2deg_8
         geomh_longu(i) = (G_xg_8(i+1)+G_xg_8(i))*HALF_8*rad2deg_8
      end do

      do i = 1, G_nj
         geomh_latgs(i) =  G_yg_8(i) * rad2deg_8
         geomh_latgv(i) = (G_yg_8(i+1)+G_yg_8(i))*HALF_8*rad2deg_8
      end do

      allocate (geomh_x_8 (l_minx:l_maxx),geomh_xu_8 (l_minx:l_maxx),&
                geomh_sx_8(l_minx:l_maxx),geomh_sy_8 (l_miny:l_maxy),&
                geomh_cx_8(l_minx:l_maxx),geomh_cy_8 (l_miny:l_maxy),&

                geomh_y_8      (l_miny:l_maxy),geomh_yv_8     (l_miny:l_maxy),&
                geomh_cy2_8    (l_miny:l_maxy),geomh_cyv_8    (l_miny:l_maxy),&
                geomh_tyoa_8   (l_miny:l_maxy),geomh_tyoav_8  (l_miny:l_maxy),&
                geomh_cyv2_8   (l_miny:l_maxy),geomh_cyM_8    (l_miny:l_maxy),&
                geomh_invDYM_8 (l_miny:l_maxy),geomh_invDYMv_8(l_miny:l_maxy),&
                geomh_invcy_8  (l_miny:l_maxy),geomh_invcyv_8 (l_miny:l_maxy),&
                geomh_invcy2_8 (l_miny:l_maxy),geomh_invcyv2_8(l_miny:l_maxy),&
                geomh_invDX_8  (l_miny:l_maxy),geomh_invDXM_8 (l_miny:l_maxy),&
                geomh_invDXMu_8(l_miny:l_maxy),geomh_invDXv_8 (l_miny:l_maxy),&
                geomh_area_8(l_ni,l_nj),geomh_mask_8(l_ni,l_nj))

      offi = Ptopo_gindx(1,Ptopo_myproc+1)-1
      offj = Ptopo_gindx(3,Ptopo_myproc+1)-1

      Del_xg= G_xg_8(2) - G_xg_8(1)
      Del_yg= G_yg_8(2) - G_yg_8(1)

      geomh_hx_8 = Del_xg
      geomh_hy_8 = Del_yg

      geomh_invDY_8  = ONE_8 / ( Dcst_rayt_8 * Del_yg )

      do i=1-G_halox, l_ni+G_halox
         indx = offi + i
         geomh_x_8 (i) =  G_xg_8(indx)
         geomh_xu_8(i) = (G_xg_8(indx+1) + G_xg_8(indx)) * HALF_8
         geomh_sx_8(i) = sin( geomh_x_8(i) )
         geomh_cx_8(i) = cos( geomh_x_8(i) )
      end do

      do j=1-G_haloy, l_nj+G_haloy
         indx = offj + j
         geomh_y_8  (j) =  G_yg_8(indx)
         geomh_yv_8 (j) = (G_yg_8(indx+1) + G_yg_8(indx)) * HALF_8
         geomh_sy_8  (j)= sin( geomh_y_8 (j) )
         geomh_cy_8  (j)= cos( geomh_y_8 (j) )
         geomh_cy2_8 (j)= cos( geomh_y_8 (j) )**2
         geomh_cyv_8 (j)= cos( geomh_yv_8(j) )
         geomh_cyv2_8(j)= cos( geomh_yv_8(j) )**2
         geomh_cyM_8 (j)= geomh_cyv_8(j)
      end do

      dimy = l_nj+2*G_haloy
      call vrec ( geomh_invcy2_8  , geomh_cy2_8 , dimy )
      call vrec ( geomh_invcyv2_8 , geomh_cyv2_8, dimy )
      call vrec ( geomh_invcy_8   , geomh_cy_8  , dimy )
      call vrec ( geomh_invcyv_8  , geomh_cyv_8 , dimy )

      do j=1-G_haloy, l_nj+G_haloy
         indx = offj + j
         geomh_invDYMv_8(j)= geomh_invDY_8
         geomh_invDYM_8 (j)= geomh_invDY_8*geomh_invcy_8(j)
         geomh_tyoa_8   (j)= 0.d0
         geomh_tyoav_8  (j)= 0.d0
      end do

      do j=1-G_haloy, l_nj+G_haloy
         geomh_tyoa_8(j) =tan(geomh_y_8 (j))/Dcst_rayt_8
         geomh_tyoav_8(j)=tan(geomh_yv_8(j))/Dcst_rayt_8
      enddo

      do j=1-G_haloy, l_nj+G_haloy
         scale_factor   = Dcst_rayt_8 * geomh_cy_8(j)
         scale_factor_v = Dcst_rayt_8 * geomh_cyv_8(j)
         geomh_invDX_8  (j) = ONE_8/(scale_factor   * Del_xg )
         geomh_invDXv_8(j)  = ONE_8/(scale_factor_v * Del_xg )
         geomh_invDXM_8 (j) = geomh_invDX_8(j)
         geomh_invDXMu_8(j) = geomh_invDX_8(j)
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

      err = gdll (Grd_local_gid, geomh_latrx, geomh_lonrx)
      do j=1,l_nj
      do i=1,l_ni
         if (geomh_lonrx(i,j).ge.180.0) geomh_lonrx(i,j)=geomh_lonrx(i,j)-360.0
      enddo
      enddo

      do i=geomh_minx, geomh_maxx
         indx = offi + i
         xfi(i) = G_xg_8(indx)*rad2deg_8
      end do
      do i=geomh_miny, geomh_maxy
         indx = offj + i
         yfi(i) = G_yg_8(indx)*rad2deg_8
      end do
      ni = geomh_maxx-geomh_minx+1
      nj = geomh_maxy-geomh_miny+1
      dgid = ezgdef_fmem (ni , nj , 'Z', 'E', Hgc_ig1ro, &
                Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xfi(geomh_minx) , yfi(geomh_miny) )
      err = gdll (dgid,geomh_latij,geomh_lonij)

 1000 format(/,'INITIALIZATION OF MODEL HORIZONTAL GEOMETRY (S/R set_geomh)', &
             /'===============================================')
!
!     ---------------------------------------------------------------
!
      return
      end subroutine set_geomh

end module geomh
