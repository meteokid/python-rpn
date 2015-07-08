module advection_set_mod
   ! 
   !Sets different advection parameters
   ! Author
   !      Rabah Aider,Stephane Gaudreault,Stephane Chamberland  -- Avril 2015 (based on old s/r: itf_adv_set, adv_set )
   !
   ! Revision
   !

 

  implicit none

   
private

#include <msg.h>
#include <arch_specific.hf>   
#include "grd.cdk"
#include "glb_ld.cdk"
#include "ver.cdk"
#include "schm.cdk"
#include "cstv.cdk"
#include "lam.cdk"
#include "constants.h"
#include "adv_grid.cdk"
#include "adv_dims.cdk"
#include "adv_interp.cdk"

public :: adv_set


contains 

     subroutine adv_set ()

   character(len=40) :: label
   integer  istat, i, j, k

  real*8,dimension(:),allocatable :: v_zm_8,v_zt_8

   
   call msg(MSG_DEBUG,'advection _set')


	
   allocate( v_zm_8(0:l_nk+1), v_zt_8(1:l_nk) )
   
   v_zm_8(0) = Cstv_Ztop_8
   
   do k=1,l_nk+1
    
      v_zm_8(k) = Ver_a_8%m(k)
   enddo
   do k=1,l_nk
   
      v_zt_8(k) = Ver_z_8%t(k)
   enddo
   
   
   adv_int_i_off = l_i0 - 1
   adv_int_j_off = l_j0 - 1
   adv_trj_i_off = 0

   adv_gminx = 1 - G_halox
   adv_gmaxx = G_ni + G_halox
   adv_gminy = 1 - G_haloy
   adv_gmaxy = G_nj + G_haloy

  
   adv_iimax = G_ni+2*G_halox-2
   adv_jjmax = G_nj+G_haloy
   adv_nit = l_maxx - l_minx + 1
   adv_njt = l_maxy - l_miny + 1
   adv_nijag = adv_nit * adv_njt
   adv_lnij = l_ni*l_nj

    allocate( adv_xg_8(adv_gminx:adv_gmaxx),  adv_yg_8(adv_gminy:adv_gmaxy) , &
                adv_verz_8%m(0:l_nk+1),  adv_verz_8%t(1:l_nk),  stat = istat)
                
    call handle_error_l(istat==0,'','problem allocating mem')

     do i = 1,G_ni
    
      adv_xg_8(i) = G_xg_8(i)
    
     enddo


     do j = 1,G_nj
      adv_yg_8(j) = G_yg_8(j)
     enddo

     adv_verz_8%m(:) = v_zm_8(:)
     adv_verz_8%t(:) = v_zt_8(:)

     deallocate(v_zm_8, v_zt_8)


   call adv_set_grid   ()

   call adv_set_interp ()
  

  end subroutine adv_set


 
  subroutine adv_set_grid ()

   integer :: istat, i, j
   real*8 :: prhxmn, prhymn

   !-------------------------------------------------------------------
   allocate( &
        adv_xx_8(l_minx:l_maxx), &
        adv_cx_8(l_ni), &
        adv_sx_8(l_ni), &
        adv_yy_8(l_miny:l_maxy), &
        adv_vsec_8(l_miny:l_maxy), &
        adv_vtan_8(l_miny:l_maxy), &
        adv_cy_8(l_nj), &
        adv_sy_8(l_nj), &
        stat = istat)
   call handle_error_l(istat==0,'adv_set_grid','problem allocating mem')


      prhxmn =  adv_xg_8(2)-adv_xg_8(1)
      do i = 0,adv_gminx,-1
         adv_xg_8(i) = adv_xg_8(i+1)  - prhxmn
      
      enddo
      do i = G_ni+1,adv_gmaxx
         adv_xg_8(i) = adv_xg_8(i-1) + prhxmn
 
      enddo

      prhymn =  adv_yg_8(2)-adv_yg_8(1)
      do j = 0,adv_gminy,-1
         adv_yg_8(j) = adv_yg_8(j+1) - prhymn
      enddo
      do j = G_nj+1,adv_gmaxy
         adv_yg_8(j) = adv_yg_8(j-1) + prhymn
      enddo
      
   !- advection grid
   do i = l_minx,l_maxx
      adv_xx_8(i) = adv_xg_8(l_i0-1+i)
   enddo
   do j = l_miny,l_maxy
      adv_yy_8(j) = adv_yg_8(l_j0-1+j)
   enddo

   do j = l_miny,l_maxy
   !- precalculation vsec, vtan for grand circle computation
      adv_vsec_8(j) = 1.0D0/(cos(adv_yy_8(j)))
      adv_vtan_8(j) = tan(adv_yy_8(j))
   enddo

   do i = 1,l_ni
      adv_cx_8(i) = cos(adv_xx_8(i))
      adv_sx_8(i) = sin(adv_xx_8(i))
   enddo

   do j = 1,l_nj
      adv_cy_8(j) = cos(adv_yy_8(j))
      adv_sy_8(j) = sin(adv_yy_8(j))
   enddo
   !---------------------------------------------------------------------
   return
end subroutine adv_set_grid



  subroutine adv_set_interp ()

   implicit none
#include <arch_specific.hf>
   !@objective set 1-D interpolation of grid reflexion across the pole 
   !@author  alain patoine
   !@revisions
   !*@/
   !
   !     Example of level positions for a 5 layer model (l_nkm=5)
   !
   !     %m (Momentum)      %t (Thermo)        %s (Superwinds)
   ! top ===============    ===============    =============== 1
   !     --------------- 1                     --------------- 2
   !                        --------------- 1  --------------- 3
   !     --------------- 2                     --------------- 4
   !                        --------------- 2  --------------- 5
   !     --------------- 3                     --------------- 6
   !                        --------------- 3  --------------- 7
   !     --------------- 4                     --------------- 8
   !                        --------------- 4  --------------- 9
   !     --------------- 5                     --------------- 10
   ! Srf ===============    =============== 5  =============== 11
   !
   !        N levels           N levels           N+1 levels        2N+1 levels

#include "adv_dims.cdk"
#include "adv_grid.cdk"
#include "adv_interp.cdk"
   
   real*8, parameter :: LARGE_8 = 1.D20
   real*8, parameter :: FRAC1OV6_8 = 1.D0/6.D0

   integer :: i, j, k, ij, pnerr, trj_i_off, nij, n, istat, istat2, ind
   integer :: i0, j0, k0, pnx, pny, pnz, nkm, nkt

   real*8 :: ra,rb,rc,rd
   real*8 :: prhxmn, prhymn, prhzmn, dummy, pdfi

   real*8 ::whx(G_ni+2*G_halox)
   real*8 ::why(G_nj+2*G_haloy)

   real*8 :: qzz_m_8(3*l_nk), qzi_m_8(4*l_nk)
   real*8 :: qzz_t_8(3*l_nk), qzi_t_8(4*l_nk)

   real*8,dimension(:),pointer :: whzt,whzm

#if !defined(TRIPROD)
#define TRIPROD(za,zb,zc,zd) ((za-zb)*(za-zc)*(za-zd))
#endif
   !---------------------------------------------------------------------
   ! See drawing above
   nkm=l_nk
   nkt=l_nk
   

   allocate( &
        whzm(0:nkm+1), &
        whzt(0:nkt+1), &
       
       
        stat = istat)

   allocate( &
        adv_xbc_8(G_ni+2*G_halox), &   ! (xc-xb)     along x
        adv_xabcd_8(G_ni+2*G_halox), & ! triproducts along x
        adv_xbacd_8(G_ni+2*G_halox), &
        adv_xcabd_8(G_ni+2*G_halox), &
        adv_xdabc_8(G_ni+2*G_halox), &
        adv_ybc_8(G_nj+2*G_haloy), &   ! (yc-yb)     along y 
        adv_yabcd_8(G_nj+2*G_haloy), & ! triproducts along y
        adv_ybacd_8(G_nj+2*G_haloy), &
        adv_ycabd_8(G_nj+2*G_haloy), &
        adv_ydabc_8(G_nj+2*G_haloy), &
        stat = istat2)

   call handle_error_l(istat==0.and.istat2==0,'adv_set_interp','problem allocating mem')

   do i = adv_gminx+1,adv_gmaxx-2
      ra = adv_xg_8(i-1)
      rb = adv_xg_8(i)
      rc = adv_xg_8(i+1)
      rd = adv_xg_8(i+2)
      adv_xabcd_8(G_halox+i) = 1.D0/TRIPROD(ra,rb,rc,rd)
      adv_xbacd_8(G_halox+i) = 1.D0/TRIPROD(rb,ra,rc,rd)
      adv_xcabd_8(G_halox+i) = 1.D0/TRIPROD(rc,ra,rb,rd)
      adv_xdabc_8(G_halox+i) = 1.D0/TRIPROD(rd,ra,rb,rc)
   enddo

   do i = adv_gminx,adv_gmaxx-1
      rb = adv_xg_8(i)
      rc = adv_xg_8(i+1)
      adv_xbc_8(G_halox+i) = 1.D0/(rc-rb)
   enddo

   do j = adv_gminy+1,adv_gmaxy-2
      ra = adv_yg_8(j-1)
      rb = adv_yg_8(j)
      rc = adv_yg_8(j+1)
      rd = adv_yg_8(j+2)
      adv_yabcd_8(G_haloy+j) = 1.D0/TRIPROD(ra,rb,rc,rd)
      adv_ybacd_8(G_haloy+j) = 1.D0/TRIPROD(rb,ra,rc,rd)
      adv_ycabd_8(G_haloy+j) = 1.D0/TRIPROD(rc,ra,rb,rd)
      adv_ydabc_8(G_haloy+j) = 1.D0/TRIPROD(rd,ra,rb,rc)
   enddo

   do j = adv_gminy,adv_gmaxy-1
      rb = adv_yg_8(j)
      rc = adv_yg_8(j+1)
      adv_ybc_8(G_haloy+j) = 1.D0/(rc-rb)
   enddo


   trj_i_off = 0

   adv_x00_8 = adv_xg_8(adv_gminx)
   adv_y00_8 = adv_yg_8(adv_gminy)

   prhxmn = LARGE_8
   prhymn = LARGE_8
   prhzmn = LARGE_8

   do i = adv_gminx,adv_gmaxx-1
      whx(G_halox+i) = adv_xg_8(i+1) - adv_xg_8(i)
      prhxmn = min(whx(G_halox+i), prhxmn)
   enddo

   do j = adv_gminy,adv_gmaxy-1
      why(G_haloy+j) = adv_yg_8(j+1) - adv_yg_8(j)
      prhymn = min(why(G_haloy+j), prhymn)
   enddo

   ! Prepare zeta on super vertical grid
  
   whzt(0    ) = 1.0
   whzt(nkt  ) = 1.0
   whzt(nkt+1) = 1.0
   do k = 1,nkt-1
      whzt(k) = adv_verZ_8%t(k+1) - adv_verZ_8%t(k)
      prhzmn = min(whzt(k), prhzmn)
   enddo

   whzm(0    ) = 1.0
   whzm(nkm  ) = 1.0
   whzm(nkm+1) = 1.0
   do k = 1,nkm-1
      whzm(k) = adv_verZ_8%m(k+1) - adv_verZ_8%m(k)
      prhzmn = min(whzm(k), prhzmn)
   enddo

     
   adv_ovdx_8 = 1.0d0/prhxmn
   adv_ovdy_8 = 1.0d0/prhymn
   adv_ovdz_8 = 1.0d0/prhzmn

   pnx = int(1.0+(adv_xg_8(adv_gmaxx)-adv_x00_8)   *adv_ovdx_8)
   pny = int(1.0+(adv_yg_8(adv_gmaxy)-adv_y00_8)   *adv_ovdy_8)
   pnz = nint(1.0+(adv_verZ_8%m(nkm+1)-adv_verZ_8%m(0))*adv_ovdz_8)

   allocate( &
        adv_lcx(pnx), &
        adv_lcy(pny), &
        adv_bsx_8(G_ni+2*G_halox), &
        adv_dlx_8(G_ni+2*G_halox), &
      
        adv_bsy_8(G_nj+2*G_haloy), &
        adv_dly_8(G_nj+2*G_haloy), &
      
        adv_lcz%t(pnz), &
        adv_lcz%m(pnz),  &
               adv_bsz_8%t(0:nkt-1)  , &
        adv_bsz_8%m(0:nkm-1), &
                adv_dlz_8%t(-1:nkt) , &
        adv_dlz_8%m(-1:nkm), &
   
        stat=istat)
   call handle_error_l(istat==0,'adv_set_interp','problem allocating mem')

   i0 = 1
   do i=1,pnx
      pdfi = adv_xg_8(adv_gminx) + (i-1) * prhxmn
      if (pdfi > adv_xg_8(i0+1-G_halox)) i0 = min(G_ni+2*G_halox-1,i0+1)
      adv_lcx(i) = i0
   enddo
   do i = adv_gminx,adv_gmaxx-1
      adv_dlx_8(G_halox+i) =       whx(G_halox+i)
   enddo
   do i = adv_gminx,adv_gmaxx
      adv_bsx_8(G_halox+i) = adv_xg_8(i)
   enddo

   j0 = 1
   do j = 1,pny
      pdfi = adv_yg_8(adv_gminy) + (j-1) * prhymn
      if (pdfi > adv_yg_8(j0+1-G_haloy)) j0 = min(G_nj+2*G_haloy-1,j0+1)
      adv_lcy(j) = j0
   enddo
   do j = adv_gminy,adv_gmaxy-1
      adv_dly_8(G_haloy+j) =       why(G_haloy+j)
   enddo
   do j = adv_gminy,adv_gmaxy
      adv_bsy_8(G_haloy+j) = adv_yg_8(j)
   enddo

   k0 = 1

   do k = 1,pnz
      pdfi = adv_verZ_8%m(0) + (k-1) * prhzmn
      if (pdfi > adv_verZ_8%t(k0+1)) k0 = min(nkt-2, k0+1)
      adv_lcz%t(k) = k0
   enddo
   do k = 0,nkt+1                    !! warning note the shift in k !!
      adv_dlz_8%t(k-1) =       whzt(k)
   enddo
   do k = 1,nkt
      adv_bsz_8%t(k-1) = adv_verZ_8%t(k)
   enddo

   k0 = 1
   do k = 1,pnz
      pdfi = adv_verZ_8%m(0) + (k-1) * prhzmn
      if (pdfi > adv_verZ_8%m(k0+1)) k0 = min(nkm-2, k0+1)
      adv_lcz%m(k) = k0
   enddo
   do k = 0,nkm+1                    !! warning note the shift in k !!
      adv_dlz_8%m(k-1) =       whzm(k)
   enddo
   do k = 1,nkm
      adv_bsz_8%m(k-1) = adv_verZ_8%m(k)
   enddo

      
           
   allocate( &
        adv_zbc_8%t(nkt),   adv_zbc_8%m(nkm),    &
        adv_zabcd_8%t(nkt), adv_zabcd_8%m(nkm),  &
        adv_zbacd_8%t(nkt), adv_zbacd_8%m(nkm),  &
        adv_zcabd_8%t(nkt), adv_zcabd_8%m(nkm),  &
        adv_zdabc_8%t(nkt), adv_zdabc_8%m(nkm), &
        stat=istat)
   call handle_error_l(istat==0,'adv_set_interp','problem allocating mem')

   do k = 2,nkm-2
      ra = adv_verZ_8%m(k-1)
      rb = adv_verZ_8%m(k)
      rc = adv_verZ_8%m(k+1)
      rd = adv_verZ_8%m(k+2)

      adv_zabcd_8%m(k) = 1.0/TRIPROD(ra,rb,rc,rd)
      adv_zbacd_8%m(k) = 1.0/TRIPROD(rb,ra,rc,rd)
      adv_zcabd_8%m(k) = 1.0/TRIPROD(rc,ra,rb,rd)
      adv_zdabc_8%m(k) = 1.0/TRIPROD(rd,ra,rb,rc)
   enddo

   do k = 2,nkt-2
      ra = adv_verZ_8%t(k-1)
      rb = adv_verZ_8%t(k)
      rc = adv_verZ_8%t(k+1)
      rd = adv_verZ_8%t(k+2)

      adv_zabcd_8%t(k) = 1.0/TRIPROD(ra,rb,rc,rd)
      adv_zbacd_8%t(k) = 1.0/TRIPROD(rb,ra,rc,rd)
      adv_zcabd_8%t(k) = 1.0/TRIPROD(rc,ra,rb,rd)
      adv_zdabc_8%t(k) = 1.0/TRIPROD(rd,ra,rb,rc)
   enddo

   
   do k = 1,nkm-1
      rb = adv_verZ_8%m(k)
      rc = adv_verZ_8%m(k+1)
      adv_zbc_8%m(k) = 1.0/(rc-rb)
   enddo

   do k = 1,nkt-1
      rb = adv_verZ_8%t(k)
      rc = adv_verZ_8%t(k+1)
      adv_zbc_8%t(k) = 1.0/(rc-rb)
   enddo

   
   deallocate(whzt, whzm)
   !---------------------------------------------------------------------
   return
end subroutine adv_set_interp


end module advection_set_mod
 

