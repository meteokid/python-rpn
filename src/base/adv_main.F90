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
      subroutine adv_main ( F_fnitraj, F_icn             ,&
                         ut0, vt0, zdt0, ut1, vt1, zdt1  ,&
                         orhsu, rhsu, orhsv, rhsv, orhsc ,&
                         rhsc, orhst,  rhst, orhsf, rhsf ,&
                         orhsq, rhsq, orhsw, rhsw, orhsx ,&
                         rhsx, Minx,Maxx,Miny,Maxy, Nk )
      implicit none
#include <arch_specific.hf>
      
      integer,intent(in) ::  F_fnitraj, F_icn
      integer,intent(in) ::  Minx,Maxx,Miny,Maxy, Nk
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: ut0, vt0, zdt0 ! winds at time t0
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: ut1, vt1, zdt1 ! winds at time t1
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(in) :: orhsu, orhsv, orhsc, orhst, orhsf, orhsq, orhsw, orhsx                                              
      real, dimension(Minx:Maxx,Miny:Maxy,NK),intent(inout) :: rhsu, rhsv, rhsc,  rhst, rhsf, rhsq, rhsw, rhsx

#include "glb_ld.cdk"
#include "grd.cdk"
#include <gmm.hf>    
#include "lam.cdk"
#include "schm.cdk"
#include "lctl.cdk"
#include "step.cdk"
#include "adv_pos.cdk"
#include "vth.cdk"
      logical, save :: setgrid = .false. ! 
      logical :: doAdwStat_L
      integer :: i0,j0,in,jn,i0u,inu,j0v,jnv ! advection computational i,j,k domain  (glb_ld.cdk)
      integer :: k0, k0m, k0t, err, jext
      real, dimension(:,:,:), allocatable  :: pxm, pym, pzm ! upstream positions valid at time t1
      real, dimension(:,:,:), allocatable  :: ua,va,wa,wat  ! arrival winds
      real, dimension(:,:,:), allocatable  :: ud,vd,wd      ! de-staggered   departure winds 
!      real, pointer, dimension(:) :: xth, yth, zth
!
!     ---------------------------------------------------------------
!      
      doAdwStat_L = .false.
      if (Step_gstat.gt.0) then
         doAdwStat_L = (mod(Lctl_step,Step_gstat) == 0)
      end if
      doAdwStat_L = doAdwStat_L .and. (F_icn == Schm_itcn)
      
      allocate ( pxm(l_ni,l_nj,l_nk), pym(l_ni,l_nj,l_nk), pzm(l_ni,l_nj,l_nk),&
                 ua (l_ni,l_nj,l_nk), va (l_ni,l_nj,l_nk), wa (l_ni,l_nj,l_nk),&
                 wat(l_ni,l_nj,l_nk) )
      allocate ( ud(l_minx:l_maxx,l_miny:l_maxy,l_nk),&
                 vd(l_minx:l_maxx,l_miny:l_maxy,l_nk),&
                 wd(l_minx:l_maxx,l_miny:l_maxy,l_nk) )
 	   if (.not.associated(pxt)) allocate (pxt(l_ni,l_nj,l_nk), &
                                       pyt(l_ni,l_nj,l_nk), &
                                       pzt(l_ni,l_nj,l_nk), &
                                       pxmu(l_ni,l_nj,l_nk), &
                                       pymu(l_ni,l_nj,l_nk), &
                                       pzmu(l_ni,l_nj,l_nk), &
                                       pxmv(l_ni,l_nj,l_nk), &
                                       pymv(l_ni,l_nj,l_nk), &
                                       pzmv(l_ni,l_nj,l_nk)  )
     nullify (xth, yth, zth)
     err = gmm_get(gmmk_xth_s , xth)
     err = gmm_get(gmmk_yth_s , yth)
     err = gmm_get(gmmk_zth_s , zth)
	  
      
!     Set advection grid & get parameters for interpolation 
      if (.not. setgrid ) then
         call adv_setgrid ()
         call adv_param   ()
         setgrid = .true.
      endif
      
!     Get advection computational i,j,k domain
      i0 = 1 ; in = l_ni
      j0 = 1 ; jn = l_nj

      jext=1
      if (Grd_yinyang_L) jext=2
      if (l_west)  i0 =        pil_w - jext
      if (l_east)  in = l_ni - pil_e + jext
      if (l_south) j0 =        pil_s - jext
      if (l_north) jn = l_nj - pil_n + jext
      i0u = i0 ;  inu = in
      j0v = j0 ;  jnv = jn

      jext = 2
      if (Grd_yinyang_L) jext = 0
      if (l_west)  i0u= i0 + jext
      if (l_east)  inu= in - jext
      if (l_south) j0v= j0 + jext
      if (l_north) jnv= jn - jext

      k0 = Lam_gbpil_t+1
      k0t= k0     
      if(Lam_gbpil_t.gt.0) k0t=k0-1
      k0m=max(k0t-2,1)

 
!     Process winds in preparation for SL advection: unstagger & interpolate from Thermo to Momentum levels
      call adv_prepareWinds ( ud, vd, wd, ua, va, wa, wat    , &
                              ut0, vt0 , zdt0, ut1, vt1, zdt1, &
                              l_minx, l_maxx, l_miny, l_maxy , &
                              l_ni , l_nj , l_nk )
 
!     calculate upstream positions at t1 using angular displacement & trapezoidal rule

      call adv_trapeze (F_fnitraj, pxm , pym , pzm,  &
                        ud, vd, wd, ua, va ,wa , wat  ,&
                        xth, yth, zth, i0, in, j0, jn, i0u, &
                        inu, j0v, jnv, k0, k0m, k0t,&
                        l_minx, l_maxx, l_miny, l_maxy, l_ni, l_nj, l_nk )

 
!     RHS interpolation : momentum levels                            
      call adv_cubic('RHSU_S', rhsu, orhsu, pxmu, pymu, pzmu, l_ni, l_nj, l_nk, &
                         l_minx, l_maxx, l_miny, l_maxy, i0u, inu, j0, jn, k0, 'm',0,0)    
 
      call adv_cubic('RHSV_S', rhsv, orhsv, pxmv, pymv, pzmv, l_ni , l_nj , l_nk, &
                         l_minx, l_maxx, l_miny, l_maxy, i0, in, j0v, jnv, k0, 'm',0,0)      

      call adv_cubic('RHSC_S', rhsc ,orhsc , pxm, pym, pzm , l_ni , l_nj , l_nk, &
                         l_minx, l_maxx, l_miny, l_maxy, i0, in, j0, jn, k0, 'm',0,0)

!     RHS Interpolation: thermo levels       
      call adv_cubic('RHST_S', rhst, orhst, pxt, pyt, pzt, l_ni, l_nj, l_nk, &
                        l_minx, l_maxx, l_miny,  l_maxy,i0, in, j0, jn, k0t, 't',0,0)

      call adv_cubic('RHSX_S', rhsx, orhsx, pxt, pyt, pzt,  l_ni , l_nj , l_nk, &
                        l_minx, l_maxx, l_miny,  l_maxy,i0, in, j0, jn, k0t, 't',0,0)      

      if(.not.Schm_hydro_L) then
         call adv_cubic('RHSF_S', rhsf, orhsf, pxt, pyt, pzt, l_ni, l_nj, l_nk, &
                        l_minx, l_maxx, l_miny,l_maxy, i0, in, j0, jn, k0t, 't',0,0)    
  
         call adv_cubic('RHSW_S', rhsw ,orhsw , pxt, pyt, pzt,  l_ni , l_nj , l_nk,&
                        l_minx, l_maxx, l_miny,l_maxy, i0, in, j0, jn, k0t, 't',0,0)

         call adv_cubic('RHSQ_S', rhsq ,orhsq , pxt, pyt, pzt,  l_ni , l_nj , l_nk,&
                        l_minx, l_maxx, l_miny,l_maxy, i0, in, j0, jn, k0t, 't',0,0)
      endif
      
! Compute Courant numbers (CFL) for stats 

     if ( doAdwStat_L ) then 
      call  adv_cfl_lam3 (pxm, pym, pzm, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'m')
      call  adv_cfl_lam3 (pxt, pyt, pzt, i0,in,j0,jn, l_ni,l_nj,k0,l_nk,'t')                                        
     endif


      deallocate (pxm,pym,pzm,ua,va,wa,wat,ud,vd,wd)
!
!     ---------------------------------------------------------------
!      
      end subroutine adv_main
