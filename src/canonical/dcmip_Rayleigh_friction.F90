!--------------------------------- LICENCE BEGIN -------------------------------
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

!**s/r dcmip_Rayleigh_friction - Apply Rayleigh Friction (Based on subr. height_sponge) 
 
      subroutine dcmip_Rayleigh_friction
 
      use gmm_vt1
      use canonical
      use dcmip_options 

      use glb_ld
      use cstv
      use lun
      use gmm_itf_mod
      implicit none

      !object
      !================================================================
      !   Apply Rayleigh Friction 
      !   -------------------------------------------------------------
      !   NOTE: The damping is not applied on zd and probably should be  
      !================================================================
    
 
      !-----------------------------------------------------------------------

      integer istat,i,j,k

      real w_fcu(l_minx:l_maxx,l_miny:l_maxy,l_nk), & 
           w_fcv(l_minx:l_maxx,l_miny:l_maxy,l_nk), &
           w_fcw(l_minx:l_maxx,l_miny:l_maxy,l_nk)

      !-----------------------------------------------------------------------

      if (.NOT.(Dcmip_case== 20.or. &
                Dcmip_case== 21.or. &
                Dcmip_case== 22.or. &
                Dcmip_case==163)) call handle_error(-1,'DCMIP_RAYLEIGH_FRICTION','FCU FCV FCW need to be prescribed') 

      if (Lun_out.gt.0) write (Lun_out,1000) 
 
      !Recover Winds u,v,w to be treated by Rayleigh damped layer 
      !----------------------------------------------------------
      istat = gmm_get(gmmk_ut1_s,ut1)
      istat = gmm_get(gmmk_vt1_s,vt1)
      istat = gmm_get(gmmk_wt1_s,wt1)

      !Recover Winds u,v,w REFERENCE for Rayleigh damped layer
      !-------------------------------------------------------
      istat = gmm_get(gmmk_uref_s,uref)
      istat = gmm_get(gmmk_vref_s,vref)
      istat = gmm_get(gmmk_wref_s,wref)

      !Recover Factors u,v,w for Rayleigh damped layer (linked to vertical grid dependence) 
      !------------------------------------------------------------------------------------
      istat = gmm_get(gmmk_fcu_s,fcu)
      istat = gmm_get(gmmk_fcv_s,fcv)
      istat = gmm_get(gmmk_fcw_s,fcw)

      w_fcu = fcu
      w_fcv = fcv
      w_fcw = fcw

      call dcmip_apply ( ut1, uref, w_fcu, l_minx, l_maxx, l_miny, l_maxy, l_niu, l_nj, l_nk)
      call dcmip_apply ( vt1, vref, w_fcv, l_minx, l_maxx, l_miny, l_maxy, l_ni,  l_njv,l_nk)
      call dcmip_apply ( wt1, wref, w_fcw, l_minx, l_maxx, l_miny, l_maxy, l_ni,  l_nj, l_nk)

      !---------------------------------------------------------------
 
      return

 1000 format( &
      /,'APPLY RAYLEIGH FRICTION: (S/R DCMIP_RAYLEIGH FRICTION)',   &
      /,'======================================================',/,/)

      end subroutine dcmip_Rayleigh_friction  

!==================================================================================

!**s/r dcmip_apply - Apply Rayleigh Friction for a given field

      subroutine dcmip_apply (F_ff,F_ffref,F_damp,Minx,Maxx,Miny,Maxy,Ni,Nj,Nk) ! Based on subr. apply 

      use glb_ld
      use cstv
      use lun
      use gmm_itf_mod
      implicit none

      !object
      !============================================
      !   Apply Rayleigh Friction for a given field 
      !============================================

      integer  Minx,Maxx,Miny,Maxy,Ni,Nj,Nk 

      real F_ff(Minx:Maxx,Miny:Maxy,Nk),F_ffref(Minx:Maxx,Miny:Maxy,Nk),F_damp(Minx:Maxx,Miny:Maxy,Nk)

      !Local variables
      !---------------
      integer i,j,k

      do k=1,Nk
         do j=1,Nj
            do i=1,Ni
               F_ff(i,j,k) = (1.-F_damp(i,j,k))*F_ff(i,j,k) + F_damp(i,j,k)*F_ffref(i,j,k)
            enddo
         enddo
      enddo

      !---------------------------------------------------------------
      
      return

      end subroutine dcmip_apply  
