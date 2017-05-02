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
!/@*
      subroutine itf_phy_update3 (F_apply_L)
      use phy_itf, only: phy_get
      use gmm_vt1
      implicit none
#include <arch_specific.hf>

      logical,intent(in) :: F_apply_L

!authors 
!     Desgagne, McTaggart-Cowan, Chamberland -- Spring 2014
!
!revision
! v4_70 - authors          - initial version
! v4_XX - Tanguay M.       - SOURCE_PS: REAL*8 with iterations
   
#include <gmm.hf>
#include "glb_ld.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "schm.cdk"
#include "tr3d.cdk"
#include "pw.cdk"
#include "lun.cdk"

      character(len=GMM_MAXNAMELENGTH) :: trname_S
      integer nelements, init, busidx, istat, i,j,k,n, cnt, iteration
      real, dimension(:,:,:), pointer :: data3d,minus,ptr3d
      real, dimension(l_minx:l_maxx,l_miny:l_maxy,G_nk), target :: tdu,tdv
      real,  dimension(l_ni,l_nj,G_nk) :: qw_phy,qw_dyn
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy)        :: pr_p0_8
      real*8,dimension(l_minx:l_maxx,l_miny:l_maxy,G_nk+1) :: pr_m_dyn_8,pr_m_phy_8,pr_t_8
!
!-----------------------------------------------------------------
!
   if (F_apply_L) then

      if (Schm_source_ps_L) then
         qw_phy = 0. ; qw_dyn = 0.
         do n= 1, Tr3d_ntr
            trname_S = 'TR/'//trim(Tr3d_name_S(n))//':P'
            istat = gmm_get(trim(trname_S),data3d)
            if ( (Tr3d_name_S(n)(1:2).eq.'HU') .or. &
                 (Schm_wload_L.and.Tr3d_wload(n)) )then
               ptr3d => tdu(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
               if ( phy_get (ptr3d, trim(trname_S), F_npath='V', F_bpath='D', &
                             F_end=(/-1,-1,l_nk/), F_quiet=.true.) .lt. 0 ) cycle
!$omp parallel
!$omp do
               do k=1, l_nk
               do j=1+pil_s,l_nj-pil_n
               do i=1+pil_w,l_ni-pil_e
                  qw_phy(i,j,k)= qw_phy(i,j,k) +    tdu(i,j,k)
                  qw_dyn(i,j,k)= qw_dyn(i,j,k) + data3d(i,j,k)
                  data3d(i,j,k)= tdu   (i,j,k)
               enddo
               enddo
               enddo 
!$omp enddo
!$omp end parallel
            else
               ptr3d => data3d(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
               istat = phy_get (ptr3d, trim(trname_S), F_npath='V', F_bpath='D',&
                                           F_end=(/-1,-1,l_nk/), F_quiet=.true. )
            endif
         enddo
      else
         do k= 1, Tr3d_ntr
            trname_S = 'TR/'//trim(Tr3d_name_S(k))//':P'
            istat = gmm_get(trim(trname_S),data3d)
            ptr3d => data3d(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
            istat = phy_get ( ptr3d, trim(trname_S), F_npath='V', F_bpath='D',&
                              F_end=(/-1,-1,l_nk/), F_quiet=.true. )
         enddo
      endif

      istat = gmm_get (gmmk_pw_uu_plus_s,pw_uu_plus)
      istat = gmm_get (gmmk_pw_vv_plus_s,pw_vv_plus)
      istat = gmm_get (gmmk_pw_tt_plus_s,pw_tt_plus)

      ptr3d => pw_uu_plus(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
      istat = phy_get(ptr3d,gmmk_pw_uu_plus_s,F_npath='V',F_bpath='D',F_end=(/-1,-1,l_nk/))

      ptr3d => pw_vv_plus(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
      istat = phy_get(ptr3d,gmmk_pw_vv_plus_s,F_npath='V',F_bpath='D',F_end=(/-1,-1,l_nk/))

      ptr3d => pw_tt_plus(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
      istat = phy_get(ptr3d,gmmk_pw_tt_plus_s,F_npath='V',F_bpath='D',F_end=(/-1,-1,l_nk/))


      if (Schm_source_ps_L) then

         iteration = 1

         istat = gmm_get(gmmk_st1_s,st1)

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 (pr_m_dyn_8,pr_t_8,pr_p0_8,st1,l_minx,l_maxx,l_miny,l_maxy,l_nk)

         pr_m_dyn_8(:,:,l_nk+1) = pr_p0_8(:,:)

     800 continue

         !Obtain pressure levels
         !----------------------
         call calc_pressure_8 (pr_m_phy_8,pr_t_8,pr_p0_8,st1,l_minx,l_maxx,l_miny,l_maxy,l_nk)

         pr_m_phy_8(:,:,l_nk+1) = pr_p0_8(:,:)

         pr_p0_8(:,:) = pr_m_phy_8(:,:,1)

         !Estimate source of surface pressure due to fluxes of water:
         !-----------------------------------------------------------------------------------------------------
         !Vertical_Integral [d(p_phy)_k+1] = Vertical_Integral [ d(p_phy)_k q_phy + d(p_dyn) (1-q_dyn) based on
         !-----------------------------------------------------------------------------------------------------
         !d(ps) = Vertical_Integral [ d(qw)/(1-qw_phy)] d(pi) (Claude Girard) 
         !-----------------------------------------------------------------------------------------------------
!$omp parallel do shared(qw_dyn,qw_phy,pr_m_dyn_8,pr_m_phy_8,pr_p0_8)
         do j=1+pil_s,l_nj-pil_n
            do k=1,l_nk
               do i=1+pil_w,l_ni-pil_e
                  pr_p0_8(i,j)= pr_p0_8(i,j) + (1.0-qw_dyn(i,j,k)) * (pr_m_dyn_8(i,j,k+1)-pr_m_dyn_8(i,j,k)) + &
                                                    qw_phy(i,j,k)  * (pr_m_phy_8(i,j,k+1)-pr_m_phy_8(i,j,k))
               enddo
            enddo
         enddo
!$omp end parallel do

!$omp parallel do shared(pr_p0_8,st1)
         do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            st1(i,j)= log(pr_p0_8(i,j)/Cstv_pref_8)
         end do
         end do
!$omp end parallel do 

         iteration = iteration + 1

         if (iteration<4) goto 800

         if (Lun_out>0) write(Lun_out,*) ''
         if (Lun_out>0) write(Lun_out,*) '--------------------------------------'
         if (Lun_out>0) write(Lun_out,*) 'SOURCE_PS is done for DRY AIR (REAL*8)'
         if (Lun_out>0) write(Lun_out,*) '--------------------------------------'
         if (Lun_out>0) write(Lun_out,*) ''

         call pw_update_GPW() 

      endif

   else

      cnt = 0
      do k= 1, Tr3d_ntr
         if (trim(Tr3d_name_S(k)) == 'HU' .or.                          &
             any(NTR_Tr3d_name_S(1:NTR_Tr3d_ntr)==trim(Tr3d_name_S(k))))&
             cycle
         trname_S = 'TR/'//trim(Tr3d_name_S(k))//':P'
         istat = gmm_get(trim(trname_S),data3d)
         ptr3d => data3d(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,1:l_nk)
         if ( phy_get ( ptr3d, trim(trname_S), F_npath='V', F_bpath='D',&
                        F_end=(/-1,-1,l_nk/), F_quiet=.true. ) .lt.0 ) cycle
         trname_S = 'TR/'//trim(Tr3d_name_S(k))//':M'
         if (Grd_yinyang_L) &
         call yyg_xchng (data3d, l_minx,l_maxx,l_miny,l_maxy, &
                         G_nk,.true., 'CUBIC')
         
         istat = gmm_get(trim(trname_S),minus)
         minus = data3d
         cnt   = cnt + 1
      end do

      if (cnt.gt.0) then
         istat = gmm_get(gmmk_tt1_s, tt1)
         call tt2virt2 (tt1, .true., l_minx,l_maxx,l_miny,l_maxy,l_nk)
         if (Grd_yinyang_L) then
            call yyg_xchng (tt1, l_minx,l_maxx,l_miny,l_maxy, &
                            G_nk, .false., 'CUBIC')
            call pw_update_T
         endif
         call pw_update_GPW
      endif

   endif
!
!-----------------------------------------------------------------
!
   return
   end subroutine itf_phy_update3
