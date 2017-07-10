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

!**s/r hzd_exp_set

      subroutine hzd_exp_set
      use hzd_mod
      use gem_options
      use tdpack
      implicit none
#include "cstv.cdk"
#include <arch_specific.hf>


#include "glb_ld.cdk"
#include "lun.cdk"
#include "ver.cdk"

      character*256 str1
      integer i,j,k,ind1,ind2,npin,nvalid
      integer, dimension(:) , allocatable :: pwr, lvl
      real   , dimension(:) , allocatable :: lnr
      real*8 c_8, x1, x2, rr, weight
      real levhyb
      real*8 pt25,nudif,epsilon
      parameter (epsilon = 1.0d-12, pt25=0.25d0)
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(Lun_out,1002)

      Hzd_lnr = min(max(0.,Hzd_lnr),0.9999999)
      Hzd_pwr = Hzd_pwr / 2
      Hzd_pwr = min(max(2,Hzd_pwr*2),8)
      
      Hzd_lnr_theta= min(max(0.,Hzd_lnr_theta),0.9999999)
      Hzd_pwr_theta= Hzd_pwr_theta / 2
      Hzd_pwr_theta= min(max(2,Hzd_pwr_theta*2),8)
      
      if (Hzd_lnr_tr.lt.0.) Hzd_lnr_tr = Hzd_lnr
      if (Hzd_pwr_tr.lt.0 ) Hzd_pwr_tr = Hzd_pwr
      Hzd_lnr_tr = min(max(0.,Hzd_lnr_tr),0.9999999)
      Hzd_pwr_tr = Hzd_pwr_tr / 2
      Hzd_pwr_tr = min(max(2,Hzd_pwr_tr*2),8)

      if ((Hzd_lnr.le.0.).and.(Hzd_lnr_theta.le.0.)  &
                         .and.(Hzd_lnr_tr   .le.0.)) then
         if((Hzd_smago_param.le.0.).and.(Hzd_smago_lnr(2).eq.0.)) then
            if (Lun_out.gt.0) write(Lun_out,1003)
         else
            if (Lun_out.gt.0) write(Lun_out,1004) &
                              Hzd_smago_param,100*Hzd_smago_lnr(2)
         endif
      endif

      call hzd_exp_geom

      call hzd_exp5p_set

 1002 format(/,'INITIALIZATING HIGH ORDER HORIZONTAL DIFFUSION ',  &
               '(S/R HZD_SET)',/,60('='))
 1003 format(/,'NO HORIZONTAL DIFFUSION REQUESTED',/,33('='))
 1004 format(/,'  HORIZONTAL DIFFUSION A LA SMAGORINSKY',/,2x,37('=')// &
              ,'  PARAMETER =',f5.2,'  BACKGROUND =',f4.1,' %/TIMESTEP')
!
!     ---------------------------------------------------------------
!
      return
      end

      subroutine sorthzdexp (F_lvl,F_pwr,F_lnr,nk)
      implicit none
#include <arch_specific.hf>

      integer nk
      integer F_lvl(nk),F_pwr(nk)
      real    F_lnr(nk)

      integer k,i,m,j,n,x1
      real x2
!
!----------------------------------------------------------------------
!
      n = nk
      do i = 1, n-1
         k = i
         do j = i+1, n
            if (F_lvl(k) .gt. F_lvl(j))  k=j
         enddo
         if (k .ne. i) then
            x2     = F_lnr(k)
            x1     = F_lvl(k)
            m      = F_pwr(k)
            F_lvl(k) = F_lvl(i)
            F_pwr(k) = F_pwr(i)
            F_lnr(k) = F_lnr(i)
            F_lvl(i) = x1
            F_pwr(i) = m
            F_lnr(i) = x2
         endif
      enddo
!
!----------------------------------------------------------------------
!
      return
      end
