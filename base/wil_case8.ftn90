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
!**s/r wil_case8: Galewsky's barotropic wave
!


      subroutine wil_case8 (gz_temp,lni,lnj,nk)
      implicit none
#include <arch_specific.hf>
!
      integer lni,lnj,nk
      real    gz_temp(lni,lnj,nk)
!
!author
!     Abdessamad Qaddouri and Vivian Lee
!
!revision
!
!object
!   To setup for Yin-Yang/Global model: The Galewvsky's barotropic wave 
!   Documentation: Qaddouri et al. 
!       "Experiment with different discretization for the Shallow-Water 
!        equations on the sphere" (2011) Quart.J.Roy.Meteor.Soc.
!
#include "lun.cdk"
#include "glb_ld.cdk"
#include "dcst.cdk"
#include "ptopo.cdk"
!
      integer i,j,k,lniu,lnju,lniv,lnjv
      real    PICLL(G_ni,G_nj),gzloc(l_minx:l_maxx,l_miny:l_maxy)
      real*8 lat2_8,alph_8,beta_8,hhat_8
      real*8 ratio1_8,ratio2_8,expos1_8,expos2_8,rad2deg_8,RLON
!
      real*8 ONE_8, CLXXX_8
      parameter( ONE_8  = 1.0 , CLXXX_8 = 180.0 )
      REAL*8  RLAT,TIME, SINT,COST,PHIAY,PHIBY,PHICY
      real*8  s(2,2),x_a,y_a,SINL,COSL
      real*8  wil_galewski_geo_8,hmean_ref_8,hmean_8
      real*8  wil_galewski_mean_8,latmean__8,xxx
      external wil_galewski_geo_8,wil_galewski_mean_8
!
!     -----------------------------------------------------------
!
      if (Lun_out.gt.0) write (Lun_out,1000)
!
      rad2deg_8 = CLXXX_8 /acos( -ONE_8 )
!
!     Bump is centred in longitude on 0E
!     ----------------------------------
      lat2_8 = Dcst_pi_8/4.
      alph_8 = 1./3.
      beta_8 = 1./15.
      hhat_8 = 120.
      hmean_ref_8 = 10.*1000.
      latmean__8 = Dcst_pi_8/2.
      hmean_8 = wil_galewski_mean_8 (Dcst_pi_8/2.)
!
      PICLL(:,:) = 0.
!
      if (Ptopo_couleur.eq.0) then !Yin

          print*,'G_ni,G_nj',G_ni,G_nj
          print*,' G_yg_8(J',( G_yg_8(J),J=1,G_nj)

          DO 151 J=1,G_nj
            RLAT = G_yg_8(J)
            COST = COS(RLAT)
            SINT = SIN(RLAT)
            DO 150 I=1,G_ni
               RLON = G_xg_8(I)
               SINL = SIN(RLON)
               COSL = COS(RLON)
!
!            NOTE: Use Latitudes SOUTH to NORTH as GEM 
!                  (not as Williamson's cases)
!            --------------------------------------------
!

               PICLL(i,j)= wil_galewski_geo_8(RLAT)
               PICLL(i,j)      =   PICLL(i,j)- hmean_8 + &
                               Dcst_grav_8*hmean_ref_8 
               if (RLON.gt. Dcst_pi_8) RLON = RLON - 2.*Dcst_pi_8
!
               if (RLON.gt.-Dcst_pi_8.and.RLON.lt.Dcst_pi_8) then
!
                   ratio1_8 = RLON/alph_8
                   expos1_8 = - ratio1_8**2
!
!            NOTE: Use Latitudes SOUTH to NORTH as GEM 
!                  (not as Williamson's cases)
!            --------------------------------------------
                   ratio2_8 = lat2_8 - RLAT
                   ratio2_8 = ratio2_8/beta_8
                   expos2_8 = - ratio2_8**2
                   xxx=hhat_8*exp(expos1_8)*exp(expos2_8) !perturbation
!
               endif
               PICLL(i,j)=PICLL(i,j)/Dcst_grav_8+xxx
!
!
  150       CONTINUE

  151     CONTINUE


      else !Yang
          DO 153 J=1,G_nj

            DO 152 I=1,G_ni
               x_a = G_xg_8(I)-acos(-1.D0)
               y_a = G_yg_8(J)
               call smat(s,RLON,RLAT,x_a,y_a)
               RLON = RLON+acos(-1.D0)
               SINT = SIN(RLAT)
               COST = COS(RLAT)

               SINL = SIN(RLON)

               COSL = COS(RLON)
!
!            NOTE: Use Latitudes SOUTH to NORTH as GEM 
!                  (not as Williamson's cases)
!            --------------------------------------------
!
               PICLL(i,j)= wil_galewski_geo_8(RLAT)

               PICLL(i,j)= PICLL(i,j)- hmean_8 + &
                               Dcst_grav_8*hmean_ref_8    

               if (RLON.gt. Dcst_pi_8) RLON = RLON - 2.*Dcst_pi_8
!
               if (RLON.gt.-Dcst_pi_8.and.RLON.lt.Dcst_pi_8) then
!
                   ratio1_8 = RLON/alph_8
                   expos1_8 = - ratio1_8**2
!
!            NOTE: Use Latitudes SOUTH to NORTH as GEM 
!                  (not as Williamson's cases)
!            --------------------------------------------
!
                   ratio2_8 = lat2_8 - RLAT
                   ratio2_8 = ratio2_8/beta_8
                   expos2_8 = - ratio2_8**2
                   xxx=hhat_8*exp(expos1_8)*exp(expos2_8) !perturbation
!
               endif
               PICLL(i,j)=PICLL(i,j)/Dcst_grav_8+xxx
!
  152       CONTINUE

  153     CONTINUE

      endif ! Yang

      call glbdist (PICLL,G_ni,G_nj,gzloc,l_minx,l_maxx,l_miny,l_maxy, &
                               1,G_halox,G_haloy)

      do k=1,nk
      do j=1,lnj
      do i=1,lni
         gz_temp(i,j,k)=gzloc(i,j)
      enddo
      enddo
      enddo
!
!     -----------------------------------------------------------
 1000  format(3X,' Add perturbation to HEIGHT field (bump localized in longitude): (S/R WIL_CASE8(Galewsky Barotropic wave)')
!     -----------------------------------------------------------
!
      return
      end
