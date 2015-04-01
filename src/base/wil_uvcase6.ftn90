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
!**s/r wil_uvcase6
!

!
      subroutine wil_uvcase6( u_temp,lniu,lnju,v_temp,lniv,lnjv,nk)
      implicit none
#include <arch_specific.hf>
      integer lniu,lnju,lniv,lnjv,nk
      real    u_temp(lniu,lnju,nk), v_temp(lniv,lnjv,nk)
!
!author 
!     Abdessamad Qaddouri & Vivian Lee 
!
!revision
!
!object
!     To setup williamson case 6 ROSSBY-HAURWITZ WAVE for Yin-Yang/Global model
!   Documentation: Nonlinear shallow-water equations on the Yin Yang grid. 
!  Quart.J.Roy.Meteor.Soc., 137, 656, 810-818.
!	
!arguments
!	none
!

#include "glb_ld.cdk"
#include "grd.cdk"
#include "dcst.cdk"
#include "schm.cdk"
#include "ptopo.cdk"

!
      integer i,j,k,R_case
      REAL*8  DLON,K_Case,OMG
      real    uloc(l_minx:l_maxx,l_miny:l_maxy),vloc(l_minx:l_maxx,l_miny:l_maxy)
      real    UICLL(G_ni,G_nj),VICLL(G_ni,G_nj)
      real*8  UI_U(G_niu,G_nj),UI_V(G_ni,G_njv)
      real*8  VI_U(G_niu,G_nj),VI_V(G_ni,G_njv)
      REAL*8  RLON,RLAT,TIME, SINT,COST
      real*8  s(2,2),x_a,y_a
      real*8  xgu_8(G_niu),ygv_8(G_njv)

!*
!     ---------------------------------------------------------------
!
!     ROSSBY-HAURWITZ WAVE AS USED BY PHILIPS IN
!     MONTHLY WEATHER REVIEW, 1959
!     
         time=0.0
         R_Case = 4
         K_Case = 7.848E-6
         OMG = 7.848E-6
         DLON = (R_Case*(3+R_Case)*OMG - 2.0*Dcst_omega_8)/ &
                ((1+R_Case)*(2+R_Case))*TIME

!        U grid
         do i=1,G_niu
            xgu_8(i)=(G_xg_8(i+1)+G_xg_8(i))*.5
         enddo
!        V grid
         do i=1,G_njv
            ygv_8(i)=(G_yg_8(i+1)+G_yg_8(i))*.5
         enddo
!
!        COMPUTE U VECTOR FOR YIN
!        ---------------------------------------------------
         if (Ptopo_couleur.eq.0) then
!
!        LONGITUDINAL CHANGE OF FEATURE
!        ------------------------------
!

         DO 151 J=1,G_nj
            RLAT = G_yg_8(J)
            DO 150 I=1,G_niu
               RLON = xgu_8(I)
               SINT = SIN(RLAT)
               COST = COS(RLAT)
               UICLL(I,J) = Dcst_rayt_8*OMG*COST +                 &
                            Dcst_rayt_8*K_Case*COST**(R_Case-1)*   &
                (R_Case*SINT*SINT-COST*COST)*COS(R_Case*(RLON-DLON))
  150       CONTINUE
  151    CONTINUE

         else

!        COMPUTE U VECTOR FOR YAN
!        ---------------------------------------------------
!
!
         DO 153 J=1,G_nj
               y_a = G_yg_8(J)
            DO 152 I=1,G_niu
               x_a = xgu_8(I)-acos(-1.D0)
               call smat(s,RLON,RLAT,x_a,y_a)
               RLON = RLON+acos(-1.D0)
               SINT = SIN(RLAT)
               COST = COS(RLAT)
               UI_u(I,J) = Dcst_rayt_8*OMG*COST +               &
                           Dcst_rayt_8*K_Case*COST**(R_Case-1)* &
              (R_Case*SINT*SINT-COST*COST)*COS(R_Case*(RLON-DLON))
               VI_u(I,J) = -Dcst_rayt_8*K_Case*R_Case*COST**(R_Case-1)*SINT &
                            * SIN(R_Case*(RLON-DLON))
               UICLL(I,J) = s(1,1)*UI_u(I,J) + s(1,2)*VI_u(I,J)
  152       CONTINUE
  153    CONTINUE

         endif

         call glbdist (UICLL,G_ni,G_nj,uloc,l_minx,l_maxx,l_miny,l_maxy, &
                               1,G_halox,G_haloy)
         do k=1,nk
         do j=1,lnju
         do i=1,lniu
            u_temp(i,j,k)=uloc(i,j)
         enddo
         enddo
         enddo
!-------------------------------------------------------------------------
!
!        COMPUTE V VECTOR FOR YIN
!        ---------------------------------------------------
         if (Ptopo_couleur.eq.0) then
!
!        LONGITUDINAL CHANGE OF FEATURE
!        ------------------------------
!

         DO 161 J=1,G_njv
            RLAT = ygv_8(J)
            DO 160 I=1,G_ni
               RLON = G_xg_8(I)
               SINT = SIN(RLAT)
               COST = COS(RLAT)

               VICLL(I,J) = -Dcst_rayt_8*K_Case*R_Case*COST**(R_Case-1)*SINT &
                            * SIN(R_Case*(RLON-DLON))
  160       CONTINUE
  161    CONTINUE

         else

!        COMPUTE V VECTOR FOR YAN
!        ---------------------------------------------------
!
!
         DO 163 J=1,G_njv

            y_a = ygv_8(J)
            DO 162 I=1,G_ni
               x_a = G_xg_8(I)-acos(-1.D0)
               call smat(s,RLON,RLAT,x_a,y_a)
               RLON = RLON+acos(-1.D0)
               SINT = SIN(RLAT)
               COST = COS(RLAT)
               UI_v(I,J) = Dcst_rayt_8*OMG*COST +               &
                           Dcst_rayt_8*K_Case*COST**(R_Case-1)* &
                   (R_Case*SINT*SINT-COST*COST)*COS(R_Case*(RLON-DLON))
               VI_v(I,J) = -Dcst_rayt_8*K_Case*R_Case*COST**(R_Case-1)*SINT &
                            * SIN(R_Case*(RLON-DLON))
               VICLL(I,J)=s(2,1)*UI_v(I,J) + s(2,2)*VI_v(I,J)
  162       CONTINUE
  163    CONTINUE

         endif

         call glbdist (VICLL,G_ni,G_nj,vloc,l_minx,l_maxx,l_miny,l_maxy, &
                               1,G_halox,G_haloy)
         do k=1,nk
         do j=1,lnjv
         do i=1,lniv
            v_temp(i,j,k)=vloc(i,j)
         enddo
         enddo
         enddo

!
!     ---------------------------------------------------------------
!
      return
      end
