!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
!**s/r ens_ptp.ftn - perturb the physical tendencies with Markov chains values

      subroutine ens_ptp1(d,v,f,dsiz,fsiz,vsiz,ni,nk,kount)
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>

      integer dsiz,fsiz,vsiz,ni,nk,kount
      real, target :: d(dsiz), f(fsiz), v(vsiz)

      real fac_ptp_m,fac_ptp_t,fac_convec

      integer i,k
      integer, dimension(ni) :: counter_w
      real   , dimension(ni) :: fac_mrk2

!author Normand Gagnon March 2010
!
!revision 
! 001      see version 5.5.0 for previous history
!object
!  To perturb the physical tendencies with Markov chains values
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! d             I                  dynamics input field
! v            I/O                 physics tendencies 
! f             I                  historic variables for the physics
! dsiz          I                  dimension of d
! fsiz          I                  dimension of f
! vsiz          I                  dimension of v
! ni            I                  horizontal running length
! nk            I                  vertical dimension
!----------------------------------------------------------------
!
! VERTICAL ENVELOPPE:
!
! Schematics of the fac_ptp values with the vertical envelope
!
!    f(mrk2+i-1)
!         |
!         |
!         |
!         | - Ens_ptp_env_u
!         \
!          \  In that layer it is a linear mix between the two values (mrk2 and 1.0)
!           \
!            \ - Ens_ptp_env_b
!             |
!             |
!             |
!             _
!            1.0
!

include "thermoconsts.inc"
#include "ens.cdk"

      real, pointer, dimension(:)   :: zcapekfc, zmrk2, ztlc
      real, pointer, dimension(:,:) :: zsigm, zsigt, &
                                       ztplus, zuplus, zvplus, zwplus, &
                                       ztphytd, zuphytd, zvphytd

      if (.not.stochphy) Return

      nullify(zcapekfc); if (capekfc > 0) zcapekfc(1:ni) => f( capekfc:)
      zmrk2  (1:ni)      => f( mrk2:)
      ztlc   (1:ni)      => f( tlc:)
      zsigm  (1:ni,1:nk) => d( sigm:)
      zsigt  (1:ni,1:nk) => d( sigt:)
      ztplus (1:ni,1:nk) => d( tplus:)
      zuplus (1:ni,1:nk) => d( uplus:)
      zvplus (1:ni,1:nk) => d( vplus:)
      zwplus (1:ni,1:nk) => d( wplus:)
      ztphytd(1:ni,1:nk) => v( tphytd:)
      zuphytd(1:ni,1:nk) => v( uphytd:)
      zvphytd(1:ni,1:nk) => v( vphytd:)
!
      if (kount.lt.1) then
         do i=1,ni
           zmrk2(i)=0.0
         enddo
         return
      endif
            
      do i=1,ni
         
         fac_convec = ptpfacreduc
         if (associated(zcapekfc) .and.convec.eq.'KFC') then
            if (zcapekfc(i).le.ptpcape) fac_convec = 1.0
         elseif (ztlc(i).le.ptptlc.and.convec.eq.'OLDKUO') then
            fac_convec = 1.0
         endif
         counter_w(i) = 0
	 fac_mrk2(i)  =(zmrk2(i)-1.)*fac_convec+1.
      enddo
!
      do k=1,nk
         do i=1,ni
            if (counter_w(i).eq.0) then
               if (zwplus(i,k).gt.ptpcritw) then
                  counter_w(i)=counter_w(i)+1   
!                  print *,'W tres grand =',zwplus(i,k),i,k,kount           
               else

                  if (zsigm(i,k).lt.ptpenvb.and.zsigm(i,k).gt.ptpenvu) then
                     fac_ptp_m=( zsigm(i,k)-ptpenvu+fac_mrk2(i)* &
                     (ptpenvb-zsigm(i,k)) )/(ptpenvb-ptpenvu)
                  else if (zsigm(i,k).le.ptpenvu) then
                     fac_ptp_m=fac_mrk2(i)
                  else
                     fac_ptp_m=1.0
                  endif
                  
                  if (zsigt(i,k).lt.ptpenvb.and.zsigt(i,k).gt.ptpenvu) then
                     fac_ptp_t=( zsigt(i,k)-ptpenvu+fac_mrk2(i)* &
                     (ptpenvb-zsigt(i,k)))/(ptpenvb-ptpenvu)
                  else if (zsigt(i,k).le.ptpenvu) then
                     fac_ptp_t=fac_mrk2(i)
                  else
                     fac_ptp_t=1.0
                  endif
                  
                  !           Tendencies of the wind
                  zuphytd(i,k) = zuphytd(i,k)*fac_ptp_m
                  zvphytd(i,k) = zvphytd(i,k)*fac_ptp_m
                  
                  !           Tendencies of temperature
                  ztphytd(i,k) = ztphytd(i,k)*fac_ptp_t
               endif
            endif
            
            zuplus(i,k)  = zuplus(i,k)+delt*zuphytd(i,k)
            zvplus(i,k)  = zvplus(i,k)+delt*zvphytd(i,k)
            ztplus(i,k)  = ztplus(i,k)+delt*ztphytd(i,k)                  
            
         enddo

      enddo
!
!--------------------------------------------------------------------
!
 999  return
      end
!
