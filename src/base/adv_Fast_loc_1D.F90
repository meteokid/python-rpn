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

!**s/r adv_Fast_Loc_1D - Fast localization of grid p_8 with respect to grid x_8.  
!                        Based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008)
!
      subroutine adv_Fast_Loc_1D (p_8,x_8,np,nx,im,shift)

         implicit none

         integer np,nx,shift
 
         real*8 p_8(0:np),x_8(0:nx)
         integer im(0:np)

         !author Tanguay/Qaddouri
         !
         !revision
         ! v4_80 - Tanguay/Qaddouri - SLICE 

         !Local variables
         !---------------
         integer ip,jl,jm,ju
         integer nps,npf,nxs,nxf

         nps = 0
         npf = np 

         nxs = 0
         nxf = nx 

         do ip = nps,npf

            jl=nxs-1
            ju=nxf+1

            do while ((ju-jl) > 1)

               jm=(ju+jl)/2
               if((x_8(nxf) > x_8(nxs)) .eqv. (p_8(ip) > x_8(jm))) then
                   jl=jm
                 else
                   ju=jm
               endif
            end do

            im(ip) = max(nps,jl)

            if (.NOT.(x_8(im(ip)) <= p_8(ip) .and. p_8(ip) <= x_8(im(ip)+1))) then
               print *,'Fast_loc_1D KO ',ip,jl,x_8(im(ip)),p_8(ip),x_8(im(ip)+1)
               call flush(6)
               STOP
            endif

            im(ip) = im(ip) + shift 

         enddo

      return 
      end 
