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
      integer function cascindx ( ideb,ifin,jdeb,jfin,&
                                  xfi,yfi,nid,njd,xpx,ypx,nis,njs )
      implicit none
#include <arch_specific.hf>

      integer ideb,ifin,jdeb,jfin,nid,njd,nis,njs
      real xfi(nid), yfi(njd), xpx(nis), ypx(njs)
!
!author
!        Michel Desgagne  -  winter 2013
!revision
! v4_50 - Desgagne M.       - initial version

      integer i
!
!---------------------------------------------------------------------
!
      cascindx = -1

      do i=1,nis
         if (xpx(i).le.xfi(1  )) ideb=i
         if (xpx(i).le.xfi(nid)) ifin=i
      enddo

      do i=1,njs
         if (ypx(i).le.yfi(1  )) jdeb=i
         if (ypx(i).le.yfi(njd)) jfin=i
      enddo

      ideb = ideb - 2 ; ifin = ifin + 3
      jdeb = jdeb - 2 ; jfin = jfin + 3 

      if ( (ideb.lt.1  ).or.(jdeb.lt.1    ) .or. &
           (ifin.gt.nis).or.(jfin.gt.njs) ) then
         ideb=0 ; ifin=0 ; jdeb=0 ; jfin=0
         return
      endif
      
      cascindx = 0
!
!---------------------------------------------------------------------
!
      return
      end

