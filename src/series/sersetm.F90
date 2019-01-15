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
subroutine sersetm(nom,rangee,valeur)
   !@object: initialize a time-series variable in multitasking mode
   implicit none
#include <arch_specific.hf>
   !@params
   ! nom      name of the variable to initialize
   ! valeur   table containing the values for initializing the variable
   ! rangee   row number
   character(len=*),intent(in) :: nom
   integer ,intent(in) :: valeur, rangee
   !@author b. bilodeau
   !*@/
   include "series.cdk"
   if (nstat <= 0) return
   if (nom == 'KA') then
      kam(rangee) = valeur
   endif
   return
end subroutine sersetm
