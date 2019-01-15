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

!**s/r var_dict

      subroutine var_dict
      implicit none
#include <arch_specific.hf>
!
!author
!     Michel Desgagne  -- fall 2013
!
!revision
! v4_70 - Desgagne M.   -   Initial version

#include "vardict.cdk"
!
!-------------------------------------------------------------------
!
      allocate (vardict(500))

      vardict%out_name = '@#$%'
      vardict%gmm_name = '@#$%'
      vardict%hor_stag = ''
      vardict%ver_stag = ''
      var_cnt          = 0
      vardict%fact_mult= 1.
      vardict%fact_add = 0.

! 9.80616
! KNAMS      0.514791

      call var_gestdic ('GMMNAME=PW_UU:P ; OUTNAME=UU ;HSTAG=U ; VSTAG=M ; fact_Mul=1.942536 ; fact_Add= 0.    ')
      call var_gestdic ('GMMNAME=PW_VV:P ; OUTNAME=VV ;HSTAG=V ; VSTAG=M ; fact_Mul=1.942536 ; fact_Add= 0.    ')
      call var_gestdic ('GMMNAME=PW_TT:P ; OUTNAME=TT ;HSTAG=Q ; VSTAG=T ; fact_Mul=1.       ; fact_Add=-273.15')
      call var_gestdic ('GMMNAME=PW_GZ:P ; OUTNAME=GZ ;HSTAG=Q ; VSTAG=M ; fact_Mul=1.       ; fact_Add= 0.    ')

!
!     ---------------------------------------------------------------
!
      return
      end
