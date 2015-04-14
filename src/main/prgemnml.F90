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

!**s/r prgemnml
      subroutine prgemnml()
      implicit none
#include <arch_specific.hf>
!
!author V.Lee - October 20,2008
!
!object
!
!----------------------------------------------------------------------

      print *,'creating gemdict.nml'
      open (6,file="gemdict.nml",access='SEQUENTIAL',form='FORMATTED')
      call nml_grid()
      call nml_step()
      call nml_adx()
      call nml_gem()
      call nml_gement()
      call nml_theo()
      close(6)
     
      return
!      
!-------------------------------------------------------------------
      end
!
      subroutine nml_grid()
      implicit none
#include <arch_specific.hf>
#include "grd.cdk"
      write (6  ,nml=grid)
      return
      end

      subroutine nml_step()
      implicit none
#include <arch_specific.hf>
#include "step.cdk"
      write (6  ,nml=step)
      return
      end
!
      subroutine nml_adx()
      implicit none
#include <arch_specific.hf>
      call adx_nml_print()
      return
      end
!
      subroutine nml_gem()
      implicit none
#include <arch_specific.hf>
#include "nml.cdk"
      write (6  ,nml=gem_cfgs)
      write (6  ,nml=grdc)
      write (6  ,nml=williamson)
      return
      end
!
      subroutine nml_gement()
      implicit none
#include <arch_specific.hf>
#include "e_nml.cdk"
      write (6  ,nml=gement)
      return
      end
!
      subroutine nml_theo()
      implicit none
#include <arch_specific.hf>
#undef TYPE_CDK
#undef GMM_IS_OK
#include "theonml.cdk"
      write (6  ,nml=theo_cfgs)
      write (6  ,nml=mtn_cfgs)
      return
      end
