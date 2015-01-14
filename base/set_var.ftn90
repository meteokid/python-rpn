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

!**s/p set_var - initialize list of variables to output
!

!
      integer function set_var (F_argc,F_argv_S,F_cmdtyp_S,F_v1,F_v2)
!
      implicit none
#include <arch_specific.hf>
!
      integer F_argc,F_v1,F_v2
      character *(*) F_argv_S(0:F_argc),F_cmdtyp_S
!
!author Vivian Lee - rpn - April 1999
!
!revision
! v2_00 - Lee V.            - initial MPI version
! v2_10 - Lee V.            - replaced CNMXPHY with Slab_pntop
! v2_21 - J. P. Toviessi    - set diez (#) slab output
! v2_31 - Lee V.            - add chemistry output list
! v2_32 - Lee V.            - gridset,levset,stepset are now IDs defined by the
! v2_32                       user so, they are matched to the SORTIE command
! v3_30 - Lee/Bilodeau      - bug fix to allow lower and upper case var names
! v4_50 - Lee V.            - Add Outd_varname_S for "longer" output names
! v4_70 - Lee V.            - Force 32 bit output for LA,LO
!
!object
!       initialization of the common blocks OUTD,OUTP. This function is
!       called when the keyword "sortie" is found in the first word
!       of the directives in the input file given in the statement
!       "process_f_callback". This feature is enabled by the
!       ARMNLIB "rpn_fortran_callback" routine (called in "srequet")
!       which allows a different way of passing user directives than
!       the conventional FORTRAN namelist. This function will process
!       the following example command read from the named input file.
!
! ie:   sortie([UU,VV,TT],levels,2,grid,3,steps,1)
!       sortie([PR,PC,RR],grid,3,steps,2,levels,1)
!
!       The "rpn_fortran_callback" routine will process the above
!       statement and return 5 arguments to this function. For more
!       information to how this is processed, see "SREQUET".
!
!	
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_argc       I    - number of elements in F_argv_S
! F_argv_S     I    - array of elements received
!                     if F_argv_S(ii) contains "[", the value in this
!                     argument indicates number of elements following it
! F_cmdtyp_S   I    - character command type - not used
! F_v1         I    - integer parameter 1 - not used
! F_v2         I    - integer parameter 2 - not used
!----------------------------------------------------------------
!
!Notes:
!    ie:   sortie([UU,VV,TT],levels,2,grid,3,steps,1)
!          sortie([PR,PC,RR],grid,3,steps,2,levels,1)
!
! sortie([vr1,vr2,vr3,...],levels,[levelset],grid,[gridset],steps,[stepset])
!
!  vr1,vr2,vr3... - set of variable names to output (max of 60)
!  levelset - levelset number to use for this set of variables
!  gridset  - gridset number to use for this set of variables
!  stepset  - stepset number (timestep set) to use for this set of variables
!
!  For each "sortie" command, the levelset, gridset and stepset must be
!  specified or an error will occur.
!

#include "glb_ld.cdk"
#include "lun.cdk"
#include "out3.cdk"
#include "setsor.cdk"
#include "outd.cdk"
#include "outp.cdk"
#include "outc.cdk"
#include "grid.cdk"
#include "level.cdk"
#include "timestep.cdk"
!
!*
!
      character*5 stuff_S
      character*8 varname_S
      character*4 string4
      character*16 string16
      integer levset,stepset,gridset,varmax
      integer i, j, k, m, pndx, ii, jj, kk
!
!----------------------------------------------------------------
!
      if (Lun_out.gt.0) then
          write(Lun_out,*)
          write(Lun_out,*) F_argv_S
      endif
      set_var=0

      if (index(F_argv_S(1),'[').gt.0) then
          stuff_S=F_argv_S(1)
          read(stuff_S(2:4),*) varmax
      else
        if (Lun_out.gt.0) write(Lun_out,*) &
                          'SET_VAR WARNING: syntax incorrect'
        set_var=1
        return
      endif
!
!     Check if chosen levels,grid and timestep sets are valid
!
      levset=-1
      gridset=-1
      stepset=-1
      do i=varmax+2, F_argc
         if (F_argv_S(i).eq.'levels') then
            read(F_argv_S(i+1),*) levset
         else if (F_argv_S(i).eq.'grid') then
            read(F_argv_S(i+1),*) gridset
         else if (F_argv_S(i).eq.'steps') then
            read(F_argv_S(i+1),*) stepset
         endif
      enddo

      if (gridset.lt.0) then
         if (Lun_out.gt.0) write(Lun_out,*) &
                           'SET_VAR WARNING: no Grid chosen'
         set_var=1
         return
      else
         do i=1,Grid_sets
            if (gridset .eq. Grid_id(i)) then
                gridset=i
                exit
            endif
         enddo
         if (i.gt.Grid_sets) then
             if (Lun_out.gt.0) write(Lun_out,*) &
                           'SET_VAR WARNING: invalid Grid set ID#'
             set_var=1
             return
         endif
      endif
      if (levset.lt.0) then
         if (Lun_out.gt.0) write(Lun_out,*) &
                           'SET_VAR WARNING: no Levels chosen'
         set_var=1
         return
      else
         do i=1,Level_sets
            if (levset .eq. Level_id(i)) then
                levset=i
                exit
            endif
         enddo
         if (i.gt. Level_sets) then
             if (Lun_out.gt.0) write(Lun_out,*) &
                           'SET_VAR WARNING: invalid Level set ID#'
             set_var=1
             return
         endif
      endif
      if (stepset.lt.0) then
          if (Lun_out.gt.0) write(Lun_out,*) &
                            'SET_VAR WARNING: no Timesteps chosen'
          set_var=1
          return
      else
         do i=1,Timestep_sets
            if (stepset .eq. Timestep_id(i)) then
                stepset=i
                exit
            endif
         enddo
         if (i .gt. Timestep_sets) then
             if (Lun_out.gt.0) write(Lun_out,*) &
                            'SET_VAR WARNING: invalid Timestep set ID#'
             set_var=1
             return
         endif
      endif
!
!     Store variables in variable sets
!
      if (F_argv_S(0).eq.'sortie') then
          j = Outd_sets + 1
          if (j.gt.MAXSET) then
          if (Lun_out.gt.0) write(Lun_out,*) &
                            'SET_VAR WARNING: too many OUTD sets'
          set_var=1
          return
          endif
!
          jj=0
          do ii=1,varmax
             jj = jj + 1
             call low2up  (F_argv_S(ii+1),string16)
             call low2up  (F_argv_S(ii+1),string4)
             Outd_varnm_S(jj,j)=string16
             Outd_var_S(jj,j)=string4
             if (Outd_var_S(jj,j)(1:3).eq.'PW_') Outd_var_S(jj,j)= 'PW'//Outd_varnm_S(jj,j)(4:5)
             Outd_nbit(jj,j)  = Out3_nbitg
             if (Outd_var_S(jj,j)(1:2).eq.'LA') Outd_nbit(jj,j)= 32
             if (Outd_var_S(jj,j)(1:2).eq.'LO') Outd_nbit(jj,j)= 32
          enddo
          if (jj.gt.0) then
              Outd_sets       = j
              Outd_var_max(j) = jj
              Outd_grid(j)    = gridset
              Outd_lev(j)     = levset
              Outd_step(j)    = stepset
          else
              if (Lun_out.gt.0) write(Lun_out,1400)
          endif
      else if (F_argv_S(0).eq.'sortie_p') then
          j = Outp_sets + 1
          if (j.gt.MAXSET) then
          if (Lun_out.gt.0) write(Lun_out,*) &
                            'SET_VAR WARNING: too many OUTP sets'
          set_var=1
          return
          endif
!                  
          jj=0
          do ii=1,varmax
             jj = jj + 1
             call low2up  (F_argv_S(ii+1),string16)
             Outp_varnm_S(jj,j)=string16
             Outp_nbit(jj,j)  = Out3_nbitg
             if (Outp_varnm_S(jj,j)(1:2).eq.'LA') Outp_nbit(jj,j)= 32
             if (Outp_varnm_S(jj,j)(1:2).eq.'LO') Outp_nbit(jj,j)= 32
          enddo
          if (jj.gt.0) then
              Outp_sets       = j
              Outp_var_max(j) = jj
              Outp_grid(j)    = gridset
              Outp_lev(j)     = levset
              Outp_step(j)    = stepset
              if (Lun_out.gt.0) then
                 write(Lun_out,*) '***PHY***Outp_sets=',Outp_sets
                 write(Lun_out,*) 'Outp_var_max=',Outp_var_max(j)
                 write(Lun_out,*) 'Outp_varnm_S=', &
                              (Outp_varnm_S(jj,j),jj=1,Outp_var_max(j))
                 write(Lun_out,*) 'Outp_grid=',Outp_grid(j)
                 write(Lun_out,*) 'Outp_lev=',Outp_lev(j)
                 write(Lun_out,*) 'Outp_step=',Outp_step(j)
              endif
          else
              if (Lun_out.gt.0) write(Lun_out,1400)
          endif
      else if (F_argv_S(0).eq.'sortie_c') then
          j = Outc_sets + 1
          if (j.gt.MAXSET) then
          if (Lun_out.gt.0) write(Lun_out,*) &
                            'SET_VAR WARNING: too many OUTC sets'
          set_var=1
          return
          endif
!                  
          jj=0
          do ii=1,varmax
             jj = jj + 1
             Outc_varnm_S(jj,j)= F_argv_S(ii+1)
             Outc_nbit(jj,j)   = Out3_nbitg
          enddo
          if (jj.gt.0) then
              Outc_sets       = j
              Outc_var_max(j) = jj
              Outc_grid(j)    = gridset
              Outc_lev(j)     = levset
              Outc_step(j)    = stepset
              if (Lun_out.gt.0) then
                 write(Lun_out,*) '***CHM***Outc_sets=',Outc_sets
                 write(Lun_out,*) 'Outc_var_max=',Outc_var_max(j)
                 write(Lun_out,*) 'Outc_varnm_S=', &
                              (Outc_varnm_S(jj,j),jj=1,Outc_var_max(j))
                 write(Lun_out,*) 'Outc_grid=',Outc_grid(j)
                 write(Lun_out,*) 'Outc_lev=',Outc_lev(j)
                 write(Lun_out,*) 'Outc_step=',Outc_step(j)
              endif
          else
              if (Lun_out.gt.0) write(Lun_out,1400)
          endif
      endif
!
!----------------------------------------------------------------
!
 1400    format('SET_VAR - WARNING: NO VARIABLES DEFINED FOR THIS SET')
      return
      end
