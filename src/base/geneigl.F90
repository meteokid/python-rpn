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

!**s/r geneigl - solves a generalised symmetric eigenproblem

      subroutine geneigl3 ( F_eval_8, F_evec_8, F_b_8, NN, NMAX, NWORK )
      implicit none
#include <arch_specific.hf>
!
      integer NN, NMAX, NWORK
      real*8 F_eval_8(NMAX), F_evec_8(NMAX,NN), F_b_8(NMAX,NN)
!
!author 
!     j. cote  March 1993, from geneig
!
!revision
! v2_00 - Desgagne M.       - initial MPI version (from geneigl v1_03)
! v3_10 - Lee V.            - SYGV calling sequence for AIX architecture
!
!object
!    To solve a generalised symmetric eigenproblem
!
!            a * x = lambda * b * x
!
!arguments:
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_eval_8    O     - eigenvalues (lambda)
!  F_evec_8    I/O   - input: matrix A 
!                     output: eigenvectors (x)
!  F_b_8       I/O   - input: matrix B 
!                     output:
!  NN          I     - order of problem
!  NMAX        I     - leading dimension of matrices in calling programm
!  NWORK       I     - size of F_work_8 >= max(1,3*n-1)
!
!note: LAPACK public domain library is required. See reference manual
!      for more information [LAPACK Users' Guide (SIAM),1992]
!      Only upper triangular parts of A and B need to be specified
!      A and  B are overwritten
!*
#include "lun.cdk"
#include "path.cdk"
#include "ptopo.cdk"

      character*8    tridi_signature_8, signa_1, signa_2, &
                     signa_1_read, signa_2_read
      character*11   basename,basename_read
      character*1024 rootdir,fn
      integer i, j, k, info, unf, err, errop, len1,len2,len3
      real*8 sav_8, faz_8, one_8, wk1(NWORK)
      data one_8 /1.0d0/
!
!--------------------------------------------------------------------
!
      rootdir = trim(Path_input_S)//'/CACHE'
      basename= '/eigenv_v1_'
!      call gemtim4 ( Lun_out, 'GENEIGL: avant DSYGV', .false. )

      signa_1= tridi_signature_8(F_evec_8,NMAX,NN)
      signa_2= tridi_signature_8(F_b_8   ,NMAX,NN)

      fn = trim(rootdir)//trim(basename)//signa_1//'_'//signa_2//'.bin'
      unf= 741

      if (Ptopo_myproc.eq.0) then
         open (unf,file=fn,status='OLD',form='unformatted',iostat=errop)

 55      if ( errop.eq.0 ) then
            write(6,1001) 'READING', trim(fn)
            read (unf) basename_read,signa_1_read,signa_2_read
            read (unf) F_evec_8,F_eval_8
            close(unf)
            if (.not.((basename_read .eq. basename) .and. &
                      (signa_1_read  .eq. signa_1 ) .and. &
                      (signa_2_read  .eq. signa_2 ) )) then
               write(6,1001) 'WRONG INPUT', trim(fn)
               errop=-1
               goto 55
            endif
         else
            info = -1
            call DSYGV( 1, 'V', 'U', NN, F_evec_8, NMAX, F_b_8, NMAX, &
                                          F_eval_8, wk1, NWORK, info )
            do j=1,NN
               faz_8 = sign( one_8, F_evec_8(1,j) )
               do i= 1, NN
                  F_evec_8(i,j) = faz_8 * F_evec_8(i,j)
               enddo
            enddo           
            do j= 1, NN/2
               k = NN - j + 1
               sav_8 = F_eval_8(j)
               F_eval_8(j) = F_eval_8(k)
               F_eval_8(k) = sav_8
               do i= 1, NN
                  sav_8 = F_evec_8(i,j)
                  F_evec_8(i,j) = F_evec_8(i,k)
                  F_evec_8(i,k) = sav_8
               enddo
            enddo
            if (Ptopo_couleur.eq.0) then
               open (unf, file=fn, form='unformatted',iostat=errop)
               if ( errop.eq.0 ) then
                  write(6,1001) 'WRITING', trim(fn)
                  write(unf) basename,signa_1,signa_2
                  write(unf) F_evec_8,F_eval_8
                  close(unf)
               endif
            endif
         endif
      endif

      call RPN_COMM_bcast (F_evec_8,NMAX*NN,"MPI_DOUBLE_PRECISION",0,"grid",err)
      call RPN_COMM_bcast (F_eval_8,NMAX   ,"MPI_DOUBLE_PRECISION",0,"grid",err)

!      call gemtim4 ( Lun_out, 'GENEIGL: apres DSYGV', .false. )

 1001 format (/' GENEIGL: ',a,' FILE ',a)
!
!--------------------------------------------------------------------
!
      return
      end

character(len=8) function tridi_signature_8(A,NMAX,NN) ! CRC32 signature of a real*8 tridiagonal matrix
implicit none
integer, intent(IN) :: NMAX    ! storage dimension
integer, intent(IN) :: NN      ! useful dimension ( <= NMAX )
real*8, dimension(NMAX,NN), intent(IN) :: A

integer :: f_crc32
external :: f_crc32
real*8, dimension(3,NN) :: sig
integer :: i
character (len=8) :: result

do I = 1 , NN  ! get diagonals
  sig(1,I) = A(max(1,i-1),i)     ! upper diagonal (duplicating first point)
  sig(2,i) = A(I,I)              ! main diagonal
  sig(3,i) = A(min(NN,i+1),i)    ! lower diagonal (duplicating last point)
enddo
i = f_crc32(0,sig,NN*3*8)        ! get 32 bit Byte CRC
write(result,100)i               ! transform into 8 character HEX string
100 format(Z8.8)

tridi_signature_8 = result

return
end function tridi_signature_8

      subroutine geneigl
      print*, 'geneigl replaced by geneigl3 - ABORT'
      stop
      end

