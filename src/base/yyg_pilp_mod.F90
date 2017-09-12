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

module yyg_pilp
   implicit none
   public
   save
!______________________________________________________________________
!                                                                      |
!  VARIABLES ASSOCIATED WITH fat BCS communication for Yin-Yang        |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! M                  | Total number of Processors in Yang or Yin       |
! Pilp_recvproc(M)   | Processor number to receive data FROM           |
! Pilp_recv_len(M)   | Number of values to receive from Pilp_recvproc(i)|
! Pilp_recv_adr(M)   | Address of starting index in vector for receiv- |
!                    | ing data from processor Pilp_recvproc(i)         |
! Pilp_recv_i(*,M)   | local gridpoint I to receive value from PE(i)   |
! Pilp_recv_j(*,M)   | local gridpoint J to receive value from PE(i)   |
! Pilp_sendproc(M)   | Processor number to send  data TO               |
! Pilp_send_len(M)   | Number of values to send to Pilp_sendproc(i)     |
! Pilp_send_adr(M)   | Address of starting index in vector for send-   |
!                    | ing data to   processor Pilp_sendproc(i)         |
! Pilp_send_imx(*,M) | closest I gridpoint on the other panel to find  |
!                    | the value for Pilp_send_xxr,Pilp_send_yyr         |
! Pilp_send_imy(*,M) | closest J gridpoint on the other panel to find  |
!                    | the value for  Pilp_send_xxr,Pilp_send_yyr        |
! Pilp_send_xxr(*,M) | longitude in the other panel to find the value  |
!                    | for receiving panel                             |
! Pilp_send_yyr(*,M) | latitude in the other panel to find the value   |
!                    | for receiving panel                             |
! Pilp_send_s1(*,M)  | element (s(1,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pilp_send_s2(*,M)  | element (s(1,2)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pilp_send_s3(*,M)  | element (s(2,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pilp_send_s4(*,M)  | element (s(2,2)) in matrix for polar vectors    |
!                    | transformation                                  |
!______________________________________________________________________|
!
!Declarations for Scalar variables (on Phi grid for cubic interpolation)
      integer ::  Pilp_send_all, Pilp_recv_all,Pilp_sendmaxproc,Pilp_recvmaxproc
      integer, dimension (:  ), pointer :: &
               Pilp_sendproc, Pilp_recvproc, &
               Pilp_recv_len, Pilp_send_len, &
               Pilp_recv_adr, Pilp_send_adr, &
               Pilp_recv_i   , Pilp_recv_j   , Pilp_send_imx , Pilp_send_imy

      real*8,  dimension (: ), pointer :: &
               Pilp_send_xxr, Pilp_send_yyr, &
               Pilp_send_s1,  Pilp_send_s2, Pilp_send_s3, Pilp_send_s4

end module yyg_pilp
