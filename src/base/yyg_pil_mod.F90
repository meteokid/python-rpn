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

module yyg_pil
   implicit none
   public
   save
!______________________________________________________________________
!                                                                      |
!  VARIABLES ASSOCIATED WITH BCS communication for Yin-Yang            |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! M                  | Total number of Processors in Yang or Yin       |
! Pil_recvproc(M)    | Processor number to receive data FROM           |
! Pil_recv_len(M)    | Number of values to receive from Pil_recvproc(i)|
! Pil_recv_adr(M)    | Address of starting index in vector for receiv- |
!                    | ing data from processor Pil_recvproc(i)         |
! Pil_recv_i(*,M)    | local gridpoint I to receive value from PE(i)   |
! Pil_recv_j(*,M)    | local gridpoint J to receive value from PE(i)   |
! Pil_sendproc(M)    | Processor number to send  data TO               |
! Pil_send_len(M)    | Number of values to send to Pil_sendproc(i)     |
! Pil_send_adr(M)    | Address of starting index in vector for send-   |
!                    | ing data to   processor Pil_sendproc(i)         |
! Pil_send_imx(*,M)  | closest I gridpoint on the other panel to find  |
!                    | the value for Pil_send_xxr,Pil_send_yyr         |
! Pil_send_imy(*,M)  | closest J gridpoint on the other panel to find  |
!                    | the value for  Pil_send_xxr,Pil_send_yyr        |
! Pil_send_xxr(*,M)  | longitude in the other panel to find the value  |
!                    | for receiving panel                             |
! Pil_send_yyr(*,M)  | latitude in the other panel to find the value   |
!                    | for receiving panel                             |
! Pil_send_s1(*,M)   | element (s(1,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pil_send_s2(*,M)   | element (s(1,2)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pil_send_s3(*,M)   | element (s(2,1)) in matrix for polar vectors    |
!                    | transformation                                  |
! Pil_send_s4(*,M)   | element (s(2,2)) in matrix for polar vectors    |
!                    | transformation                                  |
!______________________________________________________________________|
!
!Declarations for Scalar variables (on Phi grid for cubic interpolation)
   integer :: Pil_send_all, Pil_recv_all,Pil_sendmaxproc,Pil_recvmaxproc
   integer, dimension (:  ), pointer :: &
            Pil_sendproc, Pil_recvproc, &
            Pil_recv_len, Pil_send_len, &
            Pil_recv_adr, Pil_send_adr, &
            Pil_recv_i   , Pil_recv_j   , Pil_send_imx , Pil_send_imy

   real*8,  dimension (: ), pointer :: &
            Pil_send_xxr, Pil_send_yyr, &
            Pil_send_s1,  Pil_send_s2, Pil_send_s3, Pil_send_s4

end module yyg_pil
