!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     This module contains routines for writing the remapping data to 
!     a file.  Before writing the data for each mapping, the links are 
!     sorted by destination grid address.
!
!-----------------------------------------------------------------------
!
!     CVS:$Id: remap_write.f,v 1.7 2001/08/21 21:06:42 pwjones Exp $
!
!     Copyright (c) 1997, 1998 the Regents of the University of 
!       California.
!
!     This software and ancillary information (herein called software) 
!     called SCRIP is made available under the terms described here.  
!     The software has been approved for release with associated 
!     LA-CC Number 98-45.
!
!     Unless otherwise indicated, this software has been authored
!     by an employee or employees of the University of California,
!     operator of the Los Alamos National Laboratory under Contract
!     No. W-7405-ENG-36 with the U.S. Department of Energy.  The U.S.
!     Government has rights to use, reproduce, and distribute this
!     software.  The public may copy and use this software without
!     charge, provided that this Notice and any statement of authorship
!     are reproduced on all copies.  Neither the Government nor the
!     University makes any warranty, express or implied, or assumes
!     any liability or responsibility for the use of this software.
!
!     If software is modified to produce derivative works, such modified
!     software should be clearly marked, so as not to confuse it with 
!     the version available from Los Alamos National Laboratory.
!
!***********************************************************************

      module remap_write

!-----------------------------------------------------------------------

      use kinds_mod     ! defines common data types
      use constants     ! defines common scalar constants
      use grids         ! module containing grid information
      use remap_vars    ! module containing remap information

      implicit none

!-----------------------------------------------------------------------
!
!     module variables
!
!-----------------------------------------------------------------------

      character(char_len), private :: 
     &   map_method       ! character string for map_type
     &,  normalize_opt    ! character string for normalization option
     &,  history          ! character string for history information
     &,  convention       ! character string for output convention

      character(8), private :: 
     &   cdate            ! character date string

      integer (kind=int_kind), dimension(:), allocatable, private ::
     &   src_mask_int     ! integer masks to determine
     &,  dst_mask_int     ! cells that participate in map


!***********************************************************************

      contains

!***********************************************************************

      subroutine sort_add(add1, add2, weights)

!-----------------------------------------------------------------------
!
!     this routine sorts address and weight arrays based on the
!     destination address with the source address as a secondary
!     sorting criterion.  the method is a standard heap sort.
!
!-----------------------------------------------------------------------

      use kinds_mod     ! defines common data types
      use constants     ! defines common scalar constants

      implicit none

!-----------------------------------------------------------------------
!
!     Input and Output arrays
!
!-----------------------------------------------------------------------

      integer (kind=int_kind), intent(inout), dimension(:) ::
     &        add1,       ! destination address array (num_links)
     &        add2        ! source      address array

      real (kind=dbl_kind), intent(inout), dimension(:,:) ::
     &        weights     ! remapping weights (num_wts, num_links)

!-----------------------------------------------------------------------
!
!     local variables
!
!-----------------------------------------------------------------------

      integer (kind=int_kind) ::
     &          num_links,          ! num of links for this mapping
     &          num_wts,            ! num of weights for this mapping
     &          add1_tmp, add2_tmp, ! temp for addresses during swap
     &          nwgt,
     &          lvl, final_lvl,     ! level indexes for heap sort levels
     &          chk_lvl1, chk_lvl2, max_lvl

      real (kind=dbl_kind), dimension(SIZE(weights,DIM=1)) ::
     &          wgttmp              ! temp for holding wts during swap

!-----------------------------------------------------------------------
!
!     determine total number of links to sort and number of weights
!
!-----------------------------------------------------------------------

      num_links = SIZE(add1)
      num_wts   = SIZE(weights, DIM=1)

!-----------------------------------------------------------------------
!
!     start at the lowest level (N/2) of the tree and sift lower 
!     values to the bottom of the tree, promoting the larger numbers
!
!-----------------------------------------------------------------------

      do lvl=num_links/2,1,-1

        final_lvl = lvl
        add1_tmp = add1(lvl)
        add2_tmp = add2(lvl)
        wgttmp(:) = weights(:,lvl)

        !***
        !*** loop until proper level is found for this link, or reach
        !*** bottom
        !***

        sift_loop1: do

          !***
          !*** find the largest of the two daughters
          !***

          chk_lvl1 = 2*final_lvl
          chk_lvl2 = 2*final_lvl+1
          if (chk_lvl1 .EQ. num_links) chk_lvl2 = chk_lvl1

          if ((add1(chk_lvl1) >  add1(chk_lvl2)) .OR.
     &       ((add1(chk_lvl1) == add1(chk_lvl2)) .AND.
     &        (add2(chk_lvl1) >  add2(chk_lvl2)))) then
            max_lvl = chk_lvl1
          else 
            max_lvl = chk_lvl2
          endif

          !***
          !*** if the parent is greater than both daughters,
          !*** the correct level has been found
          !***

          if ((add1_tmp .GT. add1(max_lvl)) .OR.
     &       ((add1_tmp .EQ. add1(max_lvl)) .AND.
     &        (add2_tmp .GT. add2(max_lvl)))) then
            add1(final_lvl) = add1_tmp
            add2(final_lvl) = add2_tmp
            weights(:,final_lvl) = wgttmp(:)
            exit sift_loop1

          !***
          !*** otherwise, promote the largest daughter and push
          !*** down one level in the tree.  if haven't reached
          !*** the end of the tree, repeat the process.  otherwise
          !*** store last values and exit the loop
          !***

          else 
            add1(final_lvl) = add1(max_lvl)
            add2(final_lvl) = add2(max_lvl)
            weights(:,final_lvl) = weights(:,max_lvl)

            final_lvl = max_lvl
            if (2*final_lvl > num_links) then
              add1(final_lvl) = add1_tmp
              add2(final_lvl) = add2_tmp
              weights(:,final_lvl) = wgttmp(:)
              exit sift_loop1
            endif
          endif
        end do sift_loop1
      end do

!-----------------------------------------------------------------------
!
!     now that the heap has been sorted, strip off the top (largest)
!     value and promote the values below
!
!-----------------------------------------------------------------------

      do lvl=num_links,3,-1

        !***
        !*** move the top value and insert it into the correct place
        !***

        add1_tmp = add1(lvl)
        add1(lvl) = add1(1)

        add2_tmp = add2(lvl)
        add2(lvl) = add2(1)

        wgttmp(:) = weights(:,lvl)
        weights(:,lvl) = weights(:,1)

        !***
        !*** as above this loop sifts the tmp values down until proper 
        !*** level is reached
        !***

        final_lvl = 1

        sift_loop2: do

          !***
          !*** find the largest of the two daughters
          !***

          chk_lvl1 = 2*final_lvl
          chk_lvl2 = 2*final_lvl+1
          if (chk_lvl2 >= lvl) chk_lvl2 = chk_lvl1

          if ((add1(chk_lvl1) >  add1(chk_lvl2)) .OR.
     &       ((add1(chk_lvl1) == add1(chk_lvl2)) .AND.
     &        (add2(chk_lvl1) >  add2(chk_lvl2)))) then
            max_lvl = chk_lvl1
          else 
            max_lvl = chk_lvl2
          endif

          !***
          !*** if the parent is greater than both daughters,
          !*** the correct level has been found
          !***

          if ((add1_tmp >  add1(max_lvl)) .OR.
     &       ((add1_tmp == add1(max_lvl)) .AND.
     &        (add2_tmp >  add2(max_lvl)))) then
            add1(final_lvl) = add1_tmp
            add2(final_lvl) = add2_tmp
            weights(:,final_lvl) = wgttmp(:)
            exit sift_loop2

          !***
          !*** otherwise, promote the largest daughter and push
          !*** down one level in the tree.  if haven't reached
          !*** the end of the tree, repeat the process.  otherwise
          !*** store last values and exit the loop
          !***

          else 
            add1(final_lvl) = add1(max_lvl)
            add2(final_lvl) = add2(max_lvl)
            weights(:,final_lvl) = weights(:,max_lvl)

            final_lvl = max_lvl
            if (2*final_lvl >= lvl) then
              add1(final_lvl) = add1_tmp
              add2(final_lvl) = add2_tmp
              weights(:,final_lvl) = wgttmp(:)
              exit sift_loop2
            endif
          endif
        end do sift_loop2
      end do

      !***
      !*** swap the last two entries
      !***


      add1_tmp = add1(2)
      add1(2)  = add1(1)
      add1(1)  = add1_tmp

      add2_tmp = add2(2)
      add2(2)  = add2(1)
      add2(1)  = add2_tmp

      wgttmp (:)   = weights(:,2)
      weights(:,2) = weights(:,1)
      weights(:,1) = wgttmp (:)

!-----------------------------------------------------------------------

      end subroutine sort_add

!***********************************************************************

      end module remap_write

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
