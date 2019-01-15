!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

subroutine gesdict(n,nk,lindex,lachaine)
   implicit none
#include <arch_specific.hf>

   character*(*) lachaine
   integer n,nk,lindex

!@Author M. Desgagne (Oct 1995)
!@Revision
! 001      B. Bilodeau (Jan 1996) - Check name conflicts for
!                                   a given description
! 002      B. Bilodeau (Sep 1996) - Add 2-letter names
! 003      B. Bilodeau (Aug 1998) - Add staggered levels
! 004      B. Bilodeau (Dec 1998) - Add entry bus
! 005      B. Bilodeau (Feb 1999) - Add fmul to entpar, dynpar,
!                                   perpar and volpar
! 006      G. Bergeron (Oct 1999) - Test if top < maxbus
! 007      B. Bilodeau (Mar 2000) - Test conflicting output names
!                                   for a given variable
! 008      B. Bilodeau (Feb 2004) - 4-letter output names and
!                                  16-letter names
! 009      B. Bilodeau (Mar 2005) - Test conflicting variable names
!                                   and output names
! 010      B. Bilodeau (Jun 2005) - Forbid SLB*n and SLS*n for n > 1
!                                   Add mosaic capability for CLASS
! 011      V. Lee      (Mar 2011) - nmosaic=real number of mosaic tiles
!                                 - entpar(*,8),perpar(*,8),..=0 if no mosaic
!@Object
!    Manages the dictionary describing the 4 main buses of the unified
!    CMC-RPN physics package interface (BUSENT, BUSDYN, BUSPER and BUSVOL).
!    Each variable has a formal name <bus>nm(*) and a formal
!    description <bus>dc(*) along with 4 attributes <bus>par(*,4).
!    The first and second attributes are respectively the starting
!    index on <bus> and the length of the variable. The third
!    attribute is the multiplicity factor. The fourth attribute is
!    the a flag to identify variables that are defined on staggered levels.
!    The recognized token in "lachaine" are:
!         VN=  ;       ===> formal name
!         ON=  ;       ===> output name (4 letters only)
!         IN=  ;       ===> input  name (4 letters only)
!         SN=  ;       ===> series name (4 letters only)
!         VD=  ;       ===> formal description
!         VS=  ;       ===> variable shape (accepted shapes are SLB and
!                           ROW with +, - or * followed by an integer)
!         VB=  ;       ===> bus identification (D, P and V)
!         MIN= ;       ===> minimum value of the field
!         MAX= ;       ===> maximum value of the field
!@Arguments
!            - Input -
! n          horizontal dimension
! nk         vertical dimension
!            - Output -
! lindex     starting index on the bus
!            - Input -
! lachaine   string identifying the variable attributes
   
#include <rmnlib_basics.hf>
#include <msg.h>
   include "buses.cdk"

   integer, external :: splitst3

   character*1   bus
   character*4   outname,inname,sername
   character*3   shape
   character*7   struc
   character*16  varname, samename, othername
   character*48  vdescrp
   character*60  vardesc
   character*256 string
   character*512 longname
   integer :: nmosaic, fmosaik, fmul, dynini, stagg
   integer :: i, ind, esp, ivmin, ivmax, istat
   real :: vmin, vmin0, vmax

!-------------------------------------------------------------------

      call low2up(lachaine,string)
      istat = splitst3(longname, outname, inname, sername, &
           vdescrp, struc, shape, nmosaic, fmul, bus, dynini,&
           stagg, vmin, vmax, string)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_ERROR,'(gesdict) Invalid gesdict string: '//trim(string))
         call qqexit(1)
      endif

      ind= index(longname,",")
      if (ind .eq. 0) then
         varname = longname
         longname= ''
      else
         varname = longname(1:ind-1)
         longname= longname(ind+1: )
      endif
      ind= index(varname,":")
      
!     Mosaic field have an extra row for the averaged layer (1st row)
      fmosaik=nmosaic+1
      vardesc = vdescrp//';VS='//struc
      lindex  = 0

      if ((shape.eq.'SLB'.or.shape.eq.'SLS').and.fmul.gt.1) then
         write(6,908) varname,struc,shape
         call qqexit(1)
      endif

      if (inname /= ' ') then
         if ((bus == 'E' .and. any(entnm(1:enttop,BUSNM_IN) == inname)) .or. &
              (bus == 'D' .and. any(dynnm(1:dyntop,BUSNM_IN) == inname)) .or. &
              (bus == 'P' .and. any(pernm(1:pertop,BUSNM_IN) == inname)) .or. &
              (bus == 'V' .and. any(volnm(1:voltop,BUSNM_IN) == inname))) then
            write(6,'(1x,a)') "==> STOP IN GESDICT: CONFLICT FOR '"//inname// &
                 "' INPUT NAME. ALREADY ACCEPTED FOR OTHER VARIABLE in bus: "//trim(bus)
            call qqexit(1)
         endif
      endif
      if (varname /= ' ') then
         if (any(entnm(1:enttop,BUSNM_VN) == varname) .or. &
              any(dynnm(1:dyntop,BUSNM_VN) == varname) .or. &
              any(pernm(1:pertop,BUSNM_VN) == varname) .or. &
              any(volnm(1:voltop,BUSNM_VN) == varname)) then
            print *,string
            write(6,'(1x,a)') "==> STOP IN GESDICT: CONFLICT FOR '"//varname// &
                 "' NAME. ALREADY ACCEPTED FOR OTHER VARIABLE"
            call qqexit(1)
         endif
      endif

      ivmin = transfer(vmin,ivmin)
      ivmax = transfer(vmax,ivmax)

      if (bus.eq."E") then
         do 10 i=1,enttop

!           verifier si la meme description existe deja
            if (vardesc.eq.entdc(i)) then
               if (varname.ne.entnm(i,1)) then
                  write (6,903) varname,entnm(i,1),entdc(i)
                  call qqexit(1)
               endif
            endif

            if (varname.eq.entnm(i,1)) then
               if (vardesc.ne.entdc(i)) then
                  write (6,901) varname,vardesc,entdc(i)
                  call qqexit(1)
               endif
               esp = n*nk
               if (shape.eq."ROW") esp = n
               if (entpar(i,2).ne.(esp*fmul*fmosaik)) then
                  write (6,902) varname,entpar(i,2),(esp*fmul*fmosaik)
                  call qqexit(1)
               endif
               lindex = entpar(i,1)
               goto 601
            endif
 10      continue
         if (buslck) goto 601
            enttop = enttop + 1
            esp = n*nk
            entpar(enttop,7) = nk
            if (shape.eq."ROW") then
               esp = n
               entpar(enttop,7) = 1
            endif
            entpar(enttop,5) = esp
            esp = esp*fmul*fmosaik
            entnm(enttop,BUSNM_VN) = varname
            entnm(enttop,BUSNM_ON) = outname
            entnm(enttop,BUSNM_IN) = inname
            entnm(enttop,BUSNM_SN) = sername
            entdc(enttop) = vardesc
            entpar(enttop,1) = entspc + 1
            entpar(enttop,2) = esp
            entpar(enttop,3) = dynini
            entpar(enttop,4) = stagg
            entpar(enttop,6) = fmul
            entpar(enttop,8) = nmosaic
            entpar(enttop,BUSPAR_WLOAD) = 0
            entpar(enttop,BUSPAR_HZD)   = 0
            entpar(enttop,BUSPAR_MONOT) = 0
            entpar(enttop,BUSPAR_MASSC) = 0
            entpar(enttop,BUSPAR_VMIN)  = ivmin
            entpar(enttop,BUSPAR_VMAX)  = ivmax
            entspc = entpar(enttop,1) + esp - 1
            lindex = entpar(enttop,1)
      endif

      if (bus.eq."D") then
         do 20 i=1,dyntop

!           verifier si la meme description existe deja
            if (vardesc.eq.dyndc(i)) then
               if (varname.ne.dynnm(i,1)) then
                  write (6,903) varname,dynnm(i,1),dyndc(i)
                  call qqexit(1)
               endif
            endif

            if (varname.eq.dynnm(i,1)) then
               if (vardesc.ne.dyndc(i)) then
                  write (6,901) varname,vardesc,dyndc(i)
                  call qqexit(1)
               endif
               esp = n*nk
               if (shape.eq."ROW") esp = n
               if (dynpar(i,2).ne.(esp*fmul*fmosaik)) then
                  write (6,902) varname,dynpar(i,2),(esp*fmul*fmosaik)
                  call qqexit(1)
               endif
               lindex = dynpar(i,1)
               goto 601
            endif
 20      continue
         if (buslck) goto 601
            dyntop = dyntop + 1
            if (dyntop .gt. maxbus) then
               write(6,906) dyntop,maxbus
               call qqexit(1)
            end if
            esp = n*nk
            dynpar(dyntop,7) = nk
            if (shape.eq."ROW") then
               esp = n
               dynpar(dyntop,7) = 1
            endif
            dynpar(dyntop,5) =  esp
            esp = esp*fmul*fmosaik
            dynnm(dyntop,BUSNM_VN) = varname
            dynnm(dyntop,BUSNM_ON) = outname
            dynnm(dyntop,BUSNM_IN) = inname
            dynnm(dyntop,BUSNM_SN) = sername
            dyndc(dyntop) = vardesc
            dynpar(dyntop,1) = dynspc + 1
            dynpar(dyntop,2) = esp
            dynpar(dyntop,3) = dynini
            dynpar(dyntop,4) = stagg
            dynpar(dyntop,6) = fmul
            dynpar(dyntop,8) = nmosaic
            dynpar(dyntop,BUSPAR_WLOAD) = 0
            dynpar(dyntop,BUSPAR_HZD)   = 0
            dynpar(dyntop,BUSPAR_MONOT) = 0
            dynpar(dyntop,BUSPAR_MASSC) = 0
            dynpar(dyntop,BUSPAR_VMIN) = ivmin
            dynpar(dyntop,BUSPAR_VMAX) = ivmax
            dynspc = dynpar(dyntop,1) + esp - 1
            lindex = dynpar(dyntop,1)
            !#TODO: should we remove this special case for Tracers?
            if ((varname(1:3)=='TR/').and.(trim(varname(ind:))==':P')) then
               call tracers_attributes2(trim(longname), &
                    dynpar(dyntop,BUSPAR_WLOAD), dynpar(dyntop,BUSPAR_HZD), &
                    dynpar(dyntop,BUSPAR_MONOT), dynpar(dyntop,BUSPAR_MASSC), &
                    vmin0)
               vmin = max(vmin0, vmin)
               dynpar(dyntop,BUSPAR_VMIN) = transfer(vmin, dynpar(dyntop,BUSPAR_VMIN))
            endif
      endif

      if (bus.eq."P") then
         do 30 i=1,pertop

!           verifier si la meme description existe deja
            if (vardesc.eq.perdc(i)) then
               if (varname.ne.pernm(i,1)) then
                  write (6,903) varname,pernm(i,1),perdc(i)
                  call qqexit(1)
               endif
            endif

            if (varname.eq.pernm(i,1)) then
               if (vardesc.ne.perdc(i)) then
                  write (6,901) varname,vardesc,perdc(i)
                  call qqexit(1)
               endif
               esp = n*nk
               if (shape.eq."ROW") esp = n
               if (perpar(i,2).ne.(esp*fmul*fmosaik)) then
                  write (6,902) varname,perpar(i,2),(esp*fmul*fmosaik)
                  call qqexit(1)
               endif
               lindex = perpar(i,1)
               goto 601
            endif
 30      continue
         if (buslck) goto 601
            pertop = pertop + 1
            if (pertop .gt. maxbus) then
               write(6,906) pertop,maxbus
               call qqexit(1)
            end if
            esp = n*nk
            perpar(pertop,7) = nk
            if (shape.eq."ROW") then
               esp = n
               perpar(pertop,7) = 1
            endif
            perpar(pertop,5) = esp
            esp = esp*fmul*fmosaik
            pernm(pertop,BUSNM_VN) = varname
            pernm(pertop,BUSNM_ON) = outname
            pernm(pertop,BUSNM_IN) = inname
            pernm(pertop,BUSNM_SN) = sername
            perdc(pertop) = vardesc
            perpar(pertop,1) = perspc + 1
            perpar(pertop,2) = esp
            perpar(pertop,3) = dynini
            perpar(pertop,4) = stagg
            perpar(pertop,6) = fmul
            perpar(pertop,8) = nmosaic
            perpar(pertop,BUSPAR_WLOAD) = 0
            perpar(pertop,BUSPAR_HZD)   = 0
            perpar(pertop,BUSPAR_MONOT) = 0
            perpar(pertop,BUSPAR_MASSC) = 0
            perpar(pertop,BUSPAR_VMIN) = ivmin
            perpar(pertop,BUSPAR_VMAX) = ivmax
            perspc = perpar(pertop,1) + esp - 1
            lindex = perpar(pertop,1)
      endif

      if (bus.eq."V") then
         do 40 i=1,voltop

!           verifier si la meme description existe deja
            if (vardesc.eq.voldc(i)) then
               if (varname.ne.volnm(i,1)) then
                  write (6,903) varname,volnm(i,1),voldc(i)
                  call qqexit(1)
               endif
            endif

            if (varname.eq.volnm(i,1)) then
               if (vardesc.ne.voldc(i)) then
                  write (6,901) varname,vardesc,voldc(i)
                  call qqexit(1)
               endif
               esp = n*nk
               if (shape.eq."ROW") esp = n
               if (volpar(i,2).ne.(esp*fmul*fmosaik)) then
                  write (6,902) varname,volpar(i,2),(esp*fmul*fmosaik)
                  call qqexit(1)
               endif
               lindex = volpar(i,1)
               goto 601
            endif
 40      continue
         if (buslck) goto 601
            voltop = voltop + 1
            if (voltop .gt. maxbus) then
               write(6,906) voltop,maxbus
               call qqexit(1)
            end if
            esp = n*nk
            volpar(voltop,7) = nk
            if (shape.eq."ROW") then
               esp = n
               volpar(voltop,7) = 1
            endif
            volpar(voltop,5) = esp
            esp = esp*fmul*fmosaik
            volnm(voltop,BUSNM_VN) = varname
            volnm(voltop,BUSNM_ON) = outname
            volnm(voltop,BUSNM_IN) = inname
            volnm(voltop,BUSNM_SN) = sername
            voldc(voltop) = vardesc
            volpar(voltop,1) = volspc + 1
            volpar(voltop,2) = esp
            volpar(voltop,3) = dynini
            volpar(voltop,4) = stagg
            volpar(voltop,6) = fmul
            volpar(voltop,8) = nmosaic
            volpar(voltop,BUSPAR_WLOAD) = 0
            volpar(voltop,BUSPAR_HZD)   = 0
            volpar(voltop,BUSPAR_MONOT) = 0
            volpar(voltop,BUSPAR_MASSC) = 0
            volpar(voltop,BUSPAR_VMIN) = ivmin
            volpar(voltop,BUSPAR_VMAX) = ivmax
            volspc = volpar(voltop,1) + esp - 1
            lindex = volpar(voltop,1)
      endif

 601  continue


! verifier que le nom de la variable est unique

      if (bus.ne.'E') then
         do i=1,enttop
            if (varname.eq.entnm(i,1)) then
               write(6,905) varname,'E'
               call qqexit(1)
            endif
         end do
      endif

      if (bus.ne.'D') then
         do i=1,dyntop
            if (varname.eq.dynnm(i,1)) then
               write(6,905) varname,'D'
               call qqexit(1)
            endif
         end do
      endif

      if (bus.ne.'P') then
         do i=1,pertop
            if (varname.eq.pernm(i,1)) then
               write(6,905) varname,'P'
               call qqexit(1)
            endif
         end do
      endif

      if (bus.ne.'V') then
         do i=1,voltop
            if (varname.eq.volnm(i,1)) then
               write(6,905) varname,'V'
               call qqexit(1)
            endif
         end do
      endif


      do i=1,enttop
!        verifier que le nom de 4 lettres est unique
         if (outname.eq.entnm(i,2).and.varname.ne.entnm(i,1)) then
            samename = entnm(i,1)
            write(6,904) varname, outname, samename
            call qqexit(1)
         endif
!        verifier qu'une variable ne porte qu'un seul nom de 4 lettres
         if (varname.eq.entnm(i,1).and.outname.ne.entnm(i,2)) then
            othername = entnm(i,2)
            write(6,907) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "varname" ne soit pas identique a un "outname"
         if (varname.ne.entnm(i,1).and.varname.eq.entnm(i,2)) then
            othername = entnm(i,2)
            write(6,908) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "outname" ne soit pas identique a un "varname"
         if (outname.eq.entnm(i,1).and.varname.ne.entnm(i,1)) then
            write(6,908) entnm(i,1), varname, outname
            call qqexit(1)
         endif
      end do

      do i=1,dyntop
         if (outname.eq.dynnm(i,2).and.varname.ne.dynnm(i,1)) then
            samename = dynnm(i,1)
            write(6,904) varname, outname, samename
            call qqexit(1)
         endif
!        verifier qu'une variable ne porte qu'un seul nom de 4 lettres
         if (varname.eq.dynnm(i,1).and.outname.ne.dynnm(i,2)) then
            othername = dynnm(i,2)
            write(6,907) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "varname" ne soit pas identique a un "outname"
         if (varname.ne.dynnm(i,1).and.varname.eq.dynnm(i,2)) then
            othername = dynnm(i,2)
            write(6,908) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "outname" ne soit pas identique a un "varname"
         if (outname.eq.dynnm(i,1).and.varname.ne.dynnm(i,1)) then
            write(6,908) dynnm(i,1), varname, outname
            call qqexit(1)
         endif
      end do

      do i=1,pertop
         if (outname.eq.pernm(i,2).and.varname.ne.pernm(i,1)) then
            samename = pernm(i,1)
            write(6,904) varname, outname, samename
            call qqexit(1)
         endif
         if (varname.eq.pernm(i,1).and.outname.ne.pernm(i,2)) then
            othername = pernm(i,2)
            write(6,907) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "varname" ne soit pas identique a un "outname"
         if (varname.ne.pernm(i,1).and.varname.eq.pernm(i,2)) then
            othername = pernm(i,2)
            write(6,908) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "outname" ne soit pas identique a un "varname"
         if (outname.eq.pernm(i,1).and.varname.ne.pernm(i,1)) then
            write(6,908) pernm(i,1), varname, outname
            call qqexit(1)
         endif
      end do

      do i=1,voltop
         if (outname.eq.volnm(i,2).and.varname.ne.volnm(i,1)) then
            samename = volnm(i,1)
            write(6,904) varname, outname, samename
            call qqexit(1)
         endif
         if (varname.eq.volnm(i,1).and.outname.ne.volnm(i,2)) then
            othername = volnm(i,2)
            write(6,907) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "varname" ne soit pas identique a un "outname"
         if (varname.ne.volnm(i,1).and.varname.eq.volnm(i,2)) then
            othername = volnm(i,2)
            write(6,908) varname, outname, othername
            call qqexit(1)
         endif
!        verifier qu'un "outname" ne soit pas identique a un "varname"
         if (outname.eq.volnm(i,1).and.varname.ne.volnm(i,1)) then
            write(6,908) volnm(i,1), varname, outname
            call qqexit(1)
         endif
      end do


 901  format (/1x,"==> STOP IN GESDICT: CONFLICT IN '",a16, &
                  "' DESCRIPTION."/4x,"ALREADY ACCEPTED: ",a/11x, &
                  "ATTEMPTED: ",a/)
 902  format (/1x,"==> STOP IN GESDICT: CONFLICT IN '",A16, &
                  "' DIMENSION."/4x,"ALREADY ACCEPTED: ",i9/11x, &
                  "ATTEMPTED: ",i9/)
 903  format (/1x,"==> STOP IN GESDICT: NAME CONFLICT.", &
                  " VARIABLES '",a16,"' AND '",a16,"'"/, &
                  " SHARE THE SAME DESCRIPTION. DESCRIPTION IS :"/, &
                  " '",A,"'"/)
 904  format (/1x,"==> STOP IN GESDICT: CONFLICT FOR '",A16, &
                  "' OUTPUT NAME."/5x,'"',a4,'"'," ALREADY ACCEPTED", &
                  " FOR VARIABLE '",a16,"'."/)

 905  format (/1x,"==> STOP IN GESDICT: CONFLICT FOR '",A16, &
                  "' VARIABLE NAME.",/5x,"THIS NAME HAS", &
                  " ALREADY BEEN ACCEPTED IN BUS ",'"',a1,'".'/)

 906  format (/1x,"==> STOP : ",i4," EXCEEDS MAXBUS (",i4,")  &!!!")
      

 907  format (/1x,"==> STOP IN GESDICT: CONFLICT FOR '",A16, &
                  "' VARIABLE NAME.",/5x,"THIS VARIABLE HAS", &
                  " TWO DIFFERENT OUTPUT NAMES: ", &
                  '"',a4,'"'," AND ",'"',A4,'".'/)
 908  format (/1x,"==> STOP IN GESDICT: CONFLICT FOR '",A16, &
                  "' VARIABLE NAME.",/5x,"VARIABLE ", &
                  '"',a4,'"'," HAS THE OUTPUT NAME ",'"',A4,'".'/)

   !-------------------------------------------------------------------
   return
end
