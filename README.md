MIG: Model Infrastructure Group, RPN, MRD, STB, ECCC, GC, CA
============================================================

*Developed at the Recherche en Prevision Numerique (RPN), Meteorological Research
Division (MRD), Science and Technology Branch (STB), Environment and Climate
Change Canada (ECCC)*

This repository includes the source code, utilities and scripts maintained by
the Model Infrastructure Group (MIG). Its main product is the Global
Environmental Multi-scale model (GEM).  Steps required to compile and run
GEM (excluding some 3rd party dependencies) are described.

  * The *official* version of the code is on the [ECCC gitlab server](https://gitlab.science.gc.ca/MIG/mig)
  * Issue tracking is done on [CMC/GEM's Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?product=GEM&resolution=---)
  * Documentation can be found on the [CMC's wiki](https://wiki.cmc.ec.gc.ca/wiki/Gem).

*These are only available from within the ECCC or Science networks,
and requires an ECCC or science.gc.ca login.*


Getting the code
----------------

If not already done, you may clone the MIG repository and checkout the version you want to run/work on with the following command (example for version 5.0.0):

    MYVERSION=5.0.0  ## Change this to the desired version

    ## There are 2 URL options:

    ## Option 1: Main URL, GitLab.science account needed
    MYURL=git@gitlab.science.gc.ca:MIG/mig

    ## Option 2: HTTPS URL, "git push" not possible with this URL
    # MYURL=https://gitlab.science.gc.ca/MIG/mig.git

    MYTAG=mig_${MYVERSION}
    git clone ${MYURL} ${MYTAG}
    cd ${MYTAG}

    ## Check if ${MYTAG} exists
    taglist=":$(git tag -l | tr '\n' ':')"
    if [[ "x$(echo ${taglist} | grep :${MYTAG}:)" == "x" ]] ; then
        echo "===> ERROR: not such tag: ${MYTAG} <==="
    fi

    ## There are 2 branch options (existing or new branch):

    ## Option 1: Continue on existing branch - ${MYTAG} is the HEAD of its branch
    MYBRANCH=${MYTAG%.*}-branch
    git checkout ${MYBRANCH}

    ## Option 2: Develop on a new branch - ${MYTAG} is NOT the HEAD of any branch
    # MYBRANCH=${MYTAG}-${USER}-branch
    # git checkout -b ${MYBRANCH} ${MYTAG}


Building, Running and Modifying MIG/GEM
---------------------------------------

  * Building instructions: [README\_building.md](README_building.md).
  * Running instructions: [README\_running.md](README_running.md).
  * Developing instructions: [README\_developing.md](README_developing.md).


Documentation
-------------

Further documentation can be found on the [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki):

  * [GEM wiki page](https://wiki.cmc.ec.gc.ca/wiki/Gem)
      * [GEM change log page](https://wiki.cmc.ec.gc.ca/wiki/GEM/Change_Log)
  * [RPNphy wiki page](https://wiki.cmc.ec.gc.ca/wiki/Rpnphy)
      * [RPNphy change log page](https://wiki.cmc.ec.gc.ca/wiki/RPNPhy/Change_Log)
  * [SCM wiki page](https://wiki.cmc.ec.gc.ca/wiki/SCM)
  * [Modelutils wiki page](https://wiki.cmc.ec.gc.ca/wiki/Modelutils)
  * [RPNpy wiki page](https://wiki.cmc.ec.gc.ca/wiki/Rpnpy)
  * [RDE wiki page](https://wiki.cmc.ec.gc.ca/wiki/RDE)


Reporting Issues
----------------

If you identify specific bugs or problems in the code, you can report it by
creating an issue in one of MIG's CMC bugzilla products.

  * [GEM Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gem)
  * [GEMdyn Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gemdyn)
  * [RPNphy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN-Phy)
  * [RPNpy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN.py)
  * [RDE Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=rde)

For feedback on running and development guidelines use the contacts below.


See Also
--------

  * Main doc: [README.md](README.md).
  * Building instructions: [README\_building.md](README_building.md)
  * Running instructions: [README\_running.md](README_running.md)
  * Developing instructions: [README\_developing.md](README_developing.md)
  * Installing instructions: [README\_installing.md](README_installing.md)
  * Naming Conventions: [README\_version\_convention.md](README_version_convention.md)
  * [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki)
  * [CMC Bugzilla](http://bugzilla.cmc.ec.gc.ca)


Contact
-------

For questions about using the Version Control System, and guidelines:
<ec.service-gem.ec@canada.ca>, <stephane.chamberland@canada.ca>,  

> **TODO:** add other MIG members after getting their permission (StephaneG?, ChantalP?, VivianL?, RonMcTC?) <@canada.ca>,


Abbreviations
-------------

*[CMC]: Centre Meteorologique Canadien  
*[RPN]: Recherche en Previsions Numeriques (Section of MRD/STB/ECCC)  
*[MRD]: Meteorological Research Division (division of STB/ECCC)  
*[STB]: Science and Technology Branch (branch of ECCC)  
*[ECCC]: Environment and Climate Change Canada  
*[EC]: Environment and Climate Change Canada (now ECCC)  
*[GC]: Government of Canada  
*[SSC]: Shared Services Canada  

*[SPS]: Surface Prediction System, driver of RPN physics surface processes  
*[SCM]: Single Column Model, driver of RPN physics  
*[GEM]: Global Environmental Multi-scale atmospheric model from RPN, ECCC  
*[MIG]: Model Infrastructure Group at RPN, ECCC  

*[SSM]: Simple Software Manager (a super simplified package manager for software at CMC/RPN, ECCC)  
*[RDE]: Research Development Environment, a super simple code dev. env. at RPN  


> **TODO**: automate existing or new branch selection
