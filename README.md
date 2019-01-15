MIG: Model Infrastructure Group, RPN, MRD, STB, ECCC, GC, CA
============================================================

*Developed at the Recherche en Prevision Numerique (RPN), Meteorological Research Division (MRD), Science and Technology Branch (STB), Environment and Climate Change Canada (ECCC)*

This repository includes the source code, utilities and scripts maintained by the Model Infrastructure Group (MIG). Its main product is the Global Environmental Multi-scale model (GEM).  Steps required to compile and run GEM (excluding some 3rd party dependencies) are described.

  * The *official* version of the code is on the [ECCC gitlab server](https://gitlab.science.gc.ca/MIG/mig)
  * Issue tracking is done on [CMC/GEM's Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?product=GEM&resolution=---)
  * Documentation can be found on the [CMC's wiki](https://wiki.cmc.ec.gc.ca/wiki/Gem).

*These are only available from within the ECCC or Science networks, and requires an ECCC or science.gc.ca login.*


Getting the code
----------------

If you are reading this, you probably already did clone the MIG repository and checked out the version you want to run/work on.  
If not you may do (example for version 5.0.0):

        MYVERSION=5.0.0                                 ## Obviously, you'll need to change this to the desired version
        ## MYURL=git@gitlab.science.gc.ca:MIG/mig       ## You'll need a GitLab.science account for this URL
        MYURL=https://gitlab.science.gc.ca/MIG/mig.git
        git clone ${MYURL} mig_${MYVERSION}
        cd mig_${MYVERSION}
        git checkout -b mig_${MYVERSION}-${USER}-branch mig_${MYVERSION}


Running and Modifying GEM
-------------------------

  * Running instructions: [README\_running.md](README_running.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 


Reporting Issues
----------------

If you identify specific bugs or problems in the code, you can report it by creating an issue in [GEM's Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?product=GEM&resolution=---).  
For feedback on running and development guidelines use the contacts below.


See Also
--------

  * Running instructions: [README\_running.md](README_running.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 
  * Installing instructions: [README\_installing.md](README_installing.md). 


Contact
-------

For questions about using the Version Control System, and guidelines:
<ec.service-gem.ec@canada.ca>, <stephane.chamberland@canada.ca>,  
**TODO:** add other MIG members after getting their permission (StephaneG?, ChantalP?, VivianL?, RonMcTC?) <@canada.ca>,
