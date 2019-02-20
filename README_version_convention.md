MIG/GEM Version Naming Convention
=================================

> This document is a copy of the
> [GEM Naming Convention on the CMC wiki](https://wiki.cmc.ec.gc.ca/wiki/GEM/Version_Numbers)

From GEM version 4.2 a new packaging and a new naming/numbering convention is introduced.

GEM is now packaged with SSM. The name GEM now refer to all parts related to
the GEM ecosystem: dynamic, RPN physics, scripts... each with its own version
number (thus GEM may have a different number than the dynamics).

Production/Released Versions
----------------------------

Version up to GEM 4.1.4 were using a 3 digits numbering scheme where code
differences can be important between 2 versions with any of the 3 digit
changed. Minor changes (minor bugfix) were often done *in place* without
changing the version number.  
From version 4.2 on, GEM uses a 2 digits + 1 numbering scheme (see below for
details). The main difference from before being that the last digit change
only (and always) for bugfix (minor revision); a released version will never
be changed without changing its number.

The new naming convention is as follow (e.g. `GEM/4.2.3`)

    GEM/M.m.f
    GEM/M.m-b.f

where:

  * `M `: Major version number
  * `.m`: minor version number
  * `-b`: (*optional*) branch number<br>Some development following a
    version but not on the main GEM dev trunk
  * `.f`: bug fix number<br>Even minor bug fix would get a version increase
    (no more touch up). Bug fix number are increased from zero:
    * `0 `: 1st release
    * `>0`: bug fix version

> As long as the first 2 digits (`M.m-b`) are the same, model usage should
> remain the same for end user.

Development Versions
--------------------

Previously only release (production ready) versions were distributed,
development code was living in some developer's directory.
From version 4.2 on, development versions are packaged and distributed for
easier code sharing among main developers and potiential use by user needing
the latest developments.

The new naming convention for development versions is as follow
(From GEM 4.2 to GEM 4.7) (e.g. `GEM/x/4.4.0-b8`)

    GEM/x/M.m.0-Si
    GEM/x/M.m-b.0-Si

The new naming convention for development versions is as follow (From GEM 4.8)

    GEM/x/M.m.Si
    GEM/x/M.m-b.Si

  * `x/`: identify an e**X**perimental version
  * `M `: **M**ajor version number
  * `.m`: **m**inor version number
  * `-b`: (*optional*) **b**ranch number<br>Some development following a
    version but not on the main GEM dev trunk
  * `S`: development **S**tate, one of the following:
    * **a**: (*alpha*) code still under heavy development, new unstable
      features introduction<br>*Who should use alpha (a) versions*:
      **Only core GEM developers should use alpha versions**.
    * **b**: (*beta*) debugging and stabilisation of the code base, new
      feature may be introduced if they are stable/mature enough.  
      *Who should use beta (b) versions*: These versions *may be used with
      caution for cutting edge projects* and other developments needing the
      latest features/fixes.
    * **rc**: (*release candidate*) mature code in testing mode, will be made
      an official release with at most a few minor bug fix  
      *Who should use release candidates (rc)*: *Anyone planning to use the
      new GEM version should run tests with the latest "rc" version* to make
      sure needed bug fix can be introduced before release.
  * `i`: development **i**teration number<br>Developement iteration number is
    reset to 1 when *development state* change;<br>the higher the
    **i**teration number is, the closest code is to the next *development
    state/stage*, the more stable or feature full it should be.

Git Repositories Tags
---------------------

Tags in the Git repositories follow mostly the same conventions exept for
the follwoing differences:

  * component name is in lower case
  * `/x/` is never added
  * `_` (underscores) are used as separators instead of `/` (slashes)

See Also
--------

  * Main doc: [README.md](README.md).
  * Building instructions: [README\_building.md](README_building.md)
  * Running instructions: [README\_running.md](README_running.md)
  * Developing instructions: [README\_developing.md](README_developing.md)
  * Installing instructions: [README\_installing.md](README_installing.md)
  * Naming Conventions: [README\_version\_convention.md](README_version_convention.md)
  * [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki)
    * [GEM Naming Convention](https://wiki.cmc.ec.gc.ca/wiki/GEM/Version_Numbers)
  * [CMC Bugzilla](http://bugzilla.cmc.ec.gc.ca)


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
*[GEM]: Global Environmental Multi-scale atmosperic model from RPN, ECCC  
*[MIG]: Model Infrastructure Group at RPN, ECCC  

*[SSM]: Simple Software Manager (a super simplified package manager for software at CMC/RPN, ECCC)  
*[RDE]: Research Development Environment, a super simple code dev. env. at RPN  
