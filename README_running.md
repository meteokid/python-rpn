
Running MIG/GEM from Version Control
====================================

This document describes the basic steps required to run MIG/GEM using Code from Version Control.
It includes instructions to import the code, do the initial setup,
compile/build the binaries and then run with your configuration.

> **This README assumes**:  
>
> The MIG repository is already cloned and the desired version is already checked out.  
> *See [Getting the code](README.md#getting-the-code) section for more details.*  
>
> The code is already compiled and built.  
> *See [README\_building.md](README_building.md) for more details.*


To modify the code and scripts, please refer to [README\_developing.md](README_developing.md).

**Table of Contents**

  * [Configuration](#configuration)
  * [Running](#running)
    * [Running Interactively](#running-interactively)
    * [Running in Batch mode](#running-in-batch-mode)
  * [Cleaning Up](#cleaning-up)
  * [See Also](#see-also)


Configuration
-------------

Configuration files are in the `GEM_cfgs/cfg_0000/` dir.  
> **TODO:** more details

You may list all sample configurations with:

    devadd --list

You can then import a sample configuration with:

    devadd --ln --copy CONFIG_NAME

Use "`devadd --help`" for a list of all options.

To know more about what are the options to run GEM or SCM you may refer to:
  * [GEM wiki page](https://wiki.cmc.ec.gc.ca/wiki/Gem)
  * [SCM wiki page](https://wiki.cmc.ec.gc.ca/wiki/SCM)

Running
-------

### Running Interactively ###

    gemrun CONFIG_NAME --ptopo ${NPEX:-1}x${NPEY:-1}x${OMP_NUM_THREADS:-1}

> **TODO**:
> * Results are in TODO
> * log/listing files are in TODO

Use "`gemrun --help`" for a list of all options.


### Running in Batch mode ###

> *Note:* Running in batch mode is done using the maestro sequencer.
> Maestro needs to be pre-installed and configured on the machine and in
> your user account.  
> This setup is beyond the scope of this doc... (**TODO**: reference to external doc)

> Running in batch mode is currently only possible with installed GEM and SCM versions.  
> Work to support running batch from version controled code build locally is in progress.


Cleaning Up
-----------

You can remove the files, dir and links created by the setup process with
the following.

    make distclean
    rm -rf GEM_cfg*

After this, in order to be able to compile and run again,
you'll need to re-do the whole setup process.


See Also
--------

  * Main doc: [README.md](README.md)
  * Building instructions: [README\_building.md](README_building.md)
  * Running instructions: [README\_running.md](README_running.md)
  * Developing instructions: [README\_developing.md](README_developing.md)
  * Installing instructions: [README\_installing.md](README_installing.md)
  * Naming Conventions: [README\_version\_convention.md](README_version_convention.md)
  * [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki)
    * [GEM wiki page](https://wiki.cmc.ec.gc.ca/wiki/Gem)
    * [SCM wiki page](https://wiki.cmc.ec.gc.ca/wiki/SCM)
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
