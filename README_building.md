
Building MIG/GEM from Version Control
=====================================

This document describes the basic steps required to compile and build
MIG/GEM source code using Version Control.

> **This README assumes**:  
>
> The MIG repository is already cloned and the desired version is already checked out.  
> *See [Getting the code](README.md#getting-the-code) section for more details.*


**Table of Contents**

   * [Quick Start](#quick-start)
   * [Basic profile and directory setup](#basic-profile-and-directory-setup)
   * [Initial Setup](#initial-setup)
      * [Shell Env. SetUp](#shell-env-setup)
      * [Files, Directories and Links SetUp](#files-directories-and-links-setup)
   * [Compiling and Building](#compiling-and-building)
   * [Cleaning up](#cleaning-up)
   * [See Also](#see-also)


Quick Start
-----------

    export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
    # ISOFFICIAL=--official    ## For explicit use by librarian only
    . ./.setenv.dot -v ${ISOFFICIAL}

    ## Option 1: w/o running env.
    ouv_exp_mig -v
    rdemklink -v

    ## Option 2: with running env.
    # . gemdev.dot myexp -v --gitlocal

    make buildclean
    make dep
    make vfiles

    export MAKE_NPE=6
    make libs -j ${MAKE_NPE:-6}
    make abs  #-j ${MAKE_NPE:-6}

    make distclean
    rm -f */ssmusedep*bndl gem/ATM_MODEL_*


Basic profile and directory setup
------------------------------------

The build system is working on the EC/CMC and GC/Science network.  
Your account needs to have an up to date *.profile* environment setup.  
See this [SSC Getting Started doc](https://portal.science.gc.ca/confluence/display/SCIDOCS/Getting+Started+-+Setting+Up+the+Environment)

A scratch dir (big temporary space) is needed to host compilation,
building and running produced files.
This scratch dir is found by the system with the `${storage_model}` var.

    export storage_model=/PATH/TO/SCRATCH/DIR/

> **TODO**:
>  * other expected dir
>  * other expected env vars


Initial Setup
----------------

Setting up the Shell Environment and Working Directories.

The MIG/GEM compiling, building and running systems depend on a few Shell
(*Bash is the only supported Shell to date*) environment variables, PATHs,
directories, links and files.
The following commands perform that setup.


#### Shell Env. SetUp ####

*This step needs to be performed every time a new Shell is opened.*

> *Note 1*: The setup files are available for the EC/CMC and GC/Sciences networks.
> Compiler and other external dependencies are available for these networks only.  
> See [README\_developing.md](README_developing.md) on how to modify this.

> *Note 2*: The compiled objects and binaries being big files they are put under the
> build directory which is a link to a *scratch*/*big_data* space.  
> The location of the *scratch*/*big_data* space is defined with the
> `${storage_model}` environment variable.

    export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
    # ISOFFICIAL=--official    ## For explicit use by librarian only
    . ./.setenv.dot -v ${ISOFFICIAL}


#### Files, Directories and Links SetUp ####

*This step needs to be performed only once for this working directory
or after performing "`make distclean`".*

    ## Option 1: Compile/Build environment only
    ouv_exp_mig -v
    rdemklink -v

    ## Option 2: Compile/Build and interactive/batch running environment
    # . gemdev.dot myexp -v --gitlocal


Compiling and Building
----------------------

> *Note 1*: If you're planning on running in batch mode, submitting to
> another machine, make sure you do the *initial setup*, including compilation,
> on the machine where the job will be submitted.

> *Note 2*: The compiler is specified as an external dependency in
> the "`_migdep/DEPENDENCIES.external.*.bndl`" files.  
> Compiler options, rules and targets are specified in each components's
> Makefiles (`*/include/Makefile.local*mk` files).  
> See [README\_developing.md](README_developing.md) for more details.

Initially it is best to make sure to start with a clean slate.  
*This should be done once initially and every time the "dependency list" is modified.*

    make buildclean


Use the following Makefile targets to compile the build libs and abs.  
*This needs to be done initially and every time the code is modified.*

    make dep
    make vfiles
    make libs -j ${MAKE_NPE:-6}
    make abs  #-j ${MAKE_NPE:-6}

> *Note*: If you're planning on running in batch mode, submitting to another
> machine, make sure you do the *initial setup*, including compilation,
> on the machine where the job will be submitted.

Cleaning up
-----------

To remove all files created by the setup, compile and build process, use the `distclean` target.

    make distclean
    rm -f */ssmusedep*bndl gem/ATM_MODEL_*


See Also
--------

  * Main doc: [README.md](README.md)
  * Building instructions: [README\_building.md](README_building.md)
  * Running instructions: [README\_running.md](README_running.md)
  * Developing instructions: [README\_developing.md](README_developing.md)
  * Installing instructions: [README\_installing.md](README_installing.md)
  * Naming Conventions: [README\_version\_convention.md](README_version_convention.md)
  * [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki)
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
*[GEM]: Global Environmental Multi-scale atmospheric model from RPN, ECCC  
*[MIG]: Model Infrastructure Group at RPN, ECCC  

*[SSM]: Simple Software Manager (a super simplified package manager for software at CMC/RPN, ECCC)  
*[RDE]: Research Development Environment, a super simple code dev. env. at RPN  
