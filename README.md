RPN Python tools and interface to RPN libraries: RPN, MRD, STB, ECCC, GC, CA
============================================================================

This document is intended as a notebook and checklist for the RPNpy
librarian. It is only valid on EC/CMC and GC/Science Networks.

Python RPN (RPNpy) is a collection of Python modules and scripts developed
at RPN for scientific use.
Its main purpose is to act as an augmented interface to functions in: 

  * librmn, vgrid: read, manipulate, interpolate and write RPN Std files data
  * burpc: read, manipulate and write BURP files

in Python with a very simple and minimalist interface while also 
offering powerful lower level functions for special cases. 
The Python RPN package can be used for fast prototyping and full development 
of production scripts.
It allows to use powerful python data analysis and visualization tools 
like numpy, scipy and matplotlib among others.

Note: In order to use Python RPN, you must have the following shared libraries installed and 
their paths set in LD_LIBRARY_PATH, LIBPATH or PYTHONPATH: librmn, vgrid (6.0 or later) and burpc 

**Table of Contents**

   * [Quick Start: Building, Installing from Version Control](#quick-start-building-installing-from-version-control)
   * [General Information](#general-information)
      * [Basic profile and directory setup](#basic-profile-and-directory-setup)
      * [Layout](#layout)
      * [Closing Issues](#closing-issues)
      * [Documentation update](#documentation-update)
   * [Building, Installing from Version Control](#building-installing-from-version-control)
      * [Getting the code](#getting-the-code)
      * [Update Dependencies](#update-dependencies)
      * [Committing the Code](#committing-the-code)
      * [Initial Setup](#initial-setup)
            * [Shell Env. SetUp](#shell-env-setup)
            * [Files, Directories and Links SetUp](#files-directories-and-links-setup)
      * [Compiling and Building](#compiling-and-building)
      * [Testing](#testing)
      * [Install](#install)
            * [Post Install](#post-install)
      * [Uninstall](#uninstall)
      * [Cleaning up](#cleaning-up)
   * [Misc](#misc)
      * [Patching the git repo](#patching-the-git-repo)

Quick Start: Building, Installing from Version Control
======================================================

This section is just a condensed version of the steps needed to release from
the git repository. It implies the mandatory steps for your SHELL env. setup
and directories structure are already done. It also implies that the code is
fully committed in the git repository and no other changes are needed.
These steps needs to be done in the specified order.

**To Do ONLY on the Front End Machine**

        MYVERSION=2.1.b3  ## Obviously, you'll need to change this to the desired version
        NAME=rpnpy
        MYURL=https://gitlab.science.gc.ca/MIG/${NAME}.git
        git clone ${MYURL} ${NAME}_${MYVERSION} ${NAME}
        cd ${NAME}
        git checkout -b ${NAME}_${MYVERSION}-${USER}-branch ${NAME}_${MYVERSION}

        . ./.setenv.dot --dev
        rdemklink -v
        rm -f include/.restricted
        touch $(rdevar srcpathfile)

**To Do on the FrontEnd AND BackEnd Machines**

        make forced_extlibdotfile
        make rpnpy_vfiles

**To Do ONLY on the Front End Machine**

        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make components_ssm_all  SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}

        make components_install CONFIRM_INSTALL=yes \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG  SSM_BASE2=""

        make distclean

General Information
===================

Basic profile and directory setup
---------------------------------

**TODO**: 

  * list expected dir structure (~/SsmDepot, ~/SsmBundle, ...)
  * list expected env var and paths...
  * gmake based
  * supported compilers

Layout
------

This repository is built following some conventions for
RDE (RPN Development Environment) and MIG (Model Infrastructure Group).

It contains the following sub-directories:

  * `bin/       `: used for scripts, will be added to the `PATH`
  * `include/   `: used for included files and sub-Makefiles, will be added to the `INCLUDE_PATH`
  * `lib/       `: will be added to the `LIBRARY_PATH`
  * `lib/python/`: used for python modules, will be added to the `PYTHONPATH`
  * `src/       `: used for source code, will be added to the source path (`VPATH`).
  * `share/     `: used for any other content
  * `.ssm.d/    `:

It contains the following files:

  **TODO: Complete list with mandatory and optional**
  * `.name`:
  * `.setenv.dot`:
  * `bin/.env_setup.dot`:
  * `VERSION`:
  * `DEPENDENCIES.mig.bndl`:
  * `DEPENDENCIES.external.cmc.bndl`, `DEPENDENCIES.external.science.bndl`:
  * `include/Makefile.local.NAME.mk`: ... note that all these components' Makefiles will be merged/included into the main one, please make sure modifications to them does not have undesired side effect on other components.
  * `include/Makefile.ssm.mk`:
  * `.restricted`:

  **TODO: Complete list of Mandatory Makefile vars and targets**

Closing Issues
--------------

**TODO**: review/close bugzilla issues


Documentation update
--------------------

**TODO**: review doc on wiki


Building, Installing from Version Control
=========================================

Getting the code
----------------

If not already done, you may clone the git repository and checkout
the version you want to run/work on with the following command
(example for version 2.1.b3):

        NAME=rpnpy
        MYVERSION=2.1.b3                                  ## Obviously, you'll need to change this to the desired version
        MYURL=git@gitlab.science.gc.ca:MIG/${NAME}        ## You'll need a GitLab.science account for this URL
        ## MYURL=https://gitlab.science.gc.ca/MIG/${NAME}.git ## You cannot "git push" if you use the http:// URL
        git clone ${MYURL} ${NAME}_${MYVERSION} ${NAME}
        cd ${NAME}
        git checkout -b ${NAME}_${MYVERSION}-${USER}-branch ${NAME}_${MYVERSION}


Update Dependencies
-------------------

**TODO**:

  * Where/How to set change external dependencies (compiler, librmn, vgrid, rpn_comm, ...)
  * Where/How to set compiler options, new compiler rules...


Committing the Code
-------------------

**TODO Pre-commit**:

  * merge in code from other devs (and from other branches if any)
  * update version number
  * update bndl dependencies
  * test (testing section below)

**TODO**:

  * commit changes
  * tag version
  * push code and tags upstream (git push && git push --tags)


Initial Setup
-------------

Setting up the Shell Environment and Working Directories.

The compiling, building and running systems depend on a few Shell
(Bash is the only supported Shell to date) environment variables,
PATHs, directories, links and files.
The following commands perform that setup.

#### Shell Env. SetUp ####

*This step needs to be performed every time a new Shell is opened.*

*Note 1*: The setup files are available for the EC/CMC and GC/Sciences networks. 
Compiler and other external dependencies are available for these networks only.  
**TODO**: explanation on how to change this

*Note 2*: The compiled objects and binaries being big files they are put under
the build directory which is a link to a *scratch*/*big_data* space. 
The location of the *scratch*/*big_data* space is defined with the
`${storage_model}` environment variable.

        export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
        . ./.setenv.dot --dev

#### Files, Directories and Links SetUp ####

*This step needs to be performed only once for this working directory 
or after performing "`make distclean`".*

        ## Note: Once per working directory per ARCH
        rdemklink -v
        rm -f include/.restricted
        touch $(rdevar srcpathfile)


Building
--------

**To Do on the FrontEnd AND BackEnd Machines**

        make forced_extlibdotfile
        make rpnpy_vfiles


Testing
-------

**To Do on the FrontEnd AND BackEnd Machines**

        # export RPNPY_NOLONGTEST=1
        make alltests

Install
-------

Installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
> **NOTE**: Installation will be skipped if an installation is already done. To replace an existing installation first perform an `uninstall` as described below. **Make sure you do not uninstall something in use by other users.**

**To Do ONLY on the Front End Machine**

        make components_ssm_all SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}

        make components_install CONFIRM_INSTALL=yes \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG  SSM_BASE2=""


#### Post Install ####

**TODO**

  * update doc
  * send emails


Uninstall
---------

Un-installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
> **WARNING**: Uninstall cannot be reverted, make sure you do not uninstall something used by other users.

**To Do only on the Front End Machine**

        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make components_uninstall UNINSTALL_CONFIRM=yes \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG


Cleaning up
-----------

To remove all files created by the setup, compile and build process, use the `distclean` target.

        make distclean
        rm -rf ssmusedep*bndl .setenv.__extlib__.*.dot _testlog_py*/ \
               lib/${ORDENV_PLAT}/ bin/${ORDENV_PLAT}/

Misc
====

Patching the git repo
---------------------

Ref: https://www.devroom.io/2009/10/26/how-to-create-and-apply-a-patch-with-git/

Patch are produced with:

       BASETAG=   #Need to define from what tag (or hash) you want to produce patches
       git format-patch HEAD..${BASETAG}

Before applying the patch, you may check it with:

       git apply --stat PATCH.patch
       git apply --check PATCH.patch

Full apply

       git am --signoff PATCH.patch

Selective application, random list of commands

       git apply --reject PATH/TO/INCLUDE PATCH.patch
       git apply --reject --include PATH/TO/INCLUDE PATCH.patch
       git am    --include PATH/TO/INCLUDE  PATCH.patch
       git apply --exclude PATH/TO/EXCLUDE PATCH.patch
       git am    --exclude PATH/TO/EXCLUDE  PATCH.patch

Fixing apply/am problems

Ref: https://stackoverflow.com/questions/25846189/git-am-error-patch-does-not-apply
Ref: https://www.drupal.org/node/1129120

With --reject: 

  * inspect the reject
  * apply the patch manually (with an editor)
  * add file modified by the patch (git add...)
  * git am --continue
