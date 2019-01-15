
Running MIG/GEM from Version Control
====================================

This document describes the basic steps required to run MIG/GEM using Code from Version Control. 
It includes instructions to import the code, do the initial setup, 
compile/build the binaries and then run with your configuration.

To modify the code and scripts, please refer to the [Initial Setup](#initial-setup) below and to [README\_developing.md](README_developing.md).

**Table of Contents**

  * [Initial Setup](#initial-setup)
      * [0. Cloning and checkout](#0-cloning-and-checkout)
      * [1. Setting up the Shell Environment and Working Directories](#1-setting-up-the-shell-environment-and-working-directories)
      * [2. Compile and Build](#2-compile-and-build)
  * [Configuration](#configuration)
  * [Running](#running)
      * [Running Interactively](#running-interactively)
      * [Running in Batch mode](#running-in-batch-mode)
  * [Cleaning Up](#cleaning-up)
  * [See Also](#see-also)


Initial Setup
-------------

### 0. Cloning and checkout ###

If did not already clone the MIG repository and checkout the version you want to run/work on, 
see [Getting the code](README.md#getting-the-code) section in [README.md](README.md).        


### 1. Setting up the Shell Environment and Working Directories ###

The MIG/GEM compiling, building and running systems depend on a few Shell (Bash is the only supported Shell to date) environment variables, PATHs, directories, links and files. 
The following commands perform that setup.

**1.1 Shell Env. SetUp**

*This step needs to be performed every time a new Shell is opened.*

*Note 1*: The setup files are available for the EC/CMC and GC/Sciences networks. 
Compiler and other external dependencies are available for these networks only.  
See [README\_developing.md](README_developing.md) on how to modify this.

*Note 2*: The compiled objects and binaries being big files they are put under the
*build* directory which is a link to a *scratch*/*big_data* space.  
The location of the *scratch*/*big_data* space is defined with the
`${storage_model}` environment variable.

        export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
        . ./.setenv.dot

**1.2 Files, Directories and Links SetUp**

*This step needs to be performed only once for this working directory 
or after performing "`make distclean`".  This must be done after the Shell Env. Setup step above.*
       
        . gemdev.dot myexp -v --gitlocal 


### 2. Compile and Build ###

*Note 1*: If you're planning on running in batch mode, submitting to another machine,
make sure you do the *initial setup*, including compilation,
on the machine where the job will be submitted.

*Note 2*: The compiler is specified in `modelutils`'s dependencies.
Compiler options, rules and targets are specified in each components's
Makefiles (`*/include/Makefile.local*mk` files).  
See [README\_developing.md](README_developing.md) for more details.

Initially it is best to make sure to start with a clean slate.  
*This should be done once initially and every time the *dependency* list is modified.*

        make buildclean

Use the following Makefile targets to compile the build libs and abs.  
*This needs to be done initially and every time the code is modified.*

        make dep
        make vfiles
        make libs -j ${MAKE_NPE:-6}
        make abs  -j ${MAKE_NPE:-6}


Configuration
-------------

Configuration files are in the `GEM_cfgs/cfg_0000/` dir.  
**TODO:** 

You may list all sample configurations with:

        devadd --list

You can then import a sample configuration with:

        devadd --ln --copy CONFIG_NAME

Use "`devadd --help`" for a list of all options.


Running
-------

### Running Interactively ###

        gemrun CONFIG_NAME --ptopo ${NPEX:-1}x${NPEY:-1}x${OMP_NUM_THREADS:-1}

  * Results are in **TODO**
  * log/listing files are in **TODO**

Use "`gemrun --help`" for a list of all options.


### Running in Batch mode ###

*Note:* Running in batch mode is done using the maestro sequencer.
Maestro needs to be pre-installed and configured on the machine and in your user account.  
This setup is beyond the scope of this doc... (**TODO**: reference to external doc)

        # export SEQ_EXP_HOME=???? #**TODO**
        gemlaunch ...
        gemview -exp base

  * Results are in TODO
  * log/listing files are in TODO


Cleaning Up
-----------

You can remove the files, dir and links created by the Setup process with the following.

        make distclean
        rm -rf GEM_cfg*

After this, in order to be able to compile and run again, you'll need to re-do the whole setup process.


See Also
--------

  * Main doc: [README.md](README.md).
  * Developing instructions: [README\_developing.md](README_developing.md).
  * Installing instructions: [README\_installing.md](README_installing.md). 


