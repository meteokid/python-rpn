
Building, Installing MIG/GEM from Version Control
=================================================

This document is intended as a notebook and checklist for the MIG librarian. It is only valid on EC/CMC and GC/Science Networks.

For more info on how to run and develop code see:
  * Main doc: [README.md](README.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 
  * Running instructions: [README\_running.md](README_running.md).
  
**Table of Contents**

  * **TODO**


Basic profile and directory setup
---------------------------------

**TODO**: 

  * list expected dir structure (~/SsmDepot, ~/SsmBundle, ...)
  * list expected env var and paths...


Closing Issues
--------------

**TODO**: review/close bugzilla issues


Getting the code
----------------

If not already done, you may clone the MIG repository and checkout the version you want to 
run/work on with the following command (example for version 5.0.0):

        MYVERSION=5.0.0                                   ## Obviously, you'll need to change this to the desired version
        MYURL=git@gitlab.science.gc.ca:MIG/mig            ## You'll need a GitLab.science account for this URL
        ## MYURL=https://gitlab.science.gc.ca/MIG/mig.git ## You cannot "git push" if you use the http:// URL
        git clone ${MYURL} mig_${MYVERSION}
        cd mig_${MYVERSION}
        git checkout -b mig_${MYVERSION}-${USER}-branch mig_${MYVERSION}


Update Dependencies
-------------------

**TODO**:

  * Where/How to set change external dependencies (compiler, librmn, vgrid, rpn_comm, ...)
  * Where/How to set compiler options, new compiler rules...


Committing the Code
-------------------

**TODO Pre-commit**:

  * merge in code from other devs (and from other branches if any)
  * update version numbers for each modified components
  * update bndl dependencies for each components
  * update _share/gemversions.txt
  * update DEPENDENCIES files if need be (should be automated with migsplitsubtree.ksh script)
  * Add ref nml and updater data
  * test (testing section below)

**TODO**:

  * tag mig version
  * commit code and tags to mono repos (git push && git push --tags)
  * tag individual components and push the tags (use _bin/migsplitsubtree.ksh --tag)
  * split into components and push (use _bin/migsplitsubtree.ksh --tag --push)


Initial Setup
-------------

Setting up the Shell Environment and Working Directories.

The MIG/GEM compiling, building and running systems depend on a few Shell (Bash is the only supported Shell to date) environment variables, PATHs, directories, links and files. 
The following commands perform that setup.

#### Shell Env. SetUp ####

*This step needs to be performed every time a new Shell is opened.*

*Note 1*: The setup files are available for the EC/CMC and GC/Sciences networks. 
Compiler and other external dependencies are available for these networks only.  
See [README\_developing.md](README_developing.md) on how to modify this.

*Note 2*: The compiled objects and binaries being big files they are put under the
build directory which is a link to a *scratch*/*big_data* space.  
The location of the *scratch*/*big_data* space is defined with the
`${storage_model}` environment variable.

#TODO: need to auto-swap cmc/science bndl link
# for item in gem gemdyn rpnphy modelutils modelscripts rpnpy ; do (cd $item ; for item2 in $(ls *-science) ; do set -x ; ln -sf $item2 ${item2%-science} ; set +x ; done) ; done

        export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
        ISOFFICIAL=--official  # For explicit use by librarian only
        . ./.setenv.dot ${ISOFFICIAL}

#### Files, Directories and Links SetUp ####

*This step needs to be performed only once for this working directory 
or after performing "`make distclean`".*

        ## Option 1: Compile/Build environment only
        ouv_exp_mig -v
        rdemklink -v   ## or ## linkit -v
        
        ## Option 2: Compile/Build environment and interactive/batch running environment
        . gemdev.dot myexp -v --gitlocal 


Compiling and Building
----------------------

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
        make abs  #-j ${MAKE_NPE:-6}


Testing
-------

**TODO**:


Install
-------

Installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to install, default installs all components.
> **NOTE**: Installation of a component will be skipped if an installation is already done. To replace an existing installation first perform an `uninstall` of the changed components as described below. Make sure you do not uninstall something used by other users.

**To Do on All Arch**

        # COMPONENTS=""
        make ssmarch  # COMPONENTS="${COMPONENTS}"

**To Do only on the Front End Machine**

        ## Note: `make ssmarch` on all arch must be done before
        # COMPONENTS=""
        make ssmall  # COMPONENTS="${COMPONENTS}"
        make components_install CONFIRM_INSTALL=yes  # COMPONENTS="${COMPONENTS}"  # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG


**Post-install fixes (Front End Machine)**

        VERSION="$(echo $(cat ${gem:-./gem}/include/Makefile.local*.mk | grep _VERSION0 | grep -v dir | cut -d= -f2))"
        VERSION_X=$(dirname ${VERSION})
        VERSION_V=${VERSION#*/}
        VERSION_S=
        echo :${VERSION_X}:${VERSION_V}:${VERSION_S}:

        cd ~/SsmBundles/GEM/tests
        cp ~/SsmBundles/GEM/${VERSION_X}/gem/${VERSION_V}${VERSION_S}.bndl ${VERSION_V}.bndl
        chmod u+w ${VERSION_V}.bndl
        if [[ "x${RDENETWORK}" == "xscience" ]] ; then SSMPREFIX=eccc/mrd/rpn/MIG/ ; fi
        echo ${SSMPREFIX}GEM/others/renametotest >> ${VERSION_V}.bndl


#TODO: for rpnpy need to update links in ~/SsmBundle/ENV/py/?.?/rpnpy/

        ## Final operation once all tests are ok
        cd ~/SsmBundles/GEM/${VERSION_X}
        ln -s gem/${VERSION_V}.bndl ${VERSION_V}.bndl


#### Post Install ####


**TODO**

  * update doc
  * send emails


Uninstall
---------

Uninstallation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to uninstall, default uninstalls all components.
> **WARNING**: Uninstall cannot be reverted, make sure you do not uninstall something used by other users.

**To Do only on the Front End Machine**

        # COMPONENTS=""
        make components_uninstall UNINSTALL_CONFIRM=yes  # COMPONENTS="${COMPONENTS}"


Cleaning up
-----------

To remove all files created by the setup, compile and build process, use the `distclean` target.

        make distclean

You may further clean up the GEM dir by removing all imported components with the following command.
> **WARNING**: To avoid loosing your modifications. make sure you created patches (saved elsewhere) or `git push` the modified components' code before removing all imported components.

**TODO**: See bin/clean-subtree.ksh


See Also
--------

  * Main doc: [README.md](README.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 
  * Running instructions: [README\_running.md](README_running.md).

