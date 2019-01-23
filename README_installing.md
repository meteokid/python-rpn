
Updating, Building, Installing MIG/GEM from Version Control
===========================================================

This document is intended as a notebook and checklist for the Model Infrastructure Group (MIG)
librarian. It is only valid on EC/CMC and GC/Science Networks.

For more info on how to run and develop code see:

  * Main doc: [README.md](README.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 
  * Running instructions: [README\_running.md](README_running.md).
  
**Table of Contents**

   * [Updating, Building, Installing MIG/GEM from Version Control](#updating-building-installing-miggem-from-version-control)
      * [Getting the code](#getting-the-code)
      
   * [Updating the MIG Repository Content](#updating-the-mig-repository-content)
      * [1.1 - Update Code, Scripts and more](#11---update-code-scripts-and-more)
      * [1.2 - Update Compiler Options](#12---update-compiler-options)
      * [1.3 - Update Versions and Dependencies](#13---update-versions-and-dependencies)
            * [Versions](#versions)
            * [Dependencies](#dependencies)
      * [1.4 - Tag and Push Modifications](#14---tag-and-push-modifications)
      * [1.5 - Close Issues](#15---close-issues)
      * [1.6 - Remove a Component](#16---remove-a-Component)
      * [1.7 - Add a Component](#17---add-a-Component)
      
   * [Building, Installing MIG/GEM from Version Control](#building-installing-miggem-from-version-control)
      * [2.0 - Basic profile and directory setup](#20---basic-profile-and-directory-setup)
      * [2.1 - Initial Setup](#21---initial-setup)
            * [Shell Env. SetUp](#shell-env-setup)
            * [Files, Directories and Links SetUp](#files-directories-and-links-setup)
      * [2.2 - Compiling and Building](#22---compiling-and-building)
      * [2.3 - Testing](#23---testing)
      * [2.4 - Install](#24---install)
            * [Post Install](#post-install)
      * [2.5 - Update the documentation and announce](#25---update-the-documentation-and-announce)
      * [2.6 - Uninstall](#26---uninstall)
      * [2.7 - Cleaning up](#27---cleaning-up)
      
   * [Misc](#misc)
      * [Patching the git repo](#patching-the-git-repo)
      
   * [See Also](#see-also)


Getting the code
----------------

If not already done, you may clone the MIG repository and checkout the version you want to 
run/work on with the following command (example for version 5.1.0):
````bash
        MYVERSION=5.1.0                                   ## Obviously, you'll need to change this to the desired version
        MYURL=git@gitlab.science.gc.ca:MIG/mig            ## You'll need a GitLab.science account for this URL
        # MYURL=https://gitlab.science.gc.ca/MIG/mig.git  ## You cannot "git push" if you use the http:// URL
        
        MYTAG=mig_${MYVERSION}
        git clone ${MYURL} ${MYTAG}
        cd ${MYTAG}
        
        ## Check if ${MYTAG} exists
        taglist=":$(git tag -l | tr '\n' ':')"
        if [[ "x$(echo ${taglist} | grep :${MYTAG}:)" == "x" ]] ; then
            echo "===> ERROR: not such tag: ${MYTAG} <==="
        fi
        
        ## There are 2 options (existing or new branch):
        
        ## Option 1: Continue on existing branch - ${MYTAG} is the HEAD of its branch
        MYBRANCH=${MYTAG%.*}-branch
        git checkout ${MYBRANCH}

        ## Option 2: Develop on a new branch - ${MYTAG} is NOT the HEAD of any branch
        # MYBRANCH=${MYTAG}-${USER}-branch
        # git checkout -b ${MYBRANCH} ${MYTAG}
````

**TODO**: automate existing or new branch selection


Updating the MIG Repository Content
===================================

1.1 - Update Code, Scripts and more
-----------------------------------

  * **Merge in code from other devs** (and from other branches if any). You may use
    * "`git merge`", "`git rebase`" for other's repos or 
    * "`git apply`", "`git am`" for patches  
      Special care needs to be taken for patches provided on top of a 
      component's repository (not MIG).  
      (see [Patching the git repo section below](#patching-the-git-repo))
      
  * **Test**  
    Before accepting new code for others or including your own modifications   
    you may want to make sure it passes all tests  
    (See [Testing section below](#testing))


  * **Add Reference Namelists**
    You'll need to have compiled binaries for this, see:
    * [2.1 - Initial Setup](#21---initial-setup)
    * [2.2 - Compiling and Building](#22---compiling-and-building)  
````bash
        export ATM_MODEL_VERSION="$(cat gem/VERSION)"
        export ATM_MODEL_VERSION=${ATM_MODEL_VERSION#*/}
        export gem_version=${ATM_MODEL_VERSION}
        gem_nml_mkref gem_settings.${ATM_MODEL_VERSION}.ref
        rpy.nml_get -v -f gem_settings.${ATM_MODEL_VERSION}.ref \
            > gem_settings.${ATM_MODEL_VERSION}.ref.kv
        cat gem_settings.${ATM_MODEL_VERSION}.ref.kv | cut -d= -f1 \
            > gem_settings.${ATM_MODEL_VERSION}.ref.k
        mv gem_settings.${ATM_MODEL_VERSION}.ref* modelscripts/share/nml_updater/ref/
        git add modelscripts/share/nml_updater/ref/gem_settings.${ATM_MODEL_VERSION}.ref*
````

  * **Add Namelists updater Data**
````bash
export PREVIOUS_ATM_MODEL_VERSION=**TODO**  ##Set the version to compare with
cat >> modelscripts/share/nml_updater/upd/gem_nml_update_db.txt << EOF
#------
fileVersion: ${PREVIOUS_ATM_MODEL_VERSION} > ${ATM_MODEL_VERSION}
$(diff modelscripts/share/nml_updater/ref/gem_settings.${PREVIOUS_ATM_MODEL_VERSION}.ref.k \
       modelscripts/share/nml_updater/ref/gem_settings.${ATM_MODEL_VERSION}.ref.k \
       | egrep '(>|<)' \
       | sed 's/=/ = /g' | sed 's/>/#New: /' | sed 's/</rm: /') 
EOF
vi modelscripts/share/nml_updater/upd/gem_nml_update_db.txt
````

  * **Update Sample Configs and RI Settings**
````bash
        for item in $(find gem/share/cfgs modelscripts/share/gem-ref -name gem_settings.nml) ; do
            gem_nml_update -v \
                --from ${PREVIOUS_ATM_MODEL_VERSION} \
                --to ${ATM_MODEL_VERSION} \
                -i ${item} -o ${item} \
                --config=modelscripts/share/nml_updater/upd/gem_nml_update_db.txt
        done
````

  * **Update Namelists Documentation**  
    Namelists documentation produced below is added to the components' "`*/share/doc/`" directory.  
    It also needs to be updated on the
    [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki/Gem).
    Cut and past content from the "`*/share/doc/*.wiki`" files.
````bash
        ftnnml2wiki --sort --wiki
        ftnnml2wiki --sort --md
        for item in $(ls -1 *.wiki 2> /dev/null) ; do
            mkdir -p ${item%.*}/share/doc
            mv ${item%.*}.{md,wiki} ${item%%.*}/share/doc/
        done
````


1.2 - Update Compiler Options
-----------------------------

The compile/build system is based on 2 wrappers on top of the compiler:

  * RPN/CMC *Compiler Rules* used with their own "`s.compile`" script.  
    You can visualize these with (after the [Initial Setup](#21---initial-setup)):  

        cat $(s.get_compiler_rules)

    *Note that the "`s.get_compiler_rules`" script comes from SSC and  
    is not part of the MIG super repos.*
    
  * RDE Makefiles and scripts, see the *rde* component for details.  
    * RDE defines basics Makefile vars, rules and targets.  
      File: `rde/include/Makefile*`
    * every components can add specific Makefile vars, rules and targets.  
      File: `*/include/Makefile.local*.mk`
    * MIG can have specific Makefile vars, rules and targets.  
      File: `Makefile.user*.mk`
  
Compiler options, rules and targets can thus be modified in:

  * `rde/include/Makefile*`: for system-wide options
  * `*/include/Makefile*`: for components specific options
  * `Makefile.user*.mk`: for full MIG build system specific options
  
**TODO**:

  * List expected (by the build/compile system) components' Makefile targets and vars


1.3 - Update Versions and Dependencies
--------------------------------------

#### Versions ####

Every component has as "`VERSION`" file and there is one for MIG on the top level dir.
You need to update these version number for each component with modifications.

        emacs VERSION */VERSION

#### Dependencies ####

MIG has several types of dependency files:

  * **MIG's depdendencies**
    * Present version Git dependencies: "`DEPENDENCIES`" file  
      This file is used by the "`_bin/migsplitsubtree.ksh`" script.  
      This file format is one line per component:  
      `NAME=URL/TAG`
    * All MIG versions Git dependencies: "`_share/migversions.txt`" file  
      This file is used by the "`_bin/migimportall.ksh`" script.  
      This file format is one line per version:  
      `branch=MIGBRANCH; tag=MIGTAG ; components="COMP1NAME/COMP1TAG COMP2NAME/COMP2TAG ..."`
  * **Components' dependencies**
    * local, MIG, dependencies; components depend on each other.  
      These are specified in component's "`DEPENDENCIES.mig.bndl`" files  
      This file has the "`r.load.dot`" *bundle* format.
    * external dependencies, those that are not part of the MIG super repos.  
      There are 2 such files, one for ECCMC and Science networks:  
      "`DEPENDENCIES.external.cmc.bndl`" and "`DEPENDENCIES.external.science.bndl`"  
      This file has the "`r.load.dot`" *bundle* format.

Updating these files is automated with the following script 
(once the VERSION numbers have been updated)

        _bin/migdepupdater.ksh -v --gitdep --migver --check


*Note1*: external dependencies *need to be maintained manually*.  
         Make sure they are consistent with other components, especially for duplicate items.

        emacs */DEPENDENCIES.external.*.bndl

*Note2*: **TODO**: short description of the  "`r.load.dot`" *bundle* format


1.4 - Tag and Push Modifications
--------------------------------

Pre-push check list

  * make sure you completed the above steps
    * [1.1 - Update Code, Scripts and more](#11---update-code-scripts-and-more)
    * [1.2 - Update Compiler Options](#12---update-compiler-options)
    * [1.3 - Update Versions and Dependencies](#13---update-versions-and-dependencies)

  * make sure everything if fully committed

        git status

  * make sure the code compile, build and passes tests *before* pushing it
    * [2.1 - Initial Setup](#21---initial-setup)
    * [2.2 - Compiling and Building](#22---compiling-and-building)
    * [2.3 - Testing](#23---testing)

Tag this MIG version
````bash
        migversion="$(cat VERSION)"
        git tag mig_${migversion##*/}
````
Push the MIG super-repos
````bash
        migbranch="$(git symbolic-ref --short HEAD)"
        git push origin ${migbranch} && git push --tags origin
````
Tag and push individual components
````bash
        _bin/migsplitsubtree.ksh --dryrun
        ## Check that branches and tags are as expected before doing the actual push below
        
        _bin/migsplitsubtree.ksh --tag --push
````
Clean repository garbage
````bash
        _bin/migsplitsubtree.ksh --clean
````

1.5 - Close Issues
--------------------

Review, comment and close issues on MIG's CMC bugzilla products:

* http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gem
* http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gemdyn
* http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN-Phy
* http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN.py
* http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=rde


1.6 - Remove a Component
------------------------

Removing a component is as simple as removing its directory and other  
components' dependency to it.
````bash
        compname=MYNAME                      ## Set to appropriate name
        cat DEPENDENCIES | egrep -v "^${compname}="  1>  ${compname}.dep.$$
        mv ${compname}.dep.$$ DEPENDENCIES
        for item in $(ls -1 */DEPENDENCIES.mig.bndl) ; do
            cat ${item} | grep -v "/${compname}/"  1>  ${compname}.dep.$$
            mv ${compname}.dep.$$ ${item}
        done
````

Before doing anything else, it may be best to re-load the environment in a new SHELL
to make sure the removed component is removed from the SHELL env as well.  
See [Initial Setup](#21---initial-setup) below.

As for any other modifications you'll need to run the tests 
(see [testing](#23---testing) section below) then commit and push your code
(see [Tag and Push Modifications](#14---tag-and-push-modifications) section above).


1.7 - Add a Component
---------------------

Import the component from a remote Git repository using "`git subtree`":
````bash
        compname=MYNAME                      ## Set to appropriate name
        compversion=MYVERSION                ## Set to appropriate version
        comptag=${compname}_${compversion}   ## Set to appropriate tag if need be
        compurl=git@gitlab.science.gc.ca:MIG/${compname}.git
        git remote add ${compname} ${compurl}
        git fetch --tags ${compname}
        git subtree add -P ${compname} --squash ${compname} ${comptag} \
            -m "subtree_pull: tag=${comptag}; url=${compurl}; dir=${compname}"
        echo "${compname}=${compurl}/${comptag}" >> DEPENDENCIES
        for tag in $(git ls-remote --tags ${compname} | awk '{print $2}' | tr '\n' ' ') ; do
           git tag -d ${tag##*/}
        done
        git remote rm ${compname}
````

Make sure the component has the needed required directories, files and content for MIG  
integration/build system (**TODO**: see...)

Before doing anything else, it may be best to re-load the environment in a new SHELL
to make sure the new component is properly setup.  
See [Initial Setup](#21---initial-setup) below.

As for any other modifications you'll need to run the tests 
(see [testing](#23---testing) section below) then commit and push your code
(see [Tag and Push Modifications](#14---tag-and-push-modifications) section above).


Building, Installing MIG/GEM from Version Control
=================================================

2.0 - Basic profile and directory setup
---------------------------------------

The build system is working on the EC/CMC and GC/Science network.  
Your account needs to have an up to date *.profile* environment setup.  
See this [SSC Getting Started doc](https://portal.science.gc.ca/confluence/display/SCIDOCS/Getting+Started+-+Setting+Up+the+Environment)

A scratch dir (big temporary space) is needed to host compilation,
building and running produced files. 
This scratch dir is found by the system with the `${storage_model}` var.

        export storage_model=/PATH/TO/SCRATCH/DIR/

The following dir. are expected to exists for installation purpose only
(optional if you only wish to compile, build and run w/o installation).
Note that large files will be written to these dir.

        ~/SsmDepot/
        ~/SsmBundle/
        
**TODO**: 

  * other expected dir
  * other expected env vars


2.1 - Initial Setup
-------------------

Setting up the Shell Environment and Working Directories.

The MIG/GEM compiling, building and running systems depend on a few Shell 
(Bash is the only supported Shell to date) environment variables, PATHs, 
directories, links and files. 
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

        export storage_model=${storage_model:-/PATH/TO/SCRATCH/DIR/}
        ISOFFICIAL=--official  # For explicit use by librarian only
        . ./.setenv.dot -v ${ISOFFICIAL}


#### Files, Directories and Links SetUp ####

*This step needs to be performed only once for this working directory 
or after performing "`make distclean`".*

        ## Option 1: Compile/Build environment only
        ouv_exp_mig -v
        rdemklink -v
        
        ## Option 2: Compile/Build environment and interactive/batch running environment
        . gemdev.dot myexp -v --gitlocal 


2.2 - Compiling and Building
----------------------------

*Note 1*: If you're planning on running in batch mode, submitting to another machine,
make sure you do the *initial setup*, including compilation,
on the machine where the job will be submitted.

*Note 2*: The compiler is specified as an external dependency in 
the "`_migdep/DEPENDENCIES.external.*.bndl`" files.
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


2.3 - Testing
-------------

**TODO**:


2.4 - Install
-------------

Installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to install, default installs all components.
> **NOTE**: Installation of a component will be skipped if an installation is already done.  
> To replace an existing installation first perform an `uninstall` of the changed components 
> as described below. Make sure you do not uninstall something used by other users.

**To Do on All Arch**

        # COMPONENTS=""
        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make ssmarch SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}  # COMPONENTS="${COMPONENTS}"

**To Do only on the Front End Machine**

        ## Note: `make ssmarch` on all arch must be done before
        # COMPONENTS=""
        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make ssmall SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}   # COMPONENTS="${COMPONENTS}"
        
        make components_install CONFIRM_INSTALL=yes \
            SSM_SKIP_INSTALLED=--skip-installed \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG \
            # COMPONENTS="${COMPONENTS}"

#### Post Install ####

**To Do only on the Front End Machine**

        ## Final operation once all tests are ok
        VERSION="$(cat gem/VERSION)"
        if [[ ${SSM_TEST_INSTALL:-0} == 1 ]] ; then
           cd ~/SsmBundles/GEM/test/
           ln -s gem/${VERSION##*/}.bndl .
        else
           cd ~/SsmBundles/GEM/${VERSION%/*}
           ln -s gem/${VERSION##*/}.bndl .
        fi


2.5 - Update the documentation and announce
-------------------------------------------

Update documentation and change logs on the CMC wiki:

* https://wiki.cmc.ec.gc.ca/wiki/Gem
    * https://wiki.cmc.ec.gc.ca/wiki/GEM/Change_Log
* https://wiki.cmc.ec.gc.ca/wiki/Rpnphy
    * https://wiki.cmc.ec.gc.ca/wiki/RPNPhy/Change_Log
* https://wiki.cmc.ec.gc.ca/wiki/SCM
* https://wiki.cmc.ec.gc.ca/wiki/Modelutils
* https://wiki.cmc.ec.gc.ca/wiki/Rpnpy
* https://wiki.cmc.ec.gc.ca/wiki/RDE

**TODO**: send out an email (gem, phy and python-rpn mailing lists)


2.6 - Uninstall
---------------

Uninstallation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to uninstall, default uninstalls all components.
> **WARNING**: Uninstall cannot be reverted, make sure you do not uninstall something used by other users.

**To Do only on the Front End Machine**

        # COMPONENTS=""
        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make components_uninstall UNINSTALL_CONFIRM=yes \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG \
            # COMPONENTS="${COMPONENTS}"


2.7 - Cleaning up
-----------------

To remove all files created by the setup, compile and build process, use the `distclean` target.

        make distclean
        rm -f */ssmusedep*bndl gem/ATM_MODEL_*


Misc
====

Patching the git repo
---------------------

Ref: https://www.devroom.io/2009/10/26/how-to-create-and-apply-a-patch-with-git/

Patch are produced with:

       BASETAG=   #Need to define from what tag (or hash) you want to produce patches
       git format-patch HEAD..${BASETAG}

If your patch is to be applied on a sub component (sub directory),  
then set MYDIR to its name (example for gemdyn below).

        MYDIR="--directory=gemdyn"

Define the PATH/NAME of the patch file:

        MYPATH=/PATH/TO/${MYPATCH}

Before applying the patch, you may check it with:

       git apply --stat ${MYPATCH}
       git apply --check ${MYDIR} ${MYPATCH}

Full apply

       git am --signoff ${MYDIR} ${MYPATCH}

Selective application, random list of commands

       git apply --reject PATH/TO/INCLUDE   ${MYDIR} ${MYPATCH}
       git apply --reject --include PATH/TO/INCLUDE  ${MYDIR}  ${MYPATCH}
       git am    --include PATH/TO/INCLUDE  ${MYDIR} ${MYPATCH}
       git apply --exclude PATH/TO/EXCLUDE  ${MYDIR} ${MYPATCH}
       git am    --exclude PATH/TO/EXCLUDE  ${MYDIR} ${MYPATCH}

Fixing apply/am problems

Ref: https://stackoverflow.com/questions/25846189/git-am-error-patch-does-not-apply
Ref: https://www.drupal.org/node/1129120

With --reject: 

  * inspect the reject
  * apply the patch manually (with an editor)
  * add file modified by the patch (git add...)
  * git am --continue


See Also
========

  * Main doc: [README.md](README.md).
  * Developing instructions: [README\_developing.md](README_developing.md). 
  * Running instructions: [README\_running.md](README_running.md).



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

