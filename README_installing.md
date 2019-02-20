
Installing MIG/GEM from Version Control
=======================================

This document is intended as a notebook and checklist for the Model
Infrastructure Group (MIG) librarian. 
It is only valid on EC/CMC and GC/Science Networks.

> **This README assumes**:  
>
> The MIG repository is already cloned and the desired version is already checked out.  
> *See [Getting the code](README.md#getting-the-code) section for more details.*  
>
> The modifications are already committed and tested.  
> *See [README\_developing.md](README_developing.md) and 
> [README\_running.md](README_running.md) for more details.*
>
> The code is already compiled and built.  
> *See [README\_building.md](README_building.md) for more details.*


For more info see:

  * Main doc: [README.md](README.md)
  * Developing instructions: [README\_developing.md](README_developing.md)
  * Building instructions: [README\_building.md](README_building.md)
  * Running instructions: [README\_running.md](README_running.md)


**Table of Contents**

**TODO**


Quick Start
-----------

**TODO**: summary of commands
````bash
````


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
## Edit updater DB file for final adjustements.
emacs modelscripts/share/nml_updater/upd/gem_nml_update_db.txt
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

Moved to README_developping.md


1.3 - Update Versions and Dependencies
--------------------------------------

#### Versions ####

Every component has as "`VERSION`" file and there is one for MIG on the top level dir.
You need to update these version number for each component with modifications.

        emacs VERSION */VERSION

#### Dependencies ####


Partly moved to README_developping.md  
Need to add the **MIG's depdendencies** housekeeping by the librarian details in here


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

  * [GEM Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gem)
  * [GEMdyn Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gemdyn)
  * [RPNphy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN-Phy)
  * [RPNpy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN.py)
  * [RDE Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=rde)

1.6 - Remove a Component
------------------------

Moved to README_developping.md


1.7 - Add a Component
---------------------

Moved to README_developping.md


Installing MIG/GEM from Version Control
=======================================

This README assumes you already compiled and built the code you want to install.

Testing and installation must be performed in the same directory and same SHELL you used to compile and build MIG's component.

> Make sure compilation and building is done on all arch/platform you want the 
> installation to be done.

See [README\_building.md](README_building.md) for more details.


The following dir. are expected to exists for installation purpose only
(optional if you only wish to compile, build and run w/o installation).
Note that large files will be written to these dir.
````bash
~/SsmDepot/
~/SsmBundle/
````

2.3 - Testing
-------------

**TODO**:
See [README\_running.md](README_running.md) for more details.


2.4 - Install
-------------

Installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to install, default installs all components.
> **NOTE**: Installation of a component will be skipped if an installation is already done.  
> To replace an existing installation first perform an `uninstall` of the changed components 
> as described below. Make sure you do not uninstall something used by other users.

**To Do on All Arch**
````bash
# COMPONENTS=""
# export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
make ssmarch SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}  # COMPONENTS="${COMPONENTS}"
````

**To Do only on the Front End Machine**
````bash
        ## Note: `make ssmarch` on all arch must be done before
        # COMPONENTS=""
        # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
        make ssmall SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0}   # COMPONENTS="${COMPONENTS}"

        make components_install CONFIRM_INSTALL=yes \
            SSM_SKIP_INSTALLED=--skip-installed \
            SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
            # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG \
            # COMPONENTS="${COMPONENTS}"
````

#### Post Install ####

**To Do only on the Front End Machine**
````bash
        ## Final operation once all tests are ok
        VERSION="$(cat gem/VERSION)"
        if [[ ${SSM_TEST_INSTALL:-0} == 1 ]] ; then
           cd ~/SsmBundles/GEM/test/
           ln -s gem/${VERSION##*/}.bndl .
        else
           cd ~/SsmBundles/GEM/${VERSION%/*}
           ln -s gem/${VERSION##*/}.bndl .
        fi
````


2.5 - Update the documentation and announce
-------------------------------------------

Update documentation and change logs on the [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki):

  * [GEM wiki page](https://wiki.cmc.ec.gc.ca/wiki/Gem)
      * [GEM change log page](https://wiki.cmc.ec.gc.ca/wiki/GEM/Change_Log)
  * [RPNphy wiki page](https://wiki.cmc.ec.gc.ca/wiki/Rpnphy)
      * [RPNphy change log page](https://wiki.cmc.ec.gc.ca/wiki/RPNPhy/Change_Log)
  * [SCM wiki page](https://wiki.cmc.ec.gc.ca/wiki/SCM)
  * [Modelutils wiki page](https://wiki.cmc.ec.gc.ca/wiki/Modelutils)
  * [RPNpy wiki page](https://wiki.cmc.ec.gc.ca/wiki/Rpnpy)
  * [RDE wiki page](https://wiki.cmc.ec.gc.ca/wiki/RDE)

**TODO**: send out an email (gem, phy and python-rpn mailing lists)


2.6 - Uninstall
---------------

Uninstallation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to uninstall, default uninstalls all components.
> **WARNING**: Uninstall cannot be reverted, make sure you do not uninstall something used by other users.

**To Do only on the Front End Machine**
````bash
# COMPONENTS=""
# export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
make components_uninstall UNINSTALL_CONFIRM=yes \
    SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
    # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG \
    # COMPONENTS="${COMPONENTS}"
````


2.7 - Cleaning up
-----------------

To remove all files created by the setup, compile and build process, use the `distclean` target.
````bash
make distclean
rm -f */ssmusedep*bndl gem/ATM_MODEL_*
````


Misc
====

Patching the git repo
---------------------

Moved to README_developping.md


See Also
========

  * Main doc: [README.md](README.md).
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
*[GEM]: Global Environmental Multi-scale atmosperic model from RPN, ECCC  
*[MIG]: Model Infrastructure Group at RPN, ECCC  

*[SSM]: Simple Software Manager (a super simplified package manager for software at CMC/RPN, ECCC)  
*[RDE]: Research Development Environment, a super simple code dev. env. at RPN  
