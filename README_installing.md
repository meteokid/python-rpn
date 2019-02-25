
Installing MIG/GEM from Version Control
=======================================

This document is intended as a notebook and checklist for the Model
Infrastructure Group (MIG) librarian.
It is only valid on EC/CMC and GC/Science Networks.

> **This README assumes**:  
>
> The MIG repository is already cloned and the desired version is
> already checked out.  
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

   * [Updating the MIG Repository Content](#updating-the-mig-repository-content)
      * [1.1 - Update Code, Scripts and more](#11---update-code-scripts-and-more)
      * [1.2 - Update Compiler Options](#12---update-compiler-options)
      * [1.3 - Update Versions and Dependencies](#13---update-versions-and-dependencies)
      * [1.4 - Tag and Push Modifications](#14---tag-and-push-modifications)
      * [1.5 - Close Issues](#15---close-issues)
   * [Testing](#testing)
   * [Installing MIG/GEM from Version Control](#installing-miggem-from-version-control-1)
      * [Install](#install)
        * [Post Install](#post-install)
      * [Update the documentation and send announcement](#update-the-documentation-and-send-announcement)
      * [Uninstall](#uninstall)
      * [Cleaning up](#cleaning-up)
   * [See Also](#see-also)


Updating the MIG Repository Content
===================================

1.1 - Update Code, Scripts and more
-----------------------------------

  * **Merge in code from other devs** (and from other branches if any).  
    You may use
    * "`git merge`", "`git rebase`" for other's repos.  
      (see "*Merging: Pulling Others' Code*" section in
      [README\_developing.md](README_developing.md#merging-pulling-others-code)
    * "`git apply`", "`git am`" for patches  
      Special care needs to be taken for patches provided on top of a
      component's repository (not MIG).  
      (see "*Merging: Applying Patches*" section in
      [README\_developing.md](README_developing.md#merging-applying-patches)

  * **Check for code conformance**  
    **TODO** - need to set a list of things to check and maybe a "linter" script to report/fix issues (!$omp indentation, #include <arch_specific.hf>, Fortran 2004 std, ...).

  * **Test**  
    Before accepting new code from others or including your own modifications
    you may want to make sure it passes all tests.  
    See [testing section](#testing) below.


  * **Add Reference Namelists**  
    You'll need to have compiled binaries for this.  
    See [README\_building.md](README_building.md) for details.

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
export PREVIOUS_ATM_MODEL_VERSION=    ##Set the version to compare with
cat >> modelscripts/share/nml_updater/upd/gem_nml_update_db.txt << EOF
#------
fileVersion: ${PREVIOUS_ATM_MODEL_VERSION} > ${ATM_MODEL_VERSION}
$(diff modelscripts/share/nml_updater/ref/gem_settings.${PREVIOUS_ATM_MODEL_VERSION}.ref.k \
       modelscripts/share/nml_updater/ref/gem_settings.${ATM_MODEL_VERSION}.ref.k \
       | egrep '(>|<)' \
       | sed 's/=/ = /g' | sed 's/>/#New: /' | sed 's/</rm: /')
EOF
````

    ````bash
    ## Edit updater DB file for final adjustments.
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
    Namelists documentation produced below is added to the components'
    "`*/share/doc/`" directory.  
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

The compiler is specified as an external dependency in the
"`_migdep/DEPENDENCIES.external.*.bndl`" files.  
Compiler options, rules and targets are specified in each component's Makefiles (`*/include/Makefile.local*mk` files).  
For details, see:
  * [Dependencies Update](README_developing.md#dependencies-update)
  * [Compiler Options Update](README_developing.md#compiler-options-update)


1.3 - Update Versions and Dependencies
--------------------------------------

#### Versions ####

Every component has a "`VERSION`" file and there is an additional one
for MIG on the top level dir.
You need to update these version numbers for each components with modifications.

    emacs VERSION */VERSION


#### Dependencies ####

Updating these files is automated with the following script
(once the VERSION numbers have been updated)

    _bin/migdepupdater.ksh -v --gitdep --migver --check


*Note*: external dependencies *need to be maintained manually*. Most common
        external dependencies are specified in the
        "`_migdep/DEPENDENCIES.external.*.bndl`" files. Make sure they are
        consistent with other components, especially for duplicate items.

    emacs */DEPENDENCIES.external.*.bndl

For details, see:

  * [Dependencies Update](README_developing.md#dependencies-update)


1.4 - Tag and Push Modifications
--------------------------------

Pre-push check list

  * make sure you completed the above steps
    * [1.1 - Update Code, Scripts and more](#11---update-code-scripts-and-more)
    * [1.2 - Update Compiler Options](#12---update-compiler-options)
    * [1.3 - Update Versions and Dependencies](#13---update-versions-and-dependencies)

  * make sure everything if fully committed

        git status

  * make sure the code compile, build and passes tests *before* pushing it.  
    See [testing section](#testing) below.

Tag this MIG version

    migversion="$(cat VERSION)"
    git tag mig_${migversion##*/}


Push the MIG super-repos as described in the
[push upstream section](README_developing.md#push-upstream)

    ## Push to MIG's super-repos
    migbranch="$(git symbolic-ref --short HEAD)"
    git push origin ${migbranch} && git push --tags origin

    ## Check that branches and tags are as expected before doing the actual push below
    _bin/migsplitsubtree.ksh --dryrun

    ## Tag components and push
    _bin/migsplitsubtree.ksh --tag --push

    ## Cleanup repository garbage following remote import (fetch)
    _bin/migsplitsubtree.ksh --clean


1.5 - Close Issues
--------------------

Review, comment and close issues on MIG's CMC bugzilla products:

  * [GEM Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gem)
  * [GEMdyn Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=gemdyn)
  * [RPNphy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN-Phy)
  * [RPNpy Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=RPN.py)
  * [RDE Bugzilla](http://bugzilla.cmc.ec.gc.ca/buglist.cgi?cmdtype=runnamed&namedcmd=rde)


Testing
=======

This section assumes you already
[updated the MIG repository content](#updating-the-mig-repository-content) and
[compiled, built](README_building.md) the changes you want to install.

Testing must be performed in the same directory and same SHELL you used to compile and build MIG's component.

Testing should have been minimally done by developers before the code was
sent and merged in.  
See [README\_running.md](README_running.md) for more details.

Librarian should perform additional tests:

  * pre-installation canonical tests: **TODO**: (set of canonical cases, MPI/OMP conformance, ...
  * post-installation tests: **TODO**: (RI run interactive and batch, compile/build test, ...)


Installing MIG/GEM from Version Control
=======================================

This section assumes you already
[updated the MIG repository content](#updating-the-mig-repository-content),
[compiled, built](README_building.md) and [tested](#testing) the version
you want to install.

Testing and installation must be performed in the same directory and same SHELL you used to compile and build MIG's component.

> Make sure compilation and building is done on *all* arch/platform you want
> the installation to be done.

The following dir. are expected to exists for installation purpose only
(optional if you only wish to compile, build and run w/o installation).
Note that large files will be written to these dir.

    ~/SsmDepot/
    ~/SsmBundle/

> *Note*: Installation is performed by the following users:
> * `armnenv` on the EC/CMC network
> * `sgem000` on the GC/Science network


Install
-------

Installation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.  
You may specify the list of `COMPONENTS` to install, default installs all components.

> **Note**: Installation of a component will be skipped if an installation
> is already done.  
> To replace an existing installation first perform an `uninstall` of
> the changed components as described below. Make sure you do not uninstall
> something used by other users.

**To Do on All Arch**

    # COMPONENTS=""
    # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
    make ssmarch SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
         # COMPONENTS="${COMPONENTS}"

**To Do only on the Front End Machine**

    ## Note: `make ssmarch` on all arch must be done before
    # COMPONENTS=""
    # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
    make ssmall SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
         # COMPONENTS="${COMPONENTS}"

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

**To Do on All Arch**

Make sure the installed version compile, build and passes tests.  
See [testing section](#testing) below.


Update the documentation and send announcement
----------------------------------------------

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


Uninstall
---------

Uninstallation can only be performed by the librarian who has proper permissions.
This would be done on the EC/CMC and GC/Science networks.
You may specify the list of `COMPONENTS` to uninstall, default uninstalls all components.

> **WARNING**: Uninstall cannot be reverted, make sure you do not
> uninstall something used by other users.

**To Do only on the Front End Machine**

    # COMPONENTS=""
    # export SSM_TEST_INSTALL=1  ## Note: Set this to install under /tests/
    make components_uninstall UNINSTALL_CONFIRM=yes \
        SSM_TEST_INSTALL=${SSM_TEST_INSTALL:-0} \
        # SSM_BASE=/fs/ssm/eccc/mrd/rpn/MIG \
        # COMPONENTS="${COMPONENTS}"


Cleaning up
-----------

To remove all files created by the setup, compile and build process, use the `distclean` target.

    make distclean
    rm -f */ssmusedep*bndl gem/ATM_MODEL_*


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
*[GEM]: Global Environmental Multi-scale atmospheric model from RPN, ECCC  
*[MIG]: Model Infrastructure Group at RPN, ECCC  

*[SSM]: Simple Software Manager (a super simplified package manager for software at CMC/RPN, ECCC)  
*[RDE]: Research Development Environment, a super simple code dev. env. at RPN  
