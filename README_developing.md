
Developing MIG/GEM from Version Control
=======================================

This document describes the basic steps required to modify MIG/GEM
source code using Version Control.



> **TODO**: For more information on the system and its rationale see [README.md](README.md) and [Guidelines](http://wiki.cccma.ec.gc.ca/~acrnncs/CanESM5_development_guide.pdf).
> Feel free to provide feedback on the guidelines and rules.

The version control system is based on git. If you know nothing about git you should educate yourself first.
These resources may help:

  * [official git documentation](https://git-scm.com/doc)
  * [Atlassian git tutorials](https://www.atlassian.com/git/tutorials)
  * [Git subtree docs](https://github.com/git/git/blob/master/contrib/subtree/git-subtree.txt)
  * [CCCma NEMO version control help page](http://wiki.cccma.ec.gc.ca/twiki/bin/view/Main/NemoVersionControl)

Specifically, you should be comfortable with the git definitions of: "commit", "tag", "branch", "checkout", "staging", "reset", "merge".



> **This README assumes**:  
>
> The MIG repository is already cloned and the desired version is already
> checked out.  
> *See [Getting the code](README.md#getting-the-code) section for more details.*  

**Table of Contents**

   * [Layout](#layout)
      * [Detailed description](#detailed-description)
   * [Compiling, Building and Testing](#compiling-building-and-testing)
   * [Making Modifications](#making-modifications)
      * [Compile and Build](#compile-and-build)
      * [Test](#test)
      * [Commit and Tags](#commit-and-tags)
      * [Modifications: Specific Cases](#modifications-specific-cases)
         * [Shell Env. SetUp Update](#shell-env-setup-update)
         * [Dependencies Update](#dependencies-update)
         * [Compiler Options Update](#compiler-options-update)
         * [Remove a Component](#remove-a-component)
         * [Add a Component](#add-a-component)
         * [Getting Others' Modifications](#getting-others-modifications)
            * [Merging: Applying Patches](#merging-applying-patches)
            * [Merging: Pulling Others' Code](#merging-pulling-others-code)
         * [Revert Changes](#revert-changes)
         * [Branches, Parallel Development](#branches-parallel-development)
         * [Cleaning up](#cleaning-up)
      * [Contribute Back, Share](#contribute-back-share)
         * [Send Pull Request](#send-pull-request)
         * [Create Patches](#create-patches)
         * [Push Upstream](#push-upstream)
   * [See Also](#see-also)
      * [Abbreviations](#abbreviations)


Layout
======

This MIG repository is referred to as the *super-repo* (a *mono-repos* including all MIG products, all sub-repositories; called components).

The main MIG directory has the following files and directories:
  * `.gitignore `: List of path patterns (one per line) ignored by `git`
  * `.setenv.dot`: script called in the
    [Initial Setup](README_building.md#initial-setup) process to establish
    the Shell environment
  * `DEPENDENCIES`: list the name and origin of each components
  * `Makefile.user.mk`: MIG main targets for `gmake` (used in the *build-dir*)
  * `Makefile.user.root.mk`: MIG main targets for `gmake` (used in the *root-dir*)
  * `README*.md`: MIG's instructions files
  * `VERSION   `: MIG's version number
  * `_bin/     `: MIG's repository helper scripts
  * `_share/   `: some MIG misc. data files for the helper scripts
  * `_migdep/  `: dummy MIG component used to define common external dependencies to be loaded.
  * All other subdirectories are MIG's components, as described by
    the `DEPENDENCIES` file

Each component is in its own sub-directory and is imported using the
`git subtree` commands. They are considered independent repositories and
thus can be maintained and built independently. These
components/sub-directories need to contain the following mandatory
sub-directories and files:

  * `*/.name      `: Component's name, equivalent to the sub dir they are
    imported in. This file is used by some `rde` scripts.
  * `*/.setenv.dot`: script called in the
    [Initial Setup](README_building.md#initial-setup) process to establish
    the Shell environment
  * `*/DESCRIPTION`: Component's description
  * `*/VERSION    `: Component's version number
  * `*/.ssm.d/    `: SSM mandatory files
  * `*/include/   `: used for included files and sub-Makefiles, will be added to the `INCLUDE_PATH`
  * `*/include/Makefile.local.NAME.mk`: Component's targets for `gmake`
  * `*/include/Makefile.ssm.mk`: Component's targets for `gmake` used in the
    install process

Additionally these components/sub-directories may contain the following
optional sub-directories and files:

  * `*/DEPENDENCIES.mig.bndl`: Component's list of dependencies on other MIG components
  * `*/DEPENDENCIES.external.*.bndl`: Component's list of *external* (non MIG)
    dependencies, in addition to the ones listed in `_migdep/`
  * `*/bin/       `: Component's specific scripts. Could be Shell, python or else.
    For python modules or similar, it is best to put them under the
    `lib/python/` dir. Will be added to the `PATH`
  * `*/bin/.env_setup.dot`: used to define some Shell env var.
    It may be called in the [Initial Setup](README_building.md#initial-setup)
    process and, post-installation, in the SSM load process.
  * `*/lib/       `: Python, R, and other libraries or modules should be
    placed under the `lib/` dir.  
    For python modules or similar, it is best to put them under the
    `lib/python/` dir.  
    Will be added to the `LIBRARY_PATH`
  * `*/lib/python/`: used for python modules, will be added to the `PYTHONPATH`
  * `*/src/       `: Directory and sub-directories for Fortran or C source files.
    The dependency analyzer will create a list of objects files separated for
    each very `src/` sub-directories and these can thus be considered as
    individual libraries. Will be added to the `gmake` source path (`VPATH`).
  * `*/share/     `: Most of the other things exported by a component that
    is not source code, lib or scripts should be put under the `share/` dir.
  * `*/.restricted`: Directories including this file will not be added to
    the `gmake` source path (`VPATH`) unless `${ORDENV_PLAT}` match one of the
    lines in this file (whitelist).

Detailed description
--------------------

### .setenv.dot ###

`.setenv.dot` script (bash Shell script) is *sourced* in the 
[Initial Setup](README_building.md#initial-setup) process to establish the Shell 
environment. It must:

  * support `--external` argument to load only *external* dependencies (outside MIG)
  * load needed *dependencies*
  * set PATH, LIBPATH, PYTHONPATH
  * set other needed Shell environment vars.

### VERSION ###

To keep track of what is what and from where it comes from, a naming convention for version numbers is useful.

> See [README\_version\_convention.md](README_version_convention.md) for details.


### DEPENDENCIES ###

List the name and origin of each components, its URL and tag.

  * each line in `DEPENDENCIES` has the format (lowercase): `name=url/tag`
  * tags have the format `name_version` in lowercase

> *Note*: Components' version in this file can be automatically updated
> (after the [Initial Setup](README_building.md#initial-setup) process) with
> a helper script:  
> `_bin/migdepupdater.ksh -v --gitdep --check`

> See also [README\_version\_convention.md](README_version_convention.md) for details on components pathd (`d/`, `x/`, `tests/`).


### DEPENDENCIES.mig.bndl ###

Component's list of dependencies on other MIG components.

This file has the *bundle* format recognized by the `r.load.dot` RPN utility.

  * one component per line
  * relative or absolute path to the installed component's specific version

> *Note*: Components' version in these files can be automatically updated
> (after the [Initial Setup](README_building.md#initial-setup) process) with
> a helper script:  
> `_bin/migdepupdater.ksh -v --gitdep --check`

> See: [r.load.dot CMC wiki page](https://wiki.cmc.ec.gc.ca/wiki/R.load.dot)
> for details on RPN's `r.load.dot` *bundle* format.

### DEPENDENCIES.external.*.bndl ###

Component's list of external dependencies in addition to the ones listed
in `_migdep/`.

This file has the *bundle* format recognized by the `r.load.dot` RPN utility.

  * one component per line
  * relative or absolute path to the installed component's specific version

> *Note 1*: The file name must match `DEPENDENCIES.external.${RDENETWORK}.bndl`,
> where `${RDENETWORK}` should have been defined in the
> [Initial Setup](README_building.md#initial-setup) process.  
> Known `${RDENETWORK}` values: "cmc" or "science".

> *Note 2*: *external* dependencies must be defined and updated for *every*
> known `${RDENETWORK}`: "cmc" and "science".


### _migdep ###

Dummy MIG component used to define common *external* dependencies to be loaded.

These *external* dependencies are listed in the `DEPENDENCIES.external.*.bndl`
files. See above for the file format details.


### include/ ###

Fortran or C source files to be included.
Place here the files that can be included from other components.
Preferably put file that should not be included from other components in
the `src/` dir. Components' specific Makefiles variables, rules and targets
should be specified in the `include/Makefile.local*mk` file.


### include/Makefile.local.NAME.mk ###

Component's targets for `gmake` used in the compile and build process

> *Note*: all these components' Makefiles will be merged/included into
> the main one, please make sure modifications to them does not have
> undesired side effect on other components.  
> It is a good idea to use uniq prefix on vars and targets.

*Expected vars and targets*

  * `NAME_VERSION`: Version number including, normally read from the
    components's `VERSION` file for consistency
  * `NAME_VFILES` : list of version files creation targets
  * `NAME_LIBS_ALL_FILES_PLUS`: list of libs creation targets
  * `NAME_ABS`: list of abs creation targets
  * `NAME_ABS_FILES`: list of created abs file names, full path

> `NAME` in the above var names is to be replaced by the component's name
> (upper case)

Other intersting vars and targets that you may want to use in the
`Makefile.local.*.mk` file are defined in the `Makefile.dep.${ARCH}.mk` file
generated by `make dep`.
It contains auto-built source dependencies for Fortran and C source code and
exports a number of symbols that may be used in the component's
`Makefile.local.*.mk` file.
Among the most interesting ones:

  * `TOPDIRLIST_name`: list of top directories, corresponds to list of
    component's full paths
  * `SUBDIRLIST_name`: list of component's source sub directories
  * `OBJECTS`: list of objects to be created, for all components
  * `OBJECTS_name`: list of component's  objects to be created
  * `OBJECTS_name_subname`: list of component's  objects to be created
     from a specific sub directory
  * `FORTRAN_MODULES`: list of Fortran modules to be created, for all components
  * `FORTRAN_MODULES_name`: list of component's Fortran modules to be created

> `name` and `NAME` in the above var names is to be replaced by the component's name
> (lower and upper case respectively)

Finally, RDE's `Makefiles.*.mk` exports many other symboles and targets used
in the build process.

> **TODO**: other details about RDE, ouv_exp_gem/srcpath, linkit/builddir/.rde.config.dot, rules/targets, .rde.setenv.dot, ...


### include/Makefile.ssm.mk ###

Component's targets for `gmake` used in the install process.

> *Note*: all these components' Makefiles will be merged/included into the
> main one, please make sure modifications to them does not have undesired
> side effect on other components.  
> It is a good idea to use uniq prefix on vars and targets.

*Expected vars and targets*

  * `NAME_SSMALL_FILES`: list of ssm package, type all, creation targets
  * `NAME_SSMARCH_FILES`: list of ssm package, arch specific, creation targets
  * `NAME_INSTALL`: list of install targets
  * `NAME_UNINSTALL`: list of uninstall targets

> `NAME` in the above var names is to be replaced by the component's name
> (upper case)


### .restricted ###

This file is used by the automatic dependencies generator called by `make dep`.
It is a text file with white-list of arch (one per line) on which the
dependency analyzer can can operate.

  * No `.restricted` file means the dir will be analysed for dependencies.
  * An empty `.restricted` file cause, on all arch, the directory containing it to be ignored by the dependencies analyzer.
  * A not empty `.restricted` file cause, on arch NOT matching one the white-list
    arch (`${ORDENV_PLAT}`), the directory containing it to be ignored by the
    dependencies analyzer.


Compiling, Building and Testing
===============================

Please follow the

  * [Initial Setup](README_building.md#initial-setup) instructions in [README\_building.md](README_building.md) to perform the initial Shell environment and working directory setup.
  * [Compiling and Building](README_building.md#2.-compilingband-building) instructions in [README\_building.md](README_building.md).
  * [Running instructions](README_running.md) to run test configurations.

> *Note 1*: If you're planning on running in batch mode, submitting to another
> machine, make sure you do the *initial setup* and compilation,
> on the machine where the job will be submitted.

> *Note 2*: The compiler is specified as an external dependency in
> the "`_migdep/DEPENDENCIES.external.*.bndl`" files.
> Compiler options, rules and targets are specified in each components's
> Makefiles (`*/include/Makefile.local*mk` files).  
> See **TODO** below for more details.


Making Modifications
====================

Change should be made incrementally one "feature" at a time.  
All modifications should be tested before being committed or shared.

The process should be

  1. Modify: make your own modifications or integrate others' changes.  
     See details on some [specific cases](#modifications-specific-cases) below.
  2. Compile and Build: See Building instructions in
     [README\_building.md](README_building.md) for details.
  3. Test: Tests should be run *before* committing or sharing your modifications.  
     See Running instructions in [README\_running.md](README_running.md) for details.
  4. Commit and tag changes: save your changes incrementally one "feature"
     at a time with a description of the changes.  
     See details in the [commit and tags section](#commit-and-tags) below.
  5. Tag: tags can be added to more precisely identify a set of commits,
     a milestone, for future reference or a temporary point to return back to.  
     **TODO**: See naming convention
  6. Share your modifications: **TODO**

> *Note 1*: Details regarding updating components' version number and other support
> files are addressed in the [README\_installing.md](README_installing.md).

> *Note 2*: details on using `git` and its powerfull features to merge, rebase,
> allow parallel development with branches may be learnt from the abundant
> online documentation.


Compile and Build
-----------------

Obviously you need to re-compile and build the executable whenever the source
code is modified, the compiling options are changed or the dependencies updated.
But it is NOT needed for scripts, module, documentation or other non code changes.

    make buildclean  ## Optional, needed for compiler options or dependencies changes,
                     ## or when source code files where removed
    make deplocal
    make vfiles
    make libs -j ${MAKE_NPE:-6}
    make abs  # -j ${MAKE_NPE:-6}

> See the *initial Shell environment and working directory setup* instructions
> along with the *building instructions* in the
> [README\_building.md](README_building.md) file for more details.


Test
----

Tests should be run to make sure the modifications have the intended effect
as well as for making sure there are no negative side effects in
other configurations.

> Tests should be performed *before* committing or sharing your modifications.

**TODO**: some details

> See running instructions in [README\_running.md](README_running.md) for more details.


Commit and Tags
---------------

Commit is git way of *saving* your changes. Changes should be saved
incrementally, one "feature" at a time, with a description of the changes.

*Before* issuing the commit command
  * Test the changes!
  * Inpect changes to be commited, use `modelutils/bin/r.gitdiff -d --cached`
  * Make sure the needed files have been added, removed or renamed.  
    Sample commands:
````bash
    git status
    git add FILENAME    ## Add new files or add modification for next commit
    git rm FILENAME
    git mv OLDNAME NEWNAME
````
  * Make sure you are not commiting un-related changes.  
    Use `git reset -- PATH/TO/FILE` to move back files to
    *not staged for commit* status (like a `git add` inverse operation).

> It is best to *avoid committing*:
> * with option to auto add modified files (`-a`); this is to avoid committing
>   unrelated changes
> * binary files
> * very large files
> * relative links pointing outside the repository tree or absolute links.

If you are not the author of the code, you may specify the author like this:

    git commit --author='First Last <first.last@canada.ca>'

> See also *commit best practice* on the
> [who-t blog](http://who-t.blogspot.com/2009/12/on-commit-messages.html).


**Commit message best practices**

> Don't describe the code, describe the intent and the approach.  
> And keep the log in a present tense.

The git commit log is 3 parts:

  1. a short (50-78) summary on the first line
  2. a blank line
  3. a longer description that should explain:
    * why is this necessary?
    * how does it address the issue?
    * what effect does this patch has?

> See also *commit log best practice* on the
> [who-t blog](http://who-t.blogspot.com/2009/12/on-commit-messages.html) or
> on the  [CMC wiki](https://wiki.cmc.ec.gc.ca/wiki/Svn/Commit_howto).

**Tagging**

If you reached a milestone with your code or if you'd like to easily
come back to that specific version later, it may be usefull to mark that
specific version with a meaningfull tag. This is simply done with:

    git tag TAGNAME

> See *Git Repositories Tags* in the
> [README\_version\_convention.md](README_version_convention.md) file.


Modifications: Specific Cases
-----------------------------

### Shell Env. SetUp Update ###

> **Quick Ref.**
>
> Files to edit: `*/bin/.env_setup.dot`  
> Mostly defines environment vars.

> You'll need to re-do the [Initial Setup](README_building.md#initial-setup)
> process after modifying this.

The Shell environment can be set in 2 different ways:
  * when working with the full code, it is set by the `*/.setenv.dot` scripts
  * once installed, it is set by the `r.load.dot` RPN-SI script,

To minimize duplications, `*/.setenv.dot` scripts should replicate as much as
possible the `r.load.dot` functionalities. Common stuff should be moved to
other scripts that would be *sourced* by both.

  * dependencies, including compilers, are found in the
    `*/DEPENDENCIES*.bndl` files.  
    See [Dependencies Update section](#dependencies-update) for details.
  * the `*/bin/.env_setup.dot` files (bash scripts) are sourced by both.
    They define other Shell env. vars.

> See: [r.load.dot CMC wiki page](https://wiki.cmc.ec.gc.ca/wiki/R.load.dot)
> for details on RPN's `r.load.dot` utility.


### Dependencies Update ###

> **Quick Ref.**
>
> Files to edit (RPN's `r.load.dot` *bundle* format):
> * External dependencies: `*/DEPENDENCIES.external.*.bndl`
>   * common to all components: `_migdep/DEPENDENCIES.external.*.bndl`
> * Cross-MIG dependencies: `*/DEPENDENCIES.mig.bndl`
>   * May refer to common externals: `ENV/x/migdep/VERSION`
>   * Use the following script to make version numbers consistent:  
>     `_bin/migdepupdater.ksh -v --gitdep --check`

> You'll need to re-do the [Initial Setup](README_building.md#initial-setup)
> process after modifying this.

MIG has several types of dependency files:

  * **MIG's dependencies**:
    These files are mainly for housekeeping by the librarian.  
    *You do not have to edit them.*
    * Present version Git dependencies: "`DEPENDENCIES`" file  
      This file is used by the "`_bin/migsplitsubtree.ksh`" script.  
      This file format is one line per component:  
      `NAME=URL/TAG`
    * All MIG versions Git dependencies: "`_share/migversions.txt`" file  
      This file is used by the "`_bin/migimportall.ksh`" script.  
      This file format is one line per version:  
      `branch=MIGBRANCH; tag=MIGTAG ; components="COMP1NAME/COMP1TAG COMP2NAME/COMP2TAG ..."`
  * **Components' dependencies**  
    *These files are the ones you should be editing.*
    * local, cross-MIG, dependencies; components depend on each other.  
      `*/DEPENDENCIES.mig.bndl`
    * external dependencies, those that are not part of the MIG super repos.  
      `*/DEPENDENCIES.external.*.bndl`

You may update the dependencies list or versions by editing the components'
dependencies files `*/DEPENDENCIES.*.bndl`.
Theses files have the RPN's "`r.load.dot`" *bundle* format.

Notes:

- *external* dependencies must be defined and updated, kept in sync,
  for *every* known `${RDENETWORK}`: "cmc" and "science".

- *external* dependencies Common to most components must be defined in
  the speical `_migdep/DEPENDENCIES.external.*.bndl` files.

- For local, cross-MIG, dependencies versions, please use the following
  script to make them consistent instead of updating them by hand:  
  `_bin/migdepupdater.ksh -v --gitdep --check`


> See: [r.load.dot CMC wiki page](https://wiki.cmc.ec.gc.ca/wiki/R.load.dot)
> for details on RPN's `r.load.dot` *bundle* format.

> See: [README\_version\_convention.md](README_version_convention.md) for
> details on naming convention used in the `*/DEPENDENCIES.mig.bndl` files.


### Compiler Options Update ###

> **Quick Ref.**
>
> Files to edit:
>   * `*/include/Makefile.*.mk`
>   * `Makefile.user*.mk`
>   * Source files (overrides)
>
> `COMP_RULES_FILE = /PATH/TO/YOUR_COMP_RULES_FILES` can also be added to any
> `Makefile` to override default compiler options.

> You'll need to clean (`buildclean`) and re-do the 
> [compile/build process](README_building.md) after modifying this.

The compile/build system is based on `gmake` and on 2 wrappers on top
of the compiler:

  * RPN/CMC *Compiler Rules* used with their own "`s.compile`" script.  
    You can visualize these with (after the
    [Initial Setup](README_building.md#initial-setup)):  

        cat $(s.get_compiler_rules)

    *Note that the "`s.get_compiler_rules`" script comes from SSC and
    is not part of the MIG super repos.*  
    *This wrappers is needed for compiler options consistency with other
    libraries and product from EC/CMC*

  * RDE Makefiles and scripts, see the *rde* component for details.  
    Edit the following files to update the compiler rules.
    * RDE defines basic Makefile vars, rules and targets.  
      File: `rde/include/Makefile*`
    * every components can add specific Makefile vars, rules and targets.  
      File: `*/include/Makefile.local*.mk`
    * MIG can have specific Makefile vars, rules and targets.  
      *Note that this file is not used in the installed GEM.*  
      File: `Makefile.user*.mk`

On top of this, *RDE* supports per file compiler options overrides. You can add
or suppress options for a specific compiler in any source file by adding a *RDE
directive* at the top of the source file. The directive takes the forms:
   * Fortran: `!COMP_ARCH=compilername ; -add=options -suppress=options`
   * C: `/*COMP_ARCH=compilername ; -add=options -suppress=options*/`

For example:

    !COMP_ARCH=intel-2016.1.156 ; -add=-C -g -traceback -ftrapuv

> See [RDE documentation on the CMC wiki](https://wiki.cmc.ec.gc.ca/wiki/Rde)
> for details. (**TODO**: include RDE doc as a README.md in the git repos)

Finally, if the main RPN/CMC *Compiler Rules* (`s.get_compiler_rules`) is not s
uiting your needs, you can override it with you own. To do so define in any of
the Makefiles:

   COMP_RULES_FILE = /PATH/TO/YOUR_COMP_RULES_FILES

> Description of the compiler rules files is beyond the scope of this doc.  
> You may want to start from the original one and modify it:  
> `cp $(s.get_compiler_rules) MY_COMP_RULES_FILE`

Thus compiler options, rules and targets can be modified in:

  * `*/include/Makefile*`: add `COMP_RULES_FILE=` to override system-wide default options
    for system-wide default options
  * `rde/include/Makefile*`: for system-wide options
  * `*/include/Makefile*`: for components specific options
  * `*/src/*`: for file specific options
  * `Makefile.user*.mk`: for full MIG build system specific options

> **TODO**:
>   * List expected (by the build/compile system) components' Makefile targets and vars


### Remove a Component ###

> Before this step, it is best to commit any other changes you may have.

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

> You'll need to re-do the [Initial Setup](README_building.md#initial-setup)
> process after modifying this.


### Add a Component ###

> Before this step, it is best to commit any other changes you may have.

Import the component from a remote Git repository using `git subtree`:
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

Make sure the component has the needed required directories, files and content
for MIG integration/build system.  
See the [layout section](#layout) above.

> You'll need to re-do the [Initial Setup](README_building.md#initial-setup)
> process after modifying this.


### Getting Others' Modifications ###

To include other developers' MIG clone modifications, you may merge from their repository or apply patches they have provided you.

> Patching and Merging can be complex, and should not be taken lightly.

> Before this step, it is best to commit any other changes you may have.


#### Merging: Applying Patches ####

> Patches are expected to have been created with  
> `BASETAG=      ## Need to define from what tag (or hash) to produce patches`  
> `git format-patch HEAD..${BASETAG}`

If your patch is to be applied on a sub component (sub directory),
then set MYDIR to its name (example for gemdyn below).

    MYDIR="--directory=gemdyn"

Define the PATH/NAME of the patch file:

    MYPATH=/PATH/TO/${MYPATCH}

Before applying the patch, you may check it with:

    git apply --stat ${MYPATCH}
    git apply --check ${MYDIR} ${MYPATCH}

Fully apply the patch

    git am --signoff ${MYDIR} ${MYPATCH}

Selective application (if fully apply does not work or you want to exclude
some parts), random list of commands

    git apply --reject PATH/TO/INCLUDE   ${MYDIR} ${MYPATCH}
    git apply --reject --include PATH/TO/INCLUDE  ${MYDIR}  ${MYPATCH}
    git am    --include PATH/TO/INCLUDE  ${MYDIR} ${MYPATCH}
    git apply --exclude PATH/TO/EXCLUDE  ${MYDIR} ${MYPATCH}
    git am    --exclude PATH/TO/EXCLUDE  ${MYDIR} ${MYPATCH}


Fixing apply/am problems

  * inspect the reject
  * apply the patch manually (with an editor)
  * add file modified by the patch (git add...)
  * git am --continue

> Note: it is best, to avoid conflicts, to apply the patch directly on top of
> the `${BASETAG}` they were produced from. If you have modifications of your own
> on top of `${BASETAG}` you may want to
> * creat a branch from `${BASETAG}`: `git checkout -b ${BASETAG}-mine ${BASETAG}`
> * apply the patch on that branch as above
> * merge it to your branch:  
>   `git checkout MYBRANCH; git merge ${BASETAG}-mine; git branch -d ${BASETAG}-mine`

> See also:
>   * https://www.devroom.io/2009/10/26/how-to-create-and-apply-a-patch-with-git/
>   * https://stackoverflow.com/questions/25846189/git-am-error-patch-does-not-apply
>   * https://www.drupal.org/node/1129120


#### Merging: Pulling Others' Code ####

1. Add a remote branch for the other's repos
````
git remote add NAME URL/PATH
````
2. Fetch remote repos and check-it-out at the provided tag as local branch
````
git fetch NAME
git checkout -b NAMELOCAL TAGNAME
````
3. Merge that local branch in your own branch
````
  BRANCH=${NAME}_${VERSION}-${USER}-branch
  git checkout ${BRANCH}
  git merge NAMELOCAL
````
4. Fix/Merge the patch if needed

5. Remove remote
````
git remote rm NAME
````

### Revert Changes ###

Whenever you made a change you no longer want, it is easy with `git` to return
to a previous iteration.

To reset/revert all files:  
*This does erase the history, local changes and commits may be lost, proceed with care*

    git reset --hard HASH

To reset/revert only specific file(s):  
*This does not erase the history, only local, not yet commited, changes may be lost*

    git checkout HASH -- PATH/TO/FILE

> `HASH` can be:
> * a TAG (`git tag -l`), 
> * an actual commit hash (`git log --pretty=oneline`),
> * `HEAD`, a special keyword referring to the last commit

> Note: do not perform a revert (`git reset --hard HASH`) beyond your local commits,
> this would make sharing your modifications difficult.


### Branches, Parallel Development ###

Git allow you to work on several branches (kind of virtual directories) in
parallel. This may be useful to test someting befing integrating (merging)
it into your main developement branch.

Creating a new branch is as simple as:

    git checkout -b NEWBRANCH HASH     ## creates a new branch from the HASH

> `HASH` can be:
> * a BRANCH (`git branch`),
> * a TAG (`git tag -l`),
> * an actual commit hash (`git log --pretty=oneline`),
> * `HEAD`, a special keyword referring to the last commit (this is the default)

You can go back and forth between the NEWBRANCH and your MAINBRANCH with a simple
`git checkout BRANCHNAME`.
*Make sure you changes are commited before swiching branches.`

When you are satisfied with this branch's code, and it is fully commited,
you may merge it back into your main developement branch:

    git checkout MAINBRANCH
    git merge NEWBRANCH

For house keeping, it is best to remove old branches that have been fully merge
or that are no longer needed (code you no lnguer want to keep)

    git branch -D NEWBRANCH


### Cleaning up ###

To remove all files created by the setup, compile and build process,
use the `distclean` target.

        make distclean

You may further clean up the GEM dir by removing all imported components with the following command.
> **WARNING**: To avoid loosing your modifications. make sure you created patches (saved elsewhere) or `git push` the modified components' code before removing all imported components.

**TODO**: See bin/clean-subtree.ksh


Contribute Back, Share
----------------------

When sharing your modifications to the librarian or other developers, users, there are 3 ways you can go about it.


### Send Pull Request ###

The best way to give your code back to be included, merged, into the upcoming
MIG/GEM version is to send a pull request to the librarian.

* First tag the version of the code (the specific commit) you tested successfully:  
  `git tag TAGNAME`
* Then send the URL or PATH to you repository, along with the TAGNAME


### Create Patches ###

Otherwise you may create patches to be shared

**TODO**: rm VERSION number? consistent with setup_import... or save original VERSION (or TAG1) somewhere, or save the original tag name somewhere...

    TAG1=${NAME}_${VERSION}-${USER}.1
    git format-patch ${TAG1}

    expname=SOME_EXPLICIT_NAME
    patchname="${TAG1}_${expname}.patch.tgz"
    patchlist="$(ls *.patch)"
    rm -f ${patchname}
    tar czf ${patchname} ${patchlist}

Then send the PATH to your patches (`${patchname}`).


### Push Upstream ###

If your have write permissions in the original MIG Git repositories,
you can `git push` your modifications directly.

> The librarian will have to do this after merging in, or applying patch of,
> others' contributions.

**TODO**

    # loop over components
    # sash1="$(git ls-remote -t ${remote_name} ${tagname} | cut -c1-41)"
    # branch="$(git ls-remote -b ${remote_name} | grep ${sash1})"
    # branch="${branch##*/}"
    # if [[ x${branch##*/} == x ]] ; then "create new branch local at tag then push remote local_branch:remote_branch"; fi
    # git push ${remote_name} local_branch:remote_branch
    # git push --tags (or --follow-tags) ${remote_name} local_branch:remote_branch

    #git remote show <name>
    #git remote set-branches [--add] <name> <branch>...
    git push
    # --follow-tags

    # special case for top repos (gem)...


See Also
========

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


TODO
====

... librarian only? ... 
  * Update from inidividual components' repository (git subtree pull), similar to adding a component



To Be Included in Other README/Sections
====================================

**Must be done on every arch**


Developing
----------

- Edit code and scripts directly in components' directory.
- Re-Build following the instructions in Building section above.

Ground rules
------------

To keep things sane with many developers there needs to be a set of guidelines that control how the code is modified.
The set of rules governing  how users interact with the VCS and how the branches of the VCS interact is referred to as the
CCCma workflow ([Guidelines](http://wiki.cccma.ec.gc.ca/~acrnncs/CanESM5_development_guide.pdf)). The most important point is that every logical grouping of work is done on its own branch. All official CMIP6
development must respect these rules, as repeatability depends on it.

There are conceptual changes in moving from the old system to version control. Some important notes:

- there are no updates being used whatsoever. All changes are made in the version controlled source code directly.

- no bin directories are used. All runs are self contained (every piece of code used in a run is checked out from version control. 
  This is done automatically by `setup-canesm`, which provides a link to the source code). All changes are made directly in that code,
  which is specific to that run alone. 

- Given the complexity of the ESM and CCCma systems, we have chosen to structure the git repository as a CanESM super-repo, with each
  model component as *submodules*. This provides a great amount of flexibility, since each subcomponent is maintained as its own repo.
  Unfortunately, even if you are familiar with git, submodules can be complex and tricky. You must follow the steps below to make sure
  that you treat the submodules and super-repo correctly.

- Every logical grouping of work is done on its own branch. This branch must exist in the super-repo and all 
  submodules being modified (see below).

- The starting point for each piece of work must be clearly defined. Normally this will be a recent commit off the develop branch.


Modifying the code
------------------

1. **Create an issue**

    Before beggining work, an issue (under a milestone) is created on the gitlab based
    [CanESM5 issue tracker](https://gitlab.science.gc.ca/CanESM/CanESM5/issues). This allows us to keep track of what was done
    when and why. When setting up the issue, you should have a clear starting point (i.e. super-repo SHA1). If in doubt, ask Scinocca.<br><br>

2. **Clone the CanESM5 super-repo.**

    If you have setup a run on the XC40s, `setup-canesm` will have made a clone of the code
    and provided a link to it (`CanESM_source_link`), and you could modify the code directly there. However, you may want to work
    on you laptop or elsewhere, in which case get the clone using:

    git clone --recursive git@gitlab.science.gc.ca:CanESM/CanESM5.git
    cd CanESM5

    CAUTION: This might fail if you have not setup your ssh keys on gitlab. Follow [these](https://wiki.cmc.ec.gc.ca/wiki/Subscribing_To_Gitlab) instructions.
    At this stage you should checkout the starting point, e.g.:

    git checkout SHA1

    where the string following `checkout` is the hash (=SHA1=commit) or branch that you want to checkout. Be sure to update the submodules, e.g.:

    git submodule update --recursive --init --checkout

    Now you are ready to modify the code and proceed. If you already have an established repo, just do the last two steps (checkout and submodule update)
    <br><br>

3. **Checkout named branches**

    You must checkout a new branch in the super repo, and in each submodule that you plan to modify. This is now done automatically using "sbranch" and "scheckout".
    If a branch does not exist yet, then do

    git scheckout -b BRANCH-NAME

    if this does not work (we have seen issues in some versions of git used on lxwrk), you might try:

    git sbranch BRANCH-NAME
    git scheckout BRANCH-NAME

    where BRANCH-NAME is the name of your custom branch. If the BRANCH-NAME already exists, then do

    git scheckout BRANCH-NAME

    You can now go ahead and modify any code you like (in any submodule).<br><br>

4. **Add and commit the changes**

    To add changes in all submodules and the super-repo automatically, use (from the top super-repo level, i.e. CanESM):

    git sadd

    you could optionally do custom staging by adding files manually with `git add file`. To commit you can use (from the top level):

    git scommit -m '"Modifies CanAM - plus desc"'      # Commit this change using the same message in all repos.

    note here in particular the single quotes around the double quotes. If you do not specify `-m`, you can leave custom commit messages
    for each repo modified. <br><br>

5. **Push changes back to gitlab**


    Again, from the super-rpeo level, push using:

    git spush origin BRANCH-NAME

6. **Launch a test run from gitlab**

    Lauch a test run from scratch using the modified code. To do this use your CanESM5 super repo commit obtained at the end of step 4
    (you can use `git log` to look though recent commits, or look in the gitlab history to determine which commit you want to test).
    Follow the instructions in [README-Running-CanESM-VCS.md](README-Running-CanESM-VCS.md).

       Iterate through 4-6 until you are satisfied that you changes are complete, and you want to hand them back for inclusion. At this
       point it is expected that you have a documenting run, done cleanly off a super-repo commit, which is stable.<br><br>

7. **Create a merge request upon completion**

    On [gitlab](https://gitlab.science.gc.ca/CanESM/CanESM5) make a merge request between your branch and develop.


Git Cheat sheet
---------------

**TODO**: move to other file

* getting help with `git`
    git help -a
    git COMMAND --help

* Viewing the status of your working directory
  git status
  gitk --all &

* Adding new file or (not empty) dir to git, or shedule Iby default new files are not "tacked" by git
    git add /PATH/TO/FILE_OR_DIR
* Permenantly ignore files or dir
  Every file or dir matching a pattern in the `.gitignore` file will be disregarded, one pattern per line (**TODO**: shell pattern?)

* Commit
    git commit     # commit only added changes, after "git add", "git rm", "git mv", ...
    git commit -a  # commit all changes
* Commit other people's work
    git commit --author='<First Last <name@email.com>'
* Adding changes to the last commit
    git commit --amend

* Reseting all changes before committing
    git reset --hard
* Reseting all changes after committing
    git reset --hard HEAD^
* Reseting change of only one file before committing
    git checkout HEAD -- /PATH/TO/FILE
* Reseting change of only one file after committing
    git checkout HEAD^ -- /PATH/TO/FILE
* Reseting present work (not commited) and save it for later
    git stash
    # git stash pop  #to retreive it later

* Start a topic branch
    git checkout -b branch_name

* Going back to the main branch
    git checkout branch_name

**TODO**
* https://www.quora.com/What-is-the-best-Git-cheat-sheet
* https://git-scm.com/docs
* https://scotch.io/bar-talk/git-cheat-sheet
* https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf
* https://www.git-tower.com/blog/git-cheat-sheet/
* https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet
* https://docs.gitlab.com/ce/gitlab-basics/README.html

**TODO**: Version Controle best pratices at https://www.git-tower.com/blog/git-cheat-sheet/

