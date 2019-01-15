
Developing MIG/GEM from Version Control
=======================================

This document describes the basic steps required to modify MIG/GEM source code using Version Control.
**TODO**: For more information on the system and its rationale see [README.md](README.md) and [Guidelines](http://wiki.cccma.ec.gc.ca/~acrnncs/CanESM5_development_guide.pdf).
Feel free to provide feedback on the guidelines and rules.

The version control system is based on git. If you know nothing about git you should educate yourself first.
These resources may help:

  * [official git documentation](https://git-scm.com/doc)
  * [Atlassian git tutorials](https://www.atlassian.com/git/tutorials)
  * [Git subtree docs](https://github.com/git/git/blob/master/contrib/subtree/git-subtree.txt)
  * [CCCma NEMO version control help page](http://wiki.cccma.ec.gc.ca/twiki/bin/view/Main/NemoVersionControl)

Specifically, you should be comfortable with the git definitions of: "commit", "tag", "branch", "checkout", "staging", "reset", "merge".


**Table of Contents**

* [Initial Setup](#initial-setup)
* [Component's Layout](#components-layout)
    * [bin/](#bin)
    * [include/](#include)
    * [lib/](#lib)
    * [share/](#share)
    * [src/](#src)
    * [.restricted](#restricted)
    * [.name](#name)
    * [Makefile.local.mk](#makefilelocalmk)
    * [setenv.dot](#setenvdot)
* [Making Modifications](#making-modifications)
    * [Compile and build](#compile-and-build)
    * [Test](#test)
    * [Commit](#commit)
    * [Specific cases](#specific-cases)
        * [Modifying the dependencies](#modifying-the-dependencies)
        * [Modifying the Shell Env. SetUp](#modifying-the-shell-env-setup)
        * [Modifying scripts, maestro module](#modifying-scripts-maestro-module)
        * [Modifying the source code](#modifying-the-source-code)
        * [Updating the Version number](#updating-the-version-number)
        * [Reverting changes](#reverting-changes)
* [Getting other's modifications / Merging](#getting-others-modifications--merging)
    * [Merging: Applying patches](#merging-applying-patches)
    * [Merging: pulling others' code](#merging-pulling-others-code)
* [Contribute back: push, send pull request or make patches](#contribute-back-push-send-pull-request-or-make-patches)
    * [Send Pull Request](#send-pull-request)
    * [Create Patches](#create-patches)
    * [Push Upstream](#push-upstream)
* [Install](#install)
* [Uninstall](#uninstall)
* [Cleaning up](#cleaning-up)
* [See Also](#see-also)

* [To Be Included in Other README/Sections](#to-be-included-in-other-readmesections)
    * [Basic Dev.Env. Setup](#basic-devenv-setup)
    * [Building](#building)
    * [Developing](#developing)
    * [Installing](#installing)
    * [Ground rules](#ground-rules)
    * [Modifying the code](#modifying-the-code)
    * [Deleting old branches](#deleting-old-branches)
    * [Merging](#merging)
    * [Git Cheat sheet](#git-cheat-sheet)
    * [git stash pop  #to retreive it later](#git-stash-pop--to-retreive-it-later)


Layout
------

This MIG repository is referred to as the *super-repo* (a *mono-repos* including all MIG products, all sub-repositories; called components). 

  * The file `DEPENDENCIES` list the name and origin of each components, its URL and tag
  ** each line in `DEPENDENCIES` has the format (lowercase): name=url/tag
  ** tags have the format `"name_version" in lowercase
  * The `_bin` and `_share` directories contains some scripts and data files to help import components and to setup the SHELL environment for compiling and installing MIG components/products (`_bin/migimportall.ksh` and `_share/migversions.txt` were used to create this repository and can be used to update it).

Each component is in its own sub-directory and is imported using the `git subtree` commands. These components/sub-directories may contain the following sub-directories and files:

  * `bin/       `: used for scripts, will be added to the `PATH`
  * `include/   `: used for included files and sub-Makefiles, will be added to the `INCLUDE_PATH`
  * `lib/       `: will be added to the `LIBRARY_PATH`
  * `lib/python/`: used for python modules, will be added to the `PYTHONPATH`
  * `src/       `: used for source code, will be added to the source path (`VPATH`).
  * `share/     `: used for any other content
  * `.ssm.d/    `:
  
  **TODO: Complete list with mandatory and optional**
  * `.name`:
  * `.setenv.dot`:
  * `bin/.env_setp.dot`:
  * `VERSION`:
  * `DEPENDENCIES.mig`:
  * `DEPENDENCIES.external.bndl-cmc-science`:
  * `include/Makefile.local.NAME.mk`: ... note that all these components' Makefiles will be merged/included into the main one, please make sure modifications to them does not have undesired side effect on other components.
  * `.restricted`:


Initial Setup
-------------

Please follow the [Initial Setup](README_running.md#initial-setup) instructions in [README\_running.md](README_running.md) to perform the initial Shell environment and working directory setup.


Compiling and Building
----------------------

*Note 1*: If you're planning on running in batch mode, submitting to another machine,
make sure you do the *initial setup*, including compilation,
on the machine where the job will be submitted.

*Note 2*: The compiler is specified in `modelutils`'s dependencies.
Compiler options, rules and targets are specified in each components's
Makefiles (`*/include/Makefile.local*mk` files).  
See **TODO** below for more details.

Initially it is best to make sure to start with a clean slate.  
*This should be done once initially and every time the *dependency* list is modified.*

        make buildclean

Use the following Makefile targets to compile the build libs and abs.  
*This needs to be done initially and every time the code is modified.*

        make dep
        make vfiles
        make libs -j ${MAKE_NPE:-6}
        make abs  -j ${MAKE_NPE:-6}


Testing
-------

**TODO**:




Making Modifications
--------------------

Change should be made incrementally one "feature" at a time.
Should be tested before being committed or shared.


### Compile and build ###

Obviously you need to re-compile and build the executable whenever the source code is modified, the compiling options are changed or the dependencies updated. But it is NOT needed for scripts, module, documentation or other non code changes.

        make buildclean  #Optional, needed for compiler options or dependencies changes, or when source code files where removed
	    make deplocal
        make vfiles
        make libs -j ${MAKE_NPE:-6}
        make abs  -j ${MAKE_NPE:-6}


### Test ###

**TODO**: run tests before committing


### Commit ###

**TODO**: Best to test before committing
**TODO**: Commit message best practice
**TODO**: commit best practices (link?)... avoid committing unrelated stuff together, avoid binary files (espc big files)...
**TODO**: 

	    # git add ???   # git mv/rm
        # git commit -m 'desc'
        # git tag TAGNAME


### Specific cases ###

#### Modifying the dependencies ####

**TODO:** update for new migimportall.ksh/migversion.txt 

The list of *dependencies* to import from external git repositories is specified in the `DEPENDENCIES` files of each components.
The *dependencies* are specified as a list of lines with the following format (lowercase):

        NAME = URL/TAG

Where
- `NAME`: Name of the sub-dir where the components code will be imported
- `URL` : URL of the component's git repository
- `TAG` : Git tag of the component's version to be imported. Normally these tags correspond to the version specified in the component's `VERSION` file. A *TAG* must not contain any "@" character.

If a new component's *version/tag* is already present in the component's git repository and needs to be imported, you can re-run the `setup_import` (**TODO** setup_import replaced by...) script to do so.

> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


#### Modifying the Shell Env. SetUp ####

Every component is expected to have a `setenv.dot` file in its top dir.
This file should be a `bash` script that defines a set of var, update the PATH's.
You may modify this file to add, remove, change the Shell Env.Var. set by the components.
When adding a component, make sure it has this file.

> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


#### Modifying scripts, maestro module ####

**TODO**: 

> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


#### Modifying the source code ####

**TODO**: 

> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


#### Updating the Version number ####

By convention, version numbers are specified in the `VERSION` files of components' top dir. Make sure:
* a *git TAG* consistent with VERSION is created for the component
* that the *TAG* specified in the `DEPENDENCIES` file of the other components is consistent with this new *git TAG*/VERSION

**TODO**: add tag to component consistent with VERSION... and push it
**TODO**: update other components' DEPENDENCIES to match the new TAG
**TODO**: test consistency


#### Reverting changes ####

**TODO**: reverting a change (before creating the patches or pushing upstream)
**TODO**: to revert a changes that was already shared the best way is to commit the *reverted changes* explaining the rational behind it.


Getting other's modifications / Merging
---------------------------------------

### Merging: Applying patches ###

* Verify that the patch can be applied
  `git apply --check PATCHFILE`

* Apply the patch
  `git am PATCHFILE`

* Fix/Merge the patch if needed
  **TODO**: dealing with `git am` problems

> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


### Merging: pulling others' code ###

* Create your own working dir following the initial instructions in [README\_running.md](README_running.md).

* **TODO** Add a remote branch for the other's repos
  `git remote add NAME URL/PATH`
* **TODO** Fetch remote repos and check-it-out at the provided tag as local branch
   ```
   git fetch NAME
   git checkout -b NAMELOCAL TAGNAME
   ```
* **TODO** Merge that local branch in your own branch
   ```
   BRANCH=${NAME}_${VERSION}-${USER}-branch
   git checkout ${BRANCH}
   git merge NAMELOCAL
   ```

* Fix/Merge the patch if needed


> [Compile](#compile-and-build) and [test](#test) then [commit](#commit).


Contribute back: push, send pull request or make patches
--------------------------------------------------------

When contributing your modifications to the librarian or other developers, users, there are 3 ways you can go about it.

** Split components **

Before anything, it is useful to split each components into its own branch to isolate changes.

**TODO**: See bin/split-subtree.ksh


### Send Pull Request ###

The best way to give your code back to be included, merged, into the upcoming GEM version is to send a pull request to the librarian.

* First tag the version of the code (the specific commit) you tested successfully: `git tag TAGNAME`
* Then send the URL or PATH to you repository, along with the tag name, to the librarian


### Create Patches ###

Otherwise you may create patches to give to the librarian.

**TODO**: rm VERSION number? consistent with setup_import... or save original VERSION (or TAG1) somewhere, or save the original tag name somewhere...

        TAG1=${NAME}_${VERSION}-${USER}.1
        git format-patch ${TAG1}

        expname=SOME_EXPLICIT_NAME
        patchname="${TAG1}_${expname}.patch.tgz"
        patchlist="$(ls *.patch)"
        rm -f ${patchname}
        tar czf ${patchname} ${patchlist}

Then send the PATH to your patches (`${patchname}`) to the librarian.


### Push Upstream ###

If your have write permissions in the original Git repositories of GEM and its components, you can *push* your modifications directly.
The librarian will have to do this after merging in, or applying patch of, others' contributions.

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


Cleaning up
-----------

To remove all files created by the setup, compile and build process, use the `distclean` target.

        make distclean

You may further clean up the GEM dir by removing all imported components with the following command.
> **WARNING**: To avoid loosing your modifications. make sure you created patches (saved elsewhere) or `git push` the modified components' code before removing all imported components.

**TODO**: See bin/clean-subtree.ksh


Component's Layout
------------------

This section describe the general layout of a components's elements, sub directories and files. It serve as a template to create a new component or as a reference to modify existing ones.

        bin/
            .restricted
        include/
            Makefile.local.mk
        lib/
            python/
        share/
        src/
            .name -> ../.name
        .gitignore
        .name
        DEPENDENCIES
        DESCRIPTION
        README.md
        setenv.dot
        VERSION


#### bin/ ####

Component's specific scripts. Could be Shell, python or else.
For python modules or similar, it is best to put them under the `lib/python/` dir.


#### include/ ####

Fortran or C source files to be included.
Place here the files that can be included from other components.
Preferably put file that should not be included from other components in the `src/` dir.
Components specific Makefiles variables, rules and targets should be specified in the `include/Makefile.local*mk` file.


#### lib/ ####

Python, R, and other libraries or modules should be placed under the `lib/` dir.
For python modules or similar, it is best to put them under the `lib/python/` dir.


#### share/ ####

Most of the other things exported by a component that is not source code, lib or scripts should be put under the `share/` dir.


#### src/ ####

Directory and sub-directories for Fortran or C source files.
The dependency analyzer will create a list of objects files separated for each very `src/` sub-directories and these can thus be considered as individual libraries.


#### .restricted ####

This file is used by the automatic dependencies generator call by `make dep`. It is a text file with white-list of arch (one per line) on which the dependency analyzer can can operate.

  * No `.restricted` file means the dir will be analysed for dependencies.
  * An empty `.restricted` file cause, on all arch, the directory containing it to be ignored by the dependencies analyzer.
  * A not empty `.restricted` file cause, on arch NOT matching one the white-list arch, the directory containing it to be ignored by the dependencies analyzer.


#### .name ####

Name of the component equivalent to the sub dir they are imported in. This file is used by some `rde` scripts.


#### Makefile.local.mk ####

**TODO**: list of recognized var (required or not)

        NAME_VERSION

        NAME_VFILES
        NAME_LIBS_FILES
        NAME_ABS_FILES

        NAME_SSMALL_FILES
        NAME_SSMARCH_FILES

** Makefile.dep.${ARCH}.mk **

File generated by `make dep`. This files contains auto-built source dependencies for Fortran and C source code. It exports a number of symbols that may be used in the component's `Makefile.local*mk` file. Among the most interesting ones:

        TOPDIRLIST_NAMES
        SUBDIRLIST_name
        OBJECTS
        OBJECTS_name
        OBJECTS_name_subname
        FORTRAN_MODULES
        FORTRAN_MODULES_name
        ALL_LIBS

**TODO**: other details about RDE, ouv_exp_gem/srcpath, linkit/builddir/.rde.condif.dot, rules/targets, .rde.setenv.dot, ...


#### setenv.dot ####

**TODO**: script behavior, recusivity, DEPENDENCIES file

**TODO**: modeltuils special cases for specifying external dependencies including compilers

See Also
--------

  * Main doc: [README.md](README.md).
  * Running instructions: [README\_running.md](README_running.md). 
  * Installing instructions: [README\_installing.md](README_installing.md). 



To Be Included in Other README/Sections
====================================


Basic Dev.Env. Setup
--------------------

**Must be done on every arch**

        ## Sample SHELL Session
        . ./.setenv.dot
        
        ./_bin/ouv_exp_mig -v
        
        rdemklink -v
        make buildclean
        ##


Building
--------

Before building you'll need to do the Basic Dev.Env. Setup (see above section).

**Must be done on every arch**

        ## Sample SHELL Session
        make dep
        make libs -j9
        make abs
        ##


Developing
--------

- Edit code and scripts directly in components' directory.
- Re-Build following the instructions in Building section above.


Installing
----------

- Edit components version number (component dependent), may be in VERSION, Makefile or include/Makefile.local*mk files
 build all libraries and binaries (abs) as specified in the Building section above.

**Must be done on every arch**

        ## Sample SHELL Session
        make ssmarch SSM_DEPOT_DIR=~/SsmDepot
        ##

**On the "front end" arch only**

        ## Sample SHELL Session
        make ssmall SSM_DEPOT_DIR=~/SsmDepot
        make components_install CONFIRM_INSTALL=yes SSM_DEPOT_DIR=~/SsmDepot COMPONENTS="${RDECOMPONENTS}"
        ##








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
<br>

6. **Launch a test run from gitlab**

    Lauch a test run from scratch using the modified code. To do this use your CanESM5 super repo commit obtained at the end of step 4 
    (you can use `git log` to look though recent commits, or look in the gitlab history to determine which commit you want to test).
    Follow the instructions in [README-Running-CanESM-VCS.md](README-Running-CanESM-VCS.md).

       Iterate through 4-6 until you are satisfied that you changes are complete, and you want to hand them back for inclusion. At this
       point it is expected that you have a documenting run, done cleanly off a super-repo commit, which is stable.<br><br>

7. **Create a merge request upon completion**

    On [gitlab](https://gitlab.science.gc.ca/CanESM/CanESM5) make a merge request between your branch and develop.

<br>

Deleting old branches
---------------------

To delete an old, no-longer used branch both locally and remotely use:

    git sdel BRANCH-NAME

be sure that no-one else is using it though.

Merging
-------

Merging can be complex, and should not be taken lightly. If you need to merge in upstream changes to your branch,
you can use `git smerge` to help. For example, if I want to add the latest develop_canesm updates to my branch fancy-test, I would do this:

    git sfetch                                     # Fetches latest updates from gitlab, for all submodules.
    git smerge origin/develop_canesm fancy-test    # merges origin/develop_canesm into fancy-test, across all submodules.
    
At this point I would have to resolve any merge conflicts that arose. Then, I could do steps 4-6 above to complete the merge,
and push my updated branch back to gitlab.


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

