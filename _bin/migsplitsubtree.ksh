#!/bin/bash

## WARNING - bash only since it depends on ${!item}
__b=1
__a="__b"
if [[ "x${!__a}" != "x${__b}" ]] ; then
   cat 1>&2 <<EOF
ERROR: Only supported in "bash"-like SHELL with support for \${!name} syntaxe
       __b=1; __a="__b"; \${!__a} = ${!__a} != ${__b}
EOF
   unset __a __b
   return 1
fi
unset __a __b

MYSELF="${0##*/}"
DESC='Split MIG monorepos into one component per branch'
USAGE="USAGE: ${MYSELF} [-v] [--tag | --patch TAGNAME]"

usage_long() {
   toto=$(echo -e $USAGE)
   more <<EOF
$DESC

$toto

Options:
    -h, --help     : print this help
    -v, --verbose  : verbose mode
    -f, --force    : force operation (tag and push)
        --tag      : Add tag for each compoents (COMPNAME_VERSION)
        --patch    : Create patches for each components from provided tag
        --push     : Push each components upstream
                     TODO: option to provide base URL
                     TODO: option to provide remote branch name
        --clean    : remove split branches and remote branches, tags
                     [--push, --tags, --patch are then ignored]
EOF
}

_verbose=0
_tag=0
_force=""
_patch=''
_push=0
_clean=0
_mydry=0
while [[ $# -gt 0 ]] ; do
   case $1 in
      (-h|--help) usage_long; exit 0;;
      (-v|--_verbose) ((_verbose=_verbose+1));;
      (-f|--force) _force="-f" ;;
      (--tag) _tag=1 ;;
      (--patch) _patch='' ;;
      (--push) _push=1 ;;
      (--clean) _clean=1 ;;
      (--dryrun) _mydry=1 ;;
      (*)
         case ${previous} in
            (--patch) _patch="${1}" ;;
            (*)
               echo "Option Not recognized: ${1}" 1>&2
               usage_long
               exit 1;;
         esac;;
   esac
   if [[ x${1#-} != x${1} ]] ; then
      previous=${1}
   fi
   shift
done


## Get component version number
get_version() {
   _item=${1}
   _itemUC="$(echo ${_item} | tr 'a-z' 'A-Z')"
   if [[ -f ${_item}/VERSION ]] ; then
      _itemv="$(cat ${_item}/VERSION)"
   elif [[ -f ${_item}/include/Makefile.local.${_item}.mk ]] ; then
      _itemv="$(make -f _share/Makefile.print.mk print-${_itemUC}_VERSION OTHERMAKEFILE=$(pwd)/${_item}/include/Makefile.local.${_item}.mk)"
   elif [[ -f ${_item}/include/Makefile.local.mk ]] ; then
      _itemv="$(make -f _share/Makefile.print.mk print-${_itemUC}_VERSION OTHERMAKEFILE=$(pwd)/${_item}/include/Makefile.local.${_item}.mk)"
   elif [[ -f ${_item}/Makefile ]] ; then
      _itemv="$(make -f _share/Makefile.print.mk print-${_itemUC}_VERSION OTHERMAKEFILE=$(pwd)/${_item}/Makefile)"
   else
      _item2=${_item}_version
      _itemv=${!_item2}
   fi
   if [[ "x${_itemv}" != "x" && "x${_itemv}" != "xdev" ]] ; then
      _itemv=${_itemv}${_versfx}
   fi
   echo ${_itemv:-dev}
}


## Get list of components
getcomplist() {
   if [[ -f DEPENDENCIES ]] ; then
      export RDEDEPS="$(cat DEPENDENCIES | tr '\n' ' ')"
   else
      echo "ERROR: cannot find DEPENDENCIES file" 1>&2
      exit 1
      # export RDECOMPONENTS="$(find . -maxdepth 1 -type d | cut -d/ -f2 | grep -e '^[a-zA-Z]\+' | grep -v GEM_cfg | tr '\n' ' ')"
      #TODO: define RDEDEPS from RDECOMPONENTS + provided base URL + version info
   fi

   if [[ "x${RDEDEPS}" == "x" ]] ; then
      echo "ERROR: no known DEPENDENCIES" 1>&2
      exit 1
   fi
   echo ${RDEDEPS}
}


## Split component
# splitcomp() {
#    git remote add ${name} ${remoteurl} #--no-tags
#    git fetch ${name}
#    git subtree split -P ${name} -b ${mybranch1}
# }


## Push component
# pushcomp() {
#    git remote add ${name} ${remoteurl} #--no-tags
#    git fetch ${name}
#    git subtree push -P ${name} ${remoteurl} ${mybranch1}
# }


## Create git patch set
git2patch() {
   #See: https://www.devroom.io/2009/10/26/how-to-create-and-apply-a-patch-with-git/
   COMPNAME=${1:-NO_SUCH_COMP}
   COMPVER=${2:-NO_SUCH_VERSION}
   GITTAG0B=${3:-NO_SUCH_TAG}
   DESTDIR=${4:-.}
   git format-patch -o ${DESTDIR%/} -M -C ${GITTAG0B}
   # git format-patch HEAD...${GITTAG0B}  #TODO: check this
   patchname="${DESTDIR%/}/${COMPNAME}_${COMPVER}+${USER}.patch.tgz"
   __here=$(pwd)
   cd ${DESTDIR}
   patchlist="$(ls *.patch)"
   if [[ x"${patchlist}" != x"" ]] ; then
      rm -f ${patchname}
      tar czf ${patchname} ${patchlist}
      rm -f ${patchlist}
      echo "Patch: ${patchname}"
   else
      echo "Patch: Nothing to be done"
   fi
   cd ${__here}
}


## Find remote branch to push on
# * master: base version is at HEAD of master
# * existing M.m0-branch: base version is at HEAD of existing M.m0-branch ${remotetag0}
# * existing M.m1-branch: base version is at HEAD of existing M.m1-branch ${localtag1}
# * New M.m1-branch: M1.m-branch ${localtag1} does not exists
# * New M.m0.f-branch: new branch of an older version, base version is NOT HEAD of existing M1.m-branch ${localtag1}
get_remote_branch() {
   remoteurl=$1
   remotetag0=$2
   localtag1=$3

   remotehash0="$(git ls-remote --tags ${remoteurl} ${remotetag0} | awk '{ print $1}')"
   if [[ "x${remotehash0}" == "x" ]] ; then
      remotehash0="$(git ls-remote --heads ${remoteurl} ${remotetag0} | awk '{ print $1}')"
   fi

   #TODO: should we allow push to master? For sure local master is mig's master not component's master, we would need to find a way around it...
   # mybranch1=master
   # remotehash1="$(git ls-remote --heads ${remoteurl} ${mybranch1} | awk '{ print $1}')"
   # if [[ "x${remotehash0}" == "x${remotehash1}" ]] ; then
   #    # Case [HEAD of master branch]
   #    echo ${mybranch1}
   #    return 0
   # fi

   if [[ ${remotetag0%-branch} != ${remotetag0} ]] ; then
      mybranch1=${remotetag0}
   else
      mybranch1=${remotetag0%.*}-branch
   fi
   remotehash1="$(git ls-remote --heads ${remoteurl} ${mybranch1} | awk '{ print $1}')"
   if [[ "x${remotehash0}" == "x${remotehash1}" ]] ; then
      # Case [HEAD of master branch]
      echo ${mybranch1}
      return 0
   fi

   if [[ ${localtag1%.*}-branch != ${mybranch1} ]] ; then
      mybranch1=${localtag1%.*}-branch
      remotehash1="$(git ls-remote --heads ${remoteurl} ${mybranch1} | awk '{ print $1}')"
   fi

   if [[ "x${remotehash0}" == "x${remotehash1}" ]] ; then
      # Case [Existing M.m1-branch]
      echo ${mybranch1}
   elif [[ "x${remotehash1}" == "x" ]] ; then
      # Case [New M.m1-branch]
      mybranch1=${localtag1%.*}-branch
   else
     # Last resort Case [New M.m0.f-branch]: new branch off an older version
      mybranch1=${remotetag0}-branch
      cat 1>&2 <<EOF
WARNING: Pushing code off an older version (not HEAD of a branch)
         ${remoteurl}  ${mybranch1}
         Tag [${localtag1}] might conflict with an existing one
         This should be merged/rebased on top of an existing branch [e.g. ${localtag1%.*}-branch]
EOF
   fi

   echo ${mybranch1}
}


if [[ ${_push} == 1 && ${_tag} == 1 ]] ; then
   echo > DEPENDENCIES2
fi

## Split components into own branch, optionally push or create patch-set
mybranch0="$(git symbolic-ref --short HEAD)"
for item in $(getcomplist); do
   name=${item%%=*} ; rt=${item#*=} ; remoteurl=${rt%/*} ; remotetag0=${rt##*/}
   if [[ ! -d ${name} ]] ; then
      cat 1>&2 <<EOF
ERROR: cannot find ${name}
       you may either link ${name} dir
          ln -s /PATH/TO/${name} ${name}
       or remove ${name} it from the DEPENDENCIES file
       then re-run this script
EOF
      exit 1
   fi

   # remoteurl="ssh://armnsch@localhost/users/dor/armn/sch/Data/ords/big_tmp/gem-monorepos/barerepos/${name}.git"

   if [[ ${_clean} == 1 ]] ; then continue ; fi

   version="$(get_version ${name})"
   localtag1=${name}_${version}
   mybranch1="$(get_remote_branch ${remoteurl} ${remotetag0} ${localtag1})"

   echo "==== ${name} = ${mybranch1} : ${localtag1} [$remoteurl / ${remotetag0}]"
   if [[ ${_mydry} == 1 ]] ; then
      continue
   fi

   set -x
   git remote rm ${name} 2>/dev/null #TODO: should we keep this
   git remote add ${name} ${remoteurl} #--no-tags
   git fetch ${name}
   git subtree split -P ${name} -b ${mybranch1}
   git checkout ${mybranch1}

   if [[ ${_tag} == 1 ]] ; then git tag ${_force} ${localtag1} ; fi

   if [[ "x${_patch}" != "x" ]] ; then
      # origin="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
      origin="${remotetag0}"
      git2patch ${name} ${version} ${origin} ${destdir:-.}
   fi

   if [[ ${_push} == 1 ]] ; then
      git push ${_force} ${remoteurl} ${mybranch1}
      if [[ ${_tag} == 1 ]] ; then
         git push ${_force} ${remoteurl} ${localtag1}  # refs/tags/${localtag1}
         echo "${name}=${remoteurl}/${localtag1}" >> DEPENDENCIES2
      fi
   fi

   git checkout ${mybranch0}

   ## if [[ ${_push} == 1 ]] ; then git subtree push -P ${name} ${remoteurl} ${mybranch1}
   set +x
done


## Cleanup imported remote branches and tags
if [[ ${_clean} == 1 && ${_mydry} == 0 ]] ; then
   echo "==== Cleanup imported remote branches and tags"
   for item in $(getcomplist); do
      name=${item%%=*} ; rt=${item#*=} ; remoteurl=${rt%/*} ; remotetag0=${rt##*/}
      version="$(get_version ${name})"
      localtag1=${name}_${version}
      mybranch1="$(get_remote_branch ${remoteurl} ${remotetag0} ${localtag1})"
      echo "==== Clean: ${name} ${mybranch1}"
      set -x
      if [[ "x$(git remote | tr '\n' ' '| grep -e "\b${name}\b")" == x ]] ; then
         git remote add ${name} ${remoteurl} #--no-tags
         git fetch ${name}
      fi
      git tag -d $(git ls-remote --tags ${remoteurl} | awk '{ print $2}' | cut -d/ -f3 | tr '\n' ' ')
      git tag -d ${localtag1}    2>/dev/null
      git branch -D ${mybranch1} 2>/dev/null
      git remote rm ${name}
      set +x
   done
   git reflog expire --expire-unreachable=now --all
   git gc --prune=now
fi


## Update DEPENDENCIES list after push
if [[ ${_push} == 1 && ${_tag} == 1 && ${_mydry} == 0 ]] ; then
   echo "==== Update DEPENDENCIES list"
   rm -f DEPENDENCIES
   mv -f DEPENDENCIES2 DEPENDENCIES
   git add DEPENDENCIES
   git commit -m "update DEPENDENCIES"
fi

