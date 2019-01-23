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
DESC="Update every components' DEPENDNECIES.mig.bndl file from other components VERSION file."
USAGE="USAGE: ${MYSELF} [-v]"

usage_long() {
   toto=$(echo -e $USAGE)
   more <<EOF
$DESC

$toto

Options:
    -h, --help     : print this help
    -v, --verbose  : verbose mode
        --gitdep   : also update the tags in MIG's main DEPENDENCIES file
        --migver   : also add entry to the _share/migversions.txt file
        --check    : Check that components and MIG VERSION numbers were 
                     updated if changes were commited

   This command must be run from the mig repository base dir.
EOF
}

myecho() {
   if [[ ${_verbose} -ge $1 && ${_quiet} == 0 ]] ; then
      shift
      printf "$@\n" 2>&1
   fi
}
myechoerror() {
   if [[ ${_quiet} == 0 ]] ; then
      printf "ERROR: $@\n" 1>&2
   fi
}
myechowarn() {
   if [[ ${_verbose} -ge $1 && ${_quiet} == 0 ]] ; then
      shift
      printf "WARNING: $@\n" 1>&2
   fi
}

_quiet=0
_verbose=0
_gitdep=0
_migver=0
_check=0
while [[ $# -gt 0 ]] ; do
   case $1 in
      (-h|--help) usage_long; exit 0;;
      (-v|--verbose) ((_verbose=_verbose+1));;
      (--gitdep) _gitdep=1;;
      (--migver) _migver=1;;
      (--check) _check=1;;
      (-*) myechoerror "Option Not recognized: ${1}"; usage_long; return 1;;
      (*)
         case ${previous} in
            (*)
               myechoerror "Option Not recognized: ${1}"
               usage_long
               return 1;;
         esac;;
   esac
   if [[ x${1#-} != x${1} ]] ; then
      previous=${1}
   fi
   shift
done

## Get component version number
getversion() {
   _item=${1}
   if [[ -f ${_item}/VERSION ]] ; then
      _itemv="$(cat ${_item}/VERSION)"
   elif [[ -f ${!_item}/VERSION ]] ; then
      _itemv="$(cat ${!_item}/VERSION)"
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
   RDECOMPONENTS="$(cat */.name | tr '\n' ' ')"
   if [[ "x${RDECOMPONENTS}" == "x" ]] ; then
      if [[ -f DEPENDENCIES ]] ; then
         RDECOMPONENTS="$(cat DEPENDENCIES | tr '\n' ' ')"
      else
         echo "ERROR: cannot find DEPENDENCIES file" 1>&2
         exit 1
         # export RDECOMPONENTS="$(find . -maxdepth 1 -type d | cut -d/ -f2 | grep -e '^[a-zA-Z]\+' | grep -v GEM_cfg | tr '\n' ' ')"
         #TODO: define RDEDEPS from RDECOMPONENTS + provided base URL + version info
      fi
   fi
   if [[ "x${RDECOMPONENTS}" == "x" ]] ; then
      echo "ERROR: no known DEPENDENCIES" 1>&2
      exit 1
   fi
   echo ${RDECOMPONENTS}
}

## Get last tag
getlasttag() {
   hash0="$(git rev-parse HEAD)"
   lasttag="$(git describe --tags --abbrev=0)"
   # hash1="$(git rev-list -n 1 ${lasttag})"
   hash1="$(git rev-parse ${lasttag})"
   if [[ "x${hash0}" == "x${hash1}" ]] ; then
      lasttag="$(git describe --tags --abbrev=0 HEAD^)"
   fi
   echo ${lasttag}
}

##

DEPFILENAME=DEPENDENCIES.mig.bndl
DEPFILENAMEGIT=DEPENDENCIES
MIGVERFILE=_share/migversions.txt
tempsed1=${TMPDIR}/${DEPFILENAME}-sed1-$$
tempsed2=${TMPDIR}/${DEPFILENAME}-sed2-$$
tmpfile=${TMPDIR}/${DEPFILENAME}-$$

RDECOMPONENTS="$(getcomplist)"
myecho 1 "RDECOMPONENTS='${RDECOMPONENTS}'"

if [[ ${_check} == 1 ]] ; then
   _status=0
   lasttag="$(getlasttag)"
   for comp in . ${RDECOMPONENTS} ; do
      verdiff="$(git diff --name-only ${lasttag} -- ${comp}/VERSION)"
      if [[ "x${verdiff}" == "x" ]] ; then
         codediff="$(git diff --name-only ${lasttag} -- ${comp}/ | wc -l)"
         if [[ ${codediff} -gt 0 ]] ; then
            myechowarn 0 "'${comp}' Component content changed (from ${lasttag}) but not the VERSION"
            _status=1
         else
            myecho 2 "${comp}: VERSION and content unchanged (from ${lasttag})"
         fi
      else
         #myecho 2 "${comp}: VERSION was updated (from ${lasttag}): $(git diff ${lasttag} -- ${comp}/VERSION)"
         myecho 2 "${comp}: VERSION was updated (from ${lasttag})"
      fi
   done
   if [[ ${_status} != 0 ]] ; then
      echo "---- ABORTING ----" 1>&2
      exit ${_status}
   fi
fi

rm -f ${tempsed1} ${tempsed2}
for comp in ${RDECOMPONENTS} ; do
   version="$(getversion ${comp})"
   verv=${version##*/}
   verx=""
   if [[ ${verv} != ${version} ]] ; then
      verx=${version%/*}/
   fi
   cat >> ${tempsed1} <<EOF
s:\(ENV\|GEM\|SCM\)/\(d/\)*\(x/\)*\(${comp}\)\(/\|/${comp}_\)[^_/][^_/]*:\1/\2${verx}\4\5${verv}:
EOF
   cat >> ${tempsed2} <<EOF
s:\(${comp}=.*/${comp}\)_[^_/][^_/]*:\1_${verv}:
EOF
done

for comp in ${RDECOMPONENTS} ; do
   if [[ ! -f ${comp}/${DEPFILENAME} ]] ; then
      myecho 1 "# No such file: ${comp}/${DEPFILENAME}"
   else
      myecho 1 "# Updating: ${comp}/${DEPFILENAME}"
      sed -f ${tempsed1} ${comp}/${DEPFILENAME} > ${tmpfile}
      mv ${tmpfile} ${comp}/${DEPFILENAME}
   fi
done

if [[ ${_gitdep} == 1 ]] ; then
   if [[ ! -f ${DEPFILENAMEGIT} ]] ; then
      myecho 1 "# GitDep - No such file: ${DEPFILENAMEGIT}"
   else
      myecho 1 "# Updating: ${DEPFILENAMEGIT}"
      sed -f ${tempsed2} ${DEPFILENAMEGIT} > ${tmpfile}
      mv ${tmpfile} ${DEPFILENAMEGIT}
   fi
fi

if [[ ${_migver} == 1 ]] ; then
   if [[ ! -f ${MIGVERFILE} ]] ; then
      myecho 1 "# MIGver - No such file: ${MIGVERFILE}"
   else
      myecho 1 "# Updating: ${MIGVERFILE}"
      migbranch="$(git symbolic-ref --short HEAD)"
      migver=mig_$(getversion .)
      migtag=mig_${migver##*/}

      # isver="$(cat ${MIGVERFILE} | tr ' ;' '::'| grep branch=${migbranch}: | grep tag=${migtag}:)"
      isver="$(cat ${MIGVERFILE} | tr ' ;' '::'| grep tag=${migtag}:)"
      if [[ "x${isver}" != "x" ]] ; then
         #TODO: might want to automate check if consistent or different
         myechowarn 0 "'tag=${migtag}' already exists in file ${MIGVERFILE}"
      else
         components=""
         for comp in ${RDECOMPONENTS} ; do
            comp1=${comp}
            version="$(getversion ${comp})"
            verv=${version##*/}
            if [[ ${comp1} == "rpnpy" ]] ; then
               comp1="python-rpn"
            elif [[ ${comp1} == "migdep" ]] ; then
               continue
            fi
            components="${components# } ${comp1}/${comp}_${verv}"
         done
         cat >> ${MIGVERFILE} <<EOF
branch=${migbranch}; tag=${migtag} ; components="${components# }"
EOF
      fi
   fi
fi

rm -f ${tempsed1} ${tempsed2} ${tmpfile}
