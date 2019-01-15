#!/bin/ksh

## rm -rf mydir ; mkdir mydir && cd mydir && git init && ../_bin/migimportall.ksh -v

DESC='Git import components using subtree'
USAGE="USAGE: ${MYSELF} [-v] [-i VERSIONS_FILE] [-U BASE_URL]"

usage_long() {
   toto=$(echo -e $USAGE)
   more <<EOF
$DESC

$toto

Options:
    -h, --help     : print this help
    -v, --verbose  : verbose mode
    -i, --infile    : name of file describing all components/versions to be imported
                     One version per line with de following format
                     branch=BRANCH_NAME; tag=TAG_NAME ; components="REPOS1/TAG1 REPOS2/TAG2 ..."
    -U, --urlbase  : base URL for location of all component's repository
                     For the moment, all components must share a comment URL base
                     REPOS1... in the infile will be appended to this URL
        --noadd    : Do not add ../_bin ../_share to repos
        --nosquash : do not squash commits from remote repos
        --keeptags : keep remote repos tags
        --keepbranches : keep remote repos branches

   This command must be run from an already initiliez Git repository
   You may create a simple one with:
      mkdir mydir && cd mydir && git init
EOF
}

myecho() {
   if [[ ${verbose} -ge $1 && ${quiet} == 0 ]] ; then
      shift
      printf "$@\n" 2>&1
   fi
}
myechoerror() {
   if [[ ${quiet} == 0 ]] ; then
      printf "ERROR: $@\n" 1>&2
   fi
}

squash=--squash
reposbase=git@gitlab.science.gc.ca:MIG
versionsfile=./_share/migversions.txt
mygit=git
# mygit=echo
# set -x
verbose=0
quiet=0
adddir="_bin _share"
keeptags=0
keepbranches=0
while [[ $# -gt 0 ]] ; do
   case $1 in
      (-h|--help) usage_long; exit 0;;
      (-v|--verbose) ((verbose=verbose+1));;
      (-i|--infile) ;;
      (-U|--urlbase) ;;
      (--nosquash) squash="";;
      (--noadd) adddir="";;
      (--keeptags) keeptags=1;;
      (--keepbranches) keepbranches=1;;
      (--) shift ; previous=""; break;;
      (-*) myechoerror "Option Not recognized: ${1}"; usage_long; return 1;;
      (*)
         case ${previous} in
            (-i|--infile) versionsfile=${1};;
            (-U|--urlbase) reposbase=${1};;
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



TMPDIR2=${TMPDIR}/$$
mkdir -p ${TMPDIR2}

if [[ ! -d .git/ ]] ; then
   cat 1>&2 <<EOF
ERROR: not a git repository.
       Try to do "git init"
EOF
   exit 1
fi

${mygit} checkout master > /dev/null 2>&1 || true

for item in ${adddir} ; do
   if [[ -d ../${item} && ! -d ${item} ]] ; then
      cp -R ../${item}/ .
   fi
done
for item in _bin/.setenv.dot _share/README*md _share/.gitignore _share/Makefile.user*mk ; do
   if [[ -f ${item} && ! -f ${item##*/} ]] ; then
      mv ${item} ${item##*/}
   fi
done
# for item in _bin/.setenv.dot ; do
#    if [[ -f ${item} && ! -f ${item##*/} ]] ; then
#       ln -s ${item} ${item##*/}
#    fi
# done

${mygit} add .
${mygit} commit -a -m 'Base import system update'


newtags="$(${mygit} tag | tr '\n' ' ')"
for item in $(cat ${versionsfile} | tr ' ' ':' | tr '\n' ' '); do
   if [[ "x$(echo ${item} | cut -c1)" == "x#" || "x$(echo ${item} | sed 's/://g')" == "x" ]] ; then
      continue
   fi
   unset tag branch components
   item="$(echo ${item} | tr ':' ' ')"
   eval "${item}"  #TODO: double eval to only keep known vars
   # if [[ "x${tag}" == "x" || "x${branch}" == "x" || "x${components}" == "x" ]] ; then
   if [[ "x${branch}" == "x" || "x${components}" == "x" ]] ; then
      myecho 1 "WARNING: Skipping - branch/tag/components not defined in: ${item}"
      continue
   fi
   myecho 1 "==== ${branch} / ${tag} : ${components}"

   ## Skip already imported versions
   if [[ "x${tag}" != "x" ]] ; then
      if [[ "x$(${mygit} tag | grep ${tag})" != "x" ]] ; then
         myecho 1 "Skipping ${tag}, already imported"
         continue
      fi
   fi

   if [[ "x$(${mygit} branch | grep ${branch})" == "x" ]] ; then
      ${mygit} checkout -b ${branch}
   else
      ${mygit} checkout ${branch}
   fi

   echo > DEPENDENCIES
   for comp1 in $(echo ${components} | tr ':' ' ') ; do
      compname=${comp1%%/*}
      comptag=${comp1#*/}
      remoteurl=${reposbase}/${compname}.git
      if [[ "${compname}" == "python-rpn" ]] ; then
         compname=rpnpy
      fi
      remotename=${compname}
      echo "${compname}=${remoteurl}/${comptag}" >> DEPENDENCIES
   done
   ${mygit} add DEPENDENCIES
   ${mygit} commit -a -m "add DEPENDENCIES for ${branch} / ${tag}"

   for comp1 in $(echo ${components} | tr ':' ' ') ; do
      compname=${comp1%%/*}
      comptag=${comp1#*/}
      remoteurl=${reposbase}/${compname}.git
      if [[ "${compname}" == "python-rpn" ]] ; then
         compname=rpnpy
      fi
      remotename=${compname}
      cmd=add; [[ -d ${compname} ]] && cmd=pull || true

      myecho 1 "---- ${branch} / ${tag} : ${compname} ${comptag}"

      if [[ "x$(${mygit} remote -v | grep ${remotename})" == "x" ]] ; then
         ${mygit} remote add ${remotename} ${remoteurl} #--no-tags
         ${mygit} fetch --tags ${remotename}
      fi

      tagfile=${TMPDIR2}/${branch}_${compname}.txt
      logfile=${TMPDIR2}/${branch}_${compname}_${comptag}.log
      echo "subtree_pull: tag=${comptag}; url=${remoteurl}; dir=${compname}" > ${logfile}
      if [[ -f ${tagfile} ]] ; then
         comptag0="$(cat ${tagfile})"
         if [[ "${comptag0}" != "${comptag}" ]] ; then
            echo >> ${logfile}
            ${mygit} log --format=medium --date=iso ${comptag0}..${comptag} >> ${logfile}
         fi
      fi
      echo ${comptag} > ${tagfile}

      ${mygit} subtree ${cmd} -P ${compname} ${squash} ${remotename} ${comptag} -m "$(cat ${logfile})"

   done
   if [[ "x${tag}" != "x" ]] ; then
      newtags="${newtags} ${tag}"
      ${mygit} tag -f ${tag}
   fi
done


## Cleanup branches
if [[ ${keepbranches} == 0 ]] ; then
   for item in $(cat ${versionsfile} | tr ' ' ':' | tr '\n' ' '); do
      item="$(echo ${item} | tr ':' ' ')"
      eval "${item}"
      for comp1 in $(echo ${components} | tr ':' ' ') ; do
         remotename=${comp1%%/*}
         if [[ "${remotename}" == "python-rpn" ]] ; then
            remotename=rpnpy
         fi
         ${mygit} remote remove ${remotename} 2>/dev/null || true
      done
   done
fi

## Cleanup tags
if [[ ${keeptags} == 0 ]] ; then
   for tag1 in $(${mygit} tag); do
      if [[ "x$(echo ${newtags} | grep ${tag1})" == "x" ]] ; then
         ${mygit} tag -d  ${tag1}
      fi
   done
fi

set -x
## Garbage collection
${mygit} reflog expire --expire-unreachable=now --all
${mygit} gc --prune=now


