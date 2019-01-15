#!/bin/bash

reverse=0
getall=0
if [[ "x$1" == "x-r" ]] ; then
   reverse=1
   shift
fi
if [[ "x$1" == "x-a" ]] ; then
   getall=1
   shift
fi

__ROOT=${1:-$(pwd)}

depnamelist=":"
__get_deps() {
   # echo "==== Getting dep list from: $1"
   deps="$2"
   for i in ${deps} ; do \
      name=${i%%=*} ; rt=${i#*=} ; repos=${rt%@*} ; tag=${rt##*@}
      # echo name=${i%%=*} repos=${rt%@*} tag=${rt##*@}
      if [[ x$getall == x1 || -d ${__ROOT}/${name} ]] ; then
         if [[ "x$(echo ${depnamelist} | grep :${name}:)" == "x" ]] ; then
            if [[ $reverse == 0 ]] ; then
               depnamelist="${depnamelist}${name}:"
            else
               depnamelist="${name}:${depnamelist}"
            fi
            __get_deps ${name} "$(cat ${__ROOT}/${name}/DEPENDENCIES 2>/dev/null | sed 's/ //g' | tr '\n' ' ')"
         fi
      fi
      if [[ ! -d ${__ROOT}/${name} ]] ; then
         echo "WARNING: missing dep in $1 dir : ${name}" 1>&2
      fi
   done
}

__get_deps "TOP" "$(cat DEPENDENCIES 2>/dev/null | sed 's/ //g' | tr '\n' ' ')"
echo ${depnamelist} | tr ':' ' '
