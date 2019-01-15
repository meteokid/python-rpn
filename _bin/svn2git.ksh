#!/bin/ksh

url=$1
name=${url##*/}

git svn clone --stdlayout ${url}

cd ${name}

taglist="$(git branch -r | grep /tags/ | tr '\n' ' ')"
for item in ${taglist} ; do

   tag=${item##*/}
   tag2=${name}_${tag#v}

   echo ==== ${tag2} [$tag] [$item]
   set -x
   git checkout -b ${tag2}-branch ${item}
   git tag ${tag2}
   git checkout master
   git branch -D ${tag2}-branch
   set +x

done
