#!/bin/ksh

code=${3:-0}

if [ $code -eq 0 ] ; then
  string=`echo $1 | tr "[a-z]" "[A-Z]"`
  grep -i $string $2 | tr "[a-z]" "[A-Z]" | sed "s-\(.*\)\(${string} *=.*\)-\2-" \
                     | sed "s/,/ /g" | sed "s/=/ /g" | awk '{print $2}'          \
                     | sed "s/'//g" | sed 's/"//g'
fi

if [ $code -eq 1 ] ; then
  string=`grep -i $1 $2`
  s1=${string##*${1}}
  cnt=1
  while [ cnt -gt 0 ] ;do
    s1=`echo $s1 | sed 's/\(.*\)\(,.*\)/\1/'`
    cnt=`echo $s1 | grep "," | wc -m`
  done
  echo $s1 | sed 's/=//' | sed "s/'//g" | sed "s/ //g"
fi





