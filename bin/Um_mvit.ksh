#!/bin/ksh
#
#TODO: KEEP OR NOT? not used in model

yin=`echo $1 | grep "YIN" | wc -l`
yan=`echo $1 | grep "YAN" | wc -l`
yyg=$(($yin + $yan))

REP=$2/endstep_misc_files

if [ $yyg -gt 0 ] ; then
  if [ $yin -gt 0 ] ; then
    REP=$REP/YIN
  else
    REP=$REP/YAN
  fi
fi

if [ -s $1 ] ; then
  mkdir -p $REP
  mv $1 $REP
fi

