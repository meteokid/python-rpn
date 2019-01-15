#!/usr/bin/env ksh
# -*- coding: utf-8 -*-

cfgFilename=$1
fromVersion=$2
toVersion=$3
verbose=$4
debug=$5
status=0

if [[ x"$verbose" == x"True" || x"$debug" == x"True" ]] ; then
   echo "Updating gem from $fromVersion to $toVersion; $cfgFilename, $verbose, $debug"
fi

#if error ; then status=1 ; fi

exit $status
