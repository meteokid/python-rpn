#!/bin/ksh
#
sod=$1
#
if [ "$sod" = "single" ] ; then
  if [ -e configexp.dot.cfg ] ; then
    printf "\n  ############ ERROR ERROR ############\n"
    printf "\n  Batch configuration file configexp.dot.cfg is no longer\n"
    printf "  in use. You must now use file configexp.cfg\n\n"
    printf "  ----- ABORT -----\n\n"
    exit 1
  fi

  if [ ! -e configexp.cfg ] ; then
    printf "\n  ############ ERROR ERROR ############\n"
    printf "\n  Batch configuration file configexp.cfg is unavailable\n"
    printf "  ----- ABORT -----\n\n"
    exit 1
  else
    set -e
    . ./configexp.cfg
    set +e
  fi

  if [ ! -e gem_settings.nml ] ; then
    printf "\n  ############ ERROR ERROR ############\n"
    printf "\n  GEM configuration file gem_settings.nml is unavailable\n"
    printf "  ----- ABORT -----\n\n"
    exit 1
  else
    cnt=`grep "&ptopo" gem_settings.nml | wc -l`
    if [ $cnt -gt 0 ] ; then
      printf "\n  ############ ERROR ERROR ############\n"
      printf "\n  Namelist &ptopo no longer in use.\n"
      printf "  Instead use resources configuration variables\n"
      printf "  GEM_ptopo, GEM_smt, GEM_in_block, GEM_out_block\n"
      printf "  and GEM_bind in file configexp.cfg\n\n"
      printf "  ----- ABORT -----\n\n"
      exit 1
    fi
  fi
fi

if [ "$sod" = "multi" ] ; then
  if [ ! -e BATCH_config.cfg ] ; then
    printf "\n  ############ ERROR ERROR ############\n"
    printf "\n  Batch configuration file BATCH_config.cfg is unavailable\n"
    printf "  ----- ABORT -----\n\n"
    exit 1
  else
    set -e
    . ./BATCH_config.cfg
    set +e
  fi
fi

exit 0
