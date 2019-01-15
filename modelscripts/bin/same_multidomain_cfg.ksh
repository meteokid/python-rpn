#!/bin/ksh
#
arguments=$*
eval `cclargs_lite -D " " $0 \
  -f        ""          ""       "[]"\
  -deb      ""          ""       "[]"\
  -fin      ""          ""       "[]"\
  -rst      ""          ""       "[]"\
  -bkp      ""          ""       "[]"\
  -freq     ""          ""       "[]"\
  -abortf    "samecfg_abort" "samecfg_abort" "[abort file]"\
  ++ $arguments`

# This set -ex is mandatory
set -ex

abort_file=${TASK_WORK}/${abortf}-$$
touch ${abort_file}

RDEB=$(rpy.nml_get -f ${f} step/Fcst_start_S        2> /dev/null | sed 's/"//g' | sed "s/'//g")
RFIN=$(rpy.nml_get -f ${f} step/Fcst_end_S          2> /dev/null | sed 's/"//g' | sed "s/'//g")
RRST=$(rpy.nml_get -f ${f} step/Fcst_rstrt_S        2> /dev/null | sed 's/"//g' | sed "s/'//g")
BKUP=$(rpy.nml_get -f ${f} step/Fcst_bkup_S         2> /dev/null | sed 's/"//g' | sed "s/'//g")
FREQ=$(rpy.nml_get -f ${f} gem_cfgs/Out3_postfreq_s 2> /dev/null | sed 's/"//g' | sed "s/'//g")

if [ "${deb}x" != "${RDEB}x" ] ; then
    printf "\n ### same_multidomain_cfg.ksh: ERROR Fcst_start_S must be the same for all domains\n\n"
    exit 1
fi
if [ "${fin}x" != "${RFIN}x" ] ; then
    printf "\n ### same_multidomain_cfg.ksh: ERROR Fcst_end_S must be the same for all domains\n\n"
    exit 1
fi
if [ "${rst}x" != "${RRST}x" ] ; then
    printf "\n ### same_multidomain_cfg.ksh: ERROR Fcst_rstrt_S must be the same for all domains\n\n"
    exit 1
fi
if [ "${bkp}x" != "${BKUP}x" ] ; then
    printf "\n ### same_multidomain_cfg.ksh: ERROR Fcst_bkup_S must be the same for all domains\n\n"
    exit 1
fi
if [ "${freq}x" != "${FREQ}x" ] ; then
    printf "\n ### same_multidomain_cfg.ksh: ERROR Out3_postfreq_s must be the same for all domains\n\n"
    exit 1
fi

/bin/rm -f ${abort_file}

