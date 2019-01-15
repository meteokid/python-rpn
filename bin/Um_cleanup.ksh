#!/bin/ksh
#TODO: KEEP OR NOT? not used in model
arguments=$*

#====> Obtaining the arguments:
eval `cclargs_lite $0 \
     -execdir      ""       ""       "[Execution directory  ]" \
     -entry        "0"      "1"      "[Entry cleanup flag   ]" \
     -model        "0"      "1"      "[Model cleanup flag   ]" \
     -transfer     "0"      "1"      "[Transfer cleanup flag]" \
     -endstep      "0"      "0"      "[Last timestep number ]" \
  ++ $arguments`
#

printf "\n#####################################\n"
printf "########### Um_cleanup.ksh  ###########\n"
printf "#####################################\n"

set -x

# Check for a valid execution directory
if [ -z "${execdir}" ] ; then
  printf "Um_cleanup.ksh requires an '-execdir' argument for cleanup path\n"
  exit 1
fi
if [ ! -d ${execdir} ] ; then
  printf "Um_cleanup.ksh encountered an invalid execdir: ${execdir}\n"
  exit 1
fi

# Task setups
nt_taskdir=${execdir}/RUNENT
dm_taskdir=${execdir}/RUNMOD
xf_taskdir=${execdir}/XFERMOD

# Clean up in entry component on request
if [ ${entry} -gt 0 -a -d ${nt_taskdir} ] ; then
  rm -fr ${nt_taskdir}/output/*
fi

# Clean up in model component on request
if [ ${model} -gt 0 -a -d ${dm_taskdir} ] ; then
  transferred=`true_path ${xf_taskdir}/input/last_step_${endstep}`
  if [ -d "${transferred}" ] ; then
    for file in `find -L ${transferred} -type f -print` ; do
      rm -f ${dm_taskdir}/output/`basename $file`
    done
  fi
fi

# Clean up in transfer component on request
if [ ${transfer} -gt 0 -a -d ${xf_taskdir} ] ; then
  printf "No cleanup for transfer component\n"
fi

    
      
