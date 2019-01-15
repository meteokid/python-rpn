#!/bin/ksh

arguments=$*
echo $0 $arguments
date
eval `cclargs_lite -D " " $0 \
  -nmlfile        ""   ""   "[path to namelist        ]"\
  -domain         ""   ""   "[domain currently treated]"\
  -check_namelist "0"  "1"  "[to perform check or not ]"\
  -npex           "1"  "1"  "[# of Pes along-x        ]"\
  -npey           "1"  "1"  "[# of Pes along-y        ]"\
  -bin   "${TASK_BIN}"  ""  "[Binaries directory path ]"\
  -verbose        "0"  "1"  "[verbose mode            ]"\
  -cache          ""   ""   "[GEM_cache               ]"\
  ++ $arguments`

set -ex
if [ -n "${bin}" ] ; then
  bin=${bin}/
fi

unset EIGEN_FILE
if [ -n "${cache}" ] ; then
  EIGEN_FILE=${cache}/$(eigen_filename.ksh ${nmlfile})
fi
cat > checkdm.nml <<EOF
&cdm_cfgs
cdm_npex = ${npex}
cdm_npey = ${npey}
cdm_grid_L=.true.
cdm_eigen_S='${EIGEN_FILE}'
/
EOF
lis=checkdmpartlis$$
set +ex
printf "\n RUNNING checkdmpart.ksh \n\n"
checkdmpart.ksh -cfg ${domain} -nmlfile ./checkdm.nml -gemnml ${nmlfile} 1> $lis 2>&1
set -ex
checkdmpart_status='ABORT'
if [ -e checkdmpart_status.dot ] ; then . checkdmpart_status.dot; fi
if [ "${checkdmpart_status}" != 'OK' ] ; then
   printf "\n  Error: Problem with checkdmpart\n\n"
   cat $lis
   exit 1
else
   if [ ${verbose} -gt 0 ] ; then cat $lis ; fi
   printf "\n  checkdmpart is OK\n\n"
fi

flag_err=0
topo_allowed=$(grep topo_allowed checkdmpart_status.dot | sed 's/ //g' | sed 's/"//g' | sed 's/;//g'  | sed 's/=/_/g')
if [ "${topo_allowed}" != "topo_allowed_${npex}x${npey}" ] ; then
   printf "\n  Error: MPI topology NOT allowed\n\n"
   flag_err=1
else
   printf "\n  MAXIMUM number of I/O PES for this configuration is: $(echo ${MAX_PES_IO} | sed 's/^0*//')\n\n"
fi
if [ "${Fft_fast_L}" != 'OK' ] ; then
   printf "\n  Error: This G_ni is NOT FFT but you requested FFT\n\n"
   flag_err=1
fi     
if [ "${SOLVER}" != 'OK' ] ; then
   set +ex
   printf "\n  Error: VERTICAL LAYERING IS INCOMPATIBLE WITH THE TIMESTEP"
   printf "\n         THE SOLVER WILL NOT WORK\n\n"
   flag_err=1
   set -ex
fi
if [ $flag_err -gt 0 ] ; then exit 1 ; fi

rm -f ./checkdm.nml checkdmpart_status.dot $lis status_MOD.dot
