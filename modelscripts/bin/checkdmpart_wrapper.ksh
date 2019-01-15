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

flag_topo=0
if [ ${check_namelist} -gt 0 ] ; then
  BIN=$(which checkdmpart_${BASE_ARCH}.Abs)
  /bin/rm -f ./gem_settings.nml
  ln -s ${nmlfile} ./gem_settings.nml
  export Ptopo_npex=${npex:-1} ; export Ptopo_npey=${npey:-1}
  ${bin}r.run_in_parallel -pgm ${BIN} -npex 1 -npey 1 | tee checkdmpart$$
  set +ex
  topo_ok=$(grep "CHECKDMPART IS OK" checkdmpart$$ | wc -l)
  if [ $topo_ok -lt 1 ] ; then
    printf "\n  Error: Illegal domain partitionning ${npex}x${npey} \n\n"
    flag_topo=1
  else
     printf "\n CHECKDMPART IS OK: domain partitionning allowed\n\n"
     max_io_pes=$(grep MAX_PES_IO checkdmpart$$ | awk '{print $NF}')
     printf "\n MAXIMUM IO_PES= ${max_io_pes} \n\n"
  fi            
  rm -f checkdmpart$$ ./gem_settings.nml
fi
if [ ${flag_topo} -gt 0 ] ; then exit 1 ; fi
