#!/bin/ksh

arguments=$*
. r.entry.dot

eval `cclargs_lite $0 \
  -npex     "1"            "1"         "[Domain partitioning along-x]" \
  -npey     "1"            "1"         "[Domain partitioning along-y]" \
  -nomp     "1"            "1"         "[Number of OMP threads      ]" \
  -ndom     "1"            "1"         "[Number of domains          ]" \
  -inorder  "0"            "1"         "[Order listing              ]" \
  -barrier  "0"            "0"         "[DO NOT run binary          ]" \
  -nompi    "0"            "1"         "[USE mpi if = 0             ]" \
  -debug    "0"            "1"         "[Debug session if debug=1   ]"\
  -_status  "ABORT"        "ABORT"     "[return status              ]" \
  -_endstep ""             ""          "[last time step performed   ]" \
  ++ $arguments`

. r.call.dot ${TASK_BIN}/Um_cmclog.ksh -mod 1

set ${SETMEX:-+ex}

export OMP_NUM_THREADS=$nomp
if [ -n "$_CMC_LOGFILE" ] ; then export CMC_LOGFILE=$_CMC_LOGFILE ; fi

npe_total=$(( npex * npey * ndom ))

printf "\n Running `readlink ${TASK_BIN}/ATM_MOD.Abs` on $npe_total ($npex x $npey) PEs:\n"
printf " OMP_NUM_THREADS=$OMP_NUM_THREADS\n\n"
printf " ##### UM_TIMING: Um_model.ksh STARTING AT: `date`\n"

if [ $barrier -gt 0 ] ; then

  printf "GEM_Mtask 1\n"
  r.barrier
  printf "GEM_Mtask 2\n"
  r.barrier
  printf "\n =====> Um_model.ksh CONTINUING after last r.barrier\n\n"

elif [ $nompi -eq 0 ] ; then

  if [ ${inorder} -gt 0 ] ; then INORDER="-inorder -tag"; fi
  CMD="${TASK_BIN}/r.mpirun -pgm ${TASK_BIN}/ATM_MOD.Abs -npex $((npex*npey)) -npey $ndom $INORDER -minstdout 5 -nocleanup"
  if [[ x$debug != x0 ]] ; then
#     #CMD="${CMD} -gdb" #Needs r.run_in_parallel_1.1.12
#     preexec="$(which gdb) -batch -ex run -ex where"
#     DEBUGGER=${DEBUGGER:-$preexec}
#      CMD="${CMD} -preexec '$DEBUGGER'"
    CMD="${CMD} -gdb"
  fi
  printf "\n EXECUTING: $CMD\n\n"
  $CMD

else

  printf "${TASK_BIN}/ATM_MOD.Abs\n"
  if [ $debug -eq 0 ] ; then
    ${TASK_BIN}/ATM_MOD.Abs
  else
    ${DEBUGGER:-r.pgdbg -dbx} ${TASK_BIN}/ATM_MOD.Abs -I ${MODEL_SOURCE_CODE}
  fi

fi

set +ex
printf " ##### UM_TIMING: Um_model.ksh ENDING AT: `date`\n"

. r.call.dot ${TASK_BIN}/Um_cmclog.ksh -mod 2 -CMC_LOGFILE $_CMC_LOGFILE

set ${SETMEX:-+ex}
export CMC_LOGFILE=$_CMC_LOGFILE

status_file=./status_MOD.dot
nb_abort=0
nb_restart=0
nb_end=0

printf " ##### UM_TIMING: POST Um_model.ksh STARTING AT: `date`\n"
if [[ "x${GEM_NDOMAINS}" != "x" ]] ; then
   cfglist=""
   start=$(echo ${GEM_NDOMAINS} | cut -d : -f1)
   end=$(echo ${GEM_NDOMAINS} | cut -d : -f2)
   idom=${start}
   while [ ${idom} -le ${end} ] ; do
      cfglist="${cfglist} cfg_$(printf "%4.4d" ${idom})"
      idom=$((idom+1))
   done
else
   cfglist=cfg_*
fi

for i in ${cfglist} ; do

  fn=${TASK_OUTPUT}/${i}/${status_file}
  if [ -s ${fn} ] ; then
    . ${fn}
    printf "STATUS_FROM_DOMAIN: ${i} $_status\n"
    if [ "$_status" = "ABORT" ] ; then ((nb_abort=nb_abort+1))    ; fi
    if [ "$_status" = "RS"    ] ; then ((nb_restart=nb_restart+1)); fi
    if [ "$_status" = "ED"    ] ; then ((nb_end=nb_end+1))        ; fi
  else
    _status="ABORT"
    ((nb_abort=nb_abort+1))
  fi

# Deal with special files: time_series.bin, zonaux_* and *.hpm*
# Files will be transfered from ${TASK_WORK}/$i to ${TASK_OUTPUT}/$i

  cd ${i}  
  /bin/rm -rf busper
  if [ "$_status" = "ED" ] ; then
    REP=${TASK_OUTPUT}/${i}/`cat ${TASK_OUTPUT}/${i}/output_ready_MASTER | grep "\^last" | cut -d " " -f3 | sed 's/\^last//g'`/endstep_misc_files
    mkdir -p ${REP}
    if [ -d YIN ] ; then
      mkdir ${REP}/YIN       ${REP}/YAN   2> /dev/null || true
      mv YIN/time_series.bin* ${REP}/YIN   2> /dev/null || true
      mv YAN/time_series.bin* ${REP}/YAN   2> /dev/null || true
      mv YIN/[0-9]*/*.hpm    ${REP}/YIN   2> /dev/null || true
      mv YAN/[0-9]*/*.hpm    ${REP}/YAN   2> /dev/null || true
    else
      mv time_series.bin* ${REP} 2> /dev/null || true
      mv [0-9]*/*.hpm    ${REP} 2> /dev/null || true
    fi
    liste_busper=`find ./ -type f -name "BUSPER4spinphy*"`
    fn=`echo $liste_busper| awk '{print $1}'`
    if [ -n "${fn}" ] ; then
      mkdir -p ${REP}
      fn=`basename $fn`
      tar cvf ${REP}/${fn}.tar $liste_busper
      /bin/rm -f $liste_busper
    fi
  fi
  cd ../
  
done
printf " ##### UM_TIMING: POST Um_model.ksh ENDING AT: `date`\n"

if [ $nb_abort -gt 0 ] ; then 
  _status="ABORT"
else
  if [ $nb_restart -eq $ndom ] ; then
    _status="RS"
  else
    if [ $nb_end -eq $ndom ] ; then
      _status="ED"
    fi
  fi
fi
set +ex

# End of task
. r.return.dot
