#!/bin/ksh

scriptname=$(basename $0)
arguments=$*

if [ -z "${ATM_MODEL_BNDL}" ] ; then
  printf "\n  NO GEM ENVIRONMENT -- ABORT\n\n"
  exit 1
fi
if [ -z "${storage_model}" ] ; then
  printf "\n storage_model is UNDEFINED-- ABORT\n\n"
  exit 1
fi

# This next variable will eventually point to $gemdyn/VERIF some day
# root_install=$gemdyn/VERIF
# but for now:
root_install=/users/dor/armn/mid/home/VERIF

export PATH=${PATH}:${root_install}/library

# Ensure CDPATH does not screw with our cd calls
unset CDPATH
# Ensure GREP_OPTIONS does not screw with our grep calls
unset GREP_OPTIONS

LANG=en_CA.UTF-8

USE_COLOR='y'

#source library/output_format.sh
. output_format.sh

#Interface: unset all controls
unset listing outputs summary email giveup exp code buildexp intt nocolor nb_levels mach launch orig_cfg 
unset CONFIG_FILE

#Interface: obtain user specified experiment name
while [[ $# -gt 0 ]] ; do
   case $1 in
      (-exp) [[ x$(echo $2 | cut -c1) != x-   && \
                x$(echo $2 | cut -c1) != x ]] && \
                exp=$2 && shift ;;
   esac
   shift
done

version=$(basename ${ATM_MODEL_BNDL} | sed 's/\.//g')
VERSION_ref=v_4.8
#$(basename ${ATM_MODEL_BNDL} | cut -d "." -f1-2)

#Interface: retrieve options of previous execution to possibly modify defaults
if [ -n "${exp}" ] ; then
  saved_cfg_file=v${version}_${exp}
  if [ -e ${HOME}/.${scriptname}_rc/${saved_cfg_file}.cfg ] ; then
    set -x
    . r.lcfg.dot $saved_cfg_file
    set +x
    Old_exp=1
    CONFIG_FILE=${HOME}/.${scriptname}_rc/${saved_cfg_file}.cfg
  else
    saved_exp=$exp
    if [ -e ${HOME}/.${scriptname}_rc/last.cfg ] ; then
      set -x
      . r.lcfg.dot last
      set +x
    fi
    exp=$saved_exp
    Old_exp=0
  fi
else
  if [ -e ${HOME}/.${scriptname}_rc/last.cfg ] ; then
    set -x
    . r.lcfg.dot last
    set +x
    Old_exp=1
    CONFIG_FILE=${HOME}/.${scriptname}_rc/last.cfg
  else
    exp=dbg
    Old_exp=0
  fi
fi

#Interface: obtain user requirements for this execution
saved_cfg_file=v${version}_${exp}
saved_exp=$exp

eval `cclargs_lite -D " " $0 \
  -cfgs        "${cfgs}"        "${root_install}/INTEGRATION_testings/$PVERSION_ref}"     "[set an alternate test root directory]"\
  -exp         "${exp}"         "${exp}"      "[experiment name            ]"\
  -expcmp      "${expcmp}"      ""            "[compare experiment names   ]"\
  -code        "${code}"        ""            "[newcode for this experiment]"\
  -nb_levels   "${nb_levels}"   "5"           "[number of levels to tun     ]"\
  -buildexp    "${buildexp}"    "0"           "[gemdev, code copy, compile and load ]"\
  -mach        "${mach}"        "hadar pollux" "[machines to execute the experiment]"\
  -giveup      "${giveup}"      ""            "[stop execution after a test fail ]"\
  -outputs     "${outputs}"     ""            "[set location for model output files ]"\
  -email       "${email}"       ""            "[send an email when tests are completed ]"\
  -listing     "${listing}"     ""            "[set location for listing files]"\
  -summary     "${summary}"     ""            "[write summary in a file ]"\
  -nocolor     "${nocolor}"     ""            "[]"\
  -launch      "${launch}"      "0"           "[run the experiment           ]"\
  -orig_cfg    "${orig_cfg}"    "0"           "[new copy of level* config directories ]"\
  ++ $arguments`
exp=$saved_exp

#Interface: save current configuration
if [ -n "${CONFIG_FILE}" ] ; then
  cp ${CONFIG_FILE} ${TMPDIR}/gem_verif.cfg$$
fi
. r.ecfg.dot $saved_cfg_file
. r.ecfg.dot last
if [ -n "${CONFIG_FILE}" ] ; then
  cnt=$(diff ${HOME}/.${scriptname}_rc/last.cfg ${TMPDIR}/gem_verif.cfg$$ | wc -l)
  /bin/rm -f ${TMPDIR}/gem_verif.cfg$$
else
  cnt=0
fi
cat ${HOME}/.${scriptname}_rc/last.cfg
if [ $cnt -gt 0 ] ; then
  printf "\n Modification to old experiment\n\n"
fi

if [ ! -z "$nocolor" ]; then
   USE_COLOR='n'
fi

if [ ! -z "$email" ]; then
   summary=$(mktemp)
   trap "rm -rf $summary" EXIT

   if [ $USE_COLOR == 'y' ]; then
      warning "Automatically turn off colors to send an email"
      USE_COLOR='n'
   fi
fi

. term_colors.sh

# true_path of source code directories

unset liste_code_dir ; set -A liste_code_dir $(echo ${code})
nb_code=$((${#liste_code_dir[@]}-1)) ; cnt_code=0 ; unset code
set -ex
while [ $cnt_code -le ${nb_code} ] ; do
   code=${code}" "$(true_path ${liste_code_dir[$cnt_code]})
   cnt_code=$((cnt_code+1))
done
set +ex

# make machine inventory
set -A liste $(echo ${mach})
nb=$((${#liste[@]}-1)) ; cnt=0 ; unset liste_mach
while [ $cnt -le ${nb} ] ; do
   this_mach=$(echo ${liste[$cnt]} | cut -d ":" -f 1)
   deja=$(echo ${liste_mach} | grep ${this_mach} | wc -l)
   if [ $deja -eq 0 ] ; then
      liste_mach=${liste_mach}" "${this_mach}
   fi
   cnt=$((cnt+1))
done

DATABASE=${HOME}/GEMDEV_VERIFICATIONS/INTEGRATION/${version}/${exp}
launch=${launch:-0}
nb_levels=${nb_levels:-5}
orig_cfg=${orig_cfg:-0}

set -A lesmachs $(echo ${liste_mach})

# Build experiments directory, compile and load for all machines

if [ ${buildexp} -gt 0 ] ; then
  nb=$((${#lesmachs[@]}-1)) ; cnt=0
  while [ $cnt -le ${nb} ] ; do
     printf "\n Compile and Load on ${lesmachs[$cnt]} in directory ${storage_model}/gem_intt/${ATM_MODEL_VERSION}/intt_${exp}\n"
     echo "$(which Compile_and_Load.ksh) -gem_version ${ATM_MODEL_BNDL} -update ${GEM_DEV_UPDATES} -exp ${exp} -code "${code}" -buildexp ${buildexp} ; echo STATUS_COMLD=\$?" | ssh ${lesmachs[$cnt]} bash --login 1> ${TMPDIR}/ExpCL$$_${lesmachs[$cnt]} 2>&1 &
     cnt=$((cnt+1))
  done
  echo wait
  wait
  nb_errors=0
  for i in ${TMPDIR}/ExpCL$$_* ; do
     eval $(grep STATUS_COMLD= $i)
     if [ $STATUS_COMLD -ne 0 ] ; then
        nb_errors=$((nb_errors+1))
        cat $i
     fi
  done
  /bin/rm -f ${TMPDIR}/ExpCL$$_*
  if [ $nb_errors -gt 0 ] ; then exit 1 ; fi
fi

repcfgs=${HOME}/home/gem/${ATM_MODEL_VERSION}/intt_${exp}_configs

# Copy over and update list of tests
if [ -n "${cfgs}" ] ; then
  if [ -d ${cfgs} ] ; then
     if [ ${orig_cfg} -gt 0 ] ; then
        if [ -d ${repcfgs} ] ; then
           printf "\n  Config directory ${repcfgs} exists\n"
           printf "  Override [Y or N] ???\n\n"
           read over_cfgs
           if [ "${over_cfgs}" == "Y" -o "${over_cfgs}" == "y" ] ; then
              /bin/rm -rf ${repcfgs}
           fi
        fi
     fi
     if [ ! -d ${repcfgs} ] ; then
        mkdir -p ${repcfgs}
#this would be the occasion to include the namelist updater
        printf "\n  Copying config directory ${cfgs} into ${repcfgs}\n\n"
        cp -r ${cfgs}/* ${repcfgs}
     fi
  fi
fi

if [ ! -d ${repcfgs} ] ; then
   die "${repcfgs} NOT created"
fi

# Launch execution
export SETMEX="-ex"

if [ ${launch} -gt 0 ] ; then

   nb_cfgs=$(ls -1 ${repcfgs}/ | wc -l)
   if [ ${nb_cfgs} -lt 1 ] ; then
      die "NO configs to run"
   fi

   unset lesmachs liste
   set -A lesmachs $(echo ${mach})
   nb_machines=${#lesmachs[@]}
   /bin/rm -f ${DATABASE}/.ALL_DONE* 

   lajob=${HOME}/home/gem/${ATM_MODEL_VERSION}/intt_${exp}_lajob
   cat > $lajob <<EOF
export PATH=\${PATH}:${root_install}/library
repexp=\${storage_model}/gem_intt/${ATM_MODEL_VERSION}/intt_${exp}
if [ ! -d \${repexp} ] ; then
   printf "\n  Directory \${repexp} NOT available\n"
   printf "  run with -buildexp 1 first\n\n"
else
  cd \${repexp}
  . .ssmuse_gem
  run_verif.ksh -dir_configs ${repcfgs} -nb_levels ${nb_levels} -ncores \$NCORES -database $DATABASE -expcmp $expcmp
set -x
  touch ${DATABASE}/.ALL_DONE_\${TRUE_HOST}
  cnt=\$(ls -1 ${DATABASE}/.ALL_DONE* | wc -l)
  if [ \$cnt -eq $nb_machines ] ; then
    summary.ksh $DATABASE \${repexp} 
  fi
fi
EOF

   nb=$((nb_machines-1)) ; cnt=0 ; n=0
   while [ $cnt -le ${nb} ] ; do
     nw=$(echo ${lesmachs[$cnt]} | sed 's/:/ /g' | wc -w)
     if [ ${nw} == 3 ] ; then
       machine=$(echo ${lesmachs[$cnt]} | cut -d ":" -f 1)
       ltype=$(  echo ${lesmachs[$cnt]} | cut -d ":" -f 2)
       ncores=$( echo ${lesmachs[$cnt]} | cut -d ":" -f 3)
       if [ "${ltype}" == "batch" ] ; then
          printf "\n Launching experiment $exp in directoty ${storage_model}/gem_intt/${ATM_MODEL_VERSION}/intt_${exp} on $machine using $ltype\n\n"
        # does not work at this point on hadar/spica (wait for next ppp and hare)
          echo "export NCORES=$ncores" > ${lajob}.ordsoumet
          cat $lajob >> ${lajob}.ordsoumet
          ord_soumet ${lajob}.ordsoumet -mach $machine -mpi -cm 2G -t 3600 -cpus ${ncores}x1
       else
          liste_ssh[$n]=${machine}
          liste_cores[$n]=${ncores}
          n=$((n+1))
       fi
     else
        printf "\n Ignoring ${lesmachs[$cnt]} : improper syntax\n\n"
     fi
     cnt=$((cnt+1))
   done

   cat > ${lajob}.ssh <<EOF
export SETMEX=${SETMEX}
set \${SETMEX:-+ex}
ulimit -S -m unlimited
ulimit -S -s unlimited
ulimit -S -d unlimited
EOF
   cat $lajob >> ${lajob}.ssh
   chmod 755 ${lajob}.ssh

   nb=$((n-1)) ; cnt=0
   while [ $cnt -le ${nb} ] ; do
      printf "\n EXECUTING $lajob in directoty ${storage_model}/gem_intt/${ATM_MODEL_VERSION}/intt_${exp} on ${liste_ssh[$cnt]} using ${liste_cores[$cnt]} cores &\n\n"
      lis=${HOME}/listings/${liste_ssh[$cnt]}/$(basename ${lajob})_$$.o
#      export NCORES=${liste_cores[$cnt]} ; time ${lajob}.ssh 1> $lis 2>&1
      echo "export NCORES=${liste_cores[$cnt]} ; time ${lajob}.ssh 1> $lis 2>&1" | ssh ${liste_ssh[$cnt]} bash --login 1> /dev/null 2>&1 &
     cnt=$((cnt+1))
   done

   echo  wait
   wait
  
fi
