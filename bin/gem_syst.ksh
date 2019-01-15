#!/bin/ksh

scriptname=$(basename $0)
arguments=$*

if [ -z "${ATM_MODEL_BNDL}" ] ; then
  printf "\n  NO GEM ENVIRONMENT -- ABORT\n\n"
  exit 1
fi

# This next variable will eventually point to $gemdyn/VERIF some day
# root_install=$gemdyn/VERIF
# but for now:
root_install=/users/dor/armn/mid/home/VERIF

export PATH=${PATH}:${root_install}/library:${root_install}/CYCLER/v_2.0.8/bin

#Interface: unset all controls
unset exp expcmp code syst module date_server mach buildexp flow launch orig_cfg 
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
  -syst        "${syst}"        "RDPS GDPS HRDPS" "[which system               ]"\
  -exp         "${exp}"         "${exp}"          "[experiment name            ]"\
  -expcmp      "${expcmp}"      ""                "[compare experiment names   ]"\
  -code        "${code}"        ""                "[newcode for this experiment]"\
  -module      "${module}"      "${gemmod}"       "[gem module to use          ]"\
  -date_server "${date_server}" "hadar"           "[machine to use for dates   ]"\
  -mach        "${mach}"        "hadar:topo=4x4x1x3:wl=8x1:time=30 pollux:topo=4x4x1x4:wl=1x1:time=50" "[machines:threads to launch on]"\
  -buildexp    "${buildexp}"    "0"               "[gemdev, code copy, compile and load ]"\
  -flow        "${flow}"        "0"               "[build maestro experiment            ]"\
  -launch      "${launch}"      "0"               "[launch maestro experiment           ]"\
  -orig_cfg    "${orig_cfg}"    "0"               "[new copy of rc_* config directories ]"\
  ++ $arguments`
exp=$saved_exp
saved_mach=$mach

#Interface: save current configuration
if [ -n "${CONFIG_FILE}" ] ; then
  cp ${CONFIG_FILE} ${TMPDIR}/gem_verif.cfg$$
fi
. r.ecfg.dot $saved_cfg_file
. r.ecfg.dot last
if [ -n "${CONFIG_FILE}" ] ; then
  cnt=$(diff ${HOME}/.$(basename $0)_rc/last.cfg ${TMPDIR}/gem_verif.cfg$$ | wc -l)
  /bin/rm -f ${TMPDIR}/gem_verif.cfg$$
else
  cnt=0
fi
cat ${HOME}/.$(basename $0)_rc/last.cfg
if [ $cnt -gt 0 ] ; then
  printf "\n Modification to old experiment\n\n"
fi

REPBIN=${HOME}/home/gem/$(basename ${ATM_MODEL_BNDL})/syst_${exp}
DATABASE=${HOME}/GEMDEV_VERIFICATIONS/SYSTEMS
DIRCFG=${REPBIN}/suite/SYSTEM_testings
###TODO DIRCFG=$(r.read_link ${REPBIN}/suite).cfg

# Build a gemdev experiment
config=0
if [ ${buildexp} -gt 0 ] ; then
  if [ -e ${REPBIN}/suite -a ${buildexp} -lt 2 ] ; then
    printf "\n Experiment directory ${REPBIN}/suite already available\n"
    printf " run with -buildexp 2 to force re-build\n\n"
    exit 1
  fi
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

  printf "\n  Building experiment directory with \n  gemdev.dot syst_${exp} -mach ${liste_mach} -v -f\n\n"
  . gemdev.dot syst_${exp} -mach ${liste_mach} -v -f

  exp=$saved_exp
  mach=$saved_mach
  config=2

  set -A liste $(echo ${code})
  nb=$((${#liste[@]}-1)) ; cnt=0 ; flag=0
  while [ $cnt -le ${nb} ] ; do
    if [ -d ${liste[$cnt]} ] ; then
      cp ${liste[$cnt]}/* ${REPBIN} 2> /dev/null | true
    else
      printf "\n  Code directory ${liste[$cnt]} NOT available\n"
      flag=1
    fi
    cnt=$((cnt+1))
  done
  if [ ${flag} -gt 0 ] ; then exit 1 ; fi

# Compile and load on all $mach
  set -A liste $(echo ${liste_mach})
  nb=$((${#liste[@]}-1)) ; cnt=0
  while [ $cnt -le ${nb} ] ; do
    printf "  Compile and Load on ${liste[$cnt]} in directory ${REPBIN} &\n"
    echo "cd ${REPBIN} ; . .ssmuse_gem 1> lis_${liste[$cnt]}.lis 2> lis_${liste[$cnt]}.err ; linkit -f 1>> lis_${liste[$cnt]}.lis 2>> lis_${liste[$cnt]}.err ; make buildclean deplocal 1>> lis_${liste[$cnt]}.lis 2>> lis_${liste[$cnt]}.err ; make obj -j 4 1>> lis_${liste[$cnt]}.lis 2>> lis_${liste[$cnt]}.err ; make gemdm 1>> lis_${liste[$cnt]}.lis 2>> lis_${liste[$cnt]}.err" | ssh ${liste[$cnt]} bash --login 1> /dev/null 2> /dev/null &
    cnt=$((cnt+1))
  done
  wait
fi

if [ ${config} -eq 0 ] ; then
  if [ -n "${orig_cfg}" ] ; then config=${orig_cfg} ; fi
fi

if [ ! -e ${REPBIN}/suite ] ; then
  if [ ${config} -gt 0 -o ${flow} -gt 0 -o ${launch} -gt 0 ] ; then
    printf "\n  Experiment directory ${REPBIN}/suite is NOT available\n"
    printf "  run with -buildexp 1 at least once - ABORT \n\n"
    exit 1
  fi
fi

VERSION_ref=v_$(basename ${ATM_MODEL_BNDL} | cut -d "." -f1-2)

# Copy config directories from the depot $CYCLER_source/SYSTEM_testings
# Build maestro flow (-flow 1) and/or launch the execution (-launch 1)

if [ ${config} -gt 0 -o ${flow} -gt 0 -o ${launch} -gt 0 ] ; then

  set -A liste_syst $(echo ${syst})
  nb_syst=$((${#liste_syst[@]}-1))
  mkdir -p ${DIRCFG}

  if [ ${flow} -gt 0 ] ; then
    cat > ${DIRCFG}/xflow.suites.xml <<EOF
<?xml version='1.0'?>
<!DOCTYPE GroupList[]>
<GroupList>
<Group name="SYSTEM_testings">
EOF
  fi

  cnt_syst=0
  while [ $cnt_syst -le ${nb_syst} ] ; do
    this_system=${liste_syst[$cnt_syst]}
    if [ $cnt_syst -lt ${nb_syst} ] ; then
      nextsyst=${liste_syst[$((cnt_syst+1))]}
    else
      unset nextsyst
    fi

    set -A liste_sc $(find ${root_install}/SYSTEM_testings/${VERSION_ref} -type d -name "${this_system}*"|sort)
    MAESTRO_suite=${this_system}_verif_v${version}_${exp}
    if [ ${cnt_syst} -eq 0 ] ; then
      HEAD_dircfg=${DIRCFG}/${MAESTRO_suite}
    fi
    cnt=0 ; nb_elm=$((${#liste_sc[@]}-1))
    while [ $cnt -le ${nb_elm} ] ; do
      dir=${liste_sc[$cnt]}
      bdir=$(basename $dir)
      edir=$(echo ${bdir}_ | cut -d "_" -f 2)
      dircfg=${DIRCFG}/rc_${bdir}_${exp}
      if [ ${config} -gt 0 ] ; then
        if [ -e ${dircfg} ] ; then
          if [ ${config} -lt 2 ] ; then
            printf "\n Directory ${dircfg} already available\n"
            printf " run with -orig_cfg 2 to force a new copy from the depot\n\n"
          else
            /bin/rm -rf ${dircfg}
          fi
        fi
# Copy over default configurations from depot ${root_install}/SYSTEM_testings/${VERSION_ref}
        if [ ! -d ${dircfg} ] ; then
          if [ $cnt -lt ${nb_elm} ] ; then
            nextdir=${liste_sc[$((cnt+1))]}
            bdir=$(basename $nextdir)
            nextdir=$(echo ${bdir}_ | cut -d "_" -f 2)
            nextsuite=${DIRCFG}/${MAESTRO_suite}${nextdir}
          else
            unset nextsuite
            if [ -n "${nextsyst}" ] ; then
              nextsuite=${DIRCFG}/${nextsyst}_verif_v${version}_${exp}
            fi
          fi
          printf "\n ====> Copying $dir\n\n"
          cp -r $dir ${dircfg}
          if [ -n "${code}" ] ; then cp -r ${code} ${dircfg} ; fi
          cat > $TMPDIR/gemst_cfg$$ <<EOF
RC_system=${this_system}
RC_version_gem=$(echo ${ATM_MODEL_BNDL} | cut -d "/" -f2-3 | sed 's-/--g')
RC_season=verif
RC_expname=${exp}${edir}
RC_postcmp="${expcmp}"
RC_gem_ovbin=${REPBIN}
RC_nextsuite=${nextsuite}
RC_machines="${mach}"
RC_DESTINATION=datasvr:${DATABASE}
RC_module_gem=${module}
RC_date_server=${date_server}
EOF
          cat ${dircfg}/suite.cfg >> $TMPDIR/gemst_cfg$$
          mv $TMPDIR/gemst_cfg$$ ${dircfg}/suite.cfg
        fi
      fi

# Build maestro flow
      if [ ${flow} -gt 0 ] ; then
        if [ ! -e ${dircfg} ] ; then
           printf "\n Config directory ${dircfg} NOT available\n"
           printf " run with -build or -orig_cfg at least once - ABORT\n\n"
           exit 1
        fi        
        set -ex
        launch_cycle.ksh -dircfg ${dircfg} -suite_dir ${DIRCFG} -launch 0
        set +ex
      fi

      cnt=$((cnt+1))
    done
    cnt_syst=$((cnt_syst+1))
  done

  if [ ${flow} -gt 0 ] ; then
    cnt_syst=0 ; cd ${DIRCFG}
    while [ $cnt_syst -le ${nb_syst} ] ; do
      this_system=${liste_syst[$cnt_syst]}
      if [ -e ${this_system}_verif_v${version}_${exp} ] ; then
        echo "<Group name=\"${this_system}\">" >> ${DIRCFG}/xflow.suites.xml
        cd ${this_system}_verif_v${version}_${exp}
        for i in $(find ./ -type d -name "${this_system}_verif_*") ; do
          echo "<Exp>${PWD}/${i}</Exp>" >> ${DIRCFG}/xflow.suites.xml
        done
        echo "</Group>" >> ${DIRCFG}/xflow.suites.xml
        cd ../
      fi
      cnt_syst=$((cnt_syst+1))
    done
    cat >> ${DIRCFG}/xflow.suites.xml <<EOF
</Group>
</GroupList>
EOF
  fi

# Launch execution
  if [ ${launch} -gt 0 ] ; then
     if [ ! -e ${HEAD_dircfg} ] ; then
        printf "\n Maestro flow ${HEAD_dircfg} NOT available\n"
        printf " run with -flow 1 at least once - ABORT\n\n"
        exit 1
     fi
     submit_cycle.ksh ${HEAD_dircfg}
  fi

fi
