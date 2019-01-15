#!/bin/ksh
# Launch a GEM integration, handling errors and time slice submissions

# Retrieve command line arguments
arguments=$*

# Check that running on the local machine is allowed
if [[ "${ATM_MODEL_NORUN_HOST}" == "1" ]] ; then
   echo "Model integrations are not permitted on this host (ATM_MODEL_NORUN_HOST=${ATM_MODEL_NORUN_HOST})" 1>&2
   exit 1
fi

# Process command line arguments
echo $0 ${arguments}
. r.entry.dot

eval $(rpy.cclargs \
   -D " " \
   --desc   "Run GEM Model task" \
   ${0##*/} \
   -cfg           "0:0"        "0:0"       "[Configurations number/range (START:END)]"\
   -dircfg        "GEM_cfgs"   "GEM_cfgs"  "[Location of config files]"\
   -barrier       "0"          "0"         "[DO NOT run binary]"\
   -timing        "0"          "0"         "[Report performance timers]"\
   -ptopo         "1x1x1"      "1x1x1"     "[MPI & OMP PEs topology (NPEXxNPEYx NOMP)]"\
   -instances     ""           ""          "[(yinyang x ndomains)]"\
   -smt           ""           ""          "[SMT controler (AIX) (smtdyn x smtphy)]"\
   -bind          "0"          "1"         "[Processor binding logical controler]"\
   -inorder       "0"          "1"         "[Order listing]"\
   -nompi         "0"          "1"         "[Run without MPI]"\
   -debug         "0"          "1"         "[Debug session]"\
   -task_basedir  "RUNMOD"     "RUNMOD"    "[Task dir name]"\
   -no_setup      "0"          "1"         "[Do not run setup]"\
   -_status       "ABORT"      "ABORT"     "[Return status]"\
   -_endstep      ""           ""          "[Last time step performed]"\
   -_npe          "1"          "1"         "[Number of subdomains]"\
  ++ ${arguments})

check_instance() {
   if [ $1 -ne $2 ] ; then
      printf "\n Mismatch between required and provided # of instances:\n"
      printf "      provided: $1\n"
      printf "      required: $2\n"
      printf " ------ ABORT ------\n\n"
      return 1
   fi
}

printf "\n=====>  Um_runmod.ksh starts: cfg_$cfg `date` ###########\n\n"
restart=0

if [ ${no_setup} = 0 ] ; then
   set ${SETMEX:-+ex}
   if [ -d ${task_basedir}/work ] ; then
      restart=$(find -L ${task_basedir}/work -type f -name "gem_restart" | wc -l)
   fi
   if [ ${restart} -gt 0 ] ; then
      export TASK_BASEDIR=$(true_path ${task_basedir})
      export TASK_INPUT=${TASK_BASEDIR}/input
      export TASK_BIN=${TASK_BASEDIR}/bin
      export TASK_WORK=${TASK_BASEDIR}/work
      export TASK_OUTPUT=${TASK_BASEDIR}/output
      printf "Running in NORMAL RESTART mode\n"
   else
      datadir=${TMPDIR}/modeldata.$$
      /bin/rm -fr ${task_basedir:-yenapas}/* ${task_basedir:-yenapas}/.setup ${datadir}
      mkdir -p ${datadir}
      TASK_CFGFILE=${TMPDIR}/mod$$.cfg
      Um_setmod.ksh \
         -cfg $(echo ${cfg} | cut -d : -f 1):$(echo ${cfg} | cut -d : -f 2) \
         -dircfg ${dircfg} -tsk_cfgfile ${TASK_CFGFILE} -dirdata ${datadir}
      if [ -z "${TASK_SETUP}" ] ; then
         cat <<EOF
  Error in $(basename $0)
  Env variable TASK_SETUP not defined
  --- ABORT ---

EOF
         . r.return.dot
      fi
      printf "\n##### EXECUTING TASK_SETUP ##### ${TASK_SETUP}\n"
      set -e
      . ${TASK_SETUP} -f $TASK_CFGFILE --base $(pwd)/${task_basedir} \
         --verbose --clean
      set ${SETMEX:-+ex}
      printf "##### EXECUTING TASK_SETUP DONE...#####\n"
      printf "\n##### RESULT OF TASK_SETUP #####\n"
      ls -l ${TASK_BIN} ${TASK_INPUT}/cfg_*
      /bin/rm -fr ${TASK_CFGFILE} ${datadir}
   fi
fi

buf=`echo ${ptopo:-1x1x1} | sed 's/x/ /g'`
npex=`echo   $buf | awk '{print $1}'`
npey=`echo   $buf | awk '{print $2}'`
nomp=`echo   $buf | awk '{print $3}'`
npex=${npex:-1}
npey=${npey:-1}
nomp=${nomp:-1}
buf=`echo ${smt:-0x0} | sed 's/x/ /g'`
smtdyn=`echo $buf | awk '{print $1}'`
smtphy=`echo $buf | awk '{print $2}'`
smtdyn=${smtdyn:-0}
smtphy=${smtphy:-0}
if [ ${bind} -gt 0 ] ; then
   bind=true
else
   bind=false
fi
_npe=$((npex*npey))

if [ ${_npe} -gt 1 -a ${nompi} -gt 0 ] ; then
   printf "\n  -nompi option can only be used on a 1x1 mpi processor topology -- ABORT --\n\n"
   exit 1
fi

ngrids=1
for i in ${TASK_INPUT}/cfg_* ; do
   GRDTYP=$(rpy.nml_get -f ${i}/model_settings.nml grid/Grd_typ_s | sed 's/"//g' | sed "s/'//g")
   if [ "$GRDTYP" == "GY" ] ; then
      export GEM_YINYANG=YES
      ngrids=2
   fi
   break
done

for i in ${TASK_INPUT}/cfg_* ; do
   dname=`basename $i`
   mkdir -p ${TASK_OUTPUT}/${dname} ${TASK_WORK}/${dname}
   if [ "$GRDTYP" == "GY" ] ; then
      mkdir -p ${TASK_WORK}/${dname}/YIN ${TASK_WORK}/${dname}/YAN
   fi
   if [ -e ${TASK_INPUT}/${dname}/configexp.cfg ] ; then
      cp ${TASK_INPUT}/${dname}/configexp.cfg ${TASK_OUTPUT}/${dname}
   fi
   if [ -e ${dircfg}/cfg_all/BATCH_config.cfg ] ; then
      cat ${dircfg}/cfg_all/BATCH_config.cfg >> ${TASK_OUTPUT}/${dname}/configexp.cfg
   fi

   # Do not overwrite an existing model_settings.nml since a parent script (i.e. Maestro task Runmod) may have modified it
   if [[ ! -e ${TASK_WORK}/${dname}/model_settings.nml ]] ; then
      cp ${TASK_INPUT}/${dname}/model_settings.nml ${TASK_WORK}/${dname}
   fi
   if [[ -e ${TASK_INPUT}/${dname}/output_settings ]] ; then
      cp ${TASK_INPUT}/${dname}/output_settings ${TASK_WORK}/${dname}
   fi
   if [[ -e ${TASK_INPUT}/${dname}/coupleur_settings.nml ]] ; then
      cp ${TASK_INPUT}/${dname}/coupleur_settings.nml ${TASK_WORK}/${dname}
   fi


   chmod u+w ${TASK_WORK}/${dname}/model_settings.nml
   cat >> ${TASK_WORK}/${dname}/model_settings.nml <<EOF

 &resources
  Ptopo_npex    = $npex    ,  Ptopo_npey    = $npey
  Ptopo_nthreads_dyn= $smtdyn,  Ptopo_nthreads_phy= $smtphy
  Ptopo_bind_L= .$bind.
/

EOF

   if [ -s ${TASK_INPUT}/${dname}/BUSPER.tar ] ; then
      (mkdir -p ${TASK_WORK}/$dname/busper ; \
         cd ${TASK_WORK}/$dname/busper ; \
         tar xvf ${TASK_INPUT}/${dname}/BUSPER.tar)
   fi

   # Use the split analysis file information to set date if not in settings file already
   RUNSTART=$(rpy.nml_get step/Step_runstrt_S \
      -f ${TASK_WORK}/${dname}/model_settings.nml 2> /dev/null)
   if [[ -z "${RUNSTART}" ]] ; then
      date_file=${TASK_INPUT}/${dname}/MODEL_ANALYSIS/analysis_validity_date
      if [ -e ${date_file} ] ; then
         RUNSTART=$(cat ${date_file})
         rpy.nml_set -f ${TASK_WORK}/${dname}/model_settings.nml step/Step_runstrt_S=\"${RUNSTART}\"
      fi
   fi

done

set ${SETMEX:-+ex}
typeset -Z4 domain_number

export DOMAIN_start=$(echo ${cfg} | cut -d: -f1)
export DOMAIN_end=$(echo ${cfg} | cut -d: -f2)
export DOMAIN_total=$((DOMAIN_end - DOMAIN_start + 1))
DOMAIN_wide=$(echo ${cfg} | cut -d : -f3)
DOMAIN_wide=${DOMAIN_wide:-${DOMAIN_total}}
if [ ${DOMAIN_wide} -lt 1 ] ; then
  DOMAIN_wide=1
fi
export DOMAIN_wide=${DOMAIN_wide}

export BATCH_launch_execdir=${BATCH_launch_execdir:-${PWD}}
export GEMBNDL_VERSION=${ATM_MODEL_VERSION}

# Use performance timers on request
if [ ${timing} -gt 0 ] ; then
   export TMG_ON=YES
fi

cd $TASK_WORK

if [ -f ${TASK_INPUT}/cfg_0000/restart.tar ] ; then
  (cd cfg_0000 ; tar xvf ${TASK_INPUT}/cfg_0000/restart.tar)
  printf "\nRunning in FORCED RESTART mode\n\n"
fi

DOM=${DOMAIN_start}
while [ ${DOM} -le ${DOMAIN_end} ] ; do

   last_domain=$((DOM+DOMAIN_wide-1))
   if [ ${last_domain} -gt ${DOMAIN_end} ] ;then
      last_domain=${DOMAIN_end}
   fi
   loop_cfg=${DOM}:${last_domain}
   ndomains=$((last_domain - DOM + 1))
   if [ -n "$instances" ] ; then
      check_instances ${instances} $((ngrids*ndomains))
      if [ $? -ne 0 ] ; then
         . r.return.dot
         exit 1
      fi
   fi
   export GEM_NDOMAINS=${loop_cfg}

   domain_number=$DOM
   file2watch=${TASK_BASEDIR}/output/cfg_${domain_number}/output_ready
   MONLIS=gem_monitor_output_cfg_${domain_number}.lis
   mkdir -p ${TASK_WORK}/post_process_output_cfg_${domain_number}
   gem_monitor_output ${file2watch} ${TASK_BIN}/launch_sortie.ksh 1>> ${MONLIS} 2>&1

   # Run main program wrapper
   set +ex

   printf "\n LAUNCHING Um_model.ksh for domain: cfg_${domain_number} $(date)\n\n"
   . r.call.dot ${TASK_BIN}/Um_model.ksh \
      -npex $((npex*ngrids)) -npey $npey -nomp $nomp \
      -ndom $ndomains -nompi $nompi -debug $debug \
      -barrier ${barrier} -inorder ${inorder}

   cnt=$(ls -1 ${TASK_WORK}/post_process_output_*/output_ready_work*.active 2> /dev/null | wc -l)
   while [ ${cnt} -gt 0 ] ; do
      printf "Waiting for: $cnt launch_sortie.ksh process to end\n"
      sleep 10
      cnt=$(ls -1 ${TASK_WORK}/post_process_output_*/output_ready_work*.active 2> /dev/null| wc -l)
   done

   set ${SETMEX:-+ex}
   if [ "$_status" = "ED"    ] ; then
      echo ${Runmod} > ${TASK_OUTPUT}/last_npass
      ${TASK_BIN}/launch_sortie.ksh ${file2watch}_MASTER 1>> ${MONLIS} 2>&1
   fi

   DOM=$((DOM+DOMAIN_wide))
done
set +ex

printf "\n DONE LAUNCHING all domains $(date)\n\n"

# Config file cleanup
/bin/rm -rf ${TASK_WORK}/busper

printf "\n=====>  Um_runmod.ksh ends: `date` ###########\n\n"

# End of task
. r.return.dot

