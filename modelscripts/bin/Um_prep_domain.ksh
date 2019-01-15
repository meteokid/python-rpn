#!/bin/ksh

# Prepare a "domain" (model config instance) for execution
arguments=$*
echo $0 $arguments
date
eval `cclargs_lite -D " " $0 \
  -anal        ""          ""       "[Analysis file or archive     ]"\
  -input       ""          ""       "[Model input file path        ]"\
  -o           ""          ""       "[Domain output path           ]"\
  -work        "${TASK_WORK}" ""    "[Work directory path          ]"\
  -bin         "${TASK_BIN}"  ""    "[Binaries directory path      ]"\
  -headscript  ""          ""       "[Headscript to run            ]"\
  -check_namelist "0"      "1"      "[Check namelist validity      ]"\
  -nmlfile     ""          ""       "[Model namelist settings file ]"\
  -npex        "1"         "1"      "[# of processors along x      ]"\
  -npey        "1"         "1"      "[# of processors along y      ]"\
  -cache       ""          ""       "[GEM_cache                    ]"\
  -nthreads    "1"         "1"      "[# of simultaneous threads    ]"\
  -verbose     "0"         "1"      "[verbose mode                 ]"\
  -abort       ""          ""       "[Abort signal file prefix     ]"\
  ++ $arguments`

# Check for required inputs
set ${SETMEX:-+ex}

if [[ -z "${o}" ]] ; then
   echo "Error: output path (-o) must be defined for $0" >&2
   exit 1
fi

mydomain=$(basename ${o})

# Normalize path names
mkdir -p ${work} ${o}
for fpath in anal input o work headscript nmlfile ; do
   target=$(eval echo \$${fpath})
   if [[ -e ${target} ]] ; then 
      eval ${fpath}=$(true_path ${target})
   fi
done
# Prepare abort file in case of early return
if [[ -n "${abort}" ]] ; then 
   abort_file=${work}/${abort}-$$
   touch ${abort_file}
fi

if [ -n "${bin}" ] ; then
  bin=${bin}/
fi

work=${work}/${mydomain}
mkdir -p ${work} ; cd ${work}

date
if [ -n "${nmlfile}" ] ; then
if [ -e "${nmlfile}" ] ; then
   # Verify namelist entries on request
   if [ ${check_namelist} -gt 0 ] ; then
      ln -s ${nmlfile} ./gem_settings.nml
      ${bin}checknml -q gem_cfgs physics step grid series
      rm -f ./gem_settings.nml
   fi
   ${bin}checkdmpart_wrapper.ksh \
            -nmlfile ${nmlfile} -domain ${mydomain} -bin $bin \
            -check_namelist ${check_namelist} -cache ${cache} \
            -npex ${npex} -npey ${npey} -verbose $verbose
fi
fi
date

# Prepare output and work space
target_dir=${o}
tmp_analysis_path=tmp_analysis ; mkdir -p ${tmp_analysis_path}
tmp_analysis_path=$(true_path ${tmp_analysis_path})
cd ${tmp_analysis_path}

# Extract cmc archive file input or link in surface analysis
local_anal_file=${tmp_analysis_path}/ANALYSIS
if [ -e "${anal}" ] ; then
   if [[ -L ${anal} ]] ; then
      analysis=$(r.read_link ${anal})
   else
      analysis=${anal}
   fi

   if [ -d ${analysis} ] ; then
     set -x
     cd ${analysis}
     for item in $(ls -1 .) ; do
        if [[ -f ${item} ]] ; then
           ${bin}editfst -s ${item} -d ${local_anal_file} -i 0
        elif [[ -d ${item} ]] ; then
           ln -s $(true_path $item) ${o}/IAUREP
        fi
     done
     set +x
   else
     is_cmcarc=${work}/.is_cmcarc ; rm -f ${is_cmcarc}
     (${bin}cmcarc -t -f ${analysis} 2> /dev/null || exit 0 ; touch ${is_cmcarc})
     if [[ -e ${is_cmcarc} ]] ; then
        mkdir -p ${work}/unpack_archive
        cd ${work}/unpack_archive
        set -x
        time cmcarc -x -f ${analysis}
        for item in $(ls -1 .) ; do
          if [[ -f ${item} ]] ; then
            ${bin}editfst -s ${item} -d ${local_anal_file} -i 0
            /bin/rm -f ${item}
          elif [[ -d ${item} ]] ; then
            mv $item ${o}/IAUREP
          fi
        done
        set +x
     else
        set -x
        cp ${analysis} ${local_anal_file}
        set +x
     fi
   fi
   cd ${work}

   # Run the user-defined headscript
   final_file=${o}/ANALYSIS
   if [[ -x ${headscript} ]] ; then
      local_touchup_file=${local_anal_file}_touchup
      ${headscript} ${local_anal_file} ${local_touchup_file}
      mv ${local_touchup_file} ${final_file}
   else 
      mv ${local_anal_file} ${final_file}
   fi
fi
date
set -ex
gtype=$(${bin}getnml -f ${nmlfile} -n grid grd_typ_s)
if [ "${gtype}" == "'GU'" -o "${gtype}" == "'GY'" ] ; then
  input=''
fi

# Preparation splitting of input files
splitdir=${o}/analysis
inrepdir=${o}/model_inrep
mkdir -p ${splitdir} ${inrepdir}
wild='@NIL@'
input=$(echo $input | sed "s/'//g")
if [ -n "${input}" ] ; then
  if [ ! -d ${input%%\**} ] ; then
    wild=${input##*/}
    input=$(dirname ${input%%\**})
  fi
fi

if [[ -e ${final_file} ]] ; then
   ${bin}GEM_trim_input.ksh ${final_file} '@NIL@' ${splitdir} ${nthreads}
   dir=${splitdir}/$(ls -1 ${splitdir} | head -1)
   for i in $(ls -1 ${dir}/GEM_input_file*) ; do
      varname='TT'
      if [ $(r.fstliste.new -izfst $i -nomvar ${varname} | wc -l) -le 0 ] ; then
         varname='TT1'
      fi
      if [ $(r.fstliste.new -izfst $i -nomvar ${varname} | wc -l) -le 0 ] ; then
         varname='VT'
      fi
      valid=$(r.fstliste.new -izfst $i -nomvar ${varname} | head -1 | cut -d ":" -f 11)
      if [ -n "${valid}" ] ; then
         echo $(echo $valid | cut -c1-8).$(echo $valid | cut -c9-14) > ${splitdir}/analysis_validity_date
         break
      fi
   done
fi
date
if [[ -d ${input} ]] ; then
   ${bin}GEM_trim_input.ksh ${input} ${wild} ${inrepdir} ${nthreads}
fi
date
/bin/rm -rf ${work}/tmp_analysis ${work}/unpack_archive

# Final cleanup for return
if [[ -n "${abort_file}" ]] ; then rm -f ${abort_file} ; fi

