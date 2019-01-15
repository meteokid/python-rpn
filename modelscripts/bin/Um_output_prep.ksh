#!/bin/ksh
#
arguments=$*
. r.entry.dot

#====> Obtaining the arguments:
eval `cclargs_lite $0 \
     -input    ""      ""       "[Input directory                    ]"\
     -output   ""      ""       "[Output directory                   ]"\
     -assemble "0"     "1"      "[Reassemble or not                  ]"\
     -dplusp   "0"     "1"      "[Combine dynamics and physics output]"\
     -liste    ""      ""       "[Liste of model output type to treat]"\
     -listm    ""      ""       "[Xfer model listings along          ]"\
     -repcasc  ""      ""       "[Xfer cascade files                 ]"\
     -_status  "ABORT" "ABORT"  "[return status                      ]"\
     -nthreads "1"     "1"      "[Number of bemol process to run in parallel]"\
  ++ $arguments`

set ${SETMEX:-+ex}

printf "\n    =====> `basename $0` $arguments: `date`\n"

ici=`pwd`

nbfiles=0

##### We first deal with subdirectories ${input}/[0-9]*-[0-9]* #####

# casc_* cascade files
if [ -n "${repcasc}" ] ; then
   find -L ${input} -type f -name "casc_*" > casc_files_found
   cnt=`cat casc_files_found | wc -l`
   if [ $cnt -gt 0 ] ; then
     REP=${output}/${repcasc}
     mkdir ${REP}
     for i in `cat casc_files_found` ; do
       ln -s $i ${REP}
     done
   fi
   nbfiles=$((nbfiles+cnt))
fi

# Process all types of model output
printf "\n  =====> MODOUT starts ${nbfiles} `date`\n"
. r.call.dot ${TASK_BIN}/Um_output_modout.ksh -src ${input} -dst ${output} \
    -liste ${liste} -assemble ${assemble} -dplusp ${dplusp} -nthreads ${nthreads}
nbfiles=$((nbfiles+_nf2t))
assemble_error=${_nerr}
flag_err=${assemble_error}
printf "  =====> MODOUT ends: $nbfiles files treated: $assemble_error errors detected `date`\n"

##### Then we deal with other special files
prefix=''

# BUSPER4spinphy file to recycle BUSPER

if [ -s ${input}/endstep_misc_files/BUSPER4spinphy* ] ; then
  mv ${input}/endstep_misc_files/BUSPER4spinphy* ${output}
fi

# Restart files
find -L ${input} -type d -name "restart*" > restart_files_found
cnt=`cat restart_files_found | wc -l`
if [ $cnt -gt 0 ] ; then
  for i in `cat ${ici}/restart_files_found` ; do
     restart_name=$(basename ${i})
     cd ${i}
     tar cvf ${output}/${restart_name}.tar ${prefix}*
  done
  nbfiles=$((nbfiles+cnt))
  cd ${ici}
fi

# Time series file
find -L ${input} -type f -name 'time_series.bin*' > series_found
cnt=`cat series_found | wc -l`
series_ok=1
if [ $cnt -gt 0 ] ; then
  in_serie=`cat series_found`
  printf "\n  =====>  feseri -iserial ${in_serie} -omsorti time_series.fst \n"
  #ln -sf ${in_serie} .
  for item in ${in_serie} ; do
     itemnum=${item##*_}_YIN
     if [[ x$(echo ${item} | sed 's|/YAN/|/|') != x${item} ]] ; then
        itemnum=${item##*_}_YAN
     fi
     itemout=time_series.fst_${itemnum}
     itemin=time_series.bin_${itemnum}
     ln -s ${item} ${itemin}
     ${TASK_BIN}/feseri -iserial ${itemin} -omsorti ${itemout} 1> ${itemin}.lis
     if [ $? -ne 0 ]; then
        printf "\n feseri aborted \n\n"
        /bin/rm -f ${itemout} ${itemin}
        flag_err=$((flag_err+1))
        series_ok=0
        break
     else
        /bin/rm -f ${item} ${itemin} 2> /dev/null || true
     fi
  done
  if [[ $series_ok == 1 ]] ; then
     if [[ x"$(ls time_series.fst_*_YIN 2>/dev/null)" != x"" ]] ; then
        editfst -i 0 -s time_series.fst_*_YIN -d time_series.fst
        mv time_series.fst ${output}
        nbfiles=$((nbfiles+1))
     fi
     if [[ x"$(ls time_series.fst_*_YAN 2>/dev/null)" != x"" ]] ; then
        editfst -i 0 -s time_series.fst_*_YAN -d time_series.fst_YAN
        mv time_series.fst_YAN ${output}
        nbfiles=$((nbfiles+1))
     fi
  fi
fi

# Listings files
if [ "${listm}" ] ; then
  lalistetomv=`ls ${listm}* 2> /dev/null | xargs`
  if [ -n "$lalistetomv" ] ; then
    cp ${lalistetomv} ${output} 2> /dev/null
    nbfiles=$((nbfiles+1))
  fi
fi

if [ $flag_err -eq 0 ] ; then
  _status='OK'
  cnt=`ls -l ${output} | wc -l`
  cnt=$((cnt-1))
  printf "\n=====> Um_output_prep: ${nbfiles} elements treated\n"
  printf "=====> Um_output_prep: ${cnt} elements produced for XFER\n"
else
  printf "\n $flag_err ERRORS found in Um_output_prep.ksh - ABORT\n"
fi

printf "\n=====>  Um_output_prep.ksh ends: `date` ###########\n\n"

# End of task
. r.return.dot



