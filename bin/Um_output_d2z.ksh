#!/bin/ksh
#
#====> Obtaining the arguments:
eval `cclargs_lite $0 \
     -repdst   "bemol_output"   "bemol_output"    "[Destination directory for d2z output            ]"\
     -type     ""   ""    "[type of file being worked on (dm pm dh ph dp pp)]"\
     -prefix   ""   ""    "[Prefix for output files                         ]"\
     -dplusp   "0"  "1"   "[combine dynamics and physics output             ]"\
  ++ $*`
#

printf " Um_output_d2z.ksh -rep ${rep} -type ${type}\n"

/bin/rm -rf bemol_input
mkdir -p bemol_input

dliste=""
for i in `cat $1` ; do
  step=${i##*/}
  step=${step#*_}
  dejala=`echo $dliste | grep $step | wc -l`
  if [ $dejala -lt 1 ] ; then
    dliste="${dliste} $step"
  fi
  ln -fs $i bemol_input/
done

for i in ${dliste} ; do
  bliste=`find -L bemol_input/ -name "${type}*${i}"  | xargs`
  ienati=`echo ${bliste} | wc -w`
  if [ ${ienati} -gt 0 ] ; then
    for ii in ${bliste} ; do
      destination=${ii##*/}
      if [ ${dplusp} -gt 0 ] ; then
        destination=`echo $destination | sed 's/\(^.\)\(.*\)/\2/'`
      fi
      destination=${prefix}${destination%%-*}_${destination#*_}
      break
    done
    printf "    reassembleur $i: START: `date`\n"
    printf "    source               : ${bliste}\n"
    printf "    destination          : ${repdst}/${destination}\n"
    ${TASK_BIN}/reassembleur -src ${bliste} -dst ${repdst}/${destination} 1>/dev/null || exit 3
    if [ -s ${repdst}/${destination} ] ; then
      /bin/rm -f ${bliste}
    fi
    printf "    reassembleur: END: `date`\n"
  fi
done
#
#cd out_bemol ; find ./ -type f -exec mv {} $TASK_WORK/finished \;
#cd ../ ; rmdir out_bemol
#

/bin/rm -rf bemol_input

exit 0


